"""JIT Compilation Engine for Morphogen v0.7.4 Phase 6

This module implements Just-In-Time compilation of Morphogen programs using
MLIR's ExecutionEngine with caching support for improved performance.

Features:
- JIT compilation of MLIR modules to native code
- Function execution with automatic argument marshalling
- Compilation caching (in-memory and persistent)
- Support for all standard types (f32, f64, i32, i64, memref)
- Optimization levels (0-3)
- Thread-safe execution

Example usage:
    >>> from morphogen.mlir.context import MorphogenMLIRContext
    >>> ctx = MorphogenMLIRContext()
    >>> jit = MorphogenJIT(ctx)
    >>> jit.compile(module, opt_level=2)
    >>> result = jit.invoke("add", 3.0, 4.0)
    >>> print(result)  # 7.0
"""

from __future__ import annotations
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import ctypes

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, func, llvm
    from mlir.execution_engine import ExecutionEngine
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None
        ExecutionEngine = None


class CompilationCache:
    """Cache for compiled functions.

    Supports both in-memory and persistent disk caching.
    Thread-safe for concurrent access.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize compilation cache.

        Args:
            cache_dir: Directory for persistent cache (None = memory only)
        """
        self._memory_cache: Dict[str, Any] = {}
        self._cache_dir = cache_dir
        self._lock = threading.RLock()

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get cached compilation.

        Args:
            key: Cache key (hash of MLIR module)

        Returns:
            Cached ExecutionEngine or None
        """
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                return self._memory_cache[key]

            # Check disk cache
            if self._cache_dir:
                cache_file = self._cache_dir / f"{key}.cache"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached = pickle.load(f)
                            self._memory_cache[key] = cached
                            return cached
                    except Exception:
                        # Corrupted cache, ignore
                        pass

        return None

    def put(self, key: str, value: Any) -> None:
        """Store compilation in cache.

        Args:
            key: Cache key
            value: ExecutionEngine to cache
        """
        with self._lock:
            self._memory_cache[key] = value

            # Persist to disk if enabled
            if self._cache_dir:
                cache_file = self._cache_dir / f"{key}.cache"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(value, f)
                except Exception:
                    # Cache write failed, continue without disk cache
                    pass

    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._memory_cache.clear()
            if self._cache_dir:
                for cache_file in self._cache_dir.glob("*.cache"):
                    cache_file.unlink()


class MorphogenJIT:
    """JIT compiler for Morphogen programs.

    This class manages the compilation and execution of Morphogen programs
    via MLIR's JIT compilation infrastructure with caching support.

    Features:
    - Compiles MLIR modules to native code
    - Caches compiled functions for reuse
    - Supports optimization levels 0-3
    - Thread-safe execution
    - Automatic argument marshalling

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> jit = MorphogenJIT(ctx)
        >>> jit.compile(module, opt_level=2)
        >>> result = jit.invoke("my_function", 1.0, 2.0, 3.0)
    """

    def __init__(
        self,
        context: MorphogenMLIRContext,
        enable_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """Initialize JIT compiler.

        Args:
            context: Morphogen MLIR context
            enable_cache: Enable compilation caching
            cache_dir: Directory for persistent cache (None = memory only)

        Raises:
            RuntimeError: If MLIR is not available
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError(
                "MLIR not available. Install with: "
                "pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
            )

        self.context = context
        self.engine: Optional[ExecutionEngine] = None
        self.module: Optional[Any] = None
        self.opt_level: int = 2

        # Caching
        self.enable_cache = enable_cache
        self.cache = CompilationCache(cache_dir) if enable_cache else None

        # Thread safety
        self._lock = threading.RLock()

    def compile(
        self,
        module: Any,
        opt_level: int = 2,
        shared_libs: Optional[List[str]] = None
    ) -> None:
        """Compile module to native code.

        Args:
            module: MLIR module to compile
            opt_level: LLVM optimization level (0-3)
            shared_libs: Additional shared libraries to link

        Raises:
            RuntimeError: If compilation fails
        """
        with self._lock:
            self.module = module
            self.opt_level = opt_level

            # Check cache
            if self.enable_cache:
                cache_key = self._compute_cache_key(module, opt_level)
                cached_engine = self.cache.get(cache_key)
                if cached_engine is not None:
                    self.engine = cached_engine
                    return

            # Lower to LLVM dialect
            from ..lowering.scf_to_llvm import lower_to_llvm
            lower_to_llvm(module, self.context, opt_level)

            # Create execution engine
            with self.context.ctx:
                try:
                    # Configure execution engine
                    ee_options = {
                        'opt_level': opt_level,
                        'shared_libs': shared_libs or []
                    }

                    # Create engine
                    self.engine = ExecutionEngine(module, **ee_options)

                    # Cache the compiled engine
                    if self.enable_cache:
                        self.cache.put(cache_key, self.engine)

                except Exception as e:
                    raise RuntimeError(f"JIT compilation failed: {e}")

    def invoke(self, func_name: str, *args) -> Any:
        """Execute compiled function.

        Args:
            func_name: Name of function to execute
            *args: Arguments to pass to function (scalars or arrays)

        Returns:
            Function result (scalar or array)

        Raises:
            RuntimeError: If function not found or execution fails

        Example:
            >>> result = jit.invoke("add", 3.0, 4.0)
            >>> print(result)  # 7.0
        """
        with self._lock:
            if self.engine is None:
                raise RuntimeError("Module not compiled. Call compile() first.")

            try:
                # Look up function
                func_ptr = self.engine.lookup(func_name)
                if func_ptr is None:
                    raise RuntimeError(f"Function '{func_name}' not found")

                # Marshall arguments
                c_args = self._marshall_args(args)

                # Call function
                result = func_ptr(*c_args)

                return result

            except Exception as e:
                raise RuntimeError(f"JIT execution failed: {e}")

    def invoke_async(self, func_name: str, *args) -> Any:
        """Execute function asynchronously (non-blocking).

        Args:
            func_name: Name of function to execute
            *args: Arguments to pass

        Returns:
            Future-like object for retrieving result

        Note: For now, this is just a wrapper around invoke().
        True async support requires thread pool implementation.
        """
        # TODO: Implement true async execution with thread pool
        return self.invoke(func_name, *args)

    def get_function_signature(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get function signature information.

        Args:
            func_name: Name of function

        Returns:
            Dictionary with 'inputs' and 'output' types, or None if not found

        Example:
            >>> sig = jit.get_function_signature("add")
            >>> print(sig)
            >>> # {'inputs': ['f32', 'f32'], 'output': 'f32'}
        """
        if self.module is None:
            return None

        with self.context.ctx:
            # Walk module to find function
            for op in self.module.body.operations:
                if hasattr(op, 'name') and op.name.value == func_name:
                    # Extract signature from func.func operation
                    func_type = op.type
                    return {
                        'inputs': [str(t) for t in func_type.inputs],
                        'output': str(func_type.results[0]) if func_type.results else 'void'
                    }

        return None

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        if self.cache:
            self.cache.clear()

    def _compute_cache_key(self, module: Any, opt_level: int) -> str:
        """Compute cache key for module.

        Args:
            module: MLIR module
            opt_level: Optimization level

        Returns:
            SHA256 hash of module IR + opt_level
        """
        with self.context.ctx:
            module_str = str(module)
            combined = f"{module_str}:opt={opt_level}"
            return hashlib.sha256(combined.encode()).hexdigest()

    def _marshall_args(self, args: Tuple[Any, ...]) -> List[Any]:
        """Marshall Python arguments to C types.

        Args:
            args: Python arguments

        Returns:
            List of ctypes arguments

        Supports:
        - Python float → ctypes.c_float or ctypes.c_double
        - Python int → ctypes.c_int32 or ctypes.c_int64
        - NumPy arrays → ctypes pointers
        """
        c_args = []

        for arg in args:
            if isinstance(arg, float):
                c_args.append(ctypes.c_float(arg))
            elif isinstance(arg, int):
                c_args.append(ctypes.c_int32(arg))
            else:
                # Try NumPy array
                try:
                    import numpy as np
                    if isinstance(arg, np.ndarray):
                        c_args.append(arg.ctypes.data_as(ctypes.c_void_p))
                    else:
                        c_args.append(arg)
                except ImportError:
                    c_args.append(arg)

        return c_args


def create_jit(
    context: MorphogenMLIRContext,
    enable_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> MorphogenJIT:
    """Create JIT compiler instance.

    Args:
        context: Morphogen MLIR context
        enable_cache: Enable compilation caching
        cache_dir: Directory for persistent cache

    Returns:
        MorphogenJIT instance

    Example:
        >>> from morphogen.mlir.context import MorphogenMLIRContext
        >>> ctx = MorphogenMLIRContext()
        >>> jit = create_jit(ctx, enable_cache=True)
    """
    return MorphogenJIT(context, enable_cache, cache_dir)
