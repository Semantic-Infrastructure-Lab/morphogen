"""Execution Engine API for Morphogen v0.7.4 Phase 6

This module provides a high-level ExecutionEngine API that unifies
JIT and AOT compilation with memory management and resource cleanup.

Features:
- Unified API for JIT and AOT compilation
- Automatic memory management for buffers and arrays
- Resource cleanup on context exit
- Support for NumPy arrays and memrefs
- Context manager support (with statement)
- Type-safe function invocation

Example usage:
    >>> from morphogen.mlir.context import MorphogenMLIRContext
    >>> ctx = MorphogenMLIRContext()
    >>>
    >>> with ExecutionEngine(ctx, module, mode='jit') as engine:
    >>>     result = engine.invoke("my_func", 1.0, 2.0)
    >>>     print(result)
"""

from __future__ import annotations
import gc
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    import numpy as np
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    np = None
    if TYPE_CHECKING:
        from mlir import ir


class ExecutionMode(Enum):
    """Execution mode for ExecutionEngine."""
    JIT = "jit"          # Just-In-Time compilation
    AOT = "aot"          # Ahead-of-Time compilation
    INTERPRET = "interpret"  # Interpreter mode (future)


class MemoryBuffer:
    """Memory buffer wrapper with automatic cleanup.

    Wraps NumPy arrays and provides automatic memory management
    with reference counting and cleanup on context exit.
    """

    def __init__(self, array: Any, owner: Optional[Any] = None):
        """Initialize memory buffer.

        Args:
            array: NumPy array or ctypes buffer
            owner: Optional owner object for lifetime management
        """
        self.array = array
        self.owner = owner
        self._weakref = weakref.ref(self, self._cleanup)

    @staticmethod
    def _cleanup(ref: weakref.ref) -> None:
        """Cleanup callback when buffer is garbage collected."""
        # Memory automatically freed by NumPy/Python GC
        pass

    def __array__(self) -> Any:
        """NumPy array interface."""
        return self.array

    def __repr__(self) -> str:
        return f"MemoryBuffer(shape={self.array.shape}, dtype={self.array.dtype})"


class ExecutionEngine:
    """High-level execution engine for Morphogen programs.

    Provides unified API for both JIT and AOT compilation with
    automatic memory management and resource cleanup.

    Features:
    - Automatic memory management for buffers
    - Context manager support (with statement)
    - JIT and AOT compilation modes
    - Type-safe function invocation
    - Resource cleanup

    Example:
        >>> with ExecutionEngine(ctx, module, mode='jit') as engine:
        >>>     result = engine.invoke("add", 3.0, 4.0)
        >>>     print(result)  # 7.0
    """

    def __init__(
        self,
        context: MorphogenMLIRContext,
        module: Any,
        mode: str = 'jit',
        opt_level: int = 2,
        cache_dir: Optional[Path] = None
    ):
        """Initialize execution engine.

        Args:
            context: Morphogen MLIR context
            module: MLIR module to execute
            mode: Execution mode ('jit' or 'aot')
            opt_level: Optimization level (0-3)
            cache_dir: Cache directory for JIT (None = memory only)

        Raises:
            RuntimeError: If MLIR is not available
            ValueError: If mode is invalid
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        self.context = context
        self.module = module
        self.mode = ExecutionMode(mode)
        self.opt_level = opt_level

        # JIT or AOT backend
        if self.mode == ExecutionMode.JIT:
            from .jit import create_jit
            self.backend = create_jit(context, enable_cache=True, cache_dir=cache_dir)
            self.backend.compile(module, opt_level=opt_level)
        elif self.mode == ExecutionMode.AOT:
            from .aot import create_aot
            self.backend = create_aot(context)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Memory management
        self._buffers: List[MemoryBuffer] = []
        self._is_closed = False

    def __enter__(self) -> ExecutionEngine:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup resources."""
        self.close()

    def invoke(self, func_name: str, *args) -> Any:
        """Execute function with automatic argument marshalling.

        Args:
            func_name: Name of function to execute
            *args: Function arguments (scalars or arrays)

        Returns:
            Function result

        Raises:
            RuntimeError: If engine is closed or execution fails

        Example:
            >>> result = engine.invoke("add", 3.0, 4.0)
        """
        if self._is_closed:
            raise RuntimeError("ExecutionEngine is closed")

        if self.mode == ExecutionMode.JIT:
            return self.backend.invoke(func_name, *args)
        else:
            raise RuntimeError("AOT mode requires explicit compilation to file")

    def allocate_buffer(
        self,
        shape: tuple,
        dtype: str = 'float32',
        fill_value: Optional[float] = None
    ) -> MemoryBuffer:
        """Allocate memory buffer with automatic cleanup.

        Args:
            shape: Buffer shape (e.g., (100, 100))
            dtype: Data type ('float32', 'float64', 'int32', etc.)
            fill_value: Optional fill value (default: zeros)

        Returns:
            MemoryBuffer instance

        Example:
            >>> buffer = engine.allocate_buffer((256, 256), dtype='float32')
        """
        if not MLIR_AVAILABLE or np is None:
            raise RuntimeError("NumPy not available")

        # Create NumPy array
        if fill_value is not None:
            array = np.full(shape, fill_value, dtype=dtype)
        else:
            array = np.zeros(shape, dtype=dtype)

        # Wrap in MemoryBuffer
        buffer = MemoryBuffer(array, owner=self)
        self._buffers.append(buffer)

        return buffer

    def get_function_signature(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get function signature.

        Args:
            func_name: Function name

        Returns:
            Dictionary with 'inputs' and 'output' types

        Example:
            >>> sig = engine.get_function_signature("add")
            >>> print(sig)  # {'inputs': ['f32', 'f32'], 'output': 'f32'}
        """
        if self.mode == ExecutionMode.JIT:
            return self.backend.get_function_signature(func_name)
        return None

    def list_functions(self) -> List[str]:
        """List all functions in the module.

        Returns:
            List of function names

        Example:
            >>> funcs = engine.list_functions()
            >>> print(funcs)  # ['add', 'multiply', 'main']
        """
        functions = []

        with self.context.ctx:
            for op in self.module.body.operations:
                if hasattr(op, 'name') and hasattr(op.name, 'value'):
                    functions.append(op.name.value)

        return functions

    def compile_to_file(
        self,
        output_path: Path,
        format: str = 'shared',
        exported_symbols: Optional[List[str]] = None
    ) -> None:
        """Compile module to file (AOT compilation).

        Args:
            output_path: Output file path
            format: Output format ('shared', 'executable', 'object', etc.)
            exported_symbols: Symbols to export (for shared libs)

        Example:
            >>> engine.compile_to_file("libfoo.so", format='shared')
        """
        if self.mode != ExecutionMode.AOT:
            # Switch to AOT mode
            from .aot import create_aot
            self.backend = create_aot(self.context)
            self.mode = ExecutionMode.AOT

        from .aot import OutputFormat

        # Map format string to enum
        format_map = {
            'shared': OutputFormat.SHARED_LIB,
            'executable': OutputFormat.EXECUTABLE,
            'object': OutputFormat.OBJECT_FILE,
            'static': OutputFormat.STATIC_LIB,
            'llvm-ir': OutputFormat.LLVM_IR_TEXT,
            'llvm-bc': OutputFormat.LLVM_BC,
            'assembly': OutputFormat.ASSEMBLY,
        }

        output_format = format_map.get(format)
        if output_format is None:
            raise ValueError(f"Invalid format: {format}")

        self.backend.compile(
            self.module,
            output_path,
            format=output_format,
            opt_level=self.opt_level,
            exported_symbols=exported_symbols
        )

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory statistics

        Example:
            >>> stats = engine.get_memory_usage()
            >>> print(stats)
            >>> # {'num_buffers': 5, 'total_bytes': 262144}
        """
        total_bytes = 0
        for buffer in self._buffers:
            if hasattr(buffer.array, 'nbytes'):
                total_bytes += buffer.array.nbytes

        return {
            'num_buffers': len(self._buffers),
            'total_bytes': total_bytes
        }

    def clear_buffers(self) -> None:
        """Clear all allocated buffers.

        Releases references to all allocated buffers.
        Actual memory is freed by garbage collector.
        """
        self._buffers.clear()
        gc.collect()

    def close(self) -> None:
        """Close execution engine and cleanup resources.

        Releases all buffers and clears caches.
        """
        if not self._is_closed:
            # Clear all buffers
            self.clear_buffers()

            # Clear JIT cache if applicable
            if self.mode == ExecutionMode.JIT:
                if hasattr(self.backend, 'clear_cache'):
                    self.backend.clear_cache()

            self._is_closed = True

    def __del__(self) -> None:
        """Destructor - cleanup resources."""
        if not self._is_closed:
            self.close()


def create_execution_engine(
    context: MorphogenMLIRContext,
    module: Any,
    mode: str = 'jit',
    opt_level: int = 2,
    cache_dir: Optional[Path] = None
) -> ExecutionEngine:
    """Create execution engine instance.

    Args:
        context: Morphogen MLIR context
        module: MLIR module to execute
        mode: Execution mode ('jit' or 'aot')
        opt_level: Optimization level (0-3)
        cache_dir: Cache directory for JIT

    Returns:
        ExecutionEngine instance

    Example:
        >>> from morphogen.mlir.context import MorphogenMLIRContext
        >>> ctx = MorphogenMLIRContext()
        >>> engine = create_execution_engine(ctx, module, mode='jit')
        >>> result = engine.invoke("my_func", 1.0, 2.0)
    """
    return ExecutionEngine(context, module, mode, opt_level, cache_dir)
