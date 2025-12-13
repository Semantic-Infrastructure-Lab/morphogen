"""AOT (Ahead-of-Time) Compilation for Morphogen v0.7.4 Phase 6

This module implements Ahead-of-Time compilation of Morphogen programs to
native binaries, shared libraries, and object files using LLVM.

Features:
- Compile to native executables (.exe, no extension)
- Compile to shared libraries (.so, .dylib, .dll)
- Compile to object files (.o, .obj)
- Compile to LLVM IR (.ll, .bc)
- Cross-compilation support
- Optimization levels (0-3)
- Symbol visibility control
- Custom linker flags

Example usage:
    >>> from morphogen.mlir.context import MorphogenMLIRContext
    >>> ctx = MorphogenMLIRContext()
    >>> aot = MorphogenAOT(ctx)
    >>> aot.compile_to_shared_library(module, "libmorphogen.so", opt_level=3)
    >>> aot.compile_to_executable(module, "myprogram", entry_point="main")
"""

from __future__ import annotations
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, func, llvm as llvm_dialect
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class OutputFormat(Enum):
    """Supported output formats for AOT compilation."""
    EXECUTABLE = "executable"      # Native executable
    SHARED_LIB = "shared"          # Shared library (.so, .dylib, .dll)
    STATIC_LIB = "static"          # Static library (.a, .lib)
    OBJECT_FILE = "object"         # Object file (.o, .obj)
    LLVM_IR_TEXT = "llvm-ir"       # LLVM IR text (.ll)
    LLVM_BC = "llvm-bc"            # LLVM bitcode (.bc)
    ASSEMBLY = "assembly"          # Assembly (.s)


class MorphogenAOT:
    """AOT compiler for Morphogen programs.

    This class manages ahead-of-time compilation of Morphogen programs
    to native binaries, shared libraries, and other formats.

    Features:
    - Compile to multiple output formats
    - Cross-compilation support
    - Optimization levels 0-3
    - Custom linker flags
    - Symbol export control

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> aot = MorphogenAOT(ctx)
        >>> aot.compile_to_shared_library(module, "libfoo.so", opt_level=3)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize AOT compiler.

        Args:
            context: Morphogen MLIR context

        Raises:
            RuntimeError: If MLIR is not available
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError(
                "MLIR not available. Install with: "
                "pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
            )

        self.context = context

    def compile(
        self,
        module: Any,
        output_path: Path,
        format: OutputFormat = OutputFormat.EXECUTABLE,
        opt_level: int = 2,
        target_triple: Optional[str] = None,
        linker_flags: Optional[List[str]] = None,
        exported_symbols: Optional[List[str]] = None
    ) -> None:
        """Compile module to specified output format.

        Args:
            module: MLIR module to compile
            output_path: Output file path
            format: Output format (executable, shared lib, etc.)
            opt_level: Optimization level (0-3)
            target_triple: Target triple for cross-compilation (None = native)
            linker_flags: Additional linker flags
            exported_symbols: Symbols to export (for shared libs)

        Raises:
            RuntimeError: If compilation fails
        """
        output_path = Path(output_path)

        # Lower to LLVM dialect
        from ..lowering.scf_to_llvm import lower_to_llvm
        lower_to_llvm(module, self.context, opt_level)

        # Emit based on format
        if format == OutputFormat.LLVM_IR_TEXT:
            self._emit_llvm_ir_text(module, output_path)
        elif format == OutputFormat.LLVM_BC:
            self._emit_llvm_bitcode(module, output_path)
        elif format == OutputFormat.ASSEMBLY:
            self._emit_assembly(module, output_path, target_triple)
        elif format == OutputFormat.OBJECT_FILE:
            self._emit_object_file(module, output_path, target_triple, opt_level)
        elif format == OutputFormat.SHARED_LIB:
            self._emit_shared_library(
                module, output_path, target_triple, opt_level, linker_flags, exported_symbols
            )
        elif format == OutputFormat.STATIC_LIB:
            self._emit_static_library(module, output_path, target_triple, opt_level)
        elif format == OutputFormat.EXECUTABLE:
            self._emit_executable(
                module, output_path, target_triple, opt_level, linker_flags
            )
        else:
            raise ValueError(f"Unsupported output format: {format}")

    def compile_to_executable(
        self,
        module: Any,
        output_path: Path,
        entry_point: str = "main",
        opt_level: int = 2,
        target_triple: Optional[str] = None,
        linker_flags: Optional[List[str]] = None
    ) -> None:
        """Compile to native executable.

        Args:
            module: MLIR module
            output_path: Output executable path
            entry_point: Entry point function name
            opt_level: Optimization level
            target_triple: Target triple
            linker_flags: Linker flags
        """
        self.compile(
            module, output_path,
            format=OutputFormat.EXECUTABLE,
            opt_level=opt_level,
            target_triple=target_triple,
            linker_flags=linker_flags
        )

    def compile_to_shared_library(
        self,
        module: Any,
        output_path: Path,
        opt_level: int = 2,
        exported_symbols: Optional[List[str]] = None,
        target_triple: Optional[str] = None
    ) -> None:
        """Compile to shared library (.so/.dylib/.dll).

        Args:
            module: MLIR module
            output_path: Output library path
            opt_level: Optimization level
            exported_symbols: Symbols to export
            target_triple: Target triple
        """
        self.compile(
            module, output_path,
            format=OutputFormat.SHARED_LIB,
            opt_level=opt_level,
            exported_symbols=exported_symbols,
            target_triple=target_triple
        )

    def compile_to_object_file(
        self,
        module: Any,
        output_path: Path,
        opt_level: int = 2,
        target_triple: Optional[str] = None
    ) -> None:
        """Compile to object file (.o/.obj).

        Args:
            module: MLIR module
            output_path: Output object file path
            opt_level: Optimization level
            target_triple: Target triple
        """
        self.compile(
            module, output_path,
            format=OutputFormat.OBJECT_FILE,
            opt_level=opt_level,
            target_triple=target_triple
        )

    def _emit_llvm_ir_text(self, module: Any, output_path: Path) -> None:
        """Emit LLVM IR text (.ll file).

        Args:
            module: MLIR module (already lowered to LLVM dialect)
            output_path: Output .ll file path
        """
        with self.context.ctx:
            # Convert MLIR LLVM dialect to LLVM IR
            llvm_ir = self._translate_to_llvm_ir(module)

            # Write to file
            with open(output_path, 'w') as f:
                f.write(llvm_ir)

    def _emit_llvm_bitcode(self, module: Any, output_path: Path) -> None:
        """Emit LLVM bitcode (.bc file).

        Args:
            module: MLIR module
            output_path: Output .bc file path
        """
        # First emit to text IR
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
            ll_path = Path(f.name)

        self._emit_llvm_ir_text(module, ll_path)

        # Convert to bitcode using llvm-as
        try:
            subprocess.run(
                ['llvm-as', str(ll_path), '-o', str(output_path)],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llvm-as failed: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("llvm-as not found. Install LLVM toolchain.")
        finally:
            ll_path.unlink()

    def _emit_assembly(
        self,
        module: Any,
        output_path: Path,
        target_triple: Optional[str]
    ) -> None:
        """Emit assembly (.s file).

        Args:
            module: MLIR module
            output_path: Output .s file path
            target_triple: Target triple
        """
        # Emit LLVM IR first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
            ll_path = Path(f.name)

        self._emit_llvm_ir_text(module, ll_path)

        # Compile to assembly with llc
        cmd = ['llc', str(ll_path), '-o', str(output_path)]
        if target_triple:
            cmd.extend(['-mtriple', target_triple])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llc failed: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("llc not found. Install LLVM toolchain.")
        finally:
            ll_path.unlink()

    def _emit_object_file(
        self,
        module: Any,
        output_path: Path,
        target_triple: Optional[str],
        opt_level: int
    ) -> None:
        """Emit object file (.o/.obj).

        Args:
            module: MLIR module
            output_path: Output object file path
            target_triple: Target triple
            opt_level: Optimization level
        """
        # Emit LLVM IR first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
            ll_path = Path(f.name)

        self._emit_llvm_ir_text(module, ll_path)

        # Compile to object file with llc
        cmd = ['llc', str(ll_path), '-filetype=obj', '-o', str(output_path)]
        if target_triple:
            cmd.extend(['-mtriple', target_triple])
        cmd.extend([f'-O{opt_level}'])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llc failed: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("llc not found. Install LLVM toolchain.")
        finally:
            ll_path.unlink()

    def _emit_shared_library(
        self,
        module: Any,
        output_path: Path,
        target_triple: Optional[str],
        opt_level: int,
        linker_flags: Optional[List[str]],
        exported_symbols: Optional[List[str]]
    ) -> None:
        """Emit shared library (.so/.dylib/.dll).

        Args:
            module: MLIR module
            output_path: Output library path
            target_triple: Target triple
            opt_level: Optimization level
            linker_flags: Additional linker flags
            exported_symbols: Symbols to export
        """
        # First compile to object file
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as f:
            obj_path = Path(f.name)

        self._emit_object_file(module, obj_path, target_triple, opt_level)

        # Link to shared library
        import platform
        system = platform.system()

        if system == 'Linux':
            cmd = ['gcc', '-shared', str(obj_path), '-o', str(output_path)]
        elif system == 'Darwin':
            cmd = ['gcc', '-dynamiclib', str(obj_path), '-o', str(output_path)]
        elif system == 'Windows':
            cmd = ['link', '/DLL', str(obj_path), f'/OUT:{output_path}']
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        # Add linker flags
        if linker_flags:
            cmd.extend(linker_flags)

        # Export symbols
        if exported_symbols:
            if system == 'Linux':
                # Create version script for symbol export
                with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
                    f.write('{\n  global:\n')
                    for sym in exported_symbols:
                        f.write(f'    {sym};\n')
                    f.write('  local: *;\n};\n')
                    version_script = Path(f.name)
                cmd.extend([f'-Wl,--version-script={version_script}'])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Linker failed: {e.stderr.decode()}")
        finally:
            obj_path.unlink()
            if exported_symbols and system == 'Linux':
                version_script.unlink()

    def _emit_static_library(
        self,
        module: Any,
        output_path: Path,
        target_triple: Optional[str],
        opt_level: int
    ) -> None:
        """Emit static library (.a/.lib).

        Args:
            module: MLIR module
            output_path: Output library path
            target_triple: Target triple
            opt_level: Optimization level
        """
        # First compile to object file
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as f:
            obj_path = Path(f.name)

        self._emit_object_file(module, obj_path, target_triple, opt_level)

        # Create archive with ar
        try:
            subprocess.run(
                ['ar', 'rcs', str(output_path), str(obj_path)],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ar failed: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("ar not found. Install binutils.")
        finally:
            obj_path.unlink()

    def _emit_executable(
        self,
        module: Any,
        output_path: Path,
        target_triple: Optional[str],
        opt_level: int,
        linker_flags: Optional[List[str]]
    ) -> None:
        """Emit native executable.

        Args:
            module: MLIR module
            output_path: Output executable path
            target_triple: Target triple
            opt_level: Optimization level
            linker_flags: Linker flags
        """
        # First compile to object file
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as f:
            obj_path = Path(f.name)

        self._emit_object_file(module, obj_path, target_triple, opt_level)

        # Link to executable
        cmd = ['gcc', str(obj_path), '-o', str(output_path)]

        # Add linker flags
        if linker_flags:
            cmd.extend(linker_flags)

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Linker failed: {e.stderr.decode()}")
        finally:
            obj_path.unlink()

    def _translate_to_llvm_ir(self, module: Any) -> str:
        """Translate MLIR LLVM dialect to LLVM IR.

        Args:
            module: MLIR module (in LLVM dialect)

        Returns:
            LLVM IR as string
        """
        with self.context.ctx:
            # Use MLIR's LLVM IR translation
            # This requires mlir-translate tool
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
                mlir_path = Path(f.name)
                f.write(str(module))

            with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
                ll_path = Path(f.name)

            try:
                subprocess.run(
                    ['mlir-translate', '--mlir-to-llvmir', str(mlir_path), '-o', str(ll_path)],
                    check=True,
                    capture_output=True
                )

                with open(ll_path, 'r') as f:
                    return f.read()

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"mlir-translate failed: {e.stderr.decode()}")
            except FileNotFoundError:
                # mlir-translate not available, fall back to string representation
                # This is a simplified fallback - real LLVM IR translation requires mlir-translate
                return str(module)
            finally:
                mlir_path.unlink()
                ll_path.unlink()


def create_aot(context: MorphogenMLIRContext) -> MorphogenAOT:
    """Create AOT compiler instance.

    Args:
        context: Morphogen MLIR context

    Returns:
        MorphogenAOT instance

    Example:
        >>> from morphogen.mlir.context import MorphogenMLIRContext
        >>> ctx = MorphogenMLIRContext()
        >>> aot = create_aot(ctx)
        >>> aot.compile_to_shared_library(module, "libfoo.so")
    """
    return MorphogenAOT(context)
