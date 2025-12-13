"""SCF-to-LLVM Lowering Pass for Morphogen v0.7.4 Phase 6

This module implements the lowering pass that transforms SCF/Arith/Func dialects
into LLVM IR dialect, enabling JIT/AOT compilation to native code.

Transformation:
    scf.for/scf.if + arith.* + func.* → llvm.*

Example:
    Input (High-level):
        func.func @add(%arg0: f32, %arg1: f32) -> f32 {
          %0 = arith.addf %arg0, %arg1 : f32
          return %0 : f32
        }

    Output (LLVM dialect):
        llvm.func @add(%arg0: f32, %arg1: f32) -> f32 {
          %0 = llvm.fadd %arg0, %arg1 : f32
          llvm.return %0 : f32
        }

This pass is the critical bridge between high-level MLIR dialects and
executable native code.
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf, func, llvm
    from mlir import passmanager
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class SCFToLLVMPass:
    """Lowering pass: SCF/Arith/Func → LLVM dialect.

    This pass applies MLIR's built-in lowering passes to convert
    high-level dialects to LLVM IR dialect:

    1. Convert SCF to Control Flow (CF) dialect
    2. Convert Arith to LLVM
    3. Convert Func to LLVM
    4. Convert MemRef to LLVM
    5. Reconcile unrealized casts

    Operations Lowered:
        - scf.for → llvm.br/llvm.cond_br loops
        - scf.if → llvm.cond_br
        - arith.addf → llvm.fadd
        - arith.addi → llvm.add
        - func.func → llvm.func
        - memref.alloc → llvm.alloca/llvm.call @malloc

    Usage:
        >>> pass_obj = SCFToLLVMPass(context)
        >>> pass_obj.run(module)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize SCF-to-LLVM pass.

        Args:
            context: Morphogen MLIR context

        Raises:
            RuntimeError: If MLIR is not available
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        self.context = context

    def run(self, module: Any) -> None:
        """Run lowering pass on module.

        This applies a sequence of MLIR's built-in conversion passes
        to lower all high-level dialects to LLVM dialect.

        Args:
            module: MLIR module to transform (in-place)
        """
        with self.context.ctx:
            pm = passmanager.PassManager.parse(
                "builtin.module("
                # Lower SCF to Control Flow dialect
                "convert-scf-to-cf,"
                # Lower Arith to LLVM
                "convert-arith-to-llvm,"
                # Lower MemRef to LLVM
                "finalize-memref-to-llvm,"
                # Lower Func to LLVM
                "convert-func-to-llvm,"
                # Reconcile unrealized casts
                "reconcile-unrealized-casts"
                ")"
            )
            pm.run(module.operation)

    def run_with_optimization(self, module: Any, opt_level: int = 2) -> None:
        """Run lowering pass with optimization.

        Args:
            module: MLIR module to transform
            opt_level: Optimization level (0-3)
        """
        with self.context.ctx:
            # Build optimization pipeline based on level
            opt_passes = []

            if opt_level >= 1:
                opt_passes.extend([
                    "inline",  # Function inlining
                    "cse",     # Common subexpression elimination
                    "canonicalize",  # Canonicalization
                ])

            if opt_level >= 2:
                opt_passes.extend([
                    "loop-invariant-code-motion",  # LICM
                    "affine-loop-unroll{unroll-factor=4}",  # Loop unrolling
                ])

            if opt_level >= 3:
                opt_passes.extend([
                    "affine-loop-tile{tile-size=32}",  # Loop tiling for cache
                    "affine-vectorize{vectorize-reductions=true}",  # Auto-vectorization
                ])

            # Add lowering passes
            opt_passes.extend([
                "convert-scf-to-cf",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "reconcile-unrealized-casts",
            ])

            # Create and run pass manager
            pipeline = "builtin.module(" + ",".join(opt_passes) + ")"
            pm = passmanager.PassManager.parse(pipeline)
            pm.run(module.operation)


def create_scf_to_llvm_pass(context: MorphogenMLIRContext) -> SCFToLLVMPass:
    """Create SCF-to-LLVM lowering pass.

    Args:
        context: Morphogen MLIR context

    Returns:
        SCFToLLVMPass instance

    Example:
        >>> from morphogen.mlir.context import MorphogenMLIRContext
        >>> ctx = MorphogenMLIRContext()
        >>> pass_obj = create_scf_to_llvm_pass(ctx)
        >>> pass_obj.run(module)
    """
    return SCFToLLVMPass(context)


# Convenience function for direct lowering
def lower_to_llvm(module: Any, context: MorphogenMLIRContext, opt_level: int = 2) -> None:
    """Lower module to LLVM dialect with optimization.

    Args:
        module: MLIR module to lower
        context: Morphogen MLIR context
        opt_level: Optimization level (0-3)

    Example:
        >>> lower_to_llvm(module, context, opt_level=3)
    """
    pass_obj = create_scf_to_llvm_pass(context)
    pass_obj.run_with_optimization(module, opt_level)
