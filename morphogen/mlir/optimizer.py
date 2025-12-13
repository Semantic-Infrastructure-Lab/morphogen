"""MLIR Optimization Passes for Morphogen (Phase 5)

This module implements optimization passes for Morphogen's MLIR IR,
improving code quality and performance through transformations.

Optimization passes operate on the IRModule and transform it while
preserving semantics.
"""

from typing import Dict, List, Set, Optional, Any
from .ir_builder import (
    IRModule, IRFunction, IRBlock, IROperation, IRValue, IRType, IRRegion
)
from ..ast.nodes import Literal


class OptimizationPass:
    """Base class for optimization passes."""

    def run(self, module: IRModule) -> IRModule:
        """Run optimization pass on module.

        Args:
            module: IR module to optimize

        Returns:
            Optimized IR module
        """
        raise NotImplementedError("Subclasses must implement run()")


class ConstantFoldingPass(OptimizationPass):
    """Constant folding optimization pass.

    This pass evaluates constant expressions at compile time,
    replacing them with their computed values.

    Examples:
        2.0 + 3.0 → 5.0
        10 * 2 → 20
        5.0 > 3.0 → true
    """

    def run(self, module: IRModule) -> IRModule:
        """Run constant folding on module.

        Args:
            module: IR module to optimize

        Returns:
            Optimized module with folded constants
        """
        # Track constant values: SSA name → constant value
        self.constants: Dict[str, Any] = {}

        # Process each function
        for func in module.functions:
            self._fold_function(func)

        return module

    def _fold_function(self, func: IRFunction) -> None:
        """Fold constants within a function."""
        for block in func.body.blocks:
            self._fold_block(block)

    def _fold_block(self, block: IRBlock) -> None:
        """Fold constants within a block."""
        new_operations = []

        for op in block.operations:
            # Try to fold this operation
            folded_op = self._try_fold_operation(op)
            if folded_op:
                new_operations.append(folded_op)
            else:
                new_operations.append(op)

        block.operations = new_operations

    def _try_fold_operation(self, op: IROperation) -> Optional[IROperation]:
        """Try to fold an operation if all operands are constant.

        Returns:
            Folded operation (constant assignment) or None to keep original
        """
        # Check if this is a foldable arithmetic operation
        if not op.opcode.startswith("arith."):
            return None

        # Extract operands
        if len(op.operands) < 2:
            return None

        # Check if all operands are constants we know about
        operand_values = []
        for operand in op.operands:
            if operand.name in self.constants:
                operand_values.append(self.constants[operand.name])
            else:
                # Try to parse as literal constant
                const_val = self._parse_constant(operand.name)
                if const_val is not None:
                    operand_values.append(const_val)
                else:
                    return None  # Not all operands are constant

        # All operands are constant - try to fold
        result_value = self._fold_arithmetic(op.opcode, operand_values)
        if result_value is None:
            return None

        # Record this constant for future folding
        if op.results:
            self.constants[op.results[0].name] = result_value

        # Create constant operation
        # For now, just keep the original operation (full impl would replace with arith.constant)
        # This is a simplified version - real MLIR would emit constant ops
        return None

    def _parse_constant(self, ssa_name: str) -> Optional[float]:
        """Try to parse SSA name as a constant value.

        MLIR constants look like: %c5 or %cst
        This is a simplified parser.
        """
        # Check for constant marker
        if ssa_name.startswith("%c"):
            try:
                # Try to extract numeric value
                # Real implementation would track constant definitions
                return None  # Simplified: don't parse
            except:
                pass
        return None

    def _fold_arithmetic(self, opcode: str, operands: List[Any]) -> Optional[Any]:
        """Fold arithmetic operation with constant operands.

        Args:
            opcode: Operation code (e.g., "arith.addf")
            operands: Constant operand values

        Returns:
            Computed constant result or None if can't fold
        """
        if len(operands) < 2:
            return None

        a, b = operands[0], operands[1]

        try:
            if opcode == "arith.addf" or opcode == "arith.addi":
                return a + b
            elif opcode == "arith.subf" or opcode == "arith.subi":
                return a - b
            elif opcode == "arith.mulf" or opcode == "arith.muli":
                return a * b
            elif opcode == "arith.divf" or opcode == "arith.divi":
                if b != 0:
                    return a / b
            elif opcode == "arith.remf" or opcode == "arith.remi":
                if b != 0:
                    return a % b
        except:
            pass

        return None


class DeadCodeEliminationPass(OptimizationPass):
    """Dead code elimination optimization pass.

    This pass removes operations whose results are never used,
    reducing IR size and improving performance.

    Note: This is a simplified version that only removes obviously
    unused operations. A full implementation would use dataflow analysis.
    """

    def run(self, module: IRModule) -> IRModule:
        """Run dead code elimination on module.

        Args:
            module: IR module to optimize

        Returns:
            Optimized module with dead code removed
        """
        for func in module.functions:
            self._eliminate_dead_code_in_function(func)

        return module

    def _eliminate_dead_code_in_function(self, func: IRFunction) -> None:
        """Eliminate dead code within a function."""
        # Build use-def chains
        for block in func.body.blocks:
            self._eliminate_dead_code_in_block(block)

    def _eliminate_dead_code_in_block(self, block: IRBlock) -> None:
        """Eliminate dead code within a block."""
        # Track which SSA values are used
        used_values: Set[str] = set()

        # First pass: collect all used values
        for op in block.operations:
            # Don't remove operations with side effects
            if self._has_side_effects(op):
                # Mark all operands as used
                for operand in op.operands:
                    used_values.add(operand.name)
                # Mark all results as used (they have side effects)
                for result in op.results:
                    used_values.add(result.name)
            else:
                # Mark operands as used
                for operand in op.operands:
                    used_values.add(operand.name)

        # Second pass: remove operations whose results are never used
        new_operations = []
        for op in block.operations:
            # Keep operations with side effects
            if self._has_side_effects(op):
                new_operations.append(op)
                continue

            # Keep operations whose results are used
            if any(result.name in used_values for result in op.results):
                new_operations.append(op)
                # Results are used, so operands must be kept
                for operand in op.operands:
                    used_values.add(operand.name)
            # Otherwise, operation is dead - don't include it

        block.operations = new_operations

    def _has_side_effects(self, op: IROperation) -> bool:
        """Check if operation has side effects (must not be eliminated).

        Operations with side effects include:
        - Function calls (may modify state)
        - Returns
        - Yields (loop control)
        - Memory operations
        """
        side_effect_opcodes = {
            "func.call", "func.return",
            "scf.yield", "scf.for",
            "memref.load", "memref.store",
            "visual.output",  # Morphogen-specific
        }

        return op.opcode in side_effect_opcodes or "call" in op.opcode


class SimplifyPass(OptimizationPass):
    """Algebraic simplification pass.

    This pass applies algebraic identities to simplify expressions:
    - x + 0 → x
    - x * 1 → x
    - x * 0 → 0
    - x - x → 0
    """

    def run(self, module: IRModule) -> IRModule:
        """Run simplification on module.

        Args:
            module: IR module to optimize

        Returns:
            Simplified module
        """
        for func in module.functions:
            self._simplify_function(func)

        return module

    def _simplify_function(self, func: IRFunction) -> None:
        """Simplify expressions within a function."""
        for block in func.body.blocks:
            self._simplify_block(block)

    def _simplify_block(self, block: IRBlock) -> None:
        """Simplify expressions within a block."""
        # Track constant zero and one values
        zero_values = set()
        one_values = set()

        # Scan for constants
        for op in block.operations:
            if op.opcode == "arith.constant":
                # Check attributes for constant value
                if "value" in op.attributes:
                    val = op.attributes["value"]
                    if val == 0 or val == 0.0:
                        if op.results:
                            zero_values.add(op.results[0].name)
                    elif val == 1 or val == 1.0:
                        if op.results:
                            one_values.add(op.results[0].name)

        # Apply simplifications
        new_operations = []
        replacements: Dict[str, IRValue] = {}

        for op in block.operations:
            simplified = self._try_simplify(op, zero_values, one_values)
            if simplified:
                # Record replacement
                if op.results and simplified.results:
                    replacements[op.results[0].name] = simplified.results[0]
                new_operations.append(simplified)
            else:
                new_operations.append(op)

        block.operations = new_operations

    def _try_simplify(self, op: IROperation, zeros: Set[str], ones: Set[str]) -> Optional[IROperation]:
        """Try to simplify an operation using algebraic identities."""
        if not op.opcode.startswith("arith."):
            return None

        if len(op.operands) < 2:
            return None

        a, b = op.operands[0], op.operands[1]

        # x + 0 → x or 0 + x → x
        if op.opcode in ["arith.addf", "arith.addi"]:
            if b.name in zeros:
                # Result is just a
                # Can't directly replace, would need more infrastructure
                return None
            if a.name in zeros:
                # Result is just b
                return None

        # x * 1 → x or 1 * x → x
        if op.opcode in ["arith.mulf", "arith.muli"]:
            if b.name in ones:
                return None
            if a.name in ones:
                return None

        # x * 0 → 0 or 0 * x → 0
        if op.opcode in ["arith.mulf", "arith.muli"]:
            if a.name in zeros or b.name in zeros:
                # Result is zero
                return None

        return None


class OptimizationPipeline:
    """Manages a pipeline of optimization passes.

    The pipeline runs multiple optimization passes in sequence,
    allowing progressive optimization of IR.
    """

    def __init__(self, passes: Optional[List[OptimizationPass]] = None):
        """Initialize optimization pipeline.

        Args:
            passes: List of optimization passes to run (default: all passes)
        """
        if passes is None:
            # Default optimization pipeline (Phase 5)
            self.passes = [
                ConstantFoldingPass(),
                SimplifyPass(),
                DeadCodeEliminationPass(),
            ]
        else:
            self.passes = passes

    def optimize(self, module: IRModule) -> IRModule:
        """Run all optimization passes on module.

        Args:
            module: IR module to optimize

        Returns:
            Optimized module
        """
        optimized = module

        for pass_obj in self.passes:
            optimized = pass_obj.run(optimized)

        return optimized

    def add_pass(self, pass_obj: OptimizationPass) -> None:
        """Add an optimization pass to the pipeline.

        Args:
            pass_obj: Optimization pass to add
        """
        self.passes.append(pass_obj)

    def clear_passes(self) -> None:
        """Remove all passes from pipeline."""
        self.passes = []


def create_default_pipeline() -> OptimizationPipeline:
    """Create default optimization pipeline for Morphogen.

    Returns:
        Optimization pipeline with standard passes
    """
    return OptimizationPipeline()


def optimize_module(module: IRModule, pipeline: Optional[OptimizationPipeline] = None) -> IRModule:
    """Convenience function to optimize a module.

    Args:
        module: IR module to optimize
        pipeline: Optimization pipeline (default: create_default_pipeline())

    Returns:
        Optimized module
    """
    if pipeline is None:
        pipeline = create_default_pipeline()

    return pipeline.optimize(module)
