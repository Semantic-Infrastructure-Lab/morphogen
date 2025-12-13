"""Temporal-to-SCF Lowering Pass for Morphogen v0.7.0 Phase 3

This module implements the lowering pass that transforms Morphogen temporal operations
into Structured Control Flow (SCF) loops with memref-based state management.

Transformation:
    morphogen.temporal.* ops → scf.for/while loops + memref.load/store + arith ops

Example:
    Input (High-level):
        %flow = morphogen.temporal.flow.create %dt, %steps
        %state = morphogen.temporal.state.create %size, %init_val
        %final = morphogen.temporal.flow.run %flow, %state

    Output (Low-level):
        %mem = memref.alloc(%size) : memref<?xf32>
        [initialization loop]
        scf.for %t = %c0 to %steps step %c1 iter_args(%s = %mem) {
            // Flow body operations
            scf.yield %s_new
        }
"""

from __future__ import annotations
from typing import Any, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf, func
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class TemporalToSCFPass:
    """Lowering pass: Temporal operations → SCF loops + memref.

    This pass traverses the MLIR module and replaces temporal operations
    with nested scf.for loops operating on memref storage for state management.

    Operations Lowered:
        - morphogen.temporal.flow.create → flow metadata storage
        - morphogen.temporal.flow.run → scf.for loop with iter_args
        - morphogen.temporal.state.create → memref.alloc + initialization loop
        - morphogen.temporal.state.update → memref.store
        - morphogen.temporal.state.query → memref.load

    Usage:
        >>> pass_obj = TemporalToSCFPass(context)
        >>> pass_obj.run(module)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize temporal-to-SCF pass.

        Args:
            context: Morphogen MLIR context
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        self.context = context
        # Track flow metadata (dt, steps) for each flow handle
        self.flow_metadata: Dict[Any, Dict[str, Any]] = {}

    def run(self, module: Any) -> None:
        """Run lowering pass on module.

        Args:
            module: MLIR module to transform (in-place)
        """
        with self.context.ctx:
            # Walk through all operations in the module
            for op in module.body.operations:
                self._process_operation(op)

    def _process_operation(self, op: Any) -> None:
        """Process a single operation recursively.

        Args:
            op: MLIR operation to process
        """
        from ..dialects.temporal import TemporalDialect

        # Check if this is a temporal operation
        if TemporalDialect.is_temporal_op(op):
            op_name = TemporalDialect.get_temporal_op_name(op)
            if op_name == "morphogen.temporal.flow.create":
                self._lower_flow_create(op)
            elif op_name == "morphogen.temporal.flow.run":
                self._lower_flow_run(op)
            elif op_name == "morphogen.temporal.state.create":
                self._lower_state_create(op)
            elif op_name == "morphogen.temporal.state.update":
                self._lower_state_update(op)
            elif op_name == "morphogen.temporal.state.query":
                self._lower_state_query(op)

        # Recursively process nested regions
        if hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        self._process_operation(nested_op)

    def _lower_flow_create(self, op: Any) -> None:
        """Lower morphogen.temporal.flow.create to metadata storage.

        Input:
            %flow = morphogen.temporal.flow.create %dt, %steps

        Output:
            Store dt and steps in metadata dictionary for later use.
            Replace flow handle with a tuple of (dt, steps) values.
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: dt, steps
            operands = op.operands
            if len(operands) < 2:
                raise ValueError("flow.create requires 2 operands")

            dt = operands[0]
            steps = operands[1]

            # Store metadata for this flow handle
            flow_handle = op.results[0]
            self.flow_metadata[flow_handle] = {
                "dt": dt,
                "steps": steps
            }

            # For Phase 3, we create a tuple to represent the flow
            # In a more advanced implementation, this could be a struct
            with ir.InsertionPoint(op):
                # Create a placeholder unrealized conversion cast
                # that packages dt and steps together
                flow_tuple = builtin.UnrealizedConversionCastOp(
                    [dt.type],  # Just use dt's type for now
                    [dt, steps]
                )

                # Replace uses with the dt value (we'll access metadata separately)
                op.results[0].replace_all_uses_with(dt)

            # Erase the original op
            op.operation.erase()

    def _lower_flow_run(self, op: Any) -> None:
        """Lower morphogen.temporal.flow.run to scf.for loop with temporal iteration.

        Input:
            %final = morphogen.temporal.flow.run %flow, %initial_state

        Output:
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %final = scf.for %t = %c0 to %steps step %c1 iter_args(%state = %initial_state) -> (memref<?xf32>) {
                // Flow body operations on state
                scf.yield %state
            }

        Note: For Phase 3, we support simple state pass-through.
              More complex body operations can be added in future phases.
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: flow, initial_state
            operands = op.operands
            if len(operands) < 2:
                raise ValueError("flow.run requires 2 operands")

            flow = operands[0]
            initial_state = operands[1]

            # Retrieve flow metadata
            # Since flow handle was replaced with dt, we need to find steps
            # For simplicity, we'll extract steps from the original flow.create
            # In practice, this requires tracking through the IR
            # For Phase 3, we'll use a workaround: assume steps is available

            # Search backwards for the flow.create operation that created this flow
            # This is a simplified approach for Phase 3
            steps_val = None
            dt_val = None

            # Find the flow metadata by traversing the defining op
            # For now, we'll create default values if not found
            # This is a limitation of Phase 3 implementation
            with ir.InsertionPoint(op):
                # Create default loop parameters
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result

                # Try to extract steps from flow metadata
                # For Phase 3, we'll use a heuristic: look for nearby constant
                # In production, this would be properly tracked
                c10 = arith.ConstantOp(ir.IndexType.get(), 10).result
                steps_val = c10  # Default to 10 steps

                # Get state type
                state_type = initial_state.type

                # Create scf.for loop with iter_args
                for_op = scf.ForOp(
                    c0, steps_val, c1,
                    [initial_state]  # iter_args
                )

                with ir.InsertionPoint(for_op.body):
                    # Get iteration variable and state argument
                    t = for_op.induction_variable
                    state_arg = for_op.body.arguments[0]

                    # For Phase 3, we create a simple pass-through body
                    # In Phase 4+, this would contain actual flow operations
                    # The state is passed through unchanged

                    # Yield the state for next iteration
                    scf.YieldOp([state_arg])

                # Get final state from loop result
                final_state = for_op.results[0]

            # Replace uses
            op.results[0].replace_all_uses_with(final_state)
            op.operation.erase()

    def _lower_state_create(self, op: Any) -> None:
        """Lower morphogen.temporal.state.create to memref.alloc + initialization loop.

        Input:
            %state = morphogen.temporal.state.create %size, %init_val : !morphogen.state<f32>

        Output:
            %mem = memref.alloc(%size) : memref<?xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %i = %c0 to %size step %c1 {
                memref.store %init_val, %mem[%i]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: size, initial_value
            operands = op.operands
            if len(operands) < 2:
                raise ValueError("state.create requires 2 operands")

            size = operands[0]
            initial_value = operands[1]

            # Determine element type from initial_value
            element_type = initial_value.type

            with ir.InsertionPoint(op):
                # Allocate memref
                memref_type = ir.MemRefType.get(
                    [ir.ShapedType.get_dynamic_size()],
                    element_type
                )
                mem = memref.AllocOp(memref_type, [size], []).result

                # Create constants for loop bounds
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result

                # Create initialization loop
                # for i in range(size):
                for_op = scf.ForOp(c0, size, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # mem[i] = initial_value
                    memref.StoreOp(initial_value, mem, [i])

                    # Yield from loop
                    scf.YieldOp([])

            # Replace uses of state op with memref
            op.results[0].replace_all_uses_with(mem)

            # Erase the original op
            op.operation.erase()

    def _lower_state_update(self, op: Any) -> None:
        """Lower morphogen.temporal.state.update to memref.store.

        Input:
            %new_state = morphogen.temporal.state.update %state, %index, %value

        Output:
            memref.store %value, %state[%index]
            // new_state is same as state (SSA requires explicit handling)

        Note: In SSA form, we need to maintain the illusion of state updates
              while operating on the same memref. The result is the same memref.
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: state, index, value
            operands = op.operands
            if len(operands) < 3:
                raise ValueError("state.update requires 3 operands")

            state = operands[0]
            index = operands[1]
            value = operands[2]

            with ir.InsertionPoint(op):
                # Store value to state memref at index
                memref.StoreOp(value, state, [index])

            # In SSA form, the "updated" state is the same memref
            # Replace uses with the original state
            op.results[0].replace_all_uses_with(state)

            # Erase the original op
            op.operation.erase()

    def _lower_state_query(self, op: Any) -> None:
        """Lower morphogen.temporal.state.query to memref.load.

        Input:
            %value = morphogen.temporal.state.query %state, %index : f32

        Output:
            %value = memref.load %state[%index] : memref<?xf32>
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: state, index
            operands = op.operands
            if len(operands) < 2:
                raise ValueError("state.query requires 2 operands")

            state = operands[0]
            index = operands[1]

            with ir.InsertionPoint(op):
                # Load value from state memref at index
                loaded_val = memref.LoadOp(state, [index]).result

            # Replace uses with loaded value
            op.results[0].replace_all_uses_with(loaded_val)

            # Erase the original op
            op.operation.erase()


def create_temporal_to_scf_pass(context: MorphogenMLIRContext) -> TemporalToSCFPass:
    """Factory function to create temporal-to-SCF lowering pass.

    Args:
        context: Morphogen MLIR context

    Returns:
        Configured TemporalToSCFPass instance
    """
    return TemporalToSCFPass(context)


# Export public API
__all__ = [
    "TemporalToSCFPass",
    "create_temporal_to_scf_pass",
    "MLIR_AVAILABLE",
]
