"""Agent-to-SCF Lowering Pass for Morphogen v0.7.0 Phase 4

This module implements the lowering pass that transforms Morphogen agent operations
into Structured Control Flow (SCF) loops with memref-based agent storage.

Transformation:
    morphogen.agent.* ops → scf.for loops + memref.load/store + arith ops

Example:
    Input (High-level):
        %agents = morphogen.agent.spawn %count, %pos_x, %pos_y, %vel_x, %vel_y, %state

    Output (Low-level):
        %agents = memref.alloc(%count, %c5) : memref<?x5xf32>
        scf.for %i = %c0 to %count step %c1 {
            memref.store %pos_x, %agents[%i, %c0]
            memref.store %pos_y, %agents[%i, %c1]
            memref.store %vel_x, %agents[%i, %c2]
            memref.store %vel_y, %agents[%i, %c3]
            memref.store %state, %agents[%i, %c4]
        }
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

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


class AgentToSCFPass:
    """Lowering pass: Agent operations → SCF loops + memref.

    This pass traverses the MLIR module and replaces agent operations
    with nested scf.for loops operating on memref storage for agents.

    Agent Memory Layout:
        Agents are stored as a 2D memref: memref<?x5xT>
        where each row is an agent with 5 properties:
        - [0] position_x
        - [1] position_y
        - [2] velocity_x
        - [3] velocity_y
        - [4] state

    Operations Lowered:
        - morphogen.agent.spawn → memref.alloc + initialization loop
        - morphogen.agent.update → memref.store
        - morphogen.agent.query → memref.load
        - morphogen.agent.behavior → scf.for loop with property updates

    Usage:
        >>> pass_obj = AgentToSCFPass(context)
        >>> pass_obj.run(module)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize agent-to-SCF pass.

        Args:
            context: Morphogen MLIR context
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        self.context = context

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
        # Import agent dialect to check for agent ops
        from ..dialects.agent import AgentDialect

        # Check if this is an agent operation
        if AgentDialect.is_agent_op(op):
            op_name = AgentDialect.get_agent_op_name(op)
            if op_name == "morphogen.agent.spawn":
                self._lower_agent_spawn(op)
            elif op_name == "morphogen.agent.update":
                self._lower_agent_update(op)
            elif op_name == "morphogen.agent.query":
                self._lower_agent_query(op)
            elif op_name == "morphogen.agent.behavior":
                self._lower_agent_behavior(op)

        # Recursively process nested regions
        if hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        self._process_operation(nested_op)

    def _lower_agent_spawn(self, op: Any) -> None:
        """Lower morphogen.agent.spawn to memref.alloc + initialization loop.

        Input:
            %agents = morphogen.agent.spawn %count, %pos_x, %pos_y, %vel_x, %vel_y, %state

        Output:
            %agents = memref.alloc(%count, %c5) : memref<?x5xf32>
            scf.for %i = %c0 to %count step %c1 {
                memref.store %pos_x, %agents[%i, %c0]
                memref.store %pos_y, %agents[%i, %c1]
                memref.store %vel_x, %agents[%i, %c2]
                memref.store %vel_y, %agents[%i, %c3]
                memref.store %state, %agents[%i, %c4]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: count, pos_x, pos_y, vel_x, vel_y, state
            operands = op.operands
            if len(operands) < 6:
                raise ValueError("agent.spawn requires 6 operands")

            count = operands[0]
            pos_x = operands[1]
            pos_y = operands[2]
            vel_x = operands[3]
            vel_y = operands[4]
            state = operands[5]

            # Determine element type from position value
            element_type = pos_x.type

            # Get num_properties from attributes (default 5)
            num_props_attr = op.attributes.get("num_properties")
            if num_props_attr is not None:
                num_properties = int(str(num_props_attr))
            else:
                num_properties = 5

            with ir.InsertionPoint(op):
                # Create constants
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c2 = arith.ConstantOp(ir.IndexType.get(), 2).result
                c3 = arith.ConstantOp(ir.IndexType.get(), 3).result
                c4 = arith.ConstantOp(ir.IndexType.get(), 4).result

                # Allocate agent memref: [count x num_properties]
                memref_type = ir.MemRefType.get(
                    [ir.ShapedType.get_dynamic_size(), num_properties],
                    element_type
                )
                agents = memref.AllocOp(memref_type, [count], []).result

                # Initialize all agents with given properties
                # for i in range(count):
                for_op = scf.ForOp(c0, count, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # Store properties for agent i
                    memref.StoreOp(pos_x, agents, [i, c0])  # position_x
                    memref.StoreOp(pos_y, agents, [i, c1])  # position_y
                    memref.StoreOp(vel_x, agents, [i, c2])  # velocity_x
                    memref.StoreOp(vel_y, agents, [i, c3])  # velocity_y
                    memref.StoreOp(state, agents, [i, c4])  # state

                    # Yield from loop
                    scf.YieldOp([])

            # Replace uses of agent op with memref
            op.results[0].replace_all_uses_with(agents)

            # Erase the original op
            op.operation.erase()

    def _lower_agent_update(self, op: Any) -> None:
        """Lower morphogen.agent.update to memref.store.

        Input:
            %agents_new = morphogen.agent.update %agents, %index, %property, %value

        Output:
            memref.store %value, %agents[%index, %property]
            // agents_new is the same memref (SSA form)
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: agents, index, property_index, value
            operands = op.operands
            if len(operands) < 4:
                raise ValueError("agent.update requires 4 operands")

            agents = operands[0]
            index = operands[1]
            property_index = operands[2]
            value = operands[3]

            with ir.InsertionPoint(op):
                # Store value to agents[index, property_index]
                memref.StoreOp(value, agents, [index, property_index])

            # In SSA form, the "updated" agents is the same memref
            # Replace uses with the original agents memref
            op.results[0].replace_all_uses_with(agents)

            # Erase the original op
            op.operation.erase()

    def _lower_agent_query(self, op: Any) -> None:
        """Lower morphogen.agent.query to memref.load.

        Input:
            %value = morphogen.agent.query %agents, %index, %property

        Output:
            %value = memref.load %agents[%index, %property]
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: agents, index, property_index
            operands = op.operands
            if len(operands) < 3:
                raise ValueError("agent.query requires 3 operands")

            agents = operands[0]
            index = operands[1]
            property_index = operands[2]

            with ir.InsertionPoint(op):
                # Load value from agents[index, property_index]
                value = memref.LoadOp(agents, [index, property_index]).result

            # Replace uses with loaded value
            op.results[0].replace_all_uses_with(value)

            # Erase the original op
            op.operation.erase()

    def _lower_agent_behavior(self, op: Any) -> None:
        """Lower morphogen.agent.behavior to scf.for loop with behavior logic.

        Input:
            %agents_new = morphogen.agent.behavior %agents, [params...]

        Output:
            %count = memref.dim %agents, %c0
            scf.for %i = %c0 to %count step %c1 {
                // Load agent properties
                %x = memref.load %agents[%i, %c0]
                %y = memref.load %agents[%i, %c1]
                %vx = memref.load %agents[%i, %c2]
                %vy = memref.load %agents[%i, %c3]
                %state = memref.load %agents[%i, %c4]

                // Apply behavior (depends on behavior_type)
                // For "move": update position from velocity
                %new_x = arith.addf %x, %vx
                %new_y = arith.addf %y, %vy

                // Store updated properties
                memref.store %new_x, %agents[%i, %c0]
                memref.store %new_y, %agents[%i, %c1]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: agents, [optional params]
            operands = op.operands
            if len(operands) < 1:
                raise ValueError("agent.behavior requires at least 1 operand")

            agents = operands[0]
            params = list(operands[1:]) if len(operands) > 1 else []

            # Get behavior type from attributes
            behavior_type_attr = op.attributes.get("behavior_type")
            if behavior_type_attr is not None:
                behavior_type = str(behavior_type_attr).strip('"')
            else:
                behavior_type = "move"  # Default behavior

            with ir.InsertionPoint(op):
                # Create constants
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c2 = arith.ConstantOp(ir.IndexType.get(), 2).result
                c3 = arith.ConstantOp(ir.IndexType.get(), 3).result
                c4 = arith.ConstantOp(ir.IndexType.get(), 4).result

                # Get agent count from memref dimension
                count = memref.DimOp(agents, c0).result

                # Loop over all agents
                for_op = scf.ForOp(c0, count, c1)
                with ir.InsertionPoint(for_op.body):
                    i = for_op.induction_variable

                    # Load agent properties
                    x = memref.LoadOp(agents, [i, c0]).result
                    y = memref.LoadOp(agents, [i, c1]).result
                    vx = memref.LoadOp(agents, [i, c2]).result
                    vy = memref.LoadOp(agents, [i, c3]).result
                    state = memref.LoadOp(agents, [i, c4]).result

                    # Apply behavior based on behavior_type
                    if behavior_type == "move":
                        # Simple movement: position += velocity
                        new_x = arith.AddFOp(x, vx).result
                        new_y = arith.AddFOp(y, vy).result

                        # Store updated position
                        memref.StoreOp(new_x, agents, [i, c0])
                        memref.StoreOp(new_y, agents, [i, c1])

                    elif behavior_type == "bounce":
                        # Bounce behavior: update position and bounce off boundaries
                        # If params provided: [min_x, max_x, min_y, max_y]
                        if len(params) >= 4:
                            min_x = params[0]
                            max_x = params[1]
                            min_y = params[2]
                            max_y = params[3]

                            # Update position
                            new_x = arith.AddFOp(x, vx).result
                            new_y = arith.AddFOp(y, vy).result

                            # Check bounds and reverse velocity if needed
                            # This is simplified; full implementation would use scf.if
                            # For Phase 4, we just update position
                            memref.StoreOp(new_x, agents, [i, c0])
                            memref.StoreOp(new_y, agents, [i, c1])
                        else:
                            # No bounds, just move
                            new_x = arith.AddFOp(x, vx).result
                            new_y = arith.AddFOp(y, vy).result
                            memref.StoreOp(new_x, agents, [i, c0])
                            memref.StoreOp(new_y, agents, [i, c1])

                    elif behavior_type == "seek":
                        # Seek behavior: move towards target
                        # Params: [target_x, target_y, speed]
                        if len(params) >= 3:
                            target_x = params[0]
                            target_y = params[1]
                            speed = params[2]

                            # Compute direction to target
                            dx = arith.SubFOp(target_x, x).result
                            dy = arith.SubFOp(target_y, y).result

                            # Normalize and scale by speed (simplified)
                            # For Phase 4, we just use dx, dy directly scaled
                            new_vx = arith.MulFOp(dx, speed).result
                            new_vy = arith.MulFOp(dy, speed).result

                            # Update velocity
                            memref.StoreOp(new_vx, agents, [i, c2])
                            memref.StoreOp(new_vy, agents, [i, c3])

                            # Update position
                            new_x = arith.AddFOp(x, new_vx).result
                            new_y = arith.AddFOp(y, new_vy).result
                            memref.StoreOp(new_x, agents, [i, c0])
                            memref.StoreOp(new_y, agents, [i, c1])
                        else:
                            # No target, just move
                            new_x = arith.AddFOp(x, vx).result
                            new_y = arith.AddFOp(y, vy).result
                            memref.StoreOp(new_x, agents, [i, c0])
                            memref.StoreOp(new_y, agents, [i, c1])

                    else:
                        # Unknown behavior type, default to move
                        new_x = arith.AddFOp(x, vx).result
                        new_y = arith.AddFOp(y, vy).result
                        memref.StoreOp(new_x, agents, [i, c0])
                        memref.StoreOp(new_y, agents, [i, c1])

                    # Yield from loop
                    scf.YieldOp([])

            # In SSA form, the result is the same memref
            op.results[0].replace_all_uses_with(agents)

            # Erase the original op
            op.operation.erase()


def create_agent_to_scf_pass(context: MorphogenMLIRContext) -> AgentToSCFPass:
    """Factory function to create agent-to-SCF lowering pass.

    Args:
        context: Morphogen MLIR context

    Returns:
        Configured AgentToSCFPass instance
    """
    return AgentToSCFPass(context)


# Export public API
__all__ = [
    "AgentToSCFPass",
    "create_agent_to_scf_pass",
    "MLIR_AVAILABLE",
]
