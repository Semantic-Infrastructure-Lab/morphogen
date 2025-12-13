"""Morphogen Agent Dialect (v0.7.0 Phase 4)

This module defines the Morphogen Agent dialect for MLIR, providing high-level
operations for agent-based simulations with spawning, behavior trees, and
property management that lower to SCF loops and memref operations.

Status: Phase 4 Implementation (Months 10-12)

Operations:
- morphogen.agent.spawn: Create agents at positions with initial properties
- morphogen.agent.update: Update agent properties (position, velocity, state)
- morphogen.agent.query: Read agent properties at specific indices
- morphogen.agent.behavior: Define behavior trees/rules for agents

Type System:
- !morphogen.agent<T>: Generic agent type (opaque for Phase 4)
"""

from __future__ import annotations
from typing import Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class AgentType:
    """Wrapper for !morphogen.agent<T> type.

    In Phase 4, we use OpaqueType to represent custom agent types.
    In Phase 5+, this will be replaced with proper IRDL dialect definition.

    Agents are stored as structured arrays with properties:
    - position: (x, y) coordinates
    - velocity: (vx, vy) velocity vector
    - state: scalar state value
    - custom: additional custom properties

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> agent_type = AgentType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(agent_type)  # !morphogen.agent<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get agent type for given element type.

        Args:
            element_type: MLIR element type (e.g., F32Type, F64Type)
            context: MLIR context

        Returns:
            Opaque agent type !morphogen.agent<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        # Use OpaqueType for custom types in Phase 4
        # Format: !morphogen.agent<element_type>
        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"agent<{element_str}>", context=context)


class AgentSpawnOp:
    """Operation: morphogen.agent.spawn

    Creates agents at specified positions with initial properties.

    Syntax:
        %agents = morphogen.agent.spawn %count, %positions, %properties : !morphogen.agent<f32>

    Attributes:
        - count: Number of agents to spawn (index type)
        - positions: Initial positions (memref or values)
        - properties: Initial property values (velocity, state, etc.)

    Results:
        - Agent collection of type !morphogen.agent<element_type>

    Lowering:
        Lowers to memref.alloc + initialization loops:

        Agent structure in memory: [x, y, vx, vy, state, custom...]
        Properties per agent: 5+ values (position: 2, velocity: 2, state: 1, ...)

        %agents = memref.alloc(%count, %num_properties) : memref<?x?xf32>
        scf.for %i = %c0 to %count step %c1 {
            // Initialize agent i's properties
            memref.store %pos_x, %agents[%i, %c0]
            memref.store %pos_y, %agents[%i, %c1]
            memref.store %vel_x, %agents[%i, %c2]
            memref.store %vel_y, %agents[%i, %c3]
            memref.store %state_val, %agents[%i, %c4]
        }
    """

    @staticmethod
    def create(
        count: Any,  # ir.Value with index type
        position_x: Any,  # ir.Value with element type (initial x position)
        position_y: Any,  # ir.Value with element type (initial y position)
        velocity_x: Any,  # ir.Value with element type (initial x velocity)
        velocity_y: Any,  # ir.Value with element type (initial y velocity)
        state: Any,  # ir.Value with element type (initial state)
        element_type: Any,  # MLIR type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an agent spawn operation.

        Args:
            count: Number of agents to spawn
            position_x: Initial x position for all agents
            position_y: Initial y position for all agents
            velocity_x: Initial x velocity for all agents
            velocity_y: Initial y velocity for all agents
            state: Initial state value for all agents
            element_type: Element type (f32, f64, etc.)
            loc: Source location
            ip: Insertion point

        Returns:
            Agent collection value

        Note:
            This creates a custom operation using UnrealizedConversionCastOp for Phase 4.
            In Phase 5+, this will use proper dialect definition.
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Create the agent type
            agent_type = AgentType.get(element_type, loc.context)

            # Create placeholder operation
            result = builtin.UnrealizedConversionCastOp(
                [agent_type],
                [count, position_x, position_y, velocity_x, velocity_y, state]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.agent.spawn", context=loc.context
            )
            result.operation.attributes["num_properties"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), 5
            )

            return result.results[0]


class AgentUpdateOp:
    """Operation: morphogen.agent.update

    Updates agent properties at specified indices.

    Syntax:
        %agents_new = morphogen.agent.update %agents, %index, %property, %value : !morphogen.agent<f32>

    Arguments:
        - agents: Agent collection
        - index: Agent index to update (index type)
        - property: Property index (0=x, 1=y, 2=vx, 3=vy, 4=state, ...)
        - value: New property value (element type)

    Results:
        - Updated agent collection (SSA form requires new value)

    Lowering:
        Lowers to memref.store:

        memref.store %value, %agents[%index, %property]

    Note:
        In SSA form, the result is the same memref (agents are mutable).
    """

    @staticmethod
    def create(
        agents: Any,  # ir.Value with agent type
        index: Any,  # ir.Value with index type
        property_index: Any,  # ir.Value with index type
        value: Any,  # ir.Value with element type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an agent update operation.

        Args:
            agents: Agent collection
            index: Agent index to update
            property_index: Property index to update
            value: New property value
            loc: Source location
            ip: Insertion point

        Returns:
            Updated agent collection
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same agent type
            agent_type = agents.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [agent_type],
                [agents, index, property_index, value]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.agent.update", context=loc.context
            )

            return result.results[0]


class AgentQueryOp:
    """Operation: morphogen.agent.query

    Reads agent property values at specified indices.

    Syntax:
        %value = morphogen.agent.query %agents, %index, %property : f32

    Arguments:
        - agents: Agent collection
        - index: Agent index to query (index type)
        - property: Property index to read (index type)

    Results:
        - Property value (element type)

    Lowering:
        Lowers to memref.load:

        %value = memref.load %agents[%index, %property]
    """

    @staticmethod
    def create(
        agents: Any,  # ir.Value with agent type
        index: Any,  # ir.Value with index type
        property_index: Any,  # ir.Value with index type
        element_type: Any,  # MLIR element type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an agent query operation.

        Args:
            agents: Agent collection
            index: Agent index to query
            property_index: Property index to read
            element_type: Element type to return
            loc: Source location
            ip: Insertion point

        Returns:
            Property value
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is element type
            result_type = element_type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [result_type],
                [agents, index, property_index]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.agent.query", context=loc.context
            )

            return result.results[0]


class AgentBehaviorOp:
    """Operation: morphogen.agent.behavior

    Defines behavior trees/rules for agents. This is a simplified implementation
    for Phase 4 that applies a function to all agents.

    Syntax:
        %agents_new = morphogen.agent.behavior %agents, %behavior_fn : !morphogen.agent<f32>

    Arguments:
        - agents: Agent collection
        - behavior_fn: Behavior function/operation to apply

    Results:
        - Updated agent collection after applying behaviors

    Lowering:
        Lowers to loop over all agents applying behavior:

        scf.for %i = %c0 to %count step %c1 {
            // Load agent properties
            %x = memref.load %agents[%i, %c0]
            %y = memref.load %agents[%i, %c1]
            %vx = memref.load %agents[%i, %c2]
            %vy = memref.load %agents[%i, %c3]
            %state = memref.load %agents[%i, %c4]

            // Apply behavior function (user-defined logic)
            // For Phase 4: simple position update from velocity
            %new_x = arith.addf %x, %vx
            %new_y = arith.addf %y, %vy

            // Store updated properties
            memref.store %new_x, %agents[%i, %c0]
            memref.store %new_y, %agents[%i, %c1]
        }
    """

    @staticmethod
    def create(
        agents: Any,  # ir.Value with agent type
        behavior_type: str,  # Behavior type: "move", "flock", "seek", etc.
        params: List[Any],  # Optional behavior parameters
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create an agent behavior operation.

        Args:
            agents: Agent collection
            behavior_type: Type of behavior to apply
            params: Optional parameters for behavior
            loc: Source location
            ip: Insertion point

        Returns:
            Updated agent collection
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same agent type
            agent_type = agents.type

            # Create placeholder op
            operands = [agents] + params
            result = builtin.UnrealizedConversionCastOp(
                [agent_type],
                operands
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.agent.behavior", context=loc.context
            )
            result.operation.attributes["behavior_type"] = ir.StringAttr.get(
                behavior_type, context=loc.context
            )

            return result.results[0]


class AgentDialect:
    """Agent operations dialect.

    This class serves as a namespace for agent dialect operations
    and provides utility methods for working with agent types.

    Operations:
        - spawn: Create agents with initial properties
        - update: Update agent properties
        - query: Read agent properties
        - behavior: Apply behavior rules

    Property Indices (standard layout):
        - 0: position_x
        - 1: position_y
        - 2: velocity_x
        - 3: velocity_y
        - 4: state

    Example:
        >>> from morphogen.mlir.dialects.agent import AgentDialect
        >>>
        >>> # Spawn agents
        >>> count = arith.ConstantOp(ir.IndexType.get(), 100)
        >>> pos_x = arith.ConstantOp(ir.F32Type.get(), 0.0)
        >>> pos_y = arith.ConstantOp(ir.F32Type.get(), 0.0)
        >>> vel_x = arith.ConstantOp(ir.F32Type.get(), 0.1)
        >>> vel_y = arith.ConstantOp(ir.F32Type.get(), 0.0)
        >>> state = arith.ConstantOp(ir.F32Type.get(), 0.0)
        >>> agents = AgentDialect.spawn(count, pos_x, pos_y, vel_x, vel_y, state, f32, loc, ip)
        >>>
        >>> # Query agent property
        >>> idx = arith.ConstantOp(ir.IndexType.get(), 0)
        >>> prop_idx = arith.ConstantOp(ir.IndexType.get(), 0)  # position_x
        >>> x_val = AgentDialect.query(agents, idx, prop_idx, f32, loc, ip)
        >>>
        >>> # Update agent property
        >>> new_x = arith.ConstantOp(ir.F32Type.get(), 1.5)
        >>> agents_new = AgentDialect.update(agents, idx, prop_idx, new_x, loc, ip)
        >>>
        >>> # Apply behavior
        >>> agents_moved = AgentDialect.behavior(agents_new, "move", [], loc, ip)
    """

    # Property index constants
    PROP_POS_X = 0
    PROP_POS_Y = 1
    PROP_VEL_X = 2
    PROP_VEL_Y = 3
    PROP_STATE = 4
    NUM_BASE_PROPERTIES = 5

    spawn = AgentSpawnOp.create
    update = AgentUpdateOp.create
    query = AgentQueryOp.create
    behavior = AgentBehaviorOp.create

    @staticmethod
    def is_agent_op(op: Any) -> bool:
        """Check if an operation is an agent operation.

        Args:
            op: MLIR operation to check

        Returns:
            True if op is an agent operation
        """
        if not MLIR_AVAILABLE:
            return False

        # Check if op has our marker attribute
        if not hasattr(op, "attributes"):
            return False

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return False

        op_name = str(op_name_attr)
        return "morphogen.agent." in op_name

    @staticmethod
    def get_agent_op_name(op: Any) -> Optional[str]:
        """Get the agent operation name.

        Args:
            op: Agent operation

        Returns:
            Operation name (e.g., "morphogen.agent.spawn") or None
        """
        if not MLIR_AVAILABLE:
            return None

        if not hasattr(op, "attributes"):
            return None

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return None

        return str(op_name_attr).strip('"')

    @staticmethod
    def get_property_index_constant(
        property_name: str,
        context: Any
    ) -> int:
        """Get property index for named property.

        Args:
            property_name: Property name (pos_x, pos_y, vel_x, vel_y, state)
            context: MLIR context

        Returns:
            Property index as integer

        Raises:
            ValueError: If property name is unknown
        """
        property_map = {
            "pos_x": AgentDialect.PROP_POS_X,
            "pos_y": AgentDialect.PROP_POS_Y,
            "vel_x": AgentDialect.PROP_VEL_X,
            "vel_y": AgentDialect.PROP_VEL_Y,
            "state": AgentDialect.PROP_STATE,
        }

        if property_name not in property_map:
            raise ValueError(f"Unknown property: {property_name}")

        return property_map[property_name]


# Export public API
__all__ = [
    "AgentType",
    "AgentSpawnOp",
    "AgentUpdateOp",
    "AgentQueryOp",
    "AgentBehaviorOp",
    "AgentDialect",
    "MLIR_AVAILABLE",
]
