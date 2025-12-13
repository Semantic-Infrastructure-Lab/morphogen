"""Morphogen Temporal Dialect (v0.7.0 Phase 3)

This module defines the Morphogen Temporal dialect for MLIR, providing high-level
operations for time-evolving simulations with flow blocks and state management.

Status: Phase 3 Implementation (Months 7-9)

Operations:
- morphogen.temporal.flow.create: Define a flow block with timestep parameters
- morphogen.temporal.flow.step: Execute single timestep within a flow
- morphogen.temporal.flow.run: Execute complete flow for N timesteps
- morphogen.temporal.state.create: Allocate persistent state container
- morphogen.temporal.state.update: Update state values
- morphogen.temporal.state.query: Read current state values

Type System:
- !morphogen.flow<T>: Flow type (opaque for Phase 3)
- !morphogen.state<T>: State type (opaque for Phase 3)
"""

from __future__ import annotations
from typing import Optional, Any, List, TYPE_CHECKING

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


class FlowType:
    """Wrapper for !morphogen.flow<T> type.

    Represents a temporal flow block that executes operations over time.

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> flow_type = FlowType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(flow_type)  # !morphogen.flow<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get flow type for given element type.

        Args:
            element_type: MLIR element type (e.g., F32Type, F64Type)
            context: MLIR context

        Returns:
            Opaque flow type !morphogen.flow<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"flow<{element_str}>", context=context)


class StateType:
    """Wrapper for !morphogen.state<T> type.

    Represents persistent state storage for temporal simulations.

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> state_type = StateType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(state_type)  # !morphogen.state<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get state type for given element type.

        Args:
            element_type: MLIR element type (e.g., F32Type, F64Type)
            context: MLIR context

        Returns:
            Opaque state type !morphogen.state<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"state<{element_str}>", context=context)


class FlowCreateOp:
    """Operation: morphogen.temporal.flow.create

    Creates a flow block with temporal parameters.

    Syntax:
        %flow = morphogen.temporal.flow.create %dt, %steps : !morphogen.flow<f32>

    Arguments:
        - dt: Time step size (f32)
        - steps: Number of timesteps (index type)

    Results:
        - Flow handle of type !morphogen.flow<element_type>

    Lowering:
        Flow metadata is used to configure scf.for loop bounds
    """

    @staticmethod
    def create(
        dt: Any,  # ir.Value with f32 type
        steps: Any,  # ir.Value with index type
        element_type: Any,  # MLIR type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a flow creation operation.

        Args:
            dt: Time step size
            steps: Number of timesteps
            element_type: Element type for flow operations
            loc: Source location
            ip: Insertion point

        Returns:
            Flow handle value
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Create the flow type
            flow_type = FlowType.get(element_type, loc.context)

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [flow_type],
                [dt, steps]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.temporal.flow.create", context=loc.context
            )

            return result.results[0]


class FlowStepOp:
    """Operation: morphogen.temporal.flow.step

    Executes a single timestep within a flow, applying operations to state.

    Syntax:
        %new_state = morphogen.temporal.flow.step %flow, %state, %operations : !morphogen.state<f32>

    Arguments:
        - flow: Flow handle
        - state: Current state
        - operations: Operations to apply (represented as values)

    Results:
        - Updated state

    Lowering:
        Lowers to loop body operations with state updates
    """

    @staticmethod
    def create(
        flow: Any,  # ir.Value with flow type
        state: Any,  # ir.Value with state type
        operations: List[Any],  # List of ir.Value operations
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a flow step operation.

        Args:
            flow: Flow handle
            state: Current state
            operations: Operations to apply
            loc: Source location
            ip: Insertion point

        Returns:
            Updated state value
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same state type
            state_type = state.type

            # Create placeholder op
            operands = [flow, state] + operations
            result = builtin.UnrealizedConversionCastOp(
                [state_type],
                operands
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.temporal.flow.step", context=loc.context
            )

            return result.results[0]


class FlowRunOp:
    """Operation: morphogen.temporal.flow.run

    Executes a complete flow for all timesteps.

    Syntax:
        %final_state = morphogen.temporal.flow.run %flow, %initial_state : !morphogen.state<f32>

    Arguments:
        - flow: Flow handle with dt and steps
        - initial_state: Initial state

    Results:
        - Final state after all timesteps

    Lowering:
        Lowers to scf.for loop iterating over timesteps:

        scf.for %t = %c0 to %steps step %c1 iter_args(%state = %initial_state) {
            // Flow operations using state
            scf.yield %new_state
        }
    """

    @staticmethod
    def create(
        flow: Any,  # ir.Value with flow type
        initial_state: Any,  # ir.Value with state type
        body_builder: Optional[Any] = None,  # Optional callable to build flow body
        loc: Any = None,  # ir.Location
        ip: Any = None  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a flow run operation.

        Args:
            flow: Flow handle
            initial_state: Initial state
            body_builder: Optional callable to build flow body operations
            loc: Source location
            ip: Insertion point

        Returns:
            Final state value
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same state type
            state_type = initial_state.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [state_type],
                [flow, initial_state]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.temporal.flow.run", context=loc.context
            )

            # Store body builder as Python object if provided
            # (Will be used during lowering)
            if body_builder is not None:
                # For Phase 3, we'll handle body operations differently
                # This is a placeholder for future enhancement
                pass

            return result.results[0]


class StateCreateOp:
    """Operation: morphogen.temporal.state.create

    Allocates a persistent state container.

    Syntax:
        %state = morphogen.temporal.state.create %size, %initial_value : !morphogen.state<f32>

    Arguments:
        - size: Size of state container (index type)
        - initial_value: Initial value for all elements

    Results:
        - State container of type !morphogen.state<element_type>

    Lowering:
        Lowers to memref.alloc + initialization loop:

        %mem = memref.alloc(%size) : memref<?xf32>
        scf.for %i = %c0 to %size step %c1 {
            memref.store %initial_value, %mem[%i]
        }
    """

    @staticmethod
    def create(
        size: Any,  # ir.Value with index type
        initial_value: Any,  # ir.Value with element type
        element_type: Any,  # MLIR type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a state creation operation.

        Args:
            size: Size of state container
            initial_value: Initial value
            element_type: Element type
            loc: Source location
            ip: Insertion point

        Returns:
            State container value
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Create the state type
            state_type = StateType.get(element_type, loc.context)

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [state_type],
                [size, initial_value]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.temporal.state.create", context=loc.context
            )

            return result.results[0]


class StateUpdateOp:
    """Operation: morphogen.temporal.state.update

    Updates state values at specified indices.

    Syntax:
        %new_state = morphogen.temporal.state.update %state, %index, %value : !morphogen.state<f32>

    Arguments:
        - state: State container
        - index: Index to update (index type)
        - value: New value (element type)

    Results:
        - Updated state container

    Lowering:
        Lowers to memref.store:

        memref.store %value, %state_memref[%index]
    """

    @staticmethod
    def create(
        state: Any,  # ir.Value with state type
        index: Any,  # ir.Value with index type
        value: Any,  # ir.Value with element type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a state update operation.

        Args:
            state: State container
            index: Index to update
            value: New value
            loc: Source location
            ip: Insertion point

        Returns:
            Updated state container (SSA form requires new value)
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same state type
            state_type = state.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [state_type],
                [state, index, value]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.temporal.state.update", context=loc.context
            )

            return result.results[0]


class StateQueryOp:
    """Operation: morphogen.temporal.state.query

    Reads current state values at specified indices.

    Syntax:
        %value = morphogen.temporal.state.query %state, %index : f32

    Arguments:
        - state: State container
        - index: Index to read (index type)

    Results:
        - Value at index (element type)

    Lowering:
        Lowers to memref.load:

        %value = memref.load %state_memref[%index]
    """

    @staticmethod
    def create(
        state: Any,  # ir.Value with state type
        index: Any,  # ir.Value with index type
        element_type: Any,  # MLIR element type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a state query operation.

        Args:
            state: State container
            index: Index to read
            element_type: Element type to return
            loc: Source location
            ip: Insertion point

        Returns:
            Value at index
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is element type
            result_type = element_type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [result_type],
                [state, index]
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.temporal.state.query", context=loc.context
            )

            return result.results[0]


class TemporalDialect:
    """Temporal operations dialect.

    This class serves as a namespace for temporal dialect operations
    and provides utility methods for working with temporal types.

    Operations:
        - flow.create: Create flow with temporal parameters
        - flow.step: Execute single timestep
        - flow.run: Execute complete flow
        - state.create: Allocate state container
        - state.update: Update state values
        - state.query: Read state values

    Example:
        >>> from morphogen.mlir.dialects.temporal import TemporalDialect
        >>>
        >>> # Create a flow
        >>> dt = arith.ConstantOp(ir.F32Type.get(), 0.1)
        >>> steps = arith.ConstantOp(ir.IndexType.get(), 10)
        >>> flow = TemporalDialect.flow_create(dt, steps, f32, loc, ip)
        >>>
        >>> # Create state
        >>> size = arith.ConstantOp(ir.IndexType.get(), 100)
        >>> init_val = arith.ConstantOp(ir.F32Type.get(), 0.0)
        >>> state = TemporalDialect.state_create(size, init_val, f32, loc, ip)
        >>>
        >>> # Run flow
        >>> final_state = TemporalDialect.flow_run(flow, state, loc, ip)
    """

    # Flow operations
    flow_create = FlowCreateOp.create
    flow_step = FlowStepOp.create
    flow_run = FlowRunOp.create

    # State operations
    state_create = StateCreateOp.create
    state_update = StateUpdateOp.create
    state_query = StateQueryOp.create

    @staticmethod
    def is_temporal_op(op: Any) -> bool:
        """Check if an operation is a temporal operation.

        Args:
            op: MLIR operation to check

        Returns:
            True if op is a temporal operation
        """
        if not MLIR_AVAILABLE:
            return False

        if not hasattr(op, "attributes"):
            return False

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return False

        op_name = str(op_name_attr)
        return "morphogen.temporal." in op_name

    @staticmethod
    def get_temporal_op_name(op: Any) -> Optional[str]:
        """Get the temporal operation name.

        Args:
            op: Temporal operation

        Returns:
            Operation name (e.g., "morphogen.temporal.flow.run") or None
        """
        if not MLIR_AVAILABLE:
            return None

        if not hasattr(op, "attributes"):
            return None

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return None

        return str(op_name_attr).strip('"')


# Export public API
__all__ = [
    "FlowType",
    "StateType",
    "FlowCreateOp",
    "FlowStepOp",
    "FlowRunOp",
    "StateCreateOp",
    "StateUpdateOp",
    "StateQueryOp",
    "TemporalDialect",
    "MLIR_AVAILABLE",
]
