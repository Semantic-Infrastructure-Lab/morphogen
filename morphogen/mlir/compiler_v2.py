"""MLIR Compiler v2 for Morphogen (v0.7.0)

This module implements the new MLIR-based compiler for Morphogen, replacing
the text-based IR generation from v0.6.0 with real MLIR Python bindings.

Status: Phase 2 - Field Operations Dialect (Months 4-6)

Architecture:
    Morphogen AST → MLIR IR (real bindings) → Lowering Passes → LLVM → Native Code

This is a complete rewrite of morphogen/mlir/compiler.py to use actual MLIR
instead of string templates.

Phase 2 Additions:
- Field operations compilation
- FieldToSCF lowering pass integration
- Support for morphogen.field.* operations
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from ..ast.nodes import (
    Program, Statement, Expression,
    Function, Return, Assignment, Literal, Identifier, BinaryOp
)

if TYPE_CHECKING:
    from .context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, func, arith, scf, memref
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class MLIRCompilerV2:
    """MLIR compiler using real Python bindings.

    This replaces the legacy text-based IR generation with actual MLIR
    IR construction using Python bindings.

    Example:
        >>> from morphogen.mlir.context import MorphogenMLIRContext
        >>> ctx = MorphogenMLIRContext()
        >>> compiler = MLIRCompilerV2(ctx)
        >>> module = compiler.compile_program(ast_program)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize MLIR compiler v2.

        Args:
            context: Morphogen MLIR context

        Raises:
            RuntimeError: If MLIR bindings are not available
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError(
                "MLIR Python bindings required but not installed. "
                "Install: pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
            )

        self.context = context
        self.module: Optional[Any] = None  # Will be ir.Module when MLIR is available
        self.symbols: Dict[str, Any] = {}  # Will be Dict[str, ir.Value] when MLIR is available

    def compile_program(self, program: Program) -> Any:
        """Compile a Morphogen program to MLIR module.

        Args:
            program: Morphogen Program AST node

        Returns:
            MLIR Module

        Note:
            Full AST-to-MLIR compilation is not yet implemented.
            Use domain-specific compilation methods instead:
            - compile_field_program() for field operations
            - compile_temporal_program() for temporal operations
            - compile_agent_program() for agent operations
            - compile_audio_program() for audio operations
        """
        raise NotImplementedError(
            "Full AST compilation not implemented. "
            "Use domain-specific compile_*_program() methods instead."
        )

    def compile_literal(self, literal: Literal, builder: Optional[Any]) -> Any:
        """Compile literal using arith.constant.

        Args:
            literal: Literal AST node
            builder: MLIR insertion point

        Returns:
            MLIR Value representing the constant

        Example:
            3.0 → %0 = arith.constant 3.0 : f32
        """
        with self.context.ctx:
            if isinstance(literal.value, float):
                f32 = ir.F32Type.get()
                return arith.ConstantOp(
                    f32,
                    ir.FloatAttr.get(f32, literal.value)
                ).result
            elif isinstance(literal.value, int):
                i32 = ir.I32Type.get()
                return arith.ConstantOp(
                    i32,
                    ir.IntegerAttr.get(i32, literal.value)
                ).result
            elif isinstance(literal.value, bool):
                i1 = ir.IntegerType.get_signless(1)
                return arith.ConstantOp(
                    i1,
                    ir.IntegerAttr.get(i1, 1 if literal.value else 0)
                ).result
            else:
                raise ValueError(f"Unsupported literal type: {type(literal.value)}")

    def compile_binary_op(self, binop: BinaryOp, builder: Optional[Any]) -> Any:
        """Compile binary operation.

        Args:
            binop: BinaryOp AST node
            builder: MLIR insertion point

        Returns:
            MLIR Value representing the result

        Example:
            x + y → %result = arith.addf %x, %y : f32

        Note:
            General binary operation compilation is not yet implemented.
            Binary operations are handled within domain-specific lowering passes.
        """
        raise NotImplementedError(
            "General binary operation compilation not implemented. "
            "Binary ops are handled in domain-specific lowering passes."
        )

    # Phase 2: Field Operations Support

    def compile_field_create(
        self,
        width: Any,
        height: Any,
        fill_value: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile field creation operation.

        Args:
            width: Width dimension (ir.Value)
            height: Height dimension (ir.Value)
            fill_value: Initial fill value (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Field value

        Example:
            field.alloc((256, 256), fill_value=0.0)
            → %field = morphogen.field.create %c256, %c256, %c0_f32 : !morphogen.field<f32>
        """
        from .dialects.field import FieldDialect
        return FieldDialect.create(width, height, fill_value, element_type, loc, ip)

    def compile_field_gradient(
        self,
        field: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile gradient operation.

        Args:
            field: Input field (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Gradient field

        Example:
            field.gradient(field)
            → %grad = morphogen.field.gradient %field : !morphogen.field<f32>
        """
        from .dialects.field import FieldDialect
        return FieldDialect.gradient(field, loc, ip)

    def compile_field_laplacian(
        self,
        field: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile Laplacian operation.

        Args:
            field: Input field (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Laplacian field

        Example:
            field.laplacian(field)
            → %lapl = morphogen.field.laplacian %field : !morphogen.field<f32>
        """
        from .dialects.field import FieldDialect
        return FieldDialect.laplacian(field, loc, ip)

    def compile_field_diffuse(
        self,
        field: Any,
        rate: Any,
        dt: Any,
        iterations: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile diffusion operation.

        Args:
            field: Input field (ir.Value)
            rate: Diffusion rate (ir.Value)
            dt: Time step (ir.Value)
            iterations: Number of iterations (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Diffused field

        Example:
            field.diffuse(field, rate=0.1, dt=0.01, iterations=10)
            → %diffused = morphogen.field.diffuse %field, %rate, %dt, %iters
        """
        from .dialects.field import FieldDialect
        return FieldDialect.diffuse(field, rate, dt, iterations, loc, ip)

    def apply_field_lowering(self, module: Any) -> None:
        """Apply field-to-SCF lowering pass to module.

        This transforms high-level field operations into low-level
        SCF loops and memref operations.

        Args:
            module: MLIR module to transform (in-place)

        Example:
            >>> compiler.apply_field_lowering(module)
            # Field ops → SCF loops + memref
        """
        from .lowering import create_field_to_scf_pass

        pass_obj = create_field_to_scf_pass(self.context)
        pass_obj.run(module)

    def compile_field_program(
        self,
        operations: List[Dict[str, Any]],
        module_name: str = "field_program"
    ) -> Any:
        """Compile a sequence of field operations to MLIR module.

        This is a convenience method for Phase 2 to compile field operations
        without requiring full AST support.

        Args:
            operations: List of operation dictionaries with keys:
                - op: Operation name ("create", "gradient", "laplacian", "diffuse")
                - args: Dictionary of arguments
            module_name: Module name

        Returns:
            MLIR Module with lowered operations

        Example:
            >>> ops = [
            ...     {"op": "create", "args": {"width": 256, "height": 256, "fill": 0.0}},
            ...     {"op": "gradient", "args": {"field": "field0"}},
            ... ]
            >>> module = compiler.compile_field_program(ops)
        """
        with self.context.ctx, ir.Location.unknown():
            module = self.context.create_module(module_name)

            # Create a wrapper function
            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="main", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Process operations
                    results = {}
                    for i, operation in enumerate(operations):
                        op_name = operation["op"]
                        args = operation["args"]

                        if op_name == "create":
                            # Create constants for dimensions
                            width_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["width"])
                            ).result
                            height_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["height"])
                            ).result
                            fill_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["fill"])
                            ).result

                            result = self.compile_field_create(
                                width_val, height_val, fill_val, f32, loc, ip
                            )
                            results[f"field{i}"] = result

                        elif op_name == "gradient":
                            field_name = args["field"]
                            field_val = results[field_name]
                            result = self.compile_field_gradient(field_val, loc, ip)
                            results[f"grad{i}"] = result

                        elif op_name == "laplacian":
                            field_name = args["field"]
                            field_val = results[field_name]
                            result = self.compile_field_laplacian(field_val, loc, ip)
                            results[f"lapl{i}"] = result

                        elif op_name == "diffuse":
                            field_name = args["field"]
                            field_val = results[field_name]
                            rate_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["rate"])
                            ).result
                            dt_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["dt"])
                            ).result
                            iters_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["iterations"])
                            ).result

                            result = self.compile_field_diffuse(
                                field_val, rate_val, dt_val, iters_val, loc, ip
                            )
                            results[f"diffused{i}"] = result

                    # Return
                    func.ReturnOp([])

            # Apply lowering passes
            self.apply_field_lowering(module)

            return module

    # Phase 3: Temporal Operations Support

    def compile_flow_create(
        self,
        dt: Any,
        steps: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile flow creation operation.

        Args:
            dt: Time step size (ir.Value)
            steps: Number of timesteps (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Flow handle value

        Example:
            flow_create(dt=0.1, steps=10)
            → %flow = morphogen.temporal.flow.create %dt, %steps : !morphogen.flow<f32>
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.flow_create(dt, steps, element_type, loc, ip)

    def compile_flow_run(
        self,
        flow: Any,
        initial_state: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile flow run operation.

        Args:
            flow: Flow handle (ir.Value)
            initial_state: Initial state (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Final state value

        Example:
            flow_run(flow, initial_state)
            → %final = morphogen.temporal.flow.run %flow, %initial_state
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.flow_run(flow, initial_state, None, loc, ip)

    def compile_state_create(
        self,
        size: Any,
        initial_value: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile state creation operation.

        Args:
            size: Size of state container (ir.Value)
            initial_value: Initial value (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            State container value

        Example:
            state_create(size=100, initial_value=0.0)
            → %state = morphogen.temporal.state.create %size, %init_val : !morphogen.state<f32>
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.state_create(size, initial_value, element_type, loc, ip)

    def compile_state_update(
        self,
        state: Any,
        index: Any,
        value: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile state update operation.

        Args:
            state: State container (ir.Value)
            index: Index to update (ir.Value)
            value: New value (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Updated state container

        Example:
            state_update(state, index=5, value=1.5)
            → %new_state = morphogen.temporal.state.update %state, %idx, %val
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.state_update(state, index, value, loc, ip)

    def compile_state_query(
        self,
        state: Any,
        index: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile state query operation.

        Args:
            state: State container (ir.Value)
            index: Index to read (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Value at index

        Example:
            state_query(state, index=5)
            → %value = morphogen.temporal.state.query %state, %idx : f32
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.state_query(state, index, element_type, loc, ip)

    def apply_temporal_lowering(self, module: Any) -> None:
        """Apply temporal-to-SCF lowering pass to module.

        This transforms high-level temporal operations into low-level
        SCF loops with memref-based state management.

        Args:
            module: MLIR module to transform (in-place)

        Example:
            >>> compiler.apply_temporal_lowering(module)
            # Temporal ops → SCF loops + memref
        """
        from .lowering import create_temporal_to_scf_pass

        pass_obj = create_temporal_to_scf_pass(self.context)
        pass_obj.run(module)

    # Phase 4: Agent Operations Support

    def compile_agent_spawn(
        self,
        count: Any,
        position_x: Any,
        position_y: Any,
        velocity_x: Any,
        velocity_y: Any,
        state: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile agent spawn operation.

        Args:
            count: Number of agents to spawn (ir.Value)
            position_x: Initial x position (ir.Value)
            position_y: Initial y position (ir.Value)
            velocity_x: Initial x velocity (ir.Value)
            velocity_y: Initial y velocity (ir.Value)
            state: Initial state value (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Agent collection value

        Example:
            agent_spawn(count=100, pos_x=0.0, pos_y=0.0, vel_x=0.1, vel_y=0.0, state=0.0)
            → %agents = morphogen.agent.spawn %count, %pos_x, %pos_y, %vel_x, %vel_y, %state : !morphogen.agent<f32>
        """
        from .dialects.agent import AgentDialect
        return AgentDialect.spawn(count, position_x, position_y, velocity_x, velocity_y, state, element_type, loc, ip)

    def compile_agent_update(
        self,
        agents: Any,
        index: Any,
        property_index: Any,
        value: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile agent update operation.

        Args:
            agents: Agent collection (ir.Value)
            index: Agent index to update (ir.Value)
            property_index: Property index (ir.Value)
            value: New property value (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Updated agent collection

        Example:
            agent_update(agents, index=0, property=0, value=1.5)
            → %agents_new = morphogen.agent.update %agents, %idx, %prop, %val
        """
        from .dialects.agent import AgentDialect
        return AgentDialect.update(agents, index, property_index, value, loc, ip)

    def compile_agent_query(
        self,
        agents: Any,
        index: Any,
        property_index: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile agent query operation.

        Args:
            agents: Agent collection (ir.Value)
            index: Agent index to query (ir.Value)
            property_index: Property index to read (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Property value

        Example:
            agent_query(agents, index=0, property=0)
            → %value = morphogen.agent.query %agents, %idx, %prop : f32
        """
        from .dialects.agent import AgentDialect
        return AgentDialect.query(agents, index, property_index, element_type, loc, ip)

    def compile_agent_behavior(
        self,
        agents: Any,
        behavior_type: str,
        params: List[Any],
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile agent behavior operation.

        Args:
            agents: Agent collection (ir.Value)
            behavior_type: Type of behavior ("move", "seek", "bounce", etc.)
            params: Optional behavior parameters (list of ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Updated agent collection

        Example:
            agent_behavior(agents, "move", [])
            → %agents_new = morphogen.agent.behavior %agents : !morphogen.agent<f32>
        """
        from .dialects.agent import AgentDialect
        return AgentDialect.behavior(agents, behavior_type, params, loc, ip)

    def apply_agent_lowering(self, module: Any) -> None:
        """Apply agent-to-SCF lowering pass to module.

        This transforms high-level agent operations into low-level
        SCF loops with memref-based agent storage.

        Args:
            module: MLIR module to transform (in-place)

        Example:
            >>> compiler.apply_agent_lowering(module)
            # Agent ops → SCF loops + memref
        """
        from .lowering import create_agent_to_scf_pass

        pass_obj = create_agent_to_scf_pass(self.context)
        pass_obj.run(module)

    def compile_agent_program(
        self,
        operations: List[Dict[str, Any]],
        module_name: str = "agent_program"
    ) -> Any:
        """Compile a sequence of agent operations to MLIR module.

        This is a convenience method for Phase 4 to compile agent operations
        without requiring full AST support.

        Args:
            operations: List of operation dictionaries with keys:
                - op: Operation name ("spawn", "update", "query", "behavior")
                - args: Dictionary of arguments
            module_name: Module name

        Returns:
            MLIR Module with lowered operations

        Example:
            >>> ops = [
            ...     {"op": "spawn", "args": {"count": 100, "pos_x": 0.0, "pos_y": 0.0,
            ...                              "vel_x": 0.1, "vel_y": 0.0, "state": 0.0}},
            ...     {"op": "behavior", "args": {"agents": "agents0", "behavior": "move"}},
            ... ]
            >>> module = compiler.compile_agent_program(ops)
        """
        with self.context.ctx, ir.Location.unknown():
            module = self.context.create_module(module_name)

            # Create a wrapper function
            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="main", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Process operations
                    results = {}
                    for i, operation in enumerate(operations):
                        op_name = operation["op"]
                        args = operation["args"]

                        if op_name == "spawn":
                            # Create constants
                            count_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["count"])
                            ).result
                            pos_x_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["pos_x"])
                            ).result
                            pos_y_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["pos_y"])
                            ).result
                            vel_x_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["vel_x"])
                            ).result
                            vel_y_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["vel_y"])
                            ).result
                            state_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["state"])
                            ).result

                            result = self.compile_agent_spawn(
                                count_val, pos_x_val, pos_y_val, vel_x_val, vel_y_val, state_val, f32, loc, ip
                            )
                            results[f"agents{i}"] = result

                        elif op_name == "update":
                            agents_name = args["agents"]
                            agents_val = results[agents_name]
                            index_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["index"])
                            ).result
                            property_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["property"])
                            ).result
                            value_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["value"])
                            ).result

                            result = self.compile_agent_update(
                                agents_val, index_val, property_val, value_val, loc, ip
                            )
                            results[f"agents{i}"] = result

                        elif op_name == "query":
                            agents_name = args["agents"]
                            agents_val = results[agents_name]
                            index_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["index"])
                            ).result
                            property_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["property"])
                            ).result

                            result = self.compile_agent_query(
                                agents_val, index_val, property_val, f32, loc, ip
                            )
                            results[f"value{i}"] = result

                        elif op_name == "behavior":
                            agents_name = args["agents"]
                            agents_val = results[agents_name]
                            behavior_type = args.get("behavior", "move")
                            params = []  # Can be extended for parameterized behaviors

                            result = self.compile_agent_behavior(
                                agents_val, behavior_type, params, loc, ip
                            )
                            results[f"agents{i}"] = result

                    # Return
                    func.ReturnOp([])

            # Apply lowering passes
            self.apply_agent_lowering(module)

            return module

    def compile_temporal_program(
        self,
        operations: List[Dict[str, Any]],
        module_name: str = "temporal_program"
    ) -> Any:
        """Compile a sequence of temporal operations to MLIR module.

        This is a convenience method for Phase 3 to compile temporal operations
        without requiring full AST support.

        Args:
            operations: List of operation dictionaries with keys:
                - op: Operation name ("flow_create", "state_create", "flow_run", etc.)
                - args: Dictionary of arguments
            module_name: Module name

        Returns:
            MLIR Module with lowered operations

        Example:
            >>> ops = [
            ...     {"op": "state_create", "args": {"size": 100, "initial_value": 0.0}},
            ...     {"op": "flow_create", "args": {"dt": 0.1, "steps": 10}},
            ...     {"op": "flow_run", "args": {"flow": "flow0", "initial_state": "state0"}},
            ... ]
            >>> module = compiler.compile_temporal_program(ops)
        """
        with self.context.ctx, ir.Location.unknown():
            module = self.context.create_module(module_name)

            # Create a wrapper function
            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="main", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Process operations
                    results = {}
                    for i, operation in enumerate(operations):
                        op_name = operation["op"]
                        args = operation["args"]

                        if op_name == "state_create":
                            # Create constants
                            size_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["size"])
                            ).result
                            init_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["initial_value"])
                            ).result

                            result = self.compile_state_create(
                                size_val, init_val, f32, loc, ip
                            )
                            results[f"state{i}"] = result

                        elif op_name == "flow_create":
                            # Create constants
                            dt_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["dt"])
                            ).result
                            steps_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["steps"])
                            ).result

                            result = self.compile_flow_create(
                                dt_val, steps_val, f32, loc, ip
                            )
                            results[f"flow{i}"] = result

                        elif op_name == "flow_run":
                            flow_name = args["flow"]
                            state_name = args["initial_state"]
                            flow_val = results[flow_name]
                            state_val = results[state_name]

                            result = self.compile_flow_run(flow_val, state_val, loc, ip)
                            results[f"final_state{i}"] = result

                        elif op_name == "state_update":
                            state_name = args["state"]
                            state_val = results[state_name]
                            index_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["index"])
                            ).result
                            value_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["value"])
                            ).result

                            result = self.compile_state_update(
                                state_val, index_val, value_val, loc, ip
                            )
                            results[f"state{i}"] = result

                        elif op_name == "state_query":
                            state_name = args["state"]
                            state_val = results[state_name]
                            index_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["index"])
                            ).result

                            result = self.compile_state_query(
                                state_val, index_val, f32, loc, ip
                            )
                            results[f"value{i}"] = result

                    # Return
                    func.ReturnOp([])

            # Apply lowering passes
            self.apply_temporal_lowering(module)

            return module

    # Phase 5: Audio Operations Support

    def compile_audio_buffer_create(
        self,
        sample_rate: Any,
        channels: Any,
        duration: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile audio buffer creation operation.

        Args:
            sample_rate: Sample rate in Hz (ir.Value, index type)
            channels: Number of channels (ir.Value, index type)
            duration: Duration in seconds (ir.Value, f32 type)
            loc: Source location
            ip: Insertion point

        Returns:
            Audio buffer value

        Example:
            audio_buffer_create(sample_rate=44100, channels=1, duration=1.0)
            → %buf = morphogen.audio.buffer.create %sr, %ch, %dur : !morphogen.audio<44100, 1>
        """
        from .dialects.audio import AudioDialect
        return AudioDialect.buffer_create(sample_rate, channels, duration, loc, ip)

    def compile_audio_oscillator(
        self,
        buffer: Any,
        waveform: Any,
        frequency: Any,
        phase: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile oscillator operation.

        Args:
            buffer: Target audio buffer (ir.Value)
            waveform: Waveform type (ir.Value, index: 0=sine, 1=square, 2=saw, 3=triangle)
            frequency: Frequency in Hz (ir.Value, f32 type)
            phase: Initial phase in radians (ir.Value, f32 type)
            loc: Source location
            ip: Insertion point

        Returns:
            Audio buffer with oscillator output

        Example:
            audio_oscillator(buffer, waveform=0, freq=440.0, phase=0.0)
            → %osc = morphogen.audio.oscillator %buf, %wf, %freq, %phase
        """
        from .dialects.audio import AudioDialect
        return AudioDialect.oscillator(buffer, waveform, frequency, phase, loc, ip)

    def compile_audio_envelope(
        self,
        buffer: Any,
        attack: Any,
        decay: Any,
        sustain: Any,
        release: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile ADSR envelope operation.

        Args:
            buffer: Input audio buffer (ir.Value)
            attack: Attack time in seconds (ir.Value, f32 type)
            decay: Decay time in seconds (ir.Value, f32 type)
            sustain: Sustain level 0.0-1.0 (ir.Value, f32 type)
            release: Release time in seconds (ir.Value, f32 type)
            loc: Source location
            ip: Insertion point

        Returns:
            Audio buffer with envelope applied

        Example:
            audio_envelope(buffer, attack=0.01, decay=0.1, sustain=0.7, release=0.2)
            → %env = morphogen.audio.envelope %buf, %a, %d, %s, %r
        """
        from .dialects.audio import AudioDialect
        return AudioDialect.envelope(buffer, attack, decay, sustain, release, loc, ip)

    def compile_audio_filter(
        self,
        buffer: Any,
        filter_type: Any,
        cutoff: Any,
        resonance: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile filter operation.

        Args:
            buffer: Input audio buffer (ir.Value)
            filter_type: Filter type (ir.Value, index: 0=lowpass, 1=highpass, 2=bandpass)
            cutoff: Cutoff frequency in Hz (ir.Value, f32 type)
            resonance: Resonance/Q factor (ir.Value, f32 type)
            loc: Source location
            ip: Insertion point

        Returns:
            Filtered audio buffer

        Example:
            audio_filter(buffer, filter_type=0, cutoff=1000.0, resonance=1.0)
            → %filt = morphogen.audio.filter %buf, %type, %cutoff, %Q
        """
        from .dialects.audio import AudioDialect
        return AudioDialect.filter(buffer, filter_type, cutoff, resonance, loc, ip)

    def compile_audio_mix(
        self,
        buffers: List[Any],
        gains: List[Any],
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile audio mix operation.

        Args:
            buffers: List of audio buffers to mix (list of ir.Value)
            gains: List of gain values (list of ir.Value, f32 type)
            loc: Source location
            ip: Insertion point

        Returns:
            Mixed audio buffer

        Example:
            audio_mix([buf1, buf2], [0.5, 0.5])
            → %mix = morphogen.audio.mix %buf1, %buf2, %g1, %g2
        """
        from .dialects.audio import AudioDialect
        return AudioDialect.mix(buffers, gains, loc, ip)

    def apply_audio_lowering(self, module: Any) -> None:
        """Apply audio-to-SCF lowering pass to module.

        This transforms high-level audio operations into low-level
        SCF loops with memref-based audio buffer storage.

        Args:
            module: MLIR module to transform (in-place)

        Example:
            >>> compiler.apply_audio_lowering(module)
            # Audio ops → SCF loops + memref
        """
        from .lowering import create_audio_to_scf_pass

        pass_obj = create_audio_to_scf_pass(self.context)
        pass_obj.run(module)

    def compile_audio_program(
        self,
        operations: List[Dict[str, Any]],
        module_name: str = "audio_program"
    ) -> Any:
        """Compile a sequence of audio operations to MLIR module.

        This is a convenience method for Phase 5 to compile audio operations
        without requiring full AST support.

        Args:
            operations: List of operation dictionaries with keys:
                - op: Operation name ("buffer_create", "oscillator", "envelope", "filter", "mix")
                - args: Dictionary of arguments
            module_name: Module name

        Returns:
            MLIR Module with lowered operations

        Example:
            >>> ops = [
            ...     {"op": "buffer_create", "args": {"sample_rate": 44100, "channels": 1, "duration": 1.0}},
            ...     {"op": "oscillator", "args": {"buffer": "buf0", "waveform": 0, "freq": 440.0, "phase": 0.0}},
            ...     {"op": "envelope", "args": {"buffer": "osc0", "attack": 0.01, "decay": 0.1, "sustain": 0.7, "release": 0.2}},
            ... ]
            >>> module = compiler.compile_audio_program(ops)
        """
        with self.context.ctx, ir.Location.unknown():
            module = self.context.create_module(module_name)

            # Create a wrapper function
            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                index = ir.IndexType.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="main", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Process operations
                    results = {}
                    for i, operation in enumerate(operations):
                        op_name = operation["op"]
                        args = operation["args"]

                        if op_name == "buffer_create":
                            # Create constants for buffer parameters
                            sample_rate_val = arith.ConstantOp(
                                index,
                                ir.IntegerAttr.get(index, args["sample_rate"])
                            ).result
                            channels_val = arith.ConstantOp(
                                index,
                                ir.IntegerAttr.get(index, args["channels"])
                            ).result
                            duration_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["duration"])
                            ).result

                            result = self.compile_audio_buffer_create(
                                sample_rate_val, channels_val, duration_val, loc, ip
                            )
                            results[f"buf{i}"] = result

                        elif op_name == "oscillator":
                            buffer_name = args["buffer"]
                            buffer_val = results[buffer_name]
                            waveform_val = arith.ConstantOp(
                                index,
                                ir.IntegerAttr.get(index, args["waveform"])
                            ).result
                            freq_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["freq"])
                            ).result
                            phase_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["phase"])
                            ).result

                            result = self.compile_audio_oscillator(
                                buffer_val, waveform_val, freq_val, phase_val, loc, ip
                            )
                            results[f"osc{i}"] = result

                        elif op_name == "envelope":
                            buffer_name = args["buffer"]
                            buffer_val = results[buffer_name]
                            attack_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["attack"])
                            ).result
                            decay_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["decay"])
                            ).result
                            sustain_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["sustain"])
                            ).result
                            release_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["release"])
                            ).result

                            result = self.compile_audio_envelope(
                                buffer_val, attack_val, decay_val, sustain_val, release_val, loc, ip
                            )
                            results[f"env{i}"] = result

                        elif op_name == "filter":
                            buffer_name = args["buffer"]
                            buffer_val = results[buffer_name]
                            filter_type_val = arith.ConstantOp(
                                index,
                                ir.IntegerAttr.get(index, args["filter_type"])
                            ).result
                            cutoff_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["cutoff"])
                            ).result
                            resonance_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["resonance"])
                            ).result

                            result = self.compile_audio_filter(
                                buffer_val, filter_type_val, cutoff_val, resonance_val, loc, ip
                            )
                            results[f"filt{i}"] = result

                        elif op_name == "mix":
                            buffer_names = args["buffers"]
                            buffer_vals = [results[name] for name in buffer_names]
                            gain_vals = [
                                arith.ConstantOp(
                                    f32,
                                    ir.FloatAttr.get(f32, gain)
                                ).result
                                for gain in args["gains"]
                            ]

                            result = self.compile_audio_mix(
                                buffer_vals, gain_vals, loc, ip
                            )
                            results[f"mix{i}"] = result

                    # Return
                    func.ReturnOp([])

            # Apply lowering passes
            self.apply_audio_lowering(module)

            return module


# Export for backward compatibility check
def is_legacy_compiler() -> bool:
    """Check if we're using the legacy text-based compiler.

    Returns:
        True if legacy compiler, False if using real MLIR
    """
    return not MLIR_AVAILABLE
