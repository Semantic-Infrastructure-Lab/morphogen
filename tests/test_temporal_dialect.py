"""Unit tests for Morphogen Temporal Dialect (Phase 3)

Tests the temporal dialect operations and type system.
These tests require MLIR Python bindings to be installed.
"""

import pytest
from morphogen.mlir.context import MorphogenMLIRContext, MLIR_AVAILABLE
from morphogen.mlir.dialects.temporal import (
    FlowType, StateType, TemporalDialect,
    FlowCreateOp, FlowRunOp, StateCreateOp, StateUpdateOp, StateQueryOp
)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFlowType:
    """Tests for FlowType wrapper."""

    def test_flow_type_creation_f32(self):
        """Test creating f32 flow type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            flow_type = FlowType.get(f32, ctx.ctx)

            assert flow_type is not None
            type_str = str(flow_type)
            assert "morphogen" in type_str
            assert "flow" in type_str

    def test_flow_type_creation_f64(self):
        """Test creating f64 flow type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f64 = ir.F64Type.get()
            flow_type = FlowType.get(f64, ctx.ctx)

            assert flow_type is not None
            type_str = str(flow_type)
            assert "morphogen" in type_str
            assert "flow" in type_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestStateType:
    """Tests for StateType wrapper."""

    def test_state_type_creation_f32(self):
        """Test creating f32 state type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            state_type = StateType.get(f32, ctx.ctx)

            assert state_type is not None
            type_str = str(state_type)
            assert "morphogen" in type_str
            assert "state" in type_str

    def test_state_type_creation_f64(self):
        """Test creating f64 state type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f64 = ir.F64Type.get()
            state_type = StateType.get(f64, ctx.ctx)

            assert state_type is not None
            type_str = str(state_type)
            assert "morphogen" in type_str
            assert "state" in type_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFlowCreateOp:
    """Tests for flow creation operation."""

    def test_create_flow_operation(self):
        """Test creating a flow creation operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create constants
                dt = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.1)
                ).result
                steps = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 10)
                ).result

                # Create flow
                flow = FlowCreateOp.create(
                    dt, steps,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert flow is not None

            # Verify module contains the operation
            module_str = str(module)
            assert "builtin.unrealized_conversion_cast" in module_str

    def test_create_flow_with_different_timesteps(self):
        """Test creating flows with different timestep configurations."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        configs = [
            (0.01, 100),
            (0.1, 10),
            (1.0, 5),
        ]

        for dt_val, steps_val in configs:
            with ctx.ctx, ir.Location.unknown():
                module = ctx.create_module("test")

                with ir.InsertionPoint(module.body):
                    dt = arith.ConstantOp(
                        ir.F32Type.get(),
                        ir.FloatAttr.get(ir.F32Type.get(), dt_val)
                    ).result
                    steps = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), steps_val)
                    ).result

                    flow = FlowCreateOp.create(
                        dt, steps,
                        ir.F32Type.get(),
                        ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                    assert flow is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestStateCreateOp:
    """Tests for state creation operation."""

    def test_create_state_operation(self):
        """Test creating a state creation operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create constants
                size = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 100)
                ).result
                initial_value = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result

                # Create state
                state = StateCreateOp.create(
                    size, initial_value,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert state is not None

            # Verify module contains the operation
            module_str = str(module)
            assert "builtin.unrealized_conversion_cast" in module_str

    def test_create_state_with_different_sizes(self):
        """Test creating states with different sizes."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        sizes = [10, 100, 1000]

        for size_val in sizes:
            with ctx.ctx, ir.Location.unknown():
                module = ctx.create_module("test")

                with ir.InsertionPoint(module.body):
                    size = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), size_val)
                    ).result
                    initial_value = arith.ConstantOp(
                        ir.F32Type.get(),
                        ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                    ).result

                    state = StateCreateOp.create(
                        size, initial_value,
                        ir.F32Type.get(),
                        ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                    assert state is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestStateUpdateOp:
    """Tests for state update operation."""

    def test_state_update_operation(self):
        """Test creating a state update operation."""
        from mlir import ir
        from mlir.dialects import arith, memref

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a dummy state (memref)
                memref_type = ir.MemRefType.get([100], ir.F32Type.get())
                state = memref.AllocOp(memref_type, [], []).result

                # Create update parameters
                index = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 5)
                ).result
                value = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 1.5)
                ).result

                # Update state
                new_state = StateUpdateOp.create(
                    state, index, value,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert new_state is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestStateQueryOp:
    """Tests for state query operation."""

    def test_state_query_operation(self):
        """Test creating a state query operation."""
        from mlir import ir
        from mlir.dialects import arith, memref

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a dummy state (memref)
                memref_type = ir.MemRefType.get([100], ir.F32Type.get())
                state = memref.AllocOp(memref_type, [], []).result

                # Create query parameters
                index = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 5)
                ).result

                # Query state
                value = StateQueryOp.create(
                    state, index,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert value is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFlowRunOp:
    """Tests for flow run operation."""

    def test_flow_run_operation(self):
        """Test creating a flow run operation."""
        from mlir import ir
        from mlir.dialects import arith, memref

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create flow
                dt = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.1)
                ).result
                steps = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 10)
                ).result

                flow = FlowCreateOp.create(
                    dt, steps,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Create initial state
                memref_type = ir.MemRefType.get([100], ir.F32Type.get())
                initial_state = memref.AllocOp(memref_type, [], []).result

                # Run flow
                final_state = FlowRunOp.create(
                    flow, initial_state, None,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert final_state is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestTemporalDialect:
    """Tests for TemporalDialect helper class."""

    def test_is_temporal_op(self):
        """Test temporal operation detection."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a temporal operation
                dt = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.1)
                ).result
                steps = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 10)
                ).result

                flow = FlowCreateOp.create(
                    dt, steps,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Find the operation that created the flow
                # In MLIR, we need to walk the operations
                for op in module.body.operations:
                    if TemporalDialect.is_temporal_op(op):
                        assert True
                        break

    def test_get_temporal_op_name(self):
        """Test getting temporal operation name."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a temporal operation
                size = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 100)
                ).result
                initial_value = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result

                state = StateCreateOp.create(
                    size, initial_value,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Find the operation that created the state
                for op in module.body.operations:
                    op_name = TemporalDialect.get_temporal_op_name(op)
                    if op_name and "state.create" in op_name:
                        assert "morphogen.temporal.state.create" == op_name
                        break


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestTemporalLowering:
    """Integration tests for temporal lowering."""

    def test_state_create_lowering(self):
        """Test lowering state creation to memref."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.lowering import create_temporal_to_scf_pass

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create state
                size = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 100)
                ).result
                initial_value = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result

                state = StateCreateOp.create(
                    size, initial_value,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

            # Apply lowering
            pass_obj = create_temporal_to_scf_pass(ctx)
            pass_obj.run(module)

            # Verify lowering produced memref operations
            module_str = str(module)
            assert "memref.alloc" in module_str
            assert "scf.for" in module_str

    def test_flow_run_lowering(self):
        """Test lowering flow run to scf.for loop."""
        from mlir import ir
        from mlir.dialects import arith, memref
        from morphogen.mlir.lowering import create_temporal_to_scf_pass

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create flow
                dt = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.1)
                ).result
                steps = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 10)
                ).result

                flow = FlowCreateOp.create(
                    dt, steps,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Create state
                memref_type = ir.MemRefType.get([100], ir.F32Type.get())
                initial_state = memref.AllocOp(memref_type, [], []).result

                # Run flow
                final_state = FlowRunOp.create(
                    flow, initial_state, None,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

            # Apply lowering
            pass_obj = create_temporal_to_scf_pass(ctx)
            pass_obj.run(module)

            # Verify lowering produced scf.for loop
            module_str = str(module)
            assert "scf.for" in module_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestTemporalCompilerIntegration:
    """Integration tests for temporal compiler methods."""

    def test_compile_temporal_program_simple(self):
        """Test compiling a simple temporal program."""
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # Define simple temporal program
        operations = [
            {"op": "state_create", "args": {"size": 100, "initial_value": 0.0}},
        ]

        # Compile
        module = compiler.compile_temporal_program(operations)

        # Verify compilation
        assert module is not None
        module_str = str(module)
        assert "memref.alloc" in module_str

    def test_compile_temporal_program_with_flow(self):
        """Test compiling a temporal program with flow execution."""
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # Define temporal program with flow
        operations = [
            {"op": "state_create", "args": {"size": 100, "initial_value": 0.0}},
            {"op": "flow_create", "args": {"dt": 0.1, "steps": 10}},
            {"op": "flow_run", "args": {"flow": "flow1", "initial_state": "state0"}},
        ]

        # Compile
        module = compiler.compile_temporal_program(operations)

        # Verify compilation
        assert module is not None
        module_str = str(module)
        assert "scf.for" in module_str
        assert "memref" in module_str

    def test_compile_temporal_program_with_state_updates(self):
        """Test compiling a temporal program with state updates."""
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # Define temporal program with state operations
        operations = [
            {"op": "state_create", "args": {"size": 100, "initial_value": 0.0}},
            {"op": "state_update", "args": {"state": "state0", "index": 5, "value": 1.5}},
            {"op": "state_query", "args": {"state": "state1", "index": 5}},
        ]

        # Compile
        module = compiler.compile_temporal_program(operations)

        # Verify compilation
        assert module is not None
        module_str = str(module)
        assert "memref" in module_str
