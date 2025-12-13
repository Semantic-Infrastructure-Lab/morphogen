"""Unit tests for Morphogen Agent Dialect (Phase 4)

Tests the agent dialect operations, type system, and lowering passes.
These tests require MLIR Python bindings to be installed.
"""

import pytest
from morphogen.mlir.context import MorphogenMLIRContext, MLIR_AVAILABLE
from morphogen.mlir.dialects.agent import (
    AgentType, AgentDialect,
    AgentSpawnOp, AgentUpdateOp, AgentQueryOp, AgentBehaviorOp
)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentType:
    """Tests for AgentType wrapper."""

    def test_agent_type_creation_f32(self):
        """Test creating f32 agent type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            agent_type = AgentType.get(f32, ctx.ctx)

            assert agent_type is not None
            type_str = str(agent_type)
            assert "morphogen" in type_str
            assert "agent" in type_str

    def test_agent_type_creation_f64(self):
        """Test creating f64 agent type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f64 = ir.F64Type.get()
            agent_type = AgentType.get(f64, ctx.ctx)

            assert agent_type is not None
            type_str = str(agent_type)
            assert "morphogen" in type_str
            assert "agent" in type_str

    def test_agent_type_string_representation(self):
        """Test agent type string representation contains element type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            agent_type = AgentType.get(f32, ctx.ctx)

            type_str = str(agent_type)
            assert "f32" in type_str or "float" in type_str.lower()


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentSpawnOp:
    """Tests for agent spawn operation."""

    def test_spawn_agents_basic(self):
        """Test spawning agents with basic properties."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_spawn")

            with ir.InsertionPoint(module.body):
                # Create constants
                count = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 100)
                ).result
                pos_x = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result
                pos_y = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result
                vel_x = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.1)
                ).result
                vel_y = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result
                state = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result

                # Spawn agents
                agents = AgentSpawnOp.create(
                    count, pos_x, pos_y, vel_x, vel_y, state,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert agents is not None

            # Verify module contains the operation
            module_str = str(module)
            assert "builtin.unrealized_conversion_cast" in module_str

    def test_spawn_different_agent_counts(self):
        """Test spawning different numbers of agents."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        agent_counts = [1, 10, 100, 1000, 10000]

        for count_val in agent_counts:
            with ctx.ctx, ir.Location.unknown():
                module = ctx.create_module("test")

                with ir.InsertionPoint(module.body):
                    count = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), count_val)
                    ).result
                    pos_x = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    pos_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    vel_x = arith.ConstantOp(ir.F32Type.get(), 0.1).result
                    vel_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    state = arith.ConstantOp(ir.F32Type.get(), 0.0).result

                    agents = AgentSpawnOp.create(
                        count, pos_x, pos_y, vel_x, vel_y, state,
                        ir.F32Type.get(), ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                    assert agents is not None

    def test_spawn_agents_with_different_initial_positions(self):
        """Test spawning agents at different initial positions."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        positions = [(0.0, 0.0), (1.0, 2.0), (-5.0, 3.5), (100.0, 200.0)]

        for pos_x_val, pos_y_val in positions:
            with ctx.ctx, ir.Location.unknown():
                module = ctx.create_module("test")

                with ir.InsertionPoint(module.body):
                    count = arith.ConstantOp(ir.IndexType.get(), 10).result
                    pos_x = arith.ConstantOp(ir.F32Type.get(), pos_x_val).result
                    pos_y = arith.ConstantOp(ir.F32Type.get(), pos_y_val).result
                    vel_x = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    vel_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    state = arith.ConstantOp(ir.F32Type.get(), 0.0).result

                    agents = AgentSpawnOp.create(
                        count, pos_x, pos_y, vel_x, vel_y, state,
                        ir.F32Type.get(), ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                    assert agents is not None

    def test_spawn_agents_with_different_velocities(self):
        """Test spawning agents with different initial velocities."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        velocities = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.5, 0.5), (-0.3, 0.2)]

        for vel_x_val, vel_y_val in velocities:
            with ctx.ctx, ir.Location.unknown():
                module = ctx.create_module("test")

                with ir.InsertionPoint(module.body):
                    count = arith.ConstantOp(ir.IndexType.get(), 50).result
                    pos_x = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    pos_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                    vel_x = arith.ConstantOp(ir.F32Type.get(), vel_x_val).result
                    vel_y = arith.ConstantOp(ir.F32Type.get(), vel_y_val).result
                    state = arith.ConstantOp(ir.F32Type.get(), 0.0).result

                    agents = AgentSpawnOp.create(
                        count, pos_x, pos_y, vel_x, vel_y, state,
                        ir.F32Type.get(), ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                    assert agents is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentUpdateOp:
    """Tests for agent update operation."""

    def test_update_agent_property(self):
        """Test updating an agent property."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_update")

            with ir.InsertionPoint(module.body):
                # Create agents first
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                pos_x = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                pos_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                vel_x = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                vel_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                state = arith.ConstantOp(ir.F32Type.get(), 0.0).result

                agents = AgentSpawnOp.create(
                    count, pos_x, pos_y, vel_x, vel_y, state,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Update agent property
                index = arith.ConstantOp(ir.IndexType.get(), 0).result
                property_idx = arith.ConstantOp(ir.IndexType.get(), 0).result
                new_value = arith.ConstantOp(ir.F32Type.get(), 1.5).result

                updated_agents = AgentUpdateOp.create(
                    agents, index, property_idx, new_value,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert updated_agents is not None

    def test_update_multiple_properties(self):
        """Test updating multiple agent properties."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Update multiple properties
                index = arith.ConstantOp(ir.IndexType.get(), 0).result
                for prop_idx in range(5):
                    property_idx = arith.ConstantOp(ir.IndexType.get(), prop_idx).result
                    new_value = arith.ConstantOp(ir.F32Type.get(), float(prop_idx)).result

                    agents = AgentUpdateOp.create(
                        agents, index, property_idx, new_value,
                        ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                assert agents is not None

    def test_update_different_agents(self):
        """Test updating properties of different agents."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 20).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Update different agents
                property_idx = arith.ConstantOp(ir.IndexType.get(), 0).result
                for agent_idx in [0, 5, 10, 15]:
                    index = arith.ConstantOp(ir.IndexType.get(), agent_idx).result
                    new_value = arith.ConstantOp(ir.F32Type.get(), float(agent_idx)).result

                    agents = AgentUpdateOp.create(
                        agents, index, property_idx, new_value,
                        ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                assert agents is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentQueryOp:
    """Tests for agent query operation."""

    def test_query_agent_property(self):
        """Test querying an agent property."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_query")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 1.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 2.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.2).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Query agent property
                index = arith.ConstantOp(ir.IndexType.get(), 0).result
                property_idx = arith.ConstantOp(ir.IndexType.get(), 0).result

                value = AgentQueryOp.create(
                    agents, index, property_idx,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert value is not None

    def test_query_all_properties(self):
        """Test querying all agent properties."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 5).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 1.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 2.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.2).result,
                    arith.ConstantOp(ir.F32Type.get(), 3.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Query all 5 properties
                index = arith.ConstantOp(ir.IndexType.get(), 0).result
                for prop_idx in range(5):
                    property_idx = arith.ConstantOp(ir.IndexType.get(), prop_idx).result
                    value = AgentQueryOp.create(
                        agents, index, property_idx,
                        ir.F32Type.get(), ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )
                    assert value is not None

    def test_query_different_agents(self):
        """Test querying properties from different agents."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 100).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Query different agents
                property_idx = arith.ConstantOp(ir.IndexType.get(), 0).result
                for agent_idx in [0, 25, 50, 75, 99]:
                    index = arith.ConstantOp(ir.IndexType.get(), agent_idx).result
                    value = AgentQueryOp.create(
                        agents, index, property_idx,
                        ir.F32Type.get(), ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )
                    assert value is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentBehaviorOp:
    """Tests for agent behavior operation."""

    def test_behavior_move(self):
        """Test move behavior operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_behavior")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 50).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Apply move behavior
                updated_agents = AgentBehaviorOp.create(
                    agents, "move", [],
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert updated_agents is not None

    def test_behavior_seek(self):
        """Test seek behavior operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 30).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Seek behavior with target
                target_x = arith.ConstantOp(ir.F32Type.get(), 10.0).result
                target_y = arith.ConstantOp(ir.F32Type.get(), 10.0).result
                speed = arith.ConstantOp(ir.F32Type.get(), 0.1).result

                updated_agents = AgentBehaviorOp.create(
                    agents, "seek", [target_x, target_y, speed],
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert updated_agents is not None

    def test_behavior_bounce(self):
        """Test bounce behavior operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 20).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 5.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 5.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.5).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.5).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Bounce behavior with boundaries
                min_x = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                max_x = arith.ConstantOp(ir.F32Type.get(), 10.0).result
                min_y = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                max_y = arith.ConstantOp(ir.F32Type.get(), 10.0).result

                updated_agents = AgentBehaviorOp.create(
                    agents, "bounce", [min_x, max_x, min_y, max_y],
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert updated_agents is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentDialect:
    """Tests for AgentDialect utility methods."""

    def test_property_index_constants(self):
        """Test property index constants."""
        assert AgentDialect.PROP_POS_X == 0
        assert AgentDialect.PROP_POS_Y == 1
        assert AgentDialect.PROP_VEL_X == 2
        assert AgentDialect.PROP_VEL_Y == 3
        assert AgentDialect.PROP_STATE == 4
        assert AgentDialect.NUM_BASE_PROPERTIES == 5

    def test_get_property_index_constant(self):
        """Test getting property index by name."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            assert AgentDialect.get_property_index_constant("pos_x", ctx.ctx) == 0
            assert AgentDialect.get_property_index_constant("pos_y", ctx.ctx) == 1
            assert AgentDialect.get_property_index_constant("vel_x", ctx.ctx) == 2
            assert AgentDialect.get_property_index_constant("vel_y", ctx.ctx) == 3
            assert AgentDialect.get_property_index_constant("state", ctx.ctx) == 4

    def test_get_property_index_constant_invalid(self):
        """Test getting property index with invalid name raises error."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            with pytest.raises(ValueError):
                AgentDialect.get_property_index_constant("invalid_prop", ctx.ctx)

    def test_is_agent_op(self):
        """Test checking if operation is an agent operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agent op
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Check operations
                for op in module.body.operations:
                    if hasattr(op, "attributes") and op.attributes.get("op_name"):
                        is_agent = AgentDialect.is_agent_op(op)
                        assert isinstance(is_agent, bool)

    def test_get_agent_op_name(self):
        """Test getting agent operation name."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agent spawn op
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Check operation names
                for op in module.body.operations:
                    op_name = AgentDialect.get_agent_op_name(op)
                    if op_name:
                        assert "morphogen.agent." in op_name


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentLowering:
    """Tests for agent-to-SCF lowering pass."""

    def test_lower_agent_spawn(self):
        """Test lowering agent spawn to memref allocation."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_lower")

            with ir.InsertionPoint(module.body):
                # Create agent spawn
                count = arith.ConstantOp(ir.IndexType.get(), 50).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 1.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 2.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.2).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

            # Apply lowering
            compiler.apply_agent_lowering(module)

            # Verify lowering produced memref operations
            module_str = str(module)
            assert "memref.alloc" in module_str or "memref" in module_str.lower()

    def test_lower_agent_update(self):
        """Test lowering agent update to memref store."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create and update agents
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                index = arith.ConstantOp(ir.IndexType.get(), 0).result
                property_idx = arith.ConstantOp(ir.IndexType.get(), 0).result
                new_value = arith.ConstantOp(ir.F32Type.get(), 5.0).result

                agents = AgentUpdateOp.create(
                    agents, index, property_idx, new_value,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

            # Apply lowering
            compiler.apply_agent_lowering(module)

            # Verify lowering
            module_str = str(module)
            assert "memref.store" in module_str or "store" in module_str

    def test_lower_agent_query(self):
        """Test lowering agent query to memref load."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents and query
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 1.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 2.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                index = arith.ConstantOp(ir.IndexType.get(), 0).result
                property_idx = arith.ConstantOp(ir.IndexType.get(), 0).result

                value = AgentQueryOp.create(
                    agents, index, property_idx,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

            # Apply lowering
            compiler.apply_agent_lowering(module)

            # Verify lowering
            module_str = str(module)
            assert "memref.load" in module_str or "load" in module_str

    def test_lower_agent_behavior(self):
        """Test lowering agent behavior to loops."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create agents and apply behavior
                count = arith.ConstantOp(ir.IndexType.get(), 20).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                agents = AgentBehaviorOp.create(
                    agents, "move", [],
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

            # Apply lowering
            compiler.apply_agent_lowering(module)

            # Verify lowering produced loops
            module_str = str(module)
            assert "scf.for" in module_str or "for" in module_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentCompilerIntegration:
    """Tests for compiler integration with agent operations."""

    def test_compile_agent_program_spawn(self):
        """Test compiling agent program with spawn."""
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "spawn", "args": {
                "count": 100,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "vel_x": 0.1,
                "vel_y": 0.0,
                "state": 0.0
            }}
        ]

        module = compiler.compile_agent_program(operations)
        assert module is not None

        module_str = str(module)
        assert "memref" in module_str.lower()

    def test_compile_agent_program_with_behavior(self):
        """Test compiling agent program with spawn and behavior."""
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "spawn", "args": {
                "count": 50,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "vel_x": 0.2,
                "vel_y": 0.2,
                "state": 0.0
            }},
            {"op": "behavior", "args": {
                "agents": "agents0",
                "behavior": "move"
            }}
        ]

        module = compiler.compile_agent_program(operations)
        assert module is not None

        module_str = str(module)
        assert "scf.for" in module_str

    def test_compile_agent_program_with_update_and_query(self):
        """Test compiling agent program with update and query operations."""
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "spawn", "args": {
                "count": 10,
                "pos_x": 1.0,
                "pos_y": 2.0,
                "vel_x": 0.0,
                "vel_y": 0.0,
                "state": 0.0
            }},
            {"op": "update", "args": {
                "agents": "agents0",
                "index": 0,
                "property": 0,
                "value": 5.0
            }},
            {"op": "query", "args": {
                "agents": "agents1",
                "index": 0,
                "property": 0
            }}
        ]

        module = compiler.compile_agent_program(operations)
        assert module is not None

        module_str = str(module)
        assert "memref" in module_str.lower()


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestAgentIntegrationWithOtherDialects:
    """Tests for agent integration with field and temporal dialects."""

    def test_agents_with_temporal_state(self):
        """Test agents can work with temporal state management."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.dialects.temporal import TemporalDialect

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create temporal state
                size = arith.ConstantOp(ir.IndexType.get(), 100).result
                init_val = arith.ConstantOp(ir.F32Type.get(), 0.0).result
                state = TemporalDialect.state_create(
                    size, init_val, ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Create agents
                count = arith.ConstantOp(ir.IndexType.get(), 10).result
                agents = AgentSpawnOp.create(
                    count,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                    arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                    ir.F32Type.get(), ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert state is not None
                assert agents is not None

    def test_module_with_multiple_agent_operations(self):
        """Test module with multiple sequential agent operations."""
        from mlir import ir
        from mlir.dialects import arith
        from morphogen.mlir.compiler_v2 import MLIRCompilerV2

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create multiple agent populations
                for i in range(3):
                    count = arith.ConstantOp(ir.IndexType.get(), 20 * (i + 1)).result
                    agents = AgentSpawnOp.create(
                        count,
                        arith.ConstantOp(ir.F32Type.get(), float(i)).result,
                        arith.ConstantOp(ir.F32Type.get(), float(i)).result,
                        arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                        arith.ConstantOp(ir.F32Type.get(), 0.1).result,
                        arith.ConstantOp(ir.F32Type.get(), 0.0).result,
                        ir.F32Type.get(), ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

            # Apply lowering
            compiler.apply_agent_lowering(module)

            module_str = str(module)
            assert "memref" in module_str.lower()
