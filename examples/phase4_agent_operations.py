"""Examples: Morphogen v0.7.0 Phase 4 - Agent Operations

This module demonstrates the agent dialect operations for agent-based simulations.

Phase 4 introduces:
- Agent spawning with properties (position, velocity, state)
- Agent property updates (modify position, velocity, state)
- Agent queries (read property values)
- Agent behaviors (move, seek, bounce)
- Integration with field and temporal operations

Examples:
1. Basic agent spawning and property access
2. Agent movement and velocity updates
3. Multi-agent simulations with behaviors
4. Agents interacting with fields
5. Combined field + temporal + agent workflows
"""

from morphogen.mlir.context import MorphogenMLIRContext, MLIR_AVAILABLE

if not MLIR_AVAILABLE:
    print("MLIR Python bindings not available. Skipping examples.")
    exit(0)

from mlir import ir
from mlir.dialects import arith, func
from morphogen.mlir.compiler_v2 import MLIRCompilerV2
from morphogen.mlir.dialects.agent import AgentDialect


def example1_basic_agent_spawn():
    """Example 1: Basic agent spawning and property initialization.

    Demonstrates:
    - Creating agent populations
    - Setting initial positions, velocities, and state
    - Querying agent properties
    """
    print("\n" + "="*70)
    print("Example 1: Basic Agent Spawn")
    print("="*70)

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    # Define agent spawn operation
    operations = [
        {
            "op": "spawn",
            "args": {
                "count": 100,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "vel_x": 0.1,
                "vel_y": 0.0,
                "state": 0.0
            }
        },
        {
            "op": "query",
            "args": {
                "agents": "agents0",
                "index": 0,
                "property": 0  # Query position_x
            }
        }
    ]

    # Compile to MLIR
    module = compiler.compile_agent_program(operations, "basic_spawn")

    print("\nGenerated MLIR:")
    print(str(module))
    print("\nAgent spawn compiled successfully!")
    print(f"Spawned 100 agents at position (0, 0) with velocity (0.1, 0)")


def example2_agent_movement():
    """Example 2: Agent movement with velocity updates.

    Demonstrates:
    - Spawning agents with initial velocities
    - Applying move behavior to update positions
    - Multiple timesteps of movement
    """
    print("\n" + "="*70)
    print("Example 2: Agent Movement")
    print("="*70)

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    # Define agent movement workflow
    operations = [
        {
            "op": "spawn",
            "args": {
                "count": 50,
                "pos_x": 5.0,
                "pos_y": 5.0,
                "vel_x": 0.2,
                "vel_y": 0.3,
                "state": 0.0
            }
        },
        {
            "op": "behavior",
            "args": {
                "agents": "agents0",
                "behavior": "move"
            }
        }
    ]

    # Compile
    module = compiler.compile_agent_program(operations, "agent_movement")

    print("\nGenerated MLIR:")
    print(str(module))
    print("\nAgent movement compiled successfully!")
    print(f"50 agents moving with velocity (0.2, 0.3)")


def example3_multi_agent_behaviors():
    """Example 3: Multiple agents with different behaviors.

    Demonstrates:
    - Creating multiple agent populations
    - Different behaviors for different groups
    - Seek behavior towards targets
    """
    print("\n" + "="*70)
    print("Example 3: Multi-Agent Behaviors")
    print("="*70)

    ctx = MorphogenMLIRContext()
    with ctx.ctx, ir.Location.unknown():
        module = ctx.create_module("multi_agent_behaviors")

        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            func_type = ir.FunctionType.get([], [])
            func_op = func.FuncOp(name="main", type=func_type)
            func_op.add_entry_block()

            with ir.InsertionPoint(func_op.entry_block):
                loc = ir.Location.unknown()
                ip = ir.InsertionPoint(func_op.entry_block)

                # Group 1: Random walkers
                count1 = arith.ConstantOp(ir.IndexType.get(), 50).result
                agents1 = AgentDialect.spawn(
                    count1,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.1).result,
                    arith.ConstantOp(f32, 0.1).result,
                    arith.ConstantOp(f32, 0.0).result,
                    f32, loc, ip
                )

                # Apply move behavior
                agents1 = AgentDialect.behavior(agents1, "move", [], loc, ip)

                # Group 2: Seekers
                count2 = arith.ConstantOp(ir.IndexType.get(), 30).result
                agents2 = AgentDialect.spawn(
                    count2,
                    arith.ConstantOp(f32, 10.0).result,
                    arith.ConstantOp(f32, 10.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 1.0).result,
                    f32, loc, ip
                )

                # Apply seek behavior towards origin
                target_x = arith.ConstantOp(f32, 0.0).result
                target_y = arith.ConstantOp(f32, 0.0).result
                speed = arith.ConstantOp(f32, 0.05).result
                agents2 = AgentDialect.behavior(
                    agents2, "seek", [target_x, target_y, speed], loc, ip
                )

                # Return
                func.ReturnOp([])

        # Apply lowering
        compiler = MLIRCompilerV2(ctx)
        compiler.apply_agent_lowering(module)

        print("\nGenerated MLIR:")
        print(str(module))
        print("\nMulti-agent simulation compiled successfully!")
        print("Group 1: 50 random walkers")
        print("Group 2: 30 seekers moving towards origin")


def example4_agent_property_updates():
    """Example 4: Updating agent properties dynamically.

    Demonstrates:
    - Spawning agents
    - Updating individual agent properties
    - Querying updated values
    """
    print("\n" + "="*70)
    print("Example 4: Agent Property Updates")
    print("="*70)

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    # Define workflow with updates
    operations = [
        {
            "op": "spawn",
            "args": {
                "count": 10,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "vel_x": 0.0,
                "vel_y": 0.0,
                "state": 0.0
            }
        },
        {
            "op": "update",
            "args": {
                "agents": "agents0",
                "index": 0,
                "property": 0,  # position_x
                "value": 5.0
            }
        },
        {
            "op": "update",
            "args": {
                "agents": "agents1",
                "index": 0,
                "property": 1,  # position_y
                "value": 7.0
            }
        },
        {
            "op": "update",
            "args": {
                "agents": "agents2",
                "index": 0,
                "property": 2,  # velocity_x
                "value": 0.5
            }
        },
        {
            "op": "query",
            "args": {
                "agents": "agents3",
                "index": 0,
                "property": 0  # Read back position_x
            }
        }
    ]

    # Compile
    module = compiler.compile_agent_program(operations, "property_updates")

    print("\nGenerated MLIR:")
    print(str(module))
    print("\nAgent property updates compiled successfully!")
    print("Updated agent 0:")
    print("  - position_x = 5.0")
    print("  - position_y = 7.0")
    print("  - velocity_x = 0.5")


def example5_bounce_behavior():
    """Example 5: Agents bouncing off boundaries.

    Demonstrates:
    - Spawning agents in a bounded region
    - Applying bounce behavior with boundaries
    - Boundary collision handling
    """
    print("\n" + "="*70)
    print("Example 5: Bounce Behavior")
    print("="*70)

    ctx = MorphogenMLIRContext()
    with ctx.ctx, ir.Location.unknown():
        module = ctx.create_module("bounce_behavior")

        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            func_type = ir.FunctionType.get([], [])
            func_op = func.FuncOp(name="main", type=func_type)
            func_op.add_entry_block()

            with ir.InsertionPoint(func_op.entry_block):
                loc = ir.Location.unknown()
                ip = ir.InsertionPoint(func_op.entry_block)

                # Spawn agents in center of bounded region
                count = arith.ConstantOp(ir.IndexType.get(), 40).result
                agents = AgentDialect.spawn(
                    count,
                    arith.ConstantOp(f32, 50.0).result,  # Start at center
                    arith.ConstantOp(f32, 50.0).result,
                    arith.ConstantOp(f32, 2.0).result,   # High velocity
                    arith.ConstantOp(f32, 1.5).result,
                    arith.ConstantOp(f32, 0.0).result,
                    f32, loc, ip
                )

                # Define boundaries
                min_x = arith.ConstantOp(f32, 0.0).result
                max_x = arith.ConstantOp(f32, 100.0).result
                min_y = arith.ConstantOp(f32, 0.0).result
                max_y = arith.ConstantOp(f32, 100.0).result

                # Apply bounce behavior
                agents = AgentDialect.behavior(
                    agents, "bounce", [min_x, max_x, min_y, max_y], loc, ip
                )

                # Return
                func.ReturnOp([])

        # Apply lowering
        compiler = MLIRCompilerV2(ctx)
        compiler.apply_agent_lowering(module)

        print("\nGenerated MLIR:")
        print(str(module))
        print("\nBounce behavior compiled successfully!")
        print("40 agents bouncing in region [0, 100] x [0, 100]")


def example6_agent_field_integration():
    """Example 6: Agents interacting with fields.

    Demonstrates:
    - Creating a field
    - Spawning agents
    - Agents sampling field values (conceptual)
    - Combined field + agent operations
    """
    print("\n" + "="*70)
    print("Example 6: Agent-Field Integration (Conceptual)")
    print("="*70)

    ctx = MorphogenMLIRContext()
    with ctx.ctx, ir.Location.unknown():
        module = ctx.create_module("agent_field_integration")

        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            func_type = ir.FunctionType.get([], [])
            func_op = func.FuncOp(name="main", type=func_type)
            func_op.add_entry_block()

            with ir.InsertionPoint(func_op.entry_block):
                loc = ir.Location.unknown()
                ip = ir.InsertionPoint(func_op.entry_block)

                # Import field dialect
                from morphogen.mlir.dialects.field import FieldDialect

                # Create a field
                width = arith.ConstantOp(ir.IndexType.get(), 128).result
                height = arith.ConstantOp(ir.IndexType.get(), 128).result
                fill = arith.ConstantOp(f32, 0.0).result

                field = FieldDialect.create(width, height, fill, f32, loc, ip)

                # Compute field gradient
                grad = FieldDialect.gradient(field, loc, ip)

                # Spawn agents
                count = arith.ConstantOp(ir.IndexType.get(), 100).result
                agents = AgentDialect.spawn(
                    count,
                    arith.ConstantOp(f32, 64.0).result,
                    arith.ConstantOp(f32, 64.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    f32, loc, ip
                )

                # In future phases, agents would sample gradient at their positions
                # and update velocities accordingly (gradient descent, etc.)
                # For Phase 4, we demonstrate the operations exist side-by-side

                # Apply move behavior
                agents = AgentDialect.behavior(agents, "move", [], loc, ip)

                # Return
                func.ReturnOp([])

        # Apply lowering for both field and agent operations
        compiler = MLIRCompilerV2(ctx)
        compiler.apply_field_lowering(module)
        compiler.apply_agent_lowering(module)

        print("\nGenerated MLIR:")
        print(str(module))
        print("\nAgent-field integration compiled successfully!")
        print("Field: 128x128 grid with gradient computation")
        print("Agents: 100 agents moving on field")
        print("(Future: agents will sample field gradients)")


def example7_temporal_agent_evolution():
    """Example 7: Agents evolving over time with temporal operations.

    Demonstrates:
    - Creating temporal flow
    - Spawning agents
    - Agents evolving through timesteps
    - Combined temporal + agent operations
    """
    print("\n" + "="*70)
    print("Example 7: Temporal Agent Evolution")
    print("="*70)

    ctx = MorphogenMLIRContext()
    with ctx.ctx, ir.Location.unknown():
        module = ctx.create_module("temporal_agent_evolution")

        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            func_type = ir.FunctionType.get([], [])
            func_op = func.FuncOp(name="main", type=func_type)
            func_op.add_entry_block()

            with ir.InsertionPoint(func_op.entry_block):
                loc = ir.Location.unknown()
                ip = ir.InsertionPoint(func_op.entry_block)

                # Import temporal dialect
                from morphogen.mlir.dialects.temporal import TemporalDialect

                # Create temporal flow
                dt = arith.ConstantOp(f32, 0.1).result
                steps = arith.ConstantOp(ir.IndexType.get(), 100).result
                flow = TemporalDialect.flow_create(dt, steps, f32, loc, ip)

                # Spawn agents
                count = arith.ConstantOp(ir.IndexType.get(), 50).result
                agents = AgentDialect.spawn(
                    count,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.0).result,
                    arith.ConstantOp(f32, 0.5).result,
                    arith.ConstantOp(f32, 0.5).result,
                    arith.ConstantOp(f32, 0.0).result,
                    f32, loc, ip
                )

                # In future phases, agents would be integrated into flow
                # For Phase 4, we demonstrate coexistence
                # Behavior: agents move each timestep
                agents = AgentDialect.behavior(agents, "move", [], loc, ip)

                # Return
                func.ReturnOp([])

        # Apply lowering
        compiler = MLIRCompilerV2(ctx)
        compiler.apply_temporal_lowering(module)
        compiler.apply_agent_lowering(module)

        print("\nGenerated MLIR:")
        print(str(module))
        print("\nTemporal agent evolution compiled successfully!")
        print("Temporal flow: 100 timesteps with dt=0.1")
        print("Agents: 50 agents moving over time")


def example8_large_scale_simulation():
    """Example 8: Large-scale agent simulation.

    Demonstrates:
    - Spawning many agents (10,000+)
    - Multiple behaviors
    - Performance characteristics
    """
    print("\n" + "="*70)
    print("Example 8: Large-Scale Agent Simulation")
    print("="*70)

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    # Define large-scale simulation
    operations = [
        {
            "op": "spawn",
            "args": {
                "count": 10000,
                "pos_x": 50.0,
                "pos_y": 50.0,
                "vel_x": 0.1,
                "vel_y": 0.1,
                "state": 0.0
            }
        },
        {
            "op": "behavior",
            "args": {
                "agents": "agents0",
                "behavior": "move"
            }
        }
    ]

    # Compile
    module = compiler.compile_agent_program(operations, "large_scale")

    print("\nGenerated MLIR:")
    print(str(module)[:500] + "\n...[truncated]...")
    print("\nLarge-scale simulation compiled successfully!")
    print("Agents: 10,000 agents")
    print("Note: MLIR optimization passes will vectorize loops for performance")


def main():
    """Run all agent operation examples."""
    print("\n" + "="*70)
    print("Morphogen v0.7.0 Phase 4: Agent Operations Examples")
    print("="*70)
    print("\nThese examples demonstrate the agent dialect for agent-based simulations.")
    print("Agent operations include:")
    print("  - spawn: Create agent populations with properties")
    print("  - update: Modify agent properties")
    print("  - query: Read agent property values")
    print("  - behavior: Apply movement and interaction behaviors")

    try:
        example1_basic_agent_spawn()
        example2_agent_movement()
        example3_multi_agent_behaviors()
        example4_agent_property_updates()
        example5_bounce_behavior()
        example6_agent_field_integration()
        example7_temporal_agent_evolution()
        example8_large_scale_simulation()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        print("\nPhase 4 Agent Operations features:")
        print("✓ Agent spawning with initial properties")
        print("✓ Property updates (position, velocity, state)")
        print("✓ Property queries for reading values")
        print("✓ Behavior operations (move, seek, bounce)")
        print("✓ Integration with field operations")
        print("✓ Integration with temporal operations")
        print("✓ Scalable to 10,000+ agents")
        print("\nNext phase (Phase 5): Audio Operations or JIT/AOT Compilation")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
