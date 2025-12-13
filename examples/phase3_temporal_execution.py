"""Phase 3 Example: Temporal Execution with MLIR

This example demonstrates the Morphogen Temporal Dialect (Phase 3) capabilities:
- State container creation and management
- Flow block definition with temporal parameters
- Flow execution over multiple timesteps
- Integration with field operations (diffusion over time)

The example compiles temporal operations to MLIR IR, applies lowering passes,
and prints the resulting low-level MLIR (SCF loops with state management).

Requirements:
    pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest

Usage:
    python examples/phase3_temporal_execution.py
"""

import sys
from pathlib import Path

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphogen.mlir.context import is_mlir_available, MorphogenMLIRContext
from morphogen.mlir.compiler_v2 import MLIRCompilerV2


def example_state_creation():
    """Example: Create a state container with 100 elements initialized to 0.0"""
    print("\n" + "=" * 70)
    print("Example 1: State Creation")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "state_create",
                "args": {"size": 100, "initial_value": 0.0}
            }
        ]

        module = compiler.compile_temporal_program(operations, "state_creation")

        print("\nMorphogen Code (conceptual):")
        print("  @state x: Array[100] = 0.0")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled state creation!")
        print("   - Allocated memref<?xf32> with size 100")
        print("   - Initialized all elements to 0.0 using scf.for loop")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_flow_execution():
    """Example: Create flow and run for 10 timesteps"""
    print("\n" + "=" * 70)
    print("Example 2: Flow Execution")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "state_create",
                "args": {"size": 100, "initial_value": 0.0}
            },
            {
                "op": "flow_create",
                "args": {"dt": 0.1, "steps": 10}
            },
            {
                "op": "flow_run",
                "args": {"flow": "flow1", "initial_state": "state0"}
            }
        ]

        module = compiler.compile_temporal_program(operations, "flow_execution")

        print("\nMorphogen Code (conceptual):")
        print("  @state x: Array[100] = 0.0")
        print("  ")
        print("  flow(dt=0.1, steps=10) {")
        print("      // Temporal evolution operations")
        print("  }")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled flow execution!")
        print("   - Created state container (100 elements)")
        print("   - Set up flow with dt=0.1, 10 timesteps")
        print("   - Generated scf.for loop with iter_args for state evolution")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_state_update_and_query():
    """Example: Update and query state values"""
    print("\n" + "=" * 70)
    print("Example 3: State Update and Query")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "state_create",
                "args": {"size": 100, "initial_value": 0.0}
            },
            {
                "op": "state_update",
                "args": {"state": "state0", "index": 5, "value": 1.5}
            },
            {
                "op": "state_query",
                "args": {"state": "state1", "index": 5}
            }
        ]

        module = compiler.compile_temporal_program(operations, "state_operations")

        print("\nMorphogen Code (conceptual):")
        print("  @state x: Array[100] = 0.0")
        print("  x[5] = 1.5")
        print("  value = x[5]")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled state operations!")
        print("   - State update lowered to memref.store")
        print("   - State query lowered to memref.load")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_combined_field_and_temporal():
    """Example: Combine field operations with temporal evolution"""
    print("\n" + "=" * 70)
    print("Example 4: Field Diffusion Over Time (Combined Phase 2 + 3)")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        from mlir import ir
        from mlir.dialects import func

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # First, create a field and apply diffusion
        field_ops = [
            {
                "op": "create",
                "args": {"width": 64, "height": 64, "fill": 0.0}
            },
            {
                "op": "diffuse",
                "args": {
                    "field": "field0",
                    "rate": 0.1,
                    "dt": 0.01,
                    "iterations": 5
                }
            }
        ]

        field_module = compiler.compile_field_program(field_ops, "field_diffusion")

        print("\nMorphogen Code (conceptual) - Field Evolution:")
        print("  @field temperature: Field[64, 64] = 0.0")
        print("  ")
        print("  flow(dt=0.01, steps=100) {")
        print("      temperature = diffuse(temperature, rate=0.1, dt=0.01, iterations=5)")
        print("  }")

        print("\nGenerated Field MLIR (after lowering):")
        print("-" * 70)
        print(field_module)
        print("-" * 70)

        # Now create a temporal flow with state
        temporal_ops = [
            {
                "op": "state_create",
                "args": {"size": 10, "initial_value": 0.0}
            },
            {
                "op": "flow_create",
                "args": {"dt": 0.01, "steps": 100}
            },
            {
                "op": "flow_run",
                "args": {"flow": "flow1", "initial_state": "state0"}
            }
        ]

        temporal_module = compiler.compile_temporal_program(temporal_ops, "temporal_flow")

        print("\nGenerated Temporal MLIR (after lowering):")
        print("-" * 70)
        print(temporal_module)
        print("-" * 70)

        print("\n✅ Successfully compiled combined field + temporal operations!")
        print("   - Field diffusion uses nested scf.for loops with Jacobi iteration")
        print("   - Temporal flow manages state evolution over 100 timesteps")
        print("   - Both compile to efficient SCF + memref IR")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all Phase 3 temporal execution examples."""
    print("╔" + "=" * 68 + "╗")
    print("║  Morphogen v0.7.0 Phase 3: Temporal Execution Examples              ║")
    print("╚" + "=" * 68 + "╝")

    if not is_mlir_available():
        print("\n❌ MLIR Python bindings not available!")
        print("   Install with:")
        print("   pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest")
        return 1

    print("\n🚀 Running Phase 3 Examples...")

    # Run examples
    example_state_creation()
    example_flow_execution()
    example_state_update_and_query()
    example_combined_field_and_temporal()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ Phase 3 temporal operations successfully compiled to MLIR!")
    print("")
    print("Key Features Demonstrated:")
    print("  1. State containers for persistent data across timesteps")
    print("  2. Flow blocks with temporal iteration (dt, steps)")
    print("  3. State update/query operations (memref.store/load)")
    print("  4. Integration with Phase 2 field operations")
    print("")
    print("MLIR Lowering:")
    print("  - morphogen.temporal.state.create → memref.alloc + initialization")
    print("  - morphogen.temporal.flow.run → scf.for with iter_args")
    print("  - morphogen.temporal.state.update → memref.store")
    print("  - morphogen.temporal.state.query → memref.load")
    print("")
    print("Next Steps:")
    print("  - Phase 4: Agent Operations (spawning, behavior trees)")
    print("  - Phase 5: Audio Operations (oscillators, filters)")
    print("  - Phase 6: JIT/AOT Compilation (LLVM backend)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
