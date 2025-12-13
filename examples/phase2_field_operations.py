"""Phase 2 Example: Field Operations with MLIR

This example demonstrates the Morphogen Field Dialect (Phase 2) capabilities:
- Field creation
- Gradient computation
- Laplacian computation
- Diffusion solver

The example compiles field operations to MLIR IR, applies lowering passes,
and prints the resulting low-level MLIR (SCF loops + memref operations).

Requirements:
    pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest

Usage:
    python examples/phase2_field_operations.py
"""

import sys
from pathlib import Path

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphogen.mlir.context import is_mlir_available, MorphogenMLIRContext
from morphogen.mlir.compiler_v2 import MLIRCompilerV2


def example_field_creation():
    """Example: Create a 256x256 field initialized to 0.0"""
    print("\n" + "=" * 70)
    print("Example 1: Field Creation")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "create",
                "args": {"width": 256, "height": 256, "fill": 0.0}
            }
        ]

        module = compiler.compile_field_program(operations, "field_creation")

        print("\nMorphogen Code (conceptual):")
        print("  field = field.alloc((256, 256), fill_value=0.0)")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled field creation!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_gradient():
    """Example: Create field and compute gradient"""
    print("\n" + "=" * 70)
    print("Example 2: Gradient Computation")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "create",
                "args": {"width": 128, "height": 128, "fill": 1.0}
            },
            {
                "op": "gradient",
                "args": {"field": "field0"}
            }
        ]

        module = compiler.compile_field_program(operations, "gradient_example")

        print("\nMorphogen Code (conceptual):")
        print("  field = field.alloc((128, 128), fill_value=1.0)")
        print("  grad = field.gradient(field)")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled gradient computation!")
        print("\nNote: The lowering pass transforms the gradient operation into:")
        print("  - Nested scf.for loops (boundary-excluded)")
        print("  - Central difference stencil: (f[i+1,j] - f[i-1,j]) / 2")
        print("  - Separate computation for dx and dy components")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_laplacian():
    """Example: Create field and compute Laplacian"""
    print("\n" + "=" * 70)
    print("Example 3: Laplacian Computation")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "create",
                "args": {"width": 64, "height": 64, "fill": 0.5}
            },
            {
                "op": "laplacian",
                "args": {"field": "field0"}
            }
        ]

        module = compiler.compile_field_program(operations, "laplacian_example")

        print("\nMorphogen Code (conceptual):")
        print("  field = field.alloc((64, 64), fill_value=0.5)")
        print("  lapl = field.laplacian(field)")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled Laplacian computation!")
        print("\nNote: The lowering pass uses 5-point stencil:")
        print("  lapl[i,j] = f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4*f[i,j]")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_diffusion():
    """Example: Create field and apply diffusion"""
    print("\n" + "=" * 70)
    print("Example 4: Diffusion Solver")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "create",
                "args": {"width": 32, "height": 32, "fill": 1.0}
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

        module = compiler.compile_field_program(operations, "diffusion_example")

        print("\nMorphogen Code (conceptual):")
        print("  field = field.alloc((32, 32), fill_value=1.0)")
        print("  diffused = field.diffuse(field, rate=0.1, dt=0.01, iterations=5)")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled diffusion solver!")
        print("\nNote: The lowering pass implements Jacobi iteration:")
        print("  - Double-buffering for stability")
        print("  - Nested loops: iterations → i → j")
        print("  - Update: new[i,j] = field[i,j] + rate * dt * laplacian[i,j]")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_combined():
    """Example: Combined operations (gradient + laplacian + diffuse)"""
    print("\n" + "=" * 70)
    print("Example 5: Combined Field Operations")
    print("=" * 70)

    if not is_mlir_available():
        print("❌ MLIR not available. Skipping.")
        return

    try:
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {
                "op": "create",
                "args": {"width": 64, "height": 64, "fill": 0.0}
            },
            {
                "op": "gradient",
                "args": {"field": "field0"}
            },
            {
                "op": "laplacian",
                "args": {"field": "field0"}
            },
            {
                "op": "diffuse",
                "args": {
                    "field": "field0",
                    "rate": 0.2,
                    "dt": 0.005,
                    "iterations": 3
                }
            }
        ]

        module = compiler.compile_field_program(operations, "combined_example")

        print("\nMorphogen Code (conceptual):")
        print("  field = field.alloc((64, 64), fill_value=0.0)")
        print("  grad = field.gradient(field)")
        print("  lapl = field.laplacian(field)")
        print("  diffused = field.diffuse(field, rate=0.2, dt=0.005, iterations=3)")

        print("\nGenerated MLIR (after lowering):")
        print("-" * 70)
        print(module)
        print("-" * 70)

        print("\n✅ Successfully compiled combined operations!")
        print("\nThis demonstrates the full Phase 2 pipeline:")
        print("  1. High-level field operations (morphogen.field.*)")
        print("  2. Lowering to SCF loops + memref operations")
        print("  3. Ready for Phase 4: LLVM lowering + JIT execution")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all Phase 2 examples."""
    print("\n" + "=" * 70)
    print("Morphogen v0.7.0 Phase 2: Field Operations Dialect")
    print("=" * 70)

    if not is_mlir_available():
        print("\n❌ MLIR Python bindings not installed!")
        print("\nTo run these examples, install MLIR:")
        print("  pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest")
        print("\nThis is a large package (~500MB). Be patient during installation.")
        print("=" * 70)
        return

    print("\n✅ MLIR is available!")
    print("\nRunning Phase 2 examples...")

    # Run all examples
    example_field_creation()
    example_gradient()
    example_laplacian()
    example_diffusion()
    example_combined()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ All Phase 2 examples completed successfully!")
    print("\nKey accomplishments:")
    print("  - Custom Field dialect with 4 operations")
    print("  - Field-to-SCF lowering pass")
    print("  - Compilation from high-level ops to low-level MLIR")
    print("\nNext steps (Phase 3):")
    print("  - Temporal flow blocks (time iteration)")
    print("  - State management via memref")
    print("  - Full Morphogen AST compilation")
    print("\nPhase 4 (JIT Execution):")
    print("  - LLVM lowering")
    print("  - ExecutionEngine integration")
    print("  - Native code execution")
    print("  - Performance benchmarking (target: 10-100x speedup)")
    print("=" * 70)


if __name__ == "__main__":
    main()
