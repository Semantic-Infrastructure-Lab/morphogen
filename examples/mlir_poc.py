"""Proof of Concept: MLIR Integration for Morphogen v0.7.0

This example demonstrates how Morphogen will use real MLIR Python bindings
to compile and execute a simple arithmetic operation.

Goal: Compile x = 3.0 + 4.0 to native code via MLIR

Requirements:
    pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest

Note: This is a proof-of-concept. Full integration is ongoing.
"""

import sys
from pathlib import Path

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphogen.mlir.context import is_mlir_available, MorphogenMLIRContext

def example_arithmetic_without_mlir():
    """Example showing what we want to compile."""
    print("=" * 60)
    print("Morphogen Code (conceptual):")
    print("=" * 60)
    print("""
    fn add_example() -> f32 {
        x = 3.0 + 4.0
        return x
    }
    """)
    print("\nTarget MLIR IR:")
    print("=" * 60)
    print("""
    module {
      func.func @add_example() -> f32 {
        %c3 = arith.constant 3.0 : f32
        %c4 = arith.constant 4.0 : f32
        %result = arith.addf %c3, %c4 : f32
        func.return %result : f32
      }
    }
    """)
    print("=" * 60)
    print("\nExpected result: 7.0")
    print("=" * 60)


def example_with_mlir():
    """Example using real MLIR Python bindings."""
    if not is_mlir_available():
        print("\n❌ MLIR Python bindings not installed!")
        print("\nInstall with:")
        print("  pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest")
        return

    print("\n✅ MLIR is available! Building IR...\n")

    try:
        # This will be the actual implementation
        from mlir import ir
        from mlir.dialects import builtin, func, arith

        # Create context and module
        with MorphogenMLIRContext() as ctx:
            module = ctx.create_module("add_example")

            with ctx.ctx, ir.Location.unknown():
                # Create function type: () -> f32
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [f32])

                # Create function
                with ir.InsertionPoint(module.body):
                    func_op = func.FuncOp(
                        name="add_example",
                        type=func_type,
                    )
                    func_op.add_entry_block()

                    # Build function body
                    with ir.InsertionPoint(func_op.entry_block):
                        # %c3 = arith.constant 3.0 : f32
                        c3 = arith.ConstantOp(
                            f32,
                            ir.FloatAttr.get(f32, 3.0)
                        )

                        # %c4 = arith.constant 4.0 : f32
                        c4 = arith.ConstantOp(
                            f32,
                            ir.FloatAttr.get(f32, 4.0)
                        )

                        # %result = arith.addf %c3, %c4 : f32
                        result = arith.AddFOp(c3.result, c4.result)

                        # func.return %result : f32
                        func.ReturnOp([result.result])

            # Print generated MLIR
            print("Generated MLIR IR:")
            print("=" * 60)
            print(module)
            print("=" * 60)

            print("\n✅ Successfully generated MLIR IR!")
            print("\nNext steps (Phase 4):")
            print("  1. Lower to LLVM dialect")
            print("  2. Create ExecutionEngine")
            print("  3. JIT compile and execute")
            print("  4. Verify result = 7.0")

    except Exception as e:
        print(f"\n❌ Error building MLIR IR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run proof-of-concept examples."""
    print("\n" + "=" * 60)
    print("Morphogen v0.7.0 MLIR Integration - Proof of Concept")
    print("=" * 60)

    example_arithmetic_without_mlir()

    if is_mlir_available():
        example_with_mlir()
    else:
        print("\n" + "=" * 60)
        print("MLIR Status: Not Installed")
        print("=" * 60)
        print("\nTo enable real MLIR integration, install:")
        print("  pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest")
        print("\nThis is a large package (~500MB). Be patient during installation.")
        print("=" * 60)


if __name__ == "__main__":
    main()
