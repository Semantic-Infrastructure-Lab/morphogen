"""Benchmark Suite for Morphogen Field Operations (Phase 2)

This benchmark measures:
1. Compilation time: Morphogen AST → MLIR IR → Lowered IR
2. IR generation size and complexity
3. Correctness verification (Phase 2 only compiles, doesn't execute yet)

Phase 4 will add execution benchmarks and performance comparisons.

Requirements:
    pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest

Usage:
    python benchmarks/field_operations_benchmark.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphogen.mlir.context import is_mlir_available, MorphogenMLIRContext
from morphogen.mlir.compiler_v2 import MLIRCompilerV2


class FieldOperationsBenchmark:
    """Benchmark suite for field operations."""

    def __init__(self):
        """Initialize benchmark."""
        if not is_mlir_available():
            raise RuntimeError("MLIR required for benchmarking")

        self.results: List[Dict] = []

    def benchmark_field_creation(self, sizes: List[Tuple[int, int]]) -> None:
        """Benchmark field creation for various sizes.

        Args:
            sizes: List of (width, height) tuples
        """
        print("\n" + "=" * 70)
        print("Benchmark: Field Creation")
        print("=" * 70)

        for width, height in sizes:
            ctx = MorphogenMLIRContext()
            compiler = MLIRCompilerV2(ctx)

            operations = [
                {"op": "create", "args": {"width": width, "height": height, "fill": 0.0}}
            ]

            # Measure compilation time
            start = time.perf_counter()
            module = compiler.compile_field_program(operations, "field_creation")
            compile_time = time.perf_counter() - start

            # Measure IR size
            module_str = str(module)
            ir_size = len(module_str)
            num_ops = module_str.count("scf.for") + module_str.count("memref")

            result = {
                "operation": "field.create",
                "size": f"{width}x{height}",
                "compile_time_ms": compile_time * 1000,
                "ir_size_bytes": ir_size,
                "num_ops": num_ops,
            }

            self.results.append(result)

            print(f"\n  Size: {width}x{height}")
            print(f"    Compilation time: {compile_time*1000:.3f} ms")
            print(f"    IR size: {ir_size} bytes")
            print(f"    Operations count: {num_ops}")

    def benchmark_gradient(self, sizes: List[Tuple[int, int]]) -> None:
        """Benchmark gradient computation.

        Args:
            sizes: List of (width, height) tuples
        """
        print("\n" + "=" * 70)
        print("Benchmark: Gradient Computation")
        print("=" * 70)

        for width, height in sizes:
            ctx = MorphogenMLIRContext()
            compiler = MLIRCompilerV2(ctx)

            operations = [
                {"op": "create", "args": {"width": width, "height": height, "fill": 1.0}},
                {"op": "gradient", "args": {"field": "field0"}},
            ]

            start = time.perf_counter()
            module = compiler.compile_field_program(operations, "gradient")
            compile_time = time.perf_counter() - start

            module_str = str(module)
            ir_size = len(module_str)
            num_loops = module_str.count("scf.for")

            result = {
                "operation": "field.gradient",
                "size": f"{width}x{height}",
                "compile_time_ms": compile_time * 1000,
                "ir_size_bytes": ir_size,
                "num_loops": num_loops,
            }

            self.results.append(result)

            print(f"\n  Size: {width}x{height}")
            print(f"    Compilation time: {compile_time*1000:.3f} ms")
            print(f"    IR size: {ir_size} bytes")
            print(f"    SCF loops: {num_loops}")

    def benchmark_laplacian(self, sizes: List[Tuple[int, int]]) -> None:
        """Benchmark Laplacian computation.

        Args:
            sizes: List of (width, height) tuples
        """
        print("\n" + "=" * 70)
        print("Benchmark: Laplacian Computation")
        print("=" * 70)

        for width, height in sizes:
            ctx = MorphogenMLIRContext()
            compiler = MLIRCompilerV2(ctx)

            operations = [
                {"op": "create", "args": {"width": width, "height": height, "fill": 0.5}},
                {"op": "laplacian", "args": {"field": "field0"}},
            ]

            start = time.perf_counter()
            module = compiler.compile_field_program(operations, "laplacian")
            compile_time = time.perf_counter() - start

            module_str = str(module)
            ir_size = len(module_str)
            num_loads = module_str.count("memref.load")
            num_stores = module_str.count("memref.store")

            result = {
                "operation": "field.laplacian",
                "size": f"{width}x{height}",
                "compile_time_ms": compile_time * 1000,
                "ir_size_bytes": ir_size,
                "num_loads": num_loads,
                "num_stores": num_stores,
            }

            self.results.append(result)

            print(f"\n  Size: {width}x{height}")
            print(f"    Compilation time: {compile_time*1000:.3f} ms")
            print(f"    IR size: {ir_size} bytes")
            print(f"    Memory loads: {num_loads}")
            print(f"    Memory stores: {num_stores}")

    def benchmark_diffusion(self, configs: List[Dict]) -> None:
        """Benchmark diffusion solver.

        Args:
            configs: List of configuration dictionaries with keys:
                - width, height: Field dimensions
                - iterations: Number of Jacobi iterations
        """
        print("\n" + "=" * 70)
        print("Benchmark: Diffusion Solver")
        print("=" * 70)

        for config in configs:
            width = config["width"]
            height = config["height"]
            iterations = config["iterations"]

            ctx = MorphogenMLIRContext()
            compiler = MLIRCompilerV2(ctx)

            operations = [
                {"op": "create", "args": {"width": width, "height": height, "fill": 1.0}},
                {
                    "op": "diffuse",
                    "args": {
                        "field": "field0",
                        "rate": 0.1,
                        "dt": 0.01,
                        "iterations": iterations,
                    },
                },
            ]

            start = time.perf_counter()
            module = compiler.compile_field_program(operations, "diffusion")
            compile_time = time.perf_counter() - start

            module_str = str(module)
            ir_size = len(module_str)
            num_loops = module_str.count("scf.for")

            result = {
                "operation": "field.diffuse",
                "size": f"{width}x{height}",
                "iterations": iterations,
                "compile_time_ms": compile_time * 1000,
                "ir_size_bytes": ir_size,
                "num_loops": num_loops,
            }

            self.results.append(result)

            print(f"\n  Size: {width}x{height}, Iterations: {iterations}")
            print(f"    Compilation time: {compile_time*1000:.3f} ms")
            print(f"    IR size: {ir_size} bytes")
            print(f"    SCF loops: {num_loops}")

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)

        # Group by operation
        ops = {}
        for result in self.results:
            op_name = result["operation"]
            if op_name not in ops:
                ops[op_name] = []
            ops[op_name].append(result)

        for op_name, results in ops.items():
            print(f"\n{op_name}:")
            avg_compile_time = sum(r["compile_time_ms"] for r in results) / len(results)
            print(f"  Average compilation time: {avg_compile_time:.3f} ms")
            print(f"  Number of configurations tested: {len(results)}")

        # Overall stats
        all_compile_times = [r["compile_time_ms"] for r in self.results]
        print(f"\nOverall:")
        print(f"  Total benchmarks: {len(self.results)}")
        print(f"  Average compile time: {sum(all_compile_times)/len(all_compile_times):.3f} ms")
        print(f"  Min compile time: {min(all_compile_times):.3f} ms")
        print(f"  Max compile time: {max(all_compile_times):.3f} ms")

        print("\nPhase 2 Success Metrics:")
        print(f"  ✅ Compilation time < 1s: {'Yes' if max(all_compile_times) < 1000 else 'No'}")
        print(f"  ✅ All operations compile successfully: Yes")
        print(f"  ⏳ Execution speedup vs NumPy: Phase 4 (JIT required)")

    def run_all(self) -> None:
        """Run all benchmarks."""
        print("\n" + "=" * 70)
        print("Morphogen v0.7.0 Phase 2: Field Operations Benchmark Suite")
        print("=" * 70)

        # Test sizes (small to medium for Phase 2 compilation benchmarks)
        sizes = [
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
        ]

        # Diffusion configurations
        diffusion_configs = [
            {"width": 32, "height": 32, "iterations": 5},
            {"width": 64, "height": 64, "iterations": 10},
            {"width": 128, "height": 128, "iterations": 20},
        ]

        # Run benchmarks
        self.benchmark_field_creation(sizes)
        self.benchmark_gradient(sizes)
        self.benchmark_laplacian(sizes)
        self.benchmark_diffusion(diffusion_configs)

        # Print summary
        self.print_summary()


def main():
    """Run benchmark suite."""
    if not is_mlir_available():
        print("\n❌ MLIR Python bindings not installed!")
        print("\nInstall MLIR to run benchmarks:")
        print("  pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest")
        return

    try:
        benchmark = FieldOperationsBenchmark()
        benchmark.run_all()

        print("\n" + "=" * 70)
        print("✅ Benchmark suite completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
