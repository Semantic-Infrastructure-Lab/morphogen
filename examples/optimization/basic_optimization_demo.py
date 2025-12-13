"""
Basic Optimization Domain Demonstration
========================================

Demonstrates all Phase 1 evolutionary algorithms:
- Differential Evolution (DE)
- CMA-ES
- Particle Swarm Optimization (PSO)
- Nelder-Mead Simplex

Shows optimization on standard benchmark functions.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.optimization import (
    DifferentialEvolution,
    CMAES,
    ParticleSwarmOptimization,
    NelderMead,
    Optimizer,
    BenchmarkFunctions,
    minimize,
    differential_evolution,
    cmaes,
    particle_swarm,
    nelder_mead
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(algorithm_name, result):
    """Print optimization result."""
    print(f"\n{algorithm_name} Results:")
    print(f"  Best solution: {result.best_solution}")
    print(f"  Best fitness:  {result.best_fitness:.6e}")
    print(f"  Evaluations:   {result.n_evaluations}")
    print(f"  Converged:     {result.converged}")
    print(f"  Iterations:    {len(result.fitness_history)}")


def demonstrate_differential_evolution():
    """Demonstrate Differential Evolution."""
    print_section("1. Differential Evolution (DE)")

    # Problem: Minimize Rosenbrock function in 5D
    n_dim = 5
    bounds = [(-5, 5)] * n_dim

    print(f"\nOptimizing Rosenbrock function in {n_dim}D")
    print(f"Global minimum: f([1,1,1,1,1]) = 0")

    result = DifferentialEvolution.optimize(
        objective_fn=BenchmarkFunctions.rosenbrock,
        bounds=bounds,
        population_size=50,
        max_iterations=200,
        F=0.8,
        CR=0.9,
        seed=42
    )

    print_result("Differential Evolution", result)

    # Show convergence
    print(f"\nConvergence history (every 40 iterations):")
    for i in range(0, len(result.fitness_history), 40):
        print(f"  Iteration {i:3d}: {result.fitness_history[i]:.6e}")


def demonstrate_cmaes():
    """Demonstrate CMA-ES."""
    print_section("2. CMA-ES (Covariance Matrix Adaptation)")

    # Problem: Minimize Rosenbrock in 10D
    n_dim = 10
    initial_mean = np.zeros(n_dim)

    print(f"\nOptimizing Rosenbrock function in {n_dim}D")
    print(f"Global minimum: f([1,1,...,1]) = 0")

    result = CMAES.optimize(
        objective_fn=BenchmarkFunctions.rosenbrock,
        initial_mean=initial_mean,
        initial_sigma=2.0,
        max_iterations=300,
        seed=42
    )

    print_result("CMA-ES", result)

    # Show final statistics
    print(f"\nFinal mean:  {result.metadata['final_mean'][:3]}... (first 3)")
    print(f"Final sigma: {result.metadata['final_sigma']:.6e}")


def demonstrate_pso():
    """Demonstrate Particle Swarm Optimization."""
    print_section("3. Particle Swarm Optimization (PSO)")

    # Problem: Minimize Ackley function in 5D
    n_dim = 5
    bounds = [(-5, 5)] * n_dim

    print(f"\nOptimizing Ackley function in {n_dim}D")
    print(f"Global minimum: f([0,0,0,0,0]) = 0")

    result = ParticleSwarmOptimization.optimize(
        objective_fn=BenchmarkFunctions.ackley,
        bounds=bounds,
        n_particles=30,
        max_iterations=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42
    )

    print_result("Particle Swarm Optimization", result)

    # Show swarm statistics
    particles = result.metadata['final_particles']
    fitnesses = [p.fitness for p in particles]
    print(f"\nFinal swarm statistics:")
    print(f"  Best fitness:    {min(fitnesses):.6e}")
    print(f"  Worst fitness:   {max(fitnesses):.6e}")
    print(f"  Mean fitness:    {np.mean(fitnesses):.6e}")
    print(f"  Swarm diversity: {np.std([p.position for p in particles]):.6e}")


def demonstrate_nelder_mead():
    """Demonstrate Nelder-Mead Simplex."""
    print_section("4. Nelder-Mead Simplex")

    # Problem: Minimize Rosenbrock in 3D
    n_dim = 3
    initial = np.array([0.0, 0.0, 0.0])

    print(f"\nOptimizing Rosenbrock function in {n_dim}D")
    print(f"Global minimum: f([1,1,1]) = 0")

    result = NelderMead.optimize(
        objective_fn=BenchmarkFunctions.rosenbrock,
        initial=initial,
        max_iterations=500,
        tol=1e-8
    )

    print_result("Nelder-Mead", result)

    # Show convergence
    print(f"\nSimplex size at convergence: {result.metadata.get('final_simplex_size', 'N/A')}")


def demonstrate_unified_interface():
    """Demonstrate unified optimization interface."""
    print_section("5. Unified Optimizer Interface")

    # Problem: Minimize sphere function
    n_dim = 5
    bounds = [(-5, 5)] * n_dim

    print(f"\nOptimizing Sphere function in {n_dim}D")
    print(f"Global minimum: f([0,0,0,0,0]) = 0")

    # Auto-select algorithm
    print("\nUsing auto-selection...")
    result_auto = Optimizer.minimize(
        BenchmarkFunctions.sphere,
        bounds=bounds,
        method='auto',
        max_iterations=100,
        seed=42
    )
    print(f"Auto-selected: {result_auto.metadata['algorithm']}")
    print(f"Best fitness:  {result_auto.best_fitness:.6e}")

    # Compare different algorithms
    print("\nComparing all algorithms on same problem:")
    methods = ['de', 'cmaes', 'pso']
    for method in methods:
        result = Optimizer.minimize(
            BenchmarkFunctions.sphere,
            bounds=bounds,
            method=method,
            max_iterations=50,
            seed=42
        )
        print(f"  {method.upper():8s}: {result.best_fitness:.6e} "
              f"({result.n_evaluations} evals)")


def demonstrate_determinism():
    """Demonstrate deterministic behavior."""
    print_section("6. Determinism Verification")

    bounds = [(-5, 5)] * 3

    print("\nRunning DE twice with same seed...")
    result1 = DifferentialEvolution.optimize(
        BenchmarkFunctions.sphere,
        bounds,
        population_size=20,
        max_iterations=30,
        seed=123
    )

    result2 = DifferentialEvolution.optimize(
        BenchmarkFunctions.sphere,
        bounds,
        population_size=20,
        max_iterations=30,
        seed=123
    )

    print(f"Run 1: {result1.best_fitness:.10e}")
    print(f"Run 2: {result2.best_fitness:.10e}")
    print(f"Identical: {np.allclose(result1.best_solution, result2.best_solution)}")
    print(f"Bit-exact: {np.array_equal(result1.best_solution, result2.best_solution)}")


def demonstrate_realistic_application():
    """Demonstrate realistic optimization application."""
    print_section("7. Realistic Application: PID Controller Tuning")

    def simulate_pid_performance(params):
        """
        Simulate PID controller performance.

        Simplified model: minimize overshoot + settling time.
        Optimal parameters: Kp=2.0, Ki=1.0, Kd=0.5
        """
        kp, ki, kd = params

        # Penalize negative gains
        if kp < 0 or ki < 0 or kd < 0:
            return 1000.0

        # Simulate performance (simplified)
        overshoot = abs(kp - 2.0) * 0.5
        settling_time = abs(ki - 1.0) * 0.3
        noise_sensitivity = abs(kd - 0.5) * 0.2

        return overshoot + settling_time + noise_sensitivity

    bounds = [(0, 10), (0, 5), (0, 2)]  # [Kp, Ki, Kd]

    print("\nTuning PID controller (Kp, Ki, Kd)")
    print("Target: Kp=2.0, Ki=1.0, Kd=0.5")

    result = differential_evolution(
        simulate_pid_performance,
        bounds,
        population_size=30,
        max_iterations=100,
        seed=42
    )

    print(f"\nOptimal PID gains:")
    print(f"  Kp = {result.best_solution[0]:.4f}")
    print(f"  Ki = {result.best_solution[1]:.4f}")
    print(f"  Kd = {result.best_solution[2]:.4f}")
    print(f"  Performance metric: {result.best_fitness:.6e}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  Morphogen Optimization Domain - Phase 1 Demonstration")
    print("  Evolutionary Algorithms for Design Discovery")
    print("=" * 70)

    # Run demonstrations
    demonstrate_differential_evolution()
    demonstrate_cmaes()
    demonstrate_pso()
    demonstrate_nelder_mead()
    demonstrate_unified_interface()
    demonstrate_determinism()
    demonstrate_realistic_application()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
