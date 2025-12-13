"""
Optimization Domain
===================

Comprehensive optimization algorithms for Morphogen simulations.
Implements evolutionary algorithms, swarm intelligence, and numerical optimization
for parameter tuning, design discovery, and multi-objective optimization.

This domain provides:
- Evolutionary algorithms (GA, DE, CMA-ES)
- Swarm intelligence (PSO)
- Local optimization (Nelder-Mead)
- Multi-objective optimization (NSGA-II)
- Deterministic operations for reproducibility

Layer 1: Atomic operators (DE mutation, PSO velocity update, etc.)
Layer 2: Composite operators (DE generation, PSO iteration)
Layer 3: Optimization constructs (run_optimization loops)
Layer 4: Presets (common optimization configurations)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Common Types and Structures
# ============================================================================

@dataclass
class OptimizationResult:
    """
    Results from an optimization run.

    Attributes:
        best_solution: Best solution found
        best_fitness: Fitness of best solution
        fitness_history: History of best fitness per iteration
        n_evaluations: Total number of function evaluations
        converged: Whether optimization converged
        metadata: Algorithm-specific metadata
    """
    best_solution: np.ndarray
    best_fitness: float
    fitness_history: List[float]
    n_evaluations: int
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'OptimizationResult':
        """Return a copy of this result"""
        return OptimizationResult(
            best_solution=self.best_solution.copy(),
            best_fitness=self.best_fitness,
            fitness_history=self.fitness_history.copy(),
            n_evaluations=self.n_evaluations,
            converged=self.converged,
            metadata=self.metadata.copy()
        )


# ============================================================================
# Differential Evolution (DE) Implementation
# ============================================================================

class DifferentialEvolution:
    """
    Differential Evolution optimizer.

    Excellent for continuous real-valued optimization problems.
    More efficient and stable than GA for continuous parameters.

    Strategy: DE/rand/1/bin (classic DE)
    """

    @staticmethod
    def optimize(
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        max_iterations: int = 100,
        F: float = 0.8,  # Differential weight
        CR: float = 0.9,  # Crossover probability
        seed: Optional[int] = None,
        tol: float = 1e-6,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> OptimizationResult:
        """
        Run Differential Evolution optimization.

        Args:
            objective_fn: Function to minimize (takes array, returns float)
            bounds: List of (min, max) tuples for each dimension
            population_size: Number of candidate solutions
            max_iterations: Maximum number of generations
            F: Differential weight (mutation factor), typical: 0.5-1.0
            CR: Crossover probability, typical: 0.8-1.0
            seed: Random seed for deterministic execution
            tol: Convergence tolerance (change in best fitness)
            callback: Optional function called each iteration(iter, best, fitness)

        Returns:
            OptimizationResult with best solution and convergence history

        Example:
            >>> def sphere(x): return np.sum(x**2)
            >>> bounds = [(-5, 5)] * 10
            >>> result = DifferentialEvolution.optimize(sphere, bounds)
            >>> print(f"Best: {result.best_fitness:.6f}")
        """
        if seed is not None:
            np.random.seed(seed)

        n_dim = len(bounds)
        bounds_array = np.array(bounds)

        # Initialize population
        population = np.random.uniform(
            bounds_array[:, 0],
            bounds_array[:, 1],
            size=(population_size, n_dim)
        )

        # Evaluate initial population
        fitness = np.array([objective_fn(ind) for ind in population])
        n_evals = population_size

        # Track best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]

        # Evolution loop
        for iteration in range(max_iterations):
            new_population = np.zeros_like(population)

            for i in range(population_size):
                # DE/rand/1 mutation strategy
                # Select 3 random distinct individuals (not i)
                candidates = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]

                # Mutation: v = a + F * (b - c)
                mutant = a + F * (b - c)

                # Clip to bounds
                mutant = np.clip(mutant, bounds_array[:, 0], bounds_array[:, 1])

                # Binomial crossover
                cross_points = np.random.rand(n_dim) < CR
                # Ensure at least one parameter is from mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, n_dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = objective_fn(trial)
                n_evals += 1

                if trial_fitness <= fitness[i]:  # Minimization
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

            population = new_population

            # Update best
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx].copy()

            fitness_history.append(best_fitness)

            # Callback
            if callback is not None:
                callback(iteration, best_solution, best_fitness)

            # Convergence check
            if len(fitness_history) > 10:
                recent_improvement = abs(fitness_history[-10] - fitness_history[-1])
                if recent_improvement < tol:
                    converged = True
                    break
        else:
            converged = False

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            n_evaluations=n_evals,
            converged=converged,
            metadata={
                'algorithm': 'DE/rand/1/bin',
                'F': F,
                'CR': CR,
                'population_size': population_size,
                'final_population': population
            }
        )


# ============================================================================
# CMA-ES Implementation
# ============================================================================

class CMAES:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    State-of-the-art for black-box optimization of continuous functions.
    Handles up to 100+ dimensions, robust to noise, adapts to landscape curvature.
    """

    @staticmethod
    def optimize(
        objective_fn: Callable[[np.ndarray], float],
        initial_mean: np.ndarray,
        initial_sigma: float = 0.5,
        bounds: Optional[List[Tuple[float, float]]] = None,
        population_size: Optional[int] = None,
        max_iterations: int = 1000,
        tol_fun: float = 1e-12,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> OptimizationResult:
        """
        Run CMA-ES optimization.

        Args:
            objective_fn: Function to minimize
            initial_mean: Initial mean of search distribution
            initial_sigma: Initial step size (standard deviation)
            bounds: Optional bounds as [(min, max), ...] for each dimension
            population_size: Population size (default: 4 + floor(3*log(N)))
            max_iterations: Maximum generations
            tol_fun: Tolerance on function value changes
            seed: Random seed for determinism
            callback: Optional callback(iteration, mean, best_fitness)

        Returns:
            OptimizationResult with best solution

        Example:
            >>> def rosenbrock(x):
            ...     return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
            >>> result = CMAES.optimize(rosenbrock,
            ...                         initial_mean=np.zeros(10),
            ...                         initial_sigma=1.0)
        """
        if seed is not None:
            np.random.seed(seed)

        n_dim = len(initial_mean)

        # Strategy parameters
        if population_size is None:
            population_size = 4 + int(3 * np.log(n_dim))

        mu = population_size // 2  # Number of parents

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights**2).sum()

        # Adaptation parameters
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        cs = (mu_eff + 2) / (n_dim + mu_eff + 5)
        c1 = 2 / ((n_dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n_dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + cs

        # Initialize dynamic parameters
        mean = initial_mean.copy()
        sigma = initial_sigma
        pc = np.zeros(n_dim)  # Evolution path for C
        ps = np.zeros(n_dim)  # Evolution path for sigma
        C = np.eye(n_dim)  # Covariance matrix
        eigeneval = 0
        B = np.eye(n_dim)
        D = np.ones(n_dim)

        # Tracking
        fitness_history = []
        n_evals = 0
        best_solution = mean.copy()
        best_fitness = float('inf')

        for iteration in range(max_iterations):
            # Sample population
            population = []
            z_samples = []

            for _ in range(population_size):
                z = np.random.randn(n_dim)
                z_samples.append(z)
                y = B @ (D * z)
                x = mean + sigma * y

                # Apply bounds if specified
                if bounds is not None:
                    bounds_array = np.array(bounds)
                    x = np.clip(x, bounds_array[:, 0], bounds_array[:, 1])

                population.append(x)

            # Evaluate
            fitness = np.array([objective_fn(x) for x in population])
            n_evals += population_size

            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            population = [population[i] for i in sorted_indices]
            fitness = fitness[sorted_indices]
            z_samples = [z_samples[i] for i in sorted_indices]

            # Track best
            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                best_solution = population[0].copy()

            fitness_history.append(best_fitness)

            # Recombination
            old_mean = mean.copy()
            mean = sum(weights[i] * population[i] for i in range(mu))

            # Update evolution paths
            y_mean = (mean - old_mean) / sigma
            C_sqrt_inv = B @ np.diag(1 / D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv @ y_mean

            hsig = (np.linalg.norm(ps) /
                   np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) /
                   np.sqrt(n_dim) <
                   1.4 + 2 / (n_dim + 1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_mean

            # Update covariance matrix
            artmp = np.array([(population[i] - old_mean) / sigma for i in range(mu)])
            C = ((1 - c1 - cmu) * C +
                 c1 * (np.outer(pc, pc) +
                      (1 - hsig) * cc * (2 - cc) * C) +
                 cmu * artmp.T @ np.diag(weights) @ artmp)

            # Update step size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(n_dim) - 1))

            # Update B and D from C (eigen decomposition every few iterations)
            if iteration - eigeneval > 1 / (c1 + cmu) / n_dim / 10:
                eigeneval = iteration
                C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
                D, B = np.linalg.eigh(C)
                D = np.sqrt(D)

            # Callback
            if callback is not None:
                callback(iteration, mean, best_fitness)

            # Convergence check
            if len(fitness_history) > 10:
                recent_change = abs(fitness_history[-10] - fitness_history[-1])
                if recent_change < tol_fun:
                    converged = True
                    break
        else:
            converged = False

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            n_evaluations=n_evals,
            converged=converged,
            metadata={
                'algorithm': 'CMA-ES',
                'final_mean': mean,
                'final_sigma': sigma,
                'covariance_matrix': C,
                'population_size': population_size
            }
        )


# ============================================================================
# Particle Swarm Optimization (PSO)
# ============================================================================

@dataclass
class Particle:
    """
    Single particle in PSO swarm.

    Attributes:
        position: Current position in search space
        velocity: Current velocity
        best_position: Personal best position
        best_fitness: Fitness at personal best
        fitness: Current fitness
    """
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    fitness: float = float('inf')

    def copy(self) -> 'Particle':
        """Return a copy of this particle"""
        return Particle(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            best_position=self.best_position.copy(),
            best_fitness=self.best_fitness,
            fitness=self.fitness
        )


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization (PSO).

    Swarm-based optimization inspired by bird flocking behavior.
    Good for smooth-ish continuous landscapes.
    """

    @staticmethod
    def optimize(
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_particles: int = 30,
        max_iterations: int = 100,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive coefficient
        c2: float = 1.5,  # Social coefficient
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> OptimizationResult:
        """
        Run Particle Swarm Optimization.

        Args:
            objective_fn: Function to minimize
            bounds: Bounds as [(min, max), ...] for each dimension
            n_particles: Number of particles in swarm
            max_iterations: Maximum iterations
            w: Inertia weight (0.4-0.9, controls exploration vs exploitation)
            c1: Cognitive coefficient (1.0-2.0, attraction to personal best)
            c2: Social coefficient (1.0-2.0, attraction to global best)
            seed: Random seed for determinism
            callback: Optional callback(iteration, global_best_pos, global_best_fitness)

        Returns:
            OptimizationResult with best solution

        Example:
            >>> def ackley(x):
            ...     a, b, c = 20, 0.2, 2*np.pi
            ...     n = len(x)
            ...     return (-a * np.exp(-b * np.sqrt(np.sum(x**2) / n)) -
            ...             np.exp(np.sum(np.cos(c * x)) / n) + a + np.e)
            >>> bounds = [(-5, 5)] * 10
            >>> result = ParticleSwarmOptimization.optimize(ackley, bounds)
        """
        if seed is not None:
            np.random.seed(seed)

        n_dim = len(bounds)
        bounds_array = np.array(bounds)

        # Initialize swarm
        particles = []
        for _ in range(n_particles):
            position = np.random.uniform(
                bounds_array[:, 0],
                bounds_array[:, 1],
                size=n_dim
            )
            # Initialize velocity to small random values
            velocity = np.random.uniform(-1, 1, size=n_dim) * 0.1 * (
                bounds_array[:, 1] - bounds_array[:, 0]
            )

            fitness = objective_fn(position)

            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=fitness,
                fitness=fitness
            )
            particles.append(particle)

        # Find global best
        global_best_particle = min(particles, key=lambda p: p.fitness)
        global_best_position = global_best_particle.position.copy()
        global_best_fitness = global_best_particle.fitness

        fitness_history = [global_best_fitness]
        n_evals = n_particles

        # PSO main loop
        for iteration in range(max_iterations):
            for particle in particles:
                # Update velocity
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (particle.best_position - particle.position)
                social = c2 * r2 * (global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive + social

                # Limit velocity (optional but helps stability)
                v_max = 0.2 * (bounds_array[:, 1] - bounds_array[:, 0])
                particle.velocity = np.clip(particle.velocity, -v_max, v_max)

                # Update position
                particle.position = particle.position + particle.velocity

                # Enforce bounds
                particle.position = np.clip(
                    particle.position,
                    bounds_array[:, 0],
                    bounds_array[:, 1]
                )

                # Evaluate
                particle.fitness = objective_fn(particle.position)
                n_evals += 1

                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = particle.position.copy()

            fitness_history.append(global_best_fitness)

            # Callback
            if callback is not None:
                callback(iteration, global_best_position, global_best_fitness)

        return OptimizationResult(
            best_solution=global_best_position,
            best_fitness=global_best_fitness,
            fitness_history=fitness_history,
            n_evaluations=n_evals,
            converged=False,  # PSO doesn't have clear convergence criteria
            metadata={
                'algorithm': 'PSO',
                'w': w,
                'c1': c1,
                'c2': c2,
                'n_particles': n_particles,
                'final_particles': [p.copy() for p in particles]
            }
        )


# ============================================================================
# Nelder-Mead Simplex Algorithm
# ============================================================================

class NelderMead:
    """
    Nelder-Mead simplex algorithm.

    Derivative-free local optimization method.
    Robust to noise, simple and reliable for low-dimensional problems (<10D).
    """

    @staticmethod
    def optimize(
        objective_fn: Callable[[np.ndarray], float],
        initial: np.ndarray,
        max_iterations: int = 1000,
        tol: float = 1e-6,
        alpha: float = 1.0,  # Reflection
        gamma: float = 2.0,  # Expansion
        rho: float = 0.5,  # Contraction
        sigma: float = 0.5,  # Shrinkage
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> OptimizationResult:
        """
        Run Nelder-Mead simplex optimization.

        Args:
            objective_fn: Function to minimize
            initial: Initial guess
            max_iterations: Maximum iterations
            tol: Convergence tolerance (simplex size)
            alpha: Reflection coefficient (>0, typical: 1)
            gamma: Expansion coefficient (>1, typical: 2)
            rho: Contraction coefficient (0<rho<1, typical: 0.5)
            sigma: Shrinkage coefficient (0<sigma<1, typical: 0.5)
            callback: Optional callback(iteration, best_point, best_value)

        Returns:
            OptimizationResult with best solution

        Example:
            >>> def sphere(x): return np.sum(x**2)
            >>> result = NelderMead.optimize(sphere, np.array([1.0, 2.0, 3.0]))
            >>> print(f"Found minimum at: {result.best_solution}")
        """
        n_dim = len(initial)

        # Initialize simplex (n+1 points)
        simplex = [initial.copy()]
        for i in range(n_dim):
            point = initial.copy()
            point[i] += 0.05 if point[i] != 0 else 0.00025
            simplex.append(point)

        # Evaluate simplex
        f_values = [objective_fn(point) for point in simplex]
        n_evals = n_dim + 1

        fitness_history = [min(f_values)]

        for iteration in range(max_iterations):
            # Sort simplex by function value
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]

            # Best, worst, second worst
            f_best, f_worst, f_second_worst = f_values[0], f_values[-1], f_values[-2]
            x_best, x_worst = simplex[0], simplex[-1]

            # Centroid of all points except worst
            x_centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_reflected = x_centroid + alpha * (x_centroid - x_worst)
            f_reflected = objective_fn(x_reflected)
            n_evals += 1

            if f_best <= f_reflected < f_second_worst:
                # Accept reflected point
                simplex[-1] = x_reflected
                f_values[-1] = f_reflected
            elif f_reflected < f_best:
                # Try expansion
                x_expanded = x_centroid + gamma * (x_reflected - x_centroid)
                f_expanded = objective_fn(x_expanded)
                n_evals += 1

                if f_expanded < f_reflected:
                    simplex[-1] = x_expanded
                    f_values[-1] = f_expanded
                else:
                    simplex[-1] = x_reflected
                    f_values[-1] = f_reflected
            else:
                # Contraction
                if f_reflected < f_worst:
                    # Outside contraction
                    x_contracted = x_centroid + rho * (x_reflected - x_centroid)
                else:
                    # Inside contraction
                    x_contracted = x_centroid + rho * (x_worst - x_centroid)

                f_contracted = objective_fn(x_contracted)
                n_evals += 1

                if f_contracted < min(f_reflected, f_worst):
                    simplex[-1] = x_contracted
                    f_values[-1] = f_contracted
                else:
                    # Shrink
                    for i in range(1, n_dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        f_values[i] = objective_fn(simplex[i])
                    n_evals += n_dim

            # Track best
            current_best = min(f_values)
            fitness_history.append(current_best)

            # Callback
            if callback is not None:
                callback(iteration, simplex[0], f_values[0])

            # Convergence check (simplex size)
            simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0])
                                   for i in range(1, n_dim + 1)])
            if simplex_size < tol:
                converged = True
                break
        else:
            converged = False

        best_idx = np.argmin(f_values)

        return OptimizationResult(
            best_solution=simplex[best_idx],
            best_fitness=f_values[best_idx],
            fitness_history=fitness_history,
            n_evaluations=n_evals,
            converged=converged,
            metadata={
                'algorithm': 'Nelder-Mead',
                'final_simplex': simplex,
                'final_simplex_size': simplex_size if converged else None
            }
        )


# ============================================================================
# Unified Optimization Interface
# ============================================================================

class Optimizer:
    """
    Unified optimization interface supporting multiple algorithms.

    This class provides a consistent API for all optimization algorithms
    and automatic algorithm selection based on problem characteristics.
    """

    @staticmethod
    def minimize(
        objective_fn: Callable[[np.ndarray], float],
        bounds: Optional[List[Tuple[float, float]]] = None,
        initial: Optional[np.ndarray] = None,
        method: str = 'auto',
        max_iterations: int = 100,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Minimize an objective function using the specified method.

        Args:
            objective_fn: Function to minimize
            bounds: Optional bounds [(min, max), ...] for each dimension
            initial: Optional initial guess (required for Nelder-Mead)
            method: Algorithm to use:
                - 'auto': Automatically select based on problem
                - 'de': Differential Evolution
                - 'cmaes': CMA-ES
                - 'pso': Particle Swarm Optimization
                - 'nelder-mead': Nelder-Mead simplex
            max_iterations: Maximum iterations
            seed: Random seed for determinism
            callback: Optional progress callback
            **kwargs: Additional algorithm-specific parameters

        Returns:
            OptimizationResult

        Example:
            >>> def rosenbrock(x):
            ...     return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
            >>> bounds = [(-5, 5)] * 10
            >>> result = Optimizer.minimize(rosenbrock, bounds, method='de')
        """
        # Auto-select algorithm
        if method == 'auto':
            if bounds is not None:
                n_dim = len(bounds)
                if n_dim <= 5:
                    method = 'nelder-mead' if initial is not None else 'de'
                elif n_dim <= 20:
                    method = 'de'
                else:
                    method = 'cmaes'
            elif initial is not None:
                method = 'nelder-mead'
            else:
                raise ValueError("Either bounds or initial point must be provided")

        # Dispatch to appropriate algorithm
        if method == 'de':
            if bounds is None:
                raise ValueError("DE requires bounds")
            return DifferentialEvolution.optimize(
                objective_fn, bounds, max_iterations=max_iterations,
                seed=seed, callback=callback, **kwargs
            )

        elif method == 'cmaes':
            if initial is None and bounds is not None:
                # Use center of bounds as initial mean
                bounds_array = np.array(bounds)
                initial = (bounds_array[:, 0] + bounds_array[:, 1]) / 2
            elif initial is None:
                raise ValueError("CMA-ES requires initial_mean or bounds")

            return CMAES.optimize(
                objective_fn, initial_mean=initial, bounds=bounds,
                max_iterations=max_iterations, seed=seed, callback=callback, **kwargs
            )

        elif method == 'pso':
            if bounds is None:
                raise ValueError("PSO requires bounds")
            return ParticleSwarmOptimization.optimize(
                objective_fn, bounds, max_iterations=max_iterations,
                seed=seed, callback=callback, **kwargs
            )

        elif method == 'nelder-mead':
            if initial is None:
                if bounds is not None:
                    # Use center of bounds as initial
                    bounds_array = np.array(bounds)
                    initial = (bounds_array[:, 0] + bounds_array[:, 1]) / 2
                else:
                    raise ValueError("Nelder-Mead requires initial point or bounds")

            return NelderMead.optimize(
                objective_fn, initial, max_iterations=max_iterations,
                callback=callback, **kwargs
            )

        else:
            raise ValueError(f"Unknown optimization method: {method}")


# ============================================================================
# Convenience Functions
# ============================================================================

@operator(
    domain="optimization",
    category=OpCategory.TRANSFORM,
    signature="(objective_fn: Callable, bounds: Optional[List[Tuple[float, float]]], initial: Optional[ndarray], method: str, **kwargs) -> OptimizationResult",
    deterministic=False,
    doc="Minimize an objective function using specified optimization method"
)
def minimize(objective_fn, bounds=None, initial=None, method='auto', **kwargs):
    """
    Convenience function for minimization.
    Delegates to Optimizer.minimize().
    """
    return Optimizer.minimize(objective_fn, bounds, initial, method, **kwargs)


@operator(
    domain="optimization",
    category=OpCategory.TRANSFORM,
    signature="(objective_fn: Callable, bounds: List[Tuple[float, float]], **kwargs) -> OptimizationResult",
    deterministic=False,
    doc="Run Differential Evolution optimization"
)
def differential_evolution(objective_fn, bounds, **kwargs):
    """Convenience function for DE optimization."""
    return DifferentialEvolution.optimize(objective_fn, bounds, **kwargs)


@operator(
    domain="optimization",
    category=OpCategory.TRANSFORM,
    signature="(objective_fn: Callable, initial_mean: ndarray, **kwargs) -> OptimizationResult",
    deterministic=False,
    doc="Run CMA-ES optimization"
)
def cmaes(objective_fn, initial_mean, **kwargs):
    """Convenience function for CMA-ES optimization."""
    return CMAES.optimize(objective_fn, initial_mean, **kwargs)


@operator(
    domain="optimization",
    category=OpCategory.TRANSFORM,
    signature="(objective_fn: Callable, bounds: List[Tuple[float, float]], **kwargs) -> OptimizationResult",
    deterministic=False,
    doc="Run Particle Swarm Optimization"
)
def particle_swarm(objective_fn, bounds, **kwargs):
    """Convenience function for PSO optimization."""
    return ParticleSwarmOptimization.optimize(objective_fn, bounds, **kwargs)


@operator(
    domain="optimization",
    category=OpCategory.TRANSFORM,
    signature="(objective_fn: Callable, initial: ndarray, **kwargs) -> OptimizationResult",
    deterministic=False,
    doc="Run Nelder-Mead simplex optimization"
)
def nelder_mead(objective_fn, initial, **kwargs):
    """Convenience function for Nelder-Mead optimization."""
    return NelderMead.optimize(objective_fn, initial, **kwargs)


# ============================================================================
# Benchmark Functions for Testing
# ============================================================================

class BenchmarkFunctions:
    """Standard optimization test functions."""

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function: f(x) = sum(x^2), minimum at x=0, f(0)=0"""
        return np.sum(x**2)

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function: minimum at x=[1,1,...], f(x)=0"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function: highly multimodal, minimum at x=0, f(0)=0"""
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function: multimodal, minimum at x=0, f(0)=0"""
        n = len(x)
        a, b, c = 20, 0.2, 2 * np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e

    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """Schwefel function: deceptive, minimum at x=[420.97,...], f(x)≈0"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# Module-level exports
__all__ = [
    'OptimizationResult',
    'DifferentialEvolution',
    'CMAES',
    'ParticleSwarmOptimization',
    'NelderMead',
    'Optimizer',
    'minimize',
    'differential_evolution',
    'cmaes',
    'particle_swarm',
    'nelder_mead',
    'BenchmarkFunctions',
    'Particle',
]
