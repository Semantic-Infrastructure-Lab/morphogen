"""
Genetic Algorithm Domain
========================

Evolutionary computation domain for Morphogen simulations.
Implements genetic algorithms for parameter optimization, neural network evolution,
and general black-box optimization.

This domain provides:
- Population management (initialization, selection, elitism)
- Genetic operators (crossover, mutation)
- Fitness evaluation and ranking
- Generational evolution loop
- Deterministic operations for reproducibility

Layer 1: Atomic operators (selection, crossover, mutation)
Layer 2: Composite operators (breed, evolve_generation)
Layer 3: Evolution constructs (GA loop, island models)
"""

import numpy as np

from morphogen.core.operator import operator, OpCategory
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class Individual:
    """
    Represents a single individual in the population.

    Attributes:
        genome: Parameter vector (genes)
        fitness: Fitness score (higher = better)
        age: Number of generations this individual has existed
        id: Unique identifier
    """
    genome: np.ndarray
    fitness: float = 0.0
    age: int = 0
    id: int = 0

    def copy(self) -> 'Individual':
        """Return a copy of this individual (immutable semantics)"""
        return Individual(
            genome=self.genome.copy(),
            fitness=self.fitness,
            age=self.age,
            id=self.id
        )


@dataclass
class Population:
    """
    Represents a population of individuals.

    Attributes:
        individuals: List of individuals
        generation: Current generation number
        best_fitness_history: History of best fitness per generation
        mean_fitness_history: History of mean fitness per generation
    """
    individuals: List[Individual]
    generation: int = 0
    best_fitness_history: List[float] = None
    mean_fitness_history: List[float] = None

    def __post_init__(self):
        if self.best_fitness_history is None:
            self.best_fitness_history = []
        if self.mean_fitness_history is None:
            self.mean_fitness_history = []

    def copy(self) -> 'Population':
        """Return a copy of this population"""
        return Population(
            individuals=[ind.copy() for ind in self.individuals],
            generation=self.generation,
            best_fitness_history=self.best_fitness_history.copy(),
            mean_fitness_history=self.mean_fitness_history.copy()
        )

    @property
    def size(self) -> int:
        """Number of individuals in population"""
        return len(self.individuals)


class GeneticOperations:
    """
    Namespace for genetic algorithm operations.

    Follows Morphogen's 4-layer operator hierarchy:
    - Layer 1: Atomic (selection, crossover, mutation)
    - Layer 2: Composite (breed, evolve_generation)
    - Layer 3: Constructs (GA loop, island models)
    - Layer 4: Presets (common GA configurations)
    """

    # === LAYER 1: ATOMIC OPERATORS ===

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.CONSTRUCT,
        signature="(genome_size: int, seed: Optional[int]) -> Individual",
        deterministic=False,
        doc="Allocate a random individual"
    )
    def alloc_individual(genome_size: int, seed: Optional[int] = None) -> Individual:
        """
        Layer 1: Allocate a random individual.

        Args:
            genome_size: Length of genome vector
            seed: Random seed for deterministic initialization

        Returns:
            New Individual with random genome
        """
        if seed is not None:
            np.random.seed(seed)

        genome = np.random.randn(genome_size).astype(np.float32) * 0.5
        return Individual(genome=genome, fitness=0.0, age=0)

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.CONSTRUCT,
        signature="(pop_size: int, genome_size: int, seed: Optional[int]) -> Population",
        deterministic=False,
        doc="Allocate a random population"
    )
    def alloc_population(pop_size: int, genome_size: int,
                        seed: Optional[int] = None) -> Population:
        """
        Layer 1: Allocate a random population.

        Args:
            pop_size: Number of individuals
            genome_size: Length of each genome
            seed: Random seed for deterministic initialization

        Returns:
            New Population with random individuals
        """
        individuals = []
        for i in range(pop_size):
            ind_seed = None if seed is None else seed + i
            ind = GeneticOperations.alloc_individual(genome_size, seed=ind_seed)
            ind.id = i
            individuals.append(ind)

        return Population(individuals=individuals, generation=0)

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(population: Population, fitness_fn: Callable[[ndarray], float]) -> Population",
        deterministic=True,
        doc="Evaluate fitness for all individuals"
    )
    def evaluate_fitness(population: Population,
                        fitness_fn: Callable[[np.ndarray], float]) -> Population:
        """
        Layer 1: Evaluate fitness for all individuals.

        Args:
            population: Population to evaluate
            fitness_fn: Function mapping genome -> fitness score

        Returns:
            Population with updated fitness values
        """
        new_pop = population.copy()
        for ind in new_pop.individuals:
            ind.fitness = fitness_fn(ind.genome)
        return new_pop

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(population: Population) -> Population",
        deterministic=True,
        doc="Sort population by fitness (descending)"
    )
    def rank_population(population: Population) -> Population:
        """
        Layer 1: Sort population by fitness (descending).

        Args:
            population: Population to rank

        Returns:
            Population with individuals sorted by fitness
        """
        new_pop = population.copy()
        new_pop.individuals = sorted(new_pop.individuals,
                                    key=lambda x: x.fitness,
                                    reverse=True)
        return new_pop

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.QUERY,
        signature="(population: Population, tournament_size: int, seed: Optional[int]) -> Individual",
        deterministic=False,
        doc="Select individual via tournament selection"
    )
    def tournament_selection(population: Population, tournament_size: int = 3,
                           seed: Optional[int] = None) -> Individual:
        """
        Layer 1: Select individual via tournament selection.

        Randomly samples tournament_size individuals and returns the best.

        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament
            seed: Random seed for deterministic selection

        Returns:
            Selected individual (copy)
        """
        if seed is not None:
            np.random.seed(seed)

        tournament = np.random.choice(population.individuals, size=tournament_size, replace=False)
        winner = max(tournament, key=lambda x: x.fitness)
        return winner.copy()

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.QUERY,
        signature="(population: Population, seed: Optional[int]) -> Individual",
        deterministic=False,
        doc="Select individual via fitness-proportional roulette selection"
    )
    def roulette_selection(population: Population,
                          seed: Optional[int] = None) -> Individual:
        """
        Layer 1: Select individual via fitness-proportional roulette selection.

        Args:
            population: Population to select from
            seed: Random seed for deterministic selection

        Returns:
            Selected individual (copy)
        """
        if seed is not None:
            np.random.seed(seed)

        # Handle negative fitness by shifting
        fitnesses = np.array([ind.fitness for ind in population.individuals])
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-6

        # Normalize to probabilities
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            # Uniform selection if all fitnesses are zero
            idx = np.random.randint(0, len(population.individuals))
        else:
            probabilities = fitnesses / total_fitness
            idx = np.random.choice(len(population.individuals), p=probabilities)

        return population.individuals[idx].copy()

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.QUERY,
        signature="(population: Population, n_elite: int) -> List[Individual]",
        deterministic=True,
        doc="Select top N individuals (elitism)"
    )
    def elitism_select(population: Population, n_elite: int) -> List[Individual]:
        """
        Layer 1: Select top N individuals (elitism).

        Population should be ranked first.

        Args:
            population: Ranked population
            n_elite: Number of elite individuals to preserve

        Returns:
            List of elite individuals (copies)
        """
        n_elite = min(n_elite, population.size)
        return [ind.copy() for ind in population.individuals[:n_elite]]

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(genome: ndarray, mutation_rate: float, mutation_scale: float, seed: Optional[int]) -> ndarray",
        deterministic=False,
        doc="Mutate genome with Gaussian noise"
    )
    def mutate(genome: np.ndarray, mutation_rate: float = 0.1,
              mutation_scale: float = 0.3,
              seed: Optional[int] = None) -> np.ndarray:
        """
        Layer 1: Mutate genome with Gaussian noise.

        Args:
            genome: Genome vector to mutate
            mutation_rate: Probability of mutating each gene
            mutation_scale: Standard deviation of Gaussian noise
            seed: Random seed for deterministic mutation

        Returns:
            Mutated genome (new array)
        """
        if seed is not None:
            np.random.seed(seed)

        new_genome = genome.copy()
        mutation_mask = np.random.rand(len(genome)) < mutation_rate
        new_genome[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * mutation_scale
        return new_genome

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(genome1: ndarray, genome2: ndarray, seed: Optional[int]) -> Tuple[ndarray, ndarray]",
        deterministic=False,
        doc="Uniform crossover (each gene randomly from parent1 or parent2)"
    )
    def crossover_uniform(genome1: np.ndarray, genome2: np.ndarray,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Layer 1: Uniform crossover (each gene randomly from parent1 or parent2).

        Args:
            genome1: First parent genome
            genome2: Second parent genome
            seed: Random seed for deterministic crossover

        Returns:
            (offspring1, offspring2) genomes
        """
        if seed is not None:
            np.random.seed(seed)

        mask = np.random.rand(len(genome1)) < 0.5
        offspring1 = np.where(mask, genome1, genome2)
        offspring2 = np.where(mask, genome2, genome1)
        return offspring1, offspring2

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(genome1: ndarray, genome2: ndarray, seed: Optional[int]) -> Tuple[ndarray, ndarray]",
        deterministic=False,
        doc="Single-point crossover"
    )
    def crossover_single_point(genome1: np.ndarray, genome2: np.ndarray,
                              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Layer 1: Single-point crossover.

        Args:
            genome1: First parent genome
            genome2: Second parent genome
            seed: Random seed for deterministic crossover

        Returns:
            (offspring1, offspring2) genomes
        """
        if seed is not None:
            np.random.seed(seed)

        point = np.random.randint(0, len(genome1))
        offspring1 = np.concatenate([genome1[:point], genome2[point:]])
        offspring2 = np.concatenate([genome2[:point], genome1[point:]])
        return offspring1, offspring2

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(genome1: ndarray, genome2: ndarray, alpha: float) -> Tuple[ndarray, ndarray]",
        deterministic=True,
        doc="Blend crossover (weighted average)"
    )
    def crossover_blend(genome1: np.ndarray, genome2: np.ndarray,
                       alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Layer 1: Blend crossover (weighted average).

        Args:
            genome1: First parent genome
            genome2: Second parent genome
            alpha: Blending weight (0.5 = equal average)

        Returns:
            (offspring1, offspring2) genomes
        """
        offspring1 = alpha * genome1 + (1 - alpha) * genome2
        offspring2 = (1 - alpha) * genome1 + alpha * genome2
        return offspring1, offspring2

    # === LAYER 2: COMPOSITE OPERATORS ===

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(parent1: Individual, parent2: Individual, crossover_method: str, mutation_rate: float, mutation_scale: float, seed: Optional[int]) -> Tuple[Individual, Individual]",
        deterministic=False,
        doc="Breed two parents to produce two offspring"
    )
    def breed(parent1: Individual, parent2: Individual,
             crossover_method: str = 'uniform',
             mutation_rate: float = 0.1,
             mutation_scale: float = 0.3,
             seed: Optional[int] = None) -> Tuple[Individual, Individual]:
        """
        Layer 2: Breed two parents to produce two offspring.

        Combines crossover + mutation.

        Args:
            parent1: First parent
            parent2: Second parent
            crossover_method: 'uniform', 'single_point', or 'blend'
            mutation_rate: Probability of mutating each gene
            mutation_scale: Standard deviation of mutation noise
            seed: Random seed for deterministic breeding

        Returns:
            (offspring1, offspring2) individuals
        """
        # Crossover
        if crossover_method == 'uniform':
            genome1, genome2 = GeneticOperations.crossover_uniform(
                parent1.genome, parent2.genome, seed=seed
            )
        elif crossover_method == 'single_point':
            genome1, genome2 = GeneticOperations.crossover_single_point(
                parent1.genome, parent2.genome, seed=seed
            )
        elif crossover_method == 'blend':
            genome1, genome2 = GeneticOperations.crossover_blend(
                parent1.genome, parent2.genome
            )
        else:
            raise ValueError(f"Unknown crossover method: {crossover_method}")

        # Mutation
        mutation_seed1 = None if seed is None else seed + 1000
        mutation_seed2 = None if seed is None else seed + 2000
        genome1 = GeneticOperations.mutate(genome1, mutation_rate, mutation_scale, seed=mutation_seed1)
        genome2 = GeneticOperations.mutate(genome2, mutation_rate, mutation_scale, seed=mutation_seed2)

        offspring1 = Individual(genome=genome1, fitness=0.0, age=0)
        offspring2 = Individual(genome=genome2, fitness=0.0, age=0)

        return offspring1, offspring2

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(population: Population, fitness_fn: Callable[[ndarray], float], n_elite: int, selection_method: str, crossover_method: str, mutation_rate: float, mutation_scale: float, seed: Optional[int]) -> Population",
        deterministic=False,
        doc="Evolve population for one generation"
    )
    def evolve_generation(population: Population,
                         fitness_fn: Callable[[np.ndarray], float],
                         n_elite: int = 4,
                         selection_method: str = 'tournament',
                         crossover_method: str = 'uniform',
                         mutation_rate: float = 0.1,
                         mutation_scale: float = 0.3,
                         seed: Optional[int] = None) -> Population:
        """
        Layer 2: Evolve population for one generation.

        Process:
        1. Evaluate fitness
        2. Rank population
        3. Select elites
        4. Breed new offspring
        5. Combine elites + offspring

        Args:
            population: Current population
            fitness_fn: Function mapping genome -> fitness
            n_elite: Number of top individuals to preserve
            selection_method: 'tournament' or 'roulette'
            crossover_method: 'uniform', 'single_point', or 'blend'
            mutation_rate: Mutation probability
            mutation_scale: Mutation noise scale
            seed: Random seed for deterministic evolution

        Returns:
            New population for next generation
        """
        # Evaluate and rank
        population = GeneticOperations.evaluate_fitness(population, fitness_fn)
        population = GeneticOperations.rank_population(population)

        # Track statistics
        fitnesses = [ind.fitness for ind in population.individuals]
        best_fitness = np.max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        population.best_fitness_history.append(best_fitness)
        population.mean_fitness_history.append(mean_fitness)

        # Select elites
        elites = GeneticOperations.elitism_select(population, n_elite)

        # Breed offspring to fill remaining slots
        n_offspring_needed = population.size - n_elite
        offspring = []

        for i in range(n_offspring_needed // 2):
            # Select parents
            if selection_method == 'tournament':
                selection_seed1 = None if seed is None else seed + i * 10
                selection_seed2 = None if seed is None else seed + i * 10 + 1
                parent1 = GeneticOperations.tournament_selection(population, seed=selection_seed1)
                parent2 = GeneticOperations.tournament_selection(population, seed=selection_seed2)
            elif selection_method == 'roulette':
                selection_seed1 = None if seed is None else seed + i * 10
                selection_seed2 = None if seed is None else seed + i * 10 + 1
                parent1 = GeneticOperations.roulette_selection(population, seed=selection_seed1)
                parent2 = GeneticOperations.roulette_selection(population, seed=selection_seed2)
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")

            # Breed
            breed_seed = None if seed is None else seed + i * 100
            child1, child2 = GeneticOperations.breed(
                parent1, parent2,
                crossover_method=crossover_method,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
                seed=breed_seed
            )
            offspring.extend([child1, child2])

        # Handle odd population size
        if len(offspring) > n_offspring_needed:
            offspring = offspring[:n_offspring_needed]

        # Combine elites + offspring
        new_individuals = elites + offspring

        # Assign IDs and increment ages
        for i, ind in enumerate(new_individuals):
            ind.id = i
            if i < n_elite:
                ind.age += 1  # Elites age

        return Population(
            individuals=new_individuals,
            generation=population.generation + 1,
            best_fitness_history=population.best_fitness_history.copy(),
            mean_fitness_history=population.mean_fitness_history.copy()
        )

    # === LAYER 3: EVOLUTION CONSTRUCTS ===

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.TRANSFORM,
        signature="(population: Population, fitness_fn: Callable[[ndarray], float], n_generations: int, callback: Optional[Callable[[Population], None]]) -> Population",
        deterministic=True,
        doc="Run complete evolutionary algorithm"
    )
    def run_evolution(population: Population,
                     fitness_fn: Callable[[np.ndarray], float],
                     n_generations: int = 100,
                     callback: Optional[Callable[[Population], None]] = None,
                     **evolve_kwargs) -> Population:
        """
        Layer 3: Run complete evolutionary algorithm.

        Args:
            population: Initial population
            fitness_fn: Fitness evaluation function
            n_generations: Number of generations to evolve
            callback: Optional function called each generation with current population
            **evolve_kwargs: Additional arguments passed to evolve_generation()

        Returns:
            Final evolved population
        """
        for gen in range(n_generations):
            population = GeneticOperations.evolve_generation(
                population, fitness_fn, **evolve_kwargs
            )

            if callback is not None:
                callback(population)

        return population

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.QUERY,
        signature="(population: Population) -> Individual",
        deterministic=True,
        doc="Get the best individual from population"
    )
    def get_best_individual(population: Population) -> Individual:
        """
        Layer 3: Get the best individual from population.

        Population does not need to be ranked.

        Args:
            population: Population to search

        Returns:
            Individual with highest fitness
        """
        return max(population.individuals, key=lambda x: x.fitness).copy()

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.QUERY,
        signature="(population: Population) -> float",
        deterministic=True,
        doc="Compute genetic diversity of population"
    )
    def get_diversity(population: Population) -> float:
        """
        Layer 3: Compute genetic diversity of population.

        Uses average pairwise Euclidean distance between genomes.

        Args:
            population: Population to measure

        Returns:
            Diversity metric (higher = more diverse)
        """
        if population.size < 2:
            return 0.0

        genomes = np.array([ind.genome for ind in population.individuals])
        distances = []

        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                dist = np.linalg.norm(genomes[i] - genomes[j])
                distances.append(dist)

        return np.mean(distances)

    # === LAYER 4: PRESETS ===

    @staticmethod
    @operator(
        domain="genetic",
        category=OpCategory.CONSTRUCT,
        signature="(pop_size: int, genome_size: int, seed: Optional[int]) -> Population",
        deterministic=False,
        doc="Preset population for Flappy Bird neural controller evolution"
    )
    def flappy_bird_evolution(pop_size: int = 128,
                             genome_size: int = 73,  # [4,8,1] MLP = 32+8+8+1 = 73 params
                             seed: Optional[int] = None) -> Population:
        """
        Layer 4: Preset population for Flappy Bird neural controller evolution.

        Args:
            pop_size: Population size
            genome_size: Genome length (73 for [4,8,1] MLP)
            seed: Random seed

        Returns:
            Initialized population
        """
        return GeneticOperations.alloc_population(pop_size, genome_size, seed=seed)


# Module-level singleton for convenience
genetic = GeneticOperations()


# === HELPER FUNCTIONS ===

def parallel_fitness_evaluation(population: Population,
                                fitness_fn: Callable[[List[np.ndarray]], List[float]]) -> Population:
    """
    Helper: Evaluate fitness for entire population in parallel.

    Useful when fitness function can process batches efficiently.

    Args:
        population: Population to evaluate
        fitness_fn: Function mapping [genomes] -> [fitness scores]

    Returns:
        Population with updated fitness values
    """
    new_pop = population.copy()
    genomes = [ind.genome for ind in new_pop.individuals]
    fitnesses = fitness_fn(genomes)

    for ind, fitness in zip(new_pop.individuals, fitnesses):
        ind.fitness = fitness

    return new_pop


# Export operators for domain registry discovery
alloc_individual = GeneticOperations.alloc_individual
alloc_population = GeneticOperations.alloc_population
mutate = GeneticOperations.mutate
crossover_single_point = GeneticOperations.crossover_single_point
crossover_uniform = GeneticOperations.crossover_uniform
crossover_blend = GeneticOperations.crossover_blend
breed = GeneticOperations.breed
roulette_selection = GeneticOperations.roulette_selection
tournament_selection = GeneticOperations.tournament_selection
elitism_select = GeneticOperations.elitism_select
rank_population = GeneticOperations.rank_population
get_best_individual = GeneticOperations.get_best_individual
get_diversity = GeneticOperations.get_diversity
evolve_generation = GeneticOperations.evolve_generation
run_evolution = GeneticOperations.run_evolution
evaluate_fitness = GeneticOperations.evaluate_fitness
flappy_bird_evolution = GeneticOperations.flappy_bird_evolution
