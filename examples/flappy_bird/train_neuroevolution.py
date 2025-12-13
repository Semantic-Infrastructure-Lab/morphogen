#!/usr/bin/env python3
"""
Flappy Bird Neuroevolution Training
====================================

This example demonstrates Morphogen's multi-domain capabilities by training
neural network controllers for Flappy Bird using genetic algorithms.

Showcases:
- Game physics domain (flappy.py)
- Neural network domain (neural.py)
- Genetic algorithm domain (genetic.py)
- Parallel batch simulation
- Real-time visualization and telemetry

This is the "hello world" of multi-domain AI simulation in Morphogen.
"""

import sys
import numpy as np
from typing import List
import time

# Morphogen domains
from morphogen.stdlib.flappy import flappy, GameState
from morphogen.stdlib.neural import neural, MLP
from morphogen.stdlib.genetic import genetic, Population


# === CONFIGURATION ===

class Config:
    """Training configuration"""
    # Population
    POPULATION_SIZE = 128
    N_GENERATIONS = 50

    # Neural network architecture
    NN_ARCHITECTURE = [4, 8, 1]  # [bird_y, bird_vel, pipe_x, pipe_gap_y] -> flap_prob

    # Genetic algorithm
    N_ELITE = 4
    MUTATION_RATE = 0.10
    MUTATION_SCALE = 0.30
    CROSSOVER_METHOD = 'uniform'
    SELECTION_METHOD = 'tournament'

    # Game simulation
    N_PIPES = 3
    MAX_STEPS_PER_EPISODE = 1000
    GRAVITY = 1.5
    FLAP_STRENGTH = 0.35
    PIPE_SPEED = 0.5
    DT = 0.016  # ~60 FPS

    # Rewards
    FRAME_REWARD = 1.0
    PIPE_REWARD = 10.0

    # Misc
    SEED = 42
    VERBOSE = True


# === NEURAL NETWORK CONTROLLER ===

def create_controller(genome: np.ndarray) -> MLP:
    """
    Create MLP controller from genome.

    Args:
        genome: Flat parameter vector

    Returns:
        MLP network with parameters set
    """
    # Create template network
    template = neural.alloc_mlp(
        layer_sizes=Config.NN_ARCHITECTURE,
        activations=['tanh', 'sigmoid']
    )

    # Set parameters from genome
    return neural.set_parameters(template, genome)


def controller_decision(network: MLP, observation: np.ndarray) -> bool:
    """
    Make flap decision from network output.

    Args:
        network: MLP controller
        observation: Sensor vector [4,]

    Returns:
        True to flap, False otherwise
    """
    output = neural.forward(observation.reshape(1, -1), network)
    return output[0, 0] > 0.5


# === FITNESS EVALUATION ===

def evaluate_individual(genome: np.ndarray) -> float:
    """
    Evaluate fitness of a single genome by simulating a game episode.

    Args:
        genome: Neural network parameters

    Returns:
        Fitness score (total reward accumulated)
    """
    # Create controller
    controller = create_controller(genome)

    # Initialize game state (single bird)
    state = flappy.alloc_game(n_birds=1, n_pipes=Config.N_PIPES)

    # Run episode
    for step in range(Config.MAX_STEPS_PER_EPISODE):
        # Extract observation
        obs = flappy.extract_sensors(state, bird_idx=0)

        # Get action from controller
        action = controller_decision(controller, obs)
        actions = np.array([action])

        # Step simulation
        state = flappy.step(
            state, actions,
            gravity=Config.GRAVITY,
            flap_strength=Config.FLAP_STRENGTH,
            pipe_speed=Config.PIPE_SPEED,
            dt=Config.DT,
            frame_reward=Config.FRAME_REWARD,
            pipe_reward=Config.PIPE_REWARD
        )

        # Check if bird died
        if not state.bird_alive[0]:
            break

    # Return final score as fitness
    fitness = float(state.bird_score[0])
    return fitness


def evaluate_population_parallel(genomes: List[np.ndarray]) -> List[float]:
    """
    Evaluate entire population in parallel using vectorized simulation.

    This is MUCH faster than evaluating individuals sequentially.

    Args:
        genomes: List of parameter vectors

    Returns:
        List of fitness scores
    """
    n_birds = len(genomes)

    # Create all controllers
    controllers = [create_controller(genome) for genome in genomes]

    # Initialize batch game state (all birds at once)
    state = flappy.alloc_game(n_birds=n_birds, n_pipes=Config.N_PIPES)

    # Run episode
    for step in range(Config.MAX_STEPS_PER_EPISODE):
        # Check if any birds are still alive
        if not np.any(state.bird_alive):
            break

        # Extract observations for all birds
        observations = flappy.extract_sensors_batch(state)

        # Get actions from all controllers
        actions = np.zeros(n_birds, dtype=bool)
        for i in range(n_birds):
            if state.bird_alive[i]:
                actions[i] = controller_decision(controllers[i], observations[i])

        # Step simulation (all birds at once)
        state = flappy.step(
            state, actions,
            gravity=Config.GRAVITY,
            flap_strength=Config.FLAP_STRENGTH,
            pipe_speed=Config.PIPE_SPEED,
            dt=Config.DT,
            frame_reward=Config.FRAME_REWARD,
            pipe_reward=Config.PIPE_REWARD
        )

    # Return final scores as fitness
    fitnesses = state.bird_score.tolist()
    return fitnesses


# === TRAINING LOOP ===

def train():
    """
    Main training loop: evolve neural network controllers using genetic algorithm.
    """
    print("=" * 60)
    print("Morphogen Flappy Bird Neuroevolution")
    print("=" * 60)
    print()
    print(f"Population size: {Config.POPULATION_SIZE}")
    print(f"Generations: {Config.N_GENERATIONS}")
    print(f"Network architecture: {Config.NN_ARCHITECTURE}")
    print(f"Elite preservation: {Config.N_ELITE}")
    print(f"Mutation rate: {Config.MUTATION_RATE}")
    print(f"Mutation scale: {Config.MUTATION_SCALE}")
    print()

    # Calculate genome size
    # For [4, 8, 1]: weights=(4*8 + 8*1) + biases=(8 + 1) = 32 + 8 + 8 + 1 = 49
    template = neural.alloc_mlp(Config.NN_ARCHITECTURE, activations=['tanh', 'sigmoid'])
    genome_size = template.count_parameters()
    print(f"Genome size: {genome_size} parameters")
    print()

    # Initialize population
    print("Initializing population...")
    population = genetic.alloc_population(
        pop_size=Config.POPULATION_SIZE,
        genome_size=genome_size,
        seed=Config.SEED
    )

    # Training statistics
    best_fitness_overall = -np.inf
    best_genome_overall = None

    print("Starting evolution...")
    print()

    # Evolution loop
    for gen in range(Config.N_GENERATIONS):
        start_time = time.time()

        # Evaluate fitness (parallel)
        genomes = [ind.genome for ind in population.individuals]
        fitnesses = evaluate_population_parallel(genomes)

        # Update population fitness
        for ind, fitness in zip(population.individuals, fitnesses):
            ind.fitness = fitness

        # Rank population
        population = genetic.rank_population(population)

        # Track statistics
        best_fitness = population.individuals[0].fitness
        mean_fitness = np.mean(fitnesses)
        diversity = genetic.get_diversity(population)

        # Update overall best
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_genome_overall = population.individuals[0].genome.copy()

        # Print progress
        elapsed = time.time() - start_time
        print(f"Gen {gen + 1:3d}/{Config.N_GENERATIONS} | "
              f"Best: {best_fitness:7.1f} | "
              f"Mean: {mean_fitness:7.1f} | "
              f"Diversity: {diversity:6.2f} | "
              f"Time: {elapsed:.2f}s")

        # Evolve to next generation (if not last)
        if gen < Config.N_GENERATIONS - 1:
            population = genetic.evolve_generation(
                population,
                fitness_fn=lambda genome: 0.0,  # Already evaluated
                n_elite=Config.N_ELITE,
                selection_method=Config.SELECTION_METHOD,
                crossover_method=Config.CROSSOVER_METHOD,
                mutation_rate=Config.MUTATION_RATE,
                mutation_scale=Config.MUTATION_SCALE,
                seed=Config.SEED + gen
            )

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Best fitness achieved: {best_fitness_overall:.1f}")
    print("=" * 60)

    return best_genome_overall, population


# === REPLAY BEST AGENT ===

def replay_best_agent(genome: np.ndarray, visualize: bool = False):
    """
    Replay the best evolved agent and optionally visualize.

    Args:
        genome: Best neural network parameters
        visualize: If True, render game state (requires matplotlib)
    """
    print()
    print("Replaying best agent...")

    # Create controller
    controller = create_controller(genome)

    # Initialize game
    state = flappy.alloc_game(n_birds=1, n_pipes=Config.N_PIPES)

    # Track trajectory
    trajectory = []

    # Run episode
    for step in range(Config.MAX_STEPS_PER_EPISODE):
        # Extract observation
        obs = flappy.extract_sensors(state, bird_idx=0)

        # Get action
        action = controller_decision(controller, obs)
        actions = np.array([action])

        # Record state
        trajectory.append({
            'step': step,
            'bird_y': state.bird_y[0],
            'bird_velocity': state.bird_velocity[0],
            'score': state.bird_score[0],
            'pipes_passed': state.bird_pipes_passed[0],
            'action': action
        })

        # Step simulation
        state = flappy.step(
            state, actions,
            gravity=Config.GRAVITY,
            flap_strength=Config.FLAP_STRENGTH,
            pipe_speed=Config.PIPE_SPEED,
            dt=Config.DT,
            frame_reward=Config.FRAME_REWARD,
            pipe_reward=Config.PIPE_REWARD
        )

        # Check if bird died
        if not state.bird_alive[0]:
            break

    # Print summary
    final_score = state.bird_score[0]
    pipes_passed = state.bird_pipes_passed[0]
    steps_survived = len(trajectory)

    print(f"  Final score: {final_score:.1f}")
    print(f"  Pipes passed: {pipes_passed}")
    print(f"  Steps survived: {steps_survived}")

    if visualize:
        visualize_trajectory(trajectory, state)

    return trajectory


def visualize_trajectory(trajectory: List[dict], final_state: GameState):
    """
    Visualize trajectory using matplotlib (if available).

    Args:
        trajectory: List of state snapshots
        final_state: Final game state
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available for visualization)")
        return

    # Extract data
    steps = [t['step'] for t in trajectory]
    bird_y = [t['bird_y'] for t in trajectory]
    bird_vel = [t['bird_velocity'] for t in trajectory]
    scores = [t['score'] for t in trajectory]
    actions = [t['action'] for t in trajectory]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot 1: Bird position
    axes[0].plot(steps, bird_y, label='Bird Y', color='blue')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Bottom boundary')
    axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Top boundary')
    axes[0].set_ylabel('Bird Y Position')
    axes[0].set_title('Bird Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Velocity
    axes[1].plot(steps, bird_vel, label='Velocity', color='green')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Bird Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Score and actions
    ax3a = axes[2]
    ax3b = ax3a.twinx()

    ax3a.plot(steps, scores, label='Score', color='orange')
    ax3a.set_ylabel('Score', color='orange')
    ax3a.tick_params(axis='y', labelcolor='orange')

    flap_steps = [s for s, a in zip(steps, actions) if a]
    flap_heights = [bird_y[i] for i, a in enumerate(actions) if a]
    ax3b.scatter(flap_steps, flap_heights, label='Flaps', color='red', alpha=0.5, s=20)
    ax3b.set_ylabel('Flap Events', color='red')
    ax3b.tick_params(axis='y', labelcolor='red')
    ax3b.set_ylim(0, 1)

    axes[2].set_xlabel('Time Steps')
    axes[2].set_title('Score and Flap Actions')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/flappy_bird/best_agent_trajectory.png', dpi=150)
    print(f"  Saved visualization to: best_agent_trajectory.png")
    plt.close()


# === MAIN ===

def main():
    """Main entry point"""
    # Train
    best_genome, final_population = train()

    # Save best genome
    output_path = '/home/user/morphogen/examples/flappy_bird/best_genome.npy'
    np.save(output_path, best_genome)
    print()
    print(f"Saved best genome to: {output_path}")

    # Replay best agent
    replay_best_agent(best_genome, visualize=True)

    print()
    print("Done!")


if __name__ == '__main__':
    main()
