#!/usr/bin/env python3
"""
Flappy Bird Visualization
==========================

Render Flappy Bird game state using matplotlib.
Can replay saved genomes or visualize training in real-time.

This demonstrates integration with Morphogen's visual domain.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Optional, List

from morphogen.stdlib.flappy import flappy, GameState
from morphogen.stdlib.neural import neural, MLP


class FlappyBirdRenderer:
    """
    Renderer for Flappy Bird game state.

    Uses matplotlib to draw birds, pipes, and game info.
    """

    def __init__(self, figsize=(10, 6)):
        """
        Initialize renderer.

        Args:
            figsize: Figure size (width, height)
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_aspect('equal')
        self.ax.set_title('Morphogen Flappy Bird', fontsize=16, fontweight='bold')

        # Visual elements
        self.bird_artists = []
        self.pipe_artists = []
        self.text_artists = []

    def render_frame(self, state: GameState, highlight_best: bool = True):
        """
        Render a single game frame.

        Args:
            state: Current game state
            highlight_best: If True, highlight bird with highest score
        """
        # Clear previous frame
        for artist in self.bird_artists + self.pipe_artists + self.text_artists:
            artist.remove()
        self.bird_artists.clear()
        self.pipe_artists.clear()
        self.text_artists.clear()

        # Draw background
        self.ax.set_facecolor('#87CEEB')  # Sky blue

        # Draw ground
        ground = patches.Rectangle(
            (0, 0), 1.0, 0.05,
            facecolor='#8B4513', edgecolor='black', linewidth=2
        )
        self.pipe_artists.append(self.ax.add_patch(ground))

        # Draw pipes
        for i in range(state.n_pipes):
            x = state.pipe_x[i]
            gap_center = state.pipe_gap_y[i]
            gap_size = state.pipe_gap_size[i]
            width = 0.1

            # Only draw visible pipes
            if x < -0.2 or x > 1.2:
                continue

            # Top pipe
            top_pipe = patches.Rectangle(
                (x, gap_center + gap_size / 2), width, 1.0 - (gap_center + gap_size / 2),
                facecolor='#228B22', edgecolor='black', linewidth=2
            )
            self.pipe_artists.append(self.ax.add_patch(top_pipe))

            # Bottom pipe
            bottom_pipe = patches.Rectangle(
                (x, 0.05), width, gap_center - gap_size / 2 - 0.05,
                facecolor='#228B22', edgecolor='black', linewidth=2
            )
            self.pipe_artists.append(self.ax.add_patch(bottom_pipe))

        # Find best bird
        best_idx = np.argmax(state.bird_score) if highlight_best else -1

        # Draw birds
        bird_x = 0.2
        for i in range(state.n_birds):
            if not state.bird_alive[i]:
                continue

            y = state.bird_y[i]

            # Color: yellow for best, blue for others
            color = '#FFD700' if i == best_idx else '#4169E1'
            edge_color = 'red' if i == best_idx else 'black'
            edge_width = 3 if i == best_idx else 1.5

            # Draw bird as circle
            bird = plt.Circle(
                (bird_x, y), 0.03,
                facecolor=color, edgecolor=edge_color, linewidth=edge_width, zorder=10
            )
            self.bird_artists.append(self.ax.add_patch(bird))

        # Draw stats
        alive_count = np.sum(state.bird_alive)
        best_score = np.max(state.bird_score) if state.n_birds > 0 else 0
        mean_score = np.mean(state.bird_score[state.bird_alive]) if alive_count > 0 else 0

        stats_text = (
            f"Alive: {alive_count}/{state.n_birds}\n"
            f"Best Score: {best_score:.0f}\n"
            f"Mean Score: {mean_score:.0f}"
        )
        text = self.ax.text(
            0.02, 0.95, stats_text,
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        self.text_artists.append(text)

    def save_frame(self, filename: str):
        """Save current frame to file"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')

    def close(self):
        """Close figure"""
        plt.close(self.fig)


def visualize_episode(genome: Optional[np.ndarray] = None,
                     max_steps: int = 500,
                     save_path: Optional[str] = None):
    """
    Visualize a single game episode.

    Args:
        genome: Neural network parameters (if None, uses random control)
        max_steps: Maximum simulation steps
        save_path: If provided, save frames to this path (as .png or .gif)
    """
    print("Running visualization...")

    # Create controller if genome provided
    if genome is not None:
        from train_neuroevolution import create_controller, controller_decision
        controller = create_controller(genome)
        use_nn = True
        print("  Using neural network controller")
    else:
        use_nn = False
        print("  Using random controller")

    # Initialize game
    state = flappy.alloc_game(n_birds=1, n_pipes=3, seed=42)

    # Initialize renderer
    renderer = FlappyBirdRenderer()

    # Collect frames if saving animation
    frames = []

    # Run simulation
    for step in range(max_steps):
        # Render frame
        renderer.render_frame(state, highlight_best=True)

        if save_path and save_path.endswith('.gif'):
            # Save frame for GIF
            import io
            buf = io.BytesIO()
            renderer.fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            from PIL import Image
            frames.append(Image.open(buf))

        plt.pause(0.01)  # Small pause for rendering

        # Get action
        if use_nn:
            obs = flappy.extract_sensors(state, bird_idx=0)
            action = controller_decision(controller, obs)
        else:
            action = np.random.rand() < 0.05

        actions = np.array([action])

        # Step simulation
        state = flappy.step(state, actions)

        # Check if bird died
        if not state.bird_alive[0]:
            print(f"  Bird died at step {step}")
            print(f"  Final score: {state.bird_score[0]:.0f}")
            print(f"  Pipes passed: {state.bird_pipes_passed[0]}")
            break

    # Save output
    if save_path:
        if save_path.endswith('.gif'):
            if frames:
                frames[0].save(
                    save_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=50,
                    loop=0
                )
                print(f"  Saved animation to: {save_path}")
        else:
            renderer.save_frame(save_path)
            print(f"  Saved frame to: {save_path}")

    renderer.close()
    print("Visualization complete!")


def plot_training_progress(population_history: List,
                          save_path: str = 'training_progress.png'):
    """
    Plot training progress over generations.

    Args:
        population_history: List of populations (one per generation)
        save_path: Where to save the plot
    """
    # Extract statistics
    generations = list(range(len(population_history)))
    best_fitness = [pop.best_fitness_history[-1] for pop in population_history]
    mean_fitness = [pop.mean_fitness_history[-1] for pop in population_history]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, best_fitness, label='Best Fitness', color='blue', linewidth=2)
    ax.plot(generations, mean_fitness, label='Mean Fitness', color='orange', linewidth=2, alpha=0.7)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('Neuroevolution Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training progress plot to: {save_path}")
    plt.close()


def main():
    """Main entry point for visualization"""
    import sys
    import os

    print("Flappy Bird Visualization")
    print("=" * 50)
    print()

    # Check for saved genome
    genome_path = '/home/user/morphogen/examples/flappy_bird/best_genome.npy'

    if os.path.exists(genome_path):
        print(f"Found saved genome: {genome_path}")
        genome = np.load(genome_path)
        visualize_episode(
            genome=genome,
            max_steps=1000,
            save_path='/home/user/morphogen/examples/flappy_bird/best_agent.png'
        )
    else:
        print("No saved genome found. Using random control.")
        visualize_episode(
            genome=None,
            max_steps=200,
            save_path='/home/user/morphogen/examples/flappy_bird/random_agent.png'
        )


if __name__ == '__main__':
    main()
