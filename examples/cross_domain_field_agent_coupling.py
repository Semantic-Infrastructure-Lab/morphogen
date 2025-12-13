"""
Cross-Domain Field-Agent Coupling Example

Demonstrates the new cross-domain operator composition infrastructure by coupling
a flow field with particle agents. This shows:

1. Field → Agent: Particles sample velocity from flow field for forces
2. Agent → Field: Particle positions deposit to density field
3. Bidirectional coupling in a single simulation loop

Use case: Particles in a flow field (e.g., smoke, debris in wind, plankton in ocean currents)

This example showcases Morphogen's unique ability to compose Field and Agent domains
bidirectionally in real-time - something impossible in traditional frameworks.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Import cross-domain infrastructure
from morphogen.cross_domain.interface import FieldToAgentInterface, AgentToFieldInterface
from morphogen.cross_domain.registry import CrossDomainRegistry

# Import Morphogen visualization
from morphogen.stdlib import visual, Visual
from morphogen.stdlib.field import Field2D

# Optional: matplotlib for interactive visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FlowFieldAgentSimulation:
    """
    Simulation coupling a flow field with particle agents.

    The flow field represents a velocity field (e.g., wind, water current).
    Agents are particles that:
    1. Sample velocity from the field at their position
    2. Move according to the sampled velocity
    3. Deposit density onto a field for visualization
    """

    def __init__(self, grid_size=128, num_agents=500, seed=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.seed = seed

        # Set random seed for deterministic behavior
        if seed is not None:
            np.random.seed(seed)

        # Initialize flow field (velocity field with vortex)
        self.velocity_field = self._create_vortex_field()

        # Initialize agent positions (random)
        self.agent_positions = np.random.rand(num_agents, 2) * grid_size
        self.agent_velocities = np.zeros((num_agents, 2), dtype=np.float32)

        # Create cross-domain interfaces
        self.field_to_agent = FieldToAgentInterface(
            field=self.velocity_field,
            positions=self.agent_positions
        )

        self.agent_to_field = AgentToFieldInterface(
            positions=self.agent_positions,
            values=np.ones(num_agents),  # Density contribution
            field_shape=(grid_size, grid_size),
            method="accumulate"
        )

        # History for visualization
        self.density_field = np.zeros((grid_size, grid_size), dtype=np.float32)

    def _create_vortex_field(self):
        """Create a velocity field with a vortex pattern."""
        y, x = np.mgrid[0:self.grid_size, 0:self.grid_size]
        center_x, center_y = self.grid_size / 2, self.grid_size / 2

        # Radial vectors
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2) + 1e-10  # Avoid division by zero

        # Circular flow (vortex)
        strength = 2.0
        vx = -dy / r * strength * np.exp(-r / 30.0)
        vy = dx / r * strength * np.exp(-r / 30.0)

        # Stack into velocity field (H, W, 2)
        velocity_field = np.stack([vy, vx], axis=2).astype(np.float32)

        return velocity_field

    def step(self, dt=0.5):
        """
        Advance simulation by one timestep.

        Cross-domain flow:
        1. Field → Agent: Sample velocity from flow field
        2. Agent dynamics: Update positions
        3. Agent → Field: Deposit density for visualization
        """
        # CROSS-DOMAIN OPERATION 1: Field → Agent
        # Sample flow field velocities at agent positions
        sampled_velocities = self.field_to_agent.transform(self.velocity_field)

        # Agent dynamics: Integrate velocity
        self.agent_velocities = sampled_velocities
        self.agent_positions += self.agent_velocities * dt

        # Boundary conditions (periodic)
        self.agent_positions %= self.grid_size

        # CROSS-DOMAIN OPERATION 2: Agent → Field
        # Deposit agent density onto field for visualization
        self.agent_to_field.positions = self.agent_positions
        self.density_field = self.agent_to_field.transform(
            (self.agent_positions, np.ones(self.num_agents))
        )

        # Apply decay to density field (fade over time)
        self.density_field *= 0.95

    def render_frame(self, width=512, height=512) -> Visual:
        """
        Render current simulation state as a Visual using Morphogen stdlib.

        Returns a composite visualization showing:
        - Background: Flow field (velocity magnitude)
        - Overlay: Agent density field (hot colormap)
        - Particles: Agent positions as bright dots

        Args:
            width: Output width in pixels
            height: Output height in pixels

        Returns:
            Visual object ready for export
        """
        # Scale density field to output resolution if needed
        if width != self.grid_size or height != self.grid_size:
            # For now, assume grid_size matches output resolution
            # In production, would add proper interpolation
            pass

        # Create density field visualization
        density_field_obj = Field2D(self.density_field.astype(np.float32))
        density_vis = visual.colorize(density_field_obj, palette="fire", vmin=0.0, vmax=3.0)

        # Add particle positions as bright spots
        # Create a particle overlay field
        particle_field = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for pos in self.agent_positions:
            y, x = int(pos[0]), int(pos[1])
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                # Add bright spot at particle position
                particle_field[y, x] = 1.0
                # Add glow around particle (3x3 kernel)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                            particle_field[ny, nx] = max(particle_field[ny, nx], 0.5)

        # Blend particle field with density visualization (additive)
        # Convert particle field to RGB
        particle_rgb = np.stack([particle_field] * 3, axis=-1)

        # Additive blend
        result_data = np.clip(density_vis.data + particle_rgb * 0.8, 0.0, 1.0)

        return visual.Visual(result_data)

    def run_headless(self, steps=200):
        """Run simulation without visualization (for testing)."""
        for i in range(steps):
            self.step()
            if i % 50 == 0:
                print(f"Step {i}/{steps}")
                print(f"  Agent positions range: [{self.agent_positions.min():.2f}, {self.agent_positions.max():.2f}]")
                print(f"  Density field max: {self.density_field.max():.2f}")
                print(f"  Number of active cells: {np.sum(self.density_field > 0)}")

    def visualize(self, steps=200, interval=50):
        """
        Create an animated visualization of the simulation.

        Shows:
        - Flow field vectors (background)
        - Particle positions (red dots)
        - Density field (heat map overlay)
        """
        if not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib not available. Skipping visualization.")
            print("   Install with: pip install matplotlib")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Flow field + agents
        y, x = np.mgrid[0:self.grid_size:8, 0:self.grid_size:8]
        vx_sampled = self.velocity_field[::8, ::8, 1]
        vy_sampled = self.velocity_field[::8, ::8, 0]

        quiver = ax1.quiver(x, y, vx_sampled, vy_sampled, alpha=0.5, scale=50)
        scatter = ax1.scatter(
            self.agent_positions[:, 1],
            self.agent_positions[:, 0],
            c='red', s=10, alpha=0.6
        )
        ax1.set_xlim(0, self.grid_size)
        ax1.set_ylim(0, self.grid_size)
        ax1.set_aspect('equal')
        ax1.set_title('Flow Field + Agents')
        ax1.invert_yaxis()

        # Plot 2: Density field
        density_img = ax2.imshow(
            self.density_field,
            cmap='hot',
            vmin=0,
            vmax=5,
            interpolation='bilinear'
        )
        ax2.set_title('Agent Density Field')
        plt.colorbar(density_img, ax=ax2)

        def animate(frame):
            self.step()

            # Update agent positions
            scatter.set_offsets(self.agent_positions[:, [1, 0]])

            # Update density field
            density_img.set_data(self.density_field)

            return scatter, density_img

        anim = FuncAnimation(
            fig,
            animate,
            frames=steps,
            interval=interval,
            blit=True
        )

        plt.tight_layout()
        plt.show()


def main():
    """Run the cross-domain coupling example."""
    print("=" * 60)
    print("Cross-Domain Field-Agent Coupling Example")
    print("=" * 60)
    print()
    print("This example demonstrates bidirectional coupling:")
    print("  1. Field → Agent: Particles sample flow velocity")
    print("  2. Agent → Field: Particles deposit density")
    print()

    # Create simulation
    sim = FlowFieldAgentSimulation(grid_size=128, num_agents=500)

    # Verify cross-domain infrastructure
    print("Cross-domain transform registry:")
    print(f"  Field → Agent: {CrossDomainRegistry.has_transform('field', 'agent')}")
    print(f"  Agent → Field: {CrossDomainRegistry.has_transform('agent', 'field')}")
    print()

    # Run headless simulation
    print("Running simulation (headless)...")
    sim.run_headless(steps=100)

    print()
    print("✅ Cross-domain coupling example completed successfully!")
    print()
    print("To see animated visualization:")
    print("  Uncomment the line: sim.visualize(steps=200)")


def generate_field_agent_coupling(
    output_generator,
    seed: int = 42,
    duration_seconds: float = None
) -> Tuple[List[Visual], Optional[np.ndarray], Dict[str, Any]]:
    """
    Generate field-agent coupling visualization for showcase outputs.

    This function is compatible with the OutputGenerator framework (PR #78).

    Args:
        output_generator: OutputGenerator instance with preset configuration
        seed: Random seed for deterministic output
        duration_seconds: Duration in seconds (uses preset if None)

    Returns:
        Tuple of (frames, audio, metadata)
        - frames: List of Visual objects
        - audio: None (this example has no audio)
        - metadata: Dict with generation parameters
    """
    print("Generating cross-domain field-agent coupling...")

    if duration_seconds is None:
        duration_seconds = output_generator.preset['max_duration']

    # Get resolution from preset
    width, height = output_generator.preset['resolution']
    fps = output_generator.preset['fps']

    # Use smaller grid for higher resolutions (keeps computation manageable)
    grid_size = min(width, height, 512)

    # Create simulation with deterministic seed
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Agents: 500")
    print(f"  Duration: {duration_seconds}s @ {fps} fps")

    sim = FlowFieldAgentSimulation(grid_size=grid_size, num_agents=500, seed=seed)

    # Generate frames
    n_frames = int(duration_seconds * fps)
    frames = []

    print(f"  Simulating {n_frames} frames...")

    for frame_idx in range(n_frames):
        if frame_idx % (fps * 2) == 0:  # Progress every 2 seconds
            print(f"    Frame {frame_idx}/{n_frames} ({frame_idx/fps:.1f}s)")

        # Step simulation
        sim.step(dt=0.5)

        # Render frame
        vis = sim.render_frame(width=grid_size, height=grid_size)
        frames.append(vis)

    # Metadata
    metadata = {
        'example': 'cross_domain_field_agent_coupling',
        'description': 'Bidirectional Field ↔ Agent coupling demonstration',
        'cross_domain_operations': [
            'Field → Agent: Velocity sampling',
            'Agent → Field: Density deposition'
        ],
        'grid_size': grid_size,
        'num_agents': 500,
        'frames': len(frames),
        'fps': fps,
        'duration_seconds': len(frames) / fps,
        'resolution': [width, height],
        'seed': seed,
        'unique_features': [
            'Bidirectional cross-domain communication',
            'Real-time field-agent coupling',
            'Deterministic multi-domain simulation',
            'Impossible in traditional frameworks'
        ]
    }

    return frames, None, metadata


if __name__ == "__main__":
    main()

    # Uncomment to see animated visualization (requires display):
    # sim = FlowFieldAgentSimulation(grid_size=128, num_agents=500, seed=42)
    # sim.visualize(steps=200, interval=50)

    # To generate showcase outputs, use:
    # from examples.tools.generate_showcase_outputs import OutputGenerator
    # generator = OutputGenerator(preset='production')
    # frames, audio, metadata = generate_field_agent_coupling(generator, seed=42)
    # output_dir = generator.create_output_subdir('field_agent_coupling')
    # generator.export_frames(frames, output_dir, 'field_agent_coupling', formats=['png', 'mp4', 'gif'])
    # generator.save_metadata(output_dir, metadata)
