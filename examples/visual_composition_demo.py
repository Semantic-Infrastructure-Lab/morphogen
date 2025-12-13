"""Morphogen v0.6.0 Visual Composition Demo

Demonstrates agent visualization, layer composition, and video export.

Requirements:
    pip install morphogen[io]  # Installs imageio, imageio-ffmpeg
"""

import sys
sys.path.insert(0, '/home/user/morphogen')

import numpy as np
from morphogen.stdlib.field import field
from morphogen.stdlib.agents import Agents
from morphogen.stdlib.visual import visual


def demo_agent_visualization():
    """Demo: Visualizing particle systems."""
    print("\n=== Agent Visualization Demo ===")

    # Create random particle system
    n_particles = 100
    positions = np.random.rand(n_particles, 2)
    velocities = (np.random.rand(n_particles, 2) - 0.5) * 0.1

    particles = Agents(
        count=n_particles,
        properties={
            'pos': positions,
            'vel': velocities
        }
    )

    # Basic visualization
    vis1 = visual.agents(particles, width=256, height=256, size=3.0, color=(1, 1, 1))
    visual.output(vis1, "/tmp/morphogen_particles_basic.png")
    print("✅ Basic particle visualization: /tmp/morphogen_particles_basic.png")

    # Color by velocity magnitude
    vis2 = visual.agents(
        particles,
        width=256,
        height=256,
        color_property='vel',
        palette='viridis',
        size=4.0
    )
    visual.output(vis2, "/tmp/morphogen_particles_colored.png")
    print("✅ Velocity-colored particles: /tmp/morphogen_particles_colored.png")

    # Variable size by velocity
    vis3 = visual.agents(
        particles,
        width=256,
        height=256,
        color_property='vel',
        size_property='vel',
        palette='fire',
        size=5.0
    )
    visual.output(vis3, "/tmp/morphogen_particles_sized.png")
    print("✅ Variable-size particles: /tmp/morphogen_particles_sized.png")


def demo_layer_composition():
    """Demo: Compositing multiple visual layers."""
    print("\n=== Layer Composition Demo ===")

    # Create background field
    background_field = field.random((128, 128), seed=42)
    background = visual.colorize(background_field, palette="viridis")
    print("  Created background field layer")

    # Create agent layer 1 (red particles)
    pos1 = np.random.rand(30, 2)
    agents1 = Agents(count=30, properties={'pos': pos1})
    layer1 = visual.agents(agents1, width=128, height=128, color=(1, 0, 0), size=3.0)
    print("  Created red agent layer")

    # Create agent layer 2 (cyan particles)
    pos2 = np.random.rand(20, 2)
    agents2 = Agents(count=20, properties={'pos': pos2})
    layer2 = visual.agents(agents2, width=128, height=128, color=(0, 1, 1), size=2.0)
    print("  Created cyan agent layer")

    # Composite with additive blending
    result = visual.composite(background, layer1, layer2, mode="add")
    visual.output(result, "/tmp/morphogen_composite_add.png")
    print("✅ Additive composition: /tmp/morphogen_composite_add.png")

    # Composite with different opacities
    result2 = visual.composite(
        background, layer1, layer2,
        mode="over",
        opacity=[1.0, 0.8, 0.6]
    )
    visual.output(result2, "/tmp/morphogen_composite_opacity.png")
    print("✅ Opacity composition: /tmp/morphogen_composite_opacity.png")

    # Multiply blending for darker effect
    result3 = visual.composite(background, layer1, mode="multiply")
    visual.output(result3, "/tmp/morphogen_composite_multiply.png")
    print("✅ Multiply composition: /tmp/morphogen_composite_multiply.png")


def demo_video_export_gif():
    """Demo: Export animation as GIF."""
    print("\n=== GIF Video Export Demo ===")

    try:
        import imageio
    except ImportError:
        print("⚠️  imageio not installed, skipping video demos")
        return

    # Create animated field evolution
    print("  Generating 30 frames...")
    temp = field.random((64, 64), seed=42)

    frames = []
    for i in range(30):
        # Evolve field
        temp = field.diffuse(temp, rate=0.1, dt=0.1)

        # Visualize
        vis = visual.colorize(temp, palette="fire")
        frames.append(vis)

    # Export as GIF
    output_path = "/tmp/morphogen_diffusion.gif"
    visual.video(frames, output_path, fps=10)
    print(f"✅ GIF animation: {output_path}")
    print(f"   {len(frames)} frames @ 10 FPS")


def demo_video_export_mp4():
    """Demo: Export animation as MP4."""
    print("\n=== MP4 Video Export Demo ===")

    try:
        import imageio
    except ImportError:
        print("⚠️  imageio not installed, skipping video demos")
        return

    # Create particle animation
    print("  Simulating 60 frames of particles...")
    n_particles = 50
    positions = np.random.rand(n_particles, 2)
    velocities = (np.random.rand(n_particles, 2) - 0.5) * 0.02

    frames = []
    for step in range(60):
        # Update physics
        positions = positions + velocities

        # Wrap boundaries
        positions = positions % 1.0

        # Create agents
        particles = Agents(
            count=n_particles,
            properties={'pos': positions, 'vel': velocities}
        )

        # Render with velocity colors
        frame = visual.agents(
            particles,
            width=256,
            height=256,
            color_property='vel',
            palette='coolwarm',
            size=4.0
        )
        frames.append(frame)

    # Export as MP4
    output_path = "/tmp/morphogen_particles.mp4"
    visual.video(frames, output_path, fps=30)
    print(f"✅ MP4 animation: {output_path}")
    print(f"   {len(frames)} frames @ 30 FPS")


def demo_complex_composition():
    """Demo: Complex multi-layer animated composition."""
    print("\n=== Complex Composition Demo ===")

    try:
        import imageio
    except ImportError:
        print("⚠️  imageio not installed, skipping video demos")
        return

    print("  Creating 40-frame composition with field + 2 agent layers...")

    # Initialize simulation
    temp = field.random((128, 128), seed=42)

    # Agent populations
    n_pop1, n_pop2 = 30, 20
    pos1 = np.random.rand(n_pop1, 2)
    pos2 = np.random.rand(n_pop2, 2)
    vel1 = (np.random.rand(n_pop1, 2) - 0.5) * 0.015
    vel2 = (np.random.rand(n_pop2, 2) - 0.5) * 0.02

    frames = []
    for step in range(40):
        # Evolve field
        temp = field.diffuse(temp, rate=0.08, dt=0.1)

        # Update agents
        pos1 = (pos1 + vel1) % 1.0
        pos2 = (pos2 + vel2) % 1.0

        # Create layers
        bg_layer = visual.colorize(temp, palette="viridis")

        agents1 = Agents(count=n_pop1, properties={'pos': pos1})
        layer1 = visual.agents(agents1, width=128, height=128, color=(1, 0.5, 0), size=3)

        agents2 = Agents(count=n_pop2, properties={'pos': pos2})
        layer2 = visual.agents(agents2, width=128, height=128, color=(0, 1, 1), size=2)

        # Composite all layers
        composite = visual.composite(
            bg_layer, layer1, layer2,
            mode="add",
            opacity=[1.0, 0.7, 0.5]
        )

        frames.append(composite)

    # Export
    output_path = "/tmp/morphogen_complex.mp4"
    visual.video(frames, output_path, fps=20)
    print(f"✅ Complex composition: {output_path}")
    print(f"   Field + 2 agent layers, {len(frames)} frames @ 20 FPS")


def demo_generator_export():
    """Demo: Memory-efficient video export using generators."""
    print("\n=== Generator-Based Export Demo ===")

    try:
        import imageio
    except ImportError:
        print("⚠️  imageio not installed, skipping video demos")
        return

    print("  Using generator for memory-efficient rendering...")

    # Generator function
    temp = field.random((64, 64), seed=123)

    def frame_generator():
        nonlocal temp
        for i in range(50):
            temp = field.diffuse(temp, rate=0.1, dt=0.1)

            # Add some noise to keep it interesting
            noise = field.random((64, 64), seed=i)
            from morphogen.stdlib.field import Field2D
            temp_data = temp.data + noise.data * 0.05
            temp = Field2D(data=temp_data)

            vis = visual.colorize(temp, palette="coolwarm")
            yield vis

    # Export using generator (memory efficient!)
    output_path = "/tmp/morphogen_generator.gif"
    visual.video(frame_generator, output_path, fps=10, max_frames=50)
    print(f"✅ Generator export: {output_path}")
    print("   Memory-efficient: frames generated on-the-fly")


def demo_blending_modes():
    """Demo: Different blending modes."""
    print("\n=== Blending Modes Demo ===")

    # Create two contrasting layers
    layer1 = visual.layer(width=128, height=128, background=(0.8, 0.2, 0.2))  # Red
    layer2 = visual.layer(width=128, height=128, background=(0.2, 0.2, 0.8))  # Blue

    modes = ["over", "add", "multiply", "screen", "overlay"]

    for mode in modes:
        result = visual.composite(layer1, layer2, mode=mode, opacity=[1.0, 0.7])
        output_path = f"/tmp/morphogen_blend_{mode}.png"
        visual.output(result, output_path)
        print(f"  ✅ {mode:10s} mode: {output_path}")


def main():
    """Run all visual composition demos."""
    print("=" * 60)
    print("Morphogen v0.6.0 - Visual Composition Demonstrations")
    print("=" * 60)

    # Run demos
    demo_agent_visualization()
    demo_layer_composition()
    demo_blending_modes()
    demo_video_export_gif()
    demo_video_export_mp4()
    demo_complex_composition()
    demo_generator_export()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("\nGenerated files:")
    print("\nStatic Images:")
    print("  /tmp/morphogen_particles_*.png   - Agent visualizations")
    print("  /tmp/morphogen_composite_*.png   - Layer compositions")
    print("  /tmp/morphogen_blend_*.png       - Blending modes")
    print("\nAnimations:")
    print("  /tmp/morphogen_diffusion.gif     - Field diffusion (GIF)")
    print("  /tmp/morphogen_particles.mp4     - Particle system (MP4)")
    print("  /tmp/morphogen_complex.mp4       - Multi-layer composition")
    print("  /tmp/morphogen_generator.gif     - Generator-based export")
    print("=" * 60)


if __name__ == "__main__":
    main()
