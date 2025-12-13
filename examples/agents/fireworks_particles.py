"""Fireworks particle effects demonstration with audio synchronization.

Showcases particle emission, lifetime management, trails, and visual effects,
enhanced with cross-domain physics → audio mapping. Each firework burst
generates synchronized percussion sounds based on particle physics.

This demonstrates Morphogen's ability to compose Visual + Audio domains with
shared timing - particles driving both visual effects and sound synthesis.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from morphogen.stdlib.agents import agents, particle_behaviors
from morphogen.stdlib.visual import visual, Visual
from morphogen.stdlib import audio


def create_firework_burst(center, color_base, count=200, seed=None):
    """Create a single firework burst.

    Args:
        center: (x, y) position for burst center
        color_base: Base color for particles
        count: Number of particles
        seed: Random seed for deterministic emission

    Returns:
        Particle system for the burst
    """
    # Emit particles in sphere pattern
    particles = agents.emit(
        count=count,
        position=center,
        emission_shape="sphere",
        emission_radius=50.0,
        velocity=lambda n: np.random.randn(n, 2) * 3.0,  # Random velocities
        lifetime=(30.0, 60.0),  # Random lifetimes
        properties={
            'color_offset': np.random.rand(count) * 0.3,  # Color variation
            'size': np.random.uniform(1.0, 3.0, count)
        },
        seed=seed
    )

    return particles


def synthesize_firework_percussion(
    particle_count: int,
    position: Tuple[float, float],
    canvas_width: int,
    duration: float = 0.3,
    sample_rate: int = 44100,
    seed: int = 0
) -> audio.AudioBuffer:
    """
    Synthesize percussion sound for a firework burst.

    Maps particle physics to audio parameters:
    - particle_count → impact energy (amplitude)
    - position.x → stereo panning
    - Random variation → pitch/timbre

    Args:
        particle_count: Number of particles in burst
        position: (x, y) position on canvas
        canvas_width: Width of canvas for panning calculation
        duration: Sound duration in seconds
        sample_rate: Audio sample rate
        seed: Random seed for pitch variation

    Returns:
        AudioBuffer with percussion sound
    """
    rng = np.random.RandomState(seed)

    # Map particle count to amplitude (more particles = louder)
    amplitude = np.clip(particle_count / 200.0, 0.3, 1.0)

    # Map x position to stereo panning (-1 = left, 1 = right)
    pan = (position[0] / canvas_width) * 2.0 - 1.0
    pan = np.clip(pan, -1.0, 1.0)

    # Random pitch variation (60-120 Hz for impact)
    base_freq = rng.uniform(60, 120)

    # Create impact sound: low frequency sine + noise burst
    # 1. Low frequency "boom"
    boom = audio.sine(freq=base_freq, duration=duration, sample_rate=sample_rate)

    # Apply exponential decay envelope
    num_samples = boom.num_samples
    t = np.arange(num_samples) / sample_rate
    decay_envelope = np.exp(-t / 0.08)  # Fast decay

    boom.data = boom.data * decay_envelope * amplitude * 0.7

    # 2. High frequency "crackle" (noise burst)
    crackle = audio.noise(noise_type="white", seed=seed, duration=duration * 0.5,
                         sample_rate=sample_rate)

    # Apply faster decay to crackle
    crackle_samples = crackle.num_samples
    t_crackle = np.arange(crackle_samples) / sample_rate
    crackle_envelope = np.exp(-t_crackle / 0.02)

    crackle.data = crackle.data * crackle_envelope * amplitude * 0.3

    # Pad crackle to match boom length
    crackle_padded = np.zeros(num_samples, dtype=np.float32)
    crackle_padded[:crackle_samples] = crackle.data

    # Mix boom + crackle
    mixed = boom.data + crackle_padded

    # Convert to stereo with panning
    left_gain = np.sqrt((1.0 - pan) / 2.0)   # Equal power panning
    right_gain = np.sqrt((1.0 + pan) / 2.0)

    stereo = np.stack([mixed * left_gain, mixed * right_gain], axis=-1)

    return audio.AudioBuffer(data=stereo, sample_rate=sample_rate)


def fireworks_demo():
    """Run fireworks particle effects demo."""

    print("Fireworks Particle Effects Demo")
    print("=" * 50)

    # Simulation parameters
    width, height = 512, 512
    dt = 1.0

    # Create initial empty particle system
    all_particles = agents.alloc(count=0, properties={
        'pos': np.empty((0, 2)),
        'vel': np.empty((0, 2)),
        'age': np.empty(0),
        'lifetime': np.empty(0)
    })

    # Firework launch timing
    firework_timer = 0
    firework_interval = 20

    # Colors for different fireworks
    firework_colors = [
        (1.0, 0.2, 0.2),  # Red
        (0.2, 1.0, 0.2),  # Green
        (0.2, 0.2, 1.0),  # Blue
        (1.0, 1.0, 0.2),  # Yellow
        (1.0, 0.2, 1.0),  # Magenta
        (0.2, 1.0, 1.0),  # Cyan
    ]

    def generate_frame():
        nonlocal all_particles, firework_timer

        frame_count = 0
        max_frames = 300

        while frame_count < max_frames:
            # Launch new fireworks periodically
            if firework_timer <= 0:
                # Random position in upper half
                launch_pos = np.array([
                    np.random.uniform(100, width - 100),
                    np.random.uniform(height * 0.3, height * 0.7)
                ])

                # Random color
                color = firework_colors[np.random.randint(len(firework_colors))]

                # Create burst
                new_burst = create_firework_burst(launch_pos, color, count=150)

                # Merge with existing particles
                if all_particles.alive_count > 0:
                    all_particles = agents.merge([all_particles, new_burst])
                else:
                    all_particles = new_burst

                firework_timer = firework_interval + np.random.randint(-5, 5)

            firework_timer -= 1

            # Apply gravity
            if all_particles.alive_count > 0:
                all_particles = agents.apply_force(
                    all_particles,
                    force=np.array([0.0, -2.0]),  # Gravity
                    dt=dt
                )

                # Apply air resistance
                all_particles = agents.apply_force(
                    all_particles,
                    force=particle_behaviors.drag(coefficient=0.02),
                    dt=dt
                )

                # Update positions
                all_particles = agents.integrate(all_particles, dt=dt)

                # Age particles and remove dead ones
                all_particles = agents.age_particles(all_particles, dt=dt)

                # Update trails
                if frame_count % 2 == 0:  # Update trails every other frame
                    all_particles = agents.update_trail(all_particles, trail_length=15)

            # Render
            if all_particles.alive_count > 0:
                # Calculate alpha based on age
                alphas = agents.get_particle_alpha(
                    all_particles,
                    fade_in=0.1,
                    fade_out=0.3
                )

                # Create temporary alpha property for rendering
                all_particles.properties['alpha'] = np.zeros(all_particles.count, dtype=np.float32)
                all_particles.properties['alpha'][all_particles.alive_mask] = alphas

                # Render with trails and additive blending
                vis = visual.agents(
                    all_particles,
                    width=width,
                    height=height,
                    alpha_property='alpha',
                    size_property='size',
                    size=2.0,
                    background=(0.0, 0.0, 0.05),  # Dark blue background
                    blend_mode='additive',
                    trail=True,
                    trail_length=15,
                    trail_alpha=0.4
                )
            else:
                # Empty frame
                vis = visual.layer(width=width, height=height, background=(0.0, 0.0, 0.05))

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Frame {frame_count}, Particles alive: {all_particles.alive_count}")

            yield vis

    # Create generator
    gen = generate_frame()

    # Export to video
    print("\nExporting fireworks animation...")
    visual.video(
        lambda: next(gen),
        path="/tmp/fireworks_particles.mp4",
        fps=30,
        max_frames=300
    )

    print(f"\nFireworks animation saved to /tmp/fireworks_particles.mp4")
    print(f"Total frames: 300")


def generate_fireworks_with_audio(
    output_generator,
    seed: int = 42,
    duration_seconds: float = None
) -> Tuple[List[Visual], Optional[np.ndarray], Dict[str, Any]]:
    """
    Generate fireworks visualization with synchronized audio.

    Compatible with OutputGenerator framework (PR #78).

    Physics → Audio mapping:
    - Burst events → percussion sounds
    - Particle count → impact amplitude
    - Position X → stereo panning
    - Random variation → pitch/timbre

    Args:
        output_generator: OutputGenerator instance
        seed: Random seed for deterministic output
        duration_seconds: Duration (uses preset if None)

    Returns:
        Tuple of (frames, audio, metadata)
    """
    print("Generating fireworks with synchronized audio...")

    if duration_seconds is None:
        duration_seconds = min(20, output_generator.preset['max_duration'])

    width, height = output_generator.preset['resolution']
    fps = output_generator.preset['fps']
    sample_rate = output_generator.preset['audio_sr']

    # Simulation parameters
    dt = 1.0
    n_frames = int(duration_seconds * fps)

    # Set random seed for deterministic behavior
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Firework colors
    firework_colors = [
        (1.0, 0.2, 0.2),  # Red
        (0.2, 1.0, 0.2),  # Green
        (0.2, 0.2, 1.0),  # Blue
        (1.0, 1.0, 0.2),  # Yellow
        (1.0, 0.2, 1.0),  # Magenta
        (0.2, 1.0, 1.0),  # Cyan
    ]

    # Track burst events for audio synthesis
    burst_events = []  # List of (frame_idx, position, particle_count)

    # Initialize particle system
    all_particles = agents.alloc(count=0, properties={
        'pos': np.empty((0, 2)),
        'vel': np.empty((0, 2)),
        'age': np.empty(0),
        'lifetime': np.empty(0)
    })

    firework_timer = 0
    firework_interval = int(fps * 1.5)  # Launch every 1.5 seconds
    burst_counter = 0

    frames = []

    print(f"  Duration: {duration_seconds}s @ {fps} fps")
    print(f"  Generating {n_frames} frames...")

    for frame_idx in range(n_frames):
        if frame_idx % (fps * 2) == 0:
            print(f"    Frame {frame_idx}/{n_frames} ({frame_idx/fps:.1f}s)")

        # Launch new fireworks periodically
        if firework_timer <= 0:
            # Random position
            launch_pos = np.array([
                rng.uniform(width * 0.2, width * 0.8),
                rng.uniform(height * 0.3, height * 0.7)
            ])

            # Random color
            color = firework_colors[rng.randint(len(firework_colors))]

            # Particle count
            particle_count = rng.randint(100, 200)

            # Create burst (with deterministic seed)
            new_burst = create_firework_burst(
                launch_pos, color, count=particle_count,
                seed=seed + burst_counter
            )

            # Record burst event for audio
            burst_events.append((frame_idx, launch_pos, particle_count, seed + burst_counter))
            burst_counter += 1

            # Merge with existing particles
            if all_particles.alive_count > 0:
                all_particles = agents.merge([all_particles, new_burst])
            else:
                all_particles = new_burst

            firework_timer = firework_interval + rng.randint(-fps//2, fps//2)

        firework_timer -= 1

        # Update particles
        if all_particles.alive_count > 0:
            # Apply gravity
            all_particles = agents.apply_force(
                all_particles,
                force=np.array([0.0, -2.0]),
                dt=dt
            )

            # Apply drag
            all_particles = agents.apply_force(
                all_particles,
                force=particle_behaviors.drag(coefficient=0.02),
                dt=dt
            )

            # Integrate
            all_particles = agents.integrate(all_particles, dt=dt)

            # Age particles
            all_particles = agents.age_particles(all_particles, dt=dt)

            # Update trails
            if frame_idx % 2 == 0:
                all_particles = agents.update_trail(all_particles, trail_length=15)

        # Render frame
        if all_particles.alive_count > 0:
            alphas = agents.get_particle_alpha(all_particles, fade_in=0.1, fade_out=0.3)

            all_particles.properties['alpha'] = np.zeros(all_particles.count, dtype=np.float32)
            all_particles.properties['alpha'][all_particles.alive_mask] = alphas

            vis = visual.agents(
                all_particles,
                width=width,
                height=height,
                alpha_property='alpha',
                size_property='size',
                size=2.0,
                background=(0.0, 0.0, 0.05),
                blend_mode='additive',
                trail=True,
                trail_length=15,
                trail_alpha=0.4
            )
        else:
            vis = visual.layer(width=width, height=height, background=(0.0, 0.0, 0.05))

        frames.append(vis)

    # Generate synchronized audio
    print(f"\n  Synthesizing audio for {len(burst_events)} bursts...")

    total_samples = int(duration_seconds * sample_rate)
    audio_stereo = np.zeros((total_samples, 2), dtype=np.float32)

    for burst_idx, (frame_idx, position, particle_count, burst_seed) in enumerate(burst_events):
        # Calculate sample offset for this burst
        time_offset = frame_idx / fps
        sample_offset = int(time_offset * sample_rate)

        # Synthesize percussion sound
        percussion = synthesize_firework_percussion(
            particle_count=particle_count,
            position=position,
            canvas_width=width,
            duration=0.3,
            sample_rate=sample_rate,
            seed=burst_seed
        )

        # Mix into audio track
        end_sample = min(sample_offset + percussion.num_samples, total_samples)
        duration_to_add = end_sample - sample_offset

        if duration_to_add > 0:
            audio_stereo[sample_offset:end_sample, :] += percussion.data[:duration_to_add, :]

    # Normalize audio to prevent clipping
    max_amplitude = np.max(np.abs(audio_stereo))
    if max_amplitude > 0.95:
        audio_stereo = audio_stereo / max_amplitude * 0.95

    # Metadata
    metadata = {
        'example': 'fireworks_with_audio',
        'description': 'Fireworks particle effects with physics → audio synchronization',
        'cross_domain_operations': [
            'Particle physics → Visual rendering',
            'Burst events → Percussion synthesis',
            'Position → Stereo panning',
            'Particle count → Impact amplitude'
        ],
        'num_bursts': len(burst_events),
        'frames': len(frames),
        'fps': fps,
        'duration_seconds': duration_seconds,
        'resolution': [width, height],
        'audio_sample_rate': sample_rate,
        'seed': seed,
        'unique_features': [
            'Synchronized audio-visual output',
            'Physics-driven sound synthesis',
            'Spatial audio (stereo panning)',
            'Deterministic multi-domain generation'
        ]
    }

    return frames, audio_stereo, metadata


if __name__ == "__main__":
    fireworks_demo()

    # To generate showcase outputs with audio:
    # from examples.tools.generate_showcase_outputs import OutputGenerator
    # generator = OutputGenerator(preset='production')
    # frames, audio_data, metadata = generate_fireworks_with_audio(generator, seed=42)
    # output_dir = generator.create_output_subdir('fireworks_audio')
    # generator.export_frames(frames, output_dir, 'fireworks', formats=['png', 'mp4', 'gif'])
    # generator.export_audio(audio_data, output_dir, 'fireworks_audio', formats=['wav'])
    # generator.save_metadata(output_dir, metadata)
