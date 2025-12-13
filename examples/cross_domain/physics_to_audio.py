"""Physics to Audio - Cross-Domain Sonification

This example demonstrates cross-domain operator composition by converting
physics simulation events into audio synthesis. Specifically:

1. RigidBody → Audio: Collision events generate percussion sounds
2. Real-time parameter mapping: velocity → amplitude, material → timbre
3. Bidirectional coupling: Physics drives audio synthesis

Use case: Procedural sound design, game audio, physics sonification

This showcases Morphogen's unique ability to compose Physics and Audio domains
in real-time - creating emergent soundscapes from physical simulations.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphogen.stdlib import rigidbody, audio, visual, field, palette
from morphogen.stdlib.rigidbody import PhysicsWorld2D, create_circle_body, step_world


class CollisionEvent:
    """Represents a collision event in the physics simulation."""

    def __init__(self, time: float, position: np.ndarray, velocity: float,
                 mass: float, restitution: float):
        self.time = time
        self.position = position
        self.velocity = velocity  # Relative collision velocity
        self.mass = mass
        self.restitution = restitution  # Material property

    @property
    def impact_energy(self) -> float:
        """Compute impact energy (proportional to v^2)."""
        return 0.5 * self.mass * self.velocity ** 2

    def __repr__(self) -> str:
        return (f"CollisionEvent(t={self.time:.3f}s, "
                f"v={self.velocity:.2f}m/s, E={self.impact_energy:.2f}J)")


def detect_collisions(world: PhysicsWorld2D, prev_world: PhysicsWorld2D,
                       time: float) -> List[CollisionEvent]:
    """Detect collisions between current and previous physics state.

    Args:
        world: Current physics world state
        prev_world: Previous physics world state
        time: Current simulation time

    Returns:
        List of detected collision events
    """
    events = []

    # Simple collision detection: check for velocity sign changes
    # In a full implementation, would use world.collisions or similar
    for i, body in enumerate(world.bodies):
        if i >= len(prev_world.bodies):
            continue

        prev_body = prev_world.bodies[i]

        # Check for velocity reversals (collision signature)
        vel_y_changed = np.sign(body.velocity[1]) != np.sign(prev_body.velocity[1])

        # Check if velocity decreased significantly (energy loss in collision)
        vel_magnitude = np.linalg.norm(body.velocity)
        prev_vel_magnitude = np.linalg.norm(prev_body.velocity)
        velocity_dropped = vel_magnitude < prev_vel_magnitude * 0.8

        if vel_y_changed and velocity_dropped:
            # Collision detected
            relative_velocity = prev_vel_magnitude  # Velocity before collision

            event = CollisionEvent(
                time=time,
                position=body.position.copy(),
                velocity=relative_velocity,
                mass=body.mass,
                restitution=body.restitution
            )
            events.append(event)

    return events


def synthesize_impact_sound(event: CollisionEvent, duration: float = 0.5,
                             sample_rate: int = 44100) -> audio.AudioBuffer:
    """Synthesize percussion sound from collision event.

    Maps collision parameters to sound synthesis:
    - Velocity → Amplitude (louder impacts for faster collisions)
    - Mass → Pitch (heavier objects = lower pitch)
    - Restitution → Decay time (bouncier = longer decay)
    - Impact energy → Overtone richness

    Args:
        event: Collision event to sonify
        duration: Sound duration in seconds
        sample_rate: Audio sample rate

    Returns:
        AudioBuffer with synthesized impact sound
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Map mass to fundamental frequency (heavier = lower)
    # mass 0.5kg → 800Hz, mass 5kg → 200Hz
    mass_clamped = np.clip(event.mass, 0.1, 10.0)
    fundamental = 1000.0 / mass_clamped

    # Map velocity to amplitude (louder for faster impacts)
    # velocity 0-10 m/s → amplitude 0.0-1.0
    amplitude = np.clip(event.velocity / 10.0, 0.0, 1.0)

    # Map restitution to decay time (bouncier = longer)
    # restitution 0.0-1.0 → decay 5-20
    decay_rate = 5.0 + event.restitution * 15.0

    # Exponential decay envelope
    envelope = amplitude * np.exp(-decay_rate * t)

    # Synthesize sound with overtones (additive synthesis)
    sound = np.zeros_like(t)

    # Number of overtones based on impact energy
    num_overtones = int(3 + event.impact_energy * 2)
    num_overtones = min(num_overtones, 8)  # Cap at 8 overtones

    for n in range(1, num_overtones + 1):
        # Harmonic overtone
        freq = fundamental * n

        # Overtone amplitude falls off with frequency
        overtone_amp = 1.0 / n

        # Add phase variation for more natural sound
        phase = np.random.rand() * 2 * np.pi

        # Synthesize overtone
        sound += overtone_amp * np.sin(2 * np.pi * freq * t + phase)

    # Normalize and apply envelope
    if np.max(np.abs(sound)) > 0:
        sound = sound / np.max(np.abs(sound))

    sound = sound * envelope

    # Add a bit of noise for realism (impact transient)
    noise_amp = amplitude * 0.1
    noise = np.random.randn(num_samples) * noise_amp
    noise_envelope = np.exp(-50 * t)  # Quick decay
    sound += noise * noise_envelope

    return audio.AudioBuffer(data=sound.astype(np.float32), sample_rate=sample_rate)


def mix_audio_events(events: List[Tuple[float, audio.AudioBuffer]],
                      total_duration: float, sample_rate: int = 44100) -> audio.AudioBuffer:
    """Mix multiple audio events into a single buffer.

    Args:
        events: List of (time, audio_buffer) tuples
        total_duration: Total duration of output in seconds
        sample_rate: Audio sample rate

    Returns:
        Mixed audio buffer
    """
    total_samples = int(total_duration * sample_rate)
    mixed = np.zeros(total_samples, dtype=np.float32)

    for event_time, event_audio in events:
        # Calculate start sample
        start_sample = int(event_time * sample_rate)

        if start_sample >= total_samples:
            continue

        # Mix in the event audio
        event_data = event_audio.data
        end_sample = min(start_sample + len(event_data), total_samples)
        mix_length = end_sample - start_sample

        mixed[start_sample:end_sample] += event_data[:mix_length]

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed / peak * 0.95

    return audio.AudioBuffer(data=mixed, sample_rate=sample_rate)


def create_bouncing_balls_sonification(duration: float = 10.0,
                                        num_balls: int = 5,
                                        sample_rate: int = 44100,
                                        fps: int = 60) -> Tuple[audio.AudioBuffer, List]:
    """Create a sonified bouncing balls simulation.

    Args:
        duration: Simulation duration in seconds
        num_balls: Number of balls
        sample_rate: Audio sample rate
        fps: Physics simulation frame rate

    Returns:
        Tuple of (audio_buffer, visualization_frames)
    """
    print(f"Creating bouncing balls sonification...")
    print(f"  Duration: {duration}s")
    print(f"  Balls: {num_balls}")
    print(f"  Physics FPS: {fps}")

    # Create physics world
    world = PhysicsWorld2D(
        gravity=np.array([0.0, -9.81]),
        damping=0.99,
        dt=1.0 / fps
    )

    # Add ground (heavy static body)
    ground = create_circle_body(
        position=np.array([0.0, -5.0]),
        radius=5.0,
        mass=0.0,  # Static
        restitution=0.7,
        friction=0.3
    )
    world.add_body(ground)

    # Add balls with varying properties
    np.random.seed(42)
    for i in range(num_balls):
        # Vary mass, restitution for different sounds
        mass = 0.5 + np.random.rand() * 2.0  # 0.5 - 2.5 kg
        restitution = 0.5 + np.random.rand() * 0.4  # 0.5 - 0.9

        ball = create_circle_body(
            position=np.array([
                -3.0 + np.random.rand() * 6.0,  # x: -3 to 3
                5.0 + i * 1.0  # Stagger height
            ]),
            radius=0.3 + np.random.rand() * 0.2,
            mass=mass,
            restitution=restitution,
            friction=0.2
        )

        # Initial velocity
        ball.velocity = np.array([
            -2.0 + np.random.rand() * 4.0,  # x velocity
            0.0  # y velocity
        ])

        world.add_body(ball)

    # Simulate physics and detect collisions
    num_steps = int(duration * fps)
    collision_events = []

    prev_world = world
    for step in range(num_steps):
        current_time = step / fps

        # Step physics
        world = step_world(world)

        # Detect collisions
        events = detect_collisions(world, prev_world, current_time)
        collision_events.extend(events)

        if step % (fps * 2) == 0:
            print(f"  Physics step {step}/{num_steps} ({current_time:.1f}s) - "
                  f"{len(collision_events)} collisions so far")

        prev_world = world

    print(f"  Total collisions detected: {len(collision_events)}")

    # Synthesize audio for each collision
    print(f"  Synthesizing collision sounds...")
    audio_events = []

    for i, event in enumerate(collision_events):
        # Only synthesize significant collisions
        if event.velocity > 0.5:  # Minimum velocity threshold
            impact_sound = synthesize_impact_sound(
                event,
                duration=0.5,
                sample_rate=sample_rate
            )
            audio_events.append((event.time, impact_sound))

        if (i + 1) % 20 == 0:
            print(f"    Synthesized {i + 1}/{len(collision_events)} sounds")

    print(f"  Mixing {len(audio_events)} audio events...")
    mixed_audio = mix_audio_events(audio_events, duration, sample_rate)

    print(f"  ✓ Sonification complete!")
    print(f"    Peak amplitude: {np.max(np.abs(mixed_audio.data)):.3f}")
    print(f"    RMS level: {np.sqrt(np.mean(mixed_audio.data**2)):.3f}")

    # Create simple visualization frames (optional)
    vis_frames = []

    return mixed_audio, vis_frames


def demo_simple_bounce():
    """Demo: Single ball bounce sonification."""
    print("=" * 60)
    print("DEMO 1: Simple Bounce Sonification")
    print("=" * 60)
    print()

    # Create simple simulation
    duration = 5.0
    audio_output, _ = create_bouncing_balls_sonification(
        duration=duration,
        num_balls=1,
        sample_rate=44100,
        fps=60
    )

    # Save audio
    output_path = "output_physics_single_bounce.wav"
    audio.save(audio_output, output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Duration: {audio_output.duration:.2f}s")


def demo_multi_ball_chaos():
    """Demo: Multiple balls creating chaotic soundscape."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Ball Chaotic Sonification")
    print("=" * 60)
    print()

    # Create complex simulation
    duration = 10.0
    audio_output, _ = create_bouncing_balls_sonification(
        duration=duration,
        num_balls=8,
        sample_rate=44100,
        fps=60
    )

    # Save audio
    output_path = "output_physics_multi_bounce.wav"
    audio.save(audio_output, output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Duration: {audio_output.duration:.2f}s")


def main():
    """Run physics to audio demonstrations."""
    print("=" * 60)
    print("PHYSICS TO AUDIO - CROSS-DOMAIN SONIFICATION")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  • RigidBody physics simulation")
    print("  • Collision event detection")
    print("  • Cross-domain mapping: Physics → Audio")
    print("  • Procedural percussion synthesis")
    print("  • Real-time parameter mapping")
    print()
    print("Cross-Domain Mappings:")
    print("  • Collision velocity → Sound amplitude")
    print("  • Object mass → Pitch (heavier = lower)")
    print("  • Material restitution → Decay time")
    print("  • Impact energy → Overtone richness")
    print()

    # Run demos
    demo_simple_bounce()
    demo_multi_ball_chaos()

    print("\n" + "=" * 60)
    print("ALL PHYSICS SONIFICATION DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("Key insight: Physical events can be directly mapped to")
    print("audio parameters, creating emergent soundscapes that")
    print("perfectly synchronize with simulation dynamics!")
    print()
    print("This is IMPOSSIBLE in traditional audio engines - Morphogen's")
    print("cross-domain composition enables true physics-driven sound!")


if __name__ == "__main__":
    main()
