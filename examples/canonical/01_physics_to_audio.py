"""Canonical Example 01: Physics → Audio

Bouncing balls simulation sonified into a WAV file.

Each collision is detected and synthesized into a percussive impact
sound — pitch from mass, amplitude from velocity, decay from restitution.

Output: output/01_physics_to_audio.wav

Run: python examples/canonical/01_physics_to_audio.py
"""

from pathlib import Path
import numpy as np

from morphogen.stdlib.rigidbody import PhysicsWorld2D, create_circle_body, step_world
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WAV = OUTPUT_DIR / "01_physics_to_audio.wav"

SAMPLE_RATE = 44100
DURATION = 8.0       # seconds of simulation
FPS = 60             # physics update rate
NUM_BALLS = 5


def synthesize_impact(velocity: float, mass: float, restitution: float,
                      sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Map collision parameters to a decaying percussive tone.

    - mass      → fundamental frequency  (heavier = lower)
    - velocity  → amplitude              (faster = louder)
    - restitution → decay rate           (bouncier = longer ring)
    """
    n = int(0.4 * sample_rate)
    t = np.arange(n) / sample_rate

    # 1 kg ball → ~400 Hz; 3 kg → ~133 Hz
    fundamental = 400.0 / max(mass, 0.1)
    amplitude = np.clip(velocity / 8.0, 0.0, 1.0)
    decay = 5.0 + restitution * 15.0  # faster decay for low restitution

    # Two-partial tone with exponential decay
    wave = (
        np.sin(2 * np.pi * fundamental * t)
        + 0.4 * np.sin(2 * np.pi * fundamental * 2.5 * t)
    ) * np.exp(-decay * t)

    return wave * amplitude


def mix(events: list, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mix (time, samples) events into a stereo output buffer."""
    n_total = int(duration * sample_rate)
    out = np.zeros(n_total)
    for t_event, samples in events:
        start = int(t_event * sample_rate)
        end = min(start + len(samples), n_total)
        out[start:end] += samples[:end - start]
    # Normalise to ±0.8
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out = out * 0.8 / peak
    return out


def main():
    print("=" * 60)
    print("CANONICAL EXAMPLE 01: Physics → Audio")
    print("=" * 60)

    # --- Build world ---
    world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]), damping=0.99, dt=1.0 / FPS)

    # Ground: a large static circle at the bottom
    ground = create_circle_body(position=np.array([0.0, -4.0]), radius=4.0,
                                mass=0.0, restitution=0.7, friction=0.3)
    world.add_body(ground)

    # Balls with varied mass and bounciness
    rng = np.random.default_rng(42)
    for i in range(NUM_BALLS):
        mass = 0.5 + rng.random() * 2.5      # 0.5 – 3.0 kg
        rest = 0.5 + rng.random() * 0.4      # 0.5 – 0.9

        ball = create_circle_body(
            position=np.array([rng.uniform(-2.0, 2.0), 4.0 + i * 1.5]),
            radius=0.25 + rng.random() * 0.15,
            mass=mass,
            restitution=rest,
            friction=0.2,
        )
        ball.velocity = np.array([rng.uniform(-1.5, 1.5), 0.0])
        world.add_body(ball)

    print(f"Simulating {NUM_BALLS} balls for {DURATION}s at {FPS} fps...")

    # --- Simulate ---
    num_steps = int(DURATION * FPS)
    audio_events = []

    # Track previous velocities separately — step_world mutates bodies in-place
    prev_vy = {body.id: body.velocity[1] for body in world.bodies}

    for step in range(num_steps):
        t = step / FPS
        world = step_world(world)

        for body in world.bodies:
            if body.mass == 0.0:
                continue  # skip static ground
            cur_vy = body.velocity[1]
            p_vy = prev_vy[body.id]

            # Collision signature: y-velocity reversed AND body is near ground
            if np.sign(cur_vy) > 0 and np.sign(p_vy) < 0:
                impact_vel = abs(p_vy)
                if impact_vel > 0.5:  # ignore tiny taps
                    samples = synthesize_impact(impact_vel, body.mass, body.restitution)
                    audio_events.append((t, samples))

            prev_vy[body.id] = cur_vy

        if step % (FPS * 2) == 0:
            print(f"  t={t:.1f}s  collisions so far: {len(audio_events)}")

    print(f"\nTotal impacts detected: {len(audio_events)}")

    # --- Mix & save ---
    print("Mixing audio...")
    waveform = mix(audio_events, DURATION)
    buf = AudioBuffer(waveform, SAMPLE_RATE)
    audio.save(buf, str(OUTPUT_WAV))

    peak = np.max(np.abs(waveform))
    rms = np.sqrt(np.mean(waveform ** 2))
    print(f"\n✓ Saved: {OUTPUT_WAV}")
    print(f"  Duration: {DURATION:.1f}s | Peak: {peak:.3f} | RMS: {rms:.4f}")

    print("\nWhat happened:")
    print("  PhysicsWorld2D simulated bouncing balls under gravity")
    print("  Each ground collision → pitch (mass) + amplitude (velocity) + decay (restitution)")
    print("  Cross-domain: rigid-body mechanics directly drove audio synthesis parameters")


if __name__ == "__main__":
    main()
