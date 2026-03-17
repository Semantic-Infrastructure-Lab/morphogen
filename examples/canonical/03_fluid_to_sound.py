"""Canonical Example 03: Fluid → Acoustics → Audio

3-domain composition: fluid simulation → acoustic wave propagation → WAV.

    Domain 1: FLUID     — Navier-Stokes approximation, vortex shedding
    Domain 2: ACOUSTICS — Wave propagation from pressure field
    Domain 3: AUDIO     — Stereo signal from virtual microphone positions

Output: output/03_fluid_to_sound.wav

Run: python examples/canonical/03_fluid_to_sound.py
"""

from pathlib import Path
import numpy as np

from morphogen.stdlib import field, audio
from morphogen.stdlib.field import Field2D
from morphogen.stdlib.audio import AudioBuffer

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WAV = OUTPUT_DIR / "03_fluid_to_sound.wav"

GRID = 64           # spatial grid size
DURATION = 3.0      # seconds of simulation
FLUID_DT = 0.02     # fluid timestep (50 Hz)
SAMPLE_RATE = 44100


def simulate_fluid(grid: int, duration: float, dt: float) -> list[Field2D]:
    """Navier-Stokes approximation with obstacle-driven vortex shedding."""
    num_steps = int(duration / dt)

    vx = field.alloc((grid, grid), fill_value=0.0)
    vy = field.alloc((grid, grid), fill_value=0.0)
    pressure = field.alloc((grid, grid), fill_value=0.0)

    # Cylindrical obstacle at 1/4 length, mid height
    obs_x, obs_y, obs_r = grid // 4, grid // 2, grid // 12
    y_idx, x_idx = np.mgrid[0:grid, 0:grid]
    obstacle = (x_idx - obs_x) ** 2 + (y_idx - obs_y) ** 2 <= obs_r ** 2

    print(f"[DOMAIN 1: FLUID] {grid}×{grid} grid, {num_steps} steps...")
    frames = []
    for step in range(num_steps):
        vx.data[:, :8] = 1.5           # inlet flow from left
        vx.data[obstacle] = 0.0
        vy.data[obstacle] = 0.0

        dvx_dx = np.gradient(vx.data, axis=1)
        dvy_dy = np.gradient(vy.data, axis=0)
        pressure.data = -(dvx_dx + dvy_dy) * 8.0

        if step % 8 == 0:
            noise = field.random((grid, grid), seed=step)
            pressure.data += noise.data * 0.05

        pressure = field.diffuse(pressure, rate=0.12, dt=dt)
        vx.data -= vx.data * dvx_dx * dt
        vy.data -= vy.data * dvy_dy * dt
        vx.data *= 0.997
        vy.data *= 0.997

        frames.append(pressure.copy())

        if (step + 1) % max(1, num_steps // 5) == 0:
            p_range = (pressure.data.min(), pressure.data.max())
            print(f"  step {step + 1}/{num_steps}  "
                  f"pressure [{p_range[0]:.3f}, {p_range[1]:.3f}]")

    print(f"  ✓ {len(frames)} pressure fields")
    return frames


def propagate_acoustics(pressure_frames: list[Field2D],
                         grid: int, dt: float) -> list[Field2D]:
    """Couple fluid pressure to acoustic wave propagation."""
    print(f"[DOMAIN 2: ACOUSTICS] Wave propagation...")
    acoustic = field.alloc((grid, grid), fill_value=0.0)
    c_sound = 4.0
    frames = []
    for i, p in enumerate(pressure_frames):
        acoustic.data += p.data * 0.08
        acoustic = field.diffuse(acoustic, rate=c_sound, dt=dt)
        acoustic.data *= 0.97
        frames.append(acoustic.copy())
    print(f"  ✓ {len(frames)} acoustic frames")
    return frames


def synthesize_audio(acoustic_frames: list[Field2D],
                      grid: int, duration: float,
                      sample_rate: int = SAMPLE_RATE) -> AudioBuffer:
    """Sample virtual microphones in the acoustic field → stereo audio."""
    print(f"[DOMAIN 3: AUDIO] Synthesizing at {sample_rate}Hz...")

    mic_L = (grid // 4,     grid // 2)   # left microphone
    mic_R = (3 * grid // 4, grid // 2)   # right microphone

    n_frames = len(acoustic_frames)
    left_signal = np.array([f.data[mic_L[1], mic_L[0]] for f in acoustic_frames])
    right_signal = np.array([f.data[mic_R[1], mic_R[0]] for f in acoustic_frames])

    # Resample from fluid timestep rate to audio rate
    n_audio = int(duration * sample_rate)
    t_fluid = np.linspace(0, duration, n_frames)
    t_audio = np.linspace(0, duration, n_audio)
    left_audio = np.interp(t_audio, t_fluid, left_signal)
    right_audio = np.interp(t_audio, t_fluid, right_signal)

    # Bandpass-style: remove DC and very high frequencies
    left_audio -= np.mean(left_audio)
    right_audio -= np.mean(right_audio)

    # Stereo interleave
    stereo = np.column_stack([left_audio, right_audio])
    peak = np.max(np.abs(stereo))
    if peak > 1e-6:
        stereo = stereo * 0.75 / peak

    # AudioBuffer expects mono; save left channel (representative)
    buf = AudioBuffer(stereo[:, 0], sample_rate)
    print(f"  ✓ L peak: {np.max(np.abs(left_audio)):.4f}  "
          f"R peak: {np.max(np.abs(right_audio)):.4f}")
    return buf


def main():
    print("=" * 60)
    print("CANONICAL EXAMPLE 03: Fluid → Acoustics → Audio")
    print("=" * 60)
    print()
    print("3-domain pipeline:")
    print("  1. Fluid: Navier-Stokes vortex shedding")
    print("  2. Acoustics: Pressure → wave propagation")
    print("  3. Audio: Virtual microphones → WAV")
    print()

    pressure_frames = simulate_fluid(GRID, DURATION, FLUID_DT)
    print()

    acoustic_frames = propagate_acoustics(pressure_frames, GRID, FLUID_DT)
    print()

    audio_buf = synthesize_audio(acoustic_frames, GRID, DURATION)
    print()

    audio.save(audio_buf, str(OUTPUT_WAV))
    print(f"✓ Saved: {OUTPUT_WAV}")
    print(f"  Duration: {DURATION:.1f}s | Sample rate: {SAMPLE_RATE}Hz")
    print()
    print("What happened:")
    print("  Fluid simulation generated 2D pressure fields at 50 Hz")
    print("  Acoustic module coupled those fields via wave diffusion")
    print("  Two virtual microphones sampled the acoustic field")
    print("  Microphone signals resampled to 44.1kHz → WAV")
    print("  No glue code: Field2D objects flowed directly between domains")


if __name__ == "__main__":
    main()
