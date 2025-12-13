"""Example: Audio I/O operations.

Demonstrates audio generation, saving, and loading using the I/O domain.
"""

import numpy as np
import sys
from pathlib import Path

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from morphogen.stdlib import io_storage as io


def generate_sine_wave(freq, duration, sample_rate=44100, amplitude=0.5):
    """Generate sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return amplitude * np.sin(2 * np.pi * freq * t)


def example_1_basic_tone_generation():
    """Generate and save basic tones."""
    print("=" * 60)
    print("Example 1: Basic tone generation")
    print("=" * 60)

    sample_rate = 44100
    duration = 2.0  # seconds

    # Generate 440 Hz tone (A4)
    tone_a4 = generate_sine_wave(440, duration, sample_rate)
    io.save_audio("tone_a4.wav", tone_a4, sample_rate)
    print(f"  ✓ Saved tone_a4.wav (440 Hz, {duration}s)")

    # Generate 880 Hz tone (A5)
    tone_a5 = generate_sine_wave(880, duration, sample_rate)
    io.save_audio("tone_a5.wav", tone_a5, sample_rate)
    print(f"  ✓ Saved tone_a5.wav (880 Hz, {duration}s)")

    # Load and verify
    loaded, sr = io.load_audio("tone_a4.wav")
    print(f"  Loaded tone_a4.wav: {len(loaded)} samples @ {sr} Hz")

    print()


def example_2_stereo_audio():
    """Generate and save stereo audio."""
    print("=" * 60)
    print("Example 2: Stereo audio")
    print("=" * 60)

    sample_rate = 44100
    duration = 2.0

    # Different frequencies in left and right channels
    left = generate_sine_wave(440, duration, sample_rate)  # A4 in left
    right = generate_sine_wave(554, duration, sample_rate)  # C#5 in right

    # Combine into stereo
    stereo = np.column_stack([left, right])

    io.save_audio("stereo_tone.wav", stereo, sample_rate)
    print(f"  ✓ Saved stereo_tone.wav (L=440 Hz, R=554 Hz)")
    print(f"    Shape: {stereo.shape}")

    # Load as stereo
    loaded_stereo, sr = io.load_audio("stereo_tone.wav")
    print(f"  Loaded stereo: {loaded_stereo.shape} @ {sr} Hz")

    # Load as mono (downmix)
    loaded_mono, sr = io.load_audio("stereo_tone.wav", mono=True)
    print(f"  Loaded mono: {loaded_mono.shape} @ {sr} Hz")

    print()


def example_3_chord_synthesis():
    """Synthesize and save a musical chord."""
    print("=" * 60)
    print("Example 3: Musical chord synthesis")
    print("=" * 60)

    sample_rate = 44100
    duration = 3.0

    # A major chord (A4, C#5, E5)
    a4 = generate_sine_wave(440.0, duration, sample_rate, amplitude=0.3)
    cs5 = generate_sine_wave(554.37, duration, sample_rate, amplitude=0.3)
    e5 = generate_sine_wave(659.25, duration, sample_rate, amplitude=0.3)

    # Mix the notes
    chord = a4 + cs5 + e5

    # Add simple ADSR envelope
    attack = 0.1  # 100ms attack
    release = 0.5  # 500ms release
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)

    envelope = np.ones_like(chord)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    chord *= envelope

    io.save_audio("chord_a_major.wav", chord, sample_rate)
    print(f"  ✓ Saved chord_a_major.wav (A major chord with envelope)")
    print(f"    Notes: A4 (440 Hz), C#5 (554 Hz), E5 (659 Hz)")

    # Also save as FLAC (lossless compression)
    io.save_audio("chord_a_major.flac", chord, sample_rate)
    print(f"  ✓ Saved chord_a_major.flac (FLAC compression)")

    print()


def example_4_audio_effects():
    """Apply simple audio effects."""
    print("=" * 60)
    print("Example 4: Simple audio effects")
    print("=" * 60)

    sample_rate = 44100
    duration = 2.0

    # Generate base tone
    base = generate_sine_wave(220, duration, sample_rate, amplitude=0.4)

    # Effect 1: Tremolo (amplitude modulation)
    lfo_rate = 5  # 5 Hz tremolo
    lfo = generate_sine_wave(lfo_rate, duration, sample_rate, amplitude=0.3)
    tremolo = base * (1.0 + lfo)
    io.save_audio("effect_tremolo.wav", tremolo, sample_rate)
    print(f"  ✓ Saved effect_tremolo.wav (5 Hz tremolo)")

    # Effect 2: Vibrato (frequency modulation, approximation)
    vibrato_rate = 6  # 6 Hz vibrato
    vibrato_depth = 10  # 10 Hz depth
    t = np.linspace(0, duration, len(base), dtype=np.float32)
    vibrato_mod = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
    vibrato = generate_sine_wave(220, duration, sample_rate, amplitude=0.4)
    # Simple approximation: phase modulation
    phase_mod = np.cumsum(vibrato_mod) / sample_rate
    vibrato = 0.4 * np.sin(2 * np.pi * 220 * t + phase_mod)
    io.save_audio("effect_vibrato.wav", vibrato, sample_rate)
    print(f"  ✓ Saved effect_vibrato.wav (6 Hz vibrato)")

    # Effect 3: Simple echo/delay
    delay_time = 0.3  # 300ms delay
    delay_samples = int(delay_time * sample_rate)
    decay = 0.5  # 50% feedback

    echo = np.copy(base)
    if delay_samples < len(echo):
        echo[delay_samples:] += base[:-delay_samples] * decay

    io.save_audio("effect_echo.wav", echo, sample_rate)
    print(f"  ✓ Saved effect_echo.wav (300ms echo)")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AUDIO I/O EXAMPLES")
    print("=" * 60)
    print()

    example_1_basic_tone_generation()
    example_2_stereo_audio()
    example_3_chord_synthesis()
    example_4_audio_effects()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - tone_a4.wav (440 Hz sine wave)")
    print("  - tone_a5.wav (880 Hz sine wave)")
    print("  - stereo_tone.wav (binaural tones)")
    print("  - chord_a_major.wav (A major chord)")
    print("  - chord_a_major.flac (FLAC compressed)")
    print("  - effect_tremolo.wav (amplitude modulation)")
    print("  - effect_vibrato.wav (frequency modulation)")
    print("  - effect_echo.wav (delay effect)")
    print()


if __name__ == "__main__":
    main()
