"""Morphogen v0.6.0 Audio I/O Demo

Demonstrates real-time audio playback, file export, and recording capabilities.

Requirements:
    pip install morphogen[io]  # Installs sounddevice, soundfile, scipy, imageio
"""

import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.audio import audio
import numpy as np


def demo_playback():
    """Demo: Real-time audio playback."""
    print("\n=== Audio Playback Demo ===")
    print("Generating A440 tone...")

    # Generate 1 second of A440 (concert pitch A)
    tone = audio.sine(freq=440.0, duration=1.0)

    print(f"Playing {tone.duration:.2f}s of audio at {tone.sample_rate}Hz...")
    print("(Note: Playback requires audio device)")

    try:
        audio.play(tone, blocking=True)
        print("✅ Playback complete!")
    except Exception as e:
        print(f"⚠️  Playback failed (no audio device?): {e}")


def demo_wav_export():
    """Demo: WAV file export."""
    print("\n=== WAV Export Demo ===")

    # Create a musical chord (C major: C-E-G)
    c = audio.sine(freq=261.63, duration=2.0)  # C4
    e = audio.sine(freq=329.63, duration=2.0)  # E4
    g = audio.sine(freq=392.00, duration=2.0)  # G4

    # Mix with envelope
    chord = audio.mix(c, e, g)
    envelope = audio.adsr(attack=0.1, decay=0.2, sustain=0.7, release=0.5, duration=2.0)

    # Apply envelope (element-wise multiplication)
    from morphogen.stdlib.audio import AudioBuffer
    enveloped = AudioBuffer(
        data=chord.data * envelope.data,
        sample_rate=chord.sample_rate
    )

    # Add reverb
    final = audio.reverb(enveloped, mix=0.2, size=0.8)

    # Export to WAV
    output_path = "/tmp/morphogen_chord.wav"
    audio.save(final, output_path)
    print(f"✅ Saved to: {output_path}")

    # Load and verify
    loaded = audio.load(output_path)
    print(f"   Loaded: {loaded.num_samples} samples, {loaded.duration:.2f}s")

    # Verify round-trip accuracy
    correlation = np.corrcoef(final.data, loaded.data)[0, 1]
    print(f"   Round-trip correlation: {correlation:.4f} (should be > 0.99)")


def demo_flac_export():
    """Demo: FLAC file export (lossless)."""
    print("\n=== FLAC Export Demo ===")

    try:
        import soundfile
    except ImportError:
        print("⚠️  soundfile not installed, skipping FLAC demo")
        return

    # Generate complex audio
    melody = []
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00]  # C-D-E-F-G

    for freq in freqs:
        note = audio.sine(freq=freq, duration=0.3)
        envelope = audio.ar(attack=0.01, release=0.2, duration=0.3)

        from morphogen.stdlib.audio import AudioBuffer
        enveloped = AudioBuffer(
            data=note.data * envelope.data,
            sample_rate=note.sample_rate
        )
        melody.append(enveloped)

    # Concatenate notes
    full_melody = AudioBuffer(
        data=np.concatenate([note.data for note in melody]),
        sample_rate=melody[0].sample_rate
    )

    # Add effects
    filtered = audio.lowpass(full_melody, cutoff=3000.0)
    reverbed = audio.reverb(filtered, mix=0.15)

    # Export to FLAC (lossless)
    output_path = "/tmp/morphogen_melody.flac"
    audio.save(reverbed, output_path, format="flac")
    print(f"✅ Saved FLAC to: {output_path}")

    # Load and verify (FLAC is lossless)
    loaded = audio.load(output_path)
    correlation = np.corrcoef(reverbed.data, loaded.data)[0, 1]
    print(f"   FLAC correlation: {correlation:.4f} (should be ~1.0 for lossless)")


def demo_format_conversion():
    """Demo: Audio format conversion."""
    print("\n=== Format Conversion Demo ===")

    # Generate test audio
    original = audio.saw(freq=110.0, duration=1.0, blep=True)
    filtered = audio.lowpass(original, cutoff=2000.0)

    # Save as WAV
    wav_path = "/tmp/morphogen_test.wav"
    audio.save(filtered, wav_path)
    print(f"✅ Saved WAV: {wav_path}")

    # Load and convert to FLAC
    try:
        import soundfile

        loaded = audio.load(wav_path)
        flac_path = "/tmp/morphogen_test.flac"
        audio.save(loaded, flac_path, format="flac")
        print(f"✅ Converted to FLAC: {flac_path}")

        # Compare file sizes
        import os
        wav_size = os.path.getsize(wav_path)
        flac_size = os.path.getsize(flac_path)
        print(f"   WAV size:  {wav_size:,} bytes")
        print(f"   FLAC size: {flac_size:,} bytes")
        print(f"   Compression ratio: {wav_size/flac_size:.2f}x")
    except ImportError:
        print("⚠️  soundfile not installed, skipping FLAC conversion")


def demo_effects_chain():
    """Demo: Complex effects chain with export."""
    print("\n=== Effects Chain Demo ===")

    # Start with Karplus-Strong plucked string
    impulse = audio.impulse(amplitude=0.5, sample_rate=44100)
    pluck = audio.string(impulse, freq=82.41, t60=2.0)  # E2 (low E string)

    print("Building effects chain:")
    print("  1. Plucked string (Karplus-Strong)")

    # Add subtle chorus
    chorus = audio.chorus(pluck, rate=0.3, depth=0.005, mix=0.2)
    print("  2. Chorus")

    # Bandpass filter for tone shaping
    filtered = audio.bandpass(chorus, center=1000.0, q=1.5)
    print("  3. Bandpass filter")

    # Reverb for space
    reverb = audio.reverb(filtered, mix=0.25, size=0.9)
    print("  4. Reverb")

    # Limiter to prevent clipping
    limited = audio.limiter(reverb, threshold=-1.0)
    print("  5. Limiter")

    # Normalize
    final = audio.normalize(limited, target=0.8)
    print("  6. Normalize")

    # Export
    output_path = "/tmp/morphogen_guitar.wav"
    audio.save(final, output_path)
    print(f"\n✅ Effects chain complete: {output_path}")
    print(f"   Duration: {final.duration:.2f}s")
    print(f"   Peak level: {np.max(np.abs(final.data)):.3f}")


def demo_stereo_export():
    """Demo: Stereo audio export."""
    print("\n=== Stereo Export Demo ===")

    # Create two different tones
    left = audio.sine(freq=440.0, duration=1.0)   # A4
    right = audio.sine(freq=554.37, duration=1.0) # C#5

    # Pan them differently
    left_panned = audio.pan(left, position=-0.7)   # Left
    right_panned = audio.pan(right, position=0.7)  # Right

    # Mix to create stereo
    stereo = audio.mix(left_panned, right_panned)

    # Export
    output_path = "/tmp/morphogen_stereo.wav"
    audio.save(stereo, output_path)
    print(f"✅ Saved stereo audio: {output_path}")

    # Verify stereo
    loaded = audio.load(output_path)
    print(f"   Channels: {'stereo' if loaded.is_stereo else 'mono'}")
    print(f"   Shape: {loaded.data.shape}")


def main():
    """Run all audio I/O demos."""
    print("=" * 60)
    print("Morphogen v0.6.0 - Audio I/O Demonstrations")
    print("=" * 60)

    # Run demos
    demo_playback()
    demo_wav_export()
    demo_flac_export()
    demo_format_conversion()
    demo_effects_chain()
    demo_stereo_export()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("\nGenerated files:")
    print("  /tmp/morphogen_chord.wav    - C major chord with reverb")
    print("  /tmp/morphogen_melody.flac  - Melodic sequence (lossless)")
    print("  /tmp/morphogen_test.wav     - Filtered sawtooth")
    print("  /tmp/morphogen_test.flac    - FLAC conversion")
    print("  /tmp/morphogen_guitar.wav   - Plucked string with effects")
    print("  /tmp/morphogen_stereo.wav   - Stereo panning demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
