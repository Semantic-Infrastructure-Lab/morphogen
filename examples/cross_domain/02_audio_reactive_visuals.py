"""
Audio-Reactive Visual Generation

Demonstrates Audio → Visual transform with real-time-style processing:
- FFT spectrum visualization
- Beat-reactive particle emission
- Waveform oscilloscope
- Energy-driven color shifts

This showcases Morphogen's unique ability to couple audio analysis
with visual generation in a deterministic, reproducible way.
"""

import numpy as np
from morphogen.cross_domain import AudioToVisualInterface

print("=" * 70)
print("AUDIO-REACTIVE VISUAL GENERATION DEMO")
print("=" * 70)
print()

# ============================================================================
# Generate Test Audio Signal
# ============================================================================

print("Generating test audio signal...")
print("-" * 70)

sample_rate = 44100
duration = 5.0  # 5 seconds
t = np.linspace(0, duration, int(sample_rate * duration))

# Create complex musical signal:
# 1. Bassline (low frequency sweep)
bass_freq = 80 + 20 * np.sin(2 * np.pi * 0.5 * t)  # 60-100 Hz sweep
bass = np.sin(2 * np.pi * bass_freq * t) * 0.3

# 2. Melody (pentatonic scale notes)
melody_notes = [261.63, 293.66, 329.63, 392.00, 440.00]  # C, D, E, G, A
note_duration = 0.5
melody = np.zeros_like(t)

for i, note_freq in enumerate(melody_notes * 2):  # Repeat twice
    start_idx = int(i * note_duration * sample_rate)
    end_idx = int((i + 1) * note_duration * sample_rate)
    if end_idx > len(melody):
        break

    # ADSR envelope
    attack_samples = int(0.05 * sample_rate)
    decay_samples = int(0.1 * sample_rate)
    sustain_level = 0.7
    release_samples = int(0.2 * sample_rate)

    envelope = np.ones(end_idx - start_idx)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:attack_samples+decay_samples] = \
        np.linspace(1, sustain_level, decay_samples)
    envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

    melody[start_idx:end_idx] = np.sin(
        2 * np.pi * note_freq * t[start_idx:end_idx]
    ) * envelope * 0.2

# 3. Beat/rhythm (kick drum simulation)
beat_freq = 2.0  # 2 beats per second (120 BPM)
beat_times = np.arange(0, duration, 1.0 / beat_freq)
beats = np.zeros_like(t)

for beat_time in beat_times:
    beat_idx = int(beat_time * sample_rate)
    decay_length = int(0.3 * sample_rate)

    if beat_idx + decay_length < len(beats):
        beat_envelope = np.exp(-10 * np.linspace(0, 1, decay_length))
        beat_sound = np.sin(2 * np.pi * 60 * np.linspace(0, 0.3, decay_length)) * beat_envelope
        beats[beat_idx:beat_idx + decay_length] += beat_sound * 0.4

# Mix all components
audio_signal = bass + melody + beats

# Normalize
audio_signal = audio_signal / (np.max(np.abs(audio_signal)) + 0.01)

print(f"Generated {duration}-second audio signal")
print(f"Sample rate: {sample_rate} Hz")
print(f"Total samples: {len(audio_signal)}")
print(f"Components: bass + melody + beats")
print()

# ============================================================================
# Mode 1: Spectrum Analysis
# ============================================================================

print("MODE 1: Spectrum Analysis (FFT)")
print("-" * 70)

# Analyze multiple windows
n_windows = 10
window_size = 2048

print(f"\nAnalyzing {n_windows} time windows...")

for i in range(n_windows):
    window_start = i * (len(audio_signal) // n_windows)
    window_audio = audio_signal[window_start:window_start + window_size * 2]

    # Create transform
    spectrum_analyzer = AudioToVisualInterface(
        window_audio,
        sample_rate=sample_rate,
        fft_size=window_size,
        mode="spectrum"
    )

    # Get spectrum
    visual_params = spectrum_analyzer.transform(window_audio)

    spectrum = visual_params['spectrum']
    frequencies = visual_params['frequencies']
    brightness = visual_params['brightness']

    # Find dominant frequency
    peak_idx = np.argmax(spectrum)
    peak_freq = frequencies[peak_idx]
    peak_magnitude = spectrum[peak_idx]

    time = window_start / sample_rate

    print(f"  Window {i+1:2d} (t={time:.2f}s): "
          f"Peak={peak_freq:6.1f} Hz (mag={peak_magnitude:.3f}), "
          f"Brightness={brightness:.3f}")

print("\n✓ Spectrum analysis complete")
print("  Use case: Drive particle colors from frequency content")
print("  Example: Low freq → red, high freq → blue")

# ============================================================================
# Mode 2: Energy Detection
# ============================================================================

print("\n" + "=" * 70)
print("MODE 2: Energy Detection (RMS)")
print("-" * 70)

# Analyze energy over time
hop_length = sample_rate // 30  # 30 fps equivalent
n_frames = len(audio_signal) // hop_length

print(f"\nAnalyzing energy at {30} fps...")

energy_timeline = []

for i in range(min(n_frames, 150)):  # First 5 seconds at 30 fps
    frame_start = i * hop_length
    frame_end = frame_start + hop_length
    frame_audio = audio_signal[frame_start:frame_end]

    # Energy analysis
    energy_analyzer = AudioToVisualInterface(
        frame_audio,
        sample_rate=sample_rate,
        fft_size=min(2048, len(frame_audio)),
        mode="energy"
    )

    visual_params = energy_analyzer.transform(frame_audio)

    energy = visual_params['energy']
    intensity = visual_params['intensity']

    energy_timeline.append((i / 30.0, energy, intensity))

# Print summary statistics
energies = [e for _, e, _ in energy_timeline]
intensities = [i for _, _, i in energy_timeline]

print(f"\nEnergy statistics over {len(energy_timeline)} frames:")
print(f"  Mean energy: {np.mean(energies):.4f}")
print(f"  Peak energy: {np.max(energies):.4f}")
print(f"  Mean intensity: {np.mean(intensities):.3f}")
print(f"  Peak intensity: {np.max(intensities):.3f}")

# Show energy peaks
print(f"\nEnergy peaks (top 5):")
sorted_timeline = sorted(energy_timeline, key=lambda x: x[1], reverse=True)
for i, (time, energy, intensity) in enumerate(sorted_timeline[:5]):
    print(f"  {i+1}. t={time:.2f}s: energy={energy:.4f}, intensity={intensity:.3f}")

print("\n✓ Energy detection complete")
print("  Use case: Trigger particle emissions on beats")
print("  Example: High intensity → spawn burst of particles")

# ============================================================================
# Mode 3: Beat Detection
# ============================================================================

print("\n" + "=" * 70)
print("MODE 3: Beat Detection (Onset Strength)")
print("-" * 70)

# Use longer audio chunk for beat detection
beat_chunk_size = sample_rate * 3  # 3 seconds
beat_audio = audio_signal[:beat_chunk_size]

beat_detector = AudioToVisualInterface(
    beat_audio,
    sample_rate=sample_rate,
    fft_size=2048,
    mode="beat"
)

visual_params = beat_detector.transform(beat_audio)

onset_strength = visual_params['onset_strength']
beats = visual_params['beats']

# Find beat times
beat_indices = np.where(beats)[0]
beat_times = beat_indices * (512 / sample_rate)  # hop_length=512

print(f"\nBeat detection on {len(beat_audio) / sample_rate:.1f}s of audio:")
print(f"  Onset strength frames: {len(onset_strength)}")
print(f"  Beats detected: {np.sum(beats)}")

if len(beat_times) > 0:
    print(f"\nBeat times (seconds):")
    for i, beat_time in enumerate(beat_times[:10]):  # Show first 10
        print(f"  Beat {i+1:2d}: {beat_time:.3f}s")

    # Estimate tempo
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
        avg_interval = np.mean(beat_intervals)
        estimated_bpm = 60.0 / avg_interval
        print(f"\nEstimated tempo: {estimated_bpm:.1f} BPM")

print("\n✓ Beat detection complete")
print("  Use case: Sync visual events to music beats")
print("  Example: Flash screen, spawn particles on beat")

# ============================================================================
# Mode 4: Waveform Visualization
# ============================================================================

print("\n" + "=" * 70)
print("MODE 4: Waveform Visualization (Oscilloscope)")
print("-" * 70)

waveform_size = 2048
waveform_audio = audio_signal[:waveform_size]

waveform_viz = AudioToVisualInterface(
    waveform_audio,
    sample_rate=sample_rate,
    fft_size=waveform_size,
    mode="waveform"
)

visual_params = waveform_viz.transform(waveform_audio)

waveform = visual_params['waveform']
amplitude = visual_params['amplitude']

print(f"\nWaveform data:")
print(f"  Samples: {len(waveform)}")
print(f"  Range: [{waveform.min():.3f}, {waveform.max():.3f}]")
print(f"  Mean amplitude: {np.mean(amplitude):.3f}")
print(f"  Peak amplitude: {np.max(amplitude):.3f}")

# Simple ASCII waveform visualization
print(f"\nASCII Waveform (first 80 samples):")
display_samples = 80
step = len(waveform) // display_samples

for i in range(display_samples):
    sample_idx = i * step
    value = waveform[sample_idx]

    # Map to character position (40 chars for ±1.0 range)
    char_pos = int((value + 1.0) * 20)
    char_pos = np.clip(char_pos, 0, 40)

    line = " " * char_pos + "|"
    print(line)

print("\n✓ Waveform visualization complete")
print("  Use case: Circular oscilloscope, XY plots")
print("  Example: Draw waveform as moving line in visual")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print()
print("Demonstrated 4 audio analysis modes for visual generation:")
print("  1. Spectrum (FFT) - Frequency content → colors")
print("  2. Energy (RMS) - Loudness → particle emission")
print("  3. Beat detection - Rhythm → visual events")
print("  4. Waveform - Raw signal → oscilloscope")
print()
print("These transforms enable:")
print("  • Audio-reactive particle systems")
print("  • Music visualization")
print("  • VJ-style real-time graphics")
print("  • Procedural animation driven by sound")
print()
print("All analysis is:")
print("  ✓ Deterministic (same audio → same visuals)")
print("  ✓ Reproducible (bit-exact)")
print("  ✓ Cross-platform compatible")
print()
print("Next steps:")
print("  - Couple with visual.agents() for particle systems")
print("  - Use visual.colorize() with spectrum-driven palettes")
print("  - Sync with visual.video() for audio-visual exports")
