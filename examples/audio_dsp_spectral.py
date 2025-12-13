#!/usr/bin/env python3
"""Audio DSP & Spectral Analysis Examples

Demonstrates comprehensive audio buffer operations, FFT/STFT transforms,
spectral analysis metrics, and frequency-domain processing.

This example showcases the new audio operations added to Morphogen v0.7.0:
- Buffer operations (slice, concat, resample, reverse, fade)
- FFT/spectral transforms (fft, ifft, stft, istft)
- Spectral analysis (centroid, rolloff, flux, peaks, rms, zero crossings)
- Spectral processing (gate, filter, convolution)

Usage:
    python examples/audio_dsp_spectral.py
"""

import numpy as np
from morphogen.stdlib.audio import audio


def example_1_buffer_operations():
    """Example 1: Audio Buffer Operations

    Demonstrates slicing, concatenation, resampling, reversing, and fading.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Buffer Operations")
    print("=" * 70)

    # Create a simple melody
    note_duration = 0.3
    c4 = audio.sine(freq=261.63, duration=note_duration)
    e4 = audio.sine(freq=329.63, duration=note_duration)
    g4 = audio.sine(freq=392.00, duration=note_duration)
    c5 = audio.sine(freq=523.25, duration=note_duration)

    # Concatenate notes into melody
    melody = audio.concat(c4, e4, g4, c5)
    print(f"✓ Created melody: {melody.duration:.2f}s")

    # Apply fade in/out
    melody = audio.fade_in(melody, duration=0.1)
    melody = audio.fade_out(melody, duration=0.1)
    print(f"✓ Applied fades")

    # Slice middle section
    middle = audio.slice(melody, start=0.3, end=0.9)
    print(f"✓ Sliced middle section: {middle.duration:.2f}s")

    # Reverse the melody
    reversed_melody = audio.reverse(melody)
    print(f"✓ Reversed melody")

    # Create call-and-response pattern
    pattern = audio.concat(melody, reversed_melody)
    print(f"✓ Call-and-response pattern: {pattern.duration:.2f}s")

    # Resample to 48kHz
    resampled = audio.resample(pattern, new_sample_rate=48000)
    print(f"✓ Resampled to {resampled.sample_rate}Hz")

    print(f"\n  Final buffer: {resampled.num_samples:,} samples, {resampled.duration:.2f}s")


def example_2_fft_analysis():
    """Example 2: FFT Spectrum Analysis

    Demonstrates FFT, frequency peak detection, and magnitude spectrum.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: FFT Spectrum Analysis")
    print("=" * 70)

    # Create complex signal with multiple harmonics
    fundamental = 220.0  # A3
    signal = audio.sine(freq=fundamental, duration=1.0)

    # Add harmonics
    for harmonic in [2, 3, 4, 5]:
        harmonic_sig = audio.sine(freq=fundamental * harmonic, duration=1.0)
        harmonic_sig = audio.gain(harmonic_sig, amount_db=-6.0 * harmonic)
        signal = audio.mix(signal, harmonic_sig)

    print(f"✓ Created signal with fundamental {fundamental}Hz + 4 harmonics")

    # Compute FFT
    freqs, spectrum = audio.fft(signal)
    print(f"✓ Computed FFT: {len(spectrum):,} frequency bins")

    # Get magnitude spectrum
    freqs, magnitudes = audio.spectrum(signal)
    max_mag_idx = np.argmax(magnitudes)
    print(f"✓ Peak magnitude at {freqs[max_mag_idx]:.1f}Hz")

    # Find spectral peaks
    peak_freqs, peak_mags = audio.spectral_peaks(signal, num_peaks=5, min_freq=100.0)
    print(f"\n  Top 5 frequency peaks:")
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)):
        print(f"    {i+1}. {freq:7.1f} Hz  (magnitude: {mag:.2f})")

    # Get phase spectrum
    freqs, phases = audio.phase_spectrum(signal)
    print(f"\n  ✓ Phase spectrum computed ({len(phases):,} bins)")


def example_3_spectral_analysis_metrics():
    """Example 3: Spectral Analysis Metrics

    Demonstrates spectral centroid, rolloff, flux, RMS, and zero crossings.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Spectral Analysis Metrics")
    print("=" * 70)

    # Create different sound types
    sounds = {
        "Bass tone (100Hz)": audio.sine(freq=100.0, duration=1.0),
        "Mid tone (1kHz)": audio.sine(freq=1000.0, duration=1.0),
        "High tone (5kHz)": audio.sine(freq=5000.0, duration=1.0),
        "White noise": audio.noise(noise_type="white", seed=42, duration=1.0),
        "Pink noise": audio.noise(noise_type="pink", seed=42, duration=1.0),
    }

    print("\n  Spectral Analysis Comparison:\n")
    print(f"  {'Sound Type':<20} {'Centroid':>10} {'Rolloff':>10} {'RMS':>8} {'ZCR':>10}")
    print(f"  {'-' * 68}")

    for name, signal in sounds.items():
        centroid = audio.spectral_centroid(signal)
        rolloff = audio.spectral_rolloff(signal, threshold=0.85)
        rms = audio.rms(signal)
        zcr = audio.zero_crossings(signal)

        print(f"  {name:<20} {centroid:>9.1f}Hz {rolloff:>9.1f}Hz {rms:>7.3f} {zcr:>10,}")

    # Spectral flux for onset detection
    print("\n  Spectral Flux (Onset Detection):")
    silence = audio.sine(freq=0.0, duration=0.5, sample_rate=44100)
    silence.data[:] = 0  # Make silent
    tone = audio.sine(freq=440.0, duration=0.5)
    signal_with_onset = audio.concat(silence, tone)

    flux = audio.spectral_flux(signal_with_onset)
    onset_idx = np.argmax(flux)
    print(f"    ✓ Onset detected at frame {onset_idx} (flux spike: {flux[onset_idx]:.2f})")


def example_4_stft_time_frequency_analysis():
    """Example 4: Short-Time Fourier Transform (STFT)

    Demonstrates time-frequency analysis with STFT and ISTFT reconstruction.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: STFT Time-Frequency Analysis")
    print("=" * 70)

    # Create frequency sweep (chirp)
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Linear chirp from 200 Hz to 2000 Hz
    f0, f1 = 200.0, 2000.0
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))

    from morphogen.stdlib.audio import AudioBuffer
    signal = AudioBuffer(data=chirp, sample_rate=sample_rate)
    print(f"✓ Created chirp signal: {f0}Hz → {f1}Hz over {duration}s")

    # Compute STFT
    window_size = 2048
    hop_size = 512
    times, freqs, stft_matrix = audio.stft(signal, window_size=window_size, hop_size=hop_size)

    print(f"✓ STFT computed:")
    print(f"    Time frames: {len(times)}")
    print(f"    Frequency bins: {len(freqs)}")
    print(f"    Matrix shape: {stft_matrix.shape}")

    # Analyze time-frequency content
    spectrogram = np.abs(stft_matrix)

    # Find dominant frequency at different times
    print(f"\n  Frequency progression over time:")
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    for t_target in time_points:
        # Find closest time frame
        frame_idx = np.argmin(np.abs(times - t_target))
        # Find peak frequency in this frame
        peak_freq_idx = np.argmax(spectrogram[:, frame_idx])
        peak_freq = freqs[peak_freq_idx]
        print(f"    t = {times[frame_idx]:.2f}s → {peak_freq:.0f}Hz")

    # Reconstruct signal using ISTFT
    reconstructed = audio.istft(stft_matrix, hop_size=hop_size, sample_rate=sample_rate)
    print(f"\n  ✓ ISTFT reconstruction: {reconstructed.num_samples:,} samples")

    # Verify reconstruction quality
    min_len = min(len(signal.data), len(reconstructed.data))
    mid_start = min_len // 4
    mid_end = 3 * min_len // 4
    error = np.mean(np.abs(signal.data[mid_start:mid_end] - reconstructed.data[mid_start:mid_end]))
    print(f"  ✓ Reconstruction error (middle): {error:.6f}")


def example_5_spectral_filtering():
    """Example 5: Spectral Domain Filtering

    Demonstrates frequency-domain filtering and manipulation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Spectral Domain Filtering")
    print("=" * 70)

    # Create broadband signal (white noise + tones)
    noise = audio.noise(noise_type="white", seed=42, duration=1.0)
    noise = audio.gain(noise, amount_db=-20.0)

    tone1 = audio.sine(freq=440.0, duration=1.0)
    tone2 = audio.sine(freq=1000.0, duration=1.0)

    signal = audio.mix(noise, tone1, tone2)
    print(f"✓ Created signal: 2 tones + noise")
    print(f"  Original RMS: {audio.rms(signal):.4f}")

    # FFT-based bandpass filter (keep only 300-1500 Hz)
    freqs, spectrum = audio.fft(signal)
    mask = ((freqs >= 300) & (freqs <= 1500)).astype(float)

    filtered = audio.spectral_filter(signal, mask)
    print(f"\n  ✓ Applied bandpass filter (300-1500 Hz)")
    print(f"    Filtered RMS: {audio.rms(filtered):.4f}")

    # Analyze frequency content before/after
    peaks_orig, _ = audio.spectral_peaks(signal, num_peaks=3)
    peaks_filt, _ = audio.spectral_peaks(filtered, num_peaks=3)

    print(f"\n  Original peaks: {peaks_orig[:3].astype(int)} Hz")
    print(f"  Filtered peaks: {peaks_filt[:3].astype(int)} Hz")

    # Notch filter (remove 1000 Hz)
    freqs, spectrum = audio.fft(signal)
    # Create notch: zero out frequencies near 1000 Hz (±50 Hz)
    mask = np.ones(len(freqs))
    mask[(freqs >= 950) & (freqs <= 1050)] = 0.0

    notched = audio.spectral_filter(signal, mask)
    print(f"\n  ✓ Applied notch filter (remove 1000Hz)")

    peaks_notch, mags_notch = audio.spectral_peaks(notched, num_peaks=3)
    print(f"    Remaining peaks: {peaks_notch[:3].astype(int)} Hz")
    print(f"    Peak magnitudes: {mags_notch[:3]}")


def example_6_spectral_gate_noise_reduction():
    """Example 6: Spectral Noise Gate

    Demonstrates spectral gating for noise reduction.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Spectral Noise Gate (Noise Reduction)")
    print("=" * 70)

    # Create clean signal
    melody = audio.sine(freq=440.0, duration=1.0)
    melody = audio.fade_in(melody, duration=0.05)
    melody = audio.fade_out(melody, duration=0.05)

    # Add noise
    noise = audio.noise(noise_type="white", seed=123, duration=1.0)
    noise = audio.gain(noise, amount_db=-25.0)  # Moderate noise

    noisy_signal = audio.mix(melody, noise)

    print(f"✓ Created noisy signal")
    print(f"  Clean RMS: {audio.rms(melody):.4f}")
    print(f"  Noisy RMS: {audio.rms(noisy_signal):.4f}")

    # Apply spectral gate with different thresholds
    thresholds = [-50, -40, -30]
    print(f"\n  Spectral gate at different thresholds:")

    for threshold in thresholds:
        gated = audio.spectral_gate(noisy_signal, threshold_db=threshold)
        rms_gated = audio.rms(gated)
        print(f"    {threshold:>4}dB → RMS: {rms_gated:.4f}")

    # Analyze spectral content
    centroid_clean = audio.spectral_centroid(melody)
    centroid_noisy = audio.spectral_centroid(noisy_signal)
    gated_final = audio.spectral_gate(noisy_signal, threshold_db=-40.0)
    centroid_gated = audio.spectral_centroid(gated_final)

    print(f"\n  Spectral Centroid:")
    print(f"    Clean:  {centroid_clean:.1f}Hz")
    print(f"    Noisy:  {centroid_noisy:.1f}Hz")
    print(f"    Gated:  {centroid_gated:.1f}Hz")


def example_7_convolution_reverb():
    """Example 7: Convolution Reverb

    Demonstrates FFT-based convolution for reverb effects.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Convolution Reverb")
    print("=" * 70)

    # Create dry signal (short impulse)
    dry_signal = audio.sine(freq=440.0, duration=0.1)
    dry_signal = audio.fade_in(dry_signal, duration=0.005)
    dry_signal = audio.fade_out(dry_signal, duration=0.005)

    print(f"✓ Created dry signal: {dry_signal.duration:.2f}s")
    print(f"  RMS: {audio.rms(dry_signal):.4f}")

    # Create synthetic impulse response (exponential decay with early reflections)
    ir_duration = 0.5
    ir_samples = int(ir_duration * 44100)
    impulse_response_data = np.zeros(ir_samples)

    # Direct sound
    impulse_response_data[0] = 1.0

    # Early reflections
    reflections = [
        (1000, 0.6),   # 22.7ms
        (1500, 0.4),   # 34.0ms
        (2200, 0.3),   # 49.9ms
    ]
    for delay_samples, gain in reflections:
        if delay_samples < ir_samples:
            impulse_response_data[delay_samples] = gain

    # Exponential decay tail
    decay_start = 3000
    decay = np.exp(-np.arange(ir_samples - decay_start) / 5000.0)
    impulse_response_data[decay_start:] += decay * 0.2

    from morphogen.stdlib.audio import AudioBuffer
    impulse_response = AudioBuffer(data=impulse_response_data, sample_rate=44100)

    print(f"✓ Created impulse response: {impulse_response.duration:.2f}s")

    # Apply convolution
    reverb_signal = audio.convolution(dry_signal, impulse_response)

    print(f"\n  ✓ Convolution complete:")
    print(f"    Dry signal:    {dry_signal.num_samples:,} samples")
    print(f"    Impulse resp:  {impulse_response.num_samples:,} samples")
    print(f"    Reverb output: {reverb_signal.num_samples:,} samples")
    print(f"    Output RMS:    {audio.rms(reverb_signal):.4f}")

    # Analyze spectral change
    centroid_dry = audio.spectral_centroid(dry_signal)
    # Pad reverb signal to match dry signal length for fair comparison
    reverb_trimmed = audio.slice(reverb_signal, start=0.0, end=dry_signal.duration)
    centroid_reverb = audio.spectral_centroid(reverb_trimmed)

    print(f"\n  Spectral Analysis:")
    print(f"    Dry centroid:   {centroid_dry:.1f}Hz")
    print(f"    Reverb centroid: {centroid_reverb:.1f}Hz")


def example_8_complete_workflow():
    """Example 8: Complete Audio Processing Workflow

    Demonstrates a complete audio processing pipeline combining multiple operations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Complete Audio Processing Workflow")
    print("=" * 70)

    # Step 1: Create source material
    print("\n  Step 1: Create source material")
    bass = audio.sine(freq=110.0, duration=0.5)
    mid = audio.sine(freq=440.0, duration=0.5)
    high = audio.sine(freq=1760.0, duration=0.5)

    # Add envelopes
    bass = audio.fade_in(bass, duration=0.05)
    bass = audio.fade_out(bass, duration=0.1)

    # Step 2: Arrange sequence
    print("  Step 2: Arrange sequence")
    sequence = audio.concat(bass, mid, high)
    sequence = audio.concat(sequence, audio.reverse(sequence))
    print(f"    ✓ Created sequence: {sequence.duration:.2f}s")

    # Step 3: Spectral analysis
    print("  Step 3: Analyze spectral content")
    centroid = audio.spectral_centroid(sequence)
    rolloff = audio.spectral_rolloff(sequence)
    rms = audio.rms(sequence)
    print(f"    Centroid: {centroid:.1f}Hz")
    print(f"    Rolloff:  {rolloff:.1f}Hz")
    print(f"    RMS:      {rms:.4f}")

    # Step 4: Apply spectral processing
    print("  Step 4: Apply spectral EQ (boost mids)")
    freqs, spectrum = audio.fft(sequence)

    # Boost 500-2000 Hz range
    boost_mask = np.ones(len(freqs))
    boost_mask[(freqs >= 500) & (freqs <= 2000)] = 2.0

    processed = audio.spectral_filter(sequence, boost_mask)
    processed = audio.normalize(processed, target=0.8)
    print(f"    ✓ Applied spectral boost")

    # Step 5: Time-frequency analysis
    print("  Step 5: STFT analysis")
    times, freqs, stft_matrix = audio.stft(processed, window_size=2048, hop_size=512)
    print(f"    ✓ STFT: {stft_matrix.shape[1]} time frames")

    # Step 6: Final processing
    print("  Step 6: Final processing chain")
    final = audio.fade_in(processed, duration=0.1)
    final = audio.fade_out(final, duration=0.2)

    # Analyze final result
    peaks, mags = audio.spectral_peaks(final, num_peaks=5)
    print(f"\n  Final Analysis:")
    print(f"    Duration:   {final.duration:.2f}s")
    print(f"    RMS:        {audio.rms(final):.4f}")
    print(f"    Centroid:   {audio.spectral_centroid(final):.1f}Hz")
    print(f"    Top peaks:  {peaks[:3].astype(int)} Hz")

    print("\n  ✓ Workflow complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  MORPHOGEN AUDIO DSP & SPECTRAL ANALYSIS EXAMPLES")
    print("=" * 70)
    print("\n  Demonstrating 21 new audio operations:")
    print("    • 6 buffer operations")
    print("    • 6 spectral transforms")
    print("    • 6 analysis metrics")
    print("    • 3 spectral processing operations")

    try:
        example_1_buffer_operations()
        example_2_fft_analysis()
        example_3_spectral_analysis_metrics()
        example_4_stft_time_frequency_analysis()
        example_5_spectral_filtering()
        example_6_spectral_gate_noise_reduction()
        example_7_convolution_reverb()
        example_8_complete_workflow()

        print("\n" + "=" * 70)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY ✅")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
