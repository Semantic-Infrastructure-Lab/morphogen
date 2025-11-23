#!/usr/bin/env python3
"""Classic Subtractive Synthesizer with VCF and ADSR Modulation

This example demonstrates the new vcf_lowpass operator with control-rate
envelope modulation - a fundamental technique in analog synthesis.

Architecture:
    Sawtooth Oscillator (48kHz audio-rate)
          ↓
    VCF Lowpass (48kHz audio-rate)
          ↑ cutoff modulation
    ADSR Envelope (1kHz control-rate)

The scheduler automatically resamples the control-rate envelope to match
the audio-rate filter input, enabling efficient multi-rate synthesis.

This is the classic subtractive synthesis patch that defined the sound of
analog synthesizers from the 1960s-1980s (Moog, ARP, Sequential Circuits).
"""

import numpy as np
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer


def create_subtractive_synth_note(
    fundamental: float = 110.0,     # Note frequency in Hz (A2)
    duration: float = 3.0,          # Note duration in seconds
    attack: float = 0.01,           # Envelope attack time
    decay: float = 0.8,             # Envelope decay time
    sustain: float = 0.2,           # Envelope sustain level (0-1)
    release: float = 1.2,           # Envelope release time
    cutoff_min: float = 300.0,      # Minimum cutoff frequency
    cutoff_max: float = 4000.0,     # Maximum cutoff frequency
    resonance: float = 3.0,         # Filter resonance (Q)
    audio_rate: int = 48000,        # Audio sample rate
    control_rate: int = 1000,       # Envelope sample rate
) -> AudioBuffer:
    """Generate a single note using classic subtractive synthesis.

    Args:
        fundamental: Oscillator frequency in Hz
        duration: Note duration in seconds
        attack: ADSR attack time in seconds
        decay: ADSR decay time in seconds
        sustain: ADSR sustain level (0.0 to 1.0)
        release: ADSR release time in seconds
        cutoff_min: Minimum filter cutoff in Hz (envelope at 0)
        cutoff_max: Maximum filter cutoff in Hz (envelope at 1)
        resonance: Filter Q factor (higher = more resonant)
        audio_rate: Audio processing sample rate
        control_rate: Envelope processing sample rate

    Returns:
        Filtered audio buffer (classic analog synth sound)
    """

    print(f"\n{'='*60}")
    print(f"Generating Subtractive Synth Note")
    print(f"{'='*60}")

    # Step 1: Generate oscillator (sawtooth for rich harmonic content)
    print(f"\n1. Oscillator:")
    print(f"   Waveform: Sawtooth")
    print(f"   Frequency: {fundamental:.1f} Hz")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Sample rate: {audio_rate} Hz")

    oscillator = audio.saw(
        freq=fundamental,
        duration=duration,
        sample_rate=audio_rate
    )

    print(f"   Generated: {len(oscillator.data)} samples")

    # Step 2: Generate ADSR envelope at control rate
    print(f"\n2. ADSR Envelope (Filter Modulation):")
    print(f"   Attack: {attack:.3f}s")
    print(f"   Decay: {decay:.3f}s")
    print(f"   Sustain: {sustain:.2f}")
    print(f"   Release: {release:.3f}s")
    print(f"   Sample rate: {control_rate} Hz (control-rate)")

    envelope = audio.adsr(
        attack=attack,
        decay=decay,
        sustain=sustain,
        release=release,
        duration=duration,
        sample_rate=control_rate
    )

    print(f"   Generated: {len(envelope.data)} samples")
    print(f"   Rate ratio: {audio_rate/control_rate:.0f}x (audio / control)")

    # Step 3: Map envelope to filter cutoff range
    print(f"\n3. Cutoff Modulation:")
    print(f"   Envelope range: 0.0 → 1.0")
    print(f"   Cutoff range: {cutoff_min:.0f}Hz → {cutoff_max:.0f}Hz")

    cutoff_modulation = AudioBuffer(
        data=cutoff_min + envelope.data * (cutoff_max - cutoff_min),
        sample_rate=control_rate  # Still at control rate
    )

    print(f"   Min cutoff: {np.min(cutoff_modulation.data):.1f} Hz")
    print(f"   Max cutoff: {np.max(cutoff_modulation.data):.1f} Hz")
    print(f"   Modulation samples: {len(cutoff_modulation.data)} @ {control_rate}Hz")

    # Step 4: Apply VCF with envelope modulation
    # Note: In a real scheduler, the cutoff would be automatically resampled
    # Here we manually resample to demonstrate the concept
    print(f"\n4. Voltage-Controlled Filter:")
    print(f"   Type: Lowpass (biquad)")
    print(f"   Resonance (Q): {resonance:.1f}")
    print(f"   Modulation: Control-rate → Audio-rate")

    # Resample cutoff modulation to audio rate
    cutoff_resampled = np.interp(
        np.linspace(0, len(cutoff_modulation.data) - 1, len(oscillator.data)),
        np.arange(len(cutoff_modulation.data)),
        cutoff_modulation.data
    )
    cutoff_audio_rate = AudioBuffer(data=cutoff_resampled, sample_rate=audio_rate)

    print(f"   Cutoff resampled: {len(cutoff_audio_rate.data)} samples @ {audio_rate}Hz")

    # Apply VCF
    filtered = audio.vcf_lowpass(oscillator, cutoff_audio_rate, q=resonance)

    print(f"   Filtering complete: {len(filtered.data)} samples")

    # Step 5: Analyze output
    print(f"\n5. Output Analysis:")

    rms = np.sqrt(np.mean(filtered.data**2))
    peak = np.max(np.abs(filtered.data))
    dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))

    print(f"   RMS: {rms:.4f}")
    print(f"   Peak: {peak:.4f}")
    print(f"   Dynamic range: {dynamic_range_db:.1f} dB")

    # Analyze temporal evolution (brightness over time)
    quarter_len = len(filtered.data) // 4
    q1_rms = np.sqrt(np.mean(filtered.data[0:quarter_len]**2))
    q2_rms = np.sqrt(np.mean(filtered.data[quarter_len:2*quarter_len]**2))
    q3_rms = np.sqrt(np.mean(filtered.data[2*quarter_len:3*quarter_len]**2))
    q4_rms = np.sqrt(np.mean(filtered.data[3*quarter_len:4*quarter_len]**2))

    print(f"\n   Temporal evolution:")
    print(f"     Q1 (attack):  RMS = {q1_rms:.4f}")
    print(f"     Q2 (decay):   RMS = {q2_rms:.4f}")
    print(f"     Q3 (sustain): RMS = {q3_rms:.4f}")
    print(f"     Q4 (release): RMS = {q4_rms:.4f}")

    brightness_change = ((q2_rms - q1_rms) / q1_rms) * 100
    print(f"     Brightness change (attack→decay): {brightness_change:+.1f}%")

    print(f"\n{'='*60}")
    print(f"Synthesis Complete!")
    print(f"{'='*60}")

    return filtered


def main():
    """Demonstrate subtractive synthesis with multiple presets."""

    print("\n" + "="*70)
    print("VCF SUBTRACTIVE SYNTHESIS DEMONSTRATION")
    print("="*70)
    print("\nThis example demonstrates classic analog synthesis using:")
    print("  • Sawtooth oscillator (rich harmonic content)")
    print("  • ADSR envelope generator (control-rate @ 1kHz)")
    print("  • Voltage-controlled lowpass filter (audio-rate @ 48kHz)")
    print("  • Automatic cross-rate resampling")
    print()

    # ========================================================================
    # Preset 1: Classic Bass (short, punchy)
    # ========================================================================
    print("\n" + "="*70)
    print("PRESET 1: Classic Bass")
    print("="*70)

    bass = create_subtractive_synth_note(
        fundamental=55.0,       # A1 - low bass note
        duration=1.5,
        attack=0.005,           # Very fast attack
        decay=0.3,              # Quick decay
        sustain=0.1,            # Low sustain
        release=0.2,            # Short release
        cutoff_min=200.0,       # Dark starting point
        cutoff_max=1200.0,      # Moderate brightness peak
        resonance=4.0,          # Punchy resonance
    )

    # ========================================================================
    # Preset 2: Sweeping Lead (bright, evolving)
    # ========================================================================
    print("\n" + "="*70)
    print("PRESET 2: Sweeping Lead")
    print("="*70)

    lead = create_subtractive_synth_note(
        fundamental=440.0,      # A4 - lead range
        duration=3.0,
        attack=0.02,            # Gentle attack
        decay=1.2,              # Long decay
        sustain=0.4,            # Medium sustain
        release=0.8,            # Smooth release
        cutoff_min=800.0,       # Already bright
        cutoff_max=6000.0,      # Very bright peak
        resonance=2.5,          # Moderate resonance
    )

    # ========================================================================
    # Preset 3: Pad (slow, ethereal)
    # ========================================================================
    print("\n" + "="*70)
    print("PRESET 3: Ambient Pad")
    print("="*70)

    pad = create_subtractive_synth_note(
        fundamental=220.0,      # A3 - mid range
        duration=4.0,
        attack=0.5,             # Slow fade in
        decay=1.5,              # Long decay
        sustain=0.6,            # High sustain
        release=2.0,            # Very long release
        cutoff_min=400.0,       # Warm starting point
        cutoff_max=2500.0,      # Moderate brightness
        resonance=1.5,          # Subtle resonance
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    print(f"\n✅ Generated 3 preset examples:")
    print(f"   • Classic Bass:  {len(bass.data)} samples ({bass.duration:.2f}s)")
    print(f"   • Sweeping Lead: {len(lead.data)} samples ({lead.duration:.2f}s)")
    print(f"   • Ambient Pad:   {len(pad.data)} samples ({pad.duration:.2f}s)")
    print(f"\n✅ All examples use cross-rate modulation:")
    print(f"   • Audio rate: 48kHz (oscillator & filter)")
    print(f"   • Control rate: 1kHz (envelope)")
    print(f"   • Automatic resampling: 48x upsampling")
    print(f"\n✅ VCF features demonstrated:")
    print(f"   • Time-varying cutoff frequency")
    print(f"   • Per-sample coefficient recomputation")
    print(f"   • Continuous filter state maintenance")
    print(f"   • Resonance control (Q factor)")
    print(f"\n💡 To play audio, uncomment the save/play lines below")
    print("="*70)

    # Optional: Save to WAV files
    # Uncomment these lines to save audio files:
    #
    # audio.save(bass, "vcf_bass.wav")
    # audio.save(lead, "vcf_lead.wav")
    # audio.save(pad, "vcf_pad.wav")
    #
    # Or play in real-time:
    # audio.play(bass, blocking=True)


if __name__ == "__main__":
    main()
