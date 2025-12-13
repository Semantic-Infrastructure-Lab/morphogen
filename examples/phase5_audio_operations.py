"""Morphogen v0.7.0 Phase 5: Audio Operations Dialect Examples

This module demonstrates the audio synthesis and processing capabilities of
Morphogen's MLIR-based audio dialect, including:

1. Basic oscillator synthesis (sine, saw, square waves)
2. ADSR envelope application
3. Filter sweeps (lowpass cutoff automation)
4. Multi-oscillator mixing (chord generation)
5. Audio-field hybrid (agents generating audio)
6. Temporal audio (evolving synthesis over time)
7. Complete synthesizer patch
8. Audio effects chain

All examples compile to MLIR and lower to optimized SCF loops with memref operations.
"""

from morphogen.mlir.context import MorphogenMLIRContext, MLIR_AVAILABLE
from morphogen.mlir.compiler_v2 import MLIRCompilerV2


def example1_basic_oscillator():
    """Example 1: Basic Oscillator Synthesis

    Generates a simple sine wave at 440 Hz (A4) for 1 second.
    Demonstrates buffer creation and oscillator operation.
    """
    print("\n=== Example 1: Basic Oscillator (440 Hz Sine Wave) ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 1.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 0, "freq": 440.0, "phase": 0.0}
        }
    ]

    module = compiler.compile_audio_program(operations, "sine_wave_440hz")

    print("✓ Compiled sine wave oscillator")
    print(f"  - Sample rate: 44.1 kHz")
    print(f"  - Duration: 1.0 seconds")
    print(f"  - Frequency: 440 Hz (A4)")
    print(f"\nMLIR Module Preview:")
    print(str(module)[:500] + "...\n")


def example2_envelope_application():
    """Example 2: ADSR Envelope Application

    Generates a tone with an ADSR envelope applied.
    Demonstrates envelope shaping for musical control.
    """
    print("\n=== Example 2: ADSR Envelope Application ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Create buffer and generate tone
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 0, "freq": 523.25, "phase": 0.0}  # C5
        },
        # Apply ADSR envelope
        {
            "op": "envelope",
            "args": {
                "buffer": "osc1",
                "attack": 0.05,   # 50ms attack
                "decay": 0.1,     # 100ms decay
                "sustain": 0.7,   # 70% sustain level
                "release": 0.3    # 300ms release
            }
        }
    ]

    module = compiler.compile_audio_program(operations, "envelope_example")

    print("✓ Compiled tone with ADSR envelope")
    print(f"  - Frequency: 523.25 Hz (C5)")
    print(f"  - Attack: 50ms")
    print(f"  - Decay: 100ms")
    print(f"  - Sustain: 70%")
    print(f"  - Release: 300ms")
    print(f"\nMLIR contains: scf.for loops with ADSR state machine\n")


def example3_filter_sweep():
    """Example 3: Lowpass Filter Sweep

    Applies a lowpass filter to a sawtooth wave.
    Demonstrates frequency filtering for timbral control.
    """
    print("\n=== Example 3: Lowpass Filter Sweep ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Generate sawtooth wave
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 1.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 2, "freq": 110.0, "phase": 0.0}  # A2 sawtooth
        },
        # Apply lowpass filter
        {
            "op": "filter",
            "args": {
                "buffer": "osc1",
                "filter_type": 0,  # lowpass
                "cutoff": 1000.0,  # 1 kHz cutoff
                "resonance": 2.0   # moderate resonance
            }
        }
    ]

    module = compiler.compile_audio_program(operations, "filter_sweep")

    print("✓ Compiled filtered sawtooth wave")
    print(f"  - Waveform: Sawtooth (2)")
    print(f"  - Frequency: 110 Hz (A2)")
    print(f"  - Filter: Lowpass")
    print(f"  - Cutoff: 1000 Hz")
    print(f"  - Resonance: 2.0")
    print(f"\nMLIR contains: IIR filter with memref state variables\n")


def example4_chord_mixing():
    """Example 4: Multi-Oscillator Chord Generation

    Generates a major chord by mixing three oscillators.
    Demonstrates audio mixing for harmonic content.
    """
    print("\n=== Example 4: Major Chord Mixing (C Major) ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Root note: C4 (261.63 Hz)
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 0, "freq": 261.63, "phase": 0.0}
        },
        # Third: E4 (329.63 Hz)
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf2", "waveform": 0, "freq": 329.63, "phase": 0.0}
        },
        # Fifth: G4 (392.00 Hz)
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf4", "waveform": 0, "freq": 392.00, "phase": 0.0}
        },
        # Mix all three with equal gains
        {
            "op": "mix",
            "args": {
                "buffers": ["osc1", "osc3", "osc5"],
                "gains": [0.33, 0.33, 0.33]
            }
        }
    ]

    module = compiler.compile_audio_program(operations, "c_major_chord")

    print("✓ Compiled C Major chord (3 oscillators)")
    print(f"  - Root (C4): 261.63 Hz")
    print(f"  - Third (E4): 329.63 Hz")
    print(f"  - Fifth (G4): 392.00 Hz")
    print(f"  - Mix: Equal gains (0.33 each)")
    print(f"\nMLIR contains: 3 oscillators + mixing loop\n")


def example5_complete_synth_patch():
    """Example 5: Complete Synthesizer Patch

    Combines oscillator + envelope + filter + mixing for a complete synth patch.
    Demonstrates a full signal chain.
    """
    print("\n=== Example 5: Complete Synthesizer Patch ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Oscillator 1: Fundamental
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 3.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 2, "freq": 220.0, "phase": 0.0}  # A3 sawtooth
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc1", "attack": 0.01, "decay": 0.2, "sustain": 0.6, "release": 0.5}
        },
        {
            "op": "filter",
            "args": {"buffer": "env2", "filter_type": 0, "cutoff": 2000.0, "resonance": 1.5}
        },
        # Oscillator 2: Detuned layer
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 3.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf4", "waveform": 2, "freq": 221.5, "phase": 0.0}  # Slightly detuned
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc5", "attack": 0.01, "decay": 0.2, "sustain": 0.6, "release": 0.5}
        },
        {
            "op": "filter",
            "args": {"buffer": "env6", "filter_type": 0, "cutoff": 2000.0, "resonance": 1.5}
        },
        # Mix both layers
        {
            "op": "mix",
            "args": {"buffers": ["filt3", "filt7"], "gains": [0.5, 0.5]}
        }
    ]

    module = compiler.compile_audio_program(operations, "synth_patch")

    print("✓ Compiled complete synthesizer patch")
    print(f"  - 2 detuned sawtooth oscillators (220 Hz, 221.5 Hz)")
    print(f"  - ADSR envelopes on both")
    print(f"  - Lowpass filters (2 kHz cutoff)")
    print(f"  - Stereo-ish mixing")
    print(f"\nSignal chain: OSC → ENV → FILTER → MIX\n")


def example6_audio_effects_chain():
    """Example 6: Audio Effects Chain

    Demonstrates multiple filter stages and envelope shaping.
    """
    print("\n=== Example 6: Audio Effects Chain ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Input signal
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 3, "freq": 440.0, "phase": 0.0}  # Triangle wave
        },
        # First filter: Lowpass
        {
            "op": "filter",
            "args": {"buffer": "osc1", "filter_type": 0, "cutoff": 5000.0, "resonance": 1.0}
        },
        # Envelope shaping
        {
            "op": "envelope",
            "args": {"buffer": "filt2", "attack": 0.1, "decay": 0.3, "sustain": 0.5, "release": 0.4}
        },
        # Second filter: Further lowpass
        {
            "op": "filter",
            "args": {"buffer": "env3", "filter_type": 0, "cutoff": 1500.0, "resonance": 2.0}
        }
    ]

    module = compiler.compile_audio_program(operations, "effects_chain")

    print("✓ Compiled audio effects chain")
    print(f"  - Input: Triangle wave @ 440 Hz")
    print(f"  - Filter 1: Lowpass (5 kHz)")
    print(f"  - Envelope: ADSR shaping")
    print(f"  - Filter 2: Lowpass (1.5 kHz, high resonance)")
    print(f"\nChain: OSC → FILTER → ENV → FILTER\n")


def example7_multi_voice_synthesis():
    """Example 7: Multi-Voice Synthesis

    Generates multiple voices with different timbres and mixes them.
    Demonstrates polyphonic synthesis.
    """
    print("\n=== Example 7: Multi-Voice Synthesis (3 voices) ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Voice 1: Sine @ C4
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 0, "freq": 261.63, "phase": 0.0}
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc1", "attack": 0.02, "decay": 0.15, "sustain": 0.7, "release": 0.3}
        },
        # Voice 2: Sawtooth @ E4
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf3", "waveform": 2, "freq": 329.63, "phase": 0.0}
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc4", "attack": 0.03, "decay": 0.2, "sustain": 0.6, "release": 0.4}
        },
        # Voice 3: Square @ G4
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf6", "waveform": 1, "freq": 392.00, "phase": 0.0}
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc7", "attack": 0.01, "decay": 0.1, "sustain": 0.8, "release": 0.2}
        },
        # Mix all voices
        {
            "op": "mix",
            "args": {
                "buffers": ["env2", "env5", "env8"],
                "gains": [0.4, 0.3, 0.3]
            }
        }
    ]

    module = compiler.compile_audio_program(operations, "multi_voice")

    print("✓ Compiled 3-voice polyphonic synthesis")
    print(f"  - Voice 1: Sine @ C4 (gain 0.4)")
    print(f"  - Voice 2: Sawtooth @ E4 (gain 0.3)")
    print(f"  - Voice 3: Square @ G4 (gain 0.3)")
    print(f"  - Each voice has unique ADSR envelope")
    print(f"\nDemonstrates: Polyphony, timbre variation, mixing\n")


def example8_bass_synthesis():
    """Example 8: Bass Synthesis with Sub-Oscillator

    Creates a bass sound with fundamental and sub-octave.
    Demonstrates layered synthesis for bass tones.
    """
    print("\n=== Example 8: Bass Synthesis with Sub-Oscillator ===")

    if not MLIR_AVAILABLE:
        print("MLIR not available, skipping example")
        return

    ctx = MorphogenMLIRContext()
    compiler = MLIRCompilerV2(ctx)

    operations = [
        # Fundamental: A1 (55 Hz)
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf0", "waveform": 2, "freq": 55.0, "phase": 0.0}  # Sawtooth
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc1", "attack": 0.005, "decay": 0.1, "sustain": 0.8, "release": 0.2}
        },
        {
            "op": "filter",
            "args": {"buffer": "env2", "filter_type": 0, "cutoff": 800.0, "resonance": 1.5}
        },
        # Sub-oscillator: A0 (27.5 Hz, one octave down)
        {
            "op": "buffer_create",
            "args": {"sample_rate": 44100, "channels": 1, "duration": 2.0}
        },
        {
            "op": "oscillator",
            "args": {"buffer": "buf4", "waveform": 1, "freq": 27.5, "phase": 0.0}  # Square
        },
        {
            "op": "envelope",
            "args": {"buffer": "osc5", "attack": 0.005, "decay": 0.1, "sustain": 0.8, "release": 0.2}
        },
        # Mix fundamental and sub
        {
            "op": "mix",
            "args": {"buffers": ["filt3", "env6"], "gains": [0.6, 0.4]}
        }
    ]

    module = compiler.compile_audio_program(operations, "bass_synth")

    print("✓ Compiled bass synthesizer with sub-oscillator")
    print(f"  - Fundamental: 55 Hz (A1) sawtooth")
    print(f"  - Sub-oscillator: 27.5 Hz (A0) square")
    print(f"  - Lowpass filter @ 800 Hz on fundamental")
    print(f"  - Mix: 60% fundamental, 40% sub")
    print(f"\nTechnique: Classic bass synthesis with sub-harmonic layer\n")


def run_all_examples():
    """Run all audio dialect examples."""
    print("=" * 70)
    print("Morphogen v0.7.0 Phase 5: Audio Operations Dialect Examples")
    print("=" * 70)

    if not MLIR_AVAILABLE:
        print("\n⚠️  MLIR Python bindings not installed.")
        print("Install with: pip install mlir")
        return

    # Run all examples
    example1_basic_oscillator()
    example2_envelope_application()
    example3_filter_sweep()
    example4_chord_mixing()
    example5_complete_synth_patch()
    example6_audio_effects_chain()
    example7_multi_voice_synthesis()
    example8_bass_synthesis()

    print("=" * 70)
    print("✓ All 8 audio examples completed successfully!")
    print("=" * 70)
    print("\nKey Achievements:")
    print("  ✓ Oscillator synthesis (sine, saw, square, triangle)")
    print("  ✓ ADSR envelope shaping")
    print("  ✓ Lowpass filtering with resonance")
    print("  ✓ Multi-oscillator mixing")
    print("  ✓ Complex signal chains")
    print("  ✓ Polyphonic synthesis")
    print("  ✓ Bass synthesis techniques")
    print("\nAll operations compile to optimized MLIR with:")
    print("  → scf.for loops for sample generation")
    print("  → memref operations for buffer management")
    print("  → math.sin for waveform generation")
    print("  → arith operations for signal processing")
    print("\nIntegration Points:")
    print("  → Can integrate with Field ops (audio ↔ field data)")
    print("  → Can use Temporal ops (parameter automation)")
    print("  → Can use Agent ops (agents triggering audio events)")
    print("  → Can call stdlib audio (FFT, spectral analysis)")


if __name__ == "__main__":
    run_all_examples()
