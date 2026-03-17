"""Canonical Example 04: Analysis → Instrument Model → Synthesis

Extracts acoustic parameters from a reference signal, builds an
instrument model from those parameters, then synthesises a new melody
using the learned timbre.

Pipeline:
    synthetic reference signal
    → audio_analysis: f0, partials, decay rates, inharmonicity
    → instrument_model.analyze_instrument: learned InstrumentModel
    → instrument_model.synthesize_note: new notes at arbitrary pitches
    → audio mixing: 8-note melody WAV

Output: output/04_analysis_to_instrument.wav

Run: python examples/canonical/04_analysis_to_instrument.py
"""

from pathlib import Path
import numpy as np

from morphogen.stdlib.audio_analysis import (
    track_fundamental,
    track_partials,
    fit_exponential_decay,
    measure_t60,
    measure_inharmonicity,
)
from morphogen.stdlib.instrument_model import (
    analyze_instrument,
    synthesize_note,
    InstrumentType,
)
from morphogen.stdlib.audio import AudioBuffer
from morphogen.stdlib.audio import save

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WAV = OUTPUT_DIR / "04_analysis_to_instrument.wav"

SR = 44100


# ─── Step 1: Reference signal ──────────────────────────────────────────────

def make_reference_signal(f0: float = 220.0, duration: float = 2.0) -> np.ndarray:
    """Synthetic plucked-string reference: decaying multi-harmonic tone."""
    t = np.linspace(0, duration, int(SR * duration))
    signal = np.zeros_like(t)
    for n, (amp, decay) in enumerate([
        (1.00, 2.0),   # fundamental
        (0.55, 3.0),   # 2nd partial
        (0.30, 4.2),   # 3rd
        (0.18, 5.8),   # 4th
        (0.10, 7.5),   # 5th
        (0.06, 9.0),   # 6th
        (0.03, 11.0),  # 7th
        (0.02, 13.5),  # 8th
    ], start=1):
        signal += amp * np.sin(2 * np.pi * f0 * n * t) * np.exp(-decay * t)
    return signal.astype(np.float64)


# ─── Step 2: Analysis ──────────────────────────────────────────────────────

def analyse_signal(signal: np.ndarray, sr: int = SR):
    """Extract f0, decay, and inharmonicity from a reference recording."""
    print("  Tracking fundamental frequency...")
    f0_traj = track_fundamental(signal, sr, method="yin")
    f0 = float(np.median(f0_traj[f0_traj > 0]))
    print(f"  f0 = {f0:.1f} Hz")

    print("  Tracking partial amplitudes...")
    harmonics = track_partials(signal, sr, f0_traj, num_partials=8)

    print("  Fitting exponential decay to each partial...")
    decay_rates = fit_exponential_decay(harmonics, sr)
    for i, d in enumerate(decay_rates[:4]):
        t60 = measure_t60(float(d))
        print(f"    partial {i+1}: decay={d:.2f}/s  T60={t60:.2f}s")

    print("  Measuring inharmonicity...")
    B = measure_inharmonicity(signal, sr, f0, num_partials=8)
    print(f"  B = {B:.6f}  ({'near-ideal' if abs(B) < 0.0001 else 'stretched'})")

    return f0, decay_rates


# ─── Step 3: Build instrument model ────────────────────────────────────────

def build_model(signal: np.ndarray, sr: int = SR):
    """Wrap analysis into an InstrumentModel."""
    print("  Building instrument model from analysis...")
    model = analyze_instrument(
        signal,
        sr,
        instrument_id="canonical_string",
        instrument_type=InstrumentType.MODAL_STRING,
        num_partials=8,
    )
    print(f"  model id: {model.id}")
    print(f"  model type: {model.instrument_type.name}")
    return model


# ─── Step 4: Synthesise melody ─────────────────────────────────────────────

# C major pentatonic: C4 D4 E4 G4 A4 C5 A4 G4
MELODY_PITCHES = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 440.00, 392.00]
NOTE_DURATION  = 0.9   # seconds per note
NOTE_SPACING   = 1.0   # seconds between note starts


def synthesise_melody(model, pitches=MELODY_PITCHES, sr: int = SR) -> np.ndarray:
    """Synthesise a sequence of notes using the learned timbre."""
    total = int((len(pitches) * NOTE_SPACING + NOTE_DURATION) * sr)
    out = np.zeros(total)

    for i, pitch in enumerate(pitches):
        start = int(i * NOTE_SPACING * sr)
        note = synthesize_note(model, pitch=pitch, velocity=0.8, duration=NOTE_DURATION)
        end = min(start + len(note), total)
        out[start:end] += note[: end - start]
        print(f"  note {i+1}: {pitch:.2f} Hz  → samples [{start}:{end}]")

    # Normalise to −1 dB headroom
    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.891 / peak

    return out.astype(np.float32)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ANALYSIS → INSTRUMENT MODEL → SYNTHESIS")
    print("=" * 60)
    print()

    print("Step 1: Creating reference signal (220 Hz plucked string)")
    reference = make_reference_signal(f0=220.0, duration=2.0)
    print(f"  {len(reference)} samples @ {SR} Hz  ({len(reference)/SR:.1f}s)")
    print()

    print("Step 2: Analysing reference signal")
    f0, decay_rates = analyse_signal(reference, SR)
    print()

    print("Step 3: Building instrument model")
    model = build_model(reference, SR)
    print()

    print("Step 4: Synthesising C major pentatonic melody")
    melody = synthesise_melody(model)
    print()

    print("Step 5: Saving WAV")
    buf = AudioBuffer(melody, sample_rate=SR)
    save(buf, str(OUTPUT_WAV))
    print(f"  Saved: {OUTPUT_WAV}")
    print()

    print("Summary")
    print(f"  Reference pitch:     {f0:.1f} Hz")
    print(f"  Partial 1 T60:       {measure_t60(float(decay_rates[0])):.2f}s")
    print(f"  Melody notes:        {len(MELODY_PITCHES)}")
    print(f"  Output duration:     {len(melody)/SR:.1f}s")
    print()
    print("Key insight: audio_analysis extracts *how* an instrument sounds.")
    print("instrument_model re-applies that timbre at any pitch or velocity.")
    print("Together they answer: 'what would this string sound like playing D4?'")


if __name__ == "__main__":
    main()
