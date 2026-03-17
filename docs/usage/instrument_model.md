---
title: "Instrument Model Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - instrument_model
  - audio
  - synthesis
  - timbre
---

# Instrument Model

The `instrument_model` domain builds reusable playable models of instruments
from audio recordings — and lets you morph between them.

The pipeline is four steps:

```
audio recording  →  analyze_instrument()  →  InstrumentModel
InstrumentModel  →  synthesize_note(pitch, velocity)  →  audio
InstrumentModel A + B  →  morph_instruments(blend)  →  InstrumentModel
InstrumentModel  →  save_instrument / load_instrument
```

---

## Step 1 — Analyse a recording

`analyze_instrument` takes a single-note recording and extracts everything
needed to re-synthesise it at any pitch and velocity.

```python
import numpy as np
from morphogen.stdlib.instrument_model import (
    analyze_instrument, InstrumentType
)

sample_rate = 44100
t = np.linspace(0, 2.0, sample_rate * 2)

# Simulate a guitar pluck at A3 (220 Hz)
signal = (
    np.sin(2 * np.pi * 220 * t) * np.exp(-t * 2)
    + 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 3)
    + 0.2 * np.sin(2 * np.pi * 660 * t) * np.exp(-t * 4)
)

model = analyze_instrument(
    signal,
    sample_rate,
    instrument_id="acoustic_guitar",
    instrument_type=InstrumentType.MODAL_STRING,
    num_partials=20,   # harmonics to track
    num_modes=20,      # resonant modes to extract
)

print(f"Fundamental: {model.fundamental:.1f} Hz")
print(f"Harmonics shape: {model.harmonics.shape}")   # (n_frames, num_partials)
print(f"Decay rates (first 4): {model.decay_rates[:4].round(2)}")
print(f"Inharmonicity B: {model.inharmonicity:.5f}")
```

What `analyze_instrument` does internally:
1. `track_fundamental` — YIN pitch detection → f0 trajectory
2. `track_partials` — harmonic amplitude tracking
3. `analyze_modes` — resonant mode extraction (frequency, amplitude, decay)
4. `deconvolve` — cepstral source-filter separation (excitation + body IR)
5. `model_noise` — 32-band spectral noise envelope
6. `fit_exponential_decay` — decay rate per partial
7. `measure_inharmonicity` — string stiffness coefficient

**`InstrumentType` options:**
- `MODAL_STRING` — guitar, harp, piano (modal synthesis)
- `MODAL_MEMBRANE` — drums, marimba (modal synthesis)
- `ADDITIVE` — additive synthesis from harmonic content
- `WAVEGUIDE` — physical waveguide model
- `HYBRID` — any of the above with body IR convolution

---

## Step 2 — Synthesise notes

`synthesize_note` generates a new note at any pitch and velocity using
the extracted model. The pitch ratio scales all modal frequencies,
body IR, and decay envelopes proportionally.

```python
from morphogen.stdlib.instrument_model import synthesize_note
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer

# Synthesise a D4 (293.66 Hz) from our A3 model
note = synthesize_note(
    model,
    pitch=293.66,   # Hz
    velocity=0.9,   # 0–1
    duration=2.0,   # seconds
)
# → np.ndarray, shape (n_samples,), peak ≈ velocity * 0.9

buf = AudioBuffer(note, sample_rate)
audio.save(buf, "guitar_D4.wav")
```

**Pitch transposition range:** the model sounds most natural within ±1
octave of its analysed fundamental. Beyond that, modal synthesis
artifacts become audible.

**Velocity** maps linearly to peak amplitude — `velocity=1.0` gives a
peak of 0.9 (headroom preserved), `velocity=0.5` gives 0.45.

---

## Step 3 — Morph between two instruments

`morph_instruments` linearly interpolates every parameter of two models:
modes, harmonics, body IR, noise, decay rates, inharmonicity. The result
is a playable model that sounds like a blend of both.

```python
from morphogen.stdlib.instrument_model import morph_instruments

# Build a second model (a slower-decaying 'viola-like' tone)
slow_decay = (
    np.sin(2 * np.pi * 220 * t) * np.exp(-t * 0.5)
    + 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.8)
)
model_viola = analyze_instrument(
    slow_decay, sample_rate,
    instrument_id="viola",
    instrument_type=InstrumentType.MODAL_STRING,
)

# Blend 0 = pure guitar, 1 = pure viola, 0.5 = halfway
morphed = morph_instruments(model_guitar, model_viola, blend=0.5)
print(f"Morphed id: {morphed.id}")   # "guitar_viola_morph_0.50"

# Synthesise a sweep of blends
for blend in [0.0, 0.25, 0.5, 0.75, 1.0]:
    m = morph_instruments(model_guitar, model_viola, blend=blend)
    note = synthesize_note(m, pitch=440.0, velocity=0.8, duration=1.5)
    buf = AudioBuffer(note, sample_rate)
    audio.save(buf, f"morph_{blend:.2f}.wav")
```

This is the key creative affordance: `morph_instruments` treats timbre
as a first-class interpolatable value, not a fixed preset.

---

## Step 4 — Save and reload models

```python
from morphogen.stdlib.instrument_model import save_instrument, load_instrument

save_instrument(model, "guitar_A3.pkl")

# Later session:
model = load_instrument("guitar_A3.pkl")
note = synthesize_note(model, pitch=440.0, velocity=0.8, duration=2.0)
```

Models serialise to Python pickle (`.pkl`). They're small — a typical
20-partial, 20-mode model is ~50–200 KB.

---

## Adjusting synthesis parameters

`model.synth_params` controls how synthesis is performed:

```python
from morphogen.stdlib.instrument_model import SynthParams

model.synth_params = SynthParams(
    pluck_position=0.12,   # closer to bridge → brighter
    pluck_stiffness=0.98,  # harder pluck → more attack transient
    body_coupling=0.7,     # reduce body resonance contribution
    noise_level=-70.0,     # quieter broadband noise floor (dB)
)

note = synthesize_note(model, pitch=440.0, velocity=0.8, duration=2.0)
```

`SynthParams` is not re-analysed — it's applied at synthesis time, so
you can adjust without re-running the analysis.

---

## Full operator reference

| Operator | Returns | Notes |
|----------|---------|-------|
| `analyze_instrument(signal, sr, id, type, num_partials, num_modes)` | `InstrumentModel` | Full analysis pipeline |
| `synthesize_note(model, pitch, velocity, duration)` | `np.ndarray` | Mono audio, shape `(n_samples,)` |
| `morph_instruments(model_a, model_b, blend)` | `InstrumentModel` | blend ∈ [0, 1] |
| `save_instrument(model, path)` | `None` | Pickle serialisation |
| `load_instrument(path)` | `InstrumentModel` | Load from pickle |

---

## See also

- [`docs/usage/audio_analysis.md`](audio_analysis.md) — pitch detection, T60, inharmonicity
- [`morphogen/stdlib/instrument_model.py`](/morphogen/stdlib/instrument_model.py) — source with full docstrings
- [`examples/showcase/07_physical_instrument.py`](/examples/showcase/07_physical_instrument.py) — end-to-end demo
- [`examples/audio/`](/examples/audio/) — audio domain examples
