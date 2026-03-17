---
title: "Audio Analysis Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - audio_analysis
  - audio
  - pitch-detection
  - spectral-analysis
---

# Audio Analysis

The `audio_analysis` domain extracts acoustic properties from audio signals:
pitch trajectories, harmonic decay envelopes, T60 reverberation time, and
inharmonicity. The results are plain NumPy arrays and dataclasses — feed
them directly into `instrument_model`, `audio`, or your own code.

## Quick start

```python
import numpy as np
from morphogen.stdlib.audio_analysis import (
    track_fundamental,
    track_partials,
    fit_exponential_decay,
    measure_t60,
    measure_inharmonicity,
)
```

---

## Recipe 1 — Extract pitch from a recording

`track_fundamental` returns a frame-by-frame f0 trajectory.

```python
# Load or synthesise a signal (mono, float64)
sample_rate = 44100
t = np.linspace(0, 2.0, sample_rate * 2)
signal = (
    np.sin(2 * np.pi * 220 * t) * np.exp(-t * 2)     # fundamental: A3
    + 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 3)
)

f0_trajectory = track_fundamental(signal, sample_rate, method="yin")
# → np.ndarray, shape (n_frames,), values in Hz (0 = unvoiced)

f0_median = np.median(f0_trajectory[f0_trajectory > 0])
print(f"Detected pitch: {f0_median:.1f} Hz")   # ≈ 220 Hz
```

**Method options:**
- `"autocorrelation"` — robust, fastest, default
- `"yin"` — more accurate for clean tones
- `"hps"` — harmonic product spectrum, useful for noisy signals

The f0 trajectory has `n_frames = (len(signal) - frame_size) // hop_size + 1`
frames. Default `frame_size=2048`, `hop_size=512` → one estimate per 512
samples (~11.6 ms at 44100 Hz).

---

## Recipe 2 — Measure decay and T60

How quickly does each harmonic fade? `fit_exponential_decay` fits
`A(t) = A₀ · exp(−d·t)` to each partial and returns the decay rate `d`
in units of 1/s.  `measure_t60` converts that to the standard −60 dB
decay time used in room acoustics.

```python
# Step 1: track harmonic amplitudes (needs f0 first)
harmonics = track_partials(signal, sample_rate, f0_trajectory, num_partials=8)
# → np.ndarray, shape (n_frames, num_partials), amplitude per harmonic per frame

# Step 2: fit exponential decay to each partial
decay_rates = fit_exponential_decay(harmonics, sample_rate)
# → np.ndarray, shape (num_partials,), rate in 1/s

# Step 3: convert to T60
for i, d in enumerate(decay_rates[:4]):
    t60 = measure_t60(float(d))
    print(f"  partial {i+1}: decay={d:.2f}/s  T60={t60:.2f}s")
```

Example output for a synthetic guitar pluck (220 Hz, decay ~2/s):
```
  partial 1: decay=3.83/s  T60=1.80s
  partial 2: decay=4.61/s  T60=1.50s
  partial 3: decay=5.45/s  T60=1.27s
  partial 4: decay=6.21/s  T60=1.11s
```

Higher partials decay faster — that's the characteristic tonal evolution
of a plucked string.

---

## Recipe 3 — Measure inharmonicity

Real strings are slightly stiffer than ideal, so partials sit slightly
above their harmonic positions: `fₙ = n·f₀·√(1 + B·n²)`. The `B`
coefficient is the inharmonicity parameter.

```python
f0 = float(np.median(f0_trajectory[f0_trajectory > 0]))
B = measure_inharmonicity(signal, sample_rate, f0, num_partials=10)
print(f"Inharmonicity B = {B:.6f}")
# typical guitar: B ≈ 0.0001–0.001
# piano bass strings: B ≈ 0.001–0.01
# near-ideal (synthetic): B ≈ 0
```

Inharmonicity is a key perceptual signature — pianos sound like pianos
partly because of this stretching. The `instrument_model` domain uses
this value when transposing synthesised notes.

---

## Recipe 4 — Feed results into instrument_model

The decay rates and f0 produced here flow directly into
`instrument_model.analyze_instrument` and the resulting
`InstrumentModel` object, giving you a reusable playable model.

```python
from morphogen.stdlib.instrument_model import analyze_instrument, synthesize_note, InstrumentType

model = analyze_instrument(
    signal,
    sample_rate,
    instrument_id="acoustic_guitar_A3",
    instrument_type=InstrumentType.MODAL_STRING,
    num_partials=20,
)

# Synthesise the same note at a different pitch
note_D4 = synthesize_note(model, pitch=293.66, velocity=0.9, duration=2.0)
```

See the full instrument_model workflow in
[`docs/usage/instrument_model.md`](instrument_model.md).

---

## Full operator reference

| Operator | Returns | Notes |
|----------|---------|-------|
| `track_fundamental(signal, sr, method, frame_size, hop_size)` | `np.ndarray (n_frames,)` | f0 in Hz; 0 = unvoiced frame |
| `track_partials(signal, sr, f0_trajectory, num_partials, frame_size, hop_size)` | `np.ndarray (n_frames, num_partials)` | Amplitude of each harmonic |
| `spectral_envelope(stft_result, smoothing_factor)` | `np.ndarray (n_frames, n_bins)` | Smoothed spectral shape |
| `analyze_modes(signal, sr, num_modes, freq_range)` | `ModalModel` | Frequency, amplitude, decay, phase per mode |
| `fit_exponential_decay(harmonics, sr, hop_size)` | `np.ndarray (num_partials,)` | Decay rate per partial (1/s) |
| `measure_t60(decay_rate)` | `float` | T60 in seconds; `inf` if decay_rate ≤ 0 |
| `measure_inharmonicity(signal, sr, f0, num_partials)` | `float` | Inharmonicity coefficient B |
| `deconvolve(signal, sr, cepstral_lifter)` | `(excitation, body_ir)` | Cepstral source-filter separation |
| `model_noise(signal, sr, num_bands)` | `NoiseModel` | Spectral + temporal noise envelope |

All operators are deterministic (tag: `repro` or `strict`). They take
plain NumPy arrays and return plain NumPy arrays or small dataclasses —
no hidden state.

---

## See also

- [`docs/usage/instrument_model.md`](instrument_model.md) — build and synthesise instrument models
- [`morphogen/stdlib/audio_analysis.py`](/morphogen/stdlib/audio_analysis.py) — source with full docstrings
- [`examples/showcase/07_physical_instrument.py`](/examples/showcase/07_physical_instrument.py) — end-to-end demo
