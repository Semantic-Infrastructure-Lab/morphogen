---
title: "Morphogen Canonical Cross-Domain Examples"
type: reference
beth_topics:
  - morphogen
  - cross-domain
  - audio
  - examples
---

# Canonical Cross-Domain Examples

Four examples that demonstrate Morphogen's core value proposition:
**type-safe, zero-glue composition across radically different domains.**

Each runs end-to-end, produces a WAV file, and prints what happened.

## Examples

### 01 — Physics → Audio

```bash
python examples/canonical/01_physics_to_audio.py
# → output/01_physics_to_audio.wav
```

Bouncing ball simulation sonified into percussion.
Each ground collision drives impact sound synthesis:
mass → pitch, velocity → amplitude, restitution → decay.

Domains: `rigidbody` → `audio`

---

### 02 — Circuit → Audio

```bash
python examples/canonical/02_circuit_to_audio.py
# → output/02_circuit_clean.wav
# → output/02_circuit_overdrive.wav
# → output/02_circuit_ts.wav
```

Guitar signal processed through two analog circuit models:
- **Op-amp overdrive**: gain + RC tone-control filter
- **Tube Screamer**: antiparallel diodes in feedback → soft clipping

Kirchhoff equations solved sample-by-sample at 48kHz.
Newton-Raphson nonlinear solver activated by the diodes.

Domains: `audio` ↔ `circuit` → `audio`

---

### 03 — Fluid → Acoustics → Audio

```bash
python examples/canonical/03_fluid_to_sound.py
# → output/03_fluid_to_sound.wav
```

3-domain pipeline: Navier-Stokes vortex shedding drives acoustic wave
propagation; two virtual microphones sample the acoustic field and
produce a stereo WAV.

Domains: `field` (fluid) → `acoustics` → `audio`

---

### 04 — Analysis → Instrument Model → Synthesis

```bash
python examples/canonical/04_analysis_to_instrument.py
# → output/04_analysis_to_instrument.wav
```

Extracts acoustic parameters from a reference signal, builds a reusable
instrument model, then synthesises a melody using the learned timbre.

`audio_analysis` answers "how does this instrument sound?" — decay rates,
inharmonicity, harmonic envelope. `instrument_model` applies that sonic
fingerprint at any pitch or velocity.

Domains: `audio_analysis` → `instrument_model` → `audio`

---

## Why This Matters

Traditional approaches require NumPy → SciPy → librosa → PyBullet
with manual data marshalling between each tool. Morphogen uses typed
domain interfaces so `Field2D`, `AudioBuffer`, and `RigidBody` objects
flow directly between domains without conversion.

See also:
- [`examples/cross_domain/`](../cross_domain/) — more cross-domain examples
- [`docs/STRATEGY.md`](../../docs/STRATEGY.md) — v1.0 strategic context
- [`docs/PITCH.md`](../../docs/PITCH.md) — 2-minute overview
