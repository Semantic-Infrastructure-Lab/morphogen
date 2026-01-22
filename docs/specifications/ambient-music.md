# Ambient Music & Generative Audio Domain Specification

**Version:** 1.0 (Draft)
**Status:** Proposed
**Date:** 2025-11-16
**Related ADR:** [009-ambient-music-generative-domains.md](../adr/009-ambient-music-generative-domains.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Philosophy & Approach](#philosophy--approach)
3. [Domain Structure](#domain-structure)
4. [Core Types](#core-types)
5. [Operator Catalog](#operator-catalog)
   - [5.1 Spectral Domain](#51-spectral-domain)
   - [5.2 Ambience Domain](#52-ambience-domain)
   - [5.3 Synthesis Domain](#53-synthesis-domain)
   - [5.4 Composition Domain](#54-composition-domain)
6. [Ambient Pipeline Patterns](#ambient-pipeline-patterns)
7. [Cross-Domain Integration](#cross-domain-integration)
8. [MLIR Lowering Strategy](#mlir-lowering-strategy)
9. [Examples](#examples)
10. [Testing Strategy](#testing-strategy)
11. [Implementation Phases](#implementation-phases)
12. [References](#references)

---

## Overview

This specification defines four new Morphogen domains for ambient and generative music:

| Domain | Purpose | Operator Count | Status |
|--------|---------|----------------|--------|
| **Spectral** | Frequency-domain audio manipulation | 15 | Proposed |
| **Ambience** | High-level ambient primitives (drones, clouds) | 25 | Proposed |
| **Synthesis** | Modular-style DSP routing | 30 | Proposed |
| **Composition** | Generative pattern generation | 20 | Proposed |
| **Total** | | **90** | |

### Why Ambient Music in Morphogen?

Ambient music is a perfect match for Morphogen's architecture:

**Ambient music is:**
- **Procedural** — Parameter-driven, not performance-driven
- **Generative** — Evolves via simple rules (CA, Markov chains)
- **Layered** — Modular composition of simple transforms
- **Simulation-friendly** — Time-evolving fields, agent systems
- **GPU-accelerable** — Parallel granular synthesis, convolution

**Morphogen provides:**
- Declarative operator composition
- Cross-domain integration (physics → audio, CA → sequencing)
- GPU acceleration via MLIR
- Deterministic reproducibility
- Multi-rate scheduling (hours-long evolution)

**No existing tool unifies these capabilities.**

---

## Philosophy & Approach

### Design Principles

1. **Composability over Monoliths**
   - Small, focused operators that combine
   - Not "Eno Pad" preset, but `drone.harmonic` + `spectral.blur` + `granular.cloud`

2. **Cross-Domain by Default**
   - Physics simulations modulate audio parameters
   - CA systems drive sequencing
   - Fractals influence harmonic evolution

3. **GPU-First Implementation**
   - Granular synthesis: 1000s of parallel grains
   - Spectral processing: GPU FFT convolution
   - Additive synthesis: Massive sinusoid banks

4. **Deterministic Generative Music**
   - Same seed → same evolution (reproducible compositions)
   - Strict/repro/live profiles for quality/performance trade-offs

5. **Multi-Hour Time Scales**
   - Slow LFOs (0.0001 Hz, hour-long periods)
   - Drift modulators (evolve over minutes)
   - Long-form pattern generators

### Ambient Music Paradigm

Traditional music: **Performance → Recording**
Generative music: **Rules → Evolution → Recording**

Morphogen model:
```
Parameters + Seed + Rules → Operators → GPU Execution → Audio Output
                                      ↓
                           Deterministic (same seed = same output)
```

---

## Domain Structure

### Layer Hierarchy

```
┌─────────────────────────────────────────────────┐
│  Layer 9: Composition                           │
│  (Markov, CA sequencers, pattern generators)    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Layer 8: Ambience                              │
│  (Drones, granular clouds, textures)            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Layer 7: Spectral                              │
│  (Frequency-domain manipulation)                │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Layer 6: Synthesis (Enhanced)                  │
│  (Modular DSP routing, oscillators, filters)    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Existing Morphogen Domains                         │
│  Audio, Transform, Stochastic, Emergence        │
└─────────────────────────────────────────────────┘
```

### Dependencies

**Spectral Domain requires:**
- `morphogen.transform` (FFT, STFT, iFFT, iSTFT)
- `morphogen.audio` (oscillators, filters)
- `Field2D<complex>` type

**Ambience Domain requires:**
- `morphogen.spectral` (harmonic nebula, spectral blur)
- `morphogen.audio` (filters, effects)
- `morphogen.stochastic` (noise, random walks)

**Synthesis Domain requires:**
- `morphogen.audio` (base oscillators, filters)
- `morphogen.event` (gate/trigger timing)

**Composition Domain requires:**
- `morphogen.emergence` (CA, swarms)
- `morphogen.stochastic` (Markov, Poisson)
- `morphogen.event` (note timing)

---

## Core Types

### Spectral Domain Types

```morphogen
// Frequency-domain field (complex values)
type SpectralField = Field2D<complex, (freq: Hz, time: s)>

// Real-valued spectrogram (magnitude)
type Spectrogram = Field2D<f32, (freq: Hz, time: s)>

// Harmonic series representation
type HarmonicSeries = Field1D<f32, freq: Hz> {
  fundamental: f32[Hz],
  partials: List<Harmonic>
}

type Harmonic = {
  frequency: f32[Hz],
  amplitude: f32,
  phase: f32[rad]
}
```

### Ambience Domain Types

```morphogen
// Drone configuration
type Drone = {
  fundamental: Ctl[Hz],
  texture: Enum["smooth", "shimmer", "rough", "nebula"],
  drift: Ctl[cents/s],        // Pitch drift rate
  spread: Ctl[ratio],         // Harmonic spacing
  brightness: Ctl[0..1]       // Spectral centroid
}

// Granular cloud parameters
type GranularCloud = {
  source: Sig,                // Source audio
  density: Ctl[grains/s],     // Grain trigger rate
  grain_size: Ctl[ms],        // Grain duration
  pitch_shift: Ctl[cents],    // Pitch transposition
  randomness: Ctl[ratio],     // Position/size randomness
  freeze: bool                // Freeze playback position
}

// Texture descriptor
type Texture = {
  spectral_range: (f32[Hz], f32[Hz]),
  evolution_rate: Ctl[Hz],    // Evolution speed
  complexity: Ctl[0..1],      // Harmonic density
  seed: u64                   // RNG seed
}
```

### Synthesis Domain Types

```morphogen
// Synthesis patch (declarative graph)
type Patch = Graph<SynthOp> {
  inputs: Map<String, Sig|Ctl>,
  outputs: Map<String, Sig|Ctl>,
  params: Map<String, f32>
}

// Control voltage (modulation)
type CV = Ctl  // Alias for clarity

// Gate/trigger events
type Gate = Evt<void>
type Trigger = Evt<f32>  // Trigger with velocity
```

### Composition Domain Types

```morphogen
// Markov chain state transition matrix
type MarkovMatrix = Field2D<f32, (state_from: int, state_to: int)> {
  validate: sum(row) == 1.0  // Probabilities sum to 1
}

// Cellular automaton rule
type CARule = {
  rule: Enum["life", "rule30", "rule110", "lenia", "cyclic", "brian"],
  params: Map<String, f32>
}

// Musical pattern (event stream)
type Pattern = Evt<Note>

// Swarm state (for composition)
type SwarmState = Agents {
  position_to_pitch: (Vec2) -> f32[Hz],
  density_to_param: (f32) -> f32
}
```

---

## Operator Catalog

### 5.1 Spectral Domain

**Registry:** `morphogen.audio.spectral.*`

#### Spectral Manipulation Operators

**`spectral.blur(spectrogram: Spectrogram, bandwidth: f32[Hz]) -> Spectrogram`**

Blur spectrogram in frequency dimension (spectral smoothing).

```json
{
  "name": "spectral.blur",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "spectrogram": "Spectrogram",
    "bandwidth": "f32[Hz]"
  },
  "params": {
    "kernel": "gaussian | box",
    "normalize": "bool"
  },
  "output": "Spectrogram",
  "determinism": "repro",
  "gpu_accelerated": true,
  "description": "Gaussian blur in frequency domain for smooth spectral textures"
}
```

**Example:**
```morphogen
// Create smooth pad from noise
noise_sig = noise(seed=42, type="pink")
spec = stft(noise_sig, window_size=2048)
blurred = spectral.blur(spec, bandwidth=100Hz)
smooth_pad = istft(blurred)
```

---

**`spectral.morph(spec_a: Spectrogram, spec_b: Spectrogram, mix: Ctl) -> Spectrogram`**

Crossfade between two spectrograms in frequency domain.

```json
{
  "name": "spectral.morph",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "spec_a": "Spectrogram",
    "spec_b": "Spectrogram",
    "mix": "Ctl[0..1]"
  },
  "params": {
    "interpolation": "linear | logarithmic",
    "phase_mode": "preserve | interpolate | randomize"
  },
  "output": "Spectrogram",
  "determinism": "repro",
  "gpu_accelerated": true,
  "description": "Morph between two spectral textures"
}
```

---

**`spectral.freeze(spectrogram: Spectrogram, freeze_range: (f32[Hz], f32[Hz])) -> Spectrogram`**

Freeze spectral content in specified frequency range.

```json
{
  "name": "spectral.freeze",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "spectrogram": "Spectrogram",
    "freeze_range": "(f32[Hz], f32[Hz])"
  },
  "params": {
    "freeze_frame": "int | 'current'",
    "crossfade_time": "f32[s]"
  },
  "output": "Spectrogram",
  "determinism": "repro",
  "description": "Freeze spectral content (like Mutable Instruments Clouds)"
}
```

---

#### Harmonic Processing Operators

**`harmonic.nebula(fundamental: f32[Hz], spread: Ctl, density: int) -> HarmonicSeries`**

Generate harmonic nebula (distributed harmonic cloud).

```json
{
  "name": "harmonic.nebula",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "fundamental": "f32[Hz]",
    "spread": "Ctl[0..1]",
    "density": "int"
  },
  "params": {
    "distribution": "random | clustered | spread",
    "inharmonicity": "f32[0..1]",
    "seed": "u64"
  },
  "output": "HarmonicSeries",
  "determinism": "strict",
  "gpu_accelerated": true,
  "description": "Distributed harmonic cloud for ambient pads"
}
```

**Example:**
```morphogen
// Create shimmer pad with 64 distributed harmonics
nebula = harmonic.nebula(
  fundamental = 110Hz,
  spread = 0.3,
  density = 64
)
pad = additive.resynth(nebula) |> reverb(0.4)
```

---

**`harmonic.drift(series: HarmonicSeries, rate: Ctl[Hz], depth: Ctl[cents]) -> HarmonicSeries`**

Apply slow pitch drift to harmonic series.

```json
{
  "name": "harmonic.drift",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "series": "HarmonicSeries",
    "rate": "Ctl[Hz]",
    "depth": "Ctl[cents]"
  },
  "params": {
    "mode": "brownian | sine | perlin",
    "seed": "u64"
  },
  "output": "HarmonicSeries",
  "determinism": "strict",
  "description": "Slow harmonic drift for evolving textures"
}
```

---

#### Vocoding & Spectral Filtering

**`vocode(carrier: Sig, modulator: Sig, bands: int) -> Sig`**

Classic vocoder effect (carrier shaped by modulator spectrum).

```json
{
  "name": "vocode",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "carrier": "Sig",
    "modulator": "Sig",
    "bands": "int"
  },
  "params": {
    "band_distribution": "linear | logarithmic | bark",
    "attack": "f32[ms]",
    "release": "f32[ms]"
  },
  "output": "Sig",
  "determinism": "repro",
  "gpu_accelerated": true,
  "description": "Spectral envelope transfer (vocoding)"
}
```

---

**`spectral.filter(sig: Sig, envelope: Spectrogram) -> Sig`**

Apply arbitrary spectral envelope as filter.

```json
{
  "name": "spectral.filter",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "sig": "Sig",
    "envelope": "Spectrogram"
  },
  "params": {
    "interpolation": "linear | cubic"
  },
  "output": "Sig",
  "determinism": "repro",
  "gpu_accelerated": true,
  "description": "Arbitrary spectral filtering"
}
```

---

#### Resynthesis Operators

**`additive.resynth(harmonics: Field2D<f32>, phases: Field2D<f32>) -> Sig`**

Resynthesize audio from time-varying harmonic representation.

```json
{
  "name": "additive.resynth",
  "layer": 7,
  "category": "spectral",
  "inputs": {
    "harmonics": "Field2D<f32, (partial, time)>",
    "phases": "Field2D<f32, (partial, time)>"
  },
  "params": {
    "interpolation": "linear | cubic"
  },
  "output": "Sig",
  "determinism": "strict",
  "gpu_accelerated": true,
  "description": "Additive resynthesis from harmonic fields"
}
```

---

### 5.2 Ambience Domain

**Registry:** `morphogen.ambient.*`

#### Drone Generators

**`drone.harmonic(fundamental: f32[Hz], spread: Ctl, shimmer: Ctl) -> Stereo`**

Harmonic drone generator with shimmer.

```json
{
  "name": "drone.harmonic",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "fundamental": "f32[Hz]",
    "spread": "Ctl[0..1]",
    "shimmer": "Ctl[0..1]"
  },
  "params": {
    "num_harmonics": "int",
    "stereo_width": "f32[0..1]",
    "texture": "smooth | rough | shimmer"
  },
  "output": "Stereo",
  "determinism": "repro",
  "gpu_accelerated": true,
  "description": "Rich harmonic drone with optional shimmer modulation"
}
```

**Example:**
```morphogen
// Slowly evolving harmonic drone
drift_lfo = orbit.lfo(period_hours=1.0)
drone = drone.harmonic(
  fundamental = 55Hz,
  spread = 0.2 + drift_lfo * 0.1,
  shimmer = 0.3
)
output = drone |> reverb(0.5, decay=4.0)
```

---

**`drone.subharmonic(root: f32[Hz], divisions: List<int>) -> Sig`**

Subharmonic drone (root / divisions).

```json
{
  "name": "drone.subharmonic",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "root": "f32[Hz]",
    "divisions": "List<int>"
  },
  "params": {
    "amplitude_curve": "linear | 1/f | 1/f^2",
    "phase_randomize": "bool"
  },
  "output": "Sig",
  "determinism": "strict",
  "description": "Deep subharmonic bass drone"
}
```

---

**`drone.pad(harmonics: List<int>, bandwidth: f32[Hz], modulation: Ctl) -> Stereo`**

Ambient pad with spectral bandwidth.

```json
{
  "name": "drone.pad",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "harmonics": "List<int>",
    "bandwidth": "f32[Hz]",
    "modulation": "Ctl[Hz]"
  },
  "params": {
    "fundamental": "f32[Hz]",
    "stereo_detune": "f32[cents]"
  },
  "output": "Stereo",
  "determinism": "repro",
  "description": "Lush ambient pad with spectral spread"
}
```

---

#### Granular Synthesis Operators

**`granular.cloud(source: Sig, density: Ctl, grain_size: Ctl, pitch_shift: Ctl) -> Sig`**

Dense granular cloud synthesis.

```json
{
  "name": "granular.cloud",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "source": "Sig",
    "density": "Ctl[grains/s]",
    "grain_size": "Ctl[ms]",
    "pitch_shift": "Ctl[cents]"
  },
  "params": {
    "window": "hann | hamming | blackman",
    "randomness": "f32[0..1]",
    "seed": "u64"
  },
  "output": "Sig",
  "determinism": "strict",
  "gpu_accelerated": true,
  "description": "Granular cloud synthesis (Mutable Instruments Clouds style)"
}
```

**Example:**
```morphogen
// Granular cloud driven by CA density
life_grid = ca.life(size=64, initial=random_grid(seed=123))
ca_density = field.mean(life_grid) * 100  // CA alive cells → grain density

cloud = granular.cloud(
  source = drone.harmonic(110Hz, spread=0.2),
  density = ca_density,
  grain_size = 80ms,
  pitch_shift = 0
)
```

---

**`granular.freeze(source: Sig, freeze_position: Ctl) -> Sig`**

Freeze granular playback position.

```json
{
  "name": "granular.freeze",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "source": "Sig",
    "freeze_position": "Ctl[0..1]"
  },
  "params": {
    "grain_size": "f32[ms]",
    "density": "f32[grains/s]",
    "randomness": "f32[0..1]"
  },
  "output": "Sig",
  "determinism": "strict",
  "description": "Freeze playback position and granulate"
}
```

---

**`granular.reverse_cloud(source: Sig, density: Ctl, grain_size: Ctl) -> Sig`**

Reverse granular cloud (grains played backward).

```json
{
  "name": "granular.reverse_cloud",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "source": "Sig",
    "density": "Ctl[grains/s]",
    "grain_size": "Ctl[ms]"
  },
  "params": {
    "pitch_shift": "f32[cents]",
    "randomness": "f32[0..1]"
  },
  "output": "Sig",
  "determinism": "strict",
  "description": "Reverse granular synthesis for ethereal textures"
}
```

---

#### Long-Form Modulators

**`drift.noise(period_minutes: f32, depth: Ctl) -> Ctl`**

Slow drift modulator (for multi-minute evolution).

```json
{
  "name": "drift.noise",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "period_minutes": "f32",
    "depth": "Ctl"
  },
  "params": {
    "mode": "perlin | simplex | brownian",
    "seed": "u64"
  },
  "output": "Ctl",
  "determinism": "strict",
  "description": "Ultra-slow modulation for ambient evolution (0.0001 Hz range)"
}
```

**Example:**
```morphogen
// Reverb decay drifts over 20 minutes
decay_drift = drift.noise(period_minutes=20, depth=1.5) + 2.5
ambient_pad |> reverb(mix=0.4, decay=decay_drift)
```

---

**`orbit.lfo(period_hours: f32, orbit_shape: String) -> Ctl`**

Ultra-slow orbital LFO (hour-long periods).

```json
{
  "name": "orbit.lfo",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "period_hours": "f32",
    "orbit_shape": "circle | ellipse | lissajous"
  },
  "params": {
    "phase_offset": "f32[rad]",
    "eccentricity": "f32[0..1]"
  },
  "output": "Ctl",
  "determinism": "strict",
  "description": "Hour-scale orbital modulation"
}
```

---

**`slow.random_walk(range: (f32, f32), smoothing: Ctl) -> Ctl`**

Brownian random walk with smoothing.

```json
{
  "name": "slow.random_walk",
  "layer": 8,
  "category": "ambience",
  "inputs": {
    "range": "(f32, f32)",
    "smoothing": "Ctl[0..1]"
  },
  "params": {
    "step_size": "f32",
    "seed": "u64"
  },
  "output": "Ctl",
  "determinism": "strict",
  "description": "Smooth random walk for parameter drift"
}
```

---

### 5.3 Synthesis Domain

**Registry:** `morphogen.synthesis.*`

#### Enhanced Oscillators

**`vco(freq: Ctl, waveform: String, sync: Gate) -> Sig`**

Voltage-controlled oscillator with sync.

```json
{
  "name": "vco",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "freq": "Ctl[Hz]",
    "waveform": "saw | square | triangle | sine",
    "sync": "Gate"
  },
  "params": {
    "pwm": "f32[0..1]",  // For square wave
    "blep": "bool"       // Band-limited step
  },
  "output": "Sig",
  "determinism": "strict",
  "description": "Voltage-controlled oscillator with hard sync"
}
```

---

**`wavetable(table: Field1D, freq: Ctl, morph: Ctl) -> Sig`**

Wavetable oscillator with morphing.

```json
{
  "name": "wavetable",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "table": "Field1D<f32>",
    "freq": "Ctl[Hz]",
    "morph": "Ctl[0..1]"
  },
  "params": {
    "interpolation": "linear | cubic | none"
  },
  "output": "Sig",
  "determinism": "strict",
  "gpu_accelerated": true,
  "description": "Wavetable synthesis with morphing"
}
```

---

**`fm(carrier_freq: Ctl, mod_freq: Ctl, mod_index: Ctl) -> Sig`**

FM synthesis operator.

```json
{
  "name": "fm",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "carrier_freq": "Ctl[Hz]",
    "mod_freq": "Ctl[Hz]",
    "mod_index": "Ctl"
  },
  "params": {
    "algorithm": "simple | dx7"
  },
  "output": "Sig",
  "determinism": "strict",
  "description": "FM synthesis (carrier modulated by modulator)"
}
```

---

#### Advanced Filters

**`multimode(sig: Sig, mode: String, cutoff: Ctl, resonance: Ctl) -> Sig`**

Multimode filter (LP/HP/BP/Notch).

```json
{
  "name": "multimode",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "sig": "Sig",
    "mode": "lp | hp | bp | notch",
    "cutoff": "Ctl[Hz]",
    "resonance": "Ctl[0..1]"
  },
  "params": {
    "drive": "f32",
    "slope": "12dB | 24dB | 48dB"
  },
  "output": "Sig",
  "determinism": "repro",
  "description": "State-variable multimode filter"
}
```

---

**`formant(sig: Sig, vowel: String, morph: Ctl) -> Sig`**

Formant filter (vowel shaping).

```json
{
  "name": "formant",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "sig": "Sig",
    "vowel": "a | e | i | o | u",
    "morph": "Ctl[0..1]"
  },
  "params": {
    "interpolate": "bool"
  },
  "output": "Sig",
  "determinism": "repro",
  "description": "Formant filter for vowel-like timbres"
}
```

---

**`comb(sig: Sig, delay_time: Ctl, feedback: Ctl, damping: Ctl) -> Sig`**

Comb filter (feedback delay line).

```json
{
  "name": "comb",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "sig": "Sig",
    "delay_time": "Ctl[ms]",
    "feedback": "Ctl[0..1]",
    "damping": "Ctl[0..1]"
  },
  "params": {
    "interpolation": "linear | allpass"
  },
  "output": "Sig",
  "determinism": "repro",
  "description": "Comb filter for metallic/resonant timbres"
}
```

---

#### Modulation Sources

**`lfo(rate: Ctl, waveform: String) -> CV`**

Low-frequency oscillator.

```json
{
  "name": "lfo",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "rate": "Ctl[Hz]",
    "waveform": "sine | triangle | saw | square | random"
  },
  "params": {
    "phase_offset": "f32[rad]",
    "unipolar": "bool"  // 0..1 vs -1..1
  },
  "output": "CV",
  "determinism": "strict",
  "description": "Low-frequency oscillator for modulation"
}
```

---

**`envelope.follower(sig: Sig, attack: f32[ms], release: f32[ms]) -> CV`**

Envelope follower (amplitude tracking).

```json
{
  "name": "envelope.follower",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "sig": "Sig",
    "attack": "f32[ms]",
    "release": "f32[ms]"
  },
  "params": {
    "mode": "peak | rms"
  },
  "output": "CV",
  "determinism": "repro",
  "description": "Envelope follower for dynamic modulation"
}
```

---

**`sample_and_hold(sig: Sig, trigger: Gate) -> CV`**

Sample-and-hold (random stepped modulation).

```json
{
  "name": "sample_and_hold",
  "layer": 6,
  "category": "synthesis",
  "inputs": {
    "sig": "Sig",
    "trigger": "Gate"
  },
  "params": {
    "slew": "f32[ms]"  // Optional slew limiting
  },
  "output": "CV",
  "determinism": "strict",
  "description": "Sample-and-hold for stepped random modulation"
}
```

---

### 5.4 Composition Domain

**Registry:** `morphogen.composition.*`

#### Markov Chain Sequencing

**`markov.sequence(matrix: MarkovMatrix, initial_state: int, tempo: Ctl) -> Evt<Note>`**

Markov chain note sequencer.

```json
{
  "name": "markov.sequence",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "matrix": "MarkovMatrix",
    "initial_state": "int",
    "tempo": "Ctl[BPM]"
  },
  "params": {
    "seed": "u64",
    "note_mapping": "Map<int, Note>"
  },
  "output": "Evt<Note>",
  "determinism": "strict",
  "description": "Markov chain melodic sequencer"
}
```

**Example:**
```morphogen
// Simple 3-state Markov melody
matrix = [
  [0.5, 0.3, 0.2],  // From state 0
  [0.3, 0.4, 0.3],  // From state 1
  [0.2, 0.3, 0.5]   // From state 2
]
note_map = {
  0: note("C4", vel=0.8),
  1: note("E4", vel=0.7),
  2: note("G4", vel=0.6)
}

melody = markov.sequence(
  matrix = matrix,
  initial_state = 0,
  tempo = 80BPM
)
```

---

**`markov.parameter_walk(matrix: MarkovMatrix, value_mapping: Map<int, f32>) -> Ctl`**

Markov chain for parameter evolution.

```json
{
  "name": "markov.parameter_walk",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "matrix": "MarkovMatrix",
    "value_mapping": "Map<int, f32>"
  },
  "params": {
    "seed": "u64",
    "transition_rate": "Ctl[Hz]"
  },
  "output": "Ctl",
  "determinism": "strict",
  "description": "Markov chain for parameter modulation"
}
```

---

#### CA-Based Sequencing

**`ca.sequencer(ca_rule: CARule, grid_size: int, mapping: Fn) -> Evt<Note>`**

Cellular automaton-driven sequencer.

```json
{
  "name": "ca.sequencer",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "ca_rule": "CARule",
    "grid_size": "int",
    "mapping": "Fn((x: int, y: int) -> Note)"
  },
  "params": {
    "scan_rate": "Ctl[Hz]",
    "scan_pattern": "left_to_right | top_to_bottom | spiral",
    "seed": "u64"
  },
  "output": "Evt<Note>",
  "determinism": "strict",
  "description": "CA-driven melodic sequencer"
}
```

**Example:**
```morphogen
// Conway's Life triggers notes
life_rule = ca_rule("life")
notes = ca.sequencer(
  ca_rule = life_rule,
  grid_size = 16,
  mapping = (x, y) => {
    // Grid position → pitch
    pitch = 220Hz * (1 + (x + y) / 32.0)
    note(pitch, vel=0.7, dur=0.2s)
  }
)
```

---

**`life.triggers(grid: Field2D<bool>, birth_note: Note, death_note: Note) -> Evt<Note>`**

Trigger notes on cell birth/death events.

```json
{
  "name": "life.triggers",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "grid": "Field2D<bool>",
    "birth_note": "Note",
    "death_note": "Note"
  },
  "params": {
    "velocity_from_neighbors": "bool"
  },
  "output": "Evt<Note>",
  "determinism": "strict",
  "description": "Trigger notes on CA cell birth/death"
}
```

---

**`lenia.modulation(field: Field2D<f32>, param_name: String) -> Ctl`**

Use Lenia field density as modulation source.

```json
{
  "name": "lenia.modulation",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "field": "Field2D<f32>",
    "param_name": "String"
  },
  "params": {
    "aggregation": "mean | max | sum",
    "smoothing": "f32[0..1]"
  },
  "output": "Ctl",
  "determinism": "strict",
  "description": "Lenia field → parameter modulation"
}
```

---

#### Stochastic Generators

**`poisson.trigger(rate: Ctl[Hz], note_pool: List<Note>) -> Evt<Note>`**

Poisson process note triggering.

```json
{
  "name": "poisson.trigger",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "rate": "Ctl[Hz]",
    "note_pool": "List<Note>"
  },
  "params": {
    "seed": "u64",
    "selection": "random | sequential | weighted"
  },
  "output": "Evt<Note>",
  "determinism": "strict",
  "description": "Stochastic note triggering (Poisson process)"
}
```

---

**`brownian.melody(start_note: Note, step_size: f32[cents], bounds: (f32[Hz], f32[Hz])) -> Evt<Note>`**

Brownian motion melodic generator.

```json
{
  "name": "brownian.melody",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "start_note": "Note",
    "step_size": "f32[cents]",
    "bounds": "(f32[Hz], f32[Hz])"
  },
  "params": {
    "seed": "u64",
    "tempo": "Ctl[BPM]"
  },
  "output": "Evt<Note>",
  "determinism": "strict",
  "description": "Brownian walk melody generator"
}
```

---

#### Pattern Generators

**`euclidean.rhythm(steps: int, pulses: int, rotation: int) -> Evt<Gate>`**

Euclidean rhythm generator.

```json
{
  "name": "euclidean.rhythm",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "steps": "int",
    "pulses": "int",
    "rotation": "int"
  },
  "params": {
    "tempo": "Ctl[BPM]"
  },
  "output": "Evt<Gate>",
  "determinism": "strict",
  "description": "Euclidean rhythm pattern (Bjorklund algorithm)"
}
```

**Example:**
```morphogen
// Classic 3-2 son clave rhythm
rhythm = euclidean.rhythm(steps=16, pulses=5, rotation=0)
```

---

**`fractal.timing(iterations: int, ratio: f32, base_duration: f32[s]) -> Evt<Gate>`**

Fractal/recursive timing patterns.

```json
{
  "name": "fractal.timing",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "iterations": "int",
    "ratio": "f32",
    "base_duration": "f32[s]"
  },
  "params": {
    "pattern": "binary | ternary | fibonacci"
  },
  "output": "Evt<Gate>",
  "determinism": "strict",
  "description": "Fractal/recursive timing generator"
}
```

---

#### Swarm-Based Composition

**`swarm.pitch_map(agents: Agents, pitch_range: (f32[Hz], f32[Hz]), quantize_to_scale: List<int>) -> Evt<Note>`**

Map swarm agent positions to pitches.

```json
{
  "name": "swarm.pitch_map",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "agents": "Agents",
    "pitch_range": "(f32[Hz], f32[Hz])",
    "quantize_to_scale": "List<int>"
  },
  "params": {
    "trigger_on": "movement | proximity | velocity_change",
    "min_interval": "f32[ms]"
  },
  "output": "Evt<Note>",
  "determinism": "strict",
  "description": "Swarm positions → melodic material"
}
```

---

**`swarm.density_modulation(agents: Agents, param_name: String) -> Ctl`**

Use swarm density as modulation source.

```json
{
  "name": "swarm.density_modulation",
  "layer": 9,
  "category": "composition",
  "inputs": {
    "agents": "Agents",
    "param_name": "String"
  },
  "params": {
    "region_size": "f32",
    "aggregation": "mean | max | sum"
  },
  "output": "Ctl",
  "determinism": "strict",
  "description": "Swarm density → parameter modulation"
}
```

---

## Ambient Pipeline Patterns

### Pattern 1: Spectral Drone Evolution

```morphogen
// Generate harmonic nebula
nebula = harmonic.nebula(fundamental=110Hz, spread=0.3, density=64)

// Apply slow drift
drifted = harmonic.drift(nebula, rate=0.02Hz, depth=15cents)

// Resynthesize and process
pad = additive.resynth(drifted)
blurred = pad |> spectral.blur(bandwidth=80Hz)
output = blurred |> reverb(0.5, decay=6.0)
```

### Pattern 2: CA-Driven Granular Texture

```morphogen
// Cellular automaton drives grain density
life_grid = ca.life(size=64, seed=42)
density = field.mean(life_grid) * 150  // 0-150 grains/s

// Granular cloud with CA modulation
source = drone.harmonic(220Hz, spread=0.2)
grains = granular.cloud(
  source = source,
  density = density,
  grain_size = 60ms,
  pitch_shift = 0
)

output = grains |> stereo.width(1.5) |> reverb(0.3)
```

### Pattern 3: Multi-Hour Evolving Composition

```morphogen
// Ultra-slow modulators
harmonic_drift = orbit.lfo(period_hours=2.0)
reverb_evolution = drift.noise(period_minutes=30, depth=2.0) + 3.0

// Base drone with drift
drone = drone.subharmonic(
  root = 55Hz,
  divisions = [1, 2, 3, 4, 5]
)

// Granular layer
texture = granular.freeze(
  source = drone,
  freeze_position = 0.5 + harmonic_drift * 0.2
)

// Final mix with evolving reverb
output = mix(
  drone * 0.6,
  texture * 0.4
) |> reverb(0.4, decay=reverb_evolution)
```

### Pattern 4: Swarm → Melody + Modulation

```morphogen
// Boid swarm
boids = agents.boids(num=30, bounds=100, cohesion=0.5)

// Swarm positions → notes
notes = swarm.pitch_map(
  boids,
  pitch_range = (110Hz, 880Hz),
  quantize_to_scale = [0,2,4,5,7,9,11]  // Major scale
)

// Swarm density → filter cutoff
cutoff = swarm.density_modulation(boids, "filter_cutoff")
cutoff_scaled = cutoff * 2000Hz + 500Hz

// Synthesize
voice = (note: Note) => {
  sine(note.pitch)
  |> lpf(cutoff_scaled)
  |> adsr(5ms, 100ms, 0.5, 300ms)
}

output = spawn(notes, voice, max_voices=8) |> reverb(0.25)
```

---

## Cross-Domain Integration

### Physics → Audio

**Fluid vorticity drives granular density:**
```morphogen
fluid = navier_stokes.solve(...)
vorticity = field.curl(fluid.velocity)
grain_density = vorticity.max() * 100

grains = granular.cloud(
  source = drone.harmonic(110Hz),
  density = grain_density,
  grain_size = 50ms
)
```

### Fractals → Harmonic Evolution

**Mandelbrot escape time modulates harmonic spread:**
```morphogen
mandelbrot = fractal.iterate(zoom_center, zoom_rate)
depth = mandelbrot.escape_time.mean() / 100.0

pad = drone.harmonic(
  fundamental = 55Hz,
  spread = depth,
  shimmer = 0.3
)
```

### Reaction-Diffusion → Spectral Filtering

**R-D pattern shapes spectral envelope:**
```morphogen
rd = reaction_diffusion.solve(...)
pattern = rd.activator_field

// Convert RD field to spectral envelope
spectral_env = field.to_spectrogram(pattern)

// Filter drone through RD pattern
drone = drone.subharmonic(110Hz, divisions=[1,2,3,4,5])
filtered = spectral.filter(drone, spectral_env)
```

---

## MLIR Lowering Strategy

### Granular Synthesis Lowering

**High-level operator:**
```morphogen
grains = granular.cloud(source, density=50, grain_size=80ms)
```

**MLIR lowering:**
```mlir
// Grain spawning (parallel)
%grain_triggers = morphogen.poisson_process %density, %seed : !morphogen.evt<f32>

// Per-grain processing (GPU parallelizable)
%grains = linalg.generic {
  iterator_types = ["parallel"]
} ins(%source, %grain_triggers, %grain_size) outs(%output) {
^bb0(%src: f32, %trigger: f32, %size: f32):
  // Window function (Hann)
  %phase = ...
  %window = math.cos %phase : f32
  %windowed = arith.mulf %src, %window : f32
  linalg.yield %windowed : f32
}

// Sum all grains
%output = linalg.reduce add ins(%grains) outs(%mix)
```

**GPU acceleration:** Each grain computed independently → massive parallelism.

---

### Spectral Blur Lowering

**High-level operator:**
```morphogen
blurred = spectral.blur(spectrogram, bandwidth=100Hz)
```

**MLIR lowering:**
```mlir
// Convert to frequency domain
%spec = morphogen.transform.stft %input : !morphogen.stream<f32> -> !morphogen.field<complex>

// Gaussian convolution in frequency dimension (GPU)
%kernel = morphogen.spectral.gaussian_kernel %bandwidth : !morphogen.field<f32>
%blurred = linalg.conv_2d ins(%spec, %kernel) outs(%output)
  : (!morphogen.field<complex>, !morphogen.field<f32>) -> !morphogen.field<complex>

// Convert back to time domain
%result = morphogen.transform.istft %blurred : !morphogen.field<complex> -> !morphogen.stream<f32>
```

**GPU acceleration:** FFT and convolution both GPU-optimized via cuFFT/rocFFT.

---

### Additive Synthesis Lowering

**High-level operator:**
```morphogen
sig = additive.resynth(harmonics, phases)
```

**MLIR lowering:**
```mlir
// Parallel sinusoid bank (GPU)
%time = morphogen.stream.time : !morphogen.stream<f32>

%sines = linalg.generic {
  iterator_types = ["parallel", "reduction"]
} ins(%harmonics, %phases, %time) outs(%output) {
^bb0(%freq: f32, %phase: f32, %t: f32, %out: f32):
  %omega = arith.mulf %freq, %t : f32
  %phase_total = arith.addf %omega, %phase : f32
  %sin = math.sin %phase_total : f32
  %scaled = arith.mulf %sin, %harmonic_amplitude : f32
  %sum = arith.addf %out, %scaled : f32
  linalg.yield %sum : f32
}
```

**GPU acceleration:** 100s of sinusoids computed in parallel.

---

## Examples

### Example 1: Simple Ambient Drone

```morphogen
scene SimpleDrone {
  // Harmonic drone with slow drift
  drift = orbit.lfo(period_hours=1.0) * 10cents

  drone = drone.harmonic(
    fundamental = 55Hz + drift,
    spread = 0.2,
    shimmer = 0.3
  )

  out stereo = drone |> reverb(0.5, decay=4.0)
}
```

**Expected output:** Slowly evolving harmonic drone over 1+ hours.

---

### Example 2: CA-Driven Granular Texture

```morphogen
scene CAGranular {
  // Game of Life grid
  life = ca.life(size=64, seed=12345)

  // CA density drives grain rate
  density = field.mean(life) * 100  // 0-100 grains/s

  // Base drone
  base = drone.subharmonic(110Hz, divisions=[1,2,3])

  // Granular cloud
  texture = granular.cloud(
    source = base,
    density = density,
    grain_size = 80ms,
    pitch_shift = 0
  )

  out stereo = texture |> stereo.width(1.4) |> reverb(0.3)
}
```

**Expected output:** Granular texture that evolves based on CA patterns.

---

### Example 3: Spectral Morphing Pad

```morphogen
scene SpectralMorph {
  // Two different spectral textures
  noise_a = noise(seed=1, type="pink") |> lpf(2kHz)
  spec_a = stft(noise_a) |> spectral.blur(bandwidth=100Hz)

  harmonic = drone.harmonic(220Hz, spread=0.3)
  spec_b = stft(harmonic) |> spectral.blur(bandwidth=50Hz)

  // Slow morph between textures
  morph_lfo = orbit.lfo(period_hours=0.5)  // 30-minute cycle

  morphed = spectral.morph(spec_a, spec_b, mix=morph_lfo)
  pad = istft(morphed)

  out stereo = pad |> reverb(0.4, decay=5.0)
}
```

**Expected output:** Pad that morphs between noise and harmonic over 30 minutes.

---

### Example 4: Swarm-Based Generative Melody

```morphogen
scene SwarmMelody {
  // Boid swarm
  boids = agents.boids(
    num = 40,
    bounds = 100,
    cohesion = 0.5,
    separation = 0.3,
    alignment = 0.2
  )

  // Swarm positions → notes (natural minor scale)
  notes = swarm.pitch_map(
    boids,
    pitch_range = (220Hz, 880Hz),
    quantize_to_scale = [0,2,3,5,7,8,10]  // A natural minor
  )

  // Voice synthesis
  voice = (note: Note) => {
    let osc = sine(note.pitch)
    let env = adsr(5ms, 150ms, 0.4, 400ms)
    osc * env |> reverb(0.15)
  }

  // Polyphonic output
  melody = spawn(notes, voice, max_voices=12)

  // Swarm density modulates reverb
  density = swarm.density_modulation(boids, "reverb_mix")
  reverb_amount = density * 0.3 + 0.2

  out stereo = melody |> reverb(reverb_amount, decay=3.0)
}
```

**Expected output:** Melodic material generated from swarm motion, evolving over time.

---

### Example 5: Multi-Domain Physics → Audio

```morphogen
scene FluidAudio {
  // Fluid simulation
  fluid = navier_stokes.solve(
    viscosity = 0.01,
    dt = 0.016,
    grid_size = 128
  )

  // Vorticity field drives grain density
  vorticity = field.curl(fluid.velocity)
  grain_density = vorticity.max() * 80  // High vorticity = dense grains

  // Base sound
  base = drone.harmonic(110Hz, spread=0.2)

  // Granular synthesis modulated by fluid
  texture = granular.cloud(
    source = base,
    density = grain_density,
    grain_size = 60ms,
    pitch_shift = 0
  )

  // Fluid velocity magnitude → spectral brightness
  velocity_mag = field.magnitude(fluid.velocity)
  cutoff = velocity_mag.mean() * 3000Hz + 500Hz

  bright = texture |> lpf(cutoff)

  out stereo = bright |> reverb(0.35)
}
```

**Expected output:** Granular texture and spectral brightness driven by fluid simulation.

---

## Testing Strategy

### Unit Tests (Per Operator)

**Test categories:**
1. **Determinism:** Same seed → same output (bit-exact in strict mode)
2. **Parameter ranges:** Valid input ranges don't cause NaN/Inf
3. **Type safety:** Compile-time errors for invalid connections
4. **Golden artifacts:** Reference WAV files for regression testing

**Example test:**
```python
def test_granular_cloud_deterministic():
    """Granular cloud is deterministic given same seed"""
    source = drone.harmonic(110, spread=0.2)

    grains_1 = granular.cloud(source, density=50, grain_size=80, seed=42)
    grains_2 = granular.cloud(source, density=50, grain_size=80, seed=42)

    assert arrays_identical(grains_1, grains_2)
```

---

### Integration Tests (Pipelines)

**Test multi-operator compositions:**
```python
def test_spectral_morph_pipeline():
    """Spectral morphing pipeline produces valid output"""
    spec_a = stft(noise(seed=1))
    spec_b = stft(sine(440))

    morphed = spectral.morph(spec_a, spec_b, mix=0.5)
    audio = istft(morphed)

    # Check output is valid
    assert no_nans(audio)
    assert rms_level(audio) > 0.001  # Not silent

    # Compare to golden artifact
    assert matches_golden("spectral_morph_50.wav", audio, tolerance=0.01)
```

---

### Cross-Domain Tests

**Test physics → audio integration:**
```python
def test_ca_driven_granular():
    """CA field correctly drives granular density"""
    life = ca.life(size=32, seed=100)
    density = field.mean(life) * 100

    grains = granular.cloud(
        source = sine(220),
        density = density,
        grain_size = 50
    )

    # Density should vary with CA evolution
    assert density_varies_over_time(grains, threshold=0.1)
```

---

### Performance Benchmarks

**GPU vs CPU comparison:**
```python
def benchmark_granular_cloud():
    """Granular synthesis GPU speedup"""
    source = noise(seed=1, duration=10.0)

    # CPU version
    t_cpu = time_execution(
        granular.cloud(source, density=100, grain_size=80, backend="cpu")
    )

    # GPU version
    t_gpu = time_execution(
        granular.cloud(source, density=100, grain_size=80, backend="gpu")
    )

    speedup = t_cpu / t_gpu
    assert speedup > 3.0  # Expect >3x speedup on GPU
```

---

### Long-Form Stability Tests

**Multi-hour evolution:**
```python
def test_long_form_stability():
    """24-hour composition runs without memory leaks"""
    # Ultra-slow drift
    drift = drift.noise(period_minutes=120, depth=1.0)

    # Run for simulated 24 hours
    for hour in range(24):
        audio_chunk = render_hour(drift, hour)

        # Check no memory leaks
        assert memory_usage() < THRESHOLD

        # Check no NaN/Inf
        assert is_valid_audio(audio_chunk)
```

---

## Implementation Phases

### Phase 1: Spectral Domain (v0.6) — 3 months

**Deliverables:**
- Core types: `SpectralField`, `Spectrogram`, `HarmonicSeries`
- 15 spectral operators implemented and tested
- GPU-accelerated FFT/STFT/convolution (via MLIR)
- Golden artifacts: 10 spectral processing examples
- Documentation: Operator reference, examples

**Key operators:**
- `spectral.blur`, `spectral.morph`, `spectral.freeze`
- `harmonic.nebula`, `harmonic.drift`
- `vocode`, `spectral.filter`
- `additive.resynth`

**Success criteria:**
- All tests pass (unit + integration)
- GPU speedup >5x vs CPU for convolution
- Deterministic reproduction across platforms

---

### Phase 2: Ambience Domain (v0.7) — 4 months

**Deliverables:**
- Core types: `Drone`, `GranularCloud`, `Texture`
- 25 ambience operators implemented
- Long-form modulators (drift, orbit LFO)
- Example: 20-minute evolving ambient composition
- Golden artifacts: 5 ambient pieces

**Key operators:**
- `drone.harmonic`, `drone.subharmonic`, `drone.pad`
- `granular.cloud`, `granular.freeze`, `granular.reverse_cloud`
- `drift.noise`, `orbit.lfo`, `slow.random_walk`
- `texture.evolving`, `texture.shimmer`

**Success criteria:**
- Granular synthesis GPU acceleration (>3x speedup)
- Multi-hour compositions run stably
- Deterministic grain playback

---

### Phase 3: Synthesis Domain (v0.8) — 3 months

**Deliverables:**
- Core types: `Patch`, `CV`, `Gate`
- 30 synthesis operators (oscillators, filters, modulation)
- Declarative patch graph system
- Example: Generative modular patch
- Integration with Morphogen.Audio operators

**Key operators:**
- `vco`, `wavetable`, `fm`
- `multimode`, `formant`, `comb`
- `lfo`, `envelope.follower`, `sample_and_hold`

**Success criteria:**
- Type-safe patch routing
- Automatic operator fusion (MLIR)
- Signal flow validation tests pass

---

### Phase 4: Composition Domain (v0.9) — 4 months

**Deliverables:**
- Core types: `MarkovMatrix`, `CARule`, `Pattern`
- 20 composition operators
- CA-driven sequencing examples
- Swarm → melody integration
- Markov chain reference library

**Key operators:**
- `markov.sequence`, `markov.parameter_walk`
- `ca.sequencer`, `life.triggers`, `lenia.modulation`
- `poisson.trigger`, `brownian.melody`
- `euclidean.rhythm`, `fractal.timing`
- `swarm.pitch_map`, `swarm.density_modulation`

**Success criteria:**
- Deterministic sequence generation
- CA → audio integration validated
- Community-contributed Markov matrices

---

### Phase 5: Cross-Domain Showcase (v1.0) — 2 months

**Deliverables:**
- 10 cross-domain examples (physics → audio, fractals → melody)
- Performance benchmarks (CPU vs GPU)
- Comprehensive tutorial documentation
- Video demonstrations
- Golden artifacts (WAV + PNG + video)

**Examples:**
- Fluid vorticity → granular density
- Mandelbrot → harmonic evolution
- Swarm → spatial audio
- Reaction-diffusion → spectral filtering
- CA → multi-voice composition

**Success criteria:**
- All cross-domain examples run successfully
- Documentation complete (100% operator coverage)
- Positive community feedback
- Research paper citations

---

## References

### Academic Research

**Granular Synthesis:**
- Curtis Roads, *"Microsound"* (2001)
- Barry Truax, *"Real-Time Granular Synthesis with a Digital Signal Processor"* (1988)

**Generative Music:**
- Brian Eno, *"Generative Music"* (1996)
- David Cope, *"Computer Models of Musical Creativity"* (2005)

**Cellular Automata & Music:**
- Eduardo Miranda, *"Cellular Automata Music"* (2007)
- Agostino Di Scipio, *"Sound Is the Interface: From Interactive to Ecosystemic Signal Processing"* (2003)

**Spectral Processing:**
- Julius O. Smith III, *"Spectral Audio Signal Processing"* (online book)
- Jean-Claude Risset, *"Computer Music Experiments 1964-"* (1969)

**Physical Modeling:**
- Julius O. Smith III, *"Physical Audio Signal Processing"* (online book)
- Perry Cook, *"Real Sound Synthesis for Interactive Applications"* (2002)

### Commercial Systems

**Granular Synthesis:**
- Mutable Instruments Clouds (Eurorack module)
- Native Instruments Form (software)
- Ableton Granulator (Max4Live device)

**Generative Music:**
- Brian Eno's Bloom, Reflection, Trope (iOS apps)
- Noatikl (generative music software)
- Koan Pro (SSEYO, 1990s)

**Modular Synthesis:**
- VCV Rack (open-source modular)
- Reaktor (Native Instruments)
- Max/MSP (Cycling '74)
- SuperCollider (open-source)

### Morphogen Documentation

**Specifications & Architecture:**
- [ADR-009: Ambient Music & Generative Domains](../adr/009-ambient-music-generative-domains.md)
- [Emergence Domain Specification](emergence.md)
- [Procedural Generation Specification](procedural-generation.md)
- [Domain Architecture Overview](../architecture/domain-architecture.md)
- [Acoustics Domain Specification](acoustics.md)

**Examples & Patterns:**
- [Ambient Music Pipeline Examples](../examples/ambient-music-pipelines.md)
- [Cross-Domain Examples](../examples/README.md)

**Reference Materials:**
- [Sonification Cookbook](../reference/sonification-cookbook.md) - Sonification patterns
- [Mathematical Music Frameworks](../reference/mathematical-music-frameworks.md) - Theoretical foundations
- [Visualization Cookbook](../reference/visualization-cookbook.md) - Visual representation patterns

**Archived:**
- [AUDIO_SPECIFICATION.md](../../archive/root-level-docs/AUDIO_SPECIFICATION.md) - Historical audio spec (archived)

---

**Document Status:** Draft v1.0
**Last Updated:** 2025-11-16
**Next Review:** After Phase 1 implementation
