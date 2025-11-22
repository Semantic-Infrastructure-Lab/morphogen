# Morphogen Ecosystem Map

> **A comprehensive map of all Morphogen domains, modules, and expansion roadmap**

## Purpose of This Document

**For architects**: Understand how all pieces fit together (kernel → domains → frontends)
**For contributors**: See where to add new domains and what's already specified
**For users**: Understand Morphogen's scope and what's possible today vs. planned

This document defines the **complete Morphogen ecosystem** — from the non-negotiable kernel to optional domain libraries to user-facing frontends. It serves as both an architectural guide and a roadmap for future expansion.

> 💡 **New to Morphogen?** Start with [README.md](README.md) for the vision. This document is a detailed catalog of all domains and their relationships.

---

## Overview

Morphogen's world consists of a **minimal kernel** (types, units, domains, scheduler, transforms, registry, MLIR) that supports **optional domain libraries** (audio, physics, circuits, chemistry, video, optimization, etc.) and **human-friendly DSL frontends** that all compile to the same deterministic Graph IR.

**Key Insight**: All domains — audio synthesis, fluid dynamics, circuit simulation, geometry, optimization — share the same execution model. This enables **cross-domain composition** impossible in traditional tools.

---

## Table of Contents

1. [Core Kernel Domains](#1-core-kernel-domains-non-negotiable)
2. [Core Kernel Modules](#2-core-kernel-modules-must-exist-early)
3. [Domain Libraries](#3-domain-libraries-optional-but-powerful)
4. [Frontend Surfaces](#4-frontend-surfaces-not-domains-but-modules)
5. [How These Interrelate](#5-how-these-interrelate-the-big-picture)
6. [Quick Reference](#6-quick-reference)

---

## 1. Core Kernel Domains (non-negotiable)

These are foundational to *everything* Morphogen does — audio, physics, fractals, visuals, ML.
They define the **semantic building blocks**.

### 1.1 Domain: Time

* Discrete sample domains
* Rates (audio, control, visual, sim)
* Event timing & deterministic fences
* Multirate scheduling

### 1.2 Domain: Space

* 1D, 2D, 3D grids
* Spacing, boundaries
* Unit-aware spatial transforms (px, m, k-space)

### 1.3 Domain: Frequency / k-space

* Explicit transform targets (FFT, STFT, DCT, Laplacian spectral)

### 1.4 Domain: General Fields

* `Field<T, space>`
* `Field<T, frequency>`
* `Field<T, time>`
* Domain reparameterization (scale, warp, translate)

### 1.5 Domain: Events

* `Evt<A>`
* Timestamped, ordered
* Sample-accurate fences
* Randomness seeding and reproducibility

### 1.6 Core Transforms

These are **core transform dialect operations** and belong in the kernel:

* **FFT/iFFT** — time ↔ frequency
* **STFT/iSTFT** — short-time frequency analysis
* **DCT/IDCT** — discrete cosine transform
* **Wavelet/IWavelet** — multi-scale decomposition
* **Laplacian spectral** — graph/topology transforms
* **Space ↔ k-space** — spatial frequency transforms
* **Mel ↔ linear** — perceptual frequency mapping
* **Reparam / warps** — coordinate transformations

**Rationale:** These are universal domain-change operations, not domain-specific logic.

---

## 2. Core Kernel Modules (must exist early)

These are cross-domain kernel subsystems.

### 2.1 Type System & Units Module

* `Stream<T, Domain, Rate>`
* `Field<T, Space>`
* `Evt<A>`
* Grid metadata
* SI + custom units
* Cast rules

**See:** [SPEC-TYPE-SYSTEM.md](docs/SPEC-TYPE-SYSTEM.md)

### 2.2 Deterministic Multirate Scheduler

* LCM partitioning
* dt consistency
* Cross-rate resampling
* Event fences
* Hot reload barriers

**See:** [SPEC-SCHEDULER.md](docs/SPEC-SCHEDULER.md)

### 2.3 Operator Registry

* Metadata for every op across **7 semantic layers**:
  1. **Core** — cast, domain, rate, shape
  2. **Transforms** — FFT/STFT/DCT/wavelet/mel
  3. **Stochastic** — RNG, SDE processes, Monte Carlo
  4. **Physics/Fields** — integrators, PDEs, particle/grid coupling
  5. **Audio** — oscillators, filters, effects, spectral ops
  6. **Fractals/Visuals** — iteration, palette, coordinate mapping
  7. **Finance** — models (Heston, SABR), payoffs, pricing
* Lowering templates, numeric properties (order, symplectic, conservative)
* Profile defaults and determinism tiers

**See:** [SPEC-OPERATOR-REGISTRY.md](docs/SPEC-OPERATOR-REGISTRY.md) for complete 7-layer catalog

### 2.4 Profiles & Determinism Controller

* **strict** (bit exact)
* **repro** (deterministic FP)
* **live** (latency-first, approximate)
* Profile-based op overrides

**See:** [SPEC-PROFILES.md](docs/SPEC-PROFILES.md)

### 2.5 Snapshot ABI

* Buffer tree
* Graph hash
* Seed state
* Safe hot reload

**See:** [SPEC-SNAPSHOT-ABI.md](docs/SPEC-SNAPSHOT-ABI.md)

### 2.6 MLIR Backend

**Dialects:**
* `morphogen.stream`
* `morphogen.field`
* `morphogen.transform`
* `morphogen.agent` (optional later)

**Lowering to:**
* linalg
* affine
* vector
* gpu
* llvm

**See:** [SPEC-MLIR-DIALECTS.md](docs/SPEC-MLIR-DIALECTS.md), [GPU_MLIR_PRINCIPLES.md](docs/GPU_MLIR_PRINCIPLES.md)

---

## 3. Domain Libraries (optional, but powerful)

These are **not** core kernel logic, but they plug into it via the operator registry.
Each sits on top of the core types & transforms.

### 3.1 Audio / DSP (first-class)

**Status:** Active development

Already part of Morphogen.Audio:

* Oscillators (sine, saw, tri, square, noise)
* Filters (lpf, hpf, bpf, notch, allpass, shelves)
* Envelopes (ADSR, AR, exponential)
* Waveguides (string, tube, membrane)
* Reverb / convolution
* Spectral transforms
* Effects (delay, chorus, flanger, phaser, distortion)
* Instrument models (physical modeling components)

**Why it's a domain library:**
* Uses `Stream<f32, 1D, audio>` (kernel type)
* Uses kernel transforms (FFT, STFT, etc.)
* Plugs into operator registry
* Respects determinism profiles

**See:** [AUDIO_SPECIFICATION.md](AUDIO_SPECIFICATION.md)

### 3.1a Ambient Music & Generative Audio ⭐ **NEW - November 2025**

**Status:** Proposed (ADR-009, Specification complete)

Four specialized domains for ambient and generative music:

**Morphogen.Spectral** (15 operators)
* Spectral blurring, morphing, freezing
* Harmonic nebulae (distributed harmonic clouds)
* Vocoding and spectral filtering
* Additive resynthesis with time-varying parameters
* Pitch-shifting drones

**Morphogen.Ambience** (25 operators)
* Drone generators (harmonic pads, subharmonic bass, shimmer)
* Granular synthesis (clouds, frozen, reverse)
* Long-form modulators (drift noise, orbit LFOs, hour-scale evolution)
* Texture generators (evolving, shimmer, noise fields)

**Morphogen.Synthesis** (30 operators)
* Enhanced modular synthesis (VCO, wavetable, FM)
* Advanced filters (multimode, formant, comb)
* Modulation sources (LFO, envelope follower, sample-and-hold)
* Declarative patch routing (compute graph as signal flow)

**Morphogen.Composition** (20 operators)
* Markov chain sequencers (melodic and parameter evolution)
* CA-driven composition (Life, Lenia, Rule30 → notes)
* Stochastic generators (Poisson triggers, Brownian melodies)
* Pattern generators (Euclidean rhythms, fractal timing)
* Swarm-based composition (agent positions → pitches)

**Why this is transformative:**
* **Cross-domain integration** — Physics/CA/fractals drive audio parameters
* **GPU acceleration** — Granular synthesis, spectral convolution, additive synthesis
* **Deterministic generative music** — Same seed = same evolution
* **Multi-hour time scales** — Ultra-slow LFOs, drift modulators
* **No competing tool** — Unifies DSP + CA + physics + GPU

**Total:** 90 new operators across 4 domains

**See:**
* [ADR-009](docs/adr/009-ambient-music-generative-domains.md) — Architecture decision
* [docs/specifications/ambient-music.md](docs/specifications/ambient-music.md) — Complete specification (90 operators)
* [docs/domains/AMBIENT_MUSIC.md](docs/domains/AMBIENT_MUSIC.md) — Domain documentation (pending)

### 3.2 Physics

**Status:** Future expansion

Potential modules for:

* **N-body integrators** (Euler, RK4, Verlet, leapfrog)
* **Symplectic integrators** (energy-conserving)
* **Adaptive timesteps** (error control)
* **Particle-grid transforms** (PIC, FLIP, APIC)
* **PDE operators:**
  * Wave equation
  * Diffusion
  * Poisson solver
  * Navier-Stokes
* **Boundary conditions:**
  * Dirichlet
  * Neumann
  * Periodic
  * Reflecting

**Why it's a perfect domain library:**
* Uses `Field<T, space>` (kernel type)
* Uses kernel transforms (Laplacian, FFT for spectral methods)
* Needs time integration (kernel scheduler)
* Needs determinism profiles (strict for reproducibility)

**Example use cases:**
* Fluid simulation (smoke, water)
* Cloth simulation
* Rigid body dynamics
* Soft body physics
* Particle systems

### 3.3 Geometry & Topology

**Status:** Future expansion

Potential modules for:

* **Graph Laplacians** (spectral decomposition)
* **Mesh transforms** (subdivision, smoothing, decimation)
* **Spectral decompositions** (eigenvalue analysis)
* **Coordinate maps** (UV mapping, geodesic distance)

**Why it fits:**
* Uses kernel transforms (Laplacian spectral)
* Natural for graph-based data structures
* Can interface with `Field` for discretizations

**Example use cases:**
* Shape analysis
* Mesh processing
* Topology optimization
* Graph neural networks

### 3.4 Visuals / Image Processing

**Status:** Partial implementation (visual dialect exists)

Modules include:

* **Blur / sharpen**
* **Convolution kernels**
* **Gradient / divergence**
* **Nonlinear maps** (tone mapping, color grading)
* **Resampling** (scale, rotate, warp)
* **Warps** (perspective, lens distortion)

**Why it's a domain library:**
* Uses `Field<T, 2D>` (kernel type)
* Uses kernel transforms (FFT for filtering, DCT for compression)
* Natural composition with other domains

**Example use cases:**
* Real-time visual effects
* Generative art
* Image enhancement
* Computer vision preprocessing

### 3.5 Fractals (as its own micro-domain)

**Status:** Future expansion

We essentially designed:

* **Field iteration kernels** (Mandelbrot, Julia, Burning Ship, etc.)
* **Plane mapping transforms** (zoom, pan, rotation, complex transforms)
* **Palette transforms** (color mapping, gradient interpolation)
* **Zoom trajectories** (keyframed camera paths)

**Why it behaves like other domains:**
* Just like a PDE or particle system — domain-local transforms of a `Field`
* Iteration = time evolution
* Palette = visualization transform
* Zoom = spatial reparameterization

**Example use cases:**
* Fractal art generation
* Exploratory visualization
* Mathematical illustration
* Generative backgrounds

### 3.6 Particles / Agents

**Status:** Planned (agent dialect specified)

Modules include:

* **Boids** (flocking, swarming)
* **SPH** (smoothed particle hydrodynamics)
* **Cellular automata** (Conway's Life, Langton's Ant)
* **Physics particles** (springs, collision)
* **Force fields** (gravity, electromagnetism, custom)
* **Agent-based transforms** (state machines, behaviors)

**Why it fits:**
* Uses `Agents<T>` (kernel type)
* All are "streams of state updated per tick"
* Natural for sparse, heterogeneous systems

**Example use cases:**
* Crowd simulation
* Ecological modeling
* Particle effects (fire, smoke, sparks)
* Artificial life

### 3.7 Machine Learning (later)

**Status:** Long-term roadmap

Eventually MLIR gives you:

* **Tensor ops** (matrix multiply, convolution)
* **NN kernels** (ReLU, softmax, attention)
* **Differentiable programming** (autodiff)
* **JIT specialization** (kernel fusion)

**Why it's not needed early:**
* MLIR lowering infrastructure must mature first
* Other domains provide immediate value
* Can be added later without kernel changes

**Example use cases:**
* Neural synthesis (WaveNet, Neural DSP)
* Generative models (GANs, diffusion models)
* Differentiable physics
* Learned operators (replacing hand-tuned kernels)

### 3.8 Circuit / Electrical Engineering ⭐ **NEW - November 2025**

**Status:** Architecture complete (PR #43) — **`docs/SPEC-CIRCUIT.md`** (1,136 lines)

The most natural domain for Morphogen — circuits ARE typed operator graphs!

* **Atomic operators**: R, C, L, voltage/current sources
* **Composite operators**: Op-amps, transistors, transformers
* **Analysis**: DC, AC, transient, harmonic balance, noise
* **Multi-domain**: Circuit ↔ Audio (guitar pedals), Geometry ↔ Circuit (PCB parasitics)

**See:** `examples/circuit/` (5 complete examples)

### 3.9 Acoustics & Fluid Dynamics ⭐ **NEW - November 2025**

**Status:** Architecture complete (PR #44) — **`docs/DOMAIN_ARCHITECTURE.md`** sections 2.9, 2.10

Unified fluid + acoustic pipeline for exhaust systems, instruments, HVAC:

* **FluidDynamics**: 1D compressible flow, Navier-Stokes, thermodynamic coupling, engine pulses
* **Acoustics**: 1D waveguides, FDTD, Helmholtz resonators, radiation impedance
* **Pipeline**: FluidDynamics → Acoustics → Audio → WAV files

**See:** `docs/USE_CASES/2-stroke-muffler-modeling.md`

### 3.10 Instrument Modeling ⭐ **NEW - November 2025**

**Status:** Architecture complete (PR #45) — **`docs/SPEC-TIMBRE-EXTRACTION.md`** (752 lines)

The "holy grail" of audio DSP — recordings → synthesis models:

* **Analysis** (15 ops): Pitch tracking, harmonic extraction, modal fitting, inharmonicity
* **Synthesis** (12 ops): Additive, modal, granular, spectral filtering
* **Modeling** (8 ops): InstrumentModel type, timbre morphing, virtual acoustics

**Use case:** Record acoustic guitar → extract timbre → synthesize new MIDI notes

### 3.11 Optimization ⭐ **NEW - November 2025**

**Status:** Architecture complete (PR #48) — **`docs/LEARNINGS/OPTIMIZATION_ALGORITHMS_CATALOG.md`** (1,529 lines)

16 algorithms transforming Morphogen: simulation → design discovery:

* **Evolutionary** (4): GA, DE, CMA-ES, PSO
* **Local** (3): Gradient Descent, L-BFGS, Nelder-Mead
* **Surrogate** (3): Bayesian Optimization, Response Surface, Kriging
* **Multi-Objective** (3): NSGA-II, SPEA2, MOPSO

**Cross-domain:** Optimize J-tube geometry, muffler design, PID tuning, filter parameters

### 3.12 Multi-Physics Engineering ⭐ **NEW - November 2025**

**Status:** Architecture complete (PR #47) — **`docs/SPEC-PHYSICS-DOMAINS.md`** (1,079 lines)

Four specialized domains for engineering simulation:

* **FluidNetwork**: Lumped fluid networks (pipes, valves, pumps)
* **ThermalODE**: ODE-based thermal models (heat transfer, radiation)
* **FluidJet**: Jet/flame dynamics (velocity, spread, entrainment)
* **CombustionLight**: Reaction kinetics, flame modeling

**Pipeline:** Geometry → FluidNetwork → ThermalODE → CombustionLight

**See:** `docs/EXAMPLES/J-TUBE-FIREPIT-MULTIPHYSICS.md`

---

## 4. Frontend Surfaces (not domains, but modules)

These sit **above the kernel completely** and emit Graph IR.

### 4.1 Morphogen.Audio

**Declarative DSL for time-dependent audio scenes**

* Compositional language layer
* Scene-based organization
* Event scheduling
* Polyphony management
* Physical modeling

**See:** [AUDIO_SPECIFICATION.md](AUDIO_SPECIFICATION.md)

### 4.2 RiffStack

**YAML + RPN live-performance surface**

* Stack-based expression language
* YAML patch definitions
* Loopers and live control
* Performance-oriented ergonomics
* Compiles to same Graph IR as Morphogen.Audio

**See:** [RiffStack repository](https://github.com/scottsen/riffstack)

### 4.3 Future DSLs

Potential future frontends:

* **Morphogen.Fractal** — specialized fractal exploration language
* **Morphogen.Physics** — physics simulation DSL
* **Morphogen.Visual** — visual composition language
* **Morphogen.Agent** — agent-based modeling language

**All emit the same Graph IR.**

---

## 5. How These Interrelate (the big picture)

### Layer 1: The Kernel

**Defines:**
* Types
* Units
* Domains
* Scheduler
* Determinism
* Transforms
* Operator registry
* MLIR lowering

**Everything below this line must be extremely stable.**

---

### Layer 2: Domain Libraries

**Define:**
* Specific operators
* Integrators
* Transforms
* Field operations
* Palette ops
* Volume ops
* PDE kernels

**They all plug into the operator registry and lower into kernel dialects.**

---

### Layer 3: Frontends

**Define:**
* Human-friendly text formats
* Syntax
* Ergonomics
* Composition tools

**But they emit pure Morphogen Graph IR.**

---

### Dependency Flow

```
┌─────────────────────────────────────────────────┐
│           User-Facing Frontends                 │
│  Morphogen.Audio | RiffStack | Future DSLs          │
└────────────────┬────────────────────────────────┘
                 │
                 ▼ (Graph IR)
┌─────────────────────────────────────────────────┐
│            Domain Libraries                     │
│  Audio | Physics | Fractals | Visuals | ...     │
└────────────────┬────────────────────────────────┘
                 │
                 ▼ (Operator Registry)
┌─────────────────────────────────────────────────┐
│              Morphogen Kernel                       │
│  Types | Units | Scheduler | Transforms | MLIR  │
└─────────────────────────────────────────────────┘
```

---

## 6. Quick Reference

### Kernel Domains

* time
* frequency
* space (1D/2D/3D)
* k-space
* event
* field
* transform

### Kernel Modules

* type system
* units system
* profiles
* operator registry
* deterministic scheduler
* MLIR dialects
* snapshot ABI

### Domain Libraries (optional)

* audio / DSP ⭐ (active)
* ambient music & generative audio ⭐ (proposed) — **NEW**
  - spectral (frequency-domain processing)
  - ambience (drones, granular, long-form evolution)
  - synthesis (modular DSP routing)
  - composition (Markov, CA, swarm sequencing)
* physics / PDE / integrators 📋 (planned)
* fractals 📋 (planned)
* visuals / image processing 🚧 (partial)
* geometry / topology 📋 (planned)
* particles / agents 📋 (planned)
* ML 🔮 (future)

### Frontends

* Morphogen.Audio ⭐ (active)
* RiffStack ⭐ (active)
* Future DSLs 📋 (planned)

---

## Design Principles

### What Belongs in the Kernel?

1. **Universal abstractions** — time, space, events, fields
2. **Domain-change operations** — transforms (FFT, STFT, etc.)
3. **Execution model** — scheduler, determinism, state management
4. **Cross-domain contracts** — types, units, operator metadata

### What Belongs in Domain Libraries?

1. **Domain-specific operators** — filters, integrators, palette maps
2. **Specialized algorithms** — symplectic integration, fractal iteration
3. **Knowledge-intensive logic** — physics constants, filter designs
4. **Optional complexity** — you only pay for what you use

### What Belongs in Frontends?

1. **Human ergonomics** — syntax, composition patterns
2. **Domain-specific abstractions** — scenes, patches, agents
3. **User-facing concepts** — notes, palettes, forces
4. **Compilation to Graph IR** — all frontends emit the same thing

---

## Expansion Strategy

### Phase 1: Foundation (v0.1 - v0.4)

* Kernel complete
* Audio domain library production-ready
* Morphogen.Audio + RiffStack frontends stable
* MLIR pipeline functional

### Phase 2: Visual Expansion (v0.5 - v0.7)

* Visual domain library complete
* Fractal domain library added
* Image processing operators
* Visual frontend DSL

### Phase 3: Physics & Simulation (v0.8 - v1.0)

* Physics domain library
* PDE solvers
* Particle systems
* Agent-based modeling

### Phase 4: Advanced Domains (v1.1+)

* Geometry/topology
* Machine learning integration
* Custom domain framework
* Third-party domain plugins

---

## Conclusion

The Morphogen ecosystem is designed for **sustainable growth**. The kernel remains minimal and stable, while domain libraries and frontends can expand indefinitely without breaking existing code.

**Core insight:** Audio, physics, fractals, and visuals are not fundamentally different problems — they're all **time-evolving fields and streams transformed between domains**. Morphogen's kernel provides the universal substrate; domain libraries provide the specialized operators.

This architecture ensures that a physics simulation, a fractal renderer, and an audio synthesizer can **compose seamlessly** — because they speak the same language.

**Professional Impact:** This unified ecosystem unlocks transformative value across science, engineering, finance, agriculture, and trades — domains that currently suffer from fragmented tools and lack cross-domain integration. See **[docs/reference/professional-domains.md](docs/reference/professional-domains.md)** for detailed analysis of Morphogen's value proposition in 10 professional fields.

---

**Related Documentation:**

**Architecture**:
* [ARCHITECTURE.md](ARCHITECTURE.md) — The Morphogen Stack architecture (kernel, frontends, Graph IR, MLIR)
* [docs/architecture/domain-architecture.md](docs/architecture/domain-architecture.md) — Complete multi-domain technical vision (2,266 lines)
* [docs/architecture/gpu-mlir-principles.md](docs/architecture/gpu-mlir-principles.md) — GPU lowering and MLIR integration

**Professional Impact**:
* [docs/reference/professional-domains.md](docs/reference/professional-domains.md) — Value proposition across engineering, science, finance, creative fields

**Specifications**:
* [docs/specifications/type-system.md](docs/specifications/type-system.md) — Type system specification
* [docs/specifications/transform.md](docs/specifications/transform.md) — Transform dialect specification
* [docs/specifications/operator-registry.md](docs/specifications/operator-registry.md) — Operator registry specification
* [AUDIO_SPECIFICATION.md](AUDIO_SPECIFICATION.md) — Audio domain library specification

**Guides**:
* [docs/guides/domain-implementation.md](docs/guides/domain-implementation.md) — Step-by-step guide for adding new domains
* [docs/README.md](docs/README.md) — Complete documentation navigation

---

**Version:** 1.0 Draft
**Last Updated:** 2025-11-15
**Status:** Living Document
