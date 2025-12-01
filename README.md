---
project: morphogen
type: software
status: active
beth_topics:
- morphogen
- creative-computation
- dsl
- mlir
- audio-synthesis
- agent-simulation
- field-operations
tags:
- compiler
- simulation
- generative
- deterministic
---

# Morphogen

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> *Where computation becomes composition*

**Morphogen** is a universal, deterministic computation platform that unifies domains that have never talked to each other before: **audio synthesis meets physics simulation meets circuit design meets geometry meets optimization** — all in one type system, one scheduler, one language.

## Why Morphogen Exists

Current tools force you to:
- Export CAD → import to FEA → export mesh → import to CFD → manually couple results
- Write audio DSP in C++ → physics in Python → visualization in JavaScript
- Bridge domains with brittle scripts and incompatible data formats

**Morphogen eliminates this fragmentation.** Model a guitar string's physics, synthesize its sound, optimize its geometry, and visualize the result — all in the same deterministic execution environment.

---

## Part of the Semantic Infrastructure Lab

**Morphogen** is a production component of the [Semantic Infrastructure Lab (SIL)](https://github.com/semantic-infrastructure-lab/sil) — building the semantic substrate for intelligent systems.

**Role in the Semantic OS:**
- **Layer 2:** Domain Module (audio, physics, circuits, geometry)
- **Layer 4:** Deterministic Engine (MLIR compilation, reproducible execution)

**SIL Principles Applied:**
- ✅ **Clarity** — Explicit domain semantics (audio, field, agent, geometry)
- ✅ **Simplicity** — Four core operations unify all domains
- ✅ **Composability** — Domains compose via typed connections
- ✅ **Correctness** — Type system enforces units, rates, constraints
- ✅ **Verifiability** — Deterministic execution enables reproducibility

**Quick Links:** [SIL Manifesto](https://github.com/semantic-infrastructure-lab/sil/blob/main/docs/canonical/MANIFESTO.md) • [Unified Architecture](https://github.com/semantic-infrastructure-lab/sil/blob/main/docs/architecture/UNIFIED_ARCHITECTURE_GUIDE.md) • [Project Index](https://github.com/semantic-infrastructure-lab/sil/blob/main/projects/PROJECT_INDEX.md)

---

## Two Surfaces, One Kernel

Morphogen presents **two human-friendly faces** powered by a single semantic kernel:

- **Morphogen.Audio** — Declarative language for compositional audio, physics, and multi-domain scenes
- **RiffStack** — Live performance environment for real-time interaction and improvisation

Both compile to the same Graph IR, share the same operator registry, and guarantee deterministic, reproducible results.

> 📐 **Deep Dive**: See [docs/architecture/](docs/architecture/) for the complete stack design (kernel, frontends, Graph IR, MLIR compilation)

## What Makes Morphogen Different

**Cross-Domain Composition**
- Audio synthesis + fluid dynamics + circuit simulation in the same program
- Type-safe connections between domains (e.g., field → agent force, geometry → audio impulse response)
- Single execution model handles multiple rates (audio @ 48kHz, control @ 60Hz, physics @ 240Hz)

**Deterministic by Design**
- Bitwise-identical results across runs, platforms, and GPU vendors
- Explicit RNG seeding, sample-accurate event scheduling
- Three profiles: `strict` (bit-exact), `repro` (deterministic FP), `live` (low-latency)

**Transform-First Thinking**
- FFT, STFT, wavelets, DCT as first-class operations
- Domain changes (time ↔ frequency, space ↔ k-space) are core primitives
- Uniform transform API across all domains

**Production-Grade Compilation**
- MLIR-based compiler with 6 custom dialects
- Lowers to optimized CPU/GPU code via LLVM
- Field operations, agents, audio DSP, temporal execution all compile to native code

---

## Sister Project: Philbrick

**Morphogen** (software) and **[Philbrick](https://github.com/scottsen/philbrick)** (hardware) are **two halves of one vision** — modular computation in different substrates.

| Aspect | Morphogen (Digital) | Philbrick (Analog/Hybrid) |
|--------|----------------|---------------------------|
| **Purpose** | Digital simulation of continuous phenomena | Physical embodiment of continuous dynamics |
| **Primitives** | Streams, fields, transforms | Sum, integrate, nonlinearity, events |
| **Safety** | Type system (domain/rate/units) | Pin contracts (voltage/impedance/latency) |
| **Execution** | Multirate deterministic scheduler | Latency-aware routing fabric |
| **Philosophy** | **Computation = composition** | **Computation = composition** |

### The Bridge

- **Design in Morphogen** → Simulate and optimize continuous-time systems
- **Build in Philbrick** → Physical modules implementing the same primitives
- **Validate Together** → Software and hardware mirror each other

Both platforms share the same four core operations (sum, integrate, nonlinearity, events) and the same compositional philosophy. They will eventually compile to each other.

> 🔧 **Learn More**: [Philbrick](https://github.com/scottsen/philbrick) - Modular analog/digital hybrid computing platform

---

## Cross-Domain in Action

Here's what sets Morphogen apart — domains working together seamlessly:

```morphogen
# Couple fluid dynamics → acoustics → audio synthesis
use fluid, acoustics, audio

# Simulate airflow in a 2-stroke engine exhaust
@state flow : FluidNetwork1D = engine_exhaust(length=2.5m, diameter=50mm)
@state acoustic : AcousticField1D = waveguide_from_flow(flow)

flow(dt=0.1ms) {
    # Fluid dynamics: pressure pulses from engine
    flow = flow.advance(engine_pulse(t), method="lax_wendroff")

    # Couple to acoustics: flow → sound propagation
    acoustic = acoustic.couple_from_fluid(flow, impedance_match=true)

    # Synthesize audio from acoustic field
    let exhaust_sound = acoustic.to_audio(mic_position=1.5m)

    # Real-time output
    audio.play(exhaust_sound)
}
```

**One program. Three domains. Zero glue code.**

See [docs/use-cases/2-stroke-muffler-modeling.md](docs/use-cases/2-stroke-muffler-modeling.md) for the complete example.

---

## Quick Start

### Installation

```bash
git clone https://github.com/scottsen/morphogen.git
cd morphogen
pip install -e .
```

### Your First Program

Create `hello.kairo` (Morphogen source files use `.kairo` extension):

```morphogen
# hello.kairo - Heat diffusion

use field, visual

@state temp : Field2D<f32 [K]> = random_normal(
    seed=42,
    shape=(128, 128),
    mean=300.0,
    std=50.0
)

const KAPPA : f32 [m²/s] = 0.1

flow(dt=0.01, steps=500) {
    temp = diffuse(temp, rate=KAPPA, dt, iterations=20)
    output colorize(temp, palette="fire", min=250.0, max=350.0)
}
```

Run it:

```bash
morphogen run hello.kairo
```

**Next steps:**
- Try the [examples](examples/) directory (24 working examples)
- Read [Getting Started](docs/getting-started.md) for a guided tutorial
- Explore the [domain catalog](docs/DOMAINS.md) to see what's possible

---

## 🚀 Project Status & v1.0 Roadmap

**Current Status (v0.11.0):**
- ✅ 40 production-ready computational domains
- ✅ 900+ comprehensive tests (all passing)
- ✅ MLIR compilation pipeline complete (6 phases)
- ✅ Python runtime with NumPy backend
- ✅ Zero technical debt

**v1.0 Release Plan (24 weeks):**

Morphogen is on an aggressive path to v1.0 with a three-track strategy:

1. **Track 1 - Language Evolution** (13 weeks)
   - Symbolic + numeric execution (SymPy integration)
   - Transform space tracking with functorial translations
   - Algebraic composition (`∘` operator) + category theory optimization
   - Domain plugin system for user extensibility

2. **Track 2 - Critical Domains** (12 weeks)
   - Circuit domain with audio coupling ⭐ **Unique differentiator**
   - Fluid dynamics (Navier-Stokes)
   - Chemistry Phase 2 expansion
   - **Target: 50+ domains**

3. **Track 3 - Adoption & Polish** (ongoing)
   - PyPI release (alpha in week 4)
   - 5 showcase examples with videos
   - 7 progressive tutorials
   - Complete API documentation
   - Active community infrastructure

**Read the full plan:** [**Morphogen v1.0 Release Plan**](docs/planning/MORPHOGEN_RELEASE_PLAN.md)

**What makes v1.0 special:**
- 🔬 Symbolic + numeric execution (first platform to combine both)
- 🎵 Circuit → Audio coupling (design pedal circuits, hear sound instantly)
- 📐 Category theory optimization (verified composition, automatic fusion)
- 🔌 User extensibility (plugin system for custom domains)
- 🎯 50+ integrated domains (audio, physics, chemistry, graphics, AI)

**Timeline:** Current v0.11.0 → v1.0 release in 2026-Q2

---

## Language Overview

### Temporal Model

Morphogen programs describe time-evolving systems through `flow` blocks:

```morphogen
flow(dt=0.01, steps=1000) {
    # Execute this block 1000 times with timestep 0.01
    temp = diffuse(temp, rate=0.1, dt)
    output colorize(temp, palette="fire")
}
```

### State Management

Persistent variables are declared with `@state`:

```morphogen
@state vel : Field2D<Vec2<f32>> = zeros((256, 256))
@state agents : Agents<Particle> = alloc(count=1000)

flow(dt=0.01) {
    vel = advect(vel, vel, dt)      # Updates vel for next step
    agents = integrate(agents, dt)   # Updates agents for next step
}
```

### Deterministic Randomness

All randomness is explicit via RNG objects:

```morphogen
@state agents : Agents<Particle> = alloc(count=100, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1))
    }
}
```

### Physical Units

Types can carry dimensional information:

```morphogen
temp : Field2D<f32 [K]>           # Temperature in Kelvin
pos : Vec2<f32 [m]>               # Position in meters
vel : Vec2<f32 [m/s]>             # Velocity in m/s
force : Vec2<f32 [N]>             # Force in Newtons

# Unit checking at compile time
dist : f32 [m] = 10.0
time : f32 [s] = 2.0
speed = dist / time               # OK: f32 [m/s]

# ERROR: cannot mix incompatible units
x = dist + time                   # ERROR: m + s is invalid
```

---

## 40+ Computational Domains

Morphogen provides **40+ production-ready domains** spanning physics, audio, graphics, chemistry, and more. Each domain offers specialized operators optimized for its computational model.

### Core Domains

**Field Operations** - PDE solvers, diffusion, advection, stencils  
**Agent Systems** - Particle simulations, flocking, N-body forces  
**Audio Synthesis** - Oscillators, filters, effects, physical modeling  
**RigidBody Physics** - 2D dynamics, collisions, constraints  

### Advanced Domains

**Chemistry Suite** (9 domains) - Molecular dynamics, quantum chemistry, thermodynamics, kinetics  
**Graphics Suite** (4 domains) - Procedural noise, palettes, color spaces, image processing  
**Infrastructure** - Sparse linear algebra, integrators, I/O & storage  
**Analysis** - Signal processing, computer vision, terrain generation, audio analysis  

### Complete Domain Catalog

**📋 [View Full Domain Catalog](docs/DOMAINS.md)** - Detailed descriptions, code examples, and feature lists for all 40+ domains

**Status tracking:** See [STATUS.md](STATUS.md) for implementation status and test coverage by domain

---

## Examples

### Fluid Simulation (Navier-Stokes)

```morphogen
use field, visual

@state vel : Field2D<Vec2<f32 [m/s]>> = zeros((256, 256))
@state density : Field2D<f32> = zeros((256, 256))

const VISCOSITY : f32 = 0.001
const DIFFUSION : f32 = 0.0001

flow(dt=0.01, steps=1000) {
    # Advect velocity
    vel = advect(vel, vel, dt, method="maccormack")

    # Diffuse velocity (viscosity)
    vel = diffuse(vel, rate=VISCOSITY, dt, iterations=20)

    # Project (incompressibility)
    vel = project(vel, method="cg", max_iterations=50)

    # Advect and diffuse density
    density = advect(density, vel, dt)
    density = diffuse(density, rate=DIFFUSION, dt)

    # Dissipation
    density = density * 0.995

    # Visualize
    output colorize(density, palette="viridis")
}
```

### Reaction-Diffusion (Gray-Scott)

```morphogen
use field, visual

@state u : Field2D<f32> = ones((256, 256))
@state v : Field2D<f32> = zeros((256, 256))

const Du : f32 = 0.16
const Dv : f32 = 0.08
const F : f32 = 0.060
const K : f32 = 0.062

flow(dt=1.0, steps=10000) {
    # Gray-Scott reaction
    let uvv = u * v * v
    let du_dt = Du * laplacian(u) - uvv + F * (1.0 - u)
    let dv_dt = Dv * laplacian(v) + uvv - (F + K) * v

    u = u + du_dt * dt
    v = v + dv_dt * dt

    # Visualize
    output colorize(v, palette="viridis")
}
```

**📂 More Examples:** See the [examples/](examples/) directory for 24 working programs demonstrating all major domains!

---

## Documentation

### Getting Started

- **[Getting Started Guide](docs/getting-started.md)** - Installation, first program, core concepts
- **[Domain Catalog](docs/DOMAINS.md)** - Complete catalog of all 40+ domains with examples
- **[docs/README.md](docs/README.md)** - Documentation navigation and index

### Technical Documentation

- **[SPECIFICATION.md](SPECIFICATION.md)** - Complete language specification (2,282 lines)
- **[Architecture](docs/architecture/)** - System design and MLIR compilation
- **[Specifications](docs/specifications/)** - Domain-specific technical specs
- **[ADRs](docs/adr/)** - Architectural decision records

### Implementation Resources

- **[STATUS.md](STATUS.md)** - Current implementation status by domain
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[Domain Implementation Guide](docs/guides/domain-implementation.md)** - How to add new domains

---

## The Ecosystem Vision

Morphogen is building toward a future where professional domains seamlessly compose:

### Professional Applications

**Education & Academia** - Multi-physics simulations, interactive visualizations, reproducible research

**Digital Twins & Enterprise** - Real-time system simulation, predictive maintenance, optimization workflows

**Audio Production & Lutherie** - Physical modeling synthesis, instrument design, timbre extraction

**Scientific Computing** - Coupled PDE systems, reaction-diffusion, quantum chemistry

**Creative Coding & Generative Art** - Procedural generation, audio-reactive visuals, deterministic creativity

---

## Evolution from Creative Computation DSL

Morphogen v0.3.1 is the evolution of Creative Computation DSL v0.2.2, incorporating:

- **Better semantics**: `flow(dt)` blocks, `@state` declarations, explicit RNG
- **Clearer branding**: "Morphogen" is unique and memorable
- **Same foundation**: Frontend work carries forward, comprehensive stdlib preserved

---

## Related Projects

**[RiffStack](https://github.com/scottsen/riffstack)** - Live performance shell for Morphogen.Audio

RiffStack is a stack-based, YAML-driven performance environment that serves as the live interface to Morphogen.Audio. While Morphogen.Audio provides the compositional language layer, RiffStack offers real-time interaction and performance capabilities. Together they form a complete audio synthesis and performance ecosystem built on Morphogen's deterministic execution kernel.

---

## Contributing

Morphogen is building toward something transformative: a universal platform where professional domains that have never talked before can seamlessly compose. Contributions welcome at all levels!

**See [CONTRIBUTING.md](CONTRIBUTING.md) for:**
- Development setup
- High-impact contribution areas
- Code style guidelines
- Pull request process

**Quick links:**
- [Open Issues](https://github.com/scottsen/morphogen/issues)
- [Discussion Forum](https://github.com/scottsen/morphogen/discussions)
- [Domain Implementation Guide](docs/guides/domain-implementation.md)

---

## License

Apache 2.0 - see [LICENSE](LICENSE) for details

Copyright 2025 Semantic Infrastructure Lab Contributors

---

## Contact

- **GitHub**: https://github.com/scottsen/morphogen
- **Issues**: https://github.com/scottsen/morphogen/issues
- **Discussions**: https://github.com/scottsen/morphogen/discussions

---

**Status:** v0.11.0 → v1.0 Release Plan Active | **Current Version:** 0.11.0 | **Target:** v1.0 (2026-Q2) | **Last Updated:** 2025-11-21

**🚀 [View v1.0 Release Plan](docs/planning/MORPHOGEN_RELEASE_PLAN.md)** - 24-week roadmap to production release
