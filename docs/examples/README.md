# Morphogen Examples & Case Studies

This directory contains comprehensive examples and case studies demonstrating Morphogen's multi-domain capabilities.

---

## Examples

### 🏎️ [Racing AI Training Pipeline: Neural Evolution Example](RACING-AI-PIPELINE.md)

**Status:** Design Document
**Domains:** Physics (Racing), Neural Network, Genetic Algorithm, Rendering, Telemetry, Recording
**Complexity:** Advanced
**Hardware:** GPU-optimized (RTX 3060 12GB target)

A complete racing AI training pipeline demonstrating how Morphogen's unified operator model enables seamless integration of physics simulation, neural network inference, genetic algorithms, and real-time visualization — all in one composable graph.

**What This Demonstrates:**
- Multi-domain composition (Physics + NN + GA + Rendering + Telemetry + Recording)
- GPU-accelerated parallel evaluation (64+ agents simultaneously)
- Genetic algorithm operators for neural network evolution
- Real-time visualization and debugging
- Deterministic training with perfect reproducibility
- 10-100× performance vs traditional Unity + Python approaches

**Key Insight:** This is **one of Morphogen's strongest use cases** because it merges traditionally-separate technologies (game engine physics, ML frameworks, custom GA code) into a single clean pipeline.

**Use Cases:**
- Autonomous racing AI
- Self-driving car simulation
- Drone flight controllers
- Robot navigation
- Game AI (any physics-based agent control)
- Reinforcement learning benchmarks

**Related Documentation:**
- [reference/genetic-algorithm-operators.md](../reference/genetic-algorithm-operators.md) — GA operator catalog
- [specifications/operator-registry.md](../specifications/operator-registry.md) — Operator metadata format
- [ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md) — Unified patterns

---

### 🔥 [J-Tube Fire Pit: Multi-Physics Engineering Example](J-TUBE-FIREPIT-MULTIPHYSICS.md)

**Status:** Design Document
**Domains:** Geometry, FluidNetwork, ThermalODE, FluidJet, CombustionLight
**Complexity:** Advanced

A complete multi-physics modeling example showing how Morphogen's operator graph paradigm extends from audio/graphics into engineering physics.

**What This Demonstrates:**
- Multi-physics system modeling (draft pressure, flow networks, thermal ODEs, jets, combustion)
- Cross-domain integration (Geometry → Fluid → Thermal → Combustion → Visualization)
- Reference-based composition (anchors, frames) for physical systems
- Design optimization workflow (parameter sweeps, objective search)
- New domain requirements (FluidNetwork, ThermalODE, FluidJet, CombustionLight)

**Key Insight:** The J-tube fire pit is basically a **little multi-physics engine in steel** — and Morphogen is exactly the kind of thing that wants to model that.

**Use Cases:**
- Fire pits (secondary combustion optimization)
- Mufflers (exhaust flow and back-pressure)
- HVAC systems (air duct networks)
- Heat exchangers (thermal-fluid coupling)
- Burners (combustion quality estimation)

**Related Documentation:**
- [specifications/physics-domains.md](../specifications/physics-domains.md) — Detailed operator specifications for new physics domains
- [ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md) — Reference systems and anchors
- [architecture/domain-architecture.md](../architecture/domain-architecture.md) — Complete domain vision

---

### 🚀 [Kerbal Space Program: Multi-Domain Orbital Simulation Example](KERBAL-SPACE-PROGRAM-SIMULATION.md)

**Status:** Design Document
**Domains:** OrbitalMechanics, Aerodynamics, RocketEquation, PartsAssembly, PhysicsIntegration, FailureMechanics
**Complexity:** Advanced

Demonstrates how Morphogen can model KSP-style physics (orbital mechanics, aerodynamics, rocket staging, part assembly) and become a framework for aerospace simulations.

**What This Demonstrates:**
- Real-time multi-domain physics (orbits, aero, propulsion, structures)
- Part-based assembly system using operator graphs
- GPU-accelerated physics integration
- Level-of-detail physics switching (patched conics ↔ N-body)
- Cross-domain integration (Geometry/TiaCAD → Physics → Audio → Visualization)
- Educational platform for orbital mechanics and aerospace engineering

**Key Insight:** KSP's entire gameplay loop maps perfectly onto Morphogen's domain architecture — proving Morphogen can handle **real-time game physics** and **aerospace simulation**.

**Use Cases:**
- Spaceflight simulation games (KSP-like)
- Aerospace education (teaching orbital mechanics)
- Mission planning tools (trajectory optimization, launch windows)
- Satellite constellation design (Starlink, etc.)
- Research simulations (spacecraft dynamics)

**Unique Features:**
- Integration with J-tube combustion domain for realistic engine modeling
- AudioDomain integration for engine sounds and aerodynamic noise
- Procedural planet generation using NoiseDomain
- Part geometry from TiaCAD

**Related Documentation:**
- [specifications/physics-domains.md](../specifications/physics-domains.md) — Physics operator specifications
- [ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md) — Reference systems and anchors
- [specifications/geometry.md](../specifications/geometry.md) — TiaCAD integration for part geometry
- [examples/canonical/03_fluid_to_sound.py](../../examples/canonical/03_fluid_to_sound.py) — Fluid dynamics → acoustics → audio (working, runnable)

---

## Example Categories

### 🎨 Audio/Visual Examples
*Coming soon...*
- Real-time audio synthesis with field-driven modulation
- Fractal visualization with parameter animation
- Cross-domain: audio → visual (sonification)

### 🔬 Physics Simulations
- **J-Tube Fire Pit** (current) — Multi-physics thermal-fluid system
- **Racing AI Pipeline** (current) — Racing car dynamics with neural control
- **Kerbal Space Program Physics** (current) — Orbital mechanics, aerodynamics, rocket staging

*Coming soon...*
- N-body gravity simulation with Barnes-Hut optimization
- Fluid simulation (Navier-Stokes on GPU)
- Particle-field coupling (PIC/FLIP methods)

### 🏗️ Engineering Design
- **J-Tube Fire Pit** (current) — Parametric design with optimization
- **Kerbal Space Program** (current) — Rocket design and mission planning

*Coming soon...*
- Muffler design (exhaust flow and acoustics)
- Heat exchanger optimization
- HVAC system balancing

### 📊 Finance/Quantitative
*Coming soon...*
- Option pricing with Monte Carlo
- Stochastic volatility models (Heston)
- PDE-based pricing (finite difference methods)

### 🤖 ML/Neural
- **Racing AI Pipeline** (current) — Genetic algorithm + neural network evolution

*Coming soon...*
- Neural fields (NeRF-style)
- Fourier Neural Operators for PDE solving
- Differentiable physics simulation

---

## How to Use These Examples

### 1. **Learning Morphogen's Architecture**

Start with the J-Tube Fire Pit example to understand:
- How operators compose into pipelines
- How domains integrate via references/anchors
- How parameters flow through the graph
- How optimization wraps around pipelines

### 2. **Designing New Domains**

Use the physics domains (FluidNetwork, ThermalODE, etc.) as templates:
- Study operator specifications
- Note cross-domain coupling patterns
- Understand determinism requirements
- See MLIR lowering strategies

### 3. **Building Applications**

Examples show end-to-end workflows:
- Geometry setup
- Simulation pipeline
- Visualization
- Optimization
- Export/results

### 4. **Contributing Examples**

We welcome new examples! Guidelines:
- Focus on **multi-domain integration**
- Include complete **operator specifications**
- Show **cross-domain flows** clearly
- Provide **validation** (analytical, experimental, or CFD comparison)
- Document **use cases** and **generalizations**

**Template structure:**
```markdown
# Example Title

## 1. Physical/Conceptual System
## 2. Morphogen Modeling Pipeline
## 3. Domain Requirements
## 4. Operator Specifications
## 5. Complete Code Example
## 6. Validation & Testing
## 7. Generalizations & Extensions
```

---

## Example Complexity Levels

| Level | Description | Example |
|-------|-------------|---------|
| **Basic** | Single domain, few operators | Sine wave oscillator |
| **Intermediate** | 2-3 domains, cross-domain flows | Audio → Visual sonification |
| **Advanced** | 4+ domains, multi-physics | J-Tube Fire Pit |
| **Research** | Novel domain combinations | Differentiable physics + ML |

---

## Related Documentation

### Specifications
- [specifications/physics-domains.md](../specifications/physics-domains.md) — Physics domain operators
- [specifications/operator-registry.md](../specifications/operator-registry.md) — Operator metadata format
- [specifications/geometry.md](../specifications/geometry.md) — Geometry domain (TiaCAD patterns)
- [specifications/coordinate-frames.md](../specifications/coordinate-frames.md) — Frames and anchors

### Architecture
- [ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md) — Unified patterns
- [architecture/domain-architecture.md](../architecture/domain-architecture.md) — Complete domain vision

### Guides
- [guides/domain-implementation.md](../guides/domain-implementation.md) — Implementing new domains

---

## Questions?

- **Architecture questions:** See [ADR-002](../adr/002-cross-domain-architectural-patterns.md)
- **Domain design:** See [architecture/domain-architecture.md](../architecture/domain-architecture.md)
- **Operator specs:** See [specifications/operator-registry.md](../specifications/operator-registry.md)

---

**Morphogen is not a library. Morphogen is a platform.**
