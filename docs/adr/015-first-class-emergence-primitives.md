# ADR-015: First-Class Emergence Primitives

**Status:** Proposed
**Date:** 2026-01-22
**Authors:** Scott Sentell, TIA
**Relates to:** ADR-005 (Emergence Domain), ADR-012 (Universal Domain Translation)

---

## Context

Morphogen's vision is to be the right substrate for emergence-heavy domains: biology, materials science, engines, quantum systems. After deep analysis of what these domains share, we identified a gap: many concepts critical to reasoning about emergence are implicit in our current architecture rather than explicit, first-class objects.

Users of emergence-heavy systems reason about *forms* (attractors, phenotypes, stable configurations), not raw trajectories. They care about *what stabilizes* and *what constrains what can stabilize*. Our current API exposes continuous dynamics well, but the emergent structures those dynamics produce remain second-class citizens—side effects rather than objects.

### The Problem

When a biologist thinks "phenotype," a materials scientist thinks "phase," or an engineer thinks "resonance mode," they're all thinking about the same abstract concept: **a stable attractor in a dynamical system**. Currently, Morphogen can simulate the dynamics that produce these, but cannot:

- Name or reference the attractor directly
- Query which basin of attraction a state occupies
- Detect regime transitions as events
- Analyze parameter sensitivity of stable forms
- Distinguish noise-as-exploration from noise-as-error

This forces users to build these concepts themselves, repeatedly, across domains.

### Guiding Principle

After extensive analysis, we arrived at a unifying design rule:

> **Promote to first-class whatever the system stabilizes into—and whatever constrains what can stabilize.**

This principle works across biology, quantum mechanics, materials science, engines, and AI substrates. It provides a compass for all decisions in this ADR.

---

## Decision

We will promote the following concepts to first-class status in Morphogen's type system, operators, and runtime.

### 1. Attractors, Regimes, and Stability (Highest Priority)

**Rationale:** Users reason about *forms*, not raw trajectories. Form cannot be a side effect of the API.

**First-class objects:**

| Concept | Description | Example Uses |
|---------|-------------|--------------|
| `Attractor` | A stable state or trajectory the system converges to | Fixed points, limit cycles, strange attractors |
| `Basin` | Region of state space that flows to a given attractor | Phenotype domains, phase regions |
| `StabilityMetric` | Quantified measure of attractor robustness | Lyapunov exponents, basin volume |
| `Regime` | Named, labeled dynamical mode | "striped", "oscillatory", "steady", "chaotic" |
| `Bifurcation` | Qualitative change in attractor structure | Saddle-node, Hopf, pitchfork transitions |

**Domain mappings:**
- Biology: Attractors = phenotypes
- QM/Materials: Attractors = phases, eigenstates
- Engines: Attractors = resonance modes
- Neural: Attractors = learned representations

### 2. Parameters as Semantic Objects

**Rationale:** Genes, materials, geometries, and engines are tuned via parameters with *roles*. Parameters are not just scalars—they carry meaning.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `Parameter` | Named value with semantic role (rate, gain, threshold, diffusion) |
| `ParameterFamily` | Related parameters that vary together |
| `ParameterConstraint` | Valid relationships between parameters |
| `SensitivityAnalysis` | How attractors change with parameter variation |
| `RobustnessMetric` | How stable attractors are to parameter perturbation |
| `ParameterDrift` | Slow parameter change over time (evolution, wear) |

**Enables:**
- Genotype → phenotype reasoning
- Design space exploration
- Evolvability analysis
- Engineering intuition about tolerances

### 3. Noise as Structured, Typed Phenomenon

**Rationale:** Noise is not an error—but it is also not free. Different domains treat noise fundamentally differently.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `NoiseSource` | Explicit, named source of stochasticity |
| `NoiseInjectionPoint` | Where noise enters: state, parameters, timing |
| `NoiseSchedule` | How noise amplitude changes (early exploration → late suppression) |
| `NoiseSuppression` | Mechanisms that reduce noise effects |

**Domain-specific semantics:**
- Biology: Noise exploited for exploration
- QM/Materials: Noise suppressed (decoherence channels)
- Engines: Noise controlled within tolerances

### 4. Time-Scale Separation

**Rationale:** Emergence collapses without time-scale separation. Fast dynamics create the substrate; slow dynamics select and stabilize.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `TimeScale` | Explicit fast / medium / slow classification |
| `ScaleCoupling` | Rules for how scales interact |
| `Freeze` | Treat slow variable as constant during fast dynamics |
| `Relax` | Allow slow variable to evolve |
| `Hysteresis` | Memory effects from scale separation |

**Critical for:**
- Development (fast) vs physiology (medium) vs evolution (slow)
- Engine cycles (fast) vs heat accumulation (slow)
- Learning (slow) vs action (fast)

### 5. Discrete Commitment Events

**Rationale:** Real systems *commit*. The continuous → discrete bridge is where decisions become irreversible.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `CommitmentEvent` | Irreversible transition between regimes |
| `Bistability` | Two stable attractors with hysteresis between them |
| `Multistability` | Multiple coexisting attractors |
| `GuardedTransition` | Transition that requires conditions to be met |
| `EnergyInjection` | Event-driven energy input that triggers transitions |

**Examples:**
- Cell differentiation (irreversible commitment)
- Engine ignition (combustion event)
- Quantum measurement (wavefunction collapse)
- Phase change in materials (crystallization)

### 6. Constraints as Explicit Objects

**Rationale:** Constraints do most of the explanatory work. They explain why forms repeat and why alternatives never appear.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `ConservationLaw` | Quantity that must remain constant |
| `ResourceBudget` | Limited resource that constrains dynamics |
| `BoundaryCondition` | Fixed values at system boundaries |
| `TopologicalConstraint` | Connectivity requirements |
| `ForbiddenRegion` | Areas of state space that cannot be entered |

**Explains:**
- Why certain forms are inevitable
- Why certain alternatives are impossible
- How constraints channel dynamics toward specific attractors

### 7. Cycles and Phase-Locked Dynamics

**Rationale:** Many real systems are fundamentally cyclic. Engines, hearts, oscillators, seasons.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `Cycle` | Periodic trajectory as first-class object |
| `Phase` | Position within a cycle (0 to 2π) |
| `PhaseLocking` | Synchronization between cycles |
| `PeriodicBoundary` | Boundary conditions for cyclic domains |
| `HarmonicRelation` | Integer ratio relationships between frequencies |

**Essential for:**
- Engine modeling (thermodynamic cycles)
- Biological rhythms (circadian, cardiac)
- Coupled oscillators
- Resonant structures

### 8. Coarse-Graining and Macrostates

**Rationale:** Traits, sounds, phases, behaviors are **macrostates**, not raw fields. Users need observer-level descriptions.

**First-class objects:**

| Concept | Description |
|---------|-------------|
| `CoarseGrain` | Operator that maps microstates to macrostates |
| `Macrostate` | Aggregate description (temperature, pressure, phenotype) |
| `Invariant` | Feature that persists across microstate variation |
| `ObserverProjection` | Explicit mapping to human-relevant quantities |

**Prevents:**
- Intent smuggling (reading goals into dynamics)
- Over-interpretation of transient states
- Confusing representation with dynamics

---

## Philosophical Anchors

These are not changes but **constraints** that all implementations must respect:

### 1. Continuous Dynamics Are Primary

- Fields, PDEs, operators, transforms
- Discrete structure emerges from stability, not encoding
- We simulate dynamics; forms emerge

### 2. Discrete Outcomes Support Control, Memory, and Commitment

- Not as symbolic plans
- As stabilized modes, thresholds, phase transitions
- Discreteness is earned, not assumed

### 3. Emergence Is Not Magic

- It is constrained instability
- It should be analyzable, classifiable, and reproducible
- "Emergence" is a phenomenon to study, not a label to apply

### 4. Determinism at the Execution Layer

- Especially where actions, commitments, or irreversibility occur
- Stochasticity is explicit and controlled
- Reproducibility is non-negotiable

---

## What We Will NOT Promote

To preserve conceptual integrity:

| Anti-pattern | Why it breaks alignment |
|--------------|------------------------|
| Genes as code | Conflates parameter with program |
| Symbolic plans or goals | Smuggles intent into dynamics |
| Agent-style decision logic | Wrong abstraction for continuous systems |
| Optimization everywhere | Not all systems optimize; many just stabilize |
| "Emergence" as marketing | Degrades a precise concept |
| Default noise in execution | Noise must be explicit and justified |

---

## Domain-Specific Elevation Sets

### Biology-Focused

- Attractors = phenotypes (explicit mapping)
- Noise as developmental exploration
- Canalization metrics (robustness to perturbation)
- Parameter drift as evolution
- Developmental regime maps

### Quantum / Materials Science

- Hilbert spaces and operators
- Spectra and eigenmodes
- Symmetry groups as constraints
- Phase regimes and energy gaps
- Decoherence channels (typed noise suppression)

### Engines / Physical Systems

- Cycles and phase as primary objects
- Impulse events (combustion, ignition)
- Energy accounting (conservation constraints)
- Boundary-driven resonance
- Perceptual macrostates (sound, heat patterns)

---

## Consequences

### Positive

1. **Users can reason at the right level** — about forms, not just trajectories
2. **Cross-domain patterns become visible** — attractors in biology = phases in materials
3. **Analysis tools become possible** — bifurcation diagrams, basin maps, sensitivity analysis
4. **Determinism is preserved** — stochasticity explicit, execution reproducible
5. **Educational value** — teaches emergence properly, not as magic

### Negative

1. **API surface grows** — more concepts to learn
2. **Implementation complexity** — stability analysis is non-trivial
3. **Performance considerations** — some analyses are expensive
4. **Migration effort** — existing code may need updates

### Neutral

1. **Backward compatible** — new concepts are additions, not replacements
2. **Incremental adoption** — domains can adopt first-class primitives gradually

---

## Implementation Priority

| Priority | Concept | Rationale |
|----------|---------|-----------|
| P0 | Attractors & Regimes | Most requested, highest leverage |
| P0 | Constraints | Already implicit; make explicit |
| P1 | Parameters as semantic objects | Enables design exploration |
| P1 | Commitment events | Continuous→discrete bridge |
| P1 | Time-scale separation | Critical for multi-scale systems |
| P2 | Noise typing | Important but can be deferred |
| P2 | Cycles & phase | Domain-specific priority |
| P2 | Coarse-graining | Requires attractor infrastructure first |

---

## The One-Paragraph Synthesis

> Morphogen should evolve toward a system where stability, constraints, time scales, noise, and phase transitions are first-class citizens. Continuous fields generate possibilities; constraints and events select discrete outcomes; macrostates describe what persists. The system's job is not to simulate everything, but to make inevitabilities visible.

This is the throughline from Turing to engines to plants to materials.

---

## References

- Turing, A. M. (1952). "The Chemical Basis of Morphogenesis"
- Kauffman, S. (1993). "The Origins of Order"
- Strogatz, S. (2015). "Nonlinear Dynamics and Chaos"
- ADR-005: Emergence Domain
- ADR-012: Universal Domain Translation
- `docs/philosophy/categorical-structure.md`
- `docs/specifications/emergence.md`

---

## Appendix: The Unifying Design Rule

If Morphogen needs a single internal compass for future decisions:

> **Promote to first-class whatever the system stabilizes into—and whatever constrains what can stabilize.**

This rule works universally because:

- **Biology**: Phenotypes are what stabilizes; development constrains what can stabilize
- **Quantum mechanics**: Eigenstates are what stabilizes; Hamiltonians constrain what can stabilize
- **Materials**: Phases are what stabilizes; thermodynamics constrains what can stabilize
- **Engines**: Operating modes are what stabilizes; geometry and fuel constrain what can stabilize

When evaluating any future feature request, ask: "Does this help users see what stabilizes, or what constrains stability?" If yes, it belongs. If no, it probably doesn't.
