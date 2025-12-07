# Architectural Evolution Roadmap: Path to Maximum Capability & Composability

**Status:** Strategic Planning Document
**Date:** 2025-11-21
**Purpose:** Define the architectural evolution path from v0.11.0 → v1.0 → v2.0, with focus on composability, capability growth, and domain boundaries

---

## Executive Summary

Morphogen's evolution is guided by three principles:
1. **Composability First** - Every feature must compose cleanly with existing features
2. **Domain Boundaries** - Clear separation between kernel, domains, and frontends
3. **Capability Accretion** - Add capabilities without breaking existing code

This document defines the architectural decisions, boundaries, and growth patterns that will guide Morphogen toward its most capable and composable form.

---

## Table of Contents

1. [Architectural North Star](#architectural-north-star)
2. [Domain Boundary Principles](#domain-boundary-principles)
3. [Composability Framework](#composability-framework)
4. [Evolution Phases](#evolution-phases)
5. [Critical Architectural Decisions](#critical-architectural-decisions)
6. [Domain Growth Strategy](#domain-growth-strategy)
7. [Boundary Point Catalog](#boundary-point-catalog)
8. [Research & Discussion Tracking](#research--discussion-tracking)

---

## Architectural North Star

### The Vision Statement

> **Morphogen is the universal substrate for deterministic, multi-domain computation where professional domains that have never talked before can seamlessly compose through a single type system, scheduler, and compilation pipeline.**

### What "Most Capable" Means

**Capability = Coverage × Depth × Integration**

- **Coverage**: Span of domains (audio, physics, circuits, chemistry, geometry, optimization, etc.)
- **Depth**: Sophistication within each domain (not toy implementations, but production-grade)
- **Integration**: Ability to compose across domains (circuit → audio, geometry → physics, etc.)

### What "Most Composable" Means

**Composability = Predictability × Safety × Orthogonality**

- **Predictability**: Composition behaves as expected (no hidden interactions)
- **Safety**: Type system prevents invalid compositions (compile-time errors, not runtime)
- **Orthogonality**: Features are independent (adding domain X doesn't break domain Y)

---

## Domain Boundary Principles

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────┐
│         LAYER 3: FRONTENDS                  │
│  Morphogen.Audio | RiffStack | Future DSLs │
│                                             │
│  Purpose: Human ergonomics & syntax         │
│  Evolution: Additive (new frontends)        │
│  Stability: High churn OK                   │
└─────────────────┬───────────────────────────┘
                  │ Graph IR
                  ▼
┌─────────────────────────────────────────────┐
│         LAYER 2: DOMAIN LIBRARIES           │
│  Audio | Physics | Chemistry | Circuits    │
│  Vision | Geometry | Optimization | ...     │
│                                             │
│  Purpose: Specialized operators             │
│  Evolution: Accretive (add domains)         │
│  Stability: Medium (deprecation allowed)    │
└─────────────────┬───────────────────────────┘
                  │ Operator Registry
                  ▼
┌─────────────────────────────────────────────┐
│         LAYER 1: KERNEL                     │
│  Types | Units | Scheduler | Transforms    │
│  MLIR | Determinism | Operator Registry    │
│                                             │
│  Purpose: Universal abstractions            │
│  Evolution: Minimal (extremely stable)      │
│  Stability: Zero breaking changes           │
└─────────────────────────────────────────────┘
```

### Boundary Rules

#### Kernel → Domain Libraries Boundary

**Kernel provides (immutable contract):**
- Type system (`Stream<T, Domain, Rate>`, `Field<T, Space>`, `Agents<T>`)
- Physical units system (SI + custom units with dimensional analysis)
- Multirate scheduler (sample-accurate, deterministic, fence-based)
- Transform dialect (FFT, STFT, DCT, wavelet, Laplace, etc.)
- Operator registry interface (metadata, lowering, profiling)
- MLIR dialect infrastructure (graph IR → executable code)
- Determinism profiles (strict, repro, live)

**Domain libraries consume (stable API):**
- Register operators via registry
- Declare type signatures with units
- Specify lowering templates
- Define cross-domain couplings

**Key Principle:** Domain libraries are **clients** of the kernel. They never modify kernel internals.

#### Domain Libraries → Frontends Boundary

**Domain libraries provide:**
- Operator functions with rich type signatures
- Documentation (docstrings, examples)
- Performance characteristics (O(n), O(n log n), etc.)
- Determinism guarantees

**Frontends provide:**
- Human-friendly syntax
- Ergonomic composition patterns
- Error messages and diagnostics
- Domain-specific abstractions (scenes, patches, circuits)

**Key Principle:** Multiple frontends can emit the same domain library operators. Frontends are **translators**, not feature providers.

---

## Composability Framework

### The Four Pillars of Composability

#### 1. Type-Safe Cross-Domain Composition

**Principle:** The type system prevents invalid compositions at compile time.

**Examples:**
```morphogen
// ✅ Valid: Audio rate matches
let audio_a : Stream<f32, audio, 48kHz>
let audio_b : Stream<f32, audio, 48kHz>
let mixed = audio.mix(audio_a, audio_b)  // OK

// ❌ Invalid: Rate mismatch caught at compile time
let control : Stream<f32, control, 60Hz>
let mixed = audio.mix(audio_a, control)  // TYPE ERROR

// ✅ Valid: Automatic resampling with explicit annotation
let mixed = audio.mix(audio_a, resample(control, 48kHz))  // OK
```

**Implementation Requirements:**
- [ ] Rate checking in type system (v0.12.0)
- [ ] Domain compatibility matrix (v0.12.0)
- [ ] Unit dimensional analysis (v0.13.0)
- [ ] Explicit conversion operators (v0.12.0)

#### 2. Explicit Cross-Domain Coupling

**Principle:** Domain boundaries are crossed explicitly, never implicitly.

**Examples:**
```morphogen
// Field → Agent coupling (explicit)
agents = agents.sample_field(temperature_field, property="heat")

// Agent → Field coupling (explicit)
field = field.deposit_from_agents(agents, property="density", method="ngp")

// Circuit → Audio coupling (explicit)
audio_signal = circuit.to_audio(circuit_node="output", sample_rate=48kHz)

// Geometry → Physics coupling (explicit)
physics_mesh = geometry.to_physics_mesh(collision_geometry, resolution=0.1)
```

**Anti-pattern:**
```morphogen
// ❌ BAD: Implicit coupling (magic behavior)
audio_signal = some_circuit_node  // How does this work? What sample rate?
```

**Implementation Requirements:**
- [ ] Coupling operator catalog (v0.12.0)
- [ ] Coupling type signatures (v0.12.0)
- [ ] Coupling performance characteristics (v0.13.0)
- [ ] Documentation for all coupling operators (v0.12.0)

#### 3. Multirate Scheduling Coordination

**Principle:** Multiple rates coexist with explicit synchronization points.

**Examples:**
```morphogen
use audio, physics, visual

@state audio_buffer : AudioBuffer
@state particle_system : Agents<Particle>
@state frame_buffer : Image

// Three rates: audio (48kHz), physics (240Hz), visual (60Hz)
flow(dt=0.01) {
    // Physics runs every step (100Hz base rate)
    particle_system = physics.integrate(particle_system, dt)

    // Audio runs every 2nd step (48kHz / (1/0.01s) = 2 samples/step)
    audio_buffer = audio.synthesize_from_particles(particle_system)

    // Visual runs every 10th step (60Hz / (1/0.01s) = 1 frame every 0.166s)
    if step % 10 == 0:
        frame_buffer = visual.render(particle_system)
}
```

**Implementation Requirements:**
- [ ] LCM-based rate scheduling (v0.12.0)
- [ ] Rate validation at compile time (v0.12.0)
- [ ] Resampling operators (v0.12.0)
- [ ] Performance analysis tools (v0.13.0)

#### 4. Deterministic Composition

**Principle:** Composed systems are as deterministic as their components.

**Determinism Profiles:**

| Profile | Guarantee | Use Case | Cross-Domain Behavior |
|---------|-----------|----------|----------------------|
| **strict** | Bit-exact | Regression testing, verification | All domains must support strict mode |
| **repro** | Deterministic FP | Scientific computing, ML training | Domains can vary in precision but must be consistent |
| **live** | Latency-first | Real-time performance, interactive | Non-deterministic allowed, best-effort |

**Example:**
```morphogen
// Set profile globally
profile: strict

use audio, physics

// ALL operations now use strict determinism
@state particles : Agents<Particle>
@state audio_buffer : AudioBuffer

flow(dt=0.01) {
    particles = physics.integrate(particles, dt)  // Strict mode
    audio_buffer = audio.synthesis(particles)      // Strict mode
    // Identical seed → identical output, always
}
```

**Implementation Requirements:**
- [ ] Profile propagation through composition (v0.12.0)
- [ ] Per-operator profile metadata (v0.11.0 ✅)
- [ ] Profile validation pass (v0.12.0)
- [ ] Determinism testing framework (v0.12.0)

---

## Evolution Phases

### Phase 1: Foundation (v0.11.0 → v0.12.0) - **Q4 2025**

**Goal:** Solidify kernel boundaries and type system foundations

**Deliverables:**
1. **Type System Hardening**
   - [ ] Rate checking with automatic LCM scheduling
   - [ ] Unit dimensional analysis (compile-time errors for `m + s`)
   - [ ] Domain compatibility matrix
   - [ ] Cross-domain type signature validation

2. **Operator Registry Completion**
   - [ ] All 40 domains registered with metadata
   - [ ] Performance characteristics documented (O(n), memory, etc.)
   - [ ] Determinism profile for every operator
   - [ ] Lowering templates for all critical operators

3. **Coupling Operators**
   - [ ] Field ↔ Agent coupling (sample, deposit)
   - [ ] Audio ↔ Circuit coupling (to_audio, from_audio)
   - [ ] Geometry ↔ Physics coupling (to_mesh, boundary_conditions)
   - [ ] Visual ↔ Field coupling (colorize, render)

4. **Documentation**
   - [ ] Domain boundary reference guide
   - [ ] Composability patterns catalog
   - [ ] Type system specification (complete)
   - [ ] Operator registry API documentation

**Success Criteria:**
- All existing examples compile and run
- Type errors caught at parse time (not runtime)
- Determinism tests pass for all domains
- Clear "what can compose with what" documentation

---

### Phase 2: Domain Expansion (v0.12.0 → v0.13.0) - **Q1 2026**

**Goal:** Add critical professional domains with production-grade depth

**New Domains:**
1. **Circuit Domain** (⭐ Highest Priority)
   - DC/AC/transient analysis
   - Guitar pedal simulation (circuit → audio pipeline)
   - PCB parasitic extraction (geometry → circuit)
   - Target: 40+ operators, 100+ tests

2. **Fluid Domain**
   - 1D compressible flow (Navier-Stokes)
   - Acoustic coupling (fluid → sound)
   - Exhaust modeling (thermodynamics + acoustics)
   - Target: 30+ operators, 80+ tests

3. **Geometry Domain**
   - Declarative CAD (primitives, booleans, patterns)
   - Reference-based composition (TiaCAD-inspired)
   - Mesh generation (for physics/CFD)
   - Target: 50+ operators, 120+ tests

4. **Symbolic Math Domain**
   - SymPy integration
   - Symbolic + numeric execution
   - Equation solving and simplification
   - Target: 25+ operators, 60+ tests

**Cross-Domain Integration Examples:**
- Circuit pedal design → hear audio output
- J-tube geometry → fluid flow → thermal analysis
- Symbolic PDE → numeric field solver
- Mesh geometry → rigid body collision → visualization

**Success Criteria:**
- At least 2 professional "killer app" examples
- Cross-domain workflows validated
- Performance acceptable for professional use
- Documentation sufficient for domain experts

---

### Phase 3: Language Evolution (v0.13.0 → v1.0.0) - **Q2 2026**

**Goal:** Reach language 1.0 with stable syntax and advanced features

**Language Features:**
1. **Transform Tracking**
   - Type system tracks domain (time/frequency/space/k-space)
   - Automatic transform insertion where safe
   - Transform cost estimation
   - Example: `field_freq = fft(field_time)  # Type: Field<f32, frequency>`

2. **Category Theory Optimization**
   - Functorial composition (automatic fusion)
   - Monoidal parallelization
   - Natural transformation recognition
   - Verified composition properties

3. **Plugin System**
   - User-defined domains
   - Custom operator registration
   - Community domain marketplace
   - Isolated domain loading

4. **Advanced Types**
   - Dependent types for dimensions (`Field<T, shape=[N, M]>`)
   - Refinement types (`x : f32 where x > 0`)
   - Effect system (explicit I/O, mutation, randomness)

**Success Criteria:**
- Stable language specification (no breaking changes post-1.0)
- Transform tracking validated on real examples
- Plugin system demonstrated with community domain
- Advanced type features improve safety without complexity

---

### Phase 4: Compilation & Performance (v1.0.0 → v1.5.0) - **Q3-Q4 2026**

**Goal:** Production-grade compilation and GPU acceleration

**Compilation Improvements:**
1. **MLIR Optimization Pipeline**
   - Operator fusion passes
   - Polyhedral optimization for nested loops
   - Auto-vectorization (SIMD)
   - Constant folding and DCE

2. **GPU Acceleration**
   - Field operations → GPU kernels
   - Agent systems → GPU compute
   - Transform operations → cuFFT/rocFFT
   - Memory management (unified memory, async)

3. **JIT Compilation**
   - Hot-path JIT for live performance
   - Adaptive compilation (profile-guided)
   - Incremental recompilation
   - Low-latency mode for interactive use

4. **Multi-Backend Support**
   - CPU: LLVM → native code
   - GPU: CUDA, ROCm, Metal, Vulkan Compute
   - FPGA: Verilog generation (research)
   - Neuromorphic: Event-driven lowering (research)

**Success Criteria:**
- 10-100x speedup on GPU vs CPU for field operations
- Real-time audio processing on CPU (< 5ms latency)
- JIT latency < 100ms for typical programs
- Multi-GPU support for large simulations

---

### Phase 5: Ecosystem Maturity (v1.5.0 → v2.0.0) - **2027**

**Goal:** Complete ecosystem with community, tooling, and professional adoption

**Ecosystem Components:**
1. **Developer Tools**
   - VSCode extension (syntax highlighting, autocomplete, debugging)
   - Morphogen Language Server Protocol (LSP)
   - Interactive notebook (Jupyter kernel)
   - Profiling and optimization tools

2. **Community Infrastructure**
   - Domain marketplace (community domains)
   - Example gallery (50+ curated examples)
   - Forum and Discord
   - Monthly releases with changelogs

3. **Professional Integration**
   - MATLAB bridge (import/export)
   - Python interop (call Morphogen from Python)
   - C++ library export
   - Cloud execution (AWS, GCP, Azure)

4. **Educational Resources**
   - Online book / interactive tutorial
   - University course materials
   - Video series (YouTube)
   - Conference talks and papers

**Success Criteria:**
- 1000+ GitHub stars
- 50+ community domains
- 10+ companies using in production
- Published in academic conferences

---

## Critical Architectural Decisions

### Decision Framework

When considering new features, domains, or changes, evaluate against:

1. **Composability Impact**: Does this make composition harder or easier?
2. **Boundary Respect**: Does this blur kernel/domain/frontend boundaries?
3. **Determinism**: Can this work in all three profiles (strict/repro/live)?
4. **Performance**: What's the asymptotic complexity? GPU-friendly?
5. **Orthogonality**: Does this interact with existing features?
6. **Value**: Does this enable new use cases or just sugar?

### Standing Decisions

#### SD-1: Kernel Immutability Post-1.0

**Decision:** After v1.0, the kernel API is immutable. No breaking changes ever.

**Rationale:**
- Users must trust that Morphogen code written today will work forever
- Domain libraries can be added/deprecated without kernel changes
- Frontends can evolve independently

**Implications:**
- Get kernel right before 1.0 (v0.12-0.13 critical)
- Extensive testing and formal verification before 1.0
- Any mistakes in kernel live forever (design carefully)

#### SD-2: No Global State

**Decision:** All state is explicit (`@state` declarations, function arguments). No hidden global variables.

**Rationale:**
- Determinism requires explicit state
- Hot-reload requires knowing what state exists
- Parallelization requires independent state scopes

**Implications:**
- No `global` keyword
- No singleton patterns
- All RNG via explicit `RNG` objects
- No hidden caches or memoization (unless explicitly annotated)

#### SD-3: Cross-Domain Coupling is Explicit

**Decision:** Never auto-convert between domains. Always require explicit coupling operators.

**Rationale:**
- Prevents accidental performance cliffs (implicit resampling is expensive)
- Makes data flow visible (traceability for debugging)
- Enables optimization (compiler sees coupling points)

**Implications:**
- `audio.to_control()`, `field.to_agents()`, etc. required
- Type errors when mixing domains
- Verbose at first, but clear and safe

#### SD-4: Transform Spaces are First-Class

**Decision:** Frequency domain, k-space, phase space, etc. are first-class types, not hidden.

**Rationale:**
- Many algorithms only work in specific domains (FFT for convolution)
- Explicit transforms enable cost analysis
- Matches mathematical practice (Fourier analysis, Laplace transforms)

**Implications:**
- `Stream<f32, frequency>` is different type from `Stream<f32, time>`
- Explicit `fft()` / `ifft()` required
- Type errors prevent frequency-domain bugs

#### SD-5: Determinism is Default, Nondeterminism is Explicit

**Decision:** Default to strict determinism. Require explicit annotation for nondeterminism.

**Rationale:**
- Reproducibility is critical for science, debugging, regression testing
- Most users want determinism without thinking about it
- Nondeterminism should be a conscious choice

**Implications:**
- All RNG via `rng: RNG` parameter (seeded, deterministic)
- I/O operations marked as `IO` effect (future)
- Profile system allows relaxing to `live` mode when needed

---

## Domain Growth Strategy

### How to Decide: Add New Domain or Extend Existing?

**Add New Domain if:**
- Fundamentally different mathematical structure (e.g., graph theory vs differential equations)
- Different performance characteristics (sparse vs dense, iterative vs closed-form)
- Different operator semantics (stochastic vs deterministic, discrete vs continuous)
- Minimal overlap with existing domains (< 20% operator similarity)

**Extend Existing Domain if:**
- Same mathematical framework (e.g., add new filter types to Audio)
- Same performance profile (e.g., add new stencils to Field)
- Natural extension of existing operators (e.g., add 3D support to existing 2D operators)
- High overlap (> 50% of functionality already exists)

### Domain Prioritization Matrix

| Domain | Professional Impact | Implementation Effort | Cross-Domain Synergy | Priority |
|--------|---------------------|----------------------|---------------------|----------|
| **Circuit** | 🔥🔥🔥 High (EE, audio) | 🟡 Medium (1 month) | ⭐⭐⭐ Excellent (Audio, Geometry) | **P0** |
| **Fluid** | 🔥🔥🔥 High (Aero, automotive) | 🔴 High (2 months) | ⭐⭐⭐ Excellent (Acoustics, Thermal) | **P0** |
| **Geometry** | 🔥🔥 Medium-High (CAD, 3D printing) | 🔴 High (2 months) | ⭐⭐⭐ Excellent (Physics, Visual) | **P1** |
| **Symbolic** | 🔥 Medium (Education, research) | 🟡 Medium (1 month) | ⭐⭐ Good (all domains) | **P1** |
| **Control** | 🔥🔥 Medium-High (Robotics, engineering) | 🟢 Low (2 weeks) | ⭐⭐ Good (Physics, Optimization) | **P2** |
| **Finance** | 🔥 Medium (Quant, trading) | 🟡 Medium (1 month) | ⭐ Limited (Stochastic only) | **P3** |

### Domain Template Checklist

When adding a new domain, ensure:

**Documentation:**
- [ ] Domain specification (ADR + `docs/specifications/<domain>.md`)
- [ ] Operator catalog with signatures, semantics, performance
- [ ] At least 5 examples demonstrating key operators
- [ ] Cross-domain coupling documentation

**Implementation:**
- [ ] Domain module in `morphogen/stdlib/<domain>.py`
- [ ] Type signatures with units where applicable
- [ ] Determinism profile for all operators
- [ ] Performance characteristics (O(n), memory, etc.)

**Testing:**
- [ ] Unit tests for all operators (>90% coverage)
- [ ] Integration tests with other domains (if applicable)
- [ ] Determinism tests (same seed → same output)
- [ ] Performance benchmarks

**MLIR Integration:**
- [ ] Lowering templates for critical operators
- [ ] Optimization passes (if applicable)
- [ ] GPU support plan (future)

---

## Boundary Point Catalog

### Active Boundary Points (Need Ongoing Discussion)

#### BP-1: Field-Agent Coupling Performance

**Question:** What's the optimal algorithm for particle-in-cell (PIC) transfer?

**Current State:** NGP (Nearest Grid Point) implemented, works but first-order accurate

**Alternatives:**
- CIC (Cloud-In-Cell): Better accuracy, 2x cost
- TSC (Triangular-Shaped Cloud): Best accuracy, 4x cost
- APIC (Affine PIC): Preserves angular momentum, 3x cost

**Discussion Needed:**
- Performance tradeoffs on GPU
- Accuracy requirements for different use cases
- User control (explicit algorithm selection vs auto-selection)

**Tracking:** `docs/research/field-agent-coupling.md` (TODO)

---

#### BP-2: Transform Space Type System

**Question:** How explicit should transform space tracking be?

**Current State:** No transform tracking in type system

**Proposal (v0.13.0):**
```morphogen
// Explicit transform spaces in types
temp : Field<f32, space>         # Spatial domain
temp_freq : Field<f32, k-space>  # Frequency domain

// Transform operators change domain
temp_freq = fft(temp)            # space → k-space
temp_filtered = ifft(temp_freq * kernel)  # k-space → space
```

**Alternatives:**
1. **Implicit tracking** (compiler infers domain)
2. **No tracking** (user responsible)
3. **Explicit as above** (proposed)

**Discussion Needed:**
- Ergonomics vs safety tradeoff
- Error messages when mixing domains
- Automatic transform insertion (risky?)

**Tracking:** ADR-012 (Universal Domain Translation) - partially addresses this

---

#### BP-3: GPU Memory Model

**Question:** How should GPU memory management work?

**Current State:** CPU-only (NumPy backend)

**Alternatives:**
1. **Explicit allocation** (user calls `gpu.alloc()`, `gpu.to_cpu()`)
2. **Implicit unified memory** (compiler manages, slower but safe)
3. **Async transfers** (compiler overlaps compute + transfer, complex)

**Tradeoffs:**
- Explicit: Fastest, most control, verbose
- Implicit: Easiest, slower, hides cost
- Async: Best performance, hardest to debug

**Discussion Needed:**
- Default model for v1.0
- Profile-specific behavior (strict vs live)
- Error handling for OOM

**Tracking:** `docs/research/gpu-memory-model.md` (TODO)

---

#### BP-4: Symbolic + Numeric Integration

**Question:** How do symbolic and numeric execution interact?

**Proposal:**
```morphogen
use symbolic, numeric

# Define symbolically
@symbolic eq : Equation = "∂u/∂t = α∇²u"

# Solve numerically
@state u : Field2D<f32> = zeros((256, 256))
u = solve(eq, u, dt=0.01, alpha=0.1)  # Automatic numeric solver from symbolic
```

**Alternatives:**
1. **Separate worlds** (symbolic and numeric never mix)
2. **Explicit lowering** (`eq.to_numeric()` required)
3. **Automatic lowering** (compiler decides when to switch)

**Discussion Needed:**
- Performance of automatic symbolic → numeric
- Debugging symbolic expressions
- Type system for symbolic objects

**Tracking:** v1.0 Release Plan (Track 1 - Symbolic Execution)

---

#### BP-5: Effect System Design

**Question:** Should Morphogen have an effect system for I/O, mutation, randomness?

**Current State:** No effect tracking

**Proposal (v1.5.0):**
```morphogen
# Pure function (no effects)
fn double(x: f32) -> f32 { return x * 2.0 }

# Function with RNG effect
fn random_position(rng: RNG) -> Vec2<f32> { ... }
  # Type: RNG -> Vec2<f32> (implicit RNG effect)

# Function with I/O effect
fn load_file(path: string) -> Field2D<f32> { ... }
  # Type: string -> IO Field2D<f32>
```

**Benefits:**
- Track nondeterminism in type system
- Prevent accidental I/O in pure functions
- Enable better optimization (pure functions can be memoized)

**Concerns:**
- Complexity for users
- Verbosity (every I/O function marked)
- Compatibility with existing code

**Discussion Needed:**
- Priority (v1.0 or v2.0?)
- Syntax (explicit `IO` annotation vs inferred)
- Backward compatibility strategy

**Tracking:** `docs/roadmap/language-features.md` (Effects listed as "Under Discussion")

---

### Resolved Boundary Points (For Reference)

#### ✅ BP-R1: Determinism Profiles (Resolved v0.7.0)

**Resolution:** Three profiles (strict, repro, live) with explicit selection.

**Rationale:** Covers all use cases (testing, science, performance) without compromise.

**Reference:** `docs/specifications/profiles.md`

---

#### ✅ BP-R2: Multirate Scheduling (Resolved v0.6.0)

**Resolution:** LCM-based partitioning with explicit fence points.

**Rationale:** Deterministic, predictable, sample-accurate.

**Reference:** `docs/specifications/scheduler.md`

---

#### ✅ BP-R3: Physical Units System (Resolved v0.9.0)

**Resolution:** Units as type-level annotations, compile-time dimensional analysis (planned v0.13.0).

**Rationale:** Safety without runtime overhead.

**Reference:** `docs/specifications/level-3-type-system.md`

---

## Research & Discussion Tracking

### How to Track Architectural Discussions

**For Each Boundary Point:**
1. Create document: `docs/research/<topic>.md`
2. Track alternatives with pros/cons
3. Gather benchmarks, examples, prior art
4. Propose decision with rationale
5. Convert to ADR when decided

**Process:**
```
Boundary Point (BP-N) → Research → ADR → Implementation → Documentation
```

**Example Flow:**
```
BP-3: GPU Memory Model
  ↓
docs/research/gpu-memory-model.md (research alternatives)
  ↓
Benchmark different strategies
  ↓
ADR-013: GPU Memory Management (decision)
  ↓
Implementation in morphogen/mlir/gpu/
  ↓
Update docs/architecture/gpu-mlir-principles.md
```

### Active Research Topics

**High Priority:**
- [ ] Field-Agent coupling algorithms (BP-1)
- [ ] Transform space type system (BP-2)
- [ ] GPU memory model (BP-3)

**Medium Priority:**
- [ ] Symbolic + numeric integration (BP-4)
- [ ] Category theory optimization passes
- [ ] JIT compilation strategy

**Long Term:**
- [ ] Effect system design (BP-5)
- [ ] Quantum backend feasibility
- [ ] Neuromorphic compilation

---

## Capability Growth Principles

(See next document: `CAPABILITY_GROWTH_FRAMEWORK.md`)

---

## Summary

This roadmap defines:

1. **Architectural Boundaries** - Clear separation: kernel/domains/frontends
2. **Composability Framework** - Four pillars ensuring safe, predictable composition
3. **Evolution Phases** - v0.11 → v0.12 → v0.13 → v1.0 → v2.0
4. **Standing Decisions** - Immutable principles (kernel stability, no global state, etc.)
5. **Domain Growth** - Prioritization matrix and addition criteria
6. **Boundary Points** - Active discussion topics with research tracking

**Next Steps:**
1. Review this roadmap with stakeholders
2. Create research docs for active boundary points
3. Execute Phase 1 (v0.11 → v0.12) foundation work
4. Track progress via GitHub projects

---

**Document Status:** Draft for Review
**Review Cycle:** 2 weeks
**Next Update:** After v0.12.0 release
