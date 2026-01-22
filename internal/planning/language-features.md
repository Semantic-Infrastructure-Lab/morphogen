# Morphogen Language Features Roadmap

**Version:** 1.0
**Last Updated:** 2025-11-21
**Status:** Living Document

---

## Overview

This document provides a clear path toward the final language specification for Morphogen, outlining what features are **production-ready**, what features are **planned**, and what decisions are **finalized** vs **still under consideration**.

---

## Current Language Specification Status

**Current Version:** v0.10.0 (documented in `/SPECIFICATION.md`)
**Implementation Status:** Production-ready Python runtime

### ✅ Finalized & Production-Ready Features

These features are **stable**, **documented**, and **implemented**:

#### Core Language (v0.3.1+)
- **Temporal Model**: `flow(dt, steps)` blocks for time-stepping
- **State Management**: `@state` annotations for persistent variables
- **Type System**: Scalar types (`f32`, `f64`, `i32`, `u32`, `bool`)
- **Physical Units**: Optional unit annotations (e.g., `f32 [m/s]`)
- **Deterministic RNG**: Explicit `RNG` objects with seeding
- **Functions**: `fn` keyword for function definitions
- **Lambdas**: Anonymous functions with `|x| expr` syntax
- **Domain Imports**: `use domain1, domain2` syntax
- **Comments**: Line comments (`#`) and block comments (`/* */`)

#### Data Structures (v0.4.0+)
- **Structs**: User-defined composite types
- **Fields**: 2D/3D grid data (`Field2D<T>`, `Field3D<T>`)
- **Agents**: Sparse particle systems (`Agents<T>`)
- **Vectors**: `Vec2<T>`, `Vec3<T>` for spatial data
- **Arrays**: Fixed-size arrays `[T; N]`

#### Operators & Standard Library (v0.11.0)
- **40 Domain Modules**: Complete stdlib with 500+ operators
- **Field Operations**: `diffuse`, `advect`, `project`, `laplacian`, etc.
- **Agent Operations**: `alloc`, `map`, `filter`, `reduce`, forces
- **Audio Synthesis**: Oscillators, filters, envelopes, effects, physical modeling
- **Signal Processing**: FFT, STFT, filtering, spectral analysis
- **Visual Rendering**: Colorization, composition, video export
- **Chemistry Suite**: Molecular dynamics, quantum chemistry, thermodynamics, kinetics
- **Physics Suite**: Rigid body, integrators, acoustics, thermal, fluids
- **Graphics Suite**: Noise, palette, color, image processing, terrain generation
- **AI/Game**: Optimization, neural networks, state machines, graph algorithms
- **Computer Vision**: Edge detection, feature extraction, morphology

---

## 🎯 Strategic Language Direction

### The Vision: "One Language, Many Domains"

Morphogen is committed to being a **unified multi-domain platform** with:
1. **Cross-domain composition** — Different domains interact seamlessly
2. **Deterministic execution** — Reproducible results across platforms
3. **MLIR compilation** — High-performance GPU/CPU code generation
4. **Type safety** — Dimensional analysis and unit checking

### Path to 1.0: Language Stability

**Goal:** Stabilize core language syntax by **v1.0** (target: 2026 Q2)

**Criteria for 1.0:**
- [ ] All core syntax features finalized (no breaking changes)
- [ ] MLIR compiler production-ready (v0.7.x → v0.9.x)
- [ ] Cross-domain type checking enforced
- [ ] Physical unit validation implemented
- [ ] Standard library documented (all 40+ domains)
- [ ] Comprehensive test coverage (>90%)

---

## 🚧 Planned Features (Pre-1.0)

### High Priority (v0.12-0.15)

#### 1. **Physical Unit Checking** (v0.12)
**Status:** Syntax exists, enforcement not implemented
**Impact:** HIGH — Essential for safety in scientific/engineering domains

```morphogen
// Currently allowed but shouldn't be:
let dist : f32 [m] = 10.0
let time : f32 [s] = 2.0
let wrong = dist + time  // ERROR: incompatible units

// Should be caught at compile time
```

**Work Required:**
- Implement unit inference in type checker
- Add unit compatibility rules
- Support unit conversions (e.g., `m/s` → `km/h`)
- Error messages for unit mismatches

#### 2. **Cross-Domain Type Safety** (v0.13)
**Status:** Partial implementation, needs enforcement
**Impact:** HIGH — Enables safe cross-domain composition

```morphogen
use field, audio

@state temp : Field2D<f32 [K]> = zeros((256, 256))

// Should type-check coupling between domains:
let sound = temp.to_audio()  // Field → Audio coupling
```

**Work Required:**
- Formal interface definitions for domain boundaries
- Type-safe coupling operators
- Domain compatibility checking
- See: `/docs/specifications/level-3-type-system.md`

#### 3. **Module System Enhancements** (v0.14)
**Status:** Basic imports work, need namespacing
**Impact:** MEDIUM — Better code organization

```morphogen
// Current: flat namespace
use field, audio

// Proposed: explicit namespaces
import field.*
import audio.{sine, lowpass}  // Selective imports
import physics as phys        // Aliasing
```

#### 4. **MLIR Optimization Passes** (v0.15)
**Status:** Basic lowering implemented, optimization stubs
**Impact:** HIGH — Performance critical

**Work Required:**
- Auto-vectorization for field operations
- Kernel fusion for stencil operations
- Memory layout optimization
- GPU kernel generation (CUDA/Vulkan)

### Medium Priority (v0.16-0.20)

#### 5. **Hot-Reload for Development** (v0.16)
**Status:** Designed, not implemented
**Impact:** MEDIUM — Developer productivity

```bash
morphogen run --watch simulation.morph  # Auto-reload on changes
```

#### 6. **Geometry Domain Integration** (v0.18)
**Status:** Architecture complete (ADR-001), implementation pending
**Impact:** HIGH — Enables CAD/FEA workflows

**See:** `/docs/architecture/domain-architecture.md` Section 2.1

#### 7. **Symbolic Math Integration** (v0.19)
**Status:** Planned domain
**Impact:** MEDIUM — Useful for scientific computing

#### 8. **Advanced GPU Scheduling** (v0.20)
**Status:** Planned
**Impact:** HIGH — Multi-GPU support, distributed execution

---

## 📋 Features Under Discussion

These features are **not finalized** and may change based on community feedback:

### 1. **Macros / Code Generation**
**Status:** 🤔 Under consideration
**Pros:** Code reuse, domain-specific optimizations
**Cons:** Complexity, compile-time overhead

**Options:**
- Rust-style procedural macros
- Lisp-style hygienic macros
- Template metaprogramming (C++ style)
- **Decision pending:** Community feedback needed

### 2. **Effect System for I/O**
**Status:** 🤔 Under consideration
**Question:** Should I/O operations be tracked in the type system?

```morphogen
// Pure function (no I/O)
fn compute(x: f32) -> f32 { x * 2.0 }

// Impure function (has I/O effect)
fn save_result(x: f32) -> Result<(), Error> io {
    audio.save(x, "output.wav")
}
```

**Pros:** Safety, compiler can optimize pure functions
**Cons:** Complexity, learning curve

### 3. **Ownership/Borrow Checking**
**Status:** 🤔 Under consideration
**Question:** Should Morphogen adopt Rust-style ownership semantics?

**Current:** Python-style reference semantics (GC)
**Proposed:** Rust-style ownership for zero-copy performance

**Decision pending:** Weigh complexity vs performance gains

### 4. **Pattern Matching**
**Status:** 🤔 Likely addition
**Syntax TBD:**

```morphogen
// Option 1: Rust-style
match shape {
    Circle(r) => area_circle(r),
    Square(s) => s * s,
    _ => 0.0
}

// Option 2: ML-style
case shape of
  | Circle r -> area_circle r
  | Square s -> s * s
```

---

## 📚 Documentation Path Forward

### Current Documentation Structure

**Root Level:**
- `/README.md` — Project overview (Morphogen/Morphogen)
- `/SPECIFICATION.md` — Language specification v0.10.0
- `/STATUS.md` — Implementation status
- `/CHANGELOG.md` — Version history
- `/ECOSYSTEM_MAP.md` — Domain ecosystem overview

**Organized in `/docs/`:**
- **philosophy/** — Why Morphogen exists (formalization, DSL principles)
- **architecture/** — How Morphogen is built (domain architecture, MLIR)
- **specifications/** — Technical specs for all domains
- **roadmap/** — Implementation progress and planning
- **guides/** — How-to guides for implementers
- **examples/** — Working code examples
- **adr/** — Architectural decision records

### Documentation Priorities

**Immediate (2025 Q4):**
- ✅ Consolidate archives (DONE)
- ✅ Move analysis docs to `/docs/analysis/` (DONE)
- ✅ Move type system to `/docs/specifications/` (DONE)
- [ ] Update all references to moved documents
- [ ] Create comprehensive domain catalog (40 domains)
- [ ] Finalize language features roadmap (this document)

**Short-term (2026 Q1):**
- [ ] Complete API reference for all 40 domains
- [ ] Write migration guide from v0.10 → v1.0
- [ ] Create video tutorials for core features
- [ ] Write "Morphogen for X" guides (X = MATLAB users, audio engineers, game devs, etc.)

---

## 🎯 Decision Framework

### How We Decide on Language Features

**Criteria for accepting a new feature:**
1. **Solves a real problem** — User demand or clear use case
2. **Fits the vision** — Aligns with "unified multi-domain platform"
3. **Doesn't break determinism** — Must be reproducible
4. **Can be implemented efficiently** — Performance matters
5. **Teachable** — Not overly complex for target audience

**Rejection criteria:**
- Breaks backward compatibility without strong justification
- Adds significant complexity for marginal benefit
- Conflicts with determinism guarantee
- Cannot be efficiently compiled to MLIR

---

## 📞 Community Input

**How to propose language features:**
1. Open a GitHub issue with tag `[language-feature]`
2. Describe the problem and proposed solution
3. Provide example syntax and use cases
4. Discuss trade-offs (complexity vs benefit)

**Where to discuss:**
- GitHub Discussions: https://github.com/scottsen/morphogen/discussions
- GitHub Issues: https://github.com/scottsen/morphogen/issues

---

## Summary: Where We Stand

**✅ Stable & Production-Ready:**
- Core syntax (flow, @state, types, functions, lambdas)
- 40 domain modules with 500+ operators
- Python runtime with NumPy backend
- MLIR compilation infrastructure

**🚧 Planned & Finalized:**
- Physical unit checking (v0.12)
- Cross-domain type safety (v0.13)
- Module system enhancements (v0.14)
- MLIR optimization (v0.15)

**🤔 Under Discussion:**
- Macros / code generation
- Effect system for I/O
- Ownership / borrow checking
- Pattern matching

**🎯 Path to 1.0:**
- Stabilize core language (no breaking changes)
- Production-ready MLIR compiler
- Comprehensive documentation
- Target: 2026 Q2

---

**Next Steps:** Review this document with the community, finalize decisions on "Under Discussion" features, and execute on v0.12-0.15 roadmap.
