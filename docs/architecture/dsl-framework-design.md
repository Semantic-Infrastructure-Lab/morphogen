# DSL Framework Design: Domain Reasoning Language

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Design Vision & Gap Analysis
**Authors:** Scott Sen, with Claude

---

## Executive Summary

This document outlines a comprehensive vision for Morphogen as a **domain reasoning language** — a DSL that treats **domains**, **operators**, and **domain translations** as first-class citizens. It consolidates theoretical foundations from category theory with practical requirements for a powerful, extensible, and intuitive programming language.

**Core Philosophy:** Most programming languages make objects and functions first-class, but domains and cross-domain translations are ad-hoc. Morphogen fixes this by elevating domain reasoning to a fundamental language construct.

### 📖 Document Purpose & Scope

**This document answers: "HOW should we implement first-class domains and translations?"**

This is an **implementation vision** with concrete syntax proposals, gap analysis, and roadmap. It bridges theory (from philosophy docs) with practice (current implementation) and charts a path forward.

**Contrast with complementary documents:**
- **[Universal DSL Principles](../philosophy/universal-dsl-principles.md):** Design philosophy, theoretical principles, the "why"
- **This doc (DSL Framework Design):** Implementation vision, syntax proposals, the "how"
- **Current Implementation:** What Morphogen already has today (see [Architecture Overview](./overview.md))

**Think of it this way:**
```
Universal DSL Principles (philosophy)
    ↓ informs ↓
DSL Framework Design (this doc - architecture vision)
    ↓ guides ↓
Current Implementation (code & specs)
```

**What you'll find here:**
1. **Vision:** Ideal language features (Section 1)
2. **Current State:** What we have today (Section 2)
3. **Gap Analysis:** What's missing (Section 3)
4. **Proposed Extensions:** Concrete syntax proposals (Section 4)
5. **Implementation Roadmap:** 5-phase plan to v0.16.0 (Section 5)
6. **Examples:** Before/after comparisons (Section 6)

**Related Documentation:**

**Philosophical Foundations:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) ⭐ — Eight core design principles for universal DSLs (the "why")
- [Formalization and Knowledge](../philosophy/formalization-and-knowledge.md) — Historical context of formalization
- [Categorical Structure](../philosophy/categorical-structure.md) — Category theory in practice
- [Operator Foundations](../philosophy/operator-foundations.md) — Mathematical operator theory

**Technical Specifications:**
- [Continuous-Discrete Semantics](continuous-discrete-semantics.md) ⭐ — Dual computational models
- [Transform Composition](../specifications/transform-composition.md) ⭐ — Composable named transforms
- [Transform Specification](../specifications/transform.md) — Current transform implementation
- [Domain Architecture](./domain-architecture.md) — Domain taxonomy
- [Universal Domain Frameworks](../reference/universal-domain-frameworks.md) — Theoretical foundations

**ADRs:**
- [Universal Domain Translation](../adr/012-universal-domain-translation.md) ⭐ — Translation framework with invariants

---

## Table of Contents

1. [Vision: Ideal Language for Domain Reasoning](#1-vision-ideal-language-for-domain-reasoning)
2. [Current State Analysis](#2-current-state-analysis)
3. [Gap Analysis](#3-gap-analysis)
4. [Proposed Language Extensions](#4-proposed-language-extensions)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Examples: Current vs. Future](#6-examples-current-vs-future)
7. [Theoretical Foundations](#7-theoretical-foundations)

---

## 1. Vision: Ideal Language for Domain Reasoning

### 1.1 Core Philosophy

The language must treat as **first-class citizens**:

✔ **Domains** - explicit, definable computational universes
✔ **Operators** - functions bound to domains
✔ **Domain Translations** - structure-preserving maps between domains
✔ **Translation Composition** - algebraic composition of domain maps
✔ **Invariants** - declarative properties that must be preserved

This is the critical insight most existing languages miss:
- In Python, Java, etc.: objects exist, functions exist, but **domains do not exist as top-level constructs**
- Cross-domain translation is **ad-hoc** (manual glue code)
- Structure preservation is **manual** (easy to violate)

### 1.2 What the Language Must Express Easily

**🔹 A domain:**
```morphogen
domain Time {
    entity signal(t: Real): Real
    operator shift(t: Real): Time
    operator scale(factor: Real): Time
}
```

**🔹 A translation (functor):**
```morphogen
translation FFT from Time to Frequency {
    // Object mapping
    map signal(t) -> sum_k X[k] * exp(i*k*t)

    // Structure preservation
    preserves norm
    preserves linearity

    // Operator mapping
    operator shift maps to phase_rotate
    operator differentiate maps to multiply_by_iw
}
```

**🔹 A composed translation:**
```morphogen
let Wavelet = ScaleTransform ∘ FFT
let CWT = Wavelet ∘ TimeShift
```

**🔹 A universal operator:**
```morphogen
operator differentiate {
    in Time: finite_difference(h=dt)
    in Frequency: multiply(i*omega)
    in Laplace: multiply(s)
}
```

**🔹 Declarative invariants:**
```morphogen
invariant EnergyPreserved {
    forall signal in Time:
        norm(signal) == norm(FFT(signal))
}

invariant Linearity {
    forall f, g in Time, a, b in Real:
        FFT(a*f + b*g) == a*FFT(f) + b*FFT(g)
}
```

### 1.3 Power Through First-Class Abstractions

| Concept | Current Languages | Morphogen Vision |
|---------|------------------|--------------|
| Domains | Implied, ad-hoc | Explicit, first-class |
| Operators | Methods on objects | Bound to domains |
| Translations | Manual glue code | First-class, composable |
| Structure preservation | Manual verification | Compiler-enforced |
| Composition | Function composition only | Domain translation composition |
| Invariants | Runtime assertions | Declarative specifications |

---

## 2. Current State Analysis

### 2.1 What Morphogen/Morphogen Already Has ✅

**Strong Type System:**
- `Stream<T, Domain, Rate>` with dependent types
- Physical units as refinement types
- Domain tags (audio, physics, geometry, etc.)

**Operator Registry:**
- Four-layer hierarchy (atomic → composite → constructs → presets)
- `@operator` decorator with metadata
- 40 implemented domains

**Cross-Domain System:**
- `transform.to(...)` / `transform.from(...)` for domain changes
- Cross-domain interfaces and validators
- Domain composition support

**Categorical Structure:**
- Morphogen as a category (`Morph`) with streams as objects
- Operators as morphisms
- Functorial semantics for transforms (FFT, Laplace, etc.)
- Natural transformations for cross-domain interfaces

**MLIR Compilation:**
- Custom dialects for domains (field, agent, audio, temporal)
- Lowering passes
- JIT/AOT compilation

### 2.2 What's Missing ❌

**Declarative Domain Definition:**
- No `domain` keyword in language
- Domains defined in Python, not Morphogen
- Users cannot extend domains from within Morphogen

**First-Class Translations:**
- `transform.to(...)` exists but is procedural
- No declarative translation definition
- No explicit operator mapping specification

**Translation Composition:**
- No `∘` composition operator
- Can chain with `|>` but not algebraic composition
- No translation objects

**Invariant Specification:**
- No `invariant` keyword
- No compiler verification of invariants
- Structure preservation is manual

**Domain Inheritance:**
- No `extends` for domains
- No domain refinement or specialization
- No domain polymorphism

**Symbolic Manipulation:**
- Purely computational, no symbolic mode
- Cannot manipulate expressions algebraically
- No automatic simplification

### 2.3 Current Strengths

✅ **Solid foundation** - Type system, operator registry, MLIR pipeline
✅ **40 working domains** - Extensive library coverage
✅ **Theoretical grounding** - Category theory foundations documented
✅ **Production-ready** - Tests, examples, documentation
✅ **Cross-domain composition** - Patterns established (ADR-002)

---

## 3. Gap Analysis

### Priority 1: Essential for Domain Reasoning

| Feature | Status | Gap | Priority |
|---------|--------|-----|----------|
| Declarative domain definition | ❌ Missing | High | P0 |
| First-class translations | ⚠️ Partial | High | P0 |
| Translation composition | ❌ Missing | Medium | P1 |
| Invariant specification | ❌ Missing | Medium | P1 |
| Structure preservation | ⚠️ Manual | Medium | P1 |

### Priority 2: Advanced Features

| Feature | Status | Gap | Priority |
|---------|--------|-----|----------|
| Domain inheritance | ❌ Missing | Low | P2 |
| Symbolic manipulation | ❌ Missing | Low | P2 |
| Graphical visualization | ⚠️ Partial | Low | P2 |
| Automatic optimization | ⚠️ Partial | Low | P3 |

### Detailed Gap Analysis

#### Gap 1: Declarative Domain Definition

**Current:**
```python
# In Python: morphogen/stdlib/audio.py
@domain("audio")
class AudioDomain:
    @operator(category="generator")
    def sine(freq: float, duration: float) -> Stream:
        ...
```

**Desired:**
```morphogen
// In Morphogen source
domain Audio {
    entity signal: Stream<f32, audio, 48kHz>

    operator sine(freq: f32 [Hz], duration: f32 [s]): Audio.signal
    operator filter(cutoff: f32 [Hz], Q: f32): Audio.signal -> Audio.signal
}
```

**Impact:** Limits user extensibility, hides domain structure

---

#### Gap 2: First-Class Translations

**Current:**
```morphogen
// Procedural
let spec = transform.to(signal, domain="frequency", method="fft")
```

**Desired:**
```morphogen
// Declarative translation definition
translation Fourier from Time to Frequency {
    type: bijective

    forward: fft(window="hann", norm="ortho")
    inverse: ifft(norm="ortho")

    preserves {
        energy: norm(input) == norm(output)
        linearity: true
    }

    operator_map {
        convolve -> multiply
        shift -> phase_rotate
        differentiate -> multiply_by_iw
    }
}

// Usage
let spec = Fourier(signal)  // Applies translation
let back = Fourier.inverse(spec)
```

**Impact:** No explicit structure preservation, operator mappings are implicit

---

#### Gap 3: Translation Composition

**Current:**
```morphogen
// Manual chaining
let spec = transform.to(signal, domain="frequency", method="fft")
let mel = transform.reparam(spec, mapping=mel_scale(128))
```

**Desired:**
```morphogen
// Algebraic composition
let MelSpectrum = MelScale ∘ Fourier

// Use composed translation
let mel_spec = MelSpectrum(signal)

// Verify composition law
assert( (CWT ∘ FFT)(signal) == CWT(FFT(signal)) )
```

**Impact:** Composition is manual, not algebraic

---

#### Gap 4: Invariant Specification

**Current:**
```python
# Manual testing
def test_fft_preserves_energy():
    signal = generate_signal()
    spectrum = fft(signal)
    assert abs(norm(signal) - norm(spectrum)) < 1e-6
```

**Desired:**
```morphogen
translation Fourier from Time to Frequency {
    ...

    invariant EnergyPreservation {
        forall signal: Time.signal
        => norm(signal) == norm(transform(signal))
    }

    invariant Linearity {
        forall a, b in Real, f, g in Time.signal
        => transform(a*f + b*g) == a*transform(f) + b*transform(g)
    }
}
```

**Impact:** Invariants not verified by compiler

---

#### Gap 5: Domain Inheritance

**Current:**
```python
# No inheritance model
```

**Desired:**
```morphogen
domain VectorSpace {
    entity vector: Vec<N, T>
    operator add(a: vector, b: vector): vector
    operator scale(a: Real, v: vector): vector
}

domain InnerProductSpace extends VectorSpace {
    operator inner(a: vector, b: vector): Real
    operator norm(v: vector): Real = sqrt(inner(v, v))
}

domain HilbertSpace extends InnerProductSpace {
    constraint: complete  // Cauchy sequences converge
}
```

**Impact:** Cannot express domain hierarchies

---

## 4. Proposed Language Extensions

### 4.1 Domain Definition Syntax

```morphogen
domain <Name> [extends <Parent>] {
    // Type declarations
    entity <name>: <type>

    // Operator declarations
    operator <name>(<params>): <return_type>

    // Constraints
    constraint <name>: <expression>

    // Invariants
    invariant <name> { <property> }
}
```

**Example:**
```morphogen
domain Time {
    entity signal: Stream<f32, time, R>

    operator shift(delta: f32 [s]): signal -> signal
    operator sample(t: f32 [s]): signal -> f32

    constraint causality: forall t < 0 => signal(t) == 0
}
```

### 4.2 Translation Definition Syntax

```morphogen
translation <Name> from <SourceDomain> to <TargetDomain> {
    type: bijective | surjective | injective

    forward: <method>(<params>)
    inverse: <method>(<params>)

    preserves {
        <property_name>: <expression>
    }

    operator_map {
        <source_op> -> <target_op>
    }
}
```

**Example:**
```morphogen
translation Fourier from Time to Frequency {
    type: bijective

    forward: fft(window="hann", norm="ortho")
    inverse: ifft(norm="ortho")

    preserves {
        energy: norm(input) == norm(output)
        linearity: true
    }

    operator_map {
        Time.shift -> Frequency.phase_rotate
        Time.convolve -> Frequency.multiply
    }
}
```

### 4.3 Composition Operator

```morphogen
// Composition operator: ∘
let composed = Translation2 ∘ Translation1

// Equivalent to:
let composed(x) = Translation2(Translation1(x))

// Laws enforced:
// Associativity: (h ∘ g) ∘ f == h ∘ (g ∘ f)
// Identity: id ∘ f == f == f ∘ id
```

### 4.4 Invariant Syntax

```morphogen
invariant <Name> {
    forall <bindings>
    [where <constraints>]
    => <property>
}
```

**Example:**
```morphogen
invariant FourierEnergyPreservation {
    forall signal: Time.signal
    => abs(norm(signal) - norm(Fourier(signal))) < epsilon
}

invariant ConvolutionTheorem {
    forall f, g: Time.signal
    => Fourier(convolve(f, g)) == multiply(Fourier(f), Fourier(g))
}
```

### 4.5 Multi-Domain Operators

```morphogen
operator <name> {
    in <Domain1>: <implementation1>
    in <Domain2>: <implementation2>

    preserves: <invariants across domains>
}
```

**Example:**
```morphogen
operator differentiate {
    in Time: finite_difference(h=dt)
    in Frequency: multiply(i * omega)
    in Laplace: multiply(s)

    preserves linearity: true
}
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (v0.12.0) - Q1 2026

**Goal:** Add declarative domain definition to the language

- [ ] Add `domain` keyword to lexer/parser
- [ ] Extend AST with DomainDeclaration node
- [ ] Implement domain type checking
- [ ] Allow users to define domains in .morph files
- [ ] Generate Python domain classes from Morphogen domains

**Deliverable:** Users can write `domain MyDomain { ... }` in Morphogen

---

### Phase 2: Translations (v0.13.0) - Q2 2026

**Goal:** First-class translation objects

- [ ] Add `translation` keyword to lexer/parser
- [ ] Extend AST with TranslationDeclaration node
- [ ] Implement translation type system
- [ ] Add `preserves` clause validation
- [ ] Add `operator_map` specification
- [ ] Generate translation objects from declarations

**Deliverable:** Users can write `translation Fourier from Time to Frequency { ... }`

---

### Phase 3: Composition (v0.14.0) - Q3 2026

**Goal:** Algebraic composition of translations

- [ ] Add `∘` composition operator to language
- [ ] Implement composition type checking
- [ ] Verify composition laws (associativity, identity)
- [ ] Optimize composed translations
- [ ] Generate efficient code for compositions

**Deliverable:** Users can write `let CWT = Wavelet ∘ FFT`

---

### Phase 4: Invariants (v0.15.0) - Q4 2026

**Goal:** Declarative invariant specification

- [ ] Add `invariant` keyword to lexer/parser
- [ ] Extend AST with InvariantDeclaration node
- [ ] Implement invariant checking (runtime + optional static)
- [ ] Add property-based testing generation
- [ ] Integrate with test framework

**Deliverable:** Users can write `invariant { ... }` with compiler verification

---

### Phase 5: Advanced Features (v0.16.0+) - 2027

**Goal:** Domain inheritance, symbolic manipulation

- [ ] Add `extends` for domain inheritance
- [ ] Implement domain polymorphism
- [ ] Add symbolic expression mode
- [ ] Implement automatic simplification
- [ ] Add graphical diagram generation

**Deliverable:** Full domain reasoning language

---

## 6. Examples: Current vs. Future

### Example 1: Defining a Domain

**Current (v0.11.0):**
```python
# In morphogen/stdlib/custom.py
@domain("custom")
class CustomDomain:
    @operator(category="generator")
    def my_operator(x: float) -> Stream:
        return generate_stream(x)
```

**Future (v0.12.0+):**
```morphogen
// In examples/custom_domain.morph
domain Custom {
    entity signal: Stream<f32, custom, 1kHz>

    operator generate(x: f32): Custom.signal {
        // Implementation
    }

    operator process(input: Custom.signal, gain: f32): Custom.signal {
        // Implementation
    }
}

// Use it
use custom

flow(dt=0.001, steps=1000) {
    let sig = custom.generate(x=0.5)
    let processed = custom.process(sig, gain=2.0)
    output processed
}
```

---

### Example 2: Domain Translation

**Current (v0.11.0):**
```morphogen
use audio, transform

flow(dt=1.0/48000.0, steps=48000) {
    let signal = audio.sine(440Hz, duration=1s)
    let spec = transform.to(signal, domain="frequency", method="fft")

    // Manually verify energy preservation
    assert(abs(norm(signal) - norm(spec)) < 1e-6)

    output spec
}
```

**Future (v0.13.0+):**
```morphogen
use audio

// Define translation
translation Fourier from Time to Frequency {
    type: bijective

    forward: fft(window="hann", norm="ortho")
    inverse: ifft(norm="ortho")

    preserves {
        energy: norm(input) == norm(output)
        linearity: true
    }

    operator_map {
        shift -> phase_rotate
        convolve -> multiply
    }
}

flow(dt=1.0/48000.0, steps=48000) {
    let signal = audio.sine(440Hz, duration=1s)

    // Use translation (automatic verification)
    let spec = Fourier(signal)

    // Compiler verifies energy preservation automatically

    output spec
}
```

---

### Example 3: Composed Translations

**Current (v0.11.0):**
```morphogen
// Manual composition
let signal = audio.sine(440Hz, duration=1s)
let spec = transform.to(signal, domain="frequency", method="fft")
let mel = transform.reparam(spec, mapping=mel_scale(128))
let log_mel = log(abs(mel) + 1e-8)
```

**Future (v0.14.0+):**
```morphogen
// Define atomic translations
translation Fourier from Time to Frequency { ... }
translation MelScale from Frequency to MelFrequency { ... }
translation LogScale from MelFrequency to LogMelFrequency { ... }

// Compose translations algebraically
let MelSpectrogram = LogScale ∘ MelScale ∘ Fourier

// Use composed translation
let signal = audio.sine(440Hz, duration=1s)
let mel_spec = MelSpectrogram(signal)

// Verify composition law
assert( MelSpectrogram(signal) == LogScale(MelScale(Fourier(signal))) )
```

---

### Example 4: Invariant Verification

**Current (v0.11.0):**
```python
# In tests/test_fft.py
def test_fft_preserves_energy():
    signal = generate_signal()
    spectrum = fft(signal)
    assert abs(norm(signal) - norm(spectrum)) < 1e-6

def test_fft_linearity():
    f, g = generate_signals()
    a, b = 2.0, 3.0
    assert norm(fft(a*f + b*g) - (a*fft(f) + b*fft(g))) < 1e-6
```

**Future (v0.15.0+):**
```morphogen
translation Fourier from Time to Frequency {
    ...

    // Invariants checked by compiler
    invariant EnergyPreservation {
        forall signal: Time.signal
        => abs(norm(signal) - norm(transform(signal))) < epsilon
    }

    invariant Linearity {
        forall a, b: Real, f, g: Time.signal
        => transform(a*f + b*g) == a*transform(f) + b*transform(g)
    }

    invariant Parseval {
        forall signal: Time.signal
        => sum(abs(signal)^2) == sum(abs(transform(signal))^2)
    }
}

// Compiler generates property-based tests automatically
// Runtime checks can be enabled with --verify-invariants flag
```

---

## 7. Theoretical Foundations

### 7.1 Category Theory

**Morphogen/Morphogen as a Category:**
- **Objects:** `Stream<T, Domain, Rate>`
- **Morphisms:** Operators `op: Stream → Stream`
- **Composition:** Data flow / operator chaining
- **Identity:** `id_stream` (pass-through)

**Functors (Translations):**
- **Object map:** `F: Domain1 → Domain2`
- **Morphism map:** `F(op): F(A) → F(B)`
- **Laws:**
  - `F(id_A) = id_F(A)` (identity preservation)
  - `F(g ∘ f) = F(g) ∘ F(f)` (composition preservation)

**Natural Transformations (Cross-Domain Interfaces):**
- `η: F ⇒ G` between functors
- Components: `η_A: F(A) → G(A)` for each object A
- Naturality: `G(f) ∘ η_A = η_B ∘ F(f)`

### 7.2 Type Theory

**Dependent Types:**
```morphogen
Stream<T, Domain, Rate>
```
- `Rate` depends on `Domain`
- `T` depends on `Domain`

**Refinement Types:**
```morphogen
Length<m> = {x: f64 | unit = meters}
Time<s> = {t: f64 | unit = seconds}
```

**Curry-Howard-Lambek Correspondence:**
| Logic | Morphogen | Category |
|-------|-------|----------|
| Proposition P | Type `Stream<T,D,R>` | Object A |
| Proof of P | Value of type | Morphism |
| P ⇒ Q | Function | Morphism A → B |

### 7.3 Universal Algebra

**Four-Layer Operator Hierarchy:**
1. **Layer 1 (Free Algebra):** Atomic operators
2. **Layer 2 (Quotient Algebra):** Composites with domain equations
3. **Layer 3 (Derived Algebra):** Constructs
4. **Layer 4 (Instantiated Algebra):** Presets

**Homomorphisms:**
- FFT as ring homomorphism: `(Time, +, *) → (Frequency, +, ·)`
- Preserves addition, transforms convolution to multiplication

### 7.4 Domain Theory

**Pass System as Refinement:**
- Partial order: `OpGraph₁ ⊑ OpGraph₂` (refinement)
- Fixed point: `fix(optimize) = ⊔ₙ optimizeⁿ(OpGraph₀)`
- Continuous functions preserve suprema

---

## Summary

### What We Have ✅

1. **Solid foundation** - Type system, 40 domains, MLIR compilation
2. **Theoretical grounding** - Category theory foundations
3. **Production-ready** - Tests, docs, examples
4. **Cross-domain composition** - Established patterns

### What We're Building 🚧

1. **Declarative domain definition** - `domain { ... }` in language
2. **First-class translations** - `translation { ... }` with structure preservation
3. **Algebraic composition** - `Translation2 ∘ Translation1`
4. **Invariant specification** - `invariant { ... }` with verification
5. **Domain inheritance** - `domain X extends Y`

### Why It Matters 🎯

This transformation will make Morphogen the first language where:
- Domains are not implicit but **explicit and first-class**
- Cross-domain reasoning is not ad-hoc but **algebraic and composable**
- Structure preservation is not manual but **compiler-verified**
- Domain knowledge is not scattered but **declaratively specified**

**Result:** A programming language that thinks in domains the way humans do.

---

## References

### Internal Documentation
- [Morphogen Categorical Structure](./morphogen-categorical-structure.md) - Current theory
- [Transform Specification](../specifications/transform.md) - Current transforms
- [Domain Architecture](./domain-architecture.md) - Domain taxonomy
- [ADR-002: Cross-Domain Patterns](../adr/002-cross-domain-architectural-patterns.md) - Patterns
- [Universal Domain Frameworks](../reference/universal-domain-frameworks.md) - Theory

### External References
- Mac Lane: *Categories for the Working Mathematician*
- Pierce: *Types and Programming Languages*
- Awodey: *Category Theory* (2nd edition)
- Milewski: *Category Theory for Programmers*

---

*This document bridges the gap between our current implementation and our ultimate vision for Morphogen as a universal domain reasoning language.*
