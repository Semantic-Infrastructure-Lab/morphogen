# Capability Growth Framework: Decision-Making Guide for Morphogen Evolution

**Status:** Reference Document
**Date:** 2025-11-21
**Purpose:** Practical framework for evaluating new features, domains, and capabilities

---

## Purpose

This document provides the **decision-making framework** for growing Morphogen's capabilities. When considering any addition (new domain, feature, operator, syntax), use this framework to evaluate:

1. **Should we add this?** (Strategic value)
2. **Where does it belong?** (Architectural layer)
3. **How should it compose?** (Integration strategy)
4. **When should we add it?** (Prioritization)

---

## The Six Lenses Framework

Evaluate every proposed addition through six lenses:

### Lens 1: Strategic Value

**Question:** Does this enable new professional use cases or significantly improve existing ones?

**Scoring:**
- **🔥🔥🔥 Critical** - Unlocks entire professional field (e.g., Circuit domain → EE/audio market)
- **🔥🔥 High** - Significantly improves existing workflows (e.g., GPU acceleration)
- **🔥 Medium** - Useful but not transformative (e.g., additional filter types)
- **❄️ Low** - Nice-to-have, marginal impact (e.g., syntactic sugar)

**Examples:**

| Proposal | Value | Rationale |
|----------|-------|-----------|
| Circuit domain | 🔥🔥🔥 Critical | Unlocks EE, analog audio, PCB design markets. No competing tool. |
| GPU acceleration | 🔥🔥 High | 10-100x speedup, enables real-time and large-scale simulation. |
| Pattern matching syntax | 🔥 Medium | Cleaner code, but doesn't enable new capabilities. |
| Ternary operator | ❄️ Low | Already have if-else expressions. Pure sugar. |

**Decision Rule:** Only add features with 🔥 Medium or higher strategic value.

---

### Lens 2: Composability Impact

**Question:** Does this compose cleanly with existing features, or does it create complexity?

**Scoring:**
- **✅ Orthogonal** - Composes naturally, no interactions with existing features
- **⚠️ Minor Interactions** - Some edge cases, but manageable
- **🚨 Complex Interactions** - Significant interactions, requires careful design
- **🛑 Breaking** - Breaks existing composition patterns

**Examples:**

| Proposal | Impact | Rationale |
|----------|--------|-----------|
| Symbolic math domain | ✅ Orthogonal | Just another domain library, uses existing operator registry. |
| Effect system | ⚠️ Minor Interactions | Affects function types, but additive (opt-in). |
| Implicit auto-resampling | 🚨 Complex | Hidden behavior, interacts with every cross-domain coupling. |
| Mutable references | 🛑 Breaking | Breaks determinism, hot-reload, parallelization assumptions. |

**Decision Rule:** Strongly prefer ✅ Orthogonal. Reject 🛑 Breaking unless massive value justifies complete redesign.

---

### Lens 3: Implementation Complexity

**Question:** How much effort to implement, test, document, and maintain?

**Scoring:**
- **🟢 Low** (< 1 week) - Single contributor, straightforward
- **🟡 Medium** (1-4 weeks) - Requires design, testing, documentation
- **🔴 High** (1-3 months) - Major subsystem, complex integration
- **⚫ Extreme** (> 3 months) - Fundamental architectural change

**Examples:**

| Proposal | Effort | Details |
|----------|--------|---------|
| Add sine wave operator | 🟢 Low | 1 function, 5 tests, docstring. Done in hours. |
| Transform space tracking | 🟡 Medium | Type system changes, ~20 operators affected, 2 weeks. |
| Circuit domain | 🔴 High | 40+ operators, solver integration, 100+ tests. 4-6 weeks. |
| Complete effect system | ⚫ Extreme | Type system redesign, affects all code. 3+ months. |

**Decision Rule:** Balance effort against value. 🟢 Low effort = low bar. ⚫ Extreme effort requires 🔥🔥🔥 Critical value.

---

### Lens 4: Architectural Fit

**Question:** Does this belong in kernel, domain library, or frontend? Does it respect boundaries?

**Belongs in Kernel if:**
- Universal abstraction (types, scheduling, transforms)
- Required by multiple domains
- Zero acceptable alternatives (must be standard)
- Extremely stable (never changes)

**Belongs in Domain Library if:**
- Domain-specific operators
- Multiple implementations possible
- Can be deprecated/replaced
- Evolves with research/practice

**Belongs in Frontend if:**
- Human ergonomics (syntax, error messages)
- Multiple ways to express same thing
- Syntax sugar over existing operators
- Rapid iteration expected

**Examples:**

| Proposal | Layer | Rationale |
|----------|-------|-----------|
| FFT transform | Kernel | Universal, multiple domains need it, stable standard. |
| Guitar pedal model | Domain (Audio) | Specific use case, evolves, could have alternatives. |
| Pattern matching | Frontend | Syntax sugar, multiple syntaxes possible. |
| Operator fusion pass | Kernel (MLIR) | Compiler optimization, universal benefit. |

**Decision Rule:** Respect layer boundaries. If in doubt, put in higher layer (Frontend > Domain > Kernel). Moving down is hard, moving up is easy.

---

### Lens 5: Determinism Compatibility

**Question:** Does this work in all three determinism profiles (strict, repro, live)?

**Compatibility:**
- **✅ All Profiles** - Works in strict, repro, and live
- **⚠️ Repro/Live Only** - Can't guarantee bit-exact results
- **🚨 Live Only** - Fundamentally nondeterministic
- **🛑 Profile Breaking** - Makes determinism impossible

**Examples:**

| Proposal | Profile Support | Rationale |
|----------|----------------|-----------|
| FFT operator | ✅ All Profiles | Deterministic algorithm, bit-exact possible. |
| Adaptive integrator | ⚠️ Repro/Live | Timestep varies, but deterministic given tolerance. |
| Real-time audio input | 🚨 Live Only | External input, inherently nondeterministic. |
| Global mutable state | 🛑 Profile Breaking | Makes determinism impossible to track. |

**Decision Rule:** Prefer ✅ All Profiles. Mark ⚠️ and 🚨 explicitly in docs. Reject 🛑 Profile Breaking outright.

---

### Lens 6: Community Alignment

**Question:** Does this align with community needs, or is it niche/personal preference?

**Evidence:**
- **📊 User Requests** - GitHub issues, forum discussions, surveys
- **📚 Academic Precedent** - Research papers, established methods
- **🏭 Industry Practice** - Used in professional tools (MATLAB, Abaqus, etc.)
- **🎨 Creative Community** - Used in creative coding (Processing, SuperCollider, etc.)

**Examples:**

| Proposal | Alignment | Evidence |
|----------|-----------|----------|
| Circuit domain | 🏭 Industry | SPICE has 50+ years of use. Every EE uses it. |
| Symbolic math | 📚 Academic | Mathematica, SymPy, established need. |
| Category theory optimization | 📚 Academic | Research area, proven benefits (Conal Elliott). |
| Custom emoji operators | ❌ None | No precedent, niche personal preference. |

**Decision Rule:** Require at least one strong evidence source. Multiple sources = higher confidence.

---

## Decision Matrix

Use this table to score proposals:

| Proposal | Strategic | Composability | Effort | Arch Fit | Determinism | Community | **Total** | **Decision** |
|----------|-----------|---------------|--------|----------|-------------|-----------|-----------|--------------|
| Circuit Domain | 🔥🔥🔥 (5) | ✅ (5) | 🔴 (-3) | Domain (5) | ✅ (5) | 🏭 (5) | **22/25** | ✅ **ACCEPT** |
| Effect System | 🔥🔥 (4) | ⚠️ (3) | ⚫ (-5) | Kernel (5) | ✅ (5) | 📚 (4) | **16/25** | ⚠️ **DEFER to v2.0** |
| Pattern Matching | 🔥 (3) | ✅ (5) | 🟡 (-2) | Frontend (3) | ✅ (5) | 📚 (4) | **18/25** | ⏳ **CONSIDER** |
| Implicit Resampling | ❄️ (1) | 🚨 (1) | 🟢 (-1) | Domain (5) | ✅ (5) | ❌ (0) | **11/25** | ❌ **REJECT** |

**Scoring Guide:**
- Strategic: Critical=5, High=4, Medium=3, Low=1
- Composability: Orthogonal=5, Minor=3, Complex=1, Breaking=0
- Effort: Low=-1, Medium=-2, High=-3, Extreme=-5
- Arch Fit: Correct layer=5, Wrong layer=-5
- Determinism: All=5, Repro/Live=3, Live=1, Breaking=0
- Community: Strong evidence=5, Weak=3, None=0

**Decision Thresholds:**
- **20-25**: ACCEPT (high priority)
- **15-19**: CONSIDER (evaluate against alternatives)
- **10-14**: DEFER (low priority, maybe later)
- **< 10**: REJECT (not worth it)

---

## Common Scenarios

### Scenario 1: New Domain Addition

**Question:** Should we add domain X?

**Process:**
1. **Identify professional use case** (Lens 1: Strategic Value)
   - What field does this enable?
   - Who are the users?
   - What tools does this replace?

2. **Check domain independence** (Lens 2: Composability)
   - Does it reuse existing operators or need new ones?
   - What are cross-domain coupling points?
   - Any conflicts with existing domains?

3. **Estimate implementation scope** (Lens 3: Effort)
   - How many operators needed (minimum viable)?
   - What's the complexity (algorithms, dependencies)?
   - Testing and documentation load?

4. **Validate architectural fit** (Lens 4)
   - Pure domain library? Or needs kernel changes?
   - What operator registry entries required?
   - MLIR lowering needed?

5. **Check determinism** (Lens 5)
   - Are all operators deterministic?
   - Any inherent nondeterminism (I/O, external data)?
   - Profile support plan?

6. **Research community need** (Lens 6)
   - GitHub issues requesting this?
   - Academic papers using similar systems?
   - Industry tools to reference?

**Example: Circuit Domain**
- ✅ Strategic: Unlocks EE, analog audio, PCB markets
- ✅ Composability: Clean domain library, clear coupling (circuit→audio, geometry→circuit)
- ⚠️ Effort: High (40+ operators, solvers), but manageable
- ✅ Arch Fit: Pure domain library, no kernel changes
- ✅ Determinism: DC/AC/transient all deterministic with fixed timesteps
- ✅ Community: SPICE, LTspice, 50+ years of established practice

**Decision:** ACCEPT as Priority 0 domain

---

### Scenario 2: Language Feature Addition

**Question:** Should we add language feature Y (e.g., pattern matching, macros, effects)?

**Process:**
1. **Identify concrete use case** (Lens 1)
   - What problem does this solve?
   - Can existing features solve it?
   - How often will users need this?

2. **Analyze composition** (Lens 2)
   - Does this interact with existing syntax?
   - Edge cases with other features?
   - Does it introduce ambiguity?

3. **Prototype quickly** (Lens 3)
   - Implement proof-of-concept
   - Test on real examples
   - Measure complexity added to parser/compiler

4. **Check layer appropriateness** (Lens 4)
   - Is this kernel-level or frontend-level?
   - Could multiple frontends implement differently?
   - Does it affect Graph IR?

5. **Verify determinism** (Lens 5)
   - Does this feature introduce nondeterminism?
   - Can it work in strict mode?
   - Any hidden state?

6. **Research prior art** (Lens 6)
   - What languages have this?
   - What are known pitfalls?
   - Community consensus?

**Example: Pattern Matching**
- ✅ Strategic: Cleaner code for variant types, useful for AST matching
- ✅ Composability: Orthogonal, doesn't affect existing features
- ⚠️ Effort: Medium (parser changes, type checking, lowering)
- ✅ Arch Fit: Frontend feature (could vary across frontends)
- ✅ Determinism: Fully deterministic, no issues
- ✅ Community: Rust, ML, Haskell - established pattern

**Decision:** CONSIDER for v1.5.0 (useful, but not critical for v1.0)

---

### Scenario 3: Operator Addition

**Question:** Should we add operator Z to existing domain?

**Process:**
1. **Check necessity** (Lens 1)
   - Can this be composed from existing operators?
   - Is it a common pattern worth optimizing?
   - Does it unlock new use cases?

2. **Test composition** (Lens 2)
   - Does it compose with other domain operators?
   - Any surprising interactions?
   - Does it follow domain conventions?

3. **Estimate implementation** (Lens 3)
   - Algorithmically complex or straightforward?
   - Dependencies (new libraries)?
   - Testing scope (edge cases, performance)?

4. **Validate domain fit** (Lens 4)
   - Does it clearly belong to this domain?
   - Or is it cross-domain (needs coupling operator)?
   - Should it be in a different domain?

5. **Check determinism** (Lens 5)
   - Deterministic algorithm?
   - Floating-point reproducibility concerns?
   - RNG needed (must be explicit)?

6. **Find references** (Lens 6)
   - Standard algorithm (paper, textbook)?
   - Used in reference tools (NumPy, MATLAB, etc.)?
   - Community requests?

**Example: Karplus-Strong String Synthesis**
- ✅ Strategic: Essential for physical modeling, widely used
- ✅ Composability: Clean audio operator, standard signature
- ✅ Effort: Low (well-defined algorithm, ~50 lines)
- ✅ Arch Fit: Clearly audio domain operator
- ✅ Determinism: Fully deterministic given seed
- ✅ Community: SuperCollider, Csound, 40+ years of use

**Decision:** ACCEPT (added in v0.5.0)

---

## Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Premature Abstraction

**Problem:** Adding generic infrastructure before concrete use cases exist.

**Example:**
```python
# BAD: Generic "optimizer framework" with no specific optimizers
class Optimizer:
    def optimize(self, objective):
        pass  # To be implemented by subclasses...

# GOOD: Concrete optimizer, then extract commonality if needed
def gradient_descent(objective, x0, learning_rate):
    # Specific, working implementation
```

**Why Bad:**
- Speculative design often wrong
- Complexity without benefit
- Hard to change once entrenched

**Rule:** Always implement 2-3 concrete cases before abstracting.

---

### ❌ Anti-Pattern 2: Feature Creep

**Problem:** Adding features because they're "cool" or "easy," not because they're needed.

**Example:**
```morphogen
// BAD: Adding Unicode operators just because we can
let result = a ⊕ b ⊗ c  // What does this mean? Why not +, *?

// GOOD: Stick to ASCII, clear semantics
let result = a + b * c
```

**Why Bad:**
- Increases learning curve
- Obscures code intent
- Creates maintenance burden

**Rule:** Every feature must justify its existence with concrete use cases.

---

### ❌ Anti-Pattern 3: Breaking Orthogonality

**Problem:** Features that interact in unexpected ways.

**Example:**
```morphogen
// BAD: Implicit coupling between unrelated features
@global_setting resampling_mode = "cubic"
audio_out = some_circuit.output  // Uses global setting implicitly!

// GOOD: Explicit coupling
audio_out = circuit.to_audio(some_circuit.output,
                               resample_method="cubic",
                               sample_rate=48kHz)
```

**Why Bad:**
- Hidden dependencies
- Action-at-a-distance bugs
- Hard to reason about

**Rule:** Keep features independent. Explicit > implicit, always.

---

### ❌ Anti-Pattern 4: Ignoring Performance

**Problem:** Adding features without considering performance implications.

**Example:**
```python
# BAD: Naive field-agent coupling (O(n²))
for agent in agents:
    for i in range(field.width):
        for j in range(field.height):
            if distance(agent.pos, (i, j)) < radius:
                field[i, j] += agent.value

# GOOD: Use spatial hashing (O(n))
grid = build_spatial_hash(agents)
for (i, j), agents_in_cell in grid.items():
    for agent in agents_in_cell:
        field[i, j] += agent.value
```

**Why Bad:**
- Unusable for realistic problem sizes
- Gives Morphogen reputation for slowness
- Hard to fix later without breaking API

**Rule:** Profile and benchmark every new operator. Document complexity.

---

### ❌ Anti-Pattern 5: Documentation Debt

**Problem:** Implementing features without documentation, planning to "document later."

**Reality:** Later never comes. Undocumented features are unusable and unmaintainable.

**Rule:** Documentation is part of implementation. PR doesn't merge without:
- Docstring with signature, semantics, example
- Test demonstrating usage
- Update to domain reference guide

---

## Prioritization Framework

### The 2x2 Matrix

```
High Value  |  P0: Do Now        |  P1: Do Next
            |  (Circuit, GPU)    |  (Geometry, Symbolic)
            |--------------------|--------------------
Low Value   |  P3: Maybe Later   |  P2: Consider
            |  (Finance, BI)     |  (Pattern match)
            |
            Low Effort           High Effort
```

**P0 - Do Now (High Value, Low-Medium Effort):**
- Circuit domain (🔥🔥🔥 value, 🔴 effort)
- GPU acceleration (🔥🔥 value, 🔴 effort)
- Transform space tracking (🔥🔥 value, 🟡 effort)

**P1 - Do Next (High Value, High Effort or Medium Value, Low Effort):**
- Geometry domain (🔥🔥 value, 🔴 effort)
- Symbolic math (🔥 value, 🟡 effort)
- Category theory optimization (🔥🔥 value, 🔴 effort)

**P2 - Consider (Medium Value, Medium Effort):**
- Pattern matching (🔥 value, 🟡 effort)
- Control domain (🔥🔥 value, 🟢 effort, but less urgency)

**P3 - Maybe Later (Low Value or Extreme Effort):**
- Finance domain (🔥 value, 🟡 effort, niche)
- BI domain (🔥 value, 🟡 effort, niche)
- Effect system (🔥🔥 value, ⚫ effort, defer to v2.0)

---

## Decision Templates

### Template 1: Domain Addition Proposal

```markdown
# Domain Proposal: <Domain Name>

## Strategic Value (Lens 1)
- **Professional Field**: [Which industry/field does this enable?]
- **User Persona**: [Who uses this? Background, needs, pain points]
- **Competing Tools**: [What does this replace? MATLAB, Abaqus, etc.]
- **Value Score**: 🔥🔥🔥 Critical / 🔥🔥 High / 🔥 Medium / ❄️ Low

## Composability (Lens 2)
- **Dependencies**: [Which existing domains does this rely on?]
- **Coupling Points**: [What cross-domain operators needed?]
- **Interactions**: [Any conflicts or edge cases?]
- **Score**: ✅ Orthogonal / ⚠️ Minor / 🚨 Complex / 🛑 Breaking

## Implementation (Lens 3)
- **Operator Count**: [Minimum viable: X operators, Full: Y operators]
- **Algorithmic Complexity**: [Novel algorithms? Standard implementations?]
- **Dependencies**: [External libraries? LAPACK, FFTW, etc.]
- **Testing Scope**: [How many tests? Edge cases?]
- **Effort Score**: 🟢 Low / 🟡 Medium / 🔴 High / ⚫ Extreme

## Architecture (Lens 4)
- **Layer**: Kernel / Domain Library / Frontend
- **Kernel Changes**: [Any required? Type system, scheduler, etc.]
- **MLIR Lowering**: [Custom dialect? Or reuse existing?]
- **Fit Score**: 5 (correct layer) / -5 (wrong layer)

## Determinism (Lens 5)
- **Profile Support**: All / Repro+Live / Live Only
- **Nondeterminism Sources**: [RNG, I/O, external data, etc.]
- **Mitigation**: [How to make deterministic?]
- **Score**: ✅ All / ⚠️ Repro/Live / 🚨 Live / 🛑 Breaking

## Community (Lens 6)
- **User Requests**: [GitHub issues, forum posts]
- **Academic Precedent**: [Papers, textbooks]
- **Industry Practice**: [Professional tools using this]
- **Score**: 5 (strong evidence) / 3 (weak) / 0 (none)

## Decision Matrix

| Strategic | Composability | Effort | Arch Fit | Determinism | Community | **Total** |
|-----------|---------------|--------|----------|-------------|-----------|-----------|
| X         | X             | X      | X        | X           | X         | **XX/25** |

## Recommendation

[ACCEPT / CONSIDER / DEFER / REJECT] - [Rationale]

## Implementation Plan

[If accepted: phases, milestones, tests, docs]

## References

- [Papers, tools, prior art]
```

---

### Template 2: Feature Addition Proposal

```markdown
# Feature Proposal: <Feature Name>

## Problem Statement
[What problem does this solve? Concrete examples.]

## Proposed Solution
[Syntax, semantics, examples]

## Alternatives Considered
[What else could solve this? Pros/cons.]

## Six Lenses Analysis
1. **Strategic Value**: [Score + rationale]
2. **Composability**: [Score + edge cases]
3. **Implementation**: [Score + complexity]
4. **Architecture**: [Layer + fit]
5. **Determinism**: [Profile support]
6. **Community**: [Prior art]

## Decision Matrix
[Fill out table]

## Recommendation
[ACCEPT / CONSIDER / DEFER / REJECT]
```

---

## Governance & Process

### Who Decides?

**For Kernel Changes:**
- Requires unanimous agreement (all core contributors)
- Formal RFC process with 2-week review
- Extensive testing and validation
- **Rationale:** Kernel is immutable post-1.0, must get it right

**For Domain Additions:**
- Core contributor approval (1-2 reviewers)
- RFC recommended for complex domains (>30 operators)
- Comprehensive testing required
- **Rationale:** Domains can be deprecated, lower risk

**For Frontend Changes:**
- Single core contributor approval
- Rapid iteration encouraged
- User feedback prioritized
- **Rationale:** Frontends are syntax sugar, easily changed

### RFC Process

For significant changes (kernel, major domains, breaking changes):

1. **Draft RFC** using template above
2. **Post for community review** (GitHub Discussions)
3. **Two-week review period** (comments, alternatives)
4. **Revise based on feedback**
5. **Decision by core team** (accept, defer, reject)
6. **If accepted:** Create ADR, implement, test, document

---

## Measuring Success

### Capability Metrics

**Coverage:**
- Number of professional domains supported (target: 50+ by v1.0)
- Breadth of operator catalog (target: 750+ operators)
- Cross-domain coupling examples (target: 25+ examples)

**Depth:**
- Production-grade implementations (not toys)
- Performance benchmarks (competitive with domain-specific tools)
- Test coverage (>90% for all domains)

**Integration:**
- Cross-domain examples (audio+circuit, geometry+physics, etc.)
- Type safety (compile-time errors for invalid compositions)
- Performance (no cliffs at domain boundaries)

### Composability Metrics

**Predictability:**
- User survey: "Composition behaved as I expected" (target: >80% agree)
- GitHub issues tagged "surprising behavior" (target: <5 per release)

**Safety:**
- Compile-time error rate (vs runtime errors)
- Type system catch rate (invalid compositions caught)

**Orthogonality:**
- Feature interaction bugs (target: <2% of bugs)
- Regression rate (new features break old code, target: <1%)

---

## Summary

This framework provides:

1. **Six Lenses** for evaluating additions (strategic, composability, effort, architecture, determinism, community)
2. **Decision Matrix** with scoring and thresholds
3. **Common Scenarios** (domains, features, operators) with processes
4. **Anti-Patterns** to avoid (premature abstraction, feature creep, etc.)
5. **Prioritization Framework** (P0-P3 based on value/effort)
6. **Decision Templates** (structured proposals for domains and features)
7. **Governance Process** (who decides, RFC process, timelines)
8. **Success Metrics** (measuring capability growth)

**Use this document** when:
- Proposing a new domain or feature
- Evaluating community requests
- Planning roadmap priorities
- Resolving architectural discussions

**Next Steps:**
1. Apply this framework to active proposals (Circuit, Geometry, Symbolic)
2. Create RFCs for high-priority additions
3. Track decisions in ADRs
4. Review framework quarterly (is it working?)

---

**Document Status:** Reference Guide (Living Document)
**Review Cycle:** Quarterly
**Next Review:** After v0.12.0 release
