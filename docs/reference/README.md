# Reference Documentation

**Purpose:** Quick-reference catalogs, operator libraries, theoretical frameworks, and practical patterns for Morphogen developers.

**Audience:** Implementers, domain experts, researchers looking for specific operators or patterns.

---

## 📖 How to Use This Section

| If you want to... | Start here... |
|-------------------|---------------|
| **Find a specific operator** | Search the [Operator Catalogs](#operator-catalogs) |
| **Understand a domain** | Check [Domain Overviews](#domain-overviews) |
| **Build a multiphysics system** | Read [Multiphysics Success Patterns](#patterns--best-practices) |
| **Understand transformations intuitively** | See [Mathematical Transformation Metaphors](#theoretical-frameworks) |
| **Visualize your domain** | Browse [Visualization & Sonification](#visualization--sonification) |
| **Explore music theory integration** | Review [Mathematical Music Frameworks](#theoretical-frameworks) |
| **Understand theoretical foundations** | See [Universal Domain Frameworks](#theoretical-frameworks) |

---

## Operator Catalogs

**Complete, implementation-ready catalogs** of all operators in each domain:

### Core Operator Libraries

| Catalog | Size | Description |
|---------|------|-------------|
| [Emergence Operators](emergence-operators.md) | 30K | Genetic algorithms, particle swarms, cellular automata, ant colonies |
| [Procedural Operators](procedural-operators.md) | 21K | Noise functions, fractals, L-systems, Wave Function Collapse |
| [Genetic Algorithm Operators](genetic-algorithm-operators.md) | 30K | Selection, crossover, mutation, fitness functions |
| [Optimization Algorithms](optimization-algorithms.md) | 44K | Gradient descent, PSO, simulated annealing, evolution strategies |
| [Time Alignment Operators](time-alignment-operators.md) | 29K | Temporal synchronization, alignment, warping |

**Each catalog includes:**
- ✅ Type-safe signatures
- ✅ Complete parameter specifications
- ✅ Semantics and use cases
- ✅ Code examples
- ✅ Cross-domain integration patterns

**For implementation details**, see [Specifications](../specifications/).

---

## Domain Overviews

**High-level overviews** of domain capabilities and organization:

| Overview | Size | Description |
|----------|------|-------------|
| [Procedural Graphics Domains](procedural-graphics-domains.md) | 14K | Noise, palettes, color theory, image operations |
| [Visual Scene Domain](visual-scene-domain.md) | 18K | 3D scene representation, rendering, lighting |
| [Visual Domain Quick Reference](visual-domain-quickref.md) | 8K | Quick reference for 2D visual/animation operations |
| [Visual3D Quick Reference](visual3d-quickref.md) ⭐ | 12K | **3D visualization API** - PyVista-based mesh, volume, molecular viz |

**For complete domain specifications**, see [Specifications](../specifications/).

---

## Visualization & Sonification

**Comprehensive catalogs** of visualization and sonification techniques:

| Resource | Size | Description |
|----------|------|-------------|
| [Visualization Ideas by Domain](visualization-ideas-by-domain.md) ⭐ | 56K | **Comprehensive catalog** - Visualization concepts organized by domain (audio, physics, fields, agents, optimization, graphs, chemistry, etc.) |
| [Audio Visualization Ideas](audio-visualization-ideas.md) ⭐ | 40K | **Sonification & audio-visual coupling** - Making computation audible through data sonification |
| [Advanced Visualizations](advanced-visualizations.md) | 19K | Implemented visualization techniques (spectrograms, phase space, vector fields, particle systems) |
| [Visual3D Quick Reference](visual3d-quickref.md) ⭐ | 12K | **3D visualization API** - Mesh, volume rendering, streamlines, molecular viz (PyVista) |

**Key Features:**
- Real-time visualization patterns
- Cross-domain visualization mappings
- Interactive visualization techniques
- Sonification strategies for debugging and exploration

---

## Patterns & Best Practices

**Battle-tested patterns** for building robust Morphogen systems:

| Pattern Guide | Size | Description |
|---------------|------|-------------|
| [Multiphysics Success Patterns](multiphysics-success-patterns.md) ⭐ | 23K | **12 architectural patterns** for multiphysics simulation (coupling strategies, time integration, boundary conditions, stability, etc.) |
| [Operator Registry Expansion](operator-registry-expansion.md) | 49K | How the operator registry grows and evolves - adding new operators, domains, and compositions |

**When to use:**
- Building coupled multi-domain systems
- Ensuring stability and accuracy
- Optimizing performance
- Extending Morphogen with new domains

---

## Theoretical Frameworks

**Mathematical and conceptual foundations** for understanding Morphogen's design:

### Universal Frameworks

| Framework | Size | Description |
|-----------|------|-------------|
| [Universal Domain Frameworks](universal-domain-frameworks.md) ⭐ | 28K | **Category theory, functorial semantics, universal algebra** - The mathematical foundations of universal domain computation |
| [Mathematical Transformation Metaphors](math-transformation-metaphors.md) | 25K | **Pedagogical metaphors** - Intuitive frameworks for understanding transformations (rotations, projections, flows) |
| [Mathematical Music Frameworks](mathematical-music-frameworks.md) ⭐ | 31K | **10 mathematical domains** that map to musical structure (group theory, topology, dynamical systems, information theory, spectral methods) |

**Cross-References:**
- For philosophical context, see [Philosophy](../philosophy/)
- For implementation details, see [Architecture](../architecture/)
- For formal specifications, see [Specifications](../specifications/)

### How Frameworks Connect

```
Universal Domain Frameworks (math foundations)
    ↓
Philosophy (why formalization matters)
    ↓
Architecture (how Morphogen implements it)
    ↓
Reference (practical operators and patterns)
    ↓
Specifications (detailed implementation)
```

---

## Conventions & Standards

**Naming and organizational standards** for Morphogen development:

| Standard | Size | Description |
|----------|------|-------------|
| [Naming Conventions](NAMING_CONVENTIONS.md) | 8K | Naming standards for code, operators, domains, and documentation |

---

## Document Relationships

### Philosophy ↔ Reference Connection

| Philosophy Doc | Related Reference | Relationship |
|----------------|-------------------|--------------|
| [Operator Foundations](../philosophy/operator-foundations.md) | [Universal Domain Frameworks](universal-domain-frameworks.md) | Math theory (philosophy) → Practical foundations (reference) |
| [Universal DSL Principles](../philosophy/universal-dsl-principles.md) | [Multiphysics Success Patterns](multiphysics-success-patterns.md) | Design principles → Implementation patterns |
| [Formalization and Knowledge](../philosophy/formalization-and-knowledge.md) | [Mathematical Music Frameworks](mathematical-music-frameworks.md) | Why formalization matters → Specific formalization example |

### Architecture ↔ Reference Connection

| Architecture Doc | Related Reference | Relationship |
|------------------|-------------------|--------------|
| [DSL Framework Design](../architecture/dsl-framework-design.md) | [Universal Domain Frameworks](universal-domain-frameworks.md) | Implementation vision → Theoretical foundation |
| [Domain Architecture](../architecture/domain-architecture.md) | [Domain Overviews](procedural-graphics-domains.md) | Complete specification → Quick reference |
| [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) | [Multiphysics Success Patterns](multiphysics-success-patterns.md) | Dual computational models → Practical coupling patterns |

---

## Quick Navigation

**By Task:**
- 🔍 **Searching for operators** → [Operator Catalogs](#operator-catalogs)
- 📊 **Visualizing data** → [Visualization & Sonification](#visualization--sonification)
- 🏗️ **Building systems** → [Patterns & Best Practices](#patterns--best-practices)
- 🧠 **Understanding theory** → [Theoretical Frameworks](#theoretical-frameworks)
- 🎯 **Learning domains** → [Domain Overviews](#domain-overviews)

**By Experience Level:**
- **Beginners** → Start with [Domain Overviews](#domain-overviews)
- **Implementers** → Focus on [Operator Catalogs](#operator-catalogs) and [Patterns](#patterns--best-practices)
- **Researchers** → Dive into [Theoretical Frameworks](#theoretical-frameworks)
- **Visual thinkers** → Explore [Visualization & Sonification](#visualization--sonification)

---

## Statistics

**Total Reference Documentation:** ~432KB across 19 documents

**Breakdown by Category:**
- Operator Catalogs: ~154KB (5 docs)
- Visualization/Sonification: ~127KB (4 docs)
- Theoretical Frameworks: ~84KB (3 docs)
- Domain Overviews: ~52KB (4 docs)
- Patterns & Standards: ~72KB (2 docs)

---

[← Back to Documentation Home](../README.md)
