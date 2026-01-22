# Technical Specifications

Detailed technical specifications for Morphogen's language, infrastructure, and domains.

**Total:** 25 specifications

---

## Language & Type System

Core language design and type semantics.

| Specification | Status | Description |
|---------------|--------|-------------|
| [KAX Language](kax-language.md) | Research | The Morphogen language specification |
| [Type System](type-system.md) | Implemented | Type system design and semantics |
| [Level 3 Type System](level-3-type-system.md) | Implemented | Cross-domain type safety |
| [Units](units.md) | Implemented | Physical units system |

---

## Core Infrastructure

Execution engine, scheduling, and serialization.

| Specification | Status | Description |
|---------------|--------|-------------|
| [Graph IR](graph-ir.md) | Implemented | Intermediate representation for operator graphs |
| [MLIR Dialects](mlir-dialects.md) | Partial | MLIR dialect specifications |
| [Operator Registry](operator-registry.md) | Implemented | Operator registration and discovery |
| [Scheduler](scheduler.md) | Implemented | Execution scheduling and ordering |
| [Profiles](profiles.md) | Implemented | Execution profiles and optimization levels |
| [Snapshot ABI](snapshot-abi.md) | Implemented | Snapshot serialization format |

---

## Transforms & Composition

Graph transformations and operator composition.

| Specification | Status | Description |
|---------------|--------|-------------|
| [Transform](transform.md) | Implemented | Graph transformation system |
| [Transform Composition](transform-composition.md) | Implemented | Composing multiple transforms |

---

## Spatial & Geometric

Coordinate systems and geometry primitives.

| Specification | Status | Description |
|---------------|--------|-------------|
| [Coordinate Frames](coordinate-frames.md) | Implemented | Coordinate system handling |
| [Geometry](geometry.md) | Implemented | Geometric primitives and operations |

---

## Physics & Simulation Domains

Physical simulation and emergent systems.

| Specification | Status | Description |
|---------------|--------|-------------|
| [Physics Domains](physics-domains.md) | Implemented | Physics simulation (fluids, fields, particles) |
| [Chemistry](chemistry.md) | Implemented | Chemistry and chemical engineering |
| [Circuit](circuit.md) | Implemented | Electrical circuit modeling |
| [Emergence](emergence.md) | Partial | Emergence and complex systems |

---

## Procedural & Creative Domains

Content generation and GPU-first rendering.

| Specification | Status | Description |
|---------------|--------|-------------|
| [Procedural Generation](procedural-generation.md) | Implemented | Noise, fractals, L-systems, WFC |
| [BI Domain](bi-domain.md) | Research | GPU-first buffer imaging |

---

## Audio & Media Domains

Audio synthesis, analysis, and media encoding.

| Specification | Status | Description |
|---------------|--------|-------------|
| [Audio Synthesis](audio-synthesis.md) | Implemented | Audio synthesis operators |
| [Acoustics](acoustics.md) | Implemented | Acoustic modeling and wave equations |
| [Ambient Music](ambient-music.md) | Draft | Ambient music generation |
| [Timbre Extraction](timbre-extraction.md) | Implemented | Audio timbre analysis |
| [Video/Audio Encoding](video-audio-encoding.md) | Implemented | Media encoding and processing |

---

## Status Legend

| Status | Meaning |
|--------|---------|
| **Implemented** | Specification complete and implemented in code |
| **Partial** | Specification complete, implementation in progress |
| **Draft** | Specification draft, implementation planned |
| **Research** | Research specification, not yet scheduled |

---

## Navigation

- **New to Morphogen?** Start with [KAX Language](kax-language.md) and [Graph IR](graph-ir.md)
- **Implementing a domain?** Check the relevant spec, then [Domain Implementation Guide](../guides/domain-implementation.md)
- **Understanding why?** See [ADRs](../adr/) for design rationale
- **Need examples?** See [Examples](../examples/)

[← Back to Documentation Home](../README.md)
