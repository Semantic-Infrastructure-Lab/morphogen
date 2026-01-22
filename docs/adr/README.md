# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting key architectural decisions in Morphogen's development.

## What is an ADR?

An ADR captures:
- **Context**: What problem we're solving
- **Decision**: What we decided to do
- **Consequences**: What the decision means for the project

ADRs explain *why* decisions were made, not *how* to implement them (see [Specifications](../specifications/) for implementation details).

## Current ADRs

### [ADR-001: Unified Reference Model](001-unified-reference-model.md)
Establishes Morphogen's unified reference model for cross-domain simulation.

### [ADR-002: Cross-Domain Architectural Patterns](002-cross-domain-architectural-patterns.md)
Defines patterns for integrating multiple domains in a single system.

### [ADR-003: Circuit Modeling Domain](003-circuit-modeling-domain.md)
Decision to add circuit modeling as a first-class domain.

### [ADR-004: Instrument Modeling Domain](004-instrument-modeling-domain.md)
Decision to add audio instrument modeling capabilities.

### [ADR-005: Emergence Domain](005-emergence-domain.md)
Adds emergence and complex systems simulation domain.

### [ADR-006: Chemistry Domain](006-chemistry-domain.md)
Decision to support chemistry and chemical engineering workflows.

### [ADR-007: GPU-First Domains](007-gpu-first-domains.md)
Major paradigm shift: GPU-first execution for certain domains (BI domain).

### [ADR-008: Procedural Generation Domain](008-procedural-generation-domain.md)
Adds procedural content generation as a core domain.

### [ADR-009: Ambient Music / Generative Domains](009-ambient-music-generative-domains.md)
Adds ambient music generation and generative audio domains.

### [ADR-010: Ecosystem Branding & Naming Strategy](010-ecosystem-branding-naming-strategy.md)
Establishes naming conventions and branding for the Morphogen ecosystem.

### [ADR-011: Project Renaming - Morphogen & Philbrick](011-project-renaming-morphogen-philbrick.md)
Renames Kairo → Morphogen (honoring Turing's morphogenesis work) and establishes Philbrick as the analog hardware sister project.

### [ADR-012: Universal Domain Translation](012-universal-domain-translation.md)
Defines patterns for translating concepts between domains.

### [ADR-013: Music Stack Consolidation](013-music-stack-consolidation.md)
Consolidates audio/music domains into unified architecture.

### [ADR-014: Complexity Refactoring Plan](014-complexity-refactoring-plan.md)
Addresses technical debt and complexity in the codebase.

### [ADR-015: First-Class Emergence Primitives](015-first-class-emergence-primitives.md) *(Proposed)*
Promotes attractors, constraints, time-scales, noise typing, and phase transitions to first-class status. Establishes the design principle: "Promote to first-class whatever the system stabilizes into—and whatever constrains what can stabilize."

---

## ADR Numbering

**Note**: This directory was recently reorganized (2025-11-15) to fix numbering collisions:
- Previously had two files numbered "003" and four numbered "005"
- ADRs have been renumbered sequentially
- The "multiphysics success patterns" document was moved to [Reference](../reference/) as it's a patterns catalog, not a decision record

All file history has been preserved using `git mv`.

---

## Related Documentation

- **Implementation details?** See [Specifications](../specifications/)
- **High-level architecture?** See [Architecture](../architecture/)
- **How to implement?** See [Guides](../guides/)
- **Battle-tested patterns?** See [Multiphysics Success Patterns](../reference/multiphysics-success-patterns.md)

[← Back to Documentation Home](../README.md)
