# Morphogen Documentation

Welcome to the Morphogen documentation! This guide will help you navigate the documentation based on what you want to accomplish.

> 💡 **First time here?** Start with the main [README.md](../README.md) to understand Morphogen's vision and what makes it unique. Then come back here for detailed technical documentation.

---

## 🆕 New to Morphogen? Start Here

**Choose your path:**

### 🚀 Quick Start (15 minutes)
1. [Installation](getting-started.md#installation) (5 min)
2. [Your first program](getting-started.md#your-first-program) (5 min)
3. [Run an example](../examples/) (5 min)

### 🎓 Learning Path (2-3 hours)
1. Read [Why Morphogen Exists](../README.md#why-morphogen-exists) (10 min)
2. Try [Complete Examples](getting-started.md#complete-examples) (30 min)
3. Understand [Core Concepts](getting-started.md#core-concepts) (30 min)
4. Explore [Domain Catalog](DOMAINS.md) - pick 2-3 domains (30 min)
5. Read [Architecture Overview](architecture/overview.md) (30 min)

### 🔬 Deep Dive (Advanced)
1. [Philosophy](philosophy/) - Why Morphogen is designed this way
2. [Architecture](architecture/) - How it's implemented
3. [Specifications](specifications/) - Technical details
4. [ADRs](adr/) - Design decisions

**Not sure where to start?** See [Finding What You Need](#finding-what-you-need) below

---

## Quick Start

- **New to Morphogen?** Start with [Getting Started](getting-started.md) for installation and your first program
- **Understand the architecture?** Read [Architecture Overview](architecture/overview.md)
- **See the full ecosystem?** Check [ECOSYSTEM_MAP.md](architecture/ECOSYSTEM_MAP.md) for all domains and roadmap
- **Explore cross-domain patterns?** See [Cross-Domain Architectural Patterns](adr/002-cross-domain-architectural-patterns.md) and [Domain Mesh Catalog](reference/domain-mesh-catalog.md)
- **Browse all documentation?** See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) ⭐ — Complete map with reveal tool usage
- **Explore efficiently?** Use the [reveal tool](../scripts/README.md) (`./scripts/reveal.sh`) for incremental documentation exploration
- **Need help?** Check [Troubleshooting](troubleshooting.md)

---

## 🚀 **v1.0 Roadmap Active**

**Morphogen is on a focused path to v1.0!**

📋 **[Read the Unified Roadmap](ROADMAP.md)** ⭐

**Key Milestones:**
- **v0.12.0** (Q1 2026): Domain Migration - bring 15 legacy domains into modern registry
- **v1.0** (Q2 2026): Production release - 40+ domains, PyPI, community infrastructure

**Three-track strategy:**
- **Track 1:** Language evolution (type inference, error messages, performance)
- **Track 2:** Domain completion (migrate legacy, polish active domains)
- **Track 3:** Adoption & Polish (PyPI, docs, tutorials, community)

See [ROADMAP.md](ROADMAP.md) for complete details and implementation tracking.

---

## Documentation Structure

### 🧠 [Philosophy](philosophy/)
**Theoretical foundations and epistemological context** (answers "WHY")
- [Formalization and Knowledge](philosophy/formalization-and-knowledge.md) ⭐ — How formalization transforms human knowledge
- [Universal DSL Principles](philosophy/universal-dsl-principles.md) ⭐ **NEW** — Design brief for cross-domain DSLs (the "why")
- [Operator Foundations](philosophy/operator-foundations.md) — Mathematical operator theory and spectral methods
- [Categorical Structure](philosophy/categorical-structure.md) — Category-theoretic formalization
- [Philosophy README](philosophy/README.md) — Overview of philosophical foundations

**Note:** Philosophy docs establish "why" Morphogen is designed this way. For "how" to implement it, see [Architecture](architecture/).

### 📐 [Architecture](architecture/)
**High-level design and architectural principles** (answers "HOW")
- [Overview](architecture/overview.md) - Core Morphogen architecture
- [Continuous-Discrete Semantics](architecture/continuous-discrete-semantics.md) ⭐ **NEW** — Dual computational models
- **[DSL Framework Design](architecture/dsl-framework-design.md)** ⭐ **NEW** - Vision for domain reasoning language (the "how" - first-class domains, translations, composition)
- [Domain Architecture](architecture/domain-architecture.md) - How domains fit together (110KB - comprehensive!)
- [GPU & MLIR Principles](architecture/gpu-mlir-principles.md) - GPU execution and MLIR integration
- [Interactive Visualization](architecture/interactive-visualization.md) - Visualization approach

**Note:** Architecture docs explain "how" to implement the principles from [Philosophy](philosophy/).

### 📋 [Specifications](specifications/)
Detailed technical specifications (21 documents)
- **Language**: [KAX Language](specifications/kax-language.md), [Type System](specifications/type-system.md), **[Level 3 Type System](specifications/level-3-type-system.md)** ⭐ **NEW** — Cross-domain type safety
- **Infrastructure**: [Graph IR](specifications/graph-ir.md), [MLIR Dialects](specifications/mlir-dialects.md), [Operator Registry](specifications/operator-registry.md), [Scheduler](specifications/scheduler.md), [Transform](specifications/transform.md), [Transform Composition](specifications/transform-composition.md)
- **Domains**: [Chemistry](specifications/chemistry.md), [Circuit](specifications/circuit.md), [Emergence](specifications/emergence.md), [Procedural Generation](specifications/procedural-generation.md), [Physics](specifications/physics-domains.md), [BI](specifications/bi-domain.md), [Video/Audio Encoding](specifications/video-audio-encoding.md)
- **Other**: [Geometry](specifications/geometry.md), [Coordinate Frames](specifications/coordinate-frames.md), [Profiles](specifications/profiles.md), [Snapshot ABI](specifications/snapshot-abi.md), [Timbre Extraction](specifications/timbre-extraction.md)

### 📝 [Architecture Decision Records (ADRs)](adr/)
Why key architectural decisions were made (12 records)
- [001: Unified Reference Model](adr/001-unified-reference-model.md)
- [002: Cross-Domain Architectural Patterns](adr/002-cross-domain-architectural-patterns.md)
- [003: Circuit Modeling Domain](adr/003-circuit-modeling-domain.md)
- [004: Instrument Modeling Domain](adr/004-instrument-modeling-domain.md)
- [005: Emergence Domain](adr/005-emergence-domain.md)
- [006: Chemistry Domain](adr/006-chemistry-domain.md)
- [007: GPU-First Domains](adr/007-gpu-first-domains.md)
- [008: Procedural Generation Domain](adr/008-procedural-generation-domain.md)
- [009: Ambient Music & Generative Domains](adr/009-ambient-music-generative-domains.md)
- [010: Ecosystem Branding & Naming Strategy](adr/010-ecosystem-branding-naming-strategy.md)
- [011: Project Renaming (Morphogen/Philbrick)](adr/011-project-renaming-morphogen-philbrick.md)
- [012: Universal Domain Translation](adr/012-universal-domain-translation.md) ⭐ **NEW**

### 📖 [Guides](guides/)
How-to documentation for implementers
- [Domain Implementation Guide](guides/domain-implementation.md) - How to add new domains to Morphogen

### 💡 [Examples](examples/)
Complete working examples demonstrating Morphogen capabilities
- [Emergence Cross-Domain](examples/emergence-cross-domain.md)
- [J-Tube Firepit Multiphysics](examples/j-tube-firepit-multiphysics.md)
- [Kerbal Space Program Simulation](examples/kerbal-space-program.md)
- [Racing AI Pipeline](examples/racing-ai-pipeline.md)

### 🎯 [Use Cases](use-cases/)
Specific real-world applications
- [2-Stroke Muffler Modeling](use-cases/2-stroke-muffler-modeling.md)
- [Chemistry Unified Framework](use-cases/chemistry-unified-framework.md)

### 📚 [Reference](reference/)
**Catalogs, operator references, and domain overviews** (~420KB across 18 documents)

**📖 Start here:** [Reference README](reference/README.md) ⭐ — Comprehensive index with navigation by task and experience level

**Key sections:**
- **Operator Catalogs** (~154KB): Complete implementation-ready operator libraries
- **Visualization & Sonification** (~115KB): Comprehensive visualization and sonification techniques
- **Theoretical Frameworks** (~84KB): Mathematical foundations and pedagogical metaphors
- **Patterns & Best Practices** (~72KB): Battle-tested architectural patterns
- **Domain Overviews** (~40KB): High-level domain capabilities

**Quick links:**
- [Multiphysics Success Patterns](reference/multiphysics-success-patterns.md) ⭐ - 12 battle-tested patterns
- [Visualization Ideas by Domain](reference/visualization-ideas-by-domain.md) ⭐ - Comprehensive catalog (56K)
- [Universal Domain Frameworks](reference/universal-domain-frameworks.md) ⭐ - Mathematical foundations (28K)
- [Mathematical Transformation Metaphors](reference/math-transformation-metaphors.md) - Intuitive understanding (25K)

### 🗺️ Roadmap & Planning

**Primary**: **[ROADMAP.md](ROADMAP.md)** ⭐ — Unified development roadmap (v0.12.0 → v1.0)

**Feature-Specific**: [roadmap/](roadmap/)
- [Language Features](roadmap/language-features.md) — Language evolution roadmap
- [Testing Strategy](roadmap/testing-strategy.md) — Testing approach

**Strategic Planning**: [planning/](planning/)
- [KAIRO_2.0_STRATEGIC_ANALYSIS.md](planning/KAIRO_2.0_STRATEGIC_ANALYSIS.md) — v2.0+ language strategy
- See [ARCHIVE_HISTORY.md](ARCHIVE_HISTORY.md) for historical planning documents

### 📦 [Archive History](ARCHIVE_HISTORY.md) ⭐ NEW
Documentation of archived materials and their locations

**What's archived:**
- Session artifacts from development sessions (point-in-time snapshots)
- Historical planning documents (superseded by unified roadmap)
- Analysis reports from v0.8.0 era (outdated technical analyses)
- Mesh documentation (pre-consolidation)

**Where:** `~/Archive/morphogen/` - See [ARCHIVE_HISTORY.md](ARCHIVE_HISTORY.md) for complete details and inventory

---

## Finding What You Need

**I want to...**

- **Browse all documentation** → See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) ⭐ — Complete index with navigation tips
- **Explore documents incrementally** → Use `../scripts/reveal.sh` ([see docs](../scripts/README.md)) for token-efficient exploration
  - Quick preview: `./scripts/reveal.sh 1 <file>` (structure only)
  - Sample content: `./scripts/reveal.sh 2 <file>` (representative preview)
- **Understand why formalization matters** → Read [Formalization and Knowledge](philosophy/formalization-and-knowledge.md) ⭐
- **Understand Morphogen's mathematical foundations** → See [Philosophy](philosophy/) section
- **Understand Morphogen's vision and impact** → Read the main [README.md](../README.md)
- **Understand Morphogen's architecture** → Start with [Architecture Overview](architecture/overview.md), then [Domain Architecture](architecture/domain-architecture.md)
- **Understand transformations intuitively** → Read [Mathematical Transformation Metaphors](reference/math-transformation-metaphors.md)
- **See the complete ecosystem** → Check [ECOSYSTEM_MAP.md](architecture/ECOSYSTEM_MAP.md) for all domains and roadmap
- **Understand language roadmap** → Read [Language Features Roadmap](roadmap/language-features.md) ⭐ — Path to 1.0, finalized vs planned features
- **Understand cross-domain type safety** → Check [Level 3 Type System](specifications/level-3-type-system.md)
- **Implement a new domain** → Read [Domain Implementation Guide](guides/domain-implementation.md) and relevant [ADRs](adr/)
- **Learn about a specific domain** → Check [Specifications](specifications/) for the domain spec, then related [ADRs](adr/)
- **See Morphogen in action** → Browse [Examples](examples/) and [Use Cases](use-cases/)
- **Find specific operators** → Search [Reference](reference/) for operator catalogs
- **Understand a design decision** → Look in [ADRs](adr/)
- **Track project progress** → See [Roadmap](roadmap/)
- **Debug an issue** → Start with [Troubleshooting](troubleshooting.md)

---

## Recent Changes

**2025-11-21: Major Documentation Reorganization & Language Roadmap**
- ✅ **Documentation Organization** - Consolidated and clarified documentation structure:
  - Moved analysis docs from root to `docs/analysis/` (DOMAIN_VALIDATION_REPORT, MORPHOGEN_RENAME_ANALYSIS)
  - Moved type system spec to `docs/specifications/level-3-type-system.md`
  - Consolidated archives: moved `archive/historical/` and `archive/root-level-docs/` to `docs/archive/`
  - All documentation now properly organized in `docs/` subdirectories
- ✅ **Language Features Roadmap** ⭐ **NEW** - Created comprehensive [Language Features Roadmap](roadmap/language-features.md):
  - Clear path to language 1.0 (target: 2026 Q2)
  - Production-ready vs planned features (physical units, cross-domain types, MLIR optimization)
  - Features under discussion (macros, effect system, ownership, pattern matching)
  - Decision framework for accepting new features
  - Community input process
- ✅ **Enhanced Navigation** - Updated docs README with:
  - Links to newly organized documents
  - Language roadmap in prominent position
  - Cross-domain type system specification
- ✅ **Version Consistency** - Updated STATUS.md header to clarify "Morphogen (Morphogen)"

**2025-11-21: Documentation Improvements & Clarifications** (earlier today)
- ✅ **Enhanced Reference Section** - Created comprehensive [Reference README](reference/README.md) with:
  - Navigation by task and experience level
  - Document relationships and cross-references
  - Statistics and breakdown by category (~420KB total)
- ✅ **Clarified Overlapping Docs** - Added purpose sections to distinguish:
  - [Universal DSL Principles](philosophy/universal-dsl-principles.md) (the "why" - design philosophy)
  - [DSL Framework Design](architecture/dsl-framework-design.md) (the "how" - implementation vision)
- ✅ **Improved Navigation** - Added "WHY" vs "HOW" labels to Philosophy and Architecture sections

**2025-11-21: New Theoretical Foundation Documents (~150KB)**
- 🆕 **Philosophy Section** - Comprehensive theoretical framework:
  - [Formalization and Knowledge](philosophy/formalization-and-knowledge.md) (13K) - Historical context
  - [Operator Foundations](philosophy/operator-foundations.md) (18K) - Mathematical operator theory
  - [Categorical Structure](philosophy/categorical-structure.md) (26K) - Category-theoretic formalization
  - [Universal DSL Principles](philosophy/universal-dsl-principles.md) (18K) - 8 design principles
  - [Philosophy README](philosophy/README.md) (11K) - Section index
- 🆕 **Architecture Documents**:
  - [DSL Framework Design](architecture/dsl-framework-design.md) (23K) - Implementation vision
  - [Continuous-Discrete Semantics](architecture/continuous-discrete-semantics.md) - Dual computational models
- 🆕 **New ADR**: [012: Universal Domain Translation](adr/012-universal-domain-translation.md) (14K)
- 🆕 **New Spec**: [Transform Composition](specifications/transform-composition.md) (15K) - Composable transforms

**2025-11-16: Major Documentation Reorganization**
- ✅ Created `planning/` and `analysis/` directories for better organization
- ✅ Moved strategic planning docs from root to `docs/planning/`
- ✅ Moved analysis documents from root to `docs/analysis/`
- ✅ Moved orphaned docs to proper locations (`specifications/`, `guides/`)
- ✅ Updated all version numbers to v0.10.0 (23 domains implemented)
- ✅ Fixed inconsistencies across README, STATUS, and SPECIFICATION

**2025-11-15: Initial Documentation Reorganization**
- ✅ Consistent lowercase naming throughout
- ✅ Logical grouping by user intent (why/what/how/how-to)
- ✅ Fixed ADR numbering (resolved duplicate 003 and 005 issues)
- ✅ All specifications in one directory for easy discovery
- ✅ Clear navigation paths for different user types
- ✅ Moved patterns catalog out of ADRs to reference section

All file history has been preserved using `git mv`.

---

## Documentation Best Practices

### Using the Reveal Tool for Efficient Exploration

The reveal tool (`scripts/reveal.sh`) is designed for incremental documentation exploration, saving time and tokens:

```bash
# Start with structure (level 1) - see headings without full content
./scripts/reveal.sh 1 docs/architecture/domain-architecture.md

# Preview sample content (level 2) - representative sections
./scripts/reveal.sh 2 docs/specifications/chemistry.md

# Full content (level 3) - only when you need complete details
./scripts/reveal.sh 3 docs/guides/domain-implementation.md
```

**Recommended workflow:**
1. **Survey first**: Use level 1 to see document structure
2. **Sample strategically**: Use level 2 to preview interesting sections
3. **Read selectively**: Only read full docs (level 3 or direct read) when needed

**Token savings**: Level 1 uses ~5-10% of tokens, level 2 uses ~20-30%, versus 100% for full reads.

See [scripts/README.md](../scripts/README.md) for complete reveal tool documentation.

### Cross-Referencing Conventions

- **Philosophy vs Architecture**: Philosophy docs explain "WHY" (design principles), Architecture docs explain "HOW" (implementation approach)
- **ADRs reference specs**: Check ADRs to understand design decisions, then read related specifications for implementation details
- **Examples reference specs**: Working examples in `examples/` demonstrate concepts from `specifications/`
- **Guides reference everything**: Implementation guides in `guides/` tie together philosophy, architecture, specs, and ADRs
