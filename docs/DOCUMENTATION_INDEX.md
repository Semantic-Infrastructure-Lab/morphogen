# Morphogen Documentation Index

**Generated**: 2025-11-21
**Updated**: 2026-03-16
**Purpose**: Comprehensive index of all documentation with structure and navigation

> 💡 **Tip**: Use `reveal` to explore any document incrementally:
> ```bash
> reveal README.md              # structure (headings, functions, classes)
> reveal docs/specifications/   # directory layout
> reveal audio_analysis.py      # functions and classes in a source file
> reveal audio_analysis.py fit_exponential_decay  # single function
> ```
> **Token savings**: `reveal` = 5–50× cheaper than `Read` — always triage with reveal first

## Root-Level Documentation

### README.md (1,331 lines, 45KB)
**Purpose**: Project overview, getting started, domain catalog
**Key Sections**:
- Why Morphogen Exists
- The Name: A Structural Homage to Alan Turing
- Two Surfaces, One Kernel (Morphogen.Audio + RiffStack)
- Cross-Domain in Action
- Quick Start & Installation
- Language Overview (temporal model, state management, deterministic RNG)
- 15 Domain Catalogs (Field, RigidBody, Agent, Audio, Graph, Signal, etc.)
- Examples (Fluid Simulation, Reaction-Diffusion)
- Ecosystem Vision & Professional Applications
- Sister Project: Philbrick

**Navigate**: `reveal README.md`

---

### docs/DOMAIN_EXPANSION.md
**Purpose**: Post-v1.0 domain expansion plan — 8 candidate domains (controls, fem, orbit, epidemiology, photonics, quantum, economics, robotics) with operator specs, effort estimates, and cross-domain compositions each unlocks
**Key Sections**:
- Tier 1 (controls, fem, orbit, epidemiology) — high impact, strong cross-domain stories
- Tier 2 (photonics, quantum, robotics, economics) — strong value, more specialized
- Tier 3 — niche or long-horizon candidates
- Recommended implementation order + per-domain effort summary

**Navigate**: `reveal docs/DOMAIN_EXPANSION.md`

---

### docs/STRATEGY.md
**Purpose**: Strategic path to v1.0 — honest assessment of what's real vs. aspirational, what to work on, what to defer, and the post-v1.0 vision
**Key Sections**:
- The Honest Assessment (what's solid, aspirational, broken)
- The Strategic Choice (Python API-first vs DSL-first)
- What v1.0 Should Actually Be (5 concrete success criteria)
- Immediate Work (fix broken demos, PyPI, canonical examples)
- What NOT to Work On Before v1.0
- Post-v1.0 Vision (symbolic execution, MLIR, category-theoretic optimization, Philbrick)

**Navigate**: `reveal docs/STRATEGY.md`

---

### docs/PITCH.md
**Purpose**: 2-minute overview of what Morphogen is and why it matters — the entry point for new visitors
**Key Sections**:
- The Problem (integration tax)
- The Solution (cross-domain composition examples)
- What Makes It Different (determinism, category theory, multirate)
- The Name (Turing lineage)
- Current status and where to go deeper

**Navigate**: `reveal docs/PITCH.md`

---

### SPECIFICATION.md (2,282 lines, 54KB)
**Purpose**: Complete Morphogen language specification
**Key Sections**:
- Lexical structure (keywords, operators, literals)
- Type system (primitives, composites, physical units)
- Temporal programming model (`flow` blocks, `@state` declarations)
- Domain system (`use` statements, cross-domain composition)
- Operators across all domains
- Execution model & determinism

**Navigate**: `reveal SPECIFICATION.md`

---

### STATUS.md (1,096 lines, 45KB)
**Purpose**: Current implementation status by domain
**Key Sections**:
- Version history (v0.1 - v0.11.0)
- Domain implementation status (40+ domains)
- Test coverage statistics (900+ tests)
- Known issues & future work

**Navigate**: `reveal STATUS.md`

---

### docs/architecture/ECOSYSTEM_MAP.md (703 lines, 23KB)
**Purpose**: Map of all domains and roadmap
**Key Sections**:
- Complete domain catalog (organized by category)
- Implementation phases
- Cross-domain integration patterns
- Future expansion plans

**Navigate**: `reveal docs/architecture/ECOSYSTEM_MAP.md`

---

> **Note**: Cross-domain mesh documentation (CROSS_DOMAIN_MESH_CATALOG.md, CROSS_DOMAIN_API.md, MESH_TOPOLOGY_GUIDE.md, MESH_USER_GUIDE.md) has been archived to `~/Archive/morphogen/docs-archive/2025-12-06/mesh/`. For current cross-domain patterns, see [ADR-002: Cross-Domain Architectural Patterns](adr/002-cross-domain-architectural-patterns.md) and [Domain Mesh Catalog](reference/domain-mesh-catalog.md). See [ARCHIVE_HISTORY.md](ARCHIVE_HISTORY.md) for details.

---

### docs/CROSS_DOMAIN_MESH.png (383KB) + CROSS_DOMAIN_MESH.dot (2.3KB)
**Purpose**: Visual graph representation of cross-domain transform mesh
**Details**:
- NetworkX-based graph visualization showing domain connectivity
- Companion DOT file for GraphViz rendering

**Current cross-domain documentation**: See [ADR-002](adr/002-cross-domain-architectural-patterns.md) and [Domain Mesh Catalog](reference/domain-mesh-catalog.md)

---

### CHANGELOG.md (2,875 lines, 129KB)
**Purpose**: Detailed version history and release notes
**Key Sections**:
- Version-by-version changes (v0.1.0 through v0.11.0)
- Feature additions, bug fixes, breaking changes
- Migration guides between versions

**Navigate**: `reveal CHANGELOG.md` (large file - use preview)

---

### docs/ROADMAP.md (~350 lines) ⭐ **UNIFIED ROADMAP**
**Purpose**: Single source of truth for Morphogen development roadmap
**Key Sections**:
- Current Status (v0.12.0) - 39 domains, 606 operators, 1,912 tests, migration complete
- Remaining work - integration examples, v1.0 prep
- v1.0 - Production Release (Q2 2026) - three-track strategy
- Post-v1.0 Vision - v1.1, v1.5, v2.0 roadmap
- Implementation Tracking - all 39 domains, completion status

**Replaces**: Multiple scattered planning documents (consolidated 2025-12-07)
**See also**: [ARCHIVE_HISTORY.md](ARCHIVE_HISTORY.md) for archived planning documents

**Navigate**: `reveal docs/ROADMAP.md`

---

## Documentation Directories

### docs/README.md (207 lines, 14KB)
**Purpose**: Documentation navigation guide
**Structure**:
- Quick Start links
- Documentation by category (Philosophy, Architecture, Specifications, etc.)
- Finding what you need (task-based navigation)
- Recent changes

**Navigate**: `reveal docs/README.md`

---

## Architecture Documentation (7 documents)

### docs/architecture/overview.md
**Purpose**: System architecture overview
**Topics**: Kernel, frontends, Graph IR, MLIR compilation, execution model

### docs/architecture/domain-architecture.md (2,266 lines, 110KB)
**Purpose**: Deep technical vision for all domains
**Coverage**: 20+ domain specifications with operators, types, integration patterns

### docs/architecture/dsl-framework-design.md
**Purpose**: Vision for domain reasoning language
**Topics**: First-class domains, translations, composition patterns

### docs/architecture/continuous-discrete-semantics.md
**Purpose**: Dual computational models (continuous vs discrete)

### docs/architecture/gpu-mlir-principles.md
**Purpose**: GPU execution and MLIR integration strategy

### docs/architecture/interactive-visualization.md
**Purpose**: Visualization approach and capabilities

**Explore**: `reveal docs/architecture/ --files --ext md`

---

## Specifications (27 documents)

### Domain Specifications

| Domain | File | Lines | Focus |
|--------|------|-------|-------|
| Chemistry | chemistry.md | 1,002 | Molecular dynamics, kinetics, thermodynamics |
| Circuit | circuit.md | 1,136 | SPICE-like circuit simulation |
| Geometry | geometry.md | 3,000+ | CAD/parametric geometry |
| Physics | physics-domains.md | - | Fluid, thermal, combustion |
| Procedural Gen | procedural-generation.md | - | Noise, terrain, emergence |
| Audio | ambient-music.md | - | Compositional audio synthesis |
| Video/Audio | video-audio-encoding.md | - | Encoding and synchronization |
| Timbre | timbre-extraction.md | - | Instrument modeling |

### Infrastructure Specifications

| Specification | File | Purpose |
|--------------|------|---------|
| Graph IR | graph-ir.md | Frontend-kernel boundary |
| MLIR Dialects | mlir-dialects.md | Custom MLIR dialects (6 dialects) |
| Type System | type-system.md | Complete type system spec |
| Level 3 Types | level-3-type-system.md | Cross-domain type safety |
| Operator Registry | operator-registry.md | 500+ operator catalog |
| Scheduler | scheduler.md | Multirate deterministic scheduling |
| Transform | transform.md | Transform operators |
| Transform Comp | transform-composition.md | Composable transforms |
| Profiles | profiles.md | Execution profiles (strict/repro/live) |
| Units | units.md | Physical unit system |
| Snapshot ABI | snapshot-abi.md | State serialization |

**Explore**: `reveal docs/specifications/chemistry.md`

---

## Usage Guides (9 documents)

Narrative how-to guides for specific domains — start here if you want to
*use* a domain rather than understand its internals.

### Core Domain Tutorials

| Guide | Domain | Key Topics |
|-------|--------|-----------|
| `docs/usage/field.md` | `field` | Heat diffusion, reaction-diffusion, advection, gradient/laplacian, coupling to other domains |
| `docs/usage/rigidbody.md` | `rigidbody` | Bouncing balls, collision detection, forces/impulses, raycasting, collision→audio |
| `docs/usage/circuit.md` | `circuit` | RC filters, op-amp amplifiers, guitar pedal, transient/AC analysis, audio processing |
| `docs/usage/molecular.md` | `molecular` | Load/optimize molecules, MD (NVE/NVT), trajectory analysis, coupling to thermo |
| `docs/usage/audio_analysis.md` | `audio_analysis` | Pitch tracking, harmonic decay, T60, inharmonicity, feeding into `instrument_model` |
| `docs/usage/instrument_model.md` | `instrument_model` | Analyse recordings, synthesise notes, morph instruments, SynthParams |
| `docs/usage/fluid_jet.md` | `fluid_jet` | Single jets, entrainment, jet arrays, projecting to 2D field, coupling to acoustics |
| `docs/usage/controls.md` | `controls` | PID temperature control, LQR double integrator, Kalman 1D tracking, rigidbody coupling |

### Pipeline and Architecture Guides

| Guide | What It Covers |
|-------|---------------|
| `docs/usage/chemistry_pipeline.md` | End-to-end chemistry: molecular → thermo → kinetics (ethylene hydrogenation worked example) |
| `docs/usage/cross_domain_coupling.md` | How cross-domain interfaces work; writing and registering your own `DomainInterface` |

**Navigate**: `reveal docs/usage/` or `reveal docs/usage/field.md`

---

## Guides (5 documents)

### docs/guides/domain-implementation.md
**Purpose**: How to add new domains to Morphogen
**Topics**: Operator definition, registry integration, testing, documentation

### docs/guides/analysis-visualization.md
**Purpose**: Analyzing and visualizing Morphogen simulations with external tools
**Topics**: Built-in analysis (audio_analysis, signal, graph), PyDMD modal decomposition, showcase animations, spectral workflows

### docs/guides/EASY_TRANSFORMS_GUIDE.md
**Purpose**: Quick-win cross-domain transforms (high-impact, low-effort)

### docs/guides/optimization-implementation.md
**Purpose**: Implementing optimization algorithms

### docs/guides/output-generation.md
**Purpose**: Generating visualizations and outputs

*Archived (2026-03-16)*: `DOMAIN_FINISHING_GUIDE.md` (migration complete), `DOMAIN_MIGRATION_GUIDE.md` (migration complete)

**Explore**: `reveal docs/guides/domain-implementation.md`

---

## Examples (3 documents)

### docs/examples/
- **kerbal-space-program.md**: Orbital mechanics example
- **racing-ai-pipeline.md**: AI agent racing simulation
- **ambient-music-pipelines.md**: Generative music examples

*Archived (2026-03-16)*: `emergence-cross-domain.md` (design doc), `j-tube-firepit-multiphysics.md` (design doc)

See also: [`examples/canonical/`](../examples/canonical/) — 3 canonical cross-domain examples that run end-to-end

**Explore**: `reveal docs/examples/racing-ai-pipeline.md`

---

## Use Cases (2 documents)

### docs/use-cases/2-stroke-muffler-modeling.md
Complete example: Fluid dynamics → Acoustics → Audio synthesis

### docs/use-cases/chemistry-unified-framework.md
Chemistry simulation framework across multiple domains

---

## Reference Documentation (17 documents)

### Operator Catalogs
- **procedural-operators.md**: Procedural generation operators
- **emergence-operators.md**: Emergence domain operators
- **genetic-algorithm-operators.md**: GA operators
- **optimization-algorithms.md**: Optimization operators

### Visualization & Sonification
- **visualization-cookbook.md** (56KB): Comprehensive visualization techniques catalog
- **advanced-visualizations.md**: Advanced visualization techniques
- **sonification-cookbook.md** (40KB): Cross-domain sonification patterns
- **visual-domain-quickref.md**: Visual domain quick reference (2D animations)
- **visual3d-quickref.md** (NEW): 3D visualization API reference (PyVista-based)

### Theoretical Frameworks
- **universal-domain-frameworks.md** (28KB): Mathematical foundations
- **math-transformation-metaphors.md** (25KB): Intuitive transform explanations
- **mathematical-music-frameworks.md**: Music theory foundations

### Patterns & Best Practices
- **multiphysics-success-patterns.md** (30KB): 12 battle-tested patterns
- **time-alignment-operators.md**: Temporal synchronization

**Explore**: `reveal docs/reference/multiphysics-success-patterns.md`

---

## Architecture Decision Records (14 ADRs)

Located in `docs/adr/`:

| ADR | Title | Key Decision |
|-----|-------|--------------|
| 001 | Unified Reference Model | Cross-domain anchor system |
| 002 | Cross-Domain Patterns | Integration patterns |
| 003 | Circuit Modeling | SPICE-like domain design |
| 004 | Instrument Modeling | Physical modeling synthesis |
| 005 | Emergence Domain | Emergent behavior framework |
| 006 | Chemistry Domain | Molecular simulation approach |
| 007 | GPU-First Domains | GPU acceleration strategy |
| 008 | Procedural Generation | Noise/terrain domain design |
| 009 | Ambient Music & Generative | Generative music framework |
| 010 | Ecosystem Branding | Morphogen/Philbrick naming |
| 011 | Project Renaming | Kairo → Morphogen transition |
| 012 | Universal Domain Translation | Cross-domain translation patterns |
| 013 | Music Stack Consolidation | Audio domain unification |
| 014 | Complexity Refactoring Plan | 6-phase code quality improvement |

**Explore**: `reveal docs/adr/001-unified-reference-model.md`

---

## Philosophy Documentation (5 documents)

### docs/philosophy/
- **formalization-and-knowledge.md**: How formalization transforms knowledge
- **universal-dsl-principles.md**: Design philosophy (8 principles)
- **operator-foundations.md**: Mathematical operator theory
- **categorical-structure.md**: Category-theoretic formalization

**Explore**: `reveal docs/philosophy/formalization-and-knowledge.md`

---

## Roadmap & Planning

**⭐ Primary**: [docs/ROADMAP.md](ROADMAP.md) - Unified roadmap (v0.12.0 → v1.0)

*Note: Planning docs moved to `internal/planning/` (private, not in git) - 2026-01-22*

---

## Archive & History

### [ARCHIVE_HISTORY.md](ARCHIVE_HISTORY.md) ⭐ NEW
**Complete archive tracking document** - See this for details on all archived content.

**Key archived locations** (see ARCHIVE_HISTORY.md for full details):
- `~/Archive/morphogen/session-artifacts/` - Session summaries and status reports
- `~/Archive/morphogen/docs-archive/2025-12-06/` - Historical planning, mesh docs, summaries
- `~/Archive/morphogen/analysis-reports/` - Point-in-time technical analyses (v0.8.0 era)

**Why archived**: Point-in-time snapshots, superseded planning docs, outdated analyses preserved for historical reference without cluttering active documentation.

**Note**: Session artifacts policy previously documented in `docs/meta/` has been consolidated into ARCHIVE_HISTORY.md. All historical session documents are in `~/Archive/morphogen/session-artifacts/`.

---

## Claude Context

### claude.md (258 lines, 9KB)
**Purpose**: Context file for Claude AI assistant
**Contents**:
- Project overview and capabilities
- Key file locations
- reveal tool usage guide
- Development workflow
- Common tasks and examples

**Navigate**: `reveal claude.md`

---

## Scripts & Tools

### reveal (CLI tool)
**Purpose**: Progressive file explorer (wrapper)
**Usage**: `reveal <file>` or `reveal <dir>/` or `reveal <file> <func>`

### scripts/reveal.py
**Purpose**: Local Python implementation of reveal tool
**Features**: Metadata, structure extraction, preview, full content with line numbers

### scripts/gh.py
**Purpose**: GitHub issue/PR management without gh CLI

**Documentation**: See `scripts/README.md`

---

## Using the Reveal Tool

### Quick Examples

```bash
# Explore README structure
reveal README.md

# See directory layout
reveal docs/specifications/

# Inspect a source file (functions, classes)
reveal morphogen/stdlib/audio_analysis.py

# Extract a single function
reveal morphogen/stdlib/audio_analysis.py measure_t60

# List all markdown docs by recency
reveal docs/ --files --ext md
```

### Exploring Documentation Systematically

```bash
# See structure of all specifications
for f in docs/specifications/*.md; do
    echo "=== $f ==="
    reveal "$f"
    echo ""
done

# List all guides
reveal docs/guides/ --files --ext md

# Triage a large file before reading
reveal CHANGELOG.md
```

---

## Documentation Statistics

**Root Documentation**: 5 major files
**Architecture**: 6 documents (domain-architecture.md archived 2026-03-16)
**Specifications**: 24 documents (3 research specs marked with 🔬 badges)
**Usage Guides**: 9 documents (field, rigidbody, circuit, molecular, audio_analysis, instrument_model, fluid_jet, chemistry_pipeline, cross_domain_coupling)
**Guides**: 5 documents (3 archived 2026-03-16)
**Examples**: 3 documents (2 design docs archived 2026-03-16)
**Reference**: 17 documents
**ADRs**: 15 decision records
**Philosophy**: 5 documents

**Total**: ~107 documentation files (+7 usage guides added 2026-03-16, kufigi-0316)

---

## Navigation Tips

1. **Start with README.md** for project overview
2. **Read docs/README.md** for documentation navigation
3. **Use reveal tool** to explore incrementally:
   - `reveal <file>` — Structure first (headings, functions, classes)
   - Only `Read` the full file when you need the complete implementation
4. **Check SPECIFICATION.md** for language reference
5. **Browse docs/specifications/** for domain details
6. **Read ADRs** to understand key decisions
7. **Use DOCUMENTATION_INDEX.md** (this file) as your map

**Efficient exploration workflow:**
```bash
# Survey a directory (structure of all files)
for f in docs/specifications/*.md; do
    reveal "$f" | head -40
done

# Preview interesting files
reveal docs/guides/domain-implementation.md

# Read full content only when needed
reveal SPECIFICATION.md
```

---

**Generated**: 2025-11-21
**Updated**: 2026-03-16
**Tool**: `scripts/reveal.py` and custom documentation mapping
**Maintenance**: Regenerate when documentation structure changes significantly

**Related Resources:**
- [docs/README.md](README.md) — Navigation guide with task-based finding
- [docs/ROADMAP.md](ROADMAP.md) — Unified development roadmap ⭐
- [scripts/README.md](../scripts/README.md) — Complete reveal tool documentation
- [claude.md](../claude.md) — Claude AI assistant context and quick reference
- [../README.md](../README.md) — Project overview and getting started

---

## 🧹 Documentation Consolidation

### 2026-01-22: Directory Consolidation (117 → 121 docs)

**Merged directories by document type:**
- **Merged**: `roadmap/` → `planning/` (language-features.md, testing-strategy.md moved)
- **Consolidated**: `music/` → split by type:
  - `music/MUSIC_IMPLEMENTATION_ROADMAP.md` → `planning/music-implementation-roadmap.md`
  - `music/MUSIC_DOCUMENTATION_INDEX.md` → `reference/music-documentation-index.md`
- **Updated**: All cross-references to point to new locations
- **Left**: Redirect READMEs in `roadmap/` and `music/` for backwards compatibility

**Renamed for clarity:**
- `visualization-ideas-by-domain.md` → `visualization-cookbook.md`
- `audio-visualization-ideas.md` → `sonification-cookbook.md`

**Result**: Cleaner organization by document type rather than domain. All roadmaps in `planning/`, all reference docs in `reference/`.

### 2026-01-05: Research Spec Cleanup (117 docs)
**Final document count: 117** (from original 156, -25% reduction)

**Session 3 (crimson-twilight-0105):**
- **Archived**: `kairo-2.0-language-spec.md` (future research → `~/Archive/morphogen/docs-archive/2026-01-05/future-specs/`)
- **Archived**: `operator-registry-expansion.md` (duplicate → `~/Archive/morphogen/docs-archive/2026-01-05/reference/`)
- **Archived**: `KAIRO_2.0_STRATEGIC_ANALYSIS.md` (post-v1.0 planning → `~/Archive/morphogen/docs-archive/2026-01-05/planning/`)
- **Added**: Implementation status badges (🔬) to research specs (kax-language, bi-domain, emergence)
- **Updated**: ROADMAP.md with 3D visualization as critical blocker
- **Fixed**: Broken cross-references to archived directories

### 2026-01-05: Diligent Consolidation (120 docs)
**Document count: 120** (from original 156, -23% reduction)

**Phase 1 - Archive Consolidation:**
- **Archived**: Session artifacts → `~/Archive/morphogen/session-artifacts/`
- **Archived**: Historical planning docs → `~/Archive/morphogen/docs-archive/2025-12-06/planning/`
- **Archived**: Mesh documentation → `~/Archive/morphogen/docs-archive/2025-12-06/mesh/`
- **Archived**: Analysis reports → `~/Archive/morphogen/analysis-reports/`
- **Removed**: Empty `docs/archive/` and `docs/analysis/` directories

**Phase 2 - Doc Quality:**
- **Removed**: 2 broken stub documents (cross-domain-integration, mesh-operations)
- **Archived**: `units-summary.md` (implementation notes → session artifacts)
- **Organized**: Music docs into `docs/music/` subdirectory (3 docs + README)
- **Kept**: Planning/concept docs as active design documents
- **Kept**: Analog-platform docs (active hardware project)

**Result**: Cleaner structure, 25% total reduction, research vs production specs clearly marked

### 2025-12-07: Roadmap & Planning Consolidation
- **Created**: Unified `docs/ROADMAP.md` (single source of truth)
- **Archived**: 9 planning docs (now in ~/Archive)
- **Removed**: Outdated docs (mvp.md, FRESH_GIT_STRATEGY.md)
- **Organized**: Session artifacts → `~/Archive/morphogen/session-artifacts/`
- **Reorganized**: Moved guides and analysis to proper directories
- **Result**: Cleaner structure, single canonical roadmap

### 2025-12-06: Test Suite & Cross-Domain Cleanup
**See**: `~/Archive/morphogen/session-artifacts/2025-12-06/` for session summaries
- ✅ **0 test failures** (was 59)
- ✅ **1,381 passing** tests
- **Consolidated**: Cross-domain and mesh documentation
- **Archived**: Historical session artifacts

