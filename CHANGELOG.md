# Changelog

All notable changes to Morphogen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Documentation

#### ADR-015: First-Class Emergence Primitives
- New architectural decision record proposing promotion of emergence concepts to first-class primitives
- Covers: attractors, parameters, noise, time-scales, commitment events, constraints, cycles, coarse-graining
- Status: Proposed (see `docs/adr/015-first-class-emergence-primitives.md`)

#### Documentation Consolidation (2026-01-22)
- **Merged `roadmap/` → `planning/`**: All planning documents now in single directory
  - `language-features.md`, `testing-strategy.md` moved to `planning/`
- **Consolidated `music/`**: Split by document type
  - `MUSIC_IMPLEMENTATION_ROADMAP.md` → `planning/music-implementation-roadmap.md`
  - `MUSIC_DOCUMENTATION_INDEX.md` → `reference/music-documentation-index.md`
- **Renamed for clarity**:
  - `visualization-ideas-by-domain.md` → `visualization-cookbook.md`
  - `audio-visualization-ideas.md` → `sonification-cookbook.md`
- **Fixed**: `.gitignore` now allows `docs/planning/` (was blocking new files)
- Result: Cleaner organization by document type, all cross-references updated

---

## [0.11.0] - 2025-11-20

### 🎉 MAJOR: Project Renamed to Morphogen

**Breaking Change:** The project has been renamed from "Morphogen" to "Morphogen" to better reflect its essence as an emergence-focused continuous-time computation platform.

**Why "Morphogen"?**
- Named after Alan Turing's morphogenesis (1952): simple continuous-time differential equations creating complex emergent patterns
- Aligns perfectly with our architecture: four simple primitives (sum, integrate, nonlinearity, events) composing into emergent complexity
- Unique market positioning: "emergence-focused continuous-time computation"
- Educational value: teaches Turing's morphogenesis while teaching the platform
- See ADR-011 for full rationale

**What Changed:**
- Package name: `morphogen` → `morphogen`
- CLI command: `morphogen` → `morphogen`
- All imports: `from morphogen.*` → `from morphogen.*`
- MLIR dialects: `morphogen.*` → `morphogen.*`
- Class names: `MorphogenMLIRContext` → `MorphogenMLIRContext`
- All documentation updated

**Migration Guide:**
```python
# Before (v0.10.x)
from morphogen.stdlib import field, audio
import morphogen

# After (v0.11.0+)
from morphogen.stdlib import field, audio
import morphogen
```

**Sister Project:**
- Analog hardware platform "Philbrick" established (named after George A. Philbrick, inventor of modular analog computing)
- Morphogen (digital) + Philbrick (analog) = unified computational vision

---

## [2.0.0] - 2025-12-01

### 🔒 BREAKING: License Change to Apache 2.0

**License Migration:** MIT → Apache 2.0

**What Changed:**
- License upgraded from MIT to Apache 2.0
- Copyright updated to "Semantic Infrastructure Lab Contributors"
- Added CITATION.cff for academic citation
- Package metadata (pyproject.toml) updated
- README badge and license section updated

**Why Apache 2.0?**
- ✅ Patent protection for contributors and users
- ✅ Aligns Morphogen with SIL ecosystem standard
- ✅ Better fit for research/production hybrid
- ✅ Clear contributor license terms

**Compatibility:**
- Apache 2.0 is fully compatible with MIT
- All usage rights preserved (commercial, modification, distribution)
- Only adds patent protection grant

**SIL Ecosystem Integration:**
- Morphogen is now officially part of the Semantic Infrastructure Lab
- Unified licensing with Pantheon, SUP, BrowserBridge, GenesisGraph
- Visit https://github.com/scottsen/SIL for the ecosystem

---

## [0.12.0] - 2025-12-12

### 🎉 MAJOR: Domain Migration Complete

**All Legacy Domains Migrated**: The 3-phase migration plan is complete. All 39 computational domains are now fully integrated into the modern operator registry system.

**Achievement:**
- ✅ **39 production domains** (up from 25 in v0.11.0)
- ✅ **606 operators** accessible via `use` statement (up from 386)
- ✅ **1,705 comprehensive tests** (1,454 passing, 251 MLIR skipped)
- ✅ All legacy domains migrated to modern `@operator` decorator pattern
- ✅ Complete `use` statement coverage across all domains

### Migration Summary

**Phase 1: High-Value Chemistry ✅**
- Migrated: molecular, thermal_ode, fluid_network
- Added 40+ operators to registry
- Full test coverage created

**Phase 2: Chemistry Suite ✅**
- Migrated: qchem, thermo, kinetics, electrochem, catalysis, transport
- Added 70+ operators to registry
- Unified chemistry domain stack operational
- 9 chemistry domains now production-ready

**Phase 3: Specialized Physics ✅**
- Migrated: multiphase, combustion, fluid_jet, audio_analysis, instrument_model
- Added 40+ operators to registry
- Created 25 new tests (17 comprehensive for fluid_jet, 8 smoke tests for audio domains)
- All specialized physics domains now accessible

### New Domains Available (v0.12.0)

**Chemistry (9 domains)**:
- `molecular` (33 ops) - Molecular dynamics and force fields
- `qchem` (13 ops) - Quantum chemistry calculations
- `thermo` (12 ops) - Thermodynamic properties
- `kinetics` (11 ops) - Chemical reaction kinetics
- `electrochem` (13 ops) - Electrochemical systems
- `catalysis` (11 ops) - Catalysis modeling
- `transport` (17 ops) - Transport properties
- `thermal_ode` (4 ops) - Thermal dynamics
- `combustion` (8 ops) - Combustion modeling

**Fluid Dynamics (2 domains)**:
- `fluid_network` (4 ops) - Network flow modeling
- `fluid_jet` (7 ops) - Jet dynamics and entrainment
- `multiphase` (8 ops) - Multiphase flow systems

**Audio & Synthesis (2 domains)**:
- `audio_analysis` (9 ops) - Spectral analysis and deconvolution
- `instrument_model` (12 ops) - Physical modeling synthesis

### Testing

**New test files**:
- `tests/test_fluid_jet.py` - 17 comprehensive tests covering all 7 operators
- `tests/test_audio_analysis.py` - 5 smoke tests for domain validation
- `tests/test_instrument_model.py` - 3 smoke tests for domain validation

**Test Results**:
- Total: 1,705 tests (1,454 passed, 251 skipped)
- All 39 domains have test coverage
- 100% pass rate on active tests

### Documentation

**Updated**:
- `README.md` - Accurate domain counts (39 domains, 606 operators, 1,705 tests)
- `docs/ROADMAP.md` - v0.12.0 marked complete, priorities updated
- Removed legacy migration warnings

### Breaking Changes

None. All changes are additive - new domains and operators are now accessible.

### Migration Guide

All 39 domains are now accessible via the `use` statement:

```python
from morphogen.runtime import Runtime

rt = Runtime()

# Chemistry domains
rt.execute("use molecular")
rt.execute("use qchem")
rt.execute("use thermo")

# Fluid dynamics
rt.execute("use fluid_jet")
rt.execute("use fluid_network")

# Audio synthesis
rt.execute("use audio_analysis")
rt.execute("use instrument_model")
```

### Known Gaps

**Integration Examples**:
- Phase 3 domains (fluid_jet, audio_analysis, instrument_model) need usage examples
- Cross-domain chemistry examples needed
- Tutorial content for specialized physics

**Test Coverage**:
- audio_analysis: Smoke tests only (functional tests needed)
- instrument_model: Smoke tests only (functional tests needed)
- Target: 30+ tests per domain based on real use cases

### Next Steps → v1.0

With domain migration complete, focus shifts to:
1. Integration examples for all domains
2. Expanded test coverage (functional tests)
3. Performance benchmarking
4. Documentation polish
5. Community preparation

---

## [Unreleased]

### 🐛 Bug Fixes - 2025-11-23

#### Audio Domain - Filter State Management Fix
**Commit:** `8ab2496` - fix: Export constant operator for registry discovery

**Problem:** The constant operator was implemented but missing module-level export in `audio.py`, causing registry discovery to fail. This resulted in OperatorExecutor returning zeros for the constant operator, breaking filter state tests.

**Solution:** Added single line: `constant = AudioOperations.constant`

**Impact:**
- ✅ Operator count: 59 → 60 (constant operator now discoverable)
- ✅ Filter state test: PASSING (< 1e-6 error, was 0.984786 error)
- ✅ All tests passing: 4 GraphIR state + 4 constant operator = 8/8 ✅
- ✅ SimplifiedScheduler filter_state support confirmed working (no implementation needed!)

**Key Discovery:** SimplifiedScheduler already had complete filter_state support through OperatorExecutor delegation. What appeared to be a missing feature (3-4 hour implementation) was actually a one-line bug. Systematic debugging with reveal and instrumentation found the root cause in 40 minutes.

**Files Modified:**
- `morphogen/stdlib/audio.py` (+1 line)
- `docs/specifications/audio-synthesis.md` (documentation update)
- `tests/test_audio_basic.py` (constant operator tests)
- `tests/test_graphir_state_management.py` (filter state tests)

---

### 🚀 Morphogen v1.0 Release Plan - 2025-11-21

**Aggressive 24-week execution strategy to Morphogen v1.0 (2026-Q2)**

**Added:**
- **Planning Documents:**
  - [Morphogen v1.0 Release Plan](docs/planning/MORPHOGEN_RELEASE_PLAN.md) ⭐ **THE PLAN** — Unified 24-week execution strategy
  - [Morphogen Selective Evolution](docs/planning/MORPHOGEN_SELECTIVE_EVOLUTION.md) — Language evolution strategy (Path 2.5)
  - Updated [Planning README](docs/planning/README.md) — Clear navigation to active plans

**Three-Track Strategy:**
1. **Track 1 - Language Evolution** (13 weeks): Symbolic execution, transform tracking, category theory optimization, plugin system
2. **Track 2 - Critical Domains** (12 weeks): Circuit domain, Fluid dynamics, Chemistry Phase 2, 50+ domains total
3. **Track 3 - Adoption & Polish** (ongoing): PyPI release, showcase examples, tutorials, community

**Key Milestones:**
- Week 4: PyPI alpha release (v0.12.0-alpha)
- Week 22: Beta release (v0.99.0)
- Week 24: **v1.0 LAUNCH** 🎉

**Unique v1.0 Features:**
- Symbolic + numeric execution (first platform to combine both)
- Circuit → Audio coupling (design circuits, hear sound instantly)
- Category theory optimization (verified composition, automatic fusion)
- User extensibility (plugin system for custom domains)
- 50+ integrated domains

**Documentation Updates:**
- Updated root [README.md](README.md) with v1.0 roadmap section
- Updated [STATUS.md](STATUS.md) with v1.0 timeline
- Updated [docs/README.md](docs/README.md) with release plan callout
- Updated [docs/roadmap/README.md](docs/roadmap/README.md) to point to unified plan
- Marked old roadmaps as superseded

**Why This Matters:**
- Clear, unified execution plan (no conflicting docs)
- Aggressive but achievable timeline
- Continuous delivery every 2-4 weeks
- All tracks run in parallel for maximum velocity

---

### 📚 Philosophy & Universal DSL Documentation - 2025-11-21

**Major Documentation Expansion**: Created comprehensive theoretical foundations for universal cross-domain DSLs.

**Added:**
- **Philosophy Section** (`docs/philosophy/`)
  - [Formalization and Knowledge](docs/philosophy/formalization-and-knowledge.md) — Historical pattern of formalization from intuition to symbols (Probability, Logic, Geometry, Evolution) and Morphogen's role
  - [Universal DSL Principles](docs/philosophy/universal-dsl-principles.md) ⭐ — Eight core design principles for cross-domain DSLs (continuous/discrete, hybrid systems, transform spaces, translation semantics)
  - [Operator Foundations](docs/philosophy/operator-foundations.md) — Mathematical operator theory and spectral methods (moved from root)
  - [Categorical Structure](docs/philosophy/categorical-structure.md) — Category-theoretic formalization (moved from architecture)

- **Architecture Specifications**
  - [Continuous-Discrete Semantics](docs/architecture/continuous-discrete-semantics.md) — Dual computational models, hybrid systems, multi-rate scheduling

- **Language Specifications**
  - [Transform Composition](docs/specifications/transform-composition.md) — Composable named transforms with automatic inversion

- **ADRs**
  - [ADR 012: Universal Domain Translation](docs/adr/012-universal-domain-translation.md) — Framework for explicit domain translation with invariant preservation

**Changed:**
- Reorganized philosophical documentation into dedicated `docs/philosophy/` directory
- Updated all documentation indices and cross-references
- Enhanced philosophical docs with reading paths for different audiences

**Why This Matters:**
- Establishes Morphogen as continuation of formalization tradition (Euclid, Boole, Turing)
- Defines eight principles guiding universal DSL design
- Formalizes cross-domain translation with explicit invariant semantics
- Enables composable, named transform pipelines

---

### 🎯 Geometry Domain Implementation - Comprehensive 2D/3D Spatial Operations

**Date:** 2025-11-17
**Status:** ✅ COMPLETE
**Achievement:** New `geometry` domain added with 50+ operators for spatial computation!

This major addition brings comprehensive geometric primitives, transformations, and spatial queries to Morphogen, enabling powerful 2D/3D spatial reasoning for simulations, graphics, and computational geometry tasks.

**What Was Implemented:**

1. **New Geometry Domain** (`morphogen/stdlib/geometry.py`)
   - 50+ operators across 5 layers of functionality
   - Full type annotations and comprehensive documentation
   - Frame-aware transformations and coordinate conversions

2. **Layer 1: Primitive Construction (8 operators)**
   - Points (2D/3D), lines, circles, rectangles, polygons
   - Regular polygons (pentagon, hexagon, etc.)
   - All primitives support property queries (area, perimeter, centroid)

3. **Layer 2: Transformations (13 operators)**
   - Translation (moving shapes in 2D space)
   - Rotation (counter-clockwise around arbitrary centers)
   - Scaling (uniform and non-uniform)
   - Chainable transformations for complex operations

4. **Layer 3: Spatial Queries (10 operators)**
   - Distance calculations (point-point, point-line, point-circle)
   - Circle-circle intersection
   - Containment tests (point-in-circle, point-in-rectangle, point-in-polygon)
   - Closest point queries (on circles, lines)

5. **Layer 4: Coordinate Conversions (4 operators)**
   - Cartesian ↔ Polar (2D)
   - Cartesian ↔ Spherical (3D)
   - Roundtrip conversions with numerical precision

6. **Layer 5: Geometric Properties (4 operators)**
   - Area calculation (circles, rectangles, polygons)
   - Perimeter/circumference
   - Centroid computation
   - Axis-aligned bounding boxes

**Test Coverage:**
- 90 comprehensive tests covering all operators
- Edge cases: degenerate shapes, boundary conditions, numerical precision
- Integration tests: transformation chains, coordinate roundtrips
- 100% pass rate with full validation

**Examples:**
- `examples/geometry/01_basic_shapes.py` - Creating and querying primitives
- `examples/geometry/02_transformations.py` - Translation, rotation, scaling
- `examples/geometry/03_spatial_queries.py` - Distance, intersection, containment
- `examples/geometry/04_coordinate_systems.py` - Polar and spherical conversions

**Technical Highlights:**
- NumPy-based implementation for performance
- Type-safe with proper Python type hints
- Deterministic operations (all operators marked `deterministic=True`)
- Follows established Morphogen patterns (decorator, categories, signatures)

**Use Cases:**
- Physics simulations (collision detection, spatial partitioning)
- Procedural generation (terrain, dungeons, fractals)
- Graphics and visualization (shape rendering, transformations)
- Robotics (path planning, obstacle avoidance)
- Computational geometry (Voronoi, Delaunay, convex hulls)

**Next Steps:**
- Integration with `field` domain for spatial field queries
- Integration with `rigidbody` domain for advanced collision shapes
- 3D geometric primitives (boxes, spheres, meshes)
- Advanced operations (convex hull, Voronoi diagrams, mesh boolean ops)

---

### 🎉 Level 2 Integration COMPLETE - All 23 Domains Registered!

**Date:** 2025-11-17
**Status:** ✅ COMPLETE
**Achievement:** Level 2 integration jumped from 39.1% → 95.7% → **100% complete!**

This represents a **massive milestone**: all 23 stdlib domains are now fully integrated with the operator registry and ready for use from `.morphogen` source files.

**What Was Completed:**

1. **Created Temporal Domain (24 operators)**
   - New `morphogen/stdlib/temporal.py` with temporal logic and scheduling
   - Delay lines, timers, clocks, event sequences
   - Edge detection, threshold crossings, temporal logic operators
   - Lag operations, time series analysis (difference, cumsum)
   - Rate conversion and resampling

2. **Fixed Import Errors (13 domains)**
   - Corrected `from morphogen.core.operators` → `from morphogen.core.operator` (singular)
   - Added missing `OpCategory` imports to 3 domains

3. **Completed @operator Decorators (374 total operators)**
   - Added full decorator parameters to 273 incomplete operators
   - Fixed categories: CONSTRUCT, TRANSFORM, QUERY, INTEGRATE
   - Generated type signatures for all operators
   - Added documentation strings from docstrings

4. **Fixed Operator Discovery (15 domains)**
   - Added module-level exports for class-based operators
   - Pattern: `operator_name = ClassName.operator_name`
   - Enables domain registry discovery via introspection

**Final Statistics:**

```
✓ 23/23 domains registered and discoverable
✓ 374 total operators across all domains
✓ All operators callable from .morph source files
✓ Full metadata: domain, category, signature, determinism, docs
```

**Domain Breakdown:**

- Acoustics: 9 operators | Agents: 13 operators | Audio: 54 operators
- Cellular: 18 operators | Color: 20 operators | Field: 19 operators
- Genetic: 17 operators | Graph: 19 operators | Image: 18 operators
- Integrators: 9 operators | I/O & Storage: 10 operators | Neural: 16 operators
- Noise: 11 operators | Optimization: 5 operators | Palette: 21 operators
- RigidBody: 12 operators | Signal: 20 operators | Sparse LinAlg: 13 operators
- StateMachine: 15 operators | **Temporal: 24 operators** | Terrain: 11 operators
- Vision: 13 operators | Visual: 7 operators

**Impact:**

- ✅ All 23 domains accessible via `use {domain}` statement
- ✅ Operators discoverable via domain registry
- ✅ Ready for Level 3: Type system enforcement with physical units
- ✅ Foundation for multirate scheduler integration (Level 4)

**Next Steps:**

- Level 3: Type system with physical units (Months 5-6)
- Level 4: Multirate scheduler integration (Months 7-8)
- Level 5: MLIR native compilation for performance-critical domains

**Related:** Domain Finishing Initiative, DOMAIN_FINISHING_GUIDE.md

---

### Strategic - Domain Finishing Initiative

**Date:** 2025-11-17
**Status:** Active Roadmap (Level 2 Complete!)
**Timeline:** 10 months (Months 1-10)

We are launching a strategic pivot from breadth (adding more domains) to depth (finishing the 23 domains we have). This initiative is documented in `docs/guides/DOMAIN_FINISHING_GUIDE.md`.

**The Problem:**
We have 23 domains implemented, but most are "half-finished" — they work in Python but aren't fully integrated into Morphogen/Morphogen as a platform.

**The Solution:**
A comprehensive 10-month roadmap to bring all 23 domains through 5 levels of completion:

1. **Level 1: Python Runtime** ✅ COMPLETE (23/23 domains)
   - All domains work as Python libraries
   - Operators implemented, tested, with examples

2. **Level 2: Language Integration** (Months 3-4, 8 weeks)
   - Domain registry system
   - `use {domain}` statement support
   - Operators callable from `.morphogen` source files
   - Type signatures for all operators

3. **Level 3: Type System Enforcement** (Months 5-6, 8 weeks)
   - Physical units enforced at compile time
   - Cross-domain type validation
   - Rate compatibility checking
   - Clear error messages

4. **Level 4: Scheduler Integration** (Months 7-8, 8 weeks)
   - Multirate execution (audio @ 48kHz, control @ 1kHz, visual @ 60Hz)
   - Sample-accurate timing
   - Cross-rate communication and resampling
   - Deterministic scheduling

5. **Level 5: MLIR Native Compilation** (Ongoing, prioritized)
   - MLIR dialects for performance-critical domains
   - 5-10x performance improvements
   - JIT/AOT compilation support

**Current Status:**
- ✅ Level 1: 23/23 domains (COMPLETE)
- ✅ Level 2: 23/23 domains (COMPLETE - 100%!) 🎉
  - ✅ Domain registry system implemented
  - ✅ @operator decorator system complete
  - ✅ All 374 operators fully decorated and discoverable
  - ✅ Module-level exports for operator discovery
  - ✅ Acoustics: 9 ops | Agents: 13 ops | Audio: 54 ops
  - ✅ Cellular: 18 ops | Color: 20 ops | Field: 19 ops
  - ✅ Genetic: 17 ops | Graph: 19 ops | Image: 18 ops
  - ✅ Integrators: 9 ops | I/O & Storage: 10 ops | Neural: 16 ops
  - ✅ Noise: 11 ops | Optimization: 5 ops | Palette: 21 ops
  - ✅ RigidBody: 12 ops | Signal: 20 ops | Sparse LinAlg: 13 ops
  - ✅ StateMachine: 15 ops | **Temporal: 24 ops** | Terrain: 11 ops
  - ✅ Vision: 13 ops | Visual: 7 ops
- ⏭️ Level 3: 0/23 domains (NEXT - type system with physical units)
- ⏭️ Level 4: 0/23 domains (multirate scheduler integration)
- ⚠️ Level 5: 4/23 domains (field, agent, audio, temporal have MLIR support)

**Phase 1: Showcase & Validation** (Months 1-2)
- Generate professional outputs from existing domains
- Gather community feedback
- Validate use cases before infrastructure investment

**Phase 2: Core Infrastructure** (Months 3-8)
- Implement domain registry and language integration
- Build type system with units, domains, and rates
- Create multirate scheduler with cross-domain support

**Phase 3: Production Readiness** (Months 9-10)
- MLIR integration for high-priority domains
- Build 3 real-world applications
- Production deployment

**Key Insight:**
> "Stop adding domains. Finish the 23 we have. Then every future domain integrates from day 1."

**Success Metrics:**
- By Month 4: All 23 domains usable from `.morphogen` files
- By Month 6: Full type safety across all domains
- By Month 8: Multirate execution working
- By Month 10: 3 production applications + 5+ domains with native compilation

**Related Documentation:**
- `docs/guides/DOMAIN_FINISHING_GUIDE.md` - Complete roadmap with implementation details
- `docs/planning/EXECUTION_PLAN_Q4_2025.md` - Master timeline
- `docs/guides/domain-implementation.md` - MLIR integration guide
- `docs/CROSS_DOMAIN_API.md` - Cross-domain patterns

---

### Added - Level 2: Domain Registry Infrastructure

**Implementation of Level 2, Phase 1 from Domain Finishing Guide**

We've begun Level 2 (Language Integration) by implementing the core domain registry infrastructure. This is the foundation that will enable all 23 domains to be accessible from `.morphogen` source files.

**New Core Infrastructure:**

- **`morphogen/core/operator.py`** - Operator metadata system
  - `@operator` decorator for marking domain operators
  - `OpCategory` enum: CONSTRUCT, TRANSFORM, QUERY, INTEGRATE, COMPOSE, MUTATE, SAMPLE, RENDER
  - `OperatorMetadata` dataclass: domain, category, signature, determinism, documentation
  - Helper functions: `get_operator_metadata()`, `is_operator()`

- **`morphogen/core/domain_registry.py`** - Central domain registry
  - `DomainDescriptor`: Metadata container for domains
  - `DomainRegistry`: Singleton registry with auto-discovery of operators
  - `register_stdlib_domains()`: Auto-registers all 23 stdlib domains
  - Operator discovery via `@operator` decorator introspection

- **`tests/test_domain_registry.py`** - Comprehensive test suite
  - 19 tests covering decorator, descriptor, and registry functionality
  - All tests passing ✅

**Graph Domain - Fully Integrated (Proof of Concept):**

The graph domain is now the first fully-integrated domain, serving as the template for the remaining 22 domains.

- **19 operators** all decorated and discoverable:
  - **4 CONSTRUCT**: `create_empty`, `from_adjacency_matrix`, `grid_graph`, `random_graph`
  - **11 QUERY**: `bfs`, `dfs`, `dijkstra`, `shortest_path`, `connected_components`, `degree_centrality`, `betweenness_centrality`, `pagerank`, `clustering_coefficient`, `topological_sort`, `max_flow`
  - **4 TRANSFORM**: `add_edge`, `remove_edge`, `to_adjacency_matrix`, `minimum_spanning_tree`

- All operators have complete metadata:
  - Domain: "graph"
  - Category: Semantic grouping
  - Signature: Type information for future type checking
  - Deterministic: All graph operations are deterministic

**What This Enables:**

This infrastructure is the foundation for:
1. `use graph` statements in `.morphogen` files (parser enhancement needed)
2. Type checking of operator calls (type system needed)
3. Auto-generated documentation (tooling needed)
4. IDE autocompletion (LSP integration needed)

**Progress Tracking:**
- Week 1 of 8 for Level 2 (Language Integration)
- 2/23 domains fully integrated (8.7%)
- Infrastructure complete, ready for rapid domain integration

**Next Steps:**
- Add @operator decorators to remaining 21 domains
- Implement parser enhancement for `use` statement

---

### Added - Level 2: Signal Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the pattern established with the graph domain, the signal processing domain is now fully integrated into the domain registry system.

**Signal Domain - Fully Integrated:**

The signal domain is now the second fully-integrated domain, demonstrating the systematic approach for completing all 23 domains.

- **20 operators** all decorated and discoverable:
  - **4 CONSTRUCT**: `create_signal`, `sine_wave`, `chirp`, `white_noise`
  - **12 TRANSFORM**: `window`, `fft`, `ifft`, `rfft`, `stft`, `istft`, `lowpass`, `highpass`, `bandpass`, `resample`, `envelope`, `normalize`
  - **4 QUERY**: `correlate`, `peak_detection`, `spectrogram_power`, `welch_psd`

- All operators have complete metadata:
  - Domain: "signal"
  - Category: Semantic grouping (CONSTRUCT/TRANSFORM/QUERY)
  - Signature: Type information for future type checking
  - Deterministic: All signal operations are deterministic (except `white_noise` without seed)

**Module-Level Exports:**

All 20 operators are exported at module level in `morphogen/stdlib/signal.py` for automatic discovery by the domain registry system.

**Integration Progress:**
- ✅ Signal domain: 20/20 operators integrated (100%)
- ✅ 2/23 domains complete (8.7%)
- ⏭️ Next: Continue with remaining 21 domains

**What This Demonstrates:**

This integration validates the systematic approach:
1. Add `@operator` decorators to all domain operators
2. Categorize operators semantically (CONSTRUCT, TRANSFORM, QUERY)
3. Export operators at module level for discovery
4. Verify registration and operator metadata

**Related:**
- `morphogen/stdlib/signal.py` - Signal processing operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry
- `docs/guides/DOMAIN_FINISHING_GUIDE.md` - Complete roadmap
- Create operator syntax bindings
- Integration testing across all domains

---

### Added - Level 2: StateMachine Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach established with graph and signal domains, the statemachine domain is now fully integrated into the domain registry system.

**StateMachine Domain - Fully Integrated:**

The statemachine domain is now the third fully-integrated domain, continuing the methodical completion of all 23 domains.

- **15 operators** all decorated and discoverable:
  - **5 CONSTRUCT**: `create`, `create_sequence`, `create_selector`, `create_action`, `create_condition`
  - **6 TRANSFORM**: `add_state`, `add_transition`, `start`, `update`, `send_event`, `execute_behavior`
  - **4 QUERY**: `get_state_name`, `is_in_state`, `get_valid_transitions`, `to_graphviz`

- All operators have complete metadata:
  - Domain: "statemachine"
  - Category: Semantic grouping (CONSTRUCT/TRANSFORM/QUERY)
  - Signature: Type information for future type checking
  - Deterministic: FSM construction/queries are deterministic; operations with callbacks are non-deterministic

**Implementation Details:**

- Added `from morphogen.core.operator import operator, OpCategory` import
- Applied `@operator` decorator to all 15 methods in `StateMachineOperations` class
- Categorized operators by function:
  - CONSTRUCT: Creating state machines and behavior tree nodes
  - TRANSFORM: Modifying state machines, triggering transitions, executing behaviors
  - QUERY: Reading state machine information, visualization
- Marked callback-dependent operations (start, update, send_event, execute_behavior) as non-deterministic
- Exported all operators at module level for registry discovery

**Module-Level Exports:**

All 15 operators are exported at module level in `morphogen/stdlib/statemachine.py` for automatic discovery by the domain registry system:
- FSM operators: create, add_state, add_transition, start, update, send_event, get_state_name, is_in_state, get_valid_transitions, to_graphviz
- Behavior tree operators: create_sequence, create_selector, create_action, create_condition, execute_behavior

**Integration Progress:**
- ✅ StateMachine domain: 15/15 operators integrated (100%)
- ✅ 3/23 domains complete (13.0%)
- ⏭️ Next: Continue with remaining 20 domains

**What This Covers:**

The statemachine domain provides two complementary paradigms for behavior modeling:
1. **Finite State Machines (FSM)**: Event-driven state transitions with guards and actions
2. **Behavior Trees**: Hierarchical behavior composition (sequence, selector, action, condition nodes)

Both are essential for game AI, UI flows, protocol implementations, and workflow systems.

**Related:**
- `morphogen/stdlib/statemachine.py` - State machine and behavior tree operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry
- `docs/guides/DOMAIN_FINISHING_GUIDE.md` - Complete roadmap

---

### Added - Level 2: Terrain Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach established with graph, signal, and statemachine domains, the terrain domain is now fully integrated into the domain registry system.

**Terrain Domain - Fully Integrated:**

The terrain domain is now the fourth fully-integrated domain, continuing the methodical completion of all 23 domains.

- **11 operators** all decorated and discoverable:
  - **2 CONSTRUCT**: `create_heightmap`, `from_noise_perlin`
  - **6 TRANSFORM**: `hydraulic_erosion`, `thermal_erosion`, `terrace`, `smooth`, `normalize`, `island_mask`
  - **3 ANALYSIS**: `calculate_slope`, `calculate_aspect`, `classify_biomes`

- All operators have complete metadata:
  - Domain: "terrain"
  - Category: Semantic grouping (CONSTRUCT/TRANSFORM/ANALYSIS)
  - Signature: Type information for future type checking
  - Deterministic: Most terrain operations are deterministic; `from_noise_perlin` is non-deterministic without seed

**Implementation Details:**

- Added `from morphogen.core.operator import operator, OpCategory` import
- Applied `@operator` decorator to all 11 methods in `TerrainOperations` class
- Categorized operators by function:
  - CONSTRUCT: Creating and generating heightmaps
  - TRANSFORM: Modifying terrain (erosion, smoothing, normalization, effects)
  - ANALYSIS: Computing terrain properties (slope, aspect, biome classification)
- Marked `from_noise_perlin` as non-deterministic (uses random seed)
- Exported all operators at module level for registry discovery

**Module-Level Exports:**

All 11 operators are exported at module level in `morphogen/stdlib/terrain.py` for automatic discovery by the domain registry system:
- Creation: create_heightmap, from_noise_perlin
- Erosion: hydraulic_erosion, thermal_erosion
- Analysis: calculate_slope, calculate_aspect, classify_biomes
- Effects: terrace, smooth, normalize, island_mask

**Integration Progress:**
- ✅ Terrain domain: 11/11 operators integrated (100%)
- ✅ 4/23 domains complete (17.4%)
- ⏭️ Next: Continue with remaining 19 domains

**What This Covers:**

The terrain domain provides comprehensive procedural terrain generation and analysis:
1. **Heightmap Generation**: Perlin noise-based terrain creation
2. **Erosion Simulation**: Hydraulic (water) and thermal erosion for realistic terrain
3. **Terrain Analysis**: Slope, aspect, and biome classification
4. **Terrain Effects**: Terracing, smoothing, normalization, island masking

Essential for procedural world generation, game development, and geographic simulation.

**Related:**
- `morphogen/stdlib/terrain.py` - Terrain generation operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry
- `docs/guides/DOMAIN_FINISHING_GUIDE.md` - Complete roadmap

---

### Added - Level 2: Vision Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach established with graph, signal, statemachine, and terrain domains, the vision domain is now fully integrated into the domain registry system.

**Vision Domain - Fully Integrated:**

The vision domain is now the fifth fully-integrated domain, continuing the methodical completion of all 23 domains.

- **13 operators** all decorated and discoverable:
  - **1 CONSTRUCT**: `create_image`
  - **5 TRANSFORM**: `gaussian_blur`, `morphological`, `threshold`, `adaptive_threshold`
  - **8 ANALYSIS**: `sobel`, `laplacian`, `canny`, `harris_corners`, `find_contours`, `template_match`, `hough_lines`, `optical_flow_lucas_kanade`

- All operators have complete metadata:
  - Domain: "vision"
  - Category: Semantic grouping (CONSTRUCT/TRANSFORM/ANALYSIS)
  - Signature: Type information for future type checking
  - Deterministic: All vision operations are deterministic

**Implementation Details:**

- Added `from morphogen.core.operator import operator, OpCategory` import
- Applied `@operator` decorator to all 13 methods in `VisionOperations` class
- Categorized operators by function:
  - CONSTRUCT: Creating grayscale images from arrays
  - TRANSFORM: Image filtering and preprocessing (blur, morphology, thresholding)
  - ANALYSIS: Feature detection and computer vision algorithms (edge detection, corners, contours, template matching, optical flow)
- All operations marked as deterministic
- Exported all operators at module level for registry discovery

**Module-Level Exports:**

All 13 operators are exported at module level in `morphogen/stdlib/vision.py` for automatic discovery by the domain registry system:
- Construction: create_image
- Edge Detection: sobel, laplacian, canny
- Feature Detection: harris_corners, find_contours, hough_lines
- Template/Flow: template_match, optical_flow_lucas_kanade
- Transforms: gaussian_blur, morphological, threshold, adaptive_threshold

**Integration Progress:**
- ✅ Vision domain: 13/13 operators integrated (100%)
- ✅ 5/23 domains complete (21.7%)
- ⏭️ Next: Continue with remaining 18 domains

**What This Covers:**

The vision domain provides comprehensive computer vision and image analysis capabilities:
1. **Edge Detection**: Sobel, Laplacian, Canny algorithms for finding edges
2. **Feature Detection**: Harris corners, contour detection, Hough line detection
3. **Image Processing**: Gaussian blur, morphological operations (erode, dilate, open, close)
4. **Thresholding**: Binary and adaptive thresholding for segmentation
5. **Advanced Analysis**: Template matching, optical flow (Lucas-Kanade)

Essential for computer vision, image analysis, object detection, and visual processing applications.

**Related:**
- `morphogen/stdlib/vision.py` - Computer vision operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry
- `docs/guides/DOMAIN_FINISHING_GUIDE.md` - Complete roadmap

---

### Added - Level 2: Cellular Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach established with graph, signal, statemachine, terrain, and vision domains, the cellular automata domain is now fully integrated into the domain registry system.

**Cellular Domain - Fully Integrated:**

The cellular domain is now the sixth fully-integrated domain, continuing the methodical completion of all 23 domains.

- **18 operators** all decorated and discoverable:
  - **6 CONSTRUCT**: `alloc`, `random_init`, `count_neighbors_moore`, `count_neighbors_von_neumann`, `apply_rule`, `apply_wolfram_rule`
  - **4 TRANSFORM**: `step`, `evolve`, `history`, `analyze_pattern`
  - **8 CONSTRUCT (Classic Automata)**: `game_of_life`, `brians_brain`, `brians_brain_step`, `highlife`, `seeds`, `wolfram_ca`, `to_array`, `from_array`

- All operators have complete metadata:
  - Domain: "cellular"
  - Category: Semantic grouping (CONSTRUCT/TRANSFORM)
  - Signature: Type information for future type checking
  - Deterministic: Most cellular operations are deterministic (except `random_init` without seed)

**Implementation Details:**

- Added `from ..decorator import operator` import
- Applied `@operator` decorator to all 18 methods in `CellularOperations` class
- Categorized operators by layer:
  - Layer 1: Atomic CA operations (alloc, init, neighbor counting, rule application)
  - Layer 2: Composite CA operations (step, evolve, history, pattern analysis)
  - Layer 3: Classic automata constructs (Game of Life, Brian's Brain, HighLife, Seeds, Wolfram CA)
- All operations properly categorized
- Exported all operators at module level for registry discovery

**Module-Level Exports:**

All 18 operators are exported at module level in `morphogen/stdlib/cellular.py` for automatic discovery by the domain registry system:
- Field Creation: alloc, random_init, from_array
- Neighbor Counting: count_neighbors_moore, count_neighbors_von_neumann
- Rule Application: apply_rule, apply_wolfram_rule
- Evolution: step, evolve, history
- Analysis: analyze_pattern
- Classic Automata: game_of_life, brians_brain, brians_brain_step, highlife, seeds, wolfram_ca
- Conversion: to_array

**Integration Progress:**
- ✅ Cellular domain: 18/18 operators integrated (100%)
- ✅ 6/23 domains complete (26.1%)
- ⏭️ Next: Continue with remaining 17 domains

**What This Covers:**

The cellular domain provides comprehensive cellular automata capabilities:
1. **Classic CA**: Conway's Game of Life, Brian's Brain, HighLife, Seeds
2. **Wolfram Elementary CA**: 1D cellular automata with 256 possible rules
3. **Custom Rules**: Define birth/survival rules for custom automata
4. **Neighborhoods**: Moore (8 neighbors) and von Neumann (4 neighbors)
5. **Evolution & Analysis**: Step-by-step evolution, history tracking, pattern analysis

Essential for emergent pattern simulation, artificial life, procedural generation, and complexity research.

**Related:**
- `morphogen/stdlib/cellular.py` - Cellular automata operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry
- `docs/guides/DOMAIN_FINISHING_GUIDE.md` - Complete roadmap

---

### Added - Level 2: Optimization Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach, the optimization domain is now fully integrated into the domain registry system.

**Optimization Domain - Fully Integrated:**

The optimization domain is now the seventh fully-integrated domain, providing evolutionary algorithms and numerical optimization.

- **5 operators** all decorated and discoverable:
  - **OPTIMIZE**: `minimize`, `differential_evolution`, `cmaes`, `particle_swarm`, `nelder_mead`

- All operators have complete metadata:
  - Domain: "optimization"
  - Category: Optimization algorithms
  - Signature: Type information for future type checking
  - Deterministic: All operations are deterministic with seed parameter

**Implementation Details:**

- Added `from ..decorator import operator` import
- Applied `@operator` decorator to all 5 module-level convenience functions
- Supports multiple optimization algorithms:
  - Differential Evolution (DE): continuous parameter optimization
  - CMA-ES: covariance matrix adaptation
  - Particle Swarm Optimization (PSO): swarm intelligence
  - Nelder-Mead: gradient-free local optimization
  - Unified minimize interface with auto-selection

**Module-Level Exports:**

All 5 operators are module-level functions in `morphogen/stdlib/optimization.py`:
- minimize: unified optimization interface
- differential_evolution: DE optimizer
- cmaes: CMA-ES optimizer
- particle_swarm: PSO optimizer
- nelder_mead: simplex-based optimizer

**Integration Progress:**
- ✅ Optimization domain: 5/5 operators integrated (100%)
- ✅ 7/23 domains complete (30.4%)
- ⏭️ Next: Continue with remaining 16 domains

**What This Covers:**

The optimization domain provides comprehensive optimization capabilities:
1. **Evolutionary Algorithms**: Differential Evolution for robust global optimization
2. **Advanced Methods**: CMA-ES for high-dimensional problems
3. **Swarm Intelligence**: Particle Swarm Optimization
4. **Local Optimization**: Nelder-Mead simplex method
5. **Benchmark Functions**: Test functions for algorithm validation

Essential for parameter tuning, design discovery, multi-objective optimization, and automated system configuration.

**Related:**
- `morphogen/stdlib/optimization.py` - Optimization operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry

---

### Added - Level 2: Neural Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach, the neural network domain is now fully integrated into the domain registry system.

**Neural Domain - Fully Integrated:**

The neural domain is now the eighth fully-integrated domain, providing feedforward neural networks for agent control.

- **16 operators** all decorated and discoverable:
  - **7 TRANSFORM (Layer 1)**: `linear`, `tanh`, `relu`, `sigmoid`, `softmax`, `leaky_relu`, `apply_activation`
  - **2 COMPOSE (Layer 2)**: `dense`, `forward`
  - **6 CONSTRUCT (Layer 3)**: `alloc_layer`, `alloc_mlp`, `get_parameters`, `set_parameters`, `mutate_parameters`, `crossover_parameters`
  - **1 PRESET (Layer 4)**: `flappy_bird_controller`

- All operators have complete metadata:
  - Domain: "neural"
  - Category: Semantic grouping by layer
  - Signature: Type information for future type checking
  - Deterministic: All operations are deterministic (with seed where applicable)

**Implementation Details:**

- Added `from ..decorator import operator` import
- Applied `@operator` decorator to all 16 methods in `NeuralOperations` class
- Organized by 4-layer hierarchy:
  - Layer 1: Atomic operations (linear, activation functions)
  - Layer 2: Composite operations (dense layer, forward pass)
  - Layer 3: Network constructs (MLP creation, parameter manipulation)
  - Layer 4: Presets (domain-specific network architectures)

**Module-Level Exports:**

All 16 operators are exported via the `neural` singleton in `morphogen/stdlib/neural.py`:
- Activations: tanh, relu, sigmoid, softmax, leaky_relu
- Core Ops: linear, apply_activation, dense, forward
- Network Ops: alloc_layer, alloc_mlp
- Parameter Ops: get_parameters, set_parameters, mutate_parameters, crossover_parameters
- Presets: flappy_bird_controller

**Integration Progress:**
- ✅ Neural domain: 16/16 operators integrated (100%)
- ✅ 8/23 domains complete (34.8%)
- ⏭️ Next: Continue with remaining 15 domains

**What This Covers:**

The neural domain provides feedforward neural network capabilities:
1. **Activation Functions**: tanh, ReLU, sigmoid, softmax, leaky ReLU
2. **Layer Operations**: Dense layers with configurable activations
3. **Network Architectures**: Multi-layer perceptrons (MLPs)
4. **Genetic Algorithm Support**: Parameter extraction, mutation, crossover
5. **Batch Inference**: Efficient parallel evaluation

Essential for agent control, genetic algorithms, evolutionary robotics, and neuroevolution.

**Related:**
- `morphogen/stdlib/neural.py` - Neural network operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry

---

### Added - Level 2: Noise Domain Integration

**Date:** 2025-11-17
**Timeline:** Month 3-4, Week 1 of 8

Following the systematic approach, the noise generation domain is now fully integrated into the domain registry system.

**Noise Domain - Fully Integrated:**

The noise domain is now the ninth fully-integrated domain, providing procedural noise generation for textures and terrain.

- **11 operators** all decorated and discoverable:
  - **3 CONSTRUCT (Layer 2)**: `perlin2d`, `simplex2d`, `value2d`, `worley`
  - **4 COMPOSE (Layer 3)**: `fbm`, `ridged_fbm`, `turbulence`, `marble`
  - **3 ADVANCED (Layer 4)**: `vector_field`, `gradient_field`, `plasma`

- All operators have complete metadata:
  - Domain: "noise"
  - Category: Semantic grouping by complexity
  - Signature: Type information for future type checking
  - Deterministic: All operations are deterministic with seed parameter

**Implementation Details:**

- Added `from ..decorator import operator` import
- Applied `@operator` decorator to all 11 public methods in `NoiseOperations` class
- Organized by layer hierarchy:
  - Layer 2: Basic noise types (Perlin, Simplex, Value, Worley)
  - Layer 3: Fractal patterns (FBM, ridged, turbulence, marble)
  - Layer 4: Vector fields and advanced patterns

**Module-Level Exports:**

All 11 operators are exported via the `noise` singleton in `morphogen/stdlib/noise.py`:
- Basic Noise: perlin2d, simplex2d, value2d, worley
- Fractal Noise: fbm, ridged_fbm, turbulence, marble
- Advanced: vector_field, gradient_field, plasma

**Integration Progress:**
- ✅ Noise domain: 11/11 operators integrated (100%)
- ✅ 9/23 domains complete (39.1%)
- ⏭️ Next: Continue with remaining 14 domains

**What This Covers:**

The noise domain provides comprehensive procedural noise generation:
1. **Classic Noise**: Perlin, Simplex, Value noise algorithms
2. **Cellular Patterns**: Worley/Voronoi noise
3. **Fractal Composition**: Fractal Brownian Motion (FBM)
4. **Artistic Effects**: Marble, turbulence, ridged patterns
5. **Vector Fields**: Flow fields and gradients for particle systems

Essential for procedural textures, terrain generation, particle effects, and visual randomization.

**Related:**
- `morphogen/stdlib/noise.py` - Noise generation operators with @operator decorators
- `morphogen/core/operator.py` - @operator decorator system
- `morphogen/core/domain_registry.py` - Central domain registry

---

## [v0.10.0] - 2025-11-16

### Added - Five New Computational Domains ⭐⭐⭐

This release significantly expands Morphogen's domain coverage with five major new domains, bringing the total to 23 domains.

- **Graph/Network Domain** ✅
  - **Implementation**: `morphogen/stdlib/graph.py` (1,200+ lines)
  - **Data Structures**: `Graph`, `GraphMetrics`
  - **Core Operations**:
    - Graph creation: `create_empty`, `from_adjacency_matrix`, `to_adjacency_matrix`
    - Graph modification: `add_edge`, `remove_edge`
    - Graph algorithms: `dijkstra`, `shortest_path`, `bfs`, `dfs`
    - Network analysis: `degree_centrality`, `betweenness_centrality`, `pagerank`
    - Community detection: `connected_components`, `clustering_coefficient`
    - Advanced algorithms: `minimum_spanning_tree`, `topological_sort`, `max_flow`
    - Graph generators: `random_graph`, `grid_graph`
  - **Examples**: 3 comprehensive examples
    - `01_shortest_path.py` — Road network analysis, Dijkstra's algorithm
    - `02_network_analysis.py` — Social network analysis, centrality measures
    - `03_flow_network.py` — Maximum flow problem, network capacity analysis
  - **Impact**: Enables network analysis, social graphs, routing algorithms, flow optimization

- **Signal Processing Domain** ✅
  - **Implementation**: `morphogen/stdlib/signal.py` (1,000+ lines)
  - **Data Structures**: `Signal1D`, `Spectrum`, `Spectrogram`
  - **Core Operations**:
    - Signal generation: `sine_wave`, `chirp`, `white_noise`
    - Transforms: `fft`, `ifft`, `rfft`, `stft`, `istft`
    - Filtering: `lowpass`, `highpass`, `bandpass`
    - Windowing: `window` (Hann, Hamming, Blackman, Kaiser, etc.)
    - Analysis: `envelope`, `correlate`, `peak_detection`
    - Spectral analysis: `spectrogram_power`, `welch_psd`
    - Processing: `resample`, `normalize`
  - **Examples**: 2 comprehensive examples
    - `01_fft_analysis.py` — FFT spectrum analysis, frequency detection
    - `02_spectrogram.py` — STFT spectrogram, time-frequency analysis
  - **Impact**: Enables frequency analysis, spectral processing, time-frequency representations

- **State Machine Domain** ✅
  - **Implementation**: `morphogen/stdlib/statemachine.py` (900+ lines)
  - **Data Structures**: `State`, `Transition`, `StateMachine`, `BehaviorNode`
  - **Core Operations**:
    - State machine: `create`, `add_state`, `add_transition`, `start`
    - Execution: `update`, `send_event`, `get_state_name`, `is_in_state`
    - Transitions: Event-driven, automatic, timeout-based with guards and actions
    - Behavior trees: `create_sequence`, `create_selector`, `create_action`, `create_condition`
    - Visualization: `to_graphviz` (DOT format export)
  - **Examples**: 1 comprehensive example
    - `01_game_ai.py` — Game character AI with patrol, alert, chase, and attack states
  - **Impact**: Enables game AI, UI flows, protocol implementations, workflow systems

- **Terrain Generation Domain** ✅
  - **Implementation**: `morphogen/stdlib/terrain.py` (800+ lines)
  - **Data Structures**: `Heightmap`, `BiomeMap`
  - **Core Operations**:
    - Generation: `from_noise_perlin` (multi-octave procedural generation)
    - Erosion: `hydraulic_erosion`, `thermal_erosion`
    - Analysis: `calculate_slope`, `calculate_aspect`
    - Classification: `classify_biomes` (ocean, beach, grassland, forest, mountain, snow, desert)
    - Modification: `terrace`, `smooth`, `normalize`, `island_mask`
  - **Examples**: 1 comprehensive example
    - `01_island_generation.py` — Complete terrain pipeline with erosion and biomes
  - **Impact**: Enables procedural terrain generation, game worlds, geographic simulations

- **Computer Vision Domain** ✅
  - **Implementation**: `morphogen/stdlib/vision.py` (900+ lines)
  - **Data Structures**: `ImageGray`, `EdgeMap`, `Keypoint`, `Contour`
  - **Core Operations**:
    - Edge detection: `sobel`, `laplacian`, `canny`
    - Feature detection: `harris_corners`, `hough_lines`
    - Filtering: `gaussian_blur`
    - Morphology: `morphological` (erode, dilate, open, close, gradient, tophat, blackhat)
    - Segmentation: `threshold`, `adaptive_threshold`, `find_contours`
    - Analysis: `template_match`, `optical_flow_lucas_kanade`
  - **Examples**: 1 comprehensive example
    - `01_edge_detection.py` — Multi-algorithm edge detection and feature extraction
  - **Impact**: Enables image analysis, object detection, feature extraction, optical flow

### Technical Highlights

- **Total New Code**: 4,800+ lines of production code across 5 new domains
- **Total Domains**: 23 domains (up from 18)
- **New Examples**: 8 comprehensive example files with visualizations
- **Code Quality**: Follows established Morphogen patterns (immutability, NumPy backend, comprehensive docs)
- **Integration**: All domains compatible with existing cross-domain infrastructure

### Domain Ecosystem Growth

The addition of these 5 domains creates powerful new cross-domain workflows:
- **Terrain → Vision**: Analyze terrain features using computer vision algorithms
- **Signal → Audio**: Spectral analysis of audio signals
- **Graph → Agents**: Network-based agent behaviors and pathfinding
- **StateMachine → Agents**: Complex agent AI behaviors
- **Vision → Fields**: Edge detection for field boundary conditions

### Project Evolution - Morphogen Rename Preparation

**Commit**: `2e8c864` - Prepare for Morphogen rename and Philbrick integration

This release includes preparation for the project's upcoming rename from Morphogen to Morphogen, establishing the connection with sister hardware project Philbrick.

- **MORPHOGEN_RENAME_PREP.md** (437 lines)
  - Comprehensive rename execution checklist
  - User migration strategy and communication plan
  - Philbrick integration roadmap (4 phases)
  - Risk mitigation and timeline (target: 2025-12-15)

- **Sister Project Link**: README now references Philbrick
  - Philbrick: Modular analog computing hardware platform
  - Shared primitives: sum, integrate, nonlinearity, events
  - Unified philosophy: computation = composition

- **Bridge Documentation**: `docs/philbrick-bridge/`
  - Overview of Morphogen ↔ Philbrick connection (184 lines)
  - Software/hardware integration vision

- **Background**:
  - Decision documented in ADR-011 (wuluje-1116 session)
  - Morphogen: Named after Turing's morphogenesis work (1952)
  - Philbrick: Named after George A. Philbrick's modular analog computing (1952)
  - Historical grounding for educational/research positioning

**Next Steps**: Execute rename checklist when ready, tag as v0.11.0 "The Morphogen Rename"

---

## [v0.8.0] - 2025-11-15

### Added - Base-Level Domain Implementations ⭐⭐⭐

- **Integrators Dialect** (P0 - Critical) ✅
  - **Implementation**: `morphogen/stdlib/integrators.py` (520 lines)
  - **Operators**:
    - Explicit methods: `euler`, `rk2`, `rk4` (O(dt), O(dt²), O(dt⁴) accuracy)
    - Symplectic methods: `verlet`, `leapfrog`, `symplectic` (energy-conserving for physics)
    - Adaptive methods: `dormand_prince_step`, `adaptive_integrate` (automatic timestep control)
    - Generic interface: `integrate` (method selection)
  - **Tests**: 600+ lines with comprehensive accuracy, energy conservation, and determinism tests
  - **Examples**: 3 files (SHO comparison, adaptive integration, N-body gravity)
  - **Key Properties**:
    - Deterministic: Bit-exact repeatability guaranteed
    - Energy conservation: Symplectic methods < 0.01% drift over 10 periods
    - All tests passing with high accuracy (RK4 < 1e-6 error)
  - **Impact**: Unlocks principled time-stepping for all physics domains (Agent, Circuit, Fluid, Acoustics)

- **I/O & Storage Domain** (P1 - Foundational) ✅
  - **Implementation**: `morphogen/stdlib/io_storage.py` (576 lines)
  - **Operators**:
    - Image I/O: `load_image`, `save_image` (PNG, JPEG, BMP via Pillow)
      - Grayscale/RGB/RGBA support, quality control for JPEG (1-100)
      - Automatic normalization [0, 1] ↔ [0, 255] conversion
    - Audio I/O: `load_audio`, `save_audio` (WAV, FLAC, OGG via soundfile)
      - Mono/stereo support, automatic resampling with scipy
      - Stereo-to-mono downmix, format/subtype control
    - JSON I/O: `load_json`, `save_json` (with NumPy type support)
      - Auto NumPy type conversion (ndarray, int32, float32, bool_)
      - Pretty printing with configurable indentation and sorted keys
    - HDF5 I/O: `load_hdf5`, `save_hdf5` (compressed datasets)
      - Single/multiple dataset support, nested groups
      - gzip/lzf compression (214x compression ratio on zeros)
    - Checkpointing: `save_checkpoint`, `load_checkpoint` (full simulation state with metadata)
      - State + metadata, deterministic save/load
      - Supports nested dicts, arrays, scalars
  - **Tests**: 22 comprehensive tests across 6 test suites
    - pytest format: `test_io_storage.py` (600+ lines, 22 test functions)
    - Standalone verification: `verify_io_storage.py` (350+ lines, 6 suites)
    - Integration tests: Simulation workflow, field visualization pipeline
  - **Examples**: 3 comprehensive example files + README
    - `01_image_io.py` — 4 examples (gradients, grayscale, procedural textures, heatmaps)
    - `02_audio_io.py` — 4 examples (tones, stereo, chords, effects)
    - `03_simulation_checkpointing.py` — 4 examples (basic, resume, periodic, multi-field)
    - `README.md` — Complete API documentation, use cases, tips
  - **Key Properties**:
    - Supports all major formats (PNG, JPEG, WAV, FLAC, JSON, HDF5)
    - Automatic type conversion and normalization
    - Deterministic checkpoint save/load (bit-exact repeatability)
    - All roundtrip tests passing (22/22, 100% pass rate)
    - Graceful error handling (FileNotFoundError, TypeError, ValueError)
  - **Impact**: Enables asset loading, result export, checkpointing, data interchange with all external tools

- **Sparse Linear Algebra Domain** (P1 - Foundational) ✅
  - **Implementation**: `morphogen/stdlib/sparse_linalg.py` (588 lines)
  - **Operators**:
    - Sparse matrices: `csr_matrix`, `csc_matrix`, `coo_matrix`
    - Iterative solvers: `solve_cg`, `solve_bicgstab`, `solve_gmres`, `solve_sparse` (auto-select)
    - Preconditioners: `incomplete_cholesky`, `incomplete_lu`
    - Discrete operators: `laplacian_1d`, `laplacian_2d`, `gradient_2d`, `divergence_2d`
  - **Tests**: 10 comprehensive tests (solvers, operators, Poisson equation, determinism)
    - Standalone verification: `verify_sparse_linalg.py` (290 lines, 10 test functions)
    - Matrix creation, CG/BiCGSTAB/GMRES solvers, Laplacian operators, determinism
  - **Examples**: 3 comprehensive example files + README
    - `01_heat_equation.py` — 3 examples (1D/2D heat diffusion, convergence analysis)
    - `02_poisson_equation.py` — 3 examples (electrostatics, pressure projection, periodic BC)
    - `03_solver_comparison.py` — 4 examples (solver comparison, performance benchmarks up to 512×512 grids)
    - `README.md` — Complete API documentation, solver selection guidelines, performance tips
  - **Key Properties**:
    - CG solver: Converges in 25 iterations for 50×50 Laplacian (< 1e-14 error)
    - BiCGSTAB/GMRES: Robust for nonsymmetric matrices
    - 2D Laplacian: 5-point stencil with Dirichlet/Neumann/Periodic BC
    - Scales to 250K+ unknowns efficiently
    - All tests passing with high accuracy
  - **Impact**: Unlocks large-scale PDEs (1M+ unknowns), circuit simulation (1000+ nodes), graph algorithms

### Technical Highlights

- **Total Implementation**: 1,684 lines of production code (Integrators: 520, I/O: 576, Sparse: 588)
- **Total Tests**: 1,840+ lines of verification tests (Integrators: 600, I/O: 950, Sparse: 290)
- **Test Coverage**: 100% pass rate across all domains (Integrators: 100%, I/O: 22/22, Sparse: 10/10)
- **Examples**: 13 comprehensive example files + documentation
  - Integrators: 3 files (SHO, adaptive, N-body)
  - I/O & Storage: 4 files + README (12 total demonstrations)
  - Sparse Linear Algebra: 3 files + README (10 total demonstrations)
- **Dependencies Satisfied**:
  - Integrators → Agent/Circuit/Fluid domains (time-stepping)
  - I/O & Storage → All domains (asset loading, result export, checkpointing)
  - Sparse Linear Algebra → Circuit/Fields/Graph domains (large systems)

### Breaking Changes
- None (additive changes only)

---

## [v0.9.1] - 2025-11-16

### Added - Rigid Body Physics Domain ⭐⭐⭐

- **RigidBody2D Physics** (PR #74) ✅
  - **Implementation**: `morphogen/stdlib/rigidbody.py` (850+ lines)
  - **Core Types**:
    - `RigidBody2D`: Full rigid body with position, rotation, velocity, angular velocity, mass, inertia
    - `PhysicsWorld2D`: Physics world container with gravity, damping, solver settings
    - `Contact`: Collision contact points with normal, penetration, tangent
    - `ShapeType`: Collision shapes (Circle, Box, Polygon)
  - **Layer 1 - Atomic Operators**:
    - `create_circle_body`: Create circular rigid bodies with correct inertia
    - `create_box_body`: Create rectangular rigid bodies
    - `apply_force`: Apply forces with optional torque
    - `apply_impulse`: Instantaneous velocity changes
    - `integrate_body`: Semi-implicit Euler integration
    - `clear_forces`: Reset force accumulators
  - **Layer 2 - Collision Detection**:
    - `detect_circle_circle_collision`: Circle-circle narrow phase
    - `detect_collisions`: Broad-phase collision detection (O(n²) brute force)
  - **Layer 3 - Physics Simulation**:
    - `resolve_collision`: Impulse-based collision response with restitution and friction
    - `step_world`: Complete physics step (forces → integration → collision → resolution)
    - `raycast`: Ray-body intersection queries
    - `get_body_vertices`: Extract vertices for rendering
  - **Physics Properties**:
    - **Restitution**: Coefficient of restitution (0=inelastic, 1=elastic)
    - **Friction**: Coulomb friction model (tangential impulse)
    - **Damping**: Linear and angular velocity damping
    - **Static Bodies**: Infinite mass/inertia for ground, walls
    - **Position Correction**: Baumgarte stabilization to prevent sinking
  - **Tests**: 32 comprehensive tests (90%+ pass rate)
    - Body creation (circle, box, static bodies)
    - Force and impulse application
    - Integration accuracy
    - Collision detection (overlap, exact overlap, no collision)
    - Collision response (elastic, inelastic, friction)
    - World simulation (gravity, multiple bodies, stacking)
    - Physics properties (energy conservation, restitution, friction)
    - Determinism verification
  - **Examples**: 3 comprehensive examples + README
    - `01_bouncing_balls.py` — 4 balls with different properties bouncing
    - `02_collision_demo.py` — 4 demos (elastic, inelastic, mass ratio, friction)
    - `03_box_stack.py` — Stack stability demonstration
    - `README.md` — Complete guide with physics parameters, integration notes
  - **Key Features**:
    - **Deterministic**: Bit-exact repeatability for same inputs
    - **Stable**: Iterative constraint solver (configurable iterations)
    - **Physically accurate**: Correct inertia tensors, momentum/angular momentum conservation
    - **High performance**: Suitable for 100+ bodies at 60 FPS
  - **Impact**:
    - Unlocks 2D game physics, robotics simulation, mechanical engineering
    - Foundation for constraints (joints, springs, motors)
    - Integrates with Field domain (bodies in flow fields)
    - Integrates with Agents domain (hybrid rigid/particle systems)

### Technical Highlights

- **Implementation**: 850+ lines of production physics code
- **Tests**: 32 tests covering all layers (90%+ pass rate)
- **Examples**: 3 working demonstrations with detailed output
- **Collision Detection**: Currently circle-circle (box-box coming soon)
- **Integration**: Semi-implicit Euler (can use Verlet/RK4 from integrators module)
- **Solver**: Iterative impulse-based with Baumgarte stabilization

### Future Enhancements (Planned)

- Box-box and circle-box collision detection
- Polygon collision support (SAT, GJK algorithms)
- Constraints (distance joint, hinge joint, spring joint, motor)
- Spatial hashing for O(n) broad-phase
- Continuous collision detection (CCD) to prevent tunneling
- Integration with integrators module (Verlet, RK4)

### Breaking Changes
- None (additive changes only)

---

## [v0.9.0] - 2025-11-16

### Added - Advanced Domains ⭐⭐⭐

- **Particle Effects / VFX Extensions for Agents Domain** ✅
  - **Implementation**: Extended `morphogen/stdlib/agents.py` (+340 lines) and `morphogen/stdlib/visual.py` (+120 lines)
  - **New Operators**:
    - **Particle Emission**: `agents.emit()` - Flexible particle emission system
      - Multiple emission shapes: `point`, `circle`, `sphere`, `cone`
      - Configurable velocity patterns (uniform, callable, shape-based)
      - Lifetime management with random ranges
      - Custom property initialization
    - **Lifetime & Aging**: `agents.age_particles()`, `agents.get_particle_alpha()`
      - Automatic age tracking and dead particle filtering
      - Alpha fade-in/fade-out based on particle age
      - Configurable fade timing (fraction of lifetime)
    - **Force Application**: `agents.apply_force()`, `agents.integrate()`
      - Uniform or per-particle force application (F = ma)
      - Callable force functions for dynamic behaviors
      - Euler integration for position updates
    - **Trail Rendering**: `agents.update_trail()` - Historical position tracking
      - Configurable trail length (circular buffer)
      - Automatic NaN initialization for empty trails
      - Visual trail rendering with alpha decay
    - **Particle Merging**: `agents.merge()` - Combine multiple particle systems
      - Union of properties across collections
      - Preserves alive masks
      - Handles missing properties gracefully
  - **Pre-built Behaviors** (`particle_behaviors`):
    - `vortex()` - Swirling vortex force field
    - `attractor()` - Gravitational attraction to point
    - `repulsor()` - Repulsive force with radius cutoff
    - `drag()` - Velocity-proportional air resistance
    - `turbulence()` - Random force noise (seeded)
  - **Visual Rendering Enhancements** (`visual.agents()`):
    - **Alpha Blending**: `alpha_property` for per-particle transparency
    - **Blend Modes**: `alpha` (standard) and `additive` (for fire/glow effects)
    - **Rotation Visualization**: `rotation_property` for velocity-direction indicators
    - **Trail Rendering**: Full implementation with alpha decay along trail
  - **Tests**: 11 comprehensive test suites (280+ lines)
    - Particle emission (basic, shapes, lifetimes, properties, callables)
    - Lifetime management (aging, death, partial death, alpha fading)
    - Force application (uniform, callable, integration)
    - Particle behaviors (vortex, attractor, repulsor, drag, turbulence)
    - Trail management (initialization, position recording)
    - Particle merging (collections, properties, empty lists)
    - Visual rendering (alpha, rotation, blending)
    - Determinism (seeded emission, turbulence)
  - **Examples**: 3 comprehensive particle effect demos
    - `fireworks_particles.py` - Fireworks with burst emission, trails, gravity, fade-out (300 frames)
    - `fire_particles.py` - Fire/smoke with continuous emission, buoyancy, turbulence, temperature-based coloring (300 frames)
    - `vortex_magic.py` - Dual counter-rotating vortices with trail effects and rotation visualization (400 frames)
  - **Key Properties**:
    - Deterministic emission with seed control
    - Efficient trail management with circular buffers
    - Alpha blending and additive blend modes for realistic effects
    - Composable force system (combine gravity, drag, vortex, etc.)
    - Automatic dead particle cleanup
  - **Impact**: Transforms agents domain from simple circles to full particle VFX system - enables fireworks, fire/smoke, magic spells, explosions, weather effects, and more

- **Cellular Automata Domain** (PR #73) ✅
  - **Implementation**: Complete cellular automata simulation domain
  - **Operators**: Conway's Game of Life, Langton's Ant, elementary CA, and custom rules
  - **Key Features**:
    - Efficient NumPy-based state updates
    - Neighborhood detection and rule application
    - Support for multiple CA types
    - State history tracking
  - **Impact**: Enables complexity emergence, artificial life simulations, pattern generation

- **Optimization Domain - Phase 1: Evolutionary Algorithms** (PR #72) ✅
  - **Implementation**: Foundational optimization algorithms for design discovery
  - **Algorithms Implemented**:
    - Genetic Algorithm (GA): Population-based search with selection, crossover, mutation
    - Differential Evolution (DE): Vector-based optimization for continuous spaces
    - Particle Swarm Optimization (PSO): Swarm intelligence for multi-modal landscapes
  - **Key Features**:
    - Unified optimization interface
    - Configurable population sizes and hyperparameters
    - Fitness evaluation framework
    - Convergence tracking and history
  - **Impact**: Transforms Morphogen from simulation platform into design discovery platform

- **Cross-Domain Showcase Examples** (PR #71) ✅
  - **Examples**: Comprehensive demonstrations showing domain integration
  - **Showcases**:
    - Multi-domain workflows (Field + Agent + Audio + Visual)
    - Real-world use cases across different domains
    - Integration patterns and best practices
    - Performance optimization techniques
  - **Impact**: Demonstrates Morphogen's unique value in cross-domain operator composition

### Technical Highlights

- **Total Implementation**: 3 major feature additions in coordinated PRs
- **Domain Coverage**: Expanded into complexity science and optimization
- **Integration**: Showcased cross-domain capabilities
- **All Tests Passing**: 100% test coverage for new features

### Breaking Changes
- None (additive changes only)

---

## [v0.8.1] - 2025-11-16

### Added - Procedural Graphics Domains ⭐⭐⭐

- **NoiseDomain** (Tier 1 - Critical) ✅
  - **Implementation**: `morphogen/stdlib/noise.py` (850+ lines)
  - **Operators** (11 total):
    - **Layer 1 - Basic Noise**: `perlin2d`, `simplex2d`, `value2d`, `worley`
    - **Layer 2 - Fractals**: `fbm`, `ridged_fbm`, `turbulence`, `marble`
    - **Layer 3 - Advanced**: `vector_field`, `gradient_field`, `plasma`
  - **Properties**:
    - Deterministic: Seeded RNGs, bit-exact repeatability
    - Multi-octave fBm with persistence/lacunarity control
    - Multiple distance metrics (Euclidean, Manhattan, Chebyshev)
    - Worley features (F1, F2, F2-F1)
  - **Impact**: Enables fractal visualization, procedural terrain, texture synthesis, turbulence fields

- **PaletteDomain** (Tier 1 - Critical) ✅
  - **Implementation**: `morphogen/stdlib/palette.py` (550+ lines)
  - **Operators** (15+ total):
    - **Layer 1 - Creation**: `from_colors`, `from_gradient`, `greyscale`, `rainbow`, `hsv_wheel`, `inferno`, `viridis`, `plasma`, `magma`, `cosine`, `fire`, `ice`
    - **Layer 2 - Transforms**: `shift`, `cycle`, `flip`/`reverse`, `lerp`, `saturate`, `brightness`
    - **Layer 3 - Application**: `map`, `map_cyclic`
  - **Properties**:
    - Perceptually uniform scientific colormaps (Viridis family)
    - Procedural cosine gradients (IQ-style)
    - Animatable palette cycling
    - Custom gradient stops with interpolation
  - **Impact**: Enables fractal coloring, heatmaps, spectrograms, procedural art, audio-reactive visuals

- **ColorDomain** (Tier 1 - Critical) ✅
  - **Implementation**: `morphogen/stdlib/color.py` (500+ lines)
  - **Operators** (15+ total):
    - **Layer 1 - Conversions**: `rgb_to_hsv`, `hsv_to_rgb`, `rgb_to_hsl`, `hsl_to_rgb`, `hex_to_rgb`, `rgb_to_hex`, `temperature_to_rgb`
    - **Layer 2 - Manipulation**: `add`, `multiply`, `mix`, `brightness`, `saturate`, `gamma_correct`
    - **Layer 3 - Blend Modes**: `blend_overlay`, `blend_screen`, `blend_multiply`, `blend_difference`, `blend_soft_light`
    - **Layer 4 - Utility**: `posterize`, `threshold`
  - **Properties**:
    - Accurate HSV/HSL conversions
    - Physical temperature-based coloring (1000K-40000K blackbody radiation)
    - Photoshop-style blend modes
    - Vectorized array operations
  - **Impact**: Enables color grading, temperature-based lighting, photoshop-style effects, procedural color generation

- **ImageDomain** (Tier 2 - Essential) ✅
  - **Implementation**: `morphogen/stdlib/image.py` (700+ lines)
  - **Operators** (20+ total):
    - **Layer 1 - Creation**: `blank`, `rgb`, `from_field`, `compose`
    - **Layer 2 - Transforms**: `scale`, `rotate`, `warp`
    - **Layer 3 - Filters**: `blur`, `sharpen`, `edge_detect`, `erode`, `dilate`
    - **Layer 4 - Compositing**: `blend`, `overlay`, `alpha_composite`
    - **Layer 5 - Effects**: `apply_palette`, `normal_map_from_heightfield`, `gradient_map`
  - **Properties**:
    - RGB and RGBA support
    - Multiple interpolation modes (nearest, bilinear, cubic)
    - Full blend mode support (normal, multiply, screen, overlay, difference, soft_light)
    - Gaussian blur, edge detection (Sobel, Prewitt, Laplacian)
    - Morphological operations (erode, dilate)
  - **Impact**: Enables procedural texture generation, fractal visualization, post-processing, simulation rendering, normal map generation

- **FieldDomain Extensions** (Tier 2 - Essential) ✅
  - **Extended**: `morphogen/stdlib/field.py` (417 → 690 lines, +273 lines)
  - **New Operators** (10 total):
    - **Differential Operators**: `gradient`, `divergence`, `curl`
    - **Processing**: `smooth`, `normalize`, `threshold`, `clamp`, `abs`
    - **Sampling**: `sample`, `magnitude`
  - **Properties**:
    - Accurate spatial derivatives (central differences)
    - Gaussian and box filtering
    - Bilinear interpolation for arbitrary position sampling
    - Vector field magnitude computation
  - **Impact**: Enables flow field visualization, vector field analysis, gradient-based effects, field smoothing

### Technical Highlights

- **Total Implementation**: 3,873 new lines across 5 domains
  - NoiseDomain: 850+ lines
  - PaletteDomain: 550+ lines
  - ColorDomain: 500+ lines
  - ImageDomain: 700+ lines
  - FieldDomain extensions: 273 lines
- **Total Operators**: 70+ new operations
- **Examples**: 1 comprehensive demo with 8 scenarios
- **Documentation**: 400+ lines (procedural-graphics-domains.md)
- **All Demos Pass**: 100% success rate

### Examples Added

**Directory**: `/examples/procedural_graphics/`

- ✅ `demo_all_domains.py` — Comprehensive demonstration (8 scenarios):
  1. Basic noise with palette mapping
  2. Fractional Brownian Motion (fBm)
  3. Marble patterns with post-processing
  4. Procedural terrain with normal maps
  5. Color manipulation and blend modes
  6. Field operations (divergence, curl, magnitude)
  7. Animated palette cycling
  8. Cosine gradient palettes (IQ-style)
- ✅ `README.md` — Quick start guide and use cases

### Documentation Added

- ✅ `docs/reference/procedural-graphics-domains.md` — Complete reference (400+ lines)
  - Domain overviews and API documentation
  - Complete examples for each domain
  - Use cases and best practices
  - Performance notes
  - Future extensions

### Use Cases Unlocked

- ✅ **Fractal Visualization**: Mandelbrot/Julia sets with advanced coloring
- ✅ **Procedural Terrain**: Height maps, normal maps, textures
- ✅ **Audio Visualization**: Spectrograms, waveform coloring, audio-reactive effects
- ✅ **Shader-Like Effects**: Cosine gradients, procedural textures, blend modes
- ✅ **Scientific Visualization**: Perceptually uniform colormaps, field analysis
- ✅ **Procedural Art**: Noise-based patterns, marble, plasma, turbulence
- ✅ **Game Development**: Terrain generation, texture synthesis, particle effects

### Breaking Changes
- None (additive changes only)

---

## [Unreleased] - 2025-11-15

### Added

- **Agent Property Access Methods** ⭐
  - **`Agents.get(property_name)`** - Get property array for alive agents only
    - Returns NumPy array of property values filtered by alive_mask
    - Efficient retrieval without copying dead agent data
    - Raises KeyError if property doesn't exist
  - **`Agents.get_all(property_name)`** - Get property array for ALL agents (including dead)
    - Returns complete NumPy array including dead agents
    - Useful for batch operations and debugging
    - Allows inspection of full agent state
  - **Key Benefits**:
    - Separation of concerns: `get()` for active agents, `get_all()` for complete state
    - Memory efficient: `get()` only returns alive agent data
    - Type safe: Both methods raise KeyError for missing properties
    - Consistent API: Matches Morphogen's property access patterns
  - **Use Cases**:
    - Agent simulation loops (use `get()` for active agents)
    - Debugging and visualization (use `get_all()` to see full state)
    - Performance optimization (avoid processing dead agents)
  - **Implementation**: `morphogen/stdlib/agents.py:56-83`
  - **Impact**: Provides clean, efficient interface for agent property access in simulations

- **Circuit/Electrical Engineering Domain** ⭐ (PR #43)
  - **New Specifications**:
    - `docs/ADR/003-circuit-modeling-domain.md`: Complete design rationale for circuit modeling domain
    - `docs/SPEC-CIRCUIT.md`: Full circuit domain specification with 4-layer operator hierarchy
  - **Example Circuits**:
    - `examples/circuit/01_rc_filter.morph`: RC filter with frequency response
    - `examples/circuit/02_opamp_amplifier.morph`: Nonlinear op-amp modeling
    - `examples/circuit/03_guitar_pedal.morph`: Circuit → Audio integration (Tube Screamer)
    - `examples/circuit/04_pcb_trace_inductance.morph`: Geometry → Circuit parasitic extraction
    - `examples/circuit/05_unified_example.morph`: Complete multi-domain integration
    - `examples/circuit/README.md`: Comprehensive circuit examples documentation
  - **Key Capabilities**: Typed operator graphs (R → C → Op-Amp), multi-domain integration (Circuit ↔ Audio, Geometry, Physics), reference-based composition, type + unit safety
  - **Domain Position**: Core domain 1.8 in DOMAIN_ARCHITECTURE.md
  - **Impact**: Establishes Morphogen as the only tool that unifies circuit simulation, PCB layout, analog audio modeling, and multi-physics coupling

- **Fluid Dynamics & Acoustics Domains** ⭐ (PR #44)
  - **New Use Cases**:
    - `docs/USE_CASES/2-stroke-muffler-modeling.md`: Complete multi-domain 2-stroke exhaust system modeling
  - **Domain Additions**:
    - FluidDynamics domain: Compressible 1D flow, incompressible flow, thermodynamic coupling, engine-specific operators
    - Acoustics domain: 1D waveguide simulation, FDTD acoustics, Helmholtz resonators, perforated pipes, radiation impedance
  - **Cross-Domain Integration**: FluidDynamics → Acoustics → Audio → Geometry coupling
  - **Updated**: `EXAMPLE_PORTFOLIO_PLAN.md` with FluidDynamics & Acoustics examples
  - **Impact**: Demonstrates Morphogen's unique value in unifying multi-domain operator graphs for problems requiring 6+ domains

- **Instrument Modeling & Timbre Extraction Domain** ⭐ (PR #45)
  - **New Specifications**:
    - `docs/SPEC-TIMBRE-EXTRACTION.md`: Complete specification with 35 operators across analysis, synthesis, and modeling
    - `docs/ADR/003-instrument-modeling-domain.md`: Architectural decision record for instrument modeling
  - **Key Capabilities**: Record acoustic guitar → extract timbre → synthesize new notes, MIDI instrument creation, timbre morphing, luthier analysis tools
  - **Domain Position**: Layer 7 domain in operator registry
  - **Updated**: `docs/LEARNINGS/OPERATOR_REGISTRY_EXPANSION.md` and `docs/DOMAIN_ARCHITECTURE.md` (section 2.7)
  - **Impact**: Enables one of the "holy grails" of audio DSP - converting recordings into reusable synthesis models

- **Audio Time Alignment Operators** (PR #46)
  - **New Operator Subcategories** (Layer 5: Audio/DSP):
    - Measurement operators: sine_sweep, impulse_train, mls_sequence, white_noise_burst
    - Analysis operators: impulse_response_extractor, ir_peak_detect, cross_correlation, group_delay, phase_difference
    - Alignment operators: delay_designer, crossover_phase_aligner, allpass_delay, delay_compensation
    - Export operators: export_delays (miniDSP, JSON, CSV)
  - **New Reference Types**: ImpulseResponseRef, DelayMapRef
  - **Updated**: `docs/LEARNINGS/TIME_ALIGNMENT_OPERATORS.md`
  - **Impact**: Solves critical pro audio problem (car audio, studio monitors) with measurement → analysis → design workflow

- **Multi-Physics Engineering Modeling** ⭐ (PR #47)
  - **New Specifications**:
    - `docs/SPEC-PHYSICS-DOMAINS.md`: Four new physics domains (FluidNetwork, ThermalODE, FluidJet, CombustionLight)
    - `docs/EXAMPLES/J-TUBE-FIREPIT-MULTIPHYSICS.md`: Complete J-tube fire pit multi-physics modeling example
    - `docs/EXAMPLES/README.md`: Comprehensive guide to Morphogen examples and case studies
  - **Key Capabilities**: Geometry → Fluid → Thermal → Combustion pipeline, validates operator graph paradigm for engineering physics
  - **Updated**: `docs/DOMAIN_ARCHITECTURE.md` with Next-Wave physics domains
  - **Impact**: Proves Morphogen can model thermal-fluid systems, combustion, and multi-physics engineering problems

- **Optimization Domain & Algorithms** ⭐ (PR #48)
  - **New Catalog**:
    - `docs/LEARNINGS/OPTIMIZATION_ALGORITHMS_CATALOG.md`: Complete catalog of 16 optimization algorithms across 5 categories
  - **Algorithm Categories**:
    - Evolutionary/Population-Based: GA, DE, CMA-ES, PSO
    - Local Numerical: Gradient Descent, L-BFGS, Nelder-Mead
    - Surrogate/Model-Based: Bayesian Optimization, Response Surface, Kriging
    - Combinatorial/Discrete: Simulated Annealing, Tabu Search, Beam Search
    - Multi-Objective: NSGA-II, SPEA2, MOPSO
  - **Updated**: `docs/DOMAIN_ARCHITECTURE.md` (section 2.3) with algorithm summaries, operator contracts, 3-phase roadmap
  - **Impact**: Transforms Morphogen from simulation platform into design discovery platform

- **TiaCAD Integration - Unified Reference & Frame Model** ⭐
  - **New Specifications**:
    - `docs/SPEC-COORDINATE-FRAMES.md`: Complete specification for coordinate frames, anchors, and reference-based composition across all domains
    - `docs/SPEC-GEOMETRY.md`: Full geometry domain specification with TiaCAD-inspired operator families (primitives, sketches, booleans, patterns, finishing ops)
    - `docs/ADR/001-unified-reference-model.md`: Architecture Decision Record formalizing the unified reference system
  - **Enhanced Documentation**:
    - `docs/SPEC-TRANSFORM.md`: Added Section 7 on spatial transformations (affine, coordinate conversions, projective, frame-aware transforms)
    - `docs/DOMAIN_ARCHITECTURE.md`: Expanded Section 2.1 with comprehensive geometry domain coverage, TiaCAD principles, and cross-domain anchor concepts
    - `docs/SPEC-OPERATOR-REGISTRY.md`: Extended Layer 6 with geometry operators (primitives, sketches, booleans, patterns, mesh ops, anchors)
  - **Key Concepts from TiaCAD v3.x**:
    - Reference-based composition via anchors (replaces hierarchical assemblies)
    - Explicit transform origins (no implicit rotation/scale centers)
    - Deterministic transform chains (pure functions, no hidden state)
    - Auto-generated anchors for all geometric objects
    - Cross-domain applicability (geometry, audio, physics, agents, fields)
    - Backend-neutral operator semantics with multiple lowering targets
  - **Impact**: Unifies spatial, temporal, and structural references across all Morphogen domains; provides declarative, refactor-safe composition model; establishes foundation for v0.9+ geometry domain implementation

### Changed
- **Code cleanup**: Removed stale TODOs and updated documentation
  - `morphogen/mlir/context.py`: Updated dialect registration docs to reflect implemented dialects (Phases 2-5)
  - `morphogen/mlir/compiler_v2.py`: Replaced outdated "Phase 1" TODOs with clear implementation status notes

---

## [0.7.4] - 2025-11-14

**Status**: Phase 6 Complete - JIT/AOT Compilation with LLVM Backend ✅

### Overview - LLVM-Based JIT/AOT Compilation

Phase 6 of Morphogen v0.7.4 implements the complete JIT/AOT compilation infrastructure using LLVM backend. This phase bridges the gap between high-level MLIR dialects and executable native code, enabling production-ready compilation with optimization, caching, and multiple output formats.

### Added - JIT/AOT Compilation (Phase 6)

#### LLVM Lowering Pass
- **`morphogen.mlir.lowering.scf_to_llvm`** - SCF/Arith/Func to LLVM dialect lowering (199 lines):
  - `SCFToLLVMPass`: Complete lowering pipeline using MLIR's built-in passes
  - `convert-scf-to-cf`: Lower SCF to Control Flow dialect
  - `convert-arith-to-llvm`: Lower arithmetic operations to LLVM
  - `finalize-memref-to-llvm`: Lower MemRef to LLVM
  - `convert-func-to-llvm`: Lower functions to LLVM
  - `reconcile-unrealized-casts`: Clean up unrealized casts
  - Optimization levels 0-3 with inlining, CSE, LICM, loop unrolling, vectorization
  - `lower_to_llvm()` convenience function

#### JIT Compilation Engine
- **`morphogen.mlir.codegen.jit`** - JIT compilation with caching (392 lines):
  - `MorphogenJIT`: Full-featured JIT compiler
  - `CompilationCache`: In-memory and persistent disk caching
  - Thread-safe execution with locks
  - Automatic argument marshalling (scalars, NumPy arrays)
  - Function signature introspection
  - SHA256-based cache keys
  - Cache hit/miss tracking
  - Optimization levels 0-3
  - `create_jit()` factory function

#### AOT Compilation Engine
- **`morphogen.mlir.codegen.aot`** - AOT compilation to native binaries (626 lines):
  - `MorphogenAOT`: Complete AOT compiler
  - `OutputFormat` enum: 7 output formats
    - `EXECUTABLE`: Native executables (.exe, no extension)
    - `SHARED_LIB`: Shared libraries (.so, .dylib, .dll)
    - `STATIC_LIB`: Static libraries (.a, .lib)
    - `OBJECT_FILE`: Object files (.o, .obj)
    - `LLVM_IR_TEXT`: LLVM IR text (.ll)
    - `LLVM_BC`: LLVM bitcode (.bc)
    - `ASSEMBLY`: Assembly (.s)
  - Cross-compilation support (target triple)
  - Custom linker flags
  - Symbol export control
  - LLVM toolchain integration (llc, llvm-as, gcc/clang, ar)
  - `create_aot()` factory function

#### ExecutionEngine API
- **`morphogen.mlir.codegen.executor`** - High-level unified API (405 lines):
  - `ExecutionEngine`: Context manager for JIT/AOT execution
  - `ExecutionMode` enum: JIT, AOT, INTERPRET
  - `MemoryBuffer`: Automatic memory management with cleanup
  - Buffer allocation with NumPy integration
  - Memory usage tracking
  - Function listing and signature introspection
  - Automatic resource cleanup on context exit
  - `create_execution_engine()` factory function

#### Integration with Existing Dialects
- **Complete compilation pipeline**:
  - Field Dialect → FieldToSCFPass → SCFToLLVMPass → Native code
  - Temporal Dialect → TemporalToSCFPass → SCFToLLVMPass → Native code
  - Agent Dialect → AgentToSCFPass → SCFToLLVMPass → Native code
  - Audio Dialect → AudioToSCFPass → SCFToLLVMPass → Native code
- All existing dialects can now be JIT/AOT compiled
- Unified lowering infrastructure
- End-to-end optimization pipeline

### Tests Added

#### `tests/test_jit_aot_compilation.py` (1,074 lines)
- **55 comprehensive tests** covering all functionality:
  - **LLVM Lowering Tests** (10 tests):
    - Pass creation and factory functions
    - Simple function lowering
    - SCF loop lowering
    - Optimization levels (O0-O3)
    - Multiple module lowering
  - **JIT Compilation Tests** (15 tests):
    - JIT creation and compilation
    - Cache operations (get/put/clear)
    - Persistent disk cache
    - Optimization level benchmarks
    - Cache key computation
    - Argument marshalling
    - Function signature introspection
    - Thread safety
  - **AOT Compilation Tests** (12 tests):
    - Compilation to LLVM IR, object files, shared libraries, executables
    - Output format enum
    - Target triple (cross-compilation)
    - Custom linker flags
    - Symbol export control
    - LLVM toolchain integration
  - **ExecutionEngine Tests** (10 tests):
    - Context manager support
    - JIT/AOT mode switching
    - Buffer allocation
    - Memory usage tracking
    - Function listing
    - Resource cleanup
  - **Integration Tests** (8 tests):
    - Full JIT pipeline
    - Full AOT pipeline
    - Lowering + JIT
    - Cache persistence
    - Multiple compilations

### Examples Added

#### `examples/phase6_jit_aot_compilation.py` (521 lines)
- **8 complete working examples**:
  1. **Basic JIT Compilation**: Simple function JIT compilation and execution
  2. **JIT with Caching**: Persistent disk cache with cache hit demonstration
  3. **AOT to Shared Library**: Compile to .so with symbol export
  4. **AOT to Executable**: Compile to native binary with entry point
  5. **ExecutionEngine API**: High-level unified API with context manager
  6. **Field Operations JIT**: JIT compilation pipeline for field operations
  7. **Audio Synthesis JIT**: Real-time audio generation with JIT
  8. **Performance Benchmarking**: Compare optimization levels and modes

### Benchmarks Added

#### `benchmarks/jit_aot_benchmark.py` (573 lines)
- **7 performance benchmarks**:
  1. **JIT Compilation Time**: Measure compilation overhead (10 iterations)
  2. **Optimization Level Impact**: Compare O0/O1/O2/O3 performance
  3. **Cache Performance**: Cache hit/miss overhead measurement
  4. **AOT Compilation Time**: Benchmark different output formats
  5. **Memory Usage**: Track compilation memory consumption
  6. **Scalability**: Compilation time vs program size (10-200 loop iterations)
  7. **ExecutionEngine Overhead**: API wrapper overhead analysis

### Changed

#### Updated Components
- **`morphogen/mlir/codegen/__init__.py`** - Export JIT/AOT/Executor (+44 lines)
- **`morphogen/mlir/lowering/__init__.py`** - Export SCFToLLVMPass (+24 lines)

### Code Statistics

- **~2,225 lines** of production code added:
  - **199 lines**: SCF to LLVM lowering pass
  - **392 lines**: JIT compilation engine
  - **626 lines**: AOT compilation engine
  - **405 lines**: ExecutionEngine API
  - **603 lines**: Package integration
- **~2,168 lines** of test and benchmark code:
  - **1,074 lines**: Test suite (55 tests)
  - **521 lines**: Examples (8 examples)
  - **573 lines**: Benchmarks (7 benchmarks)
- **Total**: ~4,393 lines added

### Features Delivered

#### JIT Compilation ✅
- ✅ Real-time compilation to native code
- ✅ In-memory and persistent disk caching
- ✅ Thread-safe execution
- ✅ Automatic argument marshalling
- ✅ Function signature introspection
- ✅ Optimization levels 0-3

#### AOT Compilation ✅
- ✅ Native executables (.exe, ELF)
- ✅ Shared libraries (.so, .dylib, .dll)
- ✅ Static libraries (.a, .lib)
- ✅ Object files (.o, .obj)
- ✅ LLVM IR (text and bitcode)
- ✅ Assembly output (.s)
- ✅ Cross-compilation support

#### ExecutionEngine API ✅
- ✅ Unified JIT/AOT interface
- ✅ Context manager support
- ✅ Automatic memory management
- ✅ Buffer allocation
- ✅ Resource cleanup
- ✅ Memory usage tracking

#### Integration ✅
- ✅ Field dialect → Native code
- ✅ Temporal dialect → Native code
- ✅ Agent dialect → Native code
- ✅ Audio dialect → Native code
- ✅ Complete lowering pipeline
- ✅ End-to-end optimization

### Success Metrics ✅

- ✅ All 55 tests pass (100% pass rate)
- ✅ JIT compilation works with caching
- ✅ AOT compilation produces valid binaries (with LLVM toolchain)
- ✅ ExecutionEngine provides clean API
- ✅ Memory management works correctly
- ✅ Integration with all dialects functional
- ✅ Comprehensive examples and benchmarks
- ✅ Complete documentation

### Performance Characteristics

- **JIT Compilation**: ~1-10ms overhead (acceptable for runtime)
- **Cache Speedup**: Near-instant for cache hits
- **Optimization Levels**:
  - O0: Fastest compilation, no optimization
  - O1: Basic optimization, ~2x slower compilation
  - O2: Balanced (recommended), ~3x slower compilation
  - O3: Maximum performance, ~5x slower compilation
- **Memory Overhead**: ~few MB per compiled module
- **Scalability**: Linear scaling with program size

### Dependencies

- **MLIR Python bindings** (>= 18.0.0) - Required for JIT/AOT
- **LLVM toolchain** (optional) - For AOT to native binaries:
  - `llc` - LLVM compiler
  - `llvm-as` - LLVM assembler
  - `gcc` or `clang` - C compiler/linker
  - `ar` - Archive tool (for static libs)
- **NumPy** (>= 1.20.0) - For buffer management

### Known Limitations

- **ExecutionEngine pickling**: Compiled engines cannot be pickled (cache stores metadata only)
- **LLVM toolchain required**: AOT to native binaries requires external LLVM tools
- **Platform-specific**: Shared library and executable formats vary by OS
- **No GPU support yet**: LLVM CPU backend only (GPU in future phases)

### Next Phase

**Phase 7: GPU Compilation** - NVIDIA/AMD GPU Support (Months 16-18)
- CUDA/ROCm lowering passes
- GPU memory management
- Kernel fusion and optimization
- Multi-GPU support

---

## [0.7.3] - 2025-11-14

**Status**: Audio DSP & Spectral Analysis Complete ✅

### Overview - Audio Buffer Operations & Spectral Processing

Comprehensive audio buffer operations, DSP, and spectral analysis capabilities added to the audio module. This enhancement provides advanced signal processing tools including FFT/STFT transforms, spectral analysis metrics, frequency-domain processing, and flexible buffer manipulation operations.

### Added - Audio Operations

#### Buffer Operations (Section 5.6)
- **`audio.slice(signal, start, end)`** - Extract portion of audio buffer by time
- **`audio.concat(*signals)`** - Concatenate multiple audio buffers
- **`audio.resample(signal, new_sample_rate)`** - Resample to different sample rate with linear interpolation
- **`audio.reverse(signal)`** - Reverse audio buffer
- **`audio.fade_in(signal, duration)`** - Apply linear fade-in envelope
- **`audio.fade_out(signal, duration)`** - Apply linear fade-out envelope

#### FFT / Spectral Transforms (Section 5.7)
- **`audio.fft(signal)`** - Fast Fourier Transform with frequency bins
- **`audio.ifft(spectrum, sample_rate)`** - Inverse FFT reconstruction
- **`audio.stft(signal, window_size, hop_size)`** - Short-Time Fourier Transform (time-frequency analysis)
- **`audio.istft(stft_matrix, hop_size, sample_rate)`** - Inverse STFT with overlap-add reconstruction
- **`audio.spectrum(signal)`** - Get magnitude spectrum
- **`audio.phase_spectrum(signal)`** - Get phase spectrum

#### Spectral Analysis (Section 5.8)
- **`audio.spectral_centroid(signal)`** - Calculate spectral centroid (brightness measure)
- **`audio.spectral_rolloff(signal, threshold)`** - Find rolloff frequency (high-frequency content)
- **`audio.spectral_flux(signal, hop_size)`** - Measure spectral change over time (onset detection)
- **`audio.spectral_peaks(signal, num_peaks, min_freq)`** - Find dominant frequency peaks
- **`audio.rms(signal)`** - Calculate RMS level (loudness measure)
- **`audio.zero_crossings(signal)`** - Count zero crossing rate (noisiness/pitch indicator)

#### Spectral Processing (Section 5.9)
- **`audio.spectral_gate(signal, threshold_db, window_size, hop_size)`** - Spectral noise gate for noise reduction
- **`audio.spectral_filter(signal, freq_mask)`** - Apply arbitrary frequency-domain filter
- **`audio.convolution(signal, impulse)`** - FFT-based convolution (reverb, filtering)

### Implementation Details

#### Transform Operations
- **FFT/IFFT**: Uses NumPy's `rfft`/`irfft` for real-valued signals (efficient half-spectrum computation)
- **STFT/ISTFT**: Hann windowing with configurable window size and hop
- **Overlap-add synthesis**: Proper window normalization for perfect reconstruction
- **Time-frequency resolution**: Configurable trade-offs via window/hop parameters

#### Analysis Metrics
- **Spectral centroid**: Weighted average of frequencies (brightness)
- **Spectral rolloff**: Frequency below which N% of energy is contained
- **Spectral flux**: Sum of squared positive spectral differences (onset detection)
- **Peak detection**: Local maxima finding with magnitude sorting
- **RMS**: Root mean square for loudness estimation
- **Zero crossings**: Sign change counting (correlated with pitch/noise)

#### Processing Operations
- **Spectral gate**: Magnitude thresholding in STFT domain
- **Spectral filter**: Arbitrary frequency masking with FFT
- **Convolution**: Fast FFT-based convolution with auto-normalization

### Tests Added

#### `tests/test_audio_buffer_ops.py` (188 lines)
- 18 comprehensive tests covering:
  - Slice operations (basic, start-only, boundary clamping)
  - Concatenation (basic, multiple, sample rate validation)
  - Resampling (basic, same-rate, stereo)
  - Reverse operations (basic, double-reverse identity)
  - Fade operations (in, out, stereo, long duration)
  - Integration workflows (slice+concat, resample chains, reverse+fade)

#### `tests/test_audio_spectral.py` (425 lines)
- 33 comprehensive tests covering:
  - **FFT Operations** (9 tests): Basic FFT, peak detection, IFFT reconstruction, spectrum/phase extraction, STFT/ISTFT
  - **Spectral Analysis** (12 tests): Centroid, rolloff, flux, peaks, RMS, zero crossings with various signal types
  - **Spectral Processing** (9 tests): Spectral gate, filter, convolution
  - **Integration Workflows** (3 tests): FFT→modify→IFFT, STFT→modify→ISTFT, full analysis pipeline

### Code Statistics

- **~670 lines** of production code added to `morphogen/stdlib/audio.py`
- **~610 lines** of comprehensive test coverage
- **51 total tests** (18 buffer ops + 33 spectral ops)
- **100% test pass rate**
- **6 buffer operations**
- **6 transform operations**
- **6 analysis operations**
- **3 processing operations**
- All operations fully typed with NumPy arrays
- Comprehensive docstrings with examples

### Success Metrics ✅

- ✅ All 51 tests pass
- ✅ FFT/IFFT perfect reconstruction (5 decimal places)
- ✅ STFT/ISTFT near-perfect reconstruction (edge effects handled)
- ✅ Spectral metrics accurate (centroid, rolloff, flux)
- ✅ Peak detection finds multiple tones correctly
- ✅ Buffer operations preserve sample rates
- ✅ Stereo and mono signals handled appropriately
- ✅ Comprehensive error handling (stereo restrictions, boundary checks)

---

## [0.7.3] - 2025-11-14

**Status**: Phase 5 Complete - Audio Operations Dialect ✅

### Overview - Audio Operations Dialect

Phase 5 of Morphogen v0.7.0 integrates audio synthesis and processing into the MLIR compilation pipeline through a new Audio Operations dialect. This phase enables compiled audio generation with oscillators, filters, envelopes, and effects, building on the stdlib audio DSP operations added previously.

### Added - Audio Dialect (Phase 5)

#### Audio Operations
- **`morphogen.mlir.dialects.audio`** - Complete audio dialect with 5 operations (618 lines):
  - `AudioBufferCreateOp`: Allocate audio buffers (sample_rate, channels, duration)
  - `AudioOscillatorOp`: Generate waveforms (sine, square, saw, triangle)
  - `AudioEnvelopeOp`: Apply ADSR envelopes to signals
  - `AudioFilterOp`: IIR/FIR filters (lowpass, highpass, bandpass)
  - `AudioMixOp`: Mix multiple audio signals with scaling
- **Type System**:
  - `!morphogen.audio<sample_rate, channels>`: Audio buffer type
  - Example: `!morphogen.audio<44100, 1>` (mono 44.1kHz)
  - Supports variable sample rates and channel counts

#### Lowering Pass
- **`morphogen.mlir.lowering.audio_to_scf`** - Audio-to-SCF lowering pass (658 lines):
  - `audio.buffer.create` → `memref.alloc` with zero initialization
  - `audio.oscillator` → `scf.for` loops with `math.sin` for sine waves
  - `audio.envelope` → `scf.for` with ADSR state machine (`scf.if` for stages)
  - `audio.filter` → `scf.for` with IIR biquad filter (memref state variables)
  - `audio.mix` → `scf.for` with weighted summation
- Pattern-based lowering maintaining SSA
- Efficient memref-based audio buffer storage

#### Compiler Integration
- **Extended `morphogen.mlir.compiler_v2`** with audio methods (+319 lines):
  - `compile_audio_buffer_create()`: Compile buffer creation
  - `compile_audio_oscillator()`: Compile oscillator operation
  - `compile_audio_envelope()`: Compile ADSR envelope
  - `compile_audio_filter()`: Compile filter operation
  - `compile_audio_mix()`: Compile mix operation
  - `apply_audio_lowering()`: Apply audio-to-SCF pass
  - `compile_audio_program()`: Convenience API for audio programs

#### Tests and Examples
- **`tests/test_audio_dialect.py`** - Comprehensive test suite (835 lines):
  - 24 test methods covering all functionality
  - AudioType tests (mono/stereo, various sample rates)
  - Operation tests (buffer, oscillator, envelope, filter, mix)
  - Lowering pass tests
  - Compiler integration tests
  - Complex multi-operation program tests
- **`examples/phase5_audio_operations.py`** - Working demonstrations (521 lines):
  - 8 complete examples:
    1. Basic oscillator (440 Hz sine wave)
    2. ADSR envelope application
    3. Lowpass filter sweep
    4. Chord mixing (C major, 3 oscillators)
    5. Complete synthesizer patch (OSC → ENV → FILTER → MIX)
    6. Audio effects chain
    7. Multi-voice synthesis (polyphony)
    8. Bass synthesis with sub-oscillator

#### Documentation
- **`docs/PHASE5_COMPLETION_SUMMARY.md`** - Complete Phase 5 summary
- Updated dialect and lowering exports
- Inline documentation for all operations

### Changed

#### Updated Components
- **`morphogen/mlir/dialects/__init__.py`** - Export AudioDialect and AudioType
- **`morphogen/mlir/lowering/__init__.py`** - Export AudioToSCFPass (+13 lines)
- **`morphogen/mlir/compiler_v2.py`** - Audio compilation methods (+319 lines)

### Success Metrics ✅

- ✅ All audio operations compile to valid MLIR
- ✅ Lowering produces correct scf.for structures with math ops
- ✅ Generated waveforms match expected signals (sine: `sin(2π * freq * t + phase)`)
- ✅ Integration with Field/Temporal/Agent dialects works
- ✅ Compilation time <1s for typical audio programs
- ✅ Comprehensive test coverage (24 tests)
- ✅ Complete documentation and examples (8 examples)

### Code Statistics

- **~2,951 lines** of production code added
- **618 lines**: Audio dialect
- **658 lines**: Audio-to-SCF lowering
- **319 lines**: Compiler integration
- **835 lines**: Test suite
- **521 lines**: Examples
- 5 audio operations implemented
- Complete lowering infrastructure
- Full compiler integration

### Key Algorithms

1. **Sine Oscillator**: `sin(2π * freq * t / sample_rate)`
2. **ADSR Envelope**: State machine (attack → decay → sustain → release)
3. **Lowpass Filter**: `y[n] = α*x[n] + (1-α)*y[n-1]` (simplified single-pole)
4. **Audio Mixing**: `output[i] = Σ(gain[j] * buffer[j][i])`

### Integration Points

- **With stdlib audio**: Compiled ops can call stdlib FFT/spectral functions
- **With Field ops**: Audio buffers ↔ field data (sonification/synthesis)
- **With Temporal ops**: Audio synthesis evolving over timesteps
- **With Agent ops**: Agents triggering audio events

### Performance

- **Memory Layout**: Contiguous `memref<?xf32>` for cache efficiency
- **Compilation time**: Instant for typical programs (<100ms)
- **Loop Structure**: Single-level `scf.for` for most operations

### Next Phase

**Phase 6: JIT/AOT Compilation** - LLVM Backend (Months 13-15)
- Lower SCF → LLVM dialect
- Implement LLVM execution engine
- Add JIT compilation support
- Optimize loop structures (vectorization, unrolling)

---

## [0.7.2] - 2025-11-14

**Status**: Phase 4 Complete - Agent Operations ✅

### Overview - Agent Operations Dialect

Phase 4 of Morphogen v0.7.0 implements the agent operations dialect for agent-based simulations with spawning, behavior trees, and property management. This phase builds on Phase 2 (Field Operations) and Phase 3 (Temporal Execution) to enable multi-agent simulations compiled through MLIR to efficient native code.

### Added - Agent Dialect (Phase 4)

#### Agent Operations
- **`morphogen.mlir.dialects.agent`** - Complete agent dialect with 4 operations:
  - `AgentSpawnOp`: Create agents at positions with initial properties (position, velocity, state)
  - `AgentUpdateOp`: Update agent properties at specific indices
  - `AgentQueryOp`: Read agent property values
  - `AgentBehaviorOp`: Apply behavior rules (move, seek, bounce)
- **Type System**:
  - `!morphogen.agent<T>`: Agent collection type
  - Standard property layout: `[pos_x, pos_y, vel_x, vel_y, state]` (5 properties)
  - Memory model: `memref<?x5xT>` for dynamic agent arrays

#### Lowering Pass
- **`morphogen.mlir.lowering.agent_to_scf`** - Agent-to-SCF lowering pass (434 lines):
  - `agent.spawn` → `memref.alloc` + initialization loops
  - `agent.update` → `memref.store` operations
  - `agent.query` → `memref.load` operations
  - `agent.behavior` → `scf.for` loops with property computations
- Pattern-based lowering infrastructure
- SSA-compliant transformations
- Integration with Field and Temporal dialects

#### Compiler Integration
- **Extended `morphogen.mlir.compiler_v2`** with agent methods (+280 lines):
  - `compile_agent_spawn()`: Compile agent spawning
  - `compile_agent_update()`: Compile property updates
  - `compile_agent_query()`: Compile property queries
  - `compile_agent_behavior()`: Compile behavior operations
  - `apply_agent_lowering()`: Apply agent-to-SCF pass
  - `compile_agent_program()`: Convenience API for agent programs

#### Tests and Examples
- **`tests/test_agent_dialect.py`** - Comprehensive test suite (908 lines):
  - 36 test methods covering all functionality
  - Type system tests
  - Operation tests (spawn, update, query, behavior)
  - Lowering pass tests
  - Compiler integration tests
  - Integration tests with Field and Temporal dialects
- **`examples/phase4_agent_operations.py`** - Working demonstrations (547 lines):
  - 8 complete examples:
    1. Basic agent spawning
    2. Agent movement
    3. Multi-agent behaviors
    4. Property updates
    5. Bounce behavior
    6. Agent-field integration
    7. Temporal agent evolution
    8. Large-scale simulation (10,000+ agents)

#### Documentation
- **`docs/PHASE4_COMPLETION_SUMMARY.md`** - Complete Phase 4 summary
- **`docs/v0.7.0_DESIGN.md`** - Updated with Phase 4 deliverables
- **`STATUS.md`** - Updated with Phase 4 completion

### Changed

#### Updated Components
- **`morphogen/mlir/dialects/__init__.py`** - Export agent dialect
- **`morphogen/mlir/lowering/__init__.py`** - Export agent lowering pass (+13 lines)
- **`morphogen/mlir/compiler_v2.py`** - Agent compilation methods (+280 lines)

### Success Metrics ✅

- ✅ All agent operations compile to valid MLIR
- ✅ Lowering produces correct memref array structures
- ✅ Agent properties update correctly across timesteps
- ✅ Integration with field + temporal operations works
- ✅ Compilation time <1s for typical agent counts (<500ms for 10K agents)
- ✅ Comprehensive test coverage (36 tests)
- ✅ Complete documentation and examples (8 examples)

### Code Statistics

- **~2,700 lines** of production code added
- **526 lines**: Agent dialect
- **434 lines**: Agent-to-SCF lowering
- **280 lines**: Compiler integration
- **908 lines**: Test suite
- **547 lines**: Examples
- 4 agent operations implemented
- Complete lowering infrastructure
- Full compiler integration

### Performance

- **Compilation time**: <500ms for 10,000 agents
- **Memory usage**: 200 KB for 10,000 agents (5 properties × 4 bytes)
- **Scalability**: Tested up to 10,000+ agents

### Behaviors Implemented

1. **Move**: Simple velocity-based movement (position += velocity)
2. **Seek**: Move towards target with speed control
3. **Bounce**: Boundary collision handling

### Integration Points

- **With Field Operations**: Agents coexist with spatial fields (future: field sampling)
- **With Temporal Operations**: Agents evolve within temporal flows
- **Combined Workflow**: Field + Temporal + Agent in single program

### Next Phase

**Phase 5: Audio Operations OR JIT/AOT Compilation** - Next major feature (Months 13-15)

Options:
- Audio Operations: Audio buffers, DSP, spectral analysis, synthesis
- JIT/AOT: MLIR→LLVM lowering, execution engine, optimization passes

---

## [0.7.1] - 2025-11-14

**Status**: Phase 3 Complete - Temporal Execution ✅

### Overview - Temporal Execution Layer

Phase 3 of Morphogen v0.7.0 implements the temporal execution layer, enabling time-evolving simulations with flow blocks and state management. This builds on Phase 2's field operations to add temporal dynamics.

### Added - Temporal Dialect (Phase 3)

#### Temporal Operations
- **`morphogen.mlir.dialects.temporal`** - Complete temporal dialect with 6 operations:
  - `FlowCreateOp`: Define flow blocks with temporal parameters (dt, steps)
  - `FlowStepOp`: Single timestep execution (placeholder for future)
  - `FlowRunOp`: Execute complete flow for N timesteps
  - `StateCreateOp`: Allocate persistent state containers
  - `StateUpdateOp`: Update state values (SSA-compatible)
  - `StateQueryOp`: Read current state values
- **Type System**:
  - `!morphogen.flow<T>`: Flow type representing temporal execution blocks
  - `!morphogen.state<T>`: State type representing persistent storage

#### Lowering Pass
- **`morphogen.mlir.lowering.temporal_to_scf`** - Temporal-to-SCF lowering pass:
  - `flow.create` → Flow metadata storage
  - `flow.run` → `scf.for` loop with iter_args for state evolution
  - `state.create` → `memref.alloc` + initialization loops
  - `state.update` → `memref.store` operations
  - `state.query` → `memref.load` operations
- Pattern-based lowering infrastructure
- Maintains SSA form throughout transformations
- Integration with Phase 2 field operations

#### Compiler Integration
- **Extended `morphogen.mlir.compiler_v2`** with temporal methods:
  - `compile_flow_create()`: Compile flow creation
  - `compile_flow_run()`: Compile flow execution
  - `compile_state_create()`: Compile state allocation
  - `compile_state_update()`: Compile state updates
  - `compile_state_query()`: Compile state queries
  - `apply_temporal_lowering()`: Apply temporal-to-SCF pass
  - `compile_temporal_program()`: Convenience API for temporal programs

#### Tests and Examples
- **`tests/test_temporal_dialect.py`** - Comprehensive test suite:
  - Unit tests for all 6 temporal operations
  - Integration tests with lowering passes
  - Compiler integration tests
  - 30+ test methods covering all functionality
- **`examples/phase3_temporal_execution.py`** - Working demonstrations:
  - State creation and management
  - Flow execution with timesteps
  - State update/query operations
  - Combined field + temporal operations

#### Documentation
- **`PHASE3_COMPLETION_SUMMARY.md`** - Complete Phase 3 summary
- **`docs/v0.7.0_DESIGN.md`** - Updated with Phase 3 deliverables
- **`STATUS.md`** - Updated with Phase 3 completion

### Changed

#### Updated Components
- **`morphogen/mlir/dialects/__init__.py`** - Export temporal dialect
- **`morphogen/mlir/lowering/__init__.py`** - Export temporal lowering pass

### Success Metrics ✅

- ✅ All temporal operations compile to valid MLIR
- ✅ Lowering produces correct scf.for loop structures
- ✅ State management works across timesteps (memref-based)
- ✅ Integration with field operations functional
- ✅ Compilation time remains <1s for typical flows
- ✅ Comprehensive test coverage (30+ tests)
- ✅ Complete documentation and examples

### Code Statistics

- ~2,500 lines of production code added
- 6 temporal operations implemented
- Complete lowering infrastructure
- Full compiler integration

### Next Phase

**Phase 4: Agent Operations** - Agent spawning, behavior trees, property updates (Months 10-12)

---

## [0.7.0] - In Development (Started 2025-11-14)

**Status**: Phase 2 Complete - Field Operations Dialect (Months 1-6 of 12+ month effort)

### Overview - Real MLIR Integration

Morphogen v0.7.0 represents a fundamental transformation from text-based MLIR IR generation to **real MLIR integration** using Python bindings. This enables true native code generation, optimization passes, and JIT compilation.

### Added - Field Operations Dialect (Phase 2 - Completed 2025-11-14)

#### Field Dialect Implementation (`morphogen/mlir/dialects/field.py`)
- **FieldCreateOp** - Allocate fields with dimensions and fill value
- **FieldGradientOp** - Central difference gradient computation
- **FieldLaplacianOp** - 5-point stencil Laplacian operator
- **FieldDiffuseOp** - Jacobi diffusion solver with double-buffering
- **FieldType** - Opaque type wrapper for `!morphogen.field<T>` MLIR type

#### Field-to-SCF Lowering Pass (`morphogen/mlir/lowering/field_to_scf.py`)
- Pattern-based lowering infrastructure transforming field ops → nested `scf.for` loops + `memref`
- Stencil operations with proper boundary handling (gradient, Laplacian)
- Double-buffering strategy for iterative solvers
- In-place IR transformation with SSA management
- Boundary-aware loop bounds for edge cases

#### Compiler Integration (`morphogen/mlir/compiler_v2.py`)
- `compile_field_create/gradient/laplacian/diffuse` methods for individual operations
- `apply_field_lowering` pass integration into compilation pipeline
- `compile_field_program` convenience API for Phase 2 workflows
- Full compilation pipeline: Field dialect → SCF lowering → MLIR optimization

#### Testing & Validation (`tests/test_field_dialect.py`)
- Comprehensive test suite (35+ tests)
- Unit tests for all field operations
- Integration tests for chained operations (gradient → Laplacian → diffusion)
- Coverage for various field sizes (32x32, 64x64, 128x128)
- Conditional execution based on MLIR availability

#### Examples (`examples/phase2_field_operations.py`)
- 5 complete working examples demonstrating full compilation pipeline
- Field creation, gradient computation, Laplacian, diffusion, combined workflows
- MLIR IR visualization (before/after lowering)
- Educational documentation for Phase 2 patterns

#### Performance Benchmarking (`benchmarks/field_operations_benchmark.py`)
- Compilation time measurements (<1s for all test cases)
- IR size and complexity metrics
- Scalability analysis across field sizes
- Success criteria validation

#### Documentation
- `docs/PHASE2_IMPLEMENTATION_PLAN.md` - Detailed implementation plan
- `PHASE2_COMPLETION_SUMMARY.md` - Comprehensive completion report
- Updated `STATUS.md` with Phase 2 status
- Updated `v0.7.0_DESIGN.md` with Phase 2 achievements

**Technical Achievements**:
- Real MLIR Python bindings integration (zero text templates)
- Custom dialect operations with proper MLIR builder patterns
- Pattern-based lowering pass infrastructure
- Stencil operations with boundary handling
- ~2,800 lines of production code + tests + docs

**Success Metrics** (All Met ✅):
- All field operations compile to valid MLIR
- Lowering produces correct SCF loop structures
- Compilation time < 1s for all test cases
- Comprehensive test coverage
- Complete documentation and examples
- Performance benchmarking operational

**PR**: #32 merged 2025-11-14 (+3,015 additions, 11 files changed)

### Added - MLIR Infrastructure (Phase 1)

#### Core Architecture
- **MLIR Python bindings integration** - Replaced text-based IR generation with real MLIR
- **`morphogen.mlir.context`** - MLIR context management and dialect registration
- **`morphogen.mlir.compiler_v2`** - New compiler using real MLIR Python bindings
- **Module structure** for progressive implementation:
  - `morphogen/mlir/dialects/` - Custom Morphogen dialects (field, agent, audio, visual)
  - `morphogen/mlir/lowering/` - Lowering passes (Morphogen → SCF → LLVM)
  - `morphogen/mlir/codegen/` - JIT and AOT compilation engines

#### Documentation
- **`docs/v0.7.0_DESIGN.md`** - Comprehensive design document for 12-month implementation
  - Architecture overview and module structure
  - Phase-by-phase implementation plan
  - Testing strategy and success metrics
  - Migration path from v0.6.0
- **`requirements.txt`** - Added MLIR dependencies with installation instructions
- **`examples/mlir_poc.py`** - Proof-of-concept demonstration

#### Development Setup
- Graceful degradation when MLIR not installed (falls back to legacy)
- Feature flags for MLIR vs legacy backend
- Installation instructions for MLIR Python bindings

### Changed

#### Deprecated
- **`morphogen/mlir/ir_builder.py`** - Legacy text-based IR builder (marked deprecated)
- **`morphogen/mlir/optimizer.py`** - Legacy optimization passes (marked deprecated)
- Legacy components will be maintained for v0.6.0 compatibility during transition

### Planned (Future Phases)

#### Phase 3: Temporal Execution (Months 7-9)
- Flow block compilation to MLIR
- State management via memref
- Temporal iteration support

#### Phase 4: JIT Compilation (Months 10-12)
- JIT execution engine
- Native code generation via LLVM
- Performance optimization and benchmarking

### Dependencies

#### Required (when MLIR enabled)
- `mlir>=18.0.0` - MLIR Python bindings (install separately)
  - Installation: `pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest`
  - Or build from source: https://mlir.llvm.org/docs/Bindings/Python/

#### Optional
- `pytest>=7.0.0` - Testing
- `pytest-cov>=4.0.0` - Coverage

### Notes

- **Timeline**: 12+ month implementation effort
- **Current Status**: Design phase and foundation setup complete
- **Backward Compatibility**: Legacy text-based backend remains available
- **Performance Target**: 10-100x speedup for field operations once complete

---

## [0.6.0] - 2025-11-14

### Added - Audio I/O and Visual Dialect Extensions

#### Audio I/O Operations
- **`audio.play(buffer, blocking)`** - Real-time audio playback (sounddevice backend)
- **`audio.save(buffer, path, format)`** - Export to WAV/FLAC (soundfile/scipy backends)
- **`audio.load(path)`** - Load audio files (WAV/FLAC support)
- **`audio.record(duration, sample_rate)`** - Microphone recording (sounddevice backend)
- Sample rate conversion and format handling
- Mono and stereo support
- Round-trip accuracy verification

#### Visual Dialect Extensions
- **`visual.agents(agents, width, height, ...)`** - Render particles/agents as points/circles
  - Color-by-property support (velocity, energy, etc.) with palettes
  - Size-by-property support for variable-size agents
  - Multiple rendering styles (points, circles)
- **`visual.layer(width, height, background)`** - Create blank visual layers
- **`visual.composite(*layers, mode, opacity)`** - Multi-layer composition
  - Blending modes: `over`, `add`, `multiply`, `screen`, `overlay`
  - Per-layer opacity control
  - Arbitrary number of layers
- **`visual.video(frames, path, fps, format)`** - Video export
  - MP4 support (imageio-ffmpeg backend)
  - GIF support (imageio backend)
  - Frame generator support for memory-efficient animations
  - Configurable frame rate and quality

#### Integration
- Field + Agent visual composition workflows
- Audio-visual synchronized content examples
- Multi-modal export (audio + video)
- Complete demonstration scripts (`audio_io_demo.py`, `visual_composition_demo.py`)

#### Dependencies
- **Added**: sounddevice >= 0.4.0 (audio playback/recording)
- **Added**: soundfile >= 0.12.0 (WAV/FLAC I/O)
- **Added**: scipy >= 1.7.0 (WAV fallback)
- **Added**: imageio >= 2.9.0 (video export)
- **Added**: imageio-ffmpeg >= 0.4.0 (MP4 codec)
- Optional dependency group: `morphogen[io]` installs all I/O dependencies

#### Testing
- **64+ new I/O integration tests**:
  - 24 audio I/O tests (playback, file I/O, recording)
  - 40+ visual extension tests (agent rendering, composition, video export)
- **580+ total tests** (247 original + 85 agent + 184 audio + 64+ I/O tests)

#### Examples
- `examples/audio_io_demo.py` - Complete audio I/O demonstrations
- `examples/visual_composition_demo.py` - Visual composition and video export
- Real-time playback examples
- Video animation examples
- Multi-layer composition examples

### Documentation
- Added Audio I/O usage examples
- Added Visual composition tutorials
- Updated installation instructions for I/O dependencies
- Video export best practices

---

## [0.5.0] - 2025-11-14

### Added - Audio Dialect Implementation (Production-Ready)

#### AudioBuffer Type and Core Operations
- **`AudioBuffer`** class with NumPy backend
  - Sample rate management (default 44100 Hz)
  - Mono and stereo support
  - Duration and sample count tracking
  - Deterministic buffer operations

#### Oscillators
- **`audio.sine(freq, duration)`** - Sine wave oscillator
- **`audio.saw(freq, duration, blep)`** - Sawtooth with optional BLEP anti-aliasing
- **`audio.square(freq, duration, pulse_width, blep)`** - Square/pulse wave
- **`audio.triangle(freq, duration)`** - Triangle wave
- **`audio.noise(noise_type, seed, duration)`** - White, pink, and brown noise
- **`audio.impulse(amplitude, sample_rate)`** - Single-sample impulse

#### Filters
- **`audio.lowpass(buffer, cutoff, q)`** - Biquad lowpass filter
- **`audio.highpass(buffer, cutoff, q)`** - Biquad highpass filter
- **`audio.bandpass(buffer, center, q)`** - Biquad bandpass filter
- **`audio.notch(buffer, center, q)`** - Biquad notch filter
- **`audio.eq3(buffer, low_gain, mid_gain, high_gain)`** - 3-band equalizer

#### Envelopes
- **`audio.adsr(attack, decay, sustain, release, duration)`** - ADSR envelope generator
- **`audio.ar(attack, release, duration)`** - Attack-release envelope
- **`audio.envexp(time_constant, duration)`** - Exponential decay envelope

#### Effects
- **`audio.delay(buffer, delay_time, feedback, mix)`** - Delay line effect
- **`audio.reverb(buffer, mix, size, damping)`** - Reverb effect (feedback delay network)
- **`audio.chorus(buffer, rate, depth, mix)`** - Chorus effect (modulated delay)
- **`audio.flanger(buffer, rate, depth, feedback, mix)`** - Flanger effect
- **`audio.drive(buffer, amount)`** - Soft saturation/distortion
- **`audio.limiter(buffer, threshold, release_time)`** - Peak limiter

#### Utilities
- **`audio.mix(*buffers)`** - Mix multiple audio buffers
- **`audio.gain(buffer, amount_db)`** - Apply gain in decibels
- **`audio.pan(buffer, position)`** - Stereo panning (-1.0 to 1.0)
- **`audio.clip(buffer, threshold)`** - Hard clipping
- **`audio.normalize(buffer, target)`** - Normalize peak level
- **`audio.db2lin(db)`** - Convert decibels to linear amplitude

#### Physical Modeling
- **`audio.string(excitation, freq, t60, damping)`** - Karplus-Strong string synthesis
  - Frequency-dependent loss filter
  - Adjustable decay time (T60)
  - Tunable damping
- **`audio.modal(excitation, freqs, decays, amps)`** - Modal synthesis
  - Multiple resonant modes
  - Independent decay rates
  - Amplitude control per mode
  - Useful for bells, percussion, resonant bodies

#### Testing
- **192 comprehensive audio tests** (184 passing, 96% pass rate):
  - `tests/test_audio_basic.py` (42 tests) - Oscillators, utilities, buffers
  - `tests/test_audio_filters.py` (36 tests) - All filter operations
  - `tests/test_audio_envelopes.py` (31 tests) - Envelope generators
  - `tests/test_audio_effects.py` (35 tests) - Effects processing
  - `tests/test_audio_physical.py` (31 tests) - Physical modeling
  - `tests/test_audio_integration.py` (17 tests) - Full compositions, runtime

#### Determinism
- ✅ All operations produce identical results with same seed
- ✅ Verified through automated tests
- ✅ Noise generation uses deterministic NumPy RNG
- ✅ All effects and filters are deterministic

#### Use Cases
- ✅ Synthesized tones and pads
- ✅ Plucked string instruments (guitar, bass, harp)
- ✅ Bell and percussion sounds
- ✅ Drum synthesis
- ✅ Effect chains (guitar, vocal, mastering)
- ✅ Complete musical compositions

#### Runtime Integration
- Audio namespace available in Morphogen runtime
- Full integration with parser and type system
- AudioBuffer type registered
- Example compositions working

#### Documentation
- Complete audio operation reference
- Physical modeling examples
- Effect chain tutorials
- Composition examples

### Implementation
- **`morphogen/stdlib/audio.py`** (1,250+ lines of production code)
- NumPy-based for performance
- Modular design with clear separation of concerns
- Comprehensive docstrings and type hints

---

## [0.4.0] - 2025-11-14

### Added - Agent Dialect Implementation (Sparse Particle Systems)

#### Agents<T> Type System
- **`Agents`** class for managing collections of particles/agents
  - Property-based data structure (pos, vel, mass, etc.)
  - NumPy-backed for performance
  - Alive/dead agent masking
  - Efficient property access and updates

#### Agent Operations
- **`agents.alloc(count, properties)`** - Allocate agent collection
- **`agents.map(agents, property, func)`** - Apply function to each agent property
- **`agents.filter(agents, property, condition)`** - Filter agents by condition
- **`agents.reduce(agents, property, operation)`** - Aggregate across agents (sum, mean, min, max)
- **`agents.get(agents, property)`** - Get property array
- **`agents.update(agents, property, values)`** - Update property array

#### Force Calculations
- **`agents.compute_pairwise_forces(agents, radius, force_func, mass_property)`** - N-body force calculations
  - Spatial hashing for O(n) neighbor queries (vs O(n²) brute force)
  - Configurable interaction radius
  - Custom force functions (gravity, springs, repulsion)
  - Mass-based force scaling
- Force function examples:
  - Gravitational attraction
  - Spring forces
  - Lennard-Jones potential
  - Collision avoidance

#### Field-Agent Coupling
- **`agents.sample_field(agents, field, property)`** - Sample fields at agent positions
  - Bilinear interpolation
  - Boundary handling
  - Efficient NumPy implementation
- Use cases:
  - Particles in flow fields
  - Temperature-dependent behavior
  - Density-based interactions
  - Environmental forces

#### Testing
- **85 comprehensive tests** across 4 test files:
  - `tests/test_agents_basic.py` (25 tests) - Allocation, properties, masks
  - `tests/test_agents_operations.py` (29 tests) - Map, filter, reduce
  - `tests/test_agents_forces.py` (19 tests) - Pairwise forces, field sampling
  - `tests/test_agents_integration.py` (12 tests) - Runtime integration, simulations

#### Determinism
- ✅ All operations produce identical results
- ✅ Spatial hashing deterministic
- ✅ Force calculations reproducible
- ✅ Verified through automated tests

#### Performance
- ✅ 1,000 agents: Instant allocation
- ✅ 10,000 agents: ~0.01s allocation
- ✅ Spatial hashing: O(n) neighbor queries
- ✅ NumPy vectorization throughout

#### Use Cases
- ✅ Boids flocking simulations
- ✅ N-body gravitational systems
- ✅ Particle systems
- ✅ Agent-field coupling (particles in flow)
- ✅ Crowd simulation
- ✅ SPH (Smoothed Particle Hydrodynamics) foundations

#### Runtime Integration
- Agents namespace available in Morphogen runtime
- Full integration with parser and type system
- Agents<T> type registered
- Example simulations working

#### Documentation
- Complete agent operation reference
- Flocking and N-body examples
- Performance optimization guide
- Field-agent coupling tutorials

### Implementation
- **`morphogen/stdlib/agents.py`** (569 lines of production code)
- NumPy-backed for all operations
- Spatial hashing for efficient neighbor queries
- Modular design with clear API

---

## [0.3.1] - 2025-11-14

### Added
- **Ecosystem Map Documentation** (`ECOSYSTEM_MAP.md`) - Comprehensive map of all Morphogen domains, modules, and libraries
- **Documentation Accuracy Improvements**:
  - Complete rewrite of `STATUS.md` with honest assessment of what's implemented
  - Clear distinction between production-ready, experimental, and planned features
  - Accurate MLIR status (text-based IR, not real MLIR bindings)
- **Version Consistency**: Fixed `morphogen/__init__.py` to match setup.py version (0.3.1)
- **Branding**: Consistent "Morphogen" naming throughout (replaced "Creative Computation DSL" remnants)

### Changed
- **STATUS.md**: Complete rewrite with factual, accurate status of all components
- **CHANGELOG.md**: Rewritten to reflect actual development history accurately

### Documentation
- Clarified that MLIR implementation is text-based IR generation, not production MLIR
- Explicitly noted that Audio and Agent dialects are specifications only (no implementation)
- Accurate test counts (247 tests) and coverage details
- Honest assessment of what works vs what's planned

---

## [0.3.0] - 2025-11-06

### Added - Complete v0.3.0 Syntax Features
- **Function definitions** with typed parameters: `fn add(a: f32, b: f32) -> f32 { return a + b }`
- **Lambda expressions** with full closure support: `let f = |x| x * 2`
- **If/else expressions** returning values: `if condition then value else other`
- **Enhanced flow blocks** with dt, steps, and substeps parameters
- **Struct definitions**: `struct Point { x: f32, y: f32 }`
- **Struct literals**: `Point { x: 3.0, y: 4.0 }`
- **Return statements** with early exit
- **Recursion support**
- **Higher-order functions**
- **Physical unit type annotations**

### Implementation
- Parser support for all v0.3.0 syntax features
- Runtime interpreter support for functions, lambdas, structs, if/else
- MLIR text-based IR generation for basic operations
- Comprehensive test coverage for new features

### Testing
- **Parser tests**: Full v0.3.0 syntax coverage
- **Runtime tests**: All v0.3.0 features tested
- **MLIR tests**: Text IR generation tests (72 tests)
- **Integration tests**: End-to-end examples working

### Documentation
- `SPECIFICATION.md` updated with v0.3.0 features
- `ARCHITECTURE.md` - Finalized Morphogen Stack architecture
- Architecture specifications for all core components

---

## [0.2.2] - 2025-11-05

### Added - MVP Completion: Working Field Simulations

#### Field Operations (Production-Ready)
- **`field.alloc(shape, fill_value)`** - Field allocation
- **`field.random(shape, seed, low, high)`** - Deterministic random initialization
- **`field.advect(field, velocity, dt)`** - Semi-Lagrangian advection
- **`field.diffuse(field, rate, dt, iterations)`** - Jacobi diffusion solver (20 iterations default)
- **`field.project(velocity, iterations)`** - Pressure projection for divergence-free velocity
- **`field.combine(a, b, operation)`** - Element-wise operations (add, mul, sub, div, min, max)
- **`field.map(field, func)`** - Apply functions (abs, sin, cos, sqrt, square, exp, log)
- **`field.boundary(field, spec)`** - Boundary conditions (reflect, periodic)
- **`field.laplacian(field)`** - 5-point stencil Laplacian
- **`field.gradient(field)`** - Central difference gradient
- **`field.divergence(field)`** - Divergence operator

#### Visualization (Production-Ready)
- **`visual.colorize(field, palette, vmin, vmax)`** - Scalar field to RGB with 4 palettes:
  - `grayscale` - Black → white
  - `fire` - Black → red → orange → yellow → white
  - `viridis` - Perceptually uniform, colorblind-friendly
  - `coolwarm` - Blue → white → red
- **`visual.output(visual, path, format)`** - PNG and JPEG export with Pillow
- **`visual.display(visual)`** - Interactive Pygame window
- **sRGB gamma correction** for proper display
- **Custom value range mapping** (vmin/vmax)

#### Runtime Execution
- **ExecutionContext** class with double-buffering support
- **Runtime** interpreter with full expression evaluation
- **Step-by-step execution model**
- **CLI integration**: `morphogen run <file>` command working
- **Deterministic RNG** with seeding

#### Testing
- **27 field operation tests** - All passing, determinism verified
- **23 visual operation tests** - All passing
- **Integration tests** - End-to-end pipeline working
- **Total**: 66 tests passing (100% pass rate)

#### Documentation
- **`docs/GETTING_STARTED.md`** - Complete user guide (350+ lines)
  - Installation instructions
  - First simulation walkthrough
  - Core concepts explained
  - 3 complete working examples
  - API quick reference
  - Performance tips
- **`docs/TROUBLESHOOTING.md`** - Comprehensive troubleshooting guide (400+ lines)
  - Installation issues
  - Runtime errors with solutions
  - Visualization problems
  - Performance optimization
  - Known limitations

#### Examples
- Heat diffusion simulation
- Reaction-diffusion (Gray-Scott patterns)
- Velocity field projection
- Python test demonstrating full pipeline

### Implementation Details
- **`morphogen/runtime/runtime.py`** - 398 lines of production runtime code
- **`morphogen/stdlib/field.py`** - 369 lines of NumPy-backed field operations
- **`morphogen/stdlib/visual.py`** - 217 lines of visualization code
- **NumPy backend** for all field operations
- **Pillow** for image export
- **Pygame** for interactive display

### Performance
- Field operations scale to 512×512 grids
- Parse + type-check: <100ms for typical programs
- Field operations: <1s per frame for 256×256 grid
- Jacobi solver: 20 iterations sufficient for good quality

### Determinism
- ✅ Random fields bit-identical with same seed
- ✅ All operations reproducible across runs
- ✅ No external sources of randomness
- ✅ Verified through automated tests

---

## [0.2.0] - 2025-01 (Early Development)

### Added - Language Frontend Foundation

#### Lexer
- **60+ token types** (numbers, strings, identifiers, keywords, operators)
- **Physical unit annotations**: `[m]`, `[m/s]`, `[Hz]`, `[K]`, etc.
- **Decorator syntax**: `@state`, `@param`, `@double_buffer`
- **Comment handling** (single-line with `#`)
- **Source location tracking** for error messages
- **Error reporting** with line and column numbers

#### Parser
- **Recursive descent parser** building complete AST
- **Expression parsing**: literals, identifiers, binary/unary ops, function calls, field access
- **Statement parsing**: assignments, functions, flow blocks
- **Type annotations** with physical units: `Field2D<f32 [K]>`
- **Operator precedence** (PEMDAS)
- **Error recovery** and reporting

#### Type System
- **Scalar types**: `f32`, `f64`, `i32`, `u64`, `bool`
- **Vector types**: `Vec2<f32>`, `Vec3<f32>` with physical units
- **Field types**: `Field2D<T>`, `Field3D<T>`
- **Signal types**: `Signal<T>`
- **Agent types**: `Agents<Record>`
- **Visual type**
- **Type compatibility checking**
- **Unit compatibility** (annotations only, not enforced)

#### AST
- **Expression nodes**: Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess
- **Statement nodes**: Assignment, Flow, Function, Struct
- **Type annotation nodes**
- **Decorator nodes**
- **Visitor pattern** for traversal
- **AST printer** for debugging

#### Type Checker
- **Type inference** for expressions
- **Symbol table management** with scoping
- **Type compatibility validation**
- **Unit checking** (annotation-level)
- **Error collection and reporting**

### Testing
- **18 lexer and parser tests** - All passing
- **Full coverage** of frontend components

### Project Structure
- **Package setup**: `setup.py` and `pyproject.toml`
- **CLI framework**: `morphogen` command with subcommands
- **Directory organization**: `morphogen/lexer/`, `morphogen/parser/`, `morphogen/ast/`
- **Test infrastructure**: `tests/` with pytest configuration

### Documentation
- **`README.md`** - Project overview
- **`SPECIFICATION.md`** - Complete language specification (~47KB)
- **`LANGUAGE_REFERENCE.md`** - Quick reference guide
- **`docs/architecture.md`** - Architecture overview

---

## [0.1.0] - 2025-01 (Initial Concept)

### Added - Project Initialization
- **Project concept**: Typed, deterministic DSL for creative computation
- **Initial design**: Unifying simulation, sound, visualization, and procedural design
- **Core principles**:
  - Determinism by default
  - Explicit temporal model
  - Declarative state management
  - Physical units in type system
  - Multi-domain support
- **Repository setup**: GitHub repository, license (MIT), initial documentation

---

## Upcoming (Planned)

### [0.7.0] - Real MLIR Integration (12+ months)
- Integrate actual `mlir-python-bindings`
- Implement real MLIR dialects (not text-based)
- LLVM lowering and optimization passes
- Native code generation
- GPU compilation pipeline
- Performance benchmarking vs Python interpreter

### [1.0.0] - Production Release (18-24 months)
- All dialects complete and production-ready
- Physical unit dimensional analysis enforced at runtime
- Hot-reload implementation
- Performance optimization and tuning
- Production-ready tooling and CLI
- Comprehensive examples and tutorials
- Video documentation and courses
- Community contributions and ecosystem

---

## Release Philosophy

### Version Numbering
- **Major (X.0.0)**: Significant new dialects or breaking changes
- **Minor (0.X.0)**: New features, dialect additions, significant improvements
- **Patch (0.0.X)**: Bug fixes, documentation, minor improvements

### Development Status
- **v0.1-0.3**: Language foundation (lexer, parser, type system, runtime)
- **v0.4-0.6**: Domain libraries (agents, audio, real MLIR)
- **v0.7-0.9**: Production readiness (optimization, tooling, polish)
- **v1.0+**: Stable, production-ready platform

### Honesty in Releases
All changelog entries reflect **actual implemented features**, not aspirational roadmap items. Features are marked as:
- ✅ **Production-Ready**: Fully implemented, tested, documented
- 🚧 **Experimental**: Working but not production-quality
- 📋 **Planned**: Designed but not yet implemented
- ❌ **Not Implemented**: Specification exists, no code

---

## Notes

### Rebranding (0.3.0)
- Project renamed from "Creative Computation DSL" to "Morphogen"
- More memorable, unique branding
- Reflects evolution from DSL to full creative computation platform

### MLIR Clarification (0.3.1)
- MLIR implementation is **text-based IR generation**, not real MLIR bindings
- Designed for development and validation without full LLVM build
- Real MLIR integration planned for v0.6.0

### Audio/Agent Status (0.3.1)
- **Specifications complete**: Full design documents exist
- **No implementation**: Zero code for Audio or Agent dialects
- **Intentional**: Focus on solid foundation first (fields, runtime, visualization)
- **Timeline**: Audio planned for v0.5.0, Agents for v0.4.0

---

**For detailed status of all components:** See [STATUS.md](STATUS.md)
**For architecture overview:** See [ARCHITECTURE.md](ARCHITECTURE.md)
**For ecosystem roadmap:** See [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md)
