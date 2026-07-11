# Morphogen Roadmap

**Current Version**: v0.12.0  
**Next Milestone**: v1.0  
**Updated**: 2026-04-17

> Roadmap document for direction and priorities. For current implementation status, see [STATUS.md](../STATUS.md). For recent repair work, see [PROGRESS_2026-04-17.md](PROGRESS_2026-04-17.md).

> ⚠️ **The actionable work now lives in [/BACKLOG.md](../BACKLOG.md).** Every concrete
> P0 item below is checked ✅ done; what remains here is direction/vision. The real,
> prioritized, evidence-based work list — grounded in the 2026-07-11 code-reality audit
> ([docs/reviews/2026-07-11-code-reality-and-paths-forward.md](reviews/2026-07-11-code-reality-and-paths-forward.md))
> — is in BACKLOG.md. Start there.

---

## Quick Navigation

- [Current Status](#current-status-v0120) - Where we are now
- [v1.0 - Production Release](#v10---production-release-q2-2026) - Next milestone
- [Post-v1.0 Vision](#post-v10-vision) - Future roadmap
- [Implementation Tracking](#implementation-tracking) - Domain completion status

---

## Current Status

Morphogen currently has:

- a substantial Python runtime and stdlib
- working cross-domain examples
- a broad test suite
- a DSL surface that is usable for documented introductory flows, but not yet the main product interface
- compiler/MLIR work that remains secondary to the Python API in present-day usage

This roadmap assumes the **Python-first** path remains the most practical route to v1.0.

---

## v1.0 - Production Release (Q2 2026)

**Timeline**: 24 weeks (6 months)
**Focus**: Production-ready, community-facing release

### Three-Track Strategy

#### Track 1: Language Surface
**Goal:** make the language and CLI honest, stable, and predictable

**Key Work**:
- CLI and DSL regression fixes
- clearer error messages
- keep the documented `.morph` path working
- avoid expanding language surface faster than it can be tested

**Deliverables**:
- Improved developer experience
- Better error messages
- Comprehensive language docs
- Performance benchmarks

#### Track 2: Domain And Example Polish
**Goal:** strengthen the Python API and the examples that demonstrate it

**Key Work**:
- keep canonical examples runnable
- improve cross-domain documentation
- fix broken or misleading showcase surfaces
- harden environment-sensitive areas like `visual3d`

**Deliverables**:
- Production-quality domain implementations
- Cross-domain showcase examples
- Performance benchmarks per domain
- Domain-specific tutorials

#### Track 3: Packaging And Adoption
**Goal:** make installation, onboarding, and positioning coherent

**Key Work**:
- unify packaging metadata
- complete the path to `pip install .` from a git checkout (PyPI is not a goal)
- reconcile top-level docs and status messaging
- improve beginner-to-canonical onboarding

**Deliverables**:
- `pip install .` from a git checkout working
- Polished documentation site
- 10+ showcase examples
- Community channels established
- Launch materials ready

### v1.0 Success Criteria

**Technical Requirements**:
- stable Python runtime and CLI for documented entry paths
- canonical cross-domain examples that run end-to-end
- test suite that runs reliably in common local/CI environments
- packaging/install story that is internally consistent

**User Experience**:
- `pip install .` from a git checkout (or equivalent) works cleanly
- getting started path is short and accurate
- README/docs reflect what is primary today
- examples are discoverable and trustworthy

**Community**:
- contributors can run the important tests without environment surprises
- docs point to one coherent current-state story
- contribution and packaging surfaces are low-friction

### Immediate Priorities

1. Keep the Python-first surface stable.
2. Eliminate stale or contradictory docs/status claims.
3. Finish packaging cleanup and installation validation.
4. Keep render/GUI-dependent tests opt-in or well-gated.

---

## Post-v1.0 Vision

### v1.1 (3 months post-launch)
**Focus**: Community feedback & polish

- Bug fixes from community feedback
- Performance improvements
- Documentation improvements
- Additional examples
- Plugin system foundation

### v1.2-v1.5 (6-12 months)
**Focus**: Ecosystem growth

- Additional domains (target: 50+ domains)
- Plugin ecosystem
- Visual programming interface (experimental)
- Cloud/distributed execution (experimental)
- Hardware acceleration (GPU domains)

### v2.0 (12+ months)
**Focus**: Next-generation features

- Advanced compilation (full MLIR optimization)
- Distributed computing
- Hardware synthesis (Philbrick integration)
- AI/ML domain expansion
- Industry partnerships

---

## Implementation Tracking

### Domain Completion Status

**Legend**: ✅ Complete | 🚧 In Progress | ⏳ Planned | ❌ Not Started

#### Phase 1: Foundation Domains (v0.8.0-v0.10.0)
- ✅ Field (19 ops) - Spatial fields, diffusion, advection
- ✅ Agent (13 ops) - Particle systems, agent-based modeling
- ✅ Audio (60 ops) - Synthesis, effects, DSP
- ✅ Visual (11 ops) - Rendering, visualization
- ✅ Rigidbody (12 ops) - Physics simulation
- ✅ Cellular (18 ops) - Cellular automata
- ✅ Optimization (5 ops) - Genetic algorithms, CMA-ES

#### Phase 2: Advanced Domains (v0.9.0-v0.10.0)
- ✅ Graph (19 ops) - Network analysis
- ✅ Signal (20 ops) - Signal processing, FFT
- ✅ StateMachine (15 ops) - FSM, behavior trees
- ✅ Terrain (11 ops) - Procedural generation
- ✅ Vision (13 ops) - Computer vision
- ✅ Geometry (49 ops) - 2D/3D spatial operations
- ✅ Temporal (24 ops) - Temporal logic, scheduling

#### Phase 3: Procedural Graphics (v0.11.0)
- ✅ Noise (11 ops) - Perlin, simplex, etc.
- ✅ Palette (21 ops) - Color palette generation
- ✅ Color (20 ops) - Color transformations
- ✅ Image (18 ops) - Image processing

#### Phase 4: Specialized Domains (v0.11.0)
- ✅ Acoustics (9 ops) - Acoustic modeling
- ✅ Genetic (17 ops) - Genetic algorithms
- ✅ Integrators (9 ops) - Numerical integration
- ✅ IO_Storage (10 ops) - I/O operations
- ✅ Neural (16 ops) - Neural network primitives
- ✅ Sparse_LinAlg (13 ops) - Sparse matrices
- ✅ Circuit (15 ops) - DC/AC/transient analysis

#### Phase 5: Legacy Migration (v0.12.0 - COMPLETE ✅)
- ✅ Molecular (33 ops) - Molecular dynamics
- ✅ QChem (13 ops) - Quantum chemistry
- ✅ Thermo (12 ops) - Thermodynamics
- ✅ Kinetics (11 ops) - Chemical kinetics
- ✅ Electrochem (13 ops) - Electrochemistry
- ✅ Catalysis (11 ops) - Catalysis modeling
- ✅ Transport (17 ops) - Transport properties
- ✅ Multiphase (8 ops) - Multiphase flow
- ✅ Combustion (8 ops) - Combustion modeling
- ✅ Thermal_ODE (4 ops) - Thermal dynamics
- ✅ Fluid_Network (4 ops) - Network flow
- ✅ Fluid_Jet (7 ops) - Jet dynamics
- ✅ Audio_Analysis (9 ops) - Spectral analysis
- ✅ Instrument_Model (12 ops) - Physical modeling synthesis

**Total (v0.12.0)**: broad multi-domain stdlib, substantial operator surface, and a large automated test suite

---

## Development Priorities

### P0 - Do Now (Current Sprint - Post v0.12.0)
1. ~~**Integration Examples for Phase 3 Domains**~~ ✅ COMPLETE (heating-dawn-0316)
   - ~~Create usage examples for fluid_jet, audio_analysis, instrument_model~~
   - `examples/canonical/` — 4 working cross-domain examples (physics, circuit, fluid → audio; audio_analysis → instrument), CI-guarded by `tests/test_canonical_examples_smoke.py`
   - `docs/usage/audio_analysis.md` + `docs/usage/instrument_model.md` — narrative guides

2. ~~**Test Coverage Expansion**~~ ✅ COMPLETE (2026-03-16)
   - ~~Expand audio_analysis tests (smoke → functional)~~ — 29 functional + 29 physics tests added
   - ~~Expand instrument_model tests (smoke → functional)~~ — 35 functional tests added
   - keep domain-level regression coverage healthy

3. ~~**Documentation Polish (audio_analysis + instrument_model)**~~ ✅ COMPLETE (heating-dawn-0316)
   - ~~Add narrative usage examples for audio_analysis + instrument_model~~

4. ~~**Packaging**~~ ✅ COMPLETE — `pip install .` from a git checkout works in
   a clean venv; wheel scoped to the `morphogen` package; CLI entry point
   verified (guarded by `tests/test_packaging.py`). **PyPI publication is not
   a goal** — git-based install is the distribution story for v1.0.

5. **Documentation polish**
   - Chemistry domain tutorials (remaining)
   - ~~Cross-domain coupling guide ("how typed domain interfaces work")~~ ✅
     `docs/usage/cross_domain_coupling.md` — includes the `@DomainTransform`
     shorthand with real `input_types` validation

### P1 - Next Quarter (Q1 2026 - v1.0 Prep)
1. **Performance Benchmarking**
   - Establish baseline metrics across the major domain groups
   - Domain-specific benchmarks
   - Optimization opportunities identification

2. **Developer Experience**
   - Improved error messages
   - Better documentation
   - Example gallery expansion (target: 50+ examples)

3. **Community Infrastructure**
   - Documentation site updates
   - Tutorial series expansion

### P2 - v1.0 Preparation (Q2 2026)
1. **Language Polish**
   - Type inference improvements
   - Standard library expansion
   - Cross-domain patterns

2. **Community Infrastructure**
   - Documentation site
   - Tutorial series

3. **Marketing Preparation**
   - Showcase examples
   - Launch materials
   - Community channels

---

## Resources & Links

### Documentation
- [STATUS.md](../STATUS.md) - Detailed current status
- [DOMAINS.md](DOMAINS.md) - Domain catalog
- [SPECIFICATION.md](../SPECIFICATION.md) - Language specification
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Full doc index

### Planning
- [STRATEGY.md](STRATEGY.md) - strategic framing and v1 positioning
- [ARCHIVE_HISTORY.md](ARCHIVE_HISTORY.md) - Historical documents moved to `~/Archive/morphogen/`

### Development
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [docs/guides/domain-implementation.md](guides/domain-implementation.md) - Domain dev guide
- [CHANGELOG.md](../CHANGELOG.md) - Version history

---

**Last Updated**: 2026-04-17
**Next Review**: Monthly (first week of each month)
**Maintainer**: See [CONTRIBUTING.md](../CONTRIBUTING.md)
