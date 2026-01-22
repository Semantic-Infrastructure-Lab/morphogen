# Project Planning & Strategy

This directory contains active strategic planning and roadmap documents for Morphogen.

> **Primary Roadmap**: See [/docs/ROADMAP.md](../ROADMAP.md) for the canonical project roadmap.

---

## Active Documents

### Language Features Roadmap
**[language-features.md](language-features.md)** - Language evolution v0.12 → v1.0
- Physical unit checking enhancements
- Cross-domain type safety improvements
- Module system evolution
- MLIR optimization passes
- Features under discussion (macros, effects, ownership)

**Status**: Active reference for language work

### Testing Strategy
**[testing-strategy.md](testing-strategy.md)** - Comprehensive testing approach
- Unit testing strategy
- Integration testing patterns
- Performance benchmarking framework
- Domain-specific validation
- CI/CD pipeline recommendations

**Status**: Active reference for testing work

### Music Implementation Roadmap
**[music-implementation-roadmap.md](music-implementation-roadmap.md)** - Music stack phases
- Phase 1: Feature extraction layer
- Phase 2: Symbolic music layer
- Phase 3: Structural analysis layer
- Phase 4: Compositional layer
- Phase 5: RiffStack frontend integration

**Status**: Active roadmap for music domain

### Performance & Infrastructure Research
**[PERFORMANCE_INFRASTRUCTURE_RESEARCH.md](PERFORMANCE_INFRASTRUCTURE_RESEARCH.md)** - SIL Tech & Optimization
- MLIR current status and optimization paths
- GPU acceleration options (Taichi, JAX, Mojo, Triton)
- Pantheon/Prism integration possibilities
- Morphogen architecture bottlenecks and solutions

**Status**: Complete reference document (created 2026-01-05)

---

## Related Documentation

- **Unified Roadmap**: [/docs/ROADMAP.md](../ROADMAP.md) - Overall project roadmap
- **Current Status**: [/STATUS.md](../../STATUS.md) - Implementation status
- **ADRs**: [/docs/adr/](../adr/) - Architecture decisions
- **Specifications**: [/docs/specifications/](../specifications/) - Technical specs
- **Music Roadmap**: [music-implementation-roadmap.md](music-implementation-roadmap.md) - Music stack plans

---

## Archived Planning Documents

Historical planning documents in `~/Archive/morphogen/docs-archive/2025-12-06/planning/`:
- `3D_VISUALIZATION_SYSTEM_PLAN.md` - 3D viz roadmap (COMPLETE, archived 2026-01-22)
- `KAIRO_2.0_STRATEGIC_ANALYSIS.md` - v2.0+ language strategy
- `ROADMAP_2025_Q4.md` - Q4 2025 tactical plan
- And other historical documents

---

## How to Use

| Need | Document |
|------|----------|
| Current priorities | [/docs/ROADMAP.md](../ROADMAP.md) |
| Language features | [language-features.md](language-features.md) |
| Testing approach | [testing-strategy.md](testing-strategy.md) |
| Performance optimization | [PERFORMANCE_INFRASTRUCTURE_RESEARCH.md](PERFORMANCE_INFRASTRUCTURE_RESEARCH.md) |
| Music stack | [music-implementation-roadmap.md](music-implementation-roadmap.md) |

---

**Directory Structure**:
```
docs/planning/
├── README.md                              ← You are here
├── language-features.md                   ← Language evolution roadmap
├── testing-strategy.md                    ← Testing strategy
├── music-implementation-roadmap.md        ← Music stack roadmap
└── PERFORMANCE_INFRASTRUCTURE_RESEARCH.md ← Performance & SIL tech research
```

---

*Last Updated: 2026-01-22*
