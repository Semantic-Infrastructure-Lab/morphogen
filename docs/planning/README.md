# Project Planning & Strategy

This directory contains active strategic planning documents for Morphogen.

> **NOTE**: Most planning documents have been consolidated into [/docs/ROADMAP.md](../ROADMAP.md). See [archive/planning/](../archive/planning/) for historical planning documents.

---

## 📍 Current Active Documents

### Primary Roadmap
**[/docs/ROADMAP.md](../ROADMAP.md)** - **Unified Roadmap (CANONICAL)** 🎯
- **Status**: ✅ Active - Single source of truth
- **Scope**: v0.11.0 → v0.12.0 → v1.0 (Q2 2026)
- **Content**: Current status, migration plans, v1.0 roadmap, implementation tracking
- **Last Updated**: 2025-12-07
- **Use**: For all current planning, priorities, and execution tracking

### 3D Visualization System
**[3D_VISUALIZATION_SYSTEM_PLAN.md](3D_VISUALIZATION_SYSTEM_PLAN.md)** - Comprehensive 3D Viz Roadmap
- **Status**: ✅ COMPLETE - All phases implemented
- **Scope**: Phases 1-5 complete (26 operators, 55 tests)
- **Content**: PyVista backend, mesh/volume/molecular viz, camera animation, streamlines
- **Completed**: 2026-01-05
- **API Reference**: [visual3d-quickref.md](../reference/visual3d-quickref.md)

### Performance & Infrastructure Research
**[PERFORMANCE_INFRASTRUCTURE_RESEARCH.md](PERFORMANCE_INFRASTRUCTURE_RESEARCH.md)** - SIL Tech & Optimization Paths
- **Status**: ✅ Complete - Reference document
- **Scope**: MLIR, GPU acceleration, Pantheon/Prism integration
- **Content**: Technology evaluation (Taichi, JAX, Mojo), Morphogen GPU-readiness analysis
- **Created**: 2026-01-05 (session pearl-ray-0105)
- **Use**: When optimizing 3D viz performance or integrating with SIL infrastructure

### Language Evolution Strategy
**[KAIRO_2.0_STRATEGIC_ANALYSIS.md](KAIRO_2.0_STRATEGIC_ANALYSIS.md)** - v2.0+ Language Strategy
- **Status**: 📚 Reference - Future planning
- **Scope**: Post-v1.0 language evolution
- **Content**: Gap analysis, architectural requirements, v2.0 feature set
- **Use**: When planning major language evolution (v1.5+, v2.0)

---

## 📚 Archived Planning Documents

Historical planning documents have been moved to `~/Archive/morphogen/docs-archive/2025-12-06/planning/` including:

- `ROADMAP_2025_Q4.md` - Q4 2025 tactical plan (superseded)
- `MORPHOGEN_RELEASE_PLAN.md` - Original v1.0 plan (consolidated)
- `PATH_FORWARD_OVERVIEW.md` - Long-term vision (consolidated)
- `DOMAIN_VALUE_ANALYSIS.md` - Domain ROI analysis (historical reference)
- `ARCHITECTURAL_EVOLUTION_ROADMAP.md` - Architecture evolution (historical)

**Why archived**: Information consolidated into unified roadmap for maintainability.

---

## How to Use These Documents

### For New Contributors
1. **Start here**: [/docs/ROADMAP.md](../ROADMAP.md) - Current priorities and plans
2. Review [/STATUS.md](../../STATUS.md) - Current implementation status
3. Check [/docs/roadmap/](../roadmap/) - Feature-specific roadmaps
4. Reference [KAIRO_2.0_STRATEGIC_ANALYSIS.md](KAIRO_2.0_STRATEGIC_ANALYSIS.md) for future vision

### For Planning Work
- **Current quarter**: Use [/docs/ROADMAP.md](../ROADMAP.md)
- **Future features**: Check [../roadmap/](../roadmap/) (language-features.md, testing-strategy.md)
- **Long-term vision**: See KAIRO_2.0_STRATEGIC_ANALYSIS.md
- **Historical context**: See `~/Archive/morphogen/docs-archive/2025-12-06/planning/`

### For Project Maintainers
- **ROADMAP.md is the single source of truth** for current planning
- Update it monthly (first week of each month)
- Archive superseded planning docs to keep directory clean
- Document major decisions in [/docs/adr/](../adr/) (Architecture Decision Records)

---

## Related Documentation

- **Status**: [/STATUS.md](../../STATUS.md) - Current implementation status
- **Unified Roadmap**: [/docs/ROADMAP.md](../ROADMAP.md) - Primary planning doc
- **Feature Roadmaps**: [/docs/roadmap/](../roadmap/) - Feature-specific plans
- **ADRs**: [/docs/adr/](../adr/) - Architecture decisions
- **Guides**: [/docs/guides/](../guides/) - Implementation guides

---

**Directory Structure**:
```
docs/planning/
├── README.md                                    ← You are here
├── 3D_VISUALIZATION_SYSTEM_PLAN.md             ← COMPLETE: 3D viz (5 phases, 26 operators)
├── PERFORMANCE_INFRASTRUCTURE_RESEARCH.md       ← Reference: Performance & SIL tech research
├── KAIRO_2.0_STRATEGIC_ANALYSIS.md             ← Reference: v2.0+ language strategy
└── (historical planning docs in ~/Archive/)
```

**For the canonical roadmap**, see: [/docs/ROADMAP.md](../ROADMAP.md)

---

*Last Updated: 2026-01-05*
*Canonical Roadmap: /docs/ROADMAP.md*
