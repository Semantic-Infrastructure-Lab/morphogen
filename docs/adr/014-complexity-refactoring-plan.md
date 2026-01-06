# ADR-014: Complexity Refactoring Plan

**Status**: Accepted
**Date**: 2026-01-05
**Author**: TIA (with Scott)
**Sessions**: roaring-sunlight-0105 (initial), continuing across multiple sessions

## Context

Morphogen has accumulated significant technical debt in its core modules. Code quality analysis reveals:

| File | Lines | Issues | Token Cost | Primary Problems |
|------|-------|--------|------------|------------------|
| `stdlib/audio.py` | 3,105 | 44 | ~82K | Monolithic, 77 functions in one class |
| `stdlib/geometry.py` | 2,411 | 16 | ~52K | File size, property methods with logic |
| `cross_domain/interface.py` | 1,971 | 23 | ~48K | Circular dep, duplicate methods, 22 classes |
| `stdlib/molecular.py` | 1,598 | 28 | ~35K | Deep nesting (7 levels), large functions |
| `stdlib/visual.py` | 1,559 | 47 | ~44K | `agents()` 271 lines, complexity 35 |
| `stdlib/agents.py` | 1,282 | 28 | ~35K | `emit()` 174 lines, complexity 26 |
| `runtime/runtime.py` | 1,005 | 12 | ~26K | Complex dispatch, deep nesting |

**Total identified issues**: 198
**Estimated LLM token cost for full codebase read**: ~320K tokens

## Decision

Execute a 6-phase refactoring plan prioritized by impact:

1. Split `audio.py` into domain submodules (highest impact)
2. Split `interface.py` and fix circular dependency
3. Config objects for `visual.py` functions
4. Dispatch tables in `runtime.py`
5. Flatten `molecular.py` nesting
6. Deduplicate interface methods and quick wins

## Phases

---

### Phase 1: Split `audio.py` into Domain Submodules

**Goal**: Transform 3,105-line monolith into ~10 focused modules of 200-400 lines each.

**Target Structure**:
```
stdlib/audio/
├── __init__.py          # Re-exports for API compatibility
├── buffer.py            # AudioBuffer class
├── oscillators.py       # sine, saw, square, triangle, noise, impulse, constant
├── filters.py           # lowpass, highpass, bandpass, notch, eq3, vcf_*, biquad helpers
├── envelopes.py         # adsr, ar, envexp
├── effects.py           # delay, reverb, chorus, flanger, drive, limiter
├── mixing.py            # mix, gain, multiply, vca, pan, clip, normalize, db2lin, lin2db
├── synthesis.py         # string, modal (physical modeling)
├── io.py                # play, save, load, record
├── analysis.py          # fft, stft, istft, spectrum, phase_spectrum, spectral_*, rms, zero_crossings
└── transform.py         # slice, concat, resample, reverse, fade_in, fade_out
```

**Function Mapping**:

| New Module | Functions | Lines Est. |
|------------|-----------|------------|
| `buffer.py` | AudioBuffer class | ~60 |
| `oscillators.py` | sine, saw, square, triangle, noise, impulse, constant | ~280 |
| `filters.py` | lowpass, highpass, bandpass, notch, eq3, vcf_lowpass, vcf_highpass, vcf_bandpass, _biquad_*, _apply_time_varying_* | ~650 |
| `envelopes.py` | adsr, ar, envexp | ~140 |
| `effects.py` | delay, reverb, chorus, flanger, drive, limiter | ~270 |
| `mixing.py` | mix, gain, multiply, vca, pan, clip, normalize, db2lin, lin2db | ~280 |
| `synthesis.py` | string, modal, _apply_iir_filter | ~190 |
| `io.py` | play, save, load, record | ~260 |
| `analysis.py` | fft, ifft, stft, istft, spectrum, phase_spectrum, spectral_centroid, spectral_rolloff, spectral_flux, spectral_peaks, rms, zero_crossings, spectral_gate, spectral_filter, convolution | ~500 |
| `transform.py` | slice, concat, resample, reverse, fade_in, fade_out | ~200 |

**API Compatibility**: The `__init__.py` will re-export all functions so existing code continues to work:
```python
from .buffer import AudioBuffer
from .oscillators import sine, saw, square, triangle, noise, impulse, constant
from .filters import lowpass, highpass, bandpass, notch, eq3, vcf_lowpass, ...
# etc.
```

**Success Criteria**:
- [ ] All 77 functions moved to appropriate submodules
- [ ] `from morphogen.stdlib.audio import *` still works
- [ ] All existing tests pass
- [ ] reveal --check shows reduced issues per file
- [ ] No file exceeds 700 lines

---

### Phase 2: Split `interface.py` and Fix Circular Dependency

**Goal**: Break up 22-class monolith, eliminate circular import.

**Current Problem**:
```
registry.py → interface.py → registry.py (CIRCULAR)
```

**Target Structure**:
```
cross_domain/
├── __init__.py          # Public API
├── base.py              # DomainInterface, DomainMetadata, DomainTransform (shared)
├── registry.py          # CrossDomainRegistry (imports from base, not interface)
├── field_agent.py       # FieldToAgentInterface, AgentToFieldInterface
├── audio_visual.py      # AudioToVisualInterface, FieldToAudioInterface
├── physics_audio.py     # PhysicsToAudioInterface, FluidToAcousticsInterface, AcousticsToAudioInterface
├── terrain.py           # TerrainToFieldInterface, FieldToTerrainInterface
├── cellular.py          # CellularToFieldInterface
├── graph_visual.py      # GraphToVisualInterface
├── vision_field.py      # VisionToFieldInterface
├── spectral.py          # TimeToCepstralInterface, CepstralToTimeInterface, TimeToWaveletInterface
└── spatial.py           # SpatialAffineInterface, CartesianToPolarInterface, PolarToCartesianInterface
```

**Circular Dependency Fix**:
1. Extract `DomainInterface`, `DomainMetadata`, `DomainTransform` to `base.py`
2. `registry.py` imports only from `base.py`
3. Interface implementations import from `base.py` and register via decorator

**Success Criteria**:
- [ ] No circular import warnings
- [ ] `python -c "from morphogen.cross_domain import *"` works
- [ ] All 22 interface classes accessible
- [ ] Duplicate `get_input/output_interface` methods consolidated

---

### Phase 3: Config Objects for `visual.py` Functions

**Goal**: Reduce parameter explosion, improve composability.

**Problem Functions**:
| Function | Parameters | Lines | Complexity |
|----------|------------|-------|------------|
| `agents()` | 18 | 271 | 35 |
| `phase_space()` | 11 | 134 | 14 |
| `graph()` | 13 | 109 | 12 |
| `add_metrics()` | 7 | 105 | 18 |

**Solution Pattern**:
```python
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class AgentRenderConfig:
    """Configuration for agent visualization."""
    width: int = 512
    height: int = 512
    position_property: str = 'pos'
    color_property: Optional[str] = None
    size_property: Optional[str] = None
    alpha_property: Optional[str] = None
    rotation_property: Optional[str] = None
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: float = 2.0
    alpha: float = 1.0
    palette: str = "viridis"
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    blend_mode: str = "alpha"
    trail: bool = False
    trail_length: int = 10
    trail_alpha: float = 0.5
```

**Helper Extraction**:
```python
def _render_agent_points(img, positions, colors, sizes, alphas, config) -> None
def _render_agent_trails(img, trail_history, config) -> None
def _apply_blend_mode(base, overlay, mode) -> np.ndarray
def _compute_agent_colors(agents, config) -> np.ndarray
def _compute_agent_sizes(agents, config) -> np.ndarray

def agents(agents, config: AgentRenderConfig = None, **kwargs) -> Visual:
    """Render agents to a visual. Accepts config object or keyword args."""
    if config is None:
        config = AgentRenderConfig(**kwargs)
    # 30-40 line orchestration
```

**Success Criteria**:
- [ ] `agents()` reduced to <50 lines, complexity <10
- [ ] `phase_space()` reduced to <50 lines
- [ ] `graph()` reduced to <50 lines
- [ ] `add_metrics()` reduced to <50 lines
- [ ] Config objects documented with docstrings
- [ ] Backward compatibility preserved (kwargs still work)

---

### Phase 4: Dispatch Tables in `runtime.py`

**Goal**: Replace branching with data-driven dispatch.

**Problem**:
```python
# Current: 15 branches in execute_statement
def execute_statement(self, stmt):
    if isinstance(stmt, Assignment):
        return self.execute_assignment(stmt)
    elif isinstance(stmt, FlowStatement):
        return self.execute_flow(stmt)
    elif isinstance(stmt, IfStatement):
        return self.execute_if(stmt)
    # ... 12 more branches
```

**Solution**:
```python
from typing import Dict, Type, Callable

class MorphogenRuntime:
    _STATEMENT_HANDLERS: Dict[Type, str] = {
        Assignment: 'execute_assignment',
        FlowStatement: 'execute_flow',
        IfStatement: 'execute_if',
        WhileStatement: 'execute_while',
        ForStatement: 'execute_for',
        ReturnStatement: 'execute_return',
        BreakStatement: 'execute_break',
        ContinueStatement: 'execute_continue',
        UseStatement: 'execute_use',
        StructDefinition: 'execute_struct_def',
        FunctionDefinition: 'execute_function_def',
        DomainBlock: 'execute_domain_block',
        ExpressionStatement: 'execute_expression_stmt',
    }

    _EXPRESSION_HANDLERS: Dict[Type, str] = {
        BinaryOp: 'execute_binary_op',
        UnaryOp: 'execute_unary_op',
        FunctionCall: 'execute_function_call',
        FieldAccess: 'execute_field_access',
        IndexAccess: 'execute_index_access',
        Literal: 'execute_literal',
        Identifier: 'execute_identifier',
        ListLiteral: 'execute_list_literal',
        StructLiteral: 'execute_struct_literal',
        LambdaExpr: 'execute_lambda',
    }

    def execute_statement(self, stmt) -> Any:
        handler_name = self._STATEMENT_HANDLERS.get(type(stmt))
        if handler_name is None:
            raise RuntimeError(f"Unknown statement type: {type(stmt).__name__}")
        return getattr(self, handler_name)(stmt)

    def execute_expression(self, expr) -> Any:
        handler_name = self._EXPRESSION_HANDLERS.get(type(expr))
        if handler_name is None:
            raise RuntimeError(f"Unknown expression type: {type(expr).__name__}")
        return getattr(self, handler_name)(expr)
```

**Success Criteria**:
- [ ] `execute_statement` complexity reduced from 15 → 3
- [ ] `execute_expression` complexity reduced from 12 → 3
- [ ] All runtime tests pass
- [ ] Handler methods unchanged (only dispatch logic changes)

---

### Phase 5: Flatten `molecular.py` Nesting

**Goal**: Reduce max nesting depth from 7 → 4.

**Problem Functions**:
| Function | Nesting | Complexity |
|----------|---------|------------|
| `rdf()` | 7 | - |
| `hydrogen_bonds()` | 7 | - |
| `cluster_conformers()` | 6 | - |
| `generate_conformers()` | 5 | - |
| `load_pdb()` | 5 | - |

**Solution Pattern** (for `rdf()`):
```python
# Current: deeply nested loops
def rdf(trajectory, atom_type_1, atom_type_2, r_max, n_bins):
    for frame in trajectory:
        for i, atom1 in enumerate(frame.atoms):
            if atom1.type == atom_type_1:
                for j, atom2 in enumerate(frame.atoms):
                    if j > i and atom2.type == atom_type_2:
                        distance = compute_distance(atom1, atom2)
                        for bin_idx in range(n_bins):
                            if lower <= distance < upper:
                                histogram[bin_idx] += 1  # depth 7!

# Refactored: helper functions + vectorization
def _get_atom_indices_by_type(atoms, atom_type: str) -> np.ndarray:
    """Return indices of atoms matching type."""
    return np.array([i for i, a in enumerate(atoms) if a.type == atom_type])

def _compute_pairwise_distances(positions: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """Compute distances between all pairs of atoms from two index sets."""
    pos1 = positions[idx1]
    pos2 = positions[idx2]
    # Vectorized distance computation
    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1)).flatten()

def _histogram_distances(distances: np.ndarray, r_max: float, n_bins: int) -> np.ndarray:
    """Bin distances into histogram."""
    return np.histogram(distances, bins=n_bins, range=(0, r_max))[0]

def rdf(trajectory, atom_type_1, atom_type_2, r_max=10.0, n_bins=100):
    """Compute radial distribution function."""
    histograms = []
    for frame in trajectory:
        idx1 = _get_atom_indices_by_type(frame.atoms, atom_type_1)
        idx2 = _get_atom_indices_by_type(frame.atoms, atom_type_2)
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        distances = _compute_pairwise_distances(frame.positions, idx1, idx2)
        # Exclude self-pairs if same type
        if atom_type_1 == atom_type_2:
            distances = distances[distances > 0]
        histograms.append(_histogram_distances(distances, r_max, n_bins))

    return _normalize_rdf(np.mean(histograms, axis=0), r_max, n_bins, ...)
```

**Success Criteria**:
- [ ] All functions with nesting >4 reduced to ≤4
- [ ] Helper functions are reusable across molecular analysis
- [ ] Tests pass with identical results
- [ ] Performance maintained or improved (vectorization helps)

---

### Phase 6: Deduplicate Interface Methods and Quick Wins

**Goal**: Clean up remaining issues.

**6a. Deduplicate Interface Methods**

Create mixin for standard interface patterns:
```python
class ArrayIOInterface:
    """Mixin for interfaces with standard np.ndarray I/O."""
    def get_input_interface(self) -> Dict[str, Type]:
        return {"data": np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {"data": np.ndarray}

class VisionToFieldInterface(DomainInterface, ArrayIOInterface):
    # Inherits get_input/output_interface
    pass
```

**6b. Remove Unused Imports**

| File | Unused Imports |
|------|----------------|
| `audio.py` | `Callable`, `Dict`, `Any`, `Union` |
| `molecular.py` | `Union`, `Enum` |
| `runtime.py` | `Callable`, `numpy` |

**6c. Fix Long Properties in `geometry.py`**

Convert 18-line `@property` methods to regular methods:
```python
# Before
@property
def area(self) -> float:
    # 18 lines of computation

# After
def compute_area(self) -> float:
    # Same implementation

@property
def area(self) -> float:
    return self.compute_area()  # Cache if needed
```

**Success Criteria**:
- [ ] No duplicate function warnings in reveal --check
- [ ] All unused imports removed
- [ ] Property methods ≤8 lines
- [ ] Total issues reduced from 198 to <50

---

## Execution Tracking

### Session Log

| Session ID | Date | Phase | Work Completed |
|------------|------|-------|----------------|
| roaring-sunlight-0105 | 2026-01-05 | Phase 1 | Split audio.py (3105 lines) into 10 submodules. Created: buffer.py, helpers.py, oscillators.py, filters.py, envelopes.py, effects.py, mixing.py, synthesis.py, io.py, transform.py, analysis.py, __init__.py. All imports verified, backward compatibility maintained via AudioOperations class. |
| passing-dusk-0105 | 2026-01-05 | Phase 2 | Split interface.py (1971 lines) into 10 submodules + base.py. Fixed circular dependency by extracting DomainInterface/DomainTransform to base.py. Created: base.py, field_agent.py, audio_visual.py, physics_audio.py, terrain.py, vision_field.py, graph_visual.py, cellular.py, spectral.py, spatial.py. Updated registry.py and composer.py imports. All 18 transforms registered, functional tests passing. |
| onyx-twilight-0105 | 2026-01-05 | Phase 3 | Refactored visual.py with config dataclasses and helper functions. Added AgentRenderConfig, PhaseSpaceConfig, GraphConfig, MetricsConfig. agents() reduced from 271→118 lines (complexity 35→14). phase_space() 134→80 lines. graph() 109→83 lines. add_metrics() 105→67 lines. display() 169→100 lines. Fixed pre-existing Graph API bugs (n_nodes→num_nodes, adj→adjacency_list). All 8 functional tests passing. File reduced 1559→1478 lines. |
| solar-constellation-0105 | 2026-01-05 | Phase 4 | Refactored runtime.py with dispatch tables and helper extraction. execute_statement: 48→32 lines (isinstance chain→dispatch table). execute_expression: 50→27 lines (dispatch table). execute_flow: 79→25 lines, depth 7→2 (extracted _init_flow_state_vars, _execute_flow_body, _execute_flow_iteration). execute_use: depth 7→3 (extracted _import_domain_exports, moved domain_modules to class constant). Removed unused imports (Callable, numpy). Fixed audio package export (Phase 1 leftover). Issues reduced 12→2 (remaining: file size advisory, unrelated line length). All 17 runtime tests passing. File 1005→992 lines. |
| solar-constellation-0105 | 2026-01-05 | Phase 5 | Refactored molecular.py nesting. rdf: depth 7→4 (extracted _accumulate_rdf_histogram). hydrogen_bonds: depth 7→5 (extracted _is_hydrogen_bond). cluster_conformers: depth 6→3 (extracted _avg_cluster_distance, _find_closest_clusters). Removed unused imports (Union, Enum). Issues reduced 28→24. Remaining depth 5 issues in load_pdb, generate_conformers, hydrogen_bonds are acceptable (triple nested loops fundamental to algorithms). All 12 molecular tests passing (10 skipped - unimplemented features). |
| solar-constellation-0105 | 2026-01-05 | Phase 6 | Quick wins in geometry.py. Fixed long @property methods: area (18→4 lines, extracted _compute_area), perimeter (15→4 lines, extracted _compute_perimeter). Issues reduced 16→14. Note: num_vertices "duplicate" is valid polymorphism (Polygon and Mesh3D both correctly have this property). Interface deduplication deferred - methods are semantically distinct (different key names/types per domain). All 140 tests passing across runtime/molecular/geometry. |
| | | | |

### Verification Commands

```bash
# Check current status of any file
reveal /home/scottsen/src/projects/morphogen/morphogen/stdlib/audio.py --check

# Run all tests
cd /home/scottsen/src/projects/morphogen && python -m pytest tests/ -v

# Quick validation after changes
python -c "from morphogen.stdlib import audio; print('audio OK')"
python -c "from morphogen.cross_domain import *; print('cross_domain OK')"
```

---

## Consequences

### Positive
- 25x reduction in LLM token cost for module exploration
- Clearer mental model for contributors
- Independent testability of submodules
- Easier parallel development
- Reduced cognitive load per file

### Negative
- One-time migration effort
- Import paths change internally (though public API preserved)
- Need to update internal cross-references

### Risks
- Circular imports during transition (mitigate: careful ordering)
- Test breakage (mitigate: run tests after each step)
- API compatibility issues (mitigate: preserve public interface)

---

## References

- Previous refactoring: ripaka-0105, hidden-void-0105 (visual3d.py, compiler.py)
- Reveal complexity checks: `reveal <file> --check`
- Test suite: `python -m pytest tests/ -v`
