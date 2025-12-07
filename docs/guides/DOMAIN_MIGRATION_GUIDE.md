# Domain Migration Guide

**Target Version**: v0.12.0
**Status**: Planning
**Last Updated**: 2025-12-06

## Overview

Morphogen has **40 implemented domains**, but only **25 are currently accessible** via the `use` statement. The remaining **15 legacy domains** use an older class-based pattern and need migration to the new `@operator` decorator system.

## Current Situation

### Active Domains (25) ✅
These domains are fully integrated:
- Accessible via `use domain_name` in `.kairo` files
- Operators discoverable via `DomainRegistry`
- Full type safety and validation

### Legacy Domains (15) ⏳
These domains are implemented but not integrated:
- Code exists in `morphogen/stdlib/`
- Functions work when called from Python
- **Cannot** be used in `.kairo` programs
- Missing `@operator` decorators

## Why Migration is Needed

The legacy domains use a class-based wrapper pattern:

```python
# Legacy pattern (NOT discoverable)
class ThermalODEOperations:
    @staticmethod
    def heat_transfer_1D(...):
        return heat_transfer_1D(...)

def heat_transfer_1D(segment, m_dot, T_in, ...):
    """Actual implementation."""
    # ... implementation

thermal_ode = ThermalODEOperations()
```

The new pattern uses decorators for automatic discovery:

```python
# Modern pattern (DISCOVERABLE)
from morphogen.core.operator import operator, OpCategory

@operator(
    domain="thermal_ode",
    category=OpCategory.INTEGRATE,
    signature="(segment: ThermalSegment, m_dot: float, T_in: float, ...) -> float",
    deterministic=True,
    doc="1D heat transfer with convection"
)
def heat_transfer_1D(segment, m_dot, T_in, ...):
    """Actual implementation."""
    # ... implementation
```

## Migration Requirements

For each legacy domain:

### 1. Add @operator Decorators

Each function needs full metadata:

```python
@operator(
    domain="domain_name",           # Domain name (e.g., "molecular")
    category=OpCategory.XXX,        # CONSTRUCT, TRANSFORM, QUERY, etc.
    signature="(...) -> ReturnType", # Type signature
    deterministic=True/False,       # Reproducibility
    doc="Description",              # Documentation
    pure=True/False,                # Has side effects?
    stateful=False/True,            # Maintains state?
    vectorized=False/True           # Works on batches?
)
def function_name(...):
    pass
```

**OpCategory options:**
- `CONSTRUCT` - Create new data structures
- `TRANSFORM` - Transform existing data
- `QUERY` - Query/analyze data
- `INTEGRATE` - Time-stepping/integration
- `COMPOSE` - Combine multiple elements
- `MUTATE` - In-place modification
- `SAMPLE` - Sample/resample data
- `RENDER` - Render/visualize data

### 2. Register in Domain Registry

Add to `morphogen/core/domain_registry.py` (lines 195-221):

```python
("molecular", "morphogen.stdlib.molecular", "Molecular dynamics and force fields"),
("thermal_ode", "morphogen.stdlib.thermal_ode", "1D thermal modeling"),
# ... etc
```

### 3. Add Tests

Create `tests/test_domain_name.py` with:
- Basic operator smoke tests
- Integration tests for common workflows
- Edge case validation

## Migration Priority

### Phase 1 - High Value (Week 1)
Well-defined, small, tested domains:

1. **molecular** (33 functions, HAS TESTS) - Molecular dynamics, force fields
2. **thermal_ode** (4 functions) - Thermal modeling for heat exchangers
3. **fluid_network** (4 functions) - 1D flow networks

**Estimated effort:** 16-24 hours

### Phase 2 - Chemistry Suite (Week 2-3)
Complete chemistry integration:

4. **thermo** (12 functions) - Thermodynamics, equations of state
5. **kinetics** (11 functions) - Chemical kinetics, reaction rates
6. **electrochem** (13 functions) - Electrochemistry, Nernst equation
7. **qchem** (13 functions) - Quantum chemistry, electronic structure
8. **catalysis** (11 functions) - Catalytic cycles, mechanisms
9. **transport** (17 functions) - Transport properties (diffusion, viscosity)
10. **multiphase** (8 functions) - Multiphase flow, mass transfer
11. **combustion** (8 functions) - Combustion kinetics, flame dynamics

**Estimated effort:** 40-60 hours

### Phase 3 - Specialized (Week 4)

12. **audio_analysis** (9 functions) - Spectral analysis, onset detection
13. **instrument_model** (10 functions) - Physical modeling synthesis
14. **fluid_jet** (8 functions) - Jet dynamics, turbulence
15. **flappy** (5 functions) - Educational game demo

**Estimated effort:** 16-24 hours

**Total Estimated Effort:** 72-108 hours (~2-3 weeks full-time)

## Migration Workflow

### For Each Domain:

1. **Analyze Functions**
   ```bash
   python tools/analyze_domain.py morphogen/stdlib/molecular.py
   ```
   Output: List of functions with suggested categories

2. **Add Decorators**
   Manually add `@operator` to each function with proper metadata

3. **Test Import**
   ```python
   from morphogen.core.domain_registry import DomainRegistry
   DomainRegistry.register("molecular", "morphogen.stdlib.molecular", "...")
   desc = DomainRegistry.get("molecular")
   print(f"Operators: {desc.list_operators()}")
   ```

4. **Write Tests**
   ```bash
   # Create test file
   cp tests/test_template.py tests/test_molecular.py
   # Add operator tests
   pytest tests/test_molecular.py -v
   ```

5. **Register Domain**
   Add to `morphogen/core/domain_registry.py`

6. **Integration Test**
   ```python
   # Test in .kairo file
   use molecular

   @state atoms : Atoms = create_water_molecule()
   flow(dt=0.001) {
       atoms = integrate_forces(atoms, dt)
   }
   ```

## Tools

### analyze_domain.py
Analyzes a domain file and suggests decorator metadata:
```bash
python tools/analyze_domain.py morphogen/stdlib/molecular.py
```

Output:
```
molecular.py - 33 functions

create_molecule() -> Suggested: CONSTRUCT
integrate_forces() -> Suggested: INTEGRATE
compute_energy() -> Suggested: QUERY
...
```

### migrate_decorators.py (Semi-automated)
Assists with decorator addition:
```bash
python tools/migrate_decorators.py morphogen/stdlib/molecular.py --interactive
```

Prompts for each function:
```
Function: create_molecule
  Category [CONSTRUCT/TRANSFORM/QUERY/...]: CONSTRUCT
  Signature: (atoms: List[Atom], bonds: List[Bond]) -> Molecule
  Deterministic [Y/n]: Y
  Pure [Y/n]: Y

  Generated decorator:
  @operator(
      domain="molecular",
      category=OpCategory.CONSTRUCT,
      signature="(atoms: List[Atom], bonds: List[Bond]) -> Molecule",
      deterministic=True,
      pure=True
  )

  Add this decorator? [Y/n]:
```

## Testing Requirements

Each migrated domain must have:

### 1. Smoke Tests
Verify each operator can be called:
```python
def test_heat_transfer_1D_basic():
    segment = ThermalSegment(...)
    result = heat_transfer_1D(segment, m_dot=0.1, T_in=300.0, ...)
    assert result > 0
```

### 2. Integration Tests
Test realistic workflows:
```python
def test_thermal_network_simulation():
    # Create multi-segment network
    # Simulate heat transfer
    # Verify energy conservation
    pass
```

### 3. Determinism Tests
Verify reproducibility:
```python
def test_molecular_dynamics_deterministic():
    result1 = run_md_simulation(seed=42, steps=100)
    result2 = run_md_simulation(seed=42, steps=100)
    assert np.allclose(result1, result2)
```

## Success Criteria

A domain is successfully migrated when:

1. ✅ All functions have `@operator` decorators
2. ✅ Domain is registered in `domain_registry.py`
3. ✅ `DomainRegistry.get("domain_name")` returns descriptor
4. ✅ All operators listed in descriptor
5. ✅ `use domain_name` works in `.kairo` files
6. ✅ Test file exists with >80% coverage
7. ✅ All tests passing
8. ✅ Documentation updated

## Progress Tracking

| Domain | Functions | Decorated | Registered | Tests | Status |
|--------|-----------|-----------|------------|-------|--------|
| molecular | 33 | 0/33 | ❌ | ✅ Exist | ⏳ Not Started |
| thermal_ode | 4 | 0/4 | ❌ | ❌ None | ⏳ Not Started |
| fluid_network | 4 | 0/4 | ❌ | ❌ None | ⏳ Not Started |
| thermo | 12 | 0/12 | ❌ | ❌ None | ⏳ Not Started |
| kinetics | 11 | 0/11 | ❌ | ❌ None | ⏳ Not Started |
| electrochem | 13 | 0/13 | ❌ | ❌ None | ⏳ Not Started |
| qchem | 13 | 0/13 | ❌ | ❌ None | ⏳ Not Started |
| catalysis | 11 | 0/11 | ❌ | ❌ None | ⏳ Not Started |
| transport | 17 | 0/17 | ❌ | ❌ None | ⏳ Not Started |
| multiphase | 8 | 0/8 | ❌ | ❌ None | ⏳ Not Started |
| combustion | 8 | 0/8 | ❌ | ❌ None | ⏳ Not Started |
| audio_analysis | 9 | 0/9 | ❌ | ❌ None | ⏳ Not Started |
| instrument_model | 10 | 0/10 | ❌ | ❌ None | ⏳ Not Started |
| fluid_jet | 8 | 0/8 | ❌ | ❌ None | ⏳ Not Started |
| flappy | 5 | 0/5 | ❌ | ❌ None | ⏳ Not Started |

**Total:** 0/166 functions migrated (0%)

## Timeline

**Target Release:** v0.12.0 (2026-Q1)

- **Week 1-2:** Phase 1 (molecular, thermal_ode, fluid_network)
- **Week 3-5:** Phase 2 (chemistry suite - 7 domains)
- **Week 6:** Phase 3 (specialized domains - 4 domains)
- **Week 7:** Integration testing, documentation, release prep

## Resources

- **Example Migration:** See `morphogen/stdlib/field.py` for reference implementation
- **Operator Metadata Spec:** `morphogen/core/operator.py`
- **Domain Registry:** `morphogen/core/domain_registry.py`
- **Test Template:** `tests/test_template.py`

## Questions?

- Review existing registered domains in `morphogen/stdlib/` for patterns
- Check `docs/guides/domain-implementation.md` for domain creation guide
- See [DOMAIN_STATUS_ANALYSIS.md](../DOMAIN_STATUS_ANALYSIS.md) for detailed status

---

**Last Updated:** 2025-12-06
**Status:** Planning Phase - Migration starts v0.12.0
