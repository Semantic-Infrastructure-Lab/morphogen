# Domain Finishing Guide: From Python-Only to Production-Ready

**Version:** 1.0
**Date:** 2025-11-17
**Status:** Active Roadmap
**Context:** Strategic pivot from breadth (adding domains) to depth (finishing domains)

---

## Executive Summary

**The Problem:**
We have 23 domains implemented, but most are "half-finished" — they work in Python but aren't fully integrated into Morphogen/Morphogen as a platform.

**The Solution:**
This guide defines the **5 levels of completion** and provides a **clear roadmap** to finish what we started.

**Timeline:** 10 months (Months 1-10, see detailed breakdown below)

**Key Insight:**
> "Stop adding domains. Finish the 23 we have. Then every future domain integrates from day 1."

---

## Table of Contents

1. [The 5 Levels of Completion](#the-5-levels-of-completion)
2. [Current Status: The Gap Analysis](#current-status-the-gap-analysis)
3. [The Master Timeline](#the-master-timeline)
4. [Level-by-Level Finishing Guide](#level-by-level-finishing-guide)
5. [Per-Domain Finishing Checklist](#per-domain-finishing-checklist)
6. [Success Metrics](#success-metrics)
7. [Related Documentation](#related-documentation)

---

## The 5 Levels of Completion

Each domain can be at one of 5 levels. **Production-ready means Level 5.**

### Level 1: Python Runtime ✅
**What it means:**
- Operators implemented in `morphogen/stdlib/{domain}.py`
- Basic tests exist
- Examples work in Python

**Example:**
```python
# This works
from morphogen.stdlib import graph
network = graph.create_empty(directed=False)
```

**Status:** ✅ **23/23 domains have this**

---

### Level 2: Language Integration ❌
**What it means:**
- Domain registered in language runtime
- `use {domain}` statement works
- Operators callable from `.morph` source files
- Type signatures defined

**Example:**
```morphogen
# main.morph
use graph

let network = graph.create_empty(directed=false)
network = graph.add_edge(network, 0, 1, weight=1.0)
```

**Status:** ❌ **0/23 domains have this**
**Blocker:** Domain registry system not implemented
**Timeline:** Months 3-4 (8 weeks)

---

### Level 3: Type System Enforcement ❌
**What it means:**
- Physical units enforced at compile time
- Cross-domain type validation
- Rate compatibility checking
- Clear error messages

**Example:**
```morphogen
# Type errors caught at compile-time
let distance : f32 [m] = 10.0
let time : f32 [s] = 2.0

# ❌ Compile error: Cannot add incompatible units [m] and [s]
let invalid = distance + time

# ✅ OK: Division produces [m/s]
let speed : f32 [m/s] = distance / time
```

**Status:** ❌ **0/23 domains have this**
**Blocker:** Type checker doesn't enforce units/domains
**Timeline:** Months 5-6 (8 weeks)

---

### Level 4: Scheduler Integration ❌
**What it means:**
- Domain works in multirate execution
- Correct cross-rate behavior
- Sample-accurate timing
- Deterministic scheduling

**Example:**
```morphogen
# Multiple rates working together
@audio_rate(48000)
flow audio_flow(dt) {
    let synth = audio.sine(440.0)
    output synth
}

@control_rate(1000)
flow control_flow(dt) {
    let lfo = control.sine(1.0)
    # Control modulates audio (cross-rate communication)
    audio_flow.frequency = 440.0 + lfo * 50.0
}
```

**Status:** ❌ **0/23 domains have this**
**Blocker:** Multirate scheduler not fully implemented
**Timeline:** Months 7-8 (8 weeks)

---

### Level 5: MLIR Native Compilation ⚠️
**What it means:**
- MLIR dialect defined
- Lowering passes implemented
- Optimization passes
- 5-10x performance improvement
- JIT/AOT compilation

**Example:**
```bash
# Compile to native code
morphogen compile main.morph --output main.so --optimize

# 5-10x faster than Python runtime
./main.so
```

**Status:** ⚠️ **4/23 domains have this**
- ✅ field, agent, audio, temporal
- ❌ 19 domains need MLIR integration

**Timeline:** Ongoing (prioritize based on use cases)

---

## Current Status: The Gap Analysis

### Summary Table

| Level | Requirement | Status | Gap | Timeline |
|-------|-------------|--------|-----|----------|
| **1** | Python Runtime | ✅ 23/23 | None | DONE |
| **2** | Language Integration | ❌ 0/23 | All domains | M3-4 (8w) |
| **3** | Type Enforcement | ❌ 0/23 | All domains | M5-6 (8w) |
| **4** | Scheduler | ❌ 0/1 | Core system | M7-8 (8w) |
| **5** | MLIR Native | ⚠️ 4/23 | 19 domains | Ongoing |

### Critical Gaps

**Gap 1: Language Integration (CRITICAL)**
- **Impact:** Domains are Python libraries, not language features
- **User Pain:** Can't write `.morph` programs using new domains
- **Blocker:** All 23 domains stuck at Level 1
- **Fix:** Domain registry + parser enhancement (8 weeks)

**Gap 2: Type Safety (HIGH)**
- **Impact:** No compile-time safety for units, domains, rates
- **User Pain:** Runtime errors that should be compile errors
- **Blocker:** Professional workflows require type safety
- **Fix:** Type checker enhancement (8 weeks)

**Gap 3: Multirate Scheduling (HIGH)**
- **Impact:** Can't reliably run audio + control + visual simultaneously
- **User Pain:** Real-time applications don't work correctly
- **Blocker:** Most cross-domain workflows need this
- **Fix:** Scheduler implementation (8 weeks)

**Gap 4: MLIR Integration (MEDIUM)**
- **Impact:** Python runtime is 10-100x slower than compiled
- **User Pain:** Large simulations too slow for interactive use
- **Blocker:** Performance-sensitive applications
- **Fix:** Per-domain MLIR work (ongoing, prioritize)

---

## The Master Timeline

### Overview: 10 Months to Production-Ready

```
Phase 1          Phase 2                                    Phase 3
Showcase         Core Infrastructure                        Production
|-------|--------|--------|--------|--------|--------|--------|--------|
M1  M2  M3  M4  M5  M6  M7  M8  M9  M10
```

**Phase 1: Showcase & Validation (Months 1-2)**
- Generate professional outputs from existing domains
- Marketing and community feedback
- Validate use cases before infrastructure investment

**Phase 2: Core Infrastructure (Months 3-8)**
- Month 3-4: Language Integration (Level 2)
- Month 5-6: Type System (Level 3)
- Month 7-8: Scheduler + Cross-Domain (Level 4)

**Phase 3: Production Readiness (Months 9-10)**
- MLIR foundation work
- 3 real-world applications
- Production deployment

---

## Level-by-Level Finishing Guide

### Level 2: Language Integration (Months 3-4)

**Goal:** All 23 domains accessible from `.morph` source files

#### Work Items

**1. Domain Registry System (2 weeks)**

Create `morphogen/core/domain_registry.py`:

```python
"""Domain registration and lookup system."""

from typing import Dict, Callable, Any
from dataclasses import dataclass


@dataclass
class DomainDescriptor:
    """Metadata about a domain."""
    name: str
    module_path: str
    operators: Dict[str, Callable]
    types: Dict[str, type]
    version: str


class DomainRegistry:
    """Central registry for all Morphogen domains."""

    _domains: Dict[str, DomainDescriptor] = {}

    @classmethod
    def register(cls, name: str, module_path: str) -> None:
        """Register a domain."""
        # Import module
        module = __import__(module_path, fromlist=['*'])

        # Discover operators (functions decorated with @operator)
        operators = {}
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '_operator_metadata'):
                operators[attr_name] = attr

        # Create descriptor
        descriptor = DomainDescriptor(
            name=name,
            module_path=module_path,
            operators=operators,
            types={},  # TODO: type discovery
            version="0.10.0"
        )

        cls._domains[name] = descriptor

    @classmethod
    def get(cls, name: str) -> DomainDescriptor:
        """Get domain descriptor."""
        if name not in cls._domains:
            raise ValueError(f"Domain '{name}' not registered")
        return cls._domains[name]

    @classmethod
    def list_domains(cls) -> list[str]:
        """List all registered domains."""
        return list(cls._domains.keys())


# Auto-register all stdlib domains
def register_stdlib_domains():
    """Register all built-in stdlib domains."""
    domains = [
        ("field", "morphogen.stdlib.field"),
        ("agent", "morphogen.stdlib.agents"),
        ("audio", "morphogen.stdlib.audio"),
        ("visual", "morphogen.stdlib.visual"),
        ("rigidbody", "morphogen.stdlib.rigidbody"),
        ("integrators", "morphogen.stdlib.integrators"),
        ("graph", "morphogen.stdlib.graph"),
        ("signal", "morphogen.stdlib.signal"),
        ("statemachine", "morphogen.stdlib.statemachine"),
        ("terrain", "morphogen.stdlib.terrain"),
        ("vision", "morphogen.stdlib.vision"),
        ("cellular", "morphogen.stdlib.cellular"),
        ("optimization", "morphogen.stdlib.optimization"),
        ("neural", "morphogen.stdlib.neural"),
        ("sparse_linalg", "morphogen.stdlib.sparse_linalg"),
        ("io_storage", "morphogen.stdlib.io_storage"),
        ("acoustics", "morphogen.stdlib.acoustics"),
        ("noise", "morphogen.stdlib.noise"),
        ("color", "morphogen.stdlib.color"),
        ("image", "morphogen.stdlib.image"),
        ("palette", "morphogen.stdlib.palette"),
        ("genetic", "morphogen.stdlib.genetic"),
    ]

    for name, module_path in domains:
        DomainRegistry.register(name, module_path)
```

**Tests:**
```python
# tests/test_domain_registry.py
def test_register_domain():
    DomainRegistry.register("field", "morphogen.stdlib.field")
    assert "field" in DomainRegistry.list_domains()

def test_get_domain_operators():
    desc = DomainRegistry.get("field")
    assert "alloc" in desc.operators
    assert "diffuse" in desc.operators
```

**Deliverable:** ✅ Domain registry with all 23 domains registered

---

**2. Parser Enhancement for `use` Statement (2 weeks)**

Update `morphogen/parser/parser.py`:

```python
def parse_use_statement(self) -> UseStatement:
    """
    Parse: use graph
           use audio, visual
    """
    self.expect(TokenType.USE)

    domains = []
    while True:
        domain_name = self.expect(TokenType.IDENTIFIER).value
        domains.append(domain_name)

        if not self.match(TokenType.COMMA):
            break

    return UseStatement(domains=domains)
```

**Runtime binding in `morphogen/runtime/runtime.py`:**

```python
def execute_use_statement(self, stmt: UseStatement):
    """Load domain operators into current namespace."""
    for domain_name in stmt.domains:
        descriptor = DomainRegistry.get(domain_name)

        # Add all operators to runtime namespace
        for op_name, op_func in descriptor.operators.items():
            qualified_name = f"{domain_name}.{op_name}"
            self.context.add_operator(qualified_name, op_func)
```

**Deliverable:** ✅ `use graph` loads graph operators into namespace

---

**3. Operator Syntax Bindings (3 weeks)**

For each domain, create operator metadata:

```python
# morphogen/stdlib/graph.py

from morphogen.core.operator import operator, OpCategory

@operator(
    domain="graph",
    category=OpCategory.CONSTRUCT,
    signature="(directed: bool) -> Graph",
    deterministic=True,
    doc="Create an empty graph"
)
def create_empty(directed: bool = False) -> Graph:
    """Create an empty graph."""
    return Graph(directed=directed, nodes=set(), edges={})
```

**Auto-generate type signatures:**

```python
# morphogen/core/type_inference.py

def infer_operator_signature(op_func: Callable) -> TypeSignature:
    """Infer type signature from Python function."""
    import inspect
    sig = inspect.signature(op_func)

    params = []
    for name, param in sig.parameters.items():
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
        params.append(Parameter(name=name, type=param_type))

    return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else Any

    return TypeSignature(parameters=params, return_type=return_type)
```

**Deliverable:** ✅ All operators have type signatures

---

**4. Integration Testing (1 week)**

```python
# tests/test_language_integration.py

def test_use_graph_in_morphogen_program():
    """Test that graph domain works in .morph files."""
    program = """
    use graph

    let network = graph.create_empty(directed=false)
    network = graph.add_edge(network, 0, 1, weight=1.0)
    let path = graph.shortest_path(network, source=0, target=1)
    """

    runtime = Runtime()
    runtime.execute(program)

    assert "network" in runtime.context.variables
    assert "path" in runtime.context.variables

def test_all_domains_loadable():
    """Test that all 23 domains can be loaded."""
    domains = DomainRegistry.list_domains()
    assert len(domains) == 23

    for domain_name in domains:
        program = f"use {domain_name}"
        runtime = Runtime()
        runtime.execute(program)  # Should not raise
```

**Deliverable:** ✅ All 23 domains loadable from .morph files

---

#### Success Criteria for Level 2

- [x] Domain registry implemented and tested
- [x] Parser handles `use` statement
- [x] All 23 domains registered
- [x] Type signatures defined for all operators
- [x] Integration tests pass for all domains
- [x] Can write `.morph` programs using all domains
- [x] Documentation updated

**Timeline:** 8 weeks
**Status after completion:** 23/23 domains at Level 2

---

### Level 3: Type System Enforcement (Months 5-6)

**Goal:** Compile-time type safety for units, domains, and rates

#### Work Items

**1. Physical Unit System (3 weeks)**

Create `morphogen/type_system/units.py`:

```python
"""Physical unit system with dimensional analysis."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Unit:
    """Physical unit with SI dimensions."""

    # Base dimensions (SI)
    meter: int = 0      # length
    kilogram: int = 0   # mass
    second: int = 0     # time
    ampere: int = 0     # current
    kelvin: int = 0     # temperature
    mole: int = 0       # substance
    candela: int = 0    # luminosity

    def __mul__(self, other: 'Unit') -> 'Unit':
        """Multiply units."""
        return Unit(
            meter=self.meter + other.meter,
            kilogram=self.kilogram + other.kilogram,
            second=self.second + other.second,
            ampere=self.ampere + other.ampere,
            kelvin=self.kelvin + other.kelvin,
            mole=self.mole + other.mole,
            candela=self.candela + other.candela,
        )

    def __truediv__(self, other: 'Unit') -> 'Unit':
        """Divide units."""
        return Unit(
            meter=self.meter - other.meter,
            kilogram=self.kilogram - other.kilogram,
            second=self.second - other.second,
            ampere=self.ampere - other.ampere,
            kelvin=self.kelvin - other.kelvin,
            mole=self.mole - other.mole,
            candela=self.candela - other.candela,
        )

    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.meter: parts.append(f"m^{self.meter}")
        if self.kilogram: parts.append(f"kg^{self.kilogram}")
        if self.second: parts.append(f"s^{self.second}")
        return " ".join(parts) if parts else "dimensionless"


# Common units
DIMENSIONLESS = Unit()
METER = Unit(meter=1)
SECOND = Unit(second=1)
KILOGRAM = Unit(kilogram=1)
HERTZ = Unit(second=-1)
NEWTON = Unit(meter=1, kilogram=1, second=-2)
```

**Type checker enhancement:**

```python
# morphogen/type_system/type_checker.py

def check_binary_op(self, left_type: Type, op: str, right_type: Type) -> Type:
    """Check binary operation type correctness."""

    # Extract units
    left_unit = left_type.unit if hasattr(left_type, 'unit') else DIMENSIONLESS
    right_unit = right_type.unit if hasattr(right_type, 'unit') else DIMENSIONLESS

    if op in ['+', '-']:
        # Addition/subtraction requires same units
        if left_unit != right_unit:
            raise TypeError(
                f"Cannot {op} values with incompatible units: "
                f"{left_unit} and {right_unit}"
            )
        return left_type

    elif op == '*':
        # Multiplication multiplies units
        result_unit = left_unit * right_unit
        return type(left_type)(unit=result_unit)

    elif op == '/':
        # Division divides units
        result_unit = left_unit / right_unit
        return type(left_type)(unit=result_unit)
```

**Tests:**

```python
def test_unit_checking():
    """Test physical unit checking."""
    program = """
    let distance : f32 [m] = 10.0
    let time : f32 [s] = 2.0
    let speed = distance / time  # OK: [m/s]
    """

    checker = TypeChecker()
    ast = parse(program)
    checker.check(ast)  # Should pass

    # Check that speed has correct unit
    speed_type = checker.get_type("speed")
    assert speed_type.unit == METER / SECOND


def test_unit_error():
    """Test that unit errors are caught."""
    program = """
    let distance : f32 [m] = 10.0
    let time : f32 [s] = 2.0
    let invalid = distance + time  # ERROR: can't add m + s
    """

    checker = TypeChecker()
    ast = parse(program)

    with pytest.raises(TypeError, match="incompatible units"):
        checker.check(ast)
```

**Deliverable:** ✅ Physical unit checking enforced

---

**2. Cross-Domain Type Validation (2 weeks)**

```python
# morphogen/type_system/domain_types.py

class FieldType(Type):
    """Type for field values."""
    domain = "field"

class StreamType(Type):
    """Type for audio streams."""
    domain = "audio"

class AgentType(Type):
    """Type for agent collections."""
    domain = "agent"


def check_domain_compatibility(source_type: Type, target_type: Type) -> bool:
    """Check if types from different domains are compatible."""

    # Field can't be used as Stream
    if isinstance(source_type, FieldType) and isinstance(target_type, StreamType):
        return False

    # Need explicit conversion
    return source_type.domain == target_type.domain
```

**Deliverable:** ✅ Cross-domain type errors caught

---

**3. Rate Compatibility Checking (2 weeks)**

```python
# morphogen/type_system/rate_checking.py

class Rate:
    """Execution rate."""
    def __init__(self, hz: float, name: str):
        self.hz = hz
        self.name = name

    def __ge__(self, other: 'Rate') -> bool:
        return self.hz >= other.hz


AUDIO_RATE = Rate(48000, "audio")
CONTROL_RATE = Rate(1000, "control")
VISUAL_RATE = Rate(60, "visual")


def check_rate_access(source_rate: Rate, target_rate: Rate) -> bool:
    """Check if target_rate can read source_rate."""

    # Higher-rate can read lower-rate (needs resampling)
    if target_rate >= source_rate:
        return True

    # Lower-rate CANNOT directly read higher-rate
    # (needs aggregation: RMS, peak, mean)
    return False
```

**Deliverable:** ✅ Rate compatibility enforced

---

**4. Integration & Testing (1 week)**

**Deliverable:** ✅ 100+ type system tests passing

---

#### Success Criteria for Level 3

- [x] Unit system implemented and tested
- [x] Physical unit errors caught at compile time
- [x] Cross-domain type validation works
- [x] Rate compatibility checking works
- [x] Clear error messages for all type violations
- [x] Documentation and examples

**Timeline:** 8 weeks
**Status after completion:** 23/23 domains at Level 3

---

### Level 4: Scheduler Integration (Months 7-8)

**Goal:** Multirate execution with sample-accurate timing

#### Work Items

**1. LCM-Based Partitioning (2 weeks)**

```python
# morphogen/scheduler/multirate.py

from math import gcd
from functools import reduce


def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return abs(a * b) // gcd(a, b)


def lcm_multiple(rates: list[int]) -> int:
    """LCM of multiple rates."""
    return reduce(lcm, rates)


class MultirateScheduler:
    """Schedule multiple rate groups."""

    def __init__(self, rate_groups: dict[str, int]):
        """
        rate_groups: {'audio': 48000, 'control': 1000, 'visual': 60}
        """
        self.rate_groups = rate_groups
        self.master_rate = lcm_multiple(list(rate_groups.values()))

        # Calculate when each group fires
        self.firing_pattern = {}
        for name, rate in rate_groups.items():
            interval = self.master_rate // rate
            self.firing_pattern[name] = interval

    def tick(self, tick_num: int) -> list[str]:
        """Return which rate groups fire at this tick."""
        firing = []
        for name, interval in self.firing_pattern.items():
            if tick_num % interval == 0:
                firing.append(name)
        return firing
```

**Example:**
```python
scheduler = MultirateScheduler({
    'audio': 48000,
    'control': 1000,
    'visual': 60
})

# Master rate = LCM(48000, 1000, 60) = 48000
# Audio fires every tick
# Control fires every 48 ticks
# Visual fires every 800 ticks
```

**Deliverable:** ✅ LCM scheduling working

---

**2. Sample-Accurate Event Timing (2 weeks)**

```python
# morphogen/scheduler/events.py

@dataclass
class ScheduledEvent:
    """Event scheduled at specific sample time."""
    sample_time: int  # Sample number (not wall-clock time)
    callback: Callable
    data: Any


class EventScheduler:
    """Sample-accurate event scheduling."""

    def __init__(self):
        self.events: list[ScheduledEvent] = []
        self.current_sample = 0

    def schedule(self, delay_samples: int, callback: Callable, data: Any = None):
        """Schedule event N samples in the future."""
        event = ScheduledEvent(
            sample_time=self.current_sample + delay_samples,
            callback=callback,
            data=data
        )
        self.events.append(event)
        self.events.sort(key=lambda e: e.sample_time)

    def process_events(self) -> list[ScheduledEvent]:
        """Process all events due at current sample."""
        due_events = []

        while self.events and self.events[0].sample_time <= self.current_sample:
            event = self.events.pop(0)
            due_events.append(event)
            event.callback(event.data)

        return due_events

    def advance(self):
        """Advance to next sample."""
        self.current_sample += 1
```

**Deliverable:** ✅ Sample-accurate events

---

**3. Cross-Rate Resampling (2 weeks)**

```python
# morphogen/scheduler/resampling.py

def resample_hold(source_value: float, source_rate: int, target_rate: int) -> float:
    """Hold last value (zero-order hold)."""
    return source_value


def resample_linear(source_buffer: list[float], source_rate: int, target_rate: int) -> list[float]:
    """Linear interpolation."""
    import numpy as np
    source_times = np.arange(len(source_buffer)) / source_rate
    target_times = np.arange(int(len(source_buffer) * target_rate / source_rate)) / target_rate
    return np.interp(target_times, source_times, source_buffer)


def aggregate_rms(source_buffer: list[float]) -> float:
    """RMS aggregation for higher→lower rate."""
    import numpy as np
    return np.sqrt(np.mean(np.square(source_buffer)))
```

**Deliverable:** ✅ Cross-rate communication

---

**4. Integration Testing (2 weeks)**

```python
def test_multirate_execution():
    """Test audio + control + visual together."""
    program = """
    @rate(48000)
    flow audio(dt) {
        let synth = audio.sine(440.0)
        output synth
    }

    @rate(1000)
    flow control(dt) {
        let lfo = control.sine(1.0)
        audio.frequency = 440.0 + lfo * 50.0
    }

    @rate(60)
    flow visual(dt) {
        let field = visual.colorize(control.lfo)
        output field
    }
    """

    runtime = Runtime()
    runtime.execute(program)

    # Verify all rates executed
    assert runtime.audio_ticks > 0
    assert runtime.control_ticks > 0
    assert runtime.visual_ticks > 0
```

**Deliverable:** ✅ Multirate examples working

---

#### Success Criteria for Level 4

- [x] LCM scheduler implemented
- [x] Sample-accurate event timing verified
- [x] Cross-rate resampling works (hold, linear, cubic)
- [x] Deterministic execution (bit-exact results)
- [x] Real-time capable on modern hardware
- [x] Multi-rate examples demonstrate all 3 rates

**Timeline:** 8 weeks
**Status after completion:** All domains can work in multirate scenarios

---

### Level 5: MLIR Native Compilation (Ongoing)

**Goal:** 5-10x performance via native compilation

**Priority Domains** (based on performance needs):
1. **rigidbody** - Physics simulation (hot path: collision detection)
2. **integrators** - Time-stepping (called thousands of times)
3. **signal** - FFT/STFT (computationally intensive)
4. **graph** - Graph algorithms (large-scale networks)
5. **vision** - Image processing (pixel operations)

#### Per-Domain Work (2-4 weeks each)

**Step 1: Define MLIR Dialect**

See `docs/guides/domain-implementation.md` Section "Step 2: Define MLIR Dialect" for full template.

**Example for rigidbody:**

```python
# morphogen/mlir/dialects/rigidbody.py

@register_type("rigidbody")
class RigidBody2DType(Type):
    """!rigidbody.body type."""
    pass

@register_op("rigidbody")
class IntegrateOp(Operation):
    """rigidbody.integrate operation."""
    name = "rigidbody.integrate"

    def __init__(self, body, dt):
        self.body = body
        self.dt = dt
```

**Step 2: Implement Lowering**

```python
# morphogen/mlir/lowering/rigidbody_to_scf.py

class RigidBodyToSCFPass(LoweringPass):
    """Lower rigidbody ops to SCF loops."""

    def lower_integrate(self, op: IntegrateOp):
        """
        %body' = rigidbody.integrate(%body, %dt)

        Lowers to:
        %vel' = arith.add %vel, arith.mul %acc, %dt
        %pos' = arith.add %pos, arith.mul %vel', %dt
        """
        # Generate SCF + arithmetic ops
```

**Step 3: Benchmark**

```python
def benchmark_rigidbody():
    # Python runtime
    start = time.time()
    simulate_physics_python(1000_bodies, 1000_steps)
    python_time = time.time() - start

    # MLIR compiled
    start = time.time()
    simulate_physics_mlir(1000_bodies, 1000_steps)
    mlir_time = time.time() - start

    speedup = python_time / mlir_time
    assert speedup >= 5.0  # Target: 5x minimum
```

**Success Criteria per Domain:**
- [x] MLIR dialect defined
- [x] Lowering passes implemented
- [x] Compiles without errors
- [x] Produces correct results (matches Python)
- [x] 5-10x performance improvement
- [x] Integration tests pass

**Timeline:** 2-4 weeks per domain (prioritize based on use cases)

---

## Per-Domain Finishing Checklist

Use this checklist for each domain:

### Domain: ________________

**Level 1: Python Runtime** ✅ (Should already be done)
- [ ] Operators implemented in `morphogen/stdlib/{domain}.py`
- [ ] Unit tests pass (`tests/test_{domain}.py`)
- [ ] At least 3 examples work
- [ ] Documentation exists

**Level 2: Language Integration** (8 weeks total for all domains)
- [ ] Domain registered in `DomainRegistry`
- [ ] All operators have `@operator` decorator
- [ ] Type signatures defined
- [ ] `use {domain}` statement works
- [ ] Integration test passes
- [ ] Can write `.morph` programs using this domain

**Level 3: Type System** (Incremental as type system is built)
- [ ] Operators have correct input/output types
- [ ] Physical units specified (if applicable)
- [ ] Domain type defined (FieldType, StreamType, etc.)
- [ ] Rate specified (if applicable)
- [ ] Type errors produce clear messages

**Level 4: Scheduler** (Automatic once scheduler exists)
- [ ] Domain specifies rate requirements
- [ ] Works in multirate scenarios
- [ ] Cross-rate communication tested
- [ ] Determinism verified

**Level 5: MLIR Native** (Per-domain effort, 2-4 weeks each)
- [ ] MLIR dialect defined (`morphogen/mlir/dialects/{domain}.py`)
- [ ] Lowering pass implemented (`morphogen/mlir/lowering/{domain}_to_scf.py`)
- [ ] Optimization passes (optional)
- [ ] Compiles to native code
- [ ] Benchmark shows 5-10x speedup
- [ ] Correctness tests pass (output matches Python)

**Production Ready:**
- [ ] All 5 levels complete
- [ ] Real application uses this domain
- [ ] Documentation is comprehensive
- [ ] Examples showcase capabilities

---

## Success Metrics

### By End of Month 2 (Showcase Phase)
- [ ] 15+ professional outputs generated from cross-domain examples
- [ ] Community feedback collected (50+ responses)
- [ ] Top 3 use cases identified
- [ ] Validated market fit

### By End of Month 4 (Language Integration)
- [ ] 23/23 domains at Level 2
- [ ] Can write `.morph` programs using all domains
- [ ] Integration tests pass for all domains
- [ ] Documentation updated

### By End of Month 6 (Type System)
- [ ] 23/23 domains at Level 3
- [ ] Physical unit checking works
- [ ] Cross-domain type errors caught
- [ ] Type error messages are clear and helpful

### By End of Month 8 (Scheduler)
- [ ] All domains work in multirate execution
- [ ] Audio @ 48kHz + Control @ 1kHz + Visual @ 60Hz works
- [ ] Sample-accurate timing verified
- [ ] Cross-domain workflows tested

### By End of Month 10 (Production)
- [ ] 3 real-world applications built
- [ ] At least 5 domains have MLIR integration
- [ ] 5-10x performance improvement demonstrated
- [ ] Production deployment of 1+ application

---

## Related Documentation

### Core References
- **Master Plan:** `docs/planning/EXECUTION_PLAN_Q4_2025.md` (full 10-month timeline)
- **Domain Implementation:** `docs/guides/domain-implementation.md` (MLIR integration details)
- **Testing Strategy:** `docs/roadmap/testing-strategy.md` (production-ready testing)
- **Cross-Domain API:** `docs/CROSS_DOMAIN_API.md` (domain interop patterns)

### Architecture
- **Domain Architecture:** `docs/architecture/domain-architecture.md` (23 domain specs)
- **Type System:** `docs/specifications/type-system.md` (units, domains, rates)
- **Scheduler:** `docs/specifications/scheduler.md` (multirate execution)

### ADRs (Decision Records)
- **ADR-002:** Cross-Domain Architectural Patterns
- **ADR-010:** Ecosystem Branding & Naming Strategy
- **ADR-011:** Project Renaming (Morphogen & Philbrick)

---

## Quick Reference Commands

```bash
# Check domain registration
morphogen domains list

# Verify domain can be used
morphogen check main.morph --domain graph

# Run type checker
morphogen typecheck main.morph

# Compile to MLIR
morphogen compile main.morph --emit-mlir

# Benchmark domain performance
morphogen benchmark --domain rigidbody --python-vs-mlir

# Run full test suite
pytest tests/ --cov --cov-report=html

# Generate domain documentation
morphogen docs generate --domain graph
```

---

## Appendix: Frequently Asked Questions

**Q: Why not just keep adding domains?**
A: Because each "half-finished" domain creates technical debt. We have to retrofit language integration, type checking, scheduler support, and MLIR for all 23 domains. Better to finish the 23 we have, then every future domain integrates from day 1.

**Q: Do all domains need MLIR?**
A: No. MLIR (Level 5) is for performance. Focus on Levels 2-4 first (language integration, type safety, scheduling). Then prioritize MLIR for hot domains based on profiling.

**Q: How long does this really take?**
A:
- Levels 2-4: 24 weeks (6 months) for all 23 domains
- Level 5: 2-4 weeks per domain (prioritize ~5-8 domains)
- Total: ~8-10 months to production-ready

**Q: Can we parallelize?**
A: Levels 2-4 are infrastructure work (affects all domains simultaneously). Level 5 can be parallelized (different people work on different domains).

**Q: What if we find bugs in Level 1?**
A: Fix them! Level 1 (Python runtime) is the foundation. All higher levels depend on it working correctly.

**Q: What about testing?**
A: Testing happens at every level:
- Level 1: Unit tests (operators work)
- Level 2: Integration tests (.morph programs run)
- Level 3: Type tests (errors caught)
- Level 4: Multirate tests (scheduling works)
- Level 5: Performance tests (speedup verified)

---

**Last Updated:** 2025-11-17
**Status:** Active roadmap
**Next Review:** End of Month 2 (after showcase validation)
**Owner:** Morphogen/Morphogen Project

---

*This guide consolidates information from:*
- *`docs/planning/EXECUTION_PLAN_Q4_2025.md` (master timeline)*
- *`docs/guides/domain-implementation.md` (MLIR integration)*
- *`docs/roadmap/testing-strategy.md` (testing requirements)*
- *`docs/CROSS_DOMAIN_API.md` (cross-domain patterns)*
