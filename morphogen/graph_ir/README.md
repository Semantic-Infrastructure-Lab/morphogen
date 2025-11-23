# Morphogen GraphIR

**Version:** 1.0.0
**Status:** Production-ready
**Implementation Date:** 2025-11-23

## Overview

Morphogen GraphIR is the canonical intermediate representation between Morphogen frontends (DSL, RiffStack) and the Morphogen Kernel. It provides a typed, JSON-based directed acyclic graph (DAG) representation for computational graphs.

## Features

✅ **Complete Implementation** (Week 1 Deliverables)
- ✅ GraphIR data structures (Node, Edge, Graph)
- ✅ JSON serialization (load/save `.morph.json` files)
- ✅ Comprehensive validation (types, DAG, units, references)
- ✅ 28/28 unit tests passing
- ✅ Example code and documentation

## Quick Start

```python
from morphogen.graph_ir import GraphIR, GraphIROutputPort

# Create a graph
graph = GraphIR(sample_rate=48000, seed=1337)

# Add nodes
graph.add_node(
    id="osc1",
    op="sine",
    outputs=[GraphIROutputPort(name="out", type="Sig")],
    rate="audio",
    params={"freq": "440Hz"}
)

graph.add_node(
    id="lpf1",
    op="lpf",
    outputs=[GraphIROutputPort(name="out", type="Sig")],
    rate="audio",
    params={"cutoff": "2kHz", "q": 0.707}
)

# Connect nodes
graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")

# Define outputs
graph.add_output("mono", ["lpf1:out"])

# Validate
errors = graph.validate()
if not errors:
    print("✓ Graph is valid!")

# Save to file
graph.to_json("my_synth.morph.json")

# Load from file
loaded = GraphIR.from_json("my_synth.morph.json")
```

## Core Classes

### `GraphIR`
Top-level graph representation.

**Attributes:**
- `version`: GraphIR version (default: "1.0")
- `profile`: Execution profile (strict | repro | live)
- `sample_rate`: Audio sample rate in Hz
- `nodes`: List of graph nodes
- `edges`: List of graph edges
- `outputs`: Mapping of output names to port references
- `seed`: Optional random seed for deterministic execution
- `events`: Optional event stream definitions
- `metadata`: Optional metadata

**Methods:**
- `add_node()`: Add a node to the graph
- `add_edge()`: Add an edge to the graph
- `add_output()`: Add an output mapping
- `get_node(id)`: Get node by ID
- `validate()`: Validate graph structure
- `is_valid()`: Check if graph is valid
- `to_json(filepath)`: Save to JSON file
- `from_json(filepath)`: Load from JSON file

### `GraphIRNode`
Represents an operator instance in the graph.

**Attributes:**
- `id`: Unique node identifier
- `op`: Operator name (from registry)
- `outputs`: List of output ports
- `params`: Parameter values with units
- `rate`: Execution rate (audio | control | visual | sim)
- `profile_overrides`: Optional per-node profile settings

### `GraphIREdge`
Represents a connection between nodes.

**Attributes:**
- `from_port`: Source node:port reference (e.g., "osc1:out")
- `to_port`: Destination node:port reference (e.g., "lpf1:in")
- `type`: Stream type annotation (Sig, Ctl, Field2D, etc.)

**Properties:**
- `from_node`: Source node ID
- `from_port_name`: Source port name
- `to_node`: Destination node ID
- `to_port_name`: Destination port name

### `GraphIROutputPort`
Output port definition for a node.

**Attributes:**
- `name`: Port name
- `type`: Stream type (Sig, Ctl, Field2D, Field3D, Image)

### `GraphValidator`
Validates GraphIR structure and semantics.

**Validation Rules:**
1. ✅ Node ID uniqueness
2. ✅ Valid rate classes (audio, control, visual, sim)
3. ✅ Edge reference integrity (nodes and ports exist)
4. ✅ Type compatibility
5. ✅ DAG constraint (no cycles)
6. ✅ Unit consistency
7. ✅ Output mapping validity

**Methods:**
- `validate()`: Run all validation checks, returns list of errors
- `topological_sort()`: Compute execution order (returns None if cyclic)

## Type System

### Stream Types
| Type | Description | Rate |
|------|-------------|------|
| `Sig` | Audio signal (1D float stream) | audio |
| `Ctl` | Control signal (scalar value) | control |
| `Field2D` | 2D field | sim |
| `Field3D` | 3D field | sim |
| `Image` | Image/frame (RGB) | visual |

### Rate Classes
| Rate | Default Hz | Domain |
|------|-----------|---------|
| `audio` | 48000 | Audio processing |
| `control` | 1000 | Control signals |
| `visual` | 60 | Video/graphics |
| `sim` | Variable | Simulation |

### Type Compatibility
- `Sig → Sig`: ✅ Compatible
- `Ctl → Ctl`: ✅ Compatible
- `Ctl → Sig`: ✅ Compatible (upsampling)
- `Sig → Ctl`: ❌ Requires explicit downsampling
- Cross-domain types: ❌ Not compatible

## Unit Annotations

Parameters support unit annotations for clarity and validation:

```python
params = {
    "freq": "440Hz",      # Frequency
    "cutoff": "2kHz",     # Kilohertz
    "time": "0.5s",       # Seconds
    "attack": "10ms",     # Milliseconds
    "gain": "-6dB",       # Decibels
    "phase": "0.25rad",   # Radians
}
```

**Supported Units:**
- Frequency: Hz, kHz, MHz
- Time: s, ms, us
- Decibels: dB, dBFS
- Angle: rad, deg
- Distance: m, cm, mm
- Mass: kg, g

## Examples

### Simple Audio Graph
```python
graph = GraphIR()

# Oscillator
graph.add_node(
    id="osc",
    op="sine",
    outputs=[GraphIROutputPort(name="out", type="Sig")],
    params={"freq": "440Hz"}
)

# Output
graph.add_output("mono", ["osc:out"])
```

### Multirate Graph (Audio + Control)
```python
graph = GraphIR()

# Audio oscillator
graph.add_node(
    id="osc",
    op="sine",
    outputs=[GraphIROutputPort(name="out", type="Sig")],
    rate="audio",
    params={"freq": "440Hz"}
)

# Control envelope
graph.add_node(
    id="env",
    op="adsr",
    outputs=[GraphIROutputPort(name="out", type="Ctl")],
    rate="control",
    params={"attack": "0.1s"}
)

# Multiply (audio × control)
graph.add_node(
    id="mul",
    op="multiply",
    outputs=[GraphIROutputPort(name="out", type="Sig")],
    rate="audio"
)

# Cross-rate connection
graph.add_edge(from_port="osc:out", to_port="mul:in1", type="Sig")
graph.add_edge(from_port="env:out", to_port="mul:in2", type="Ctl")  # Cross-rate!
```

### Stereo Output
```python
# Panner with stereo output
graph.add_node(
    id="pan",
    op="pan",
    outputs=[
        GraphIROutputPort(name="left", type="Sig"),
        GraphIROutputPort(name="right", type="Sig"),
    ],
    params={"pos": 0.0}
)

# Stereo output mapping
graph.add_output("stereo", ["pan:left", "pan:right"])
```

## JSON Schema

GraphIR files use the `.morph.json` extension and follow this schema:

```json
{
  "version": "1.0",
  "profile": "repro",
  "sample_rate": 48000,
  "seed": 42,
  "nodes": [
    {
      "id": "osc1",
      "op": "sine",
      "params": {"freq": "440Hz"},
      "rate": "audio",
      "outputs": [{"name": "out", "type": "Sig"}]
    }
  ],
  "edges": [
    {"from": "osc1:out", "to": "lpf1:in", "type": "Sig"}
  ],
  "outputs": {
    "mono": ["lpf1:out"]
  },
  "metadata": {
    "source": "Morphogen.Audio",
    "author": "user@example.com"
  }
}
```

## Testing

Run the test suite:

```bash
pytest morphogen/graph_ir/test_graph_ir.py -v
```

**Test Coverage:**
- ✅ 28 tests covering all core functionality
- ✅ Data structure creation and serialization
- ✅ JSON load/save round-trips
- ✅ All validation rules
- ✅ Complex graph scenarios
- ✅ Edge cases and error handling

## File Structure

```
morphogen/graph_ir/
├── __init__.py          # Public API exports
├── core.py              # Core data structures
├── validation.py        # Graph validation
├── test_graph_ir.py     # Unit tests (28 tests)
└── README.md            # This file
```

## Next Steps

**Week 2-5 Roadmap** (from implementation plan):

### Week 3-4: Simplified Scheduler
- [ ] GCD master clock computation
- [ ] Rate group partitioning (audio + control)
- [ ] Topological sort (execution order)
- [ ] Linear resampling (cross-rate connections)
- [ ] Simple execution loop

### Week 5: Integration & Testing
- [ ] Connect operators to scheduler
- [ ] Test multirate graphs
- [ ] Create example projects
- [ ] Validate Pantheon adapter integration

## Integration Points

### Pantheon Adapter
GraphIR is designed to work with the Pantheon Layer 4 adapter:

```python
from pantheon.adapters.morphogen import MorphogenAdapter

# Convert GraphIR to Pantheon
adapter = MorphogenAdapter()
pantheon_graph = adapter.to_pantheon(graph_ir)

# Convert back (round-trip)
reconstructed = adapter.from_pantheon(pantheon_graph)
```

See: `/home/scottsen/src/projects/pantheon/adapters/morphogen/adapter.py`

## Specification

Full specification:
- 📄 `/home/scottsen/src/projects/morphogen/docs/specifications/graph-ir.md`

Related specifications:
- 📄 `/home/scottsen/src/projects/morphogen/docs/specifications/scheduler.md`
- 📄 `/home/scottsen/src/projects/pantheon/docs/specifications/layer-4-dynamics.md`

## Version History

**1.0.0** (2025-11-23)
- ✅ Initial implementation
- ✅ Core data structures
- ✅ JSON serialization
- ✅ Comprehensive validation
- ✅ Full test coverage (28 tests)
- ✅ Example code

## License

Part of the Morphogen project.

---

**Status:** ✅ Week 1 complete - GraphIR core implementation ready for scheduler integration
