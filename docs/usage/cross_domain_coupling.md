---
title: "Cross-Domain Coupling — How It Works and How to Write Your Own"
type: guide
beth_topics:
  - morphogen
  - cross-domain
  - domain-interface
  - coupling
  - architecture
  - extensibility
---

# Cross-Domain Coupling

Morphogen's cross-domain system lets you connect any two domains with
type-safe, composable transformation objects. This guide explains how the
built-in coupling works and how to write your own.

---

## The core idea

Every cross-domain connection is a `DomainInterface` subclass:

```
Source domain data  →  [DomainInterface.transform()]  →  Target domain data
```

An interface:
1. Declares which domains it bridges (`source_domain`, `target_domain`)
2. Validates that the input data is compatible
3. Transforms the data into the target domain's format

No magic. The result is plain Python values — NumPy arrays, dataclasses, floats —
that the target domain can consume directly.

---

## What's already built in

The `morphogen.cross_domain` module ships interfaces for common pairings:

| Interface | File | What it does |
|-----------|------|-------------|
| `PhysicsToAudioInterface` | `physics_audio.py` | Collision event list → per-event audio parameters |
| `FluidToAcousticsInterface` | `physics_audio.py` | 2D pressure field list → acoustic field |
| `AcousticsToAudioInterface` | `physics_audio.py` | Acoustic fields at mic positions → waveform |
| `AudioVisualInterface` | `audio_visual.py` | Spectral features → visual parameters |
| `FieldAgentInterface` | `field_agent.py` | `Field2D` sampled at agent positions |
| `TerrainNavigationInterface` | `terrain.py` | Terrain heightmap → agent pathfinding grid |
| `SpectralFieldInterface` | `spectral.py` | Field → spectral decomposition |
| `SpatialAudioInterface` | `spatial.py` | 3D positions → stereo/binaural audio |

```python
from morphogen.cross_domain import (
    PhysicsToAudioInterface,
    FluidToAcousticsInterface,
    AcousticsToAudioInterface,
)
```

---

## Using a built-in interface

Here is the physics → audio chain from
[`examples/canonical/01_physics_to_audio.py`](/examples/canonical/01_physics_to_audio.py):

```python
import numpy as np
from morphogen.cross_domain.physics_audio import (
    PhysicsToAudioInterface,
    AcousticsToAudioInterface,
)

# Step 1: collect collision events from rigidbody simulation
# Each event is a dict with keys: velocity, mass, position, time
collision_events = [
    {"velocity": 3.5, "mass": 1.0, "position": [0, 0], "time": 0.12},
    {"velocity": 2.1, "mass": 1.0, "position": [0, 0], "time": 0.45},
]

# Step 2: declare the mapping (source key → audio parameter)
interface = PhysicsToAudioInterface(
    events=collision_events,
    mapping={
        "velocity": "amplitude",   # impact speed → volume
        "mass":     "pitch",       # mass → fundamental frequency
    },
    sample_rate=44100,
)

# Step 3: transform
audio_params = interface.transform(collision_events)
# → Dict[str, np.ndarray]: {"amplitude": ..., "pitch": ...}

print(f"amplitude values: {audio_params['amplitude']}")
print(f"pitch values:     {audio_params['pitch']}")
```

The `__call__` shorthand validates then transforms in one step:

```python
audio_params = interface(collision_events)  # same result
```

---

## Writing a custom interface

Suppose you want to connect `molecular` simulation output to a visual
colour mapping — no built-in interface exists for this.

**Step 1: subclass `DomainInterface`**

```python
from morphogen.cross_domain.base import DomainInterface
import numpy as np

class MolecularToColorInterface(DomainInterface):
    """Map per-atom kinetic energies to RGB colours."""

    source_domain = "molecular"
    target_domain = "visual"

    def __init__(self, colormap: str = "viridis", scale_factor: float = 1.0):
        super().__init__()
        self.colormap = colormap
        self.scale_factor = scale_factor

    def transform(self, molecule) -> np.ndarray:
        """
        Args:
            molecule: A Molecule object with .positions and .velocities

        Returns:
            np.ndarray, shape (n_atoms, 3), float32 RGB values in [0, 1]
        """
        # Kinetic energy per atom: KE = ½ m v²
        masses    = molecule.masses()                    # (n_atoms,)
        velocities = getattr(molecule, "velocities", None)

        if velocities is None:
            # No velocities — fall back to atom index
            t = np.linspace(0, 1, molecule.n_atoms)
        else:
            ke = 0.5 * masses * np.sum(velocities**2, axis=1)
            t  = ke / (ke.max() + 1e-12)

        t = np.clip(t * self.scale_factor, 0, 1)

        # Simple viridis-like ramp (no matplotlib dependency)
        r = np.clip(1.7 * t - 0.4, 0, 1)
        g = np.clip(1.4 * (1 - np.abs(2*t - 0.8)), 0, 1)
        b = np.clip(1.0 - 1.5 * t, 0, 1)

        return np.stack([r, g, b], axis=1).astype(np.float32)

    def validate(self) -> bool:
        """Check that source_data is a Molecule."""
        if self.source_data is None:
            return True   # deferred validation is fine
        has_positions = hasattr(self.source_data, "positions")
        has_atoms     = hasattr(self.source_data, "atoms")
        return has_positions and has_atoms

    def get_input_interface(self):
        return {"molecule": "Molecule"}

    def get_output_interface(self):
        return {"colors": "np.ndarray (n_atoms, 3) float32 RGB"}
```

**Step 2: use it**

```python
from morphogen.stdlib.molecular import load_smiles, optimize_geometry

mol = optimize_geometry(load_smiles("c1ccccc1"))

iface = MolecularToColorInterface(colormap="viridis", scale_factor=2.0)
colors = iface(mol)
print(f"colours shape: {colors.shape}")   # (n_atoms, 3)
print(f"first atom RGB: {colors[0]}")
```

**Step 3: optionally register it**

Register makes the interface discoverable by `TransformComposer`:

```python
from morphogen.cross_domain.registry import CrossDomainRegistry

CrossDomainRegistry.register(
    source="molecular",
    target="visual",
    interface_class=MolecularToColorInterface,
    metadata={
        "description": "Per-atom kinetic energy → RGB colour",
        "version": "1.0",
    }
)

# Now the registry knows about it:
cls = CrossDomainRegistry.get("molecular", "visual")
print(cls)   # MolecularToColorInterface
```

---

## Composing interfaces with TransformComposer

`TransformComposer` finds and chains registered interfaces automatically.

```python
from morphogen.cross_domain.composer import TransformComposer, auto_compose

composer = TransformComposer()

# Find a registered path, molecular → visual
pipeline = composer.compose_path("molecular", "visual")
if pipeline:
    result = pipeline(mol)
    print(f"pipeline: {pipeline.visualize()}")
else:
    print("no registered path found — use direct interface")
```

For linear chains you control:

```python
from morphogen.cross_domain.composer import compose

# Chain two interfaces manually (no registry needed)
pipeline = compose(
    MolecularToColorInterface(),
    # AnotherInterface() ...
    validate=True,
)
output = pipeline(mol)
```

---

## The `@DomainTransform` decorator shorthand

For simple stateless transforms, the decorator is less boilerplate:

```python
from morphogen.cross_domain.base import DomainTransform

@DomainTransform(
    source="field",
    target="audio",
    description="Slice a row of a Field2D into a waveform",
)
def field_row_to_waveform(field, row: int = 64):
    """Sample row `row` of the field as a mono audio frame."""
    return field.data[row, :].astype(np.float32)
```

For inline use the function remains callable as-is
(`field_row_to_waveform(field, row=64)`). The decorator also attaches the
generated `DomainInterface` subclass to the function as `fn.interface`, so you
can drive it through the validating interface path when you want type checks:

```python
interface = field_row_to_waveform.interface()
waveform = interface(field)   # runs validate() then transform()
```

If you declare `input_types` on the decorator, `validate()` enforces them — the
first input is checked against the first declared type and a `TypeError` is
raised on a mismatch, so a bad cross-domain flow fails loudly instead of
silently producing garbage:

```python
@DomainTransform(source="field", target="audio", input_types={"field": np.ndarray})
def field_row_to_waveform(field):
    return field[64, :].astype(np.float32)
```

With no `input_types` declared, validation is permissive.

---

## Design rules for interfaces

1. **Inputs and outputs are plain values.** Return NumPy arrays, dataclasses,
   or primitives — not domain-specific wrapper objects that the target can't use.

2. **`validate()` is lightweight.** Check shapes and types; don't run the
   transform inside validate.

3. **No hidden state between calls.** Interfaces may be called multiple times
   with different `source_data`. Don't accumulate state across calls.

4. **One transformation, one interface.** If you need two mappings, write two
   interfaces and chain them with `compose()`.

5. **Document the contract.** `get_input_interface()` and `get_output_interface()`
   are documentation-as-code — fill them in even if validation is minimal.

---

## See also

- [`morphogen/cross_domain/base.py`](/morphogen/cross_domain/base.py) — `DomainInterface`, `DomainTransform`
- [`morphogen/cross_domain/registry.py`](/morphogen/cross_domain/registry.py) — `CrossDomainRegistry`
- [`morphogen/cross_domain/composer.py`](/morphogen/cross_domain/composer.py) — `TransformComposer`, `compose()`
- [`morphogen/cross_domain/physics_audio.py`](/morphogen/cross_domain/physics_audio.py) — reference implementation
- [`examples/canonical/01_physics_to_audio.py`](/examples/canonical/01_physics_to_audio.py) — end-to-end physics → audio
- [`docs/adr/002-cross-domain-architectural-patterns.md`](/docs/adr/002-cross-domain-architectural-patterns.md) — ADR with design rationale
