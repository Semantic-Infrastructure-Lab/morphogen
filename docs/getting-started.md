---
title: "Getting Started with Morphogen"
type: guide
beth_topics:
  - morphogen
  - getting-started
  - installation
  - tutorial
---

# Getting Started with Morphogen

**Version**: v0.12.0 | **Python**: 3.9+

Morphogen is a cross-domain computation platform for simulation, audio synthesis, and computational creativity. The core idea: instead of gluing together NumPy + SciPy + librosa + PyBullet with manual data conversion, you compose typed operators from different domains directly.

```python
# Physics drives audio — no glue code
from morphogen.stdlib.rigidbody import create_circle_body, step_world
from morphogen.stdlib.audio import AudioBuffer

# collision → sound, circuit → audio, fluid → acoustics → wav
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install from source

```bash
git clone https://github.com/scottsen/morphogen.git
cd morphogen
pip install -e .
```

Core dependencies (numpy, scipy, pillow) install automatically.

### Optional: audio I/O

```bash
pip install -e ".[io]"
# adds: sounddevice, soundfile, imageio
```

### Verify

```bash
morphogen version
# Morphogen v0.12.0
```

---

## Your First Program (Python API)

The recommended interface is the Python API. Let's run a field diffusion:

```python
import numpy as np
from morphogen.stdlib import field, visual

# Create a 64×64 temperature field with a hot spot in the center
temp = field.alloc((64, 64), fill_value=0.0)
temp.data[28:36, 28:36] = 1.0   # hot spot

# Diffuse for 50 steps
for _ in range(50):
    temp = field.diffuse(temp, rate=0.15, dt=0.1)

print(f"Center: {temp.data[32, 32]:.4f}  Corner: {temp.data[0, 0]:.4f}")
# Center: 0.0426  Corner: 0.0001
```

---

## Core Concepts

### 1. Domains

Every capability in Morphogen lives in a domain. Import the one you need:

```python
from morphogen.stdlib import field       # 2D/3D grid operations
from morphogen.stdlib import audio       # signal processing, WAV I/O
from morphogen.stdlib import rigidbody   # 2D physics simulation
from morphogen.stdlib import circuit     # analog circuit simulation
from morphogen.stdlib import acoustics   # wave propagation
from morphogen.stdlib import noise       # Perlin, fractal noise
# ... 39 domains total
```

### 2. Operators are functions

Every domain exposes operators as plain Python functions. They're deterministic, take typed inputs, and return typed outputs:

```python
from morphogen.stdlib.field import Field2D
import morphogen.stdlib.field as field

f = field.random((128, 128), seed=42)     # Field2D
f2 = field.diffuse(f, rate=0.1, dt=0.01) # Field2D
g = field.gradient(f)                    # Field2D (gradient magnitude)
```

### 3. Cross-domain composition

The value is that typed objects pass directly between domains:

```python
# AudioBuffer → circuit → AudioBuffer
from morphogen.stdlib.audio import AudioBuffer, AudioOperations as audio
from morphogen.stdlib.circuit import CircuitOperations as circuit

buf = AudioBuffer(np.sin(2*np.pi*440*t), 44100)   # Audio domain
pedal = circuit.create(num_nodes=4, dt=1/44100)   # Circuit domain
result = circuit.process_audio(pedal, buf, ...)    # audio ↔ circuit ↔ audio
```

### 4. Deterministic randomness

All random operators take a `seed` parameter. Same seed = same output everywhere:

```python
a = field.random((64, 64), seed=42)
b = field.random((64, 64), seed=42)
assert np.array_equal(a.data, b.data)   # always True
```

---

## Three Working Examples

### Example 1 — Physics → Audio (rigidbody + audio)

Bouncing balls generate percussive impacts as WAV. See
[`examples/canonical/01_physics_to_audio.py`](../examples/canonical/01_physics_to_audio.py).

```bash
python examples/canonical/01_physics_to_audio.py
# → examples/canonical/output/01_physics_to_audio.wav
```

### Example 2 — Circuit → Audio (circuit + audio)

Guitar signal through a Tube Screamer soft-clipper circuit. See
[`examples/canonical/02_circuit_to_audio.py`](../examples/canonical/02_circuit_to_audio.py).

```bash
python examples/canonical/02_circuit_to_audio.py
# → examples/canonical/output/02_circuit_ts.wav
```

### Example 3 — Fluid → Acoustics → Audio (3 domains)

Navier-Stokes vortex shedding drives acoustic wave propagation, sampled by virtual microphones into a WAV. See
[`examples/canonical/03_fluid_to_sound.py`](../examples/canonical/03_fluid_to_sound.py).

```bash
python examples/canonical/03_fluid_to_sound.py
# → examples/canonical/output/03_fluid_to_sound.wav
```

---

## The `.morph` DSL (optional)

Morphogen also includes a domain-specific language for describing time-evolving systems. The Python API is the recommended starting point, but the DSL is available:

```bash
morphogen run examples/01_hello_heat.morph   # run a .morph program
morphogen check program.morph                # type-check only
```

DSL programs use `flow` blocks and `@state` declarations. See
[`SPECIFICATION.md`](../SPECIFICATION.md) for the full language reference.

---

## Next Steps

| Goal | Where to go |
|------|-------------|
| Use audio analysis | [`docs/usage/audio_analysis.md`](usage/audio_analysis.md) |
| Build instrument models | [`docs/usage/instrument_model.md`](usage/instrument_model.md) |
| More cross-domain examples | [`examples/canonical/`](../examples/canonical/) |
| See all 39 domains | [`STATUS.md`](../STATUS.md) |
| Understand architecture decisions | [`docs/adr/`](adr/) |
| Full language spec | [`SPECIFICATION.md`](../SPECIFICATION.md) |

---

## Getting Help

- **Examples**: `examples/` directory — dozens of working programs
- **Source**: Each domain is in `morphogen/stdlib/<domain>.py` with full docstrings
- **Issues**: [github.com/scottsen/morphogen/issues](https://github.com/scottsen/morphogen/issues)
