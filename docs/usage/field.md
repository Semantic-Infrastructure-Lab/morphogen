---
title: "Field Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - field
  - field2d
  - heat-diffusion
  - reaction-diffusion
  - pde
---

# Field Domain

The `field` domain provides 2D scalar and vector field operations: heat diffusion,
advection, Laplacian/gradient/divergence/curl, projection, and arbitrary mapping.
All operators return new `Field2D` objects — there is no mutation in place.

## Quick start

```python
import numpy as np
from morphogen.stdlib.field import (
    alloc, diffuse, laplacian, advect,
    gradient, divergence, normalize,
    combine, smooth, threshold, sample,
)
```

---

## Recipe 1 — Heat diffusion

Classic 2D heat equation: an initial hot spot spreading through a grid.

```python
# 128×128 grid, 1 m spacing
field = alloc((128, 128), fill_value=0.0, dx=1.0, dy=1.0)

# Place a heat source at centre
field.data[64, 64] = 1.0

# Step the diffusion equation 200 times
dt = 0.01
for _ in range(200):
    field = diffuse(field, rate=0.1, dt=dt)

# Inspect
print(f"peak: {field.data.max():.4f}")   # energy spreads, peak falls
print(f"mean: {field.data.mean():.6f}")  # total energy conserved
```

`diffuse` supports two solvers (default `"jacobi"`; also `"explicit"`).
Increase `iterations` for tighter convergence at each step.

---

## Recipe 2 — Gradient and Laplacian

```python
from morphogen.stdlib.field import gradient, laplacian

# Gradient: two fields (∂f/∂x, ∂f/∂y)
gx, gy = gradient(field)
# gx.data[i,j] ≈ (f[i,j+1] - f[i,j-1]) / (2·dx)

# Laplacian: ∇²f — used in diffusion, wave equations
lap = laplacian(field)
# lap.data[i,j] ≈ (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4f[i,j]) / dx²
print(f"grad magnitude max: {(gx.data**2 + gy.data**2).max()**.5:.4f}")
```

---

## Recipe 3 — Reaction-diffusion (Gray-Scott)

Two coupled fields — `U` (feed species) and `V` (autocatalyst) — produce
Turing-like patterns via diffuse-and-react dynamics.

```python
from morphogen.stdlib.field import alloc, diffuse, combine, map, normalize, random

# Parameters for "mitosis" pattern
F, k = 0.0545, 0.062
Du, Dv = 0.16, 0.08
dt = 1.0

# Initialise: U=1 everywhere, V seeded randomly in centre
U = alloc((128, 128), fill_value=1.0)
V = alloc((128, 128), fill_value=0.0)
rng = np.random.default_rng(42)
V.data[56:72, 56:72] = rng.random((16, 16)) * 0.25

for _ in range(5000):
    # Gray-Scott reaction: U → V via UV² autocatalysis
    uvv = combine(combine(U, V, "mul"), V, "mul")    # U·V²

    dU = combine(
        diffuse(U, Du, dt),
        combine(uvv, map(U, lambda u: (F + k) * u - F * (1 - u)), "add"),
        "add",
    )
    dV = combine(
        diffuse(V, Dv, dt),
        combine(uvv, map(V, lambda v: -(F + k) * v), "add"),
        "add",
    )
    U = dU
    V = dV

pattern = normalize(V)
print(f"V range: [{V.data.min():.3f}, {V.data.max():.3f}]")
```

---

## Recipe 4 — Advection (fluid-carried transport)

Move a scalar quantity along a velocity field — smoke in wind, dye in flow.

```python
from morphogen.stdlib.field import alloc, advect, combine, normalize, random

# Build a circular velocity field (vortex)
size = 64
vx = alloc((size, size), fill_value=0.0)
vy = alloc((size, size), fill_value=0.0)
cx, cy = size // 2, size // 2
for i in range(size):
    for j in range(size):
        dx_ = j - cx
        dy_ = i - cy
        r = max(1.0, (dx_**2 + dy_**2) ** 0.5)
        vx.data[i, j] =  dy_ / r * 2.0
        vy.data[i, j] = -dx_ / r * 2.0

# Scalar dye blob, off-centre
dye = alloc((size, size), fill_value=0.0)
dye.data[32:40, 20:28] = 1.0

# Advect 60 steps — dye rotates with the vortex
velocity = combine(vx, vy, "add")   # pack as single velocity field
for _ in range(60):
    dye = advect(dye, velocity, dt=0.1)

print(f"dye total (should be stable): {dye.data.sum():.2f}")
```

> **Note on velocity fields**: `advect` expects a `Field2D` whose `.data` encodes
> the (u, v) velocity components. For a pure scalar advection you can pack both
> components into the same call as shown above.

---

## Recipe 5 — Coupling to other domains

Field data flows naturally into agents, audio, and visual domains.

```python
from morphogen.stdlib.field import sample, normalize

# Sample field values at arbitrary (x, y) positions
positions = np.array([[32.5, 32.5], [10.0, 60.0], [55.0, 15.0]])
values = sample(field, positions, method="bilinear")
# → np.ndarray, shape (3,), one float per query position

# Use sampled values to drive agent behaviour
# (see examples/cross_domain/cross_domain_field_agent_coupling.py)
for pos, val in zip(positions, values):
    print(f"  {pos} → field={val:.4f}")
```

Field → audio: export a row of the field as a waveform:

```python
from morphogen.stdlib.audio import AudioBuffer, save_wav

row = normalize(field).data[64, :]           # 1-D slice
waveform = np.tile(row, 44100 // len(row))   # loop to 1 second
buf = AudioBuffer(waveform.astype(np.float32), sample_rate=44100)
save_wav(buf, "/tmp/field_row.wav")
```

---

## Full operator reference

| Operator | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `alloc` | `(shape, dtype, fill_value, dx, dy)` | `Field2D` | Create empty or filled grid |
| `diffuse` | `(field, rate, dt, method, iterations)` | `Field2D` | Methods: `"jacobi"`, `"explicit"` |
| `advect` | `(field, velocity, dt, method)` | `Field2D` | Methods: `"semi_lagrangian"` |
| `laplacian` | `(field)` | `Field2D` | ∇²f |
| `gradient` | `(field)` | `(Field2D, Field2D)` | (∂f/∂x, ∂f/∂y) |
| `divergence` | `(velocity)` | `Field2D` | ∇·v |
| `curl` | `(velocity)` | `Field2D` | ∇×v (z-component) |
| `project` | `(velocity, method, iterations, tolerance)` | `Field2D` | Helmholtz projection — make incompressible |
| `combine` | `(a, b, operation)` | `Field2D` | `"add"`, `"sub"`, `"mul"`, `"div"`, or callable |
| `map` | `(field, func)` | `Field2D` | Apply pointwise: `"square"`, `"sqrt"`, `"abs"`, or callable |
| `boundary` | `(field, spec)` | `Field2D` | `"reflect"`, `"wrap"`, `"zero"` |
| `random` | `(shape, seed, low, high)` | `Field2D` | Uniform random field |
| `smooth` | `(field, iterations, method)` | `Field2D` | Methods: `"gaussian"`, `"box"` |
| `normalize` | `(field, target_min, target_max)` | `Field2D` | Rescale to [min, max] |
| `threshold` | `(field, threshold_value, low_value, high_value)` | `Field2D` | Binary mask |
| `sample` | `(field, positions, method)` | `np.ndarray` | Bilinear/nearest at (x,y) positions |
| `clamp` | `(field, min_value, max_value)` | `Field2D` | Hard clip |
| `abs` | `(field)` | `Field2D` | Pointwise absolute value |
| `magnitude` | `(velocity)` | `Field2D` | √(u² + v²) |

`Field2D` supports Python arithmetic (`+`, `-`, `*`, `/`, unary `-`) directly.

---

## See also

- [`docs/usage/fluid_jet.md`](fluid_jet.md) — free-jet momentum fields on a 2D grid
- [`examples/cross_domain/cross_domain_field_agent_coupling.py`](/examples/cross_domain_field_agent_coupling.py) — agents steered by field gradients
- [`examples/canonical/03_fluid_to_sound.py`](/examples/canonical/03_fluid_to_sound.py) — Navier-Stokes field → acoustics → WAV
- [`morphogen/stdlib/field.py`](/morphogen/stdlib/field.py) — source with full docstrings
