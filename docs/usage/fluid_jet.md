---
title: "Fluid Jet Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - fluid_jet
  - jets
  - combustion
  - fluid-dynamics
  - entrainment
---

# Fluid Jet Domain

The `fluid_jet` domain models free turbulent jets: momentum flux, Reynolds number,
centerline velocity decay, entrainment, and spatial spreading. Jets can be arranged
in arrays (burner rings, spray nozzles) and projected onto 2D field grids for
coupling with the `field` domain.

## Quick start

```python
import numpy as np
from morphogen.stdlib.fluid_jet import (
    jet_from_tube, jet_reynolds, jet_entrainment,
    jet_field_2d, jet_centerline_velocity, jet_spreading_width,
    create_jet_array_radial, JetArray,
)
```

---

## Recipe 1 — Single free jet

Create a jet and characterise its near-field behaviour.

```python
# 20mm tube, 50 g/s mass flow, 300 K air
jet = jet_from_tube(
    tube_diameter=0.02,                      # m
    tube_position=(0.0, 0.0, 0.0),
    tube_direction=(0.0, 0.0, 1.0),          # upward
    m_dot=0.05,                              # kg/s
    T_out=300.0,                             # K
)

print(f"Exit velocity: {jet.velocity:.2f} m/s")
print(f"Momentum flux: {jet.momentum_flux():.4f} kg·m/s²")
print(f"Diameter:      {jet.diameter():.4f} m")

# Turbulence regime
Re = jet_reynolds(jet, mu=1.8e-5)   # dynamic viscosity of air at 300 K
print(f"Reynolds number: {Re:.0f}  ({'turbulent' if Re > 4000 else 'laminar'})")

# Centerline velocity decay: ~1/x for turbulent round jet
print("\nCenterline velocity vs. distance:")
for x in [0.01, 0.05, 0.1, 0.2, 0.5]:
    v = jet_centerline_velocity(jet, distance=x)
    w = jet_spreading_width(jet, distance=x)
    print(f"  x={x:.2f} m  v_cl={v:.2f} m/s  half-width={w:.4f} m")
```

---

## Recipe 2 — Entrainment

Turbulent jets entrain surrounding fluid (plume) — critical for combustion,
mixing, and exhaust modelling.

```python
# Estimate mass entrainment rate at a cross-section
# plume conditions: 2 m/s upward, air density at 1200 K
plume_velocity  = 2.0    # m/s
plume_density   = 0.29   # kg/m³ (hot combustion gas at 1200 K)

mdot_entrained = jet_entrainment(
    jet,
    plume_velocity=plume_velocity,
    plume_density=plume_density,
    model="empirical",     # "empirical" | "integral"
)
print(f"Entrained mass flow: {mdot_entrained:.4f} kg/s")
print(f"Entrainment ratio:   {mdot_entrained / jet.m_dot:.2f}  (entrained / injected)")
```

---

## Recipe 3 — Radial jet array

Burner rings and spray nozzles use radially-arranged jets. Each jet in the array
points inward at a configurable angle, creating a converging flow pattern.

```python
# 8-jet radial burner ring, 50 mm radius
n_jets    = 8
ring_r    = 0.05      # m
d_nozzle  = 0.005     # 5 mm nozzle
m_per_jet = 0.003     # 3 g/s each

jet_array = create_jet_array_radial(
    n_jets=n_jets,
    radius=ring_r,
    jet_diameter=d_nozzle,
    m_dot_per_jet=m_per_jet,
    temperature=600.0,      # K
    height=0.0,
    angle_inward=15.0,      # degrees toward axis
)

print(f"Total jets: {jet_array.count()}")
print(f"Total mass flow: {jet_array.total_flow()*1000:.1f} g/s")
for i, j in enumerate(jet_array.jets):
    print(f"  jet {i}: pos={j.position}, vel={j.velocity:.1f} m/s")
```

---

## Recipe 4 — Project jets onto a 2D field

`jet_field_2d` maps a `JetArray` to a 2D scalar grid — momentum density or
temperature perturbation. This field slots directly into the `field` domain.

```python
from morphogen.stdlib.field import alloc, combine, normalize

# 200×200 grid, covering 0.2m × 0.2m
grid_size   = (200, 200)
grid_bounds = (-0.1, 0.1, -0.1, 0.1)   # (x_min, x_max, y_min, y_max)

vel_field = jet_field_2d(
    jet_array,
    grid_size=grid_size,
    grid_bounds=grid_bounds,
    decay=0.05,           # exponential decay coefficient
)
# vel_field: np.ndarray, shape (200, 200), velocity magnitude [m/s]

# Wrap in Field2D for further processing
from morphogen.stdlib.field import Field2D
f = Field2D(vel_field)
f_norm = normalize(f)

print(f"Peak velocity in field: {vel_field.max():.2f} m/s")
print(f"Coverage (>0.01 m/s): {(vel_field > 0.01).sum()} cells")
```

The resulting field can be diffused, advected, or used to drive agent forces
in a multi-domain simulation.

---

## Recipe 5 — Coupling to acoustics

Turbulent jets generate aeroacoustic noise — use the field projection as a
source term for the `acoustics` domain:

```python
from morphogen.stdlib.acoustics import propagate_pressure_wave

# Velocity fluctuations are sources of acoustic pressure
source_field = vel_field / vel_field.max()   # normalise to [0, 1]

# Acoustic propagation over the same grid
speed_of_sound = 343.0   # m/s
pressure_out = propagate_pressure_wave(
    source_field=source_field,
    speed_of_sound=speed_of_sound,
    grid_dx=0.001,           # 1 mm grid spacing
    dt=1e-6,
    steps=100,
)
print(f"peak pressure: {pressure_out.max():.4f} Pa")
```

---

## Full operator reference

| Operator | Returns | Notes |
|----------|---------|-------|
| `jet_from_tube(tube_diameter, tube_position, tube_direction, m_dot, T_out, rho)` | `Jet` | `rho` optional (derived from ideal gas if omitted) |
| `jet_reynolds(jet, mu)` | `float` | Dynamic viscosity `mu` default 1.8×10⁻⁵ Pa·s (air, 300 K) |
| `jet_entrainment(jet, plume_velocity, plume_density, model)` | `float` | Entrained kg/s |
| `jet_centerline_velocity(jet, distance)` | `float` | m/s at axial distance |
| `jet_spreading_width(jet, distance, spreading_rate)` | `float` | Half-width in m |
| `create_jet_array_radial(n_jets, radius, jet_diameter, m_dot_per_jet, temperature, height, angle_inward)` | `JetArray` | |
| `jet_field_2d(jet_array, grid_size, grid_bounds, decay)` | `np.ndarray` | 2D velocity magnitude field |

`Jet` properties: `velocity`, `m_dot`, `temperature`, `position`, `direction`,
`momentum_flux()`, `diameter()`.

`JetArray` properties: `jets` (list of `Jet`), `count()`, `total_flow()`.

---

## See also

- [`docs/usage/field.md`](field.md) — `Field2D` for downstream processing of jet fields
- [`examples/canonical/03_fluid_to_sound.py`](/examples/canonical/03_fluid_to_sound.py) — Navier-Stokes field → acoustics → audio
- [`examples/fluids/jet_dynamics_demo.py`](/examples/fluids/jet_dynamics_demo.py) — extended jet dynamics demo
- [`morphogen/stdlib/fluid_jet.py`](/morphogen/stdlib/fluid_jet.py) — source with full docstrings
