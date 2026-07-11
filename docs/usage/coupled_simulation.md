---
title: "Coupled simulation — co-advancing rigorous domains with feedback"
type: guide
beth_topics:
  - morphogen
  - coupling
  - co-simulation
  - coupled-multiphysics
  - cross-domain
  - feedback
  - controls
  - integrators
---

# Coupled simulation (`morphogen.coupling`)

Most of Morphogen is a library of **rigorous, independently-validated domains** —
`integrators`, `controls`, `circuit`, `thermo`, and friends, each checked against
analytic ground truth. `morphogen.coupling` adds the one coordinating layer they
were missing: a small driver that **co-advances several of those domains with
per-timestep feedback**, so their states influence each other every step.

This is what "model an engine — heat + mechanics + control together" actually
needs. It is *not* the retired cross-domain composer: that plumbed data through a
domain graph once (A→B→C). This closes a **loop** — B reads A's current output, A
then advances using B's response — every timestep, across domains that each hold
their own state.

## The idea in one loop

A **subsystem** owns some state and knows how to advance it by one timestep,
reading coupling signals published by the others and publishing its own. The
`couple(...)` driver steps them all, timestep by timestep, routing signals over a
shared bus.

```python
from morphogen.coupling import Subsystem, couple
import numpy as np

# A trivial first-order plant driven by a proportional controller.
def plant(x, sig, t, dt):
    u = sig.get("u", 0.0)              # read the controller's output (zero-order hold)
    x = x + dt * (-x + u)             # advance the plant one step
    return x, {"y": x}               # publish the measurement

def controller(_, sig, t, dt):
    err = 1.0 - sig.get("y", 0.0)     # hold y at the setpoint 1.0
    return None, {"u": 2.0 * err}     # publish the actuation

result = couple(
    [Subsystem("plant", 0.0, plant),
     Subsystem("controller", None, controller)],
    steps=500, dt=0.01, record=("y", "u"),
)

print(result["y"][-1])   # -> ~0.667, the closed-loop steady state
```

### The `step` contract

Each subsystem's `step` is a pure function:

```
step(state, signals, t, dt) -> (new_state, published_signals)
```

- `state` is opaque to the driver — a float, a numpy array, a domain dataclass
  (e.g. a `PIDState`), whatever the wrapped domain uses.
- `signals` is the read-only shared bus. Read your inputs with `signals.get(name, default)`.
- Return the updated state and a dict of the signals you want on the bus.

### Scheme: sequential, with a zero-order hold

Subsystems step **in list order**. Each reads the bus as it stands — holding last
step's signals plus anything already refreshed by an earlier subsystem this step
(a sequential / Gauss–Seidel co-simulation, with a zero-order hold on the coupling
signals). So list order is meaningful and lets you write a sensor → controller →
plant chain directly:

```python
couple([controller, plant], ...)   # plant reads the controller's fresh output this step
```

Use `initial_signals=` to seed the bus so a consumer that runs before its producer
on step 0 still sees a defined input.

## A real example: coupled multiphysics engine

`experiments/coupled-feedback/coupled_engine.py` couples **three** rigorous
domains — thermal, mechanics, control — into a cold-start warm-up:

| Subsystem | Domain | State | Reads | Publishes |
|-----------|--------|-------|-------|-----------|
| `controls`  | `controls` PID          | `PIDState` | `rpm`             | `throttle` |
| `thermal`   | `integrators` (RK4)     | block temp `T` | `throttle`, `omega` | `temp` |
| `mechanics` | `integrators` (RK4)     | crank speed `omega` | `throttle`, `temp` | `omega`, `rpm` |

The couplings that make it genuine multiphysics: **drag depends on temperature**
(cold oil drags harder) and **temperature depends on the work done** (combustion +
friction heat the block). So all three states feed back on each other every step.
The governor floors the throttle while the engine is cold and high-drag, then
self-relaxes to ~60% as the block warms and drag falls — the classic
fast-idle-then-settle warm-up — holding idle throughout. You can't get that from
any single-domain library.

## Multi-rate

Give a subsystem `stride=N` to advance it only every `N` ticks (its last outputs
are held between advances) — e.g. a fast plant sampled by a slower controller:

```python
couple([Subsystem("plant", x0, plant),               # every tick
        Subsystem("controller", None, ctrl, stride=10)],  # every 10th tick
       steps=1000, dt=1e-4)
```

## What you get back

`couple(...)` returns a `CoupledResult`:

- `result.signals` — the final signal bus.
- `result.history` — `{name: np.ndarray[steps]}` for each `record`ed signal
  (`result["name"]` is a shortcut).
- `result.states` — `{subsystem_name: final_state}`.
- `result.times` — sample times.

## When to use this vs. a single domain

Reach for `couple(...)` when **two or more domains have to advance together and
influence each other over time** — a plant and its controller, thermal and
mechanical states, a circuit driving and loaded by a mechanism. For a one-shot
"transform this into that" you don't need it: instantiate the relevant domain or
cross-domain bridge directly.

## See also

- `morphogen/coupling/substrate.py` — the driver (small; read it).
- `tests/test_coupling.py` — the contract and the value claim, as tests.
- [`docs/reviews/2026-07-11-vision-vs-reality.md`](../reviews/2026-07-11-vision-vs-reality.md)
  — why this is the layer the vision always needed.
- [`docs/usage/cross_domain_coupling.md`](cross_domain_coupling.md) — the one-shot
  cross-domain *bridges* (kept; distinct from this loop-closing substrate).
