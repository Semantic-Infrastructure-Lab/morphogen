---
title: "Rigidbody Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - rigidbody
  - physics
  - collision
  - rigid-body-simulation
---

# Rigidbody Domain

The `rigidbody` domain is a 2D discrete-event physics engine: circles and boxes,
impulse-based collision resolution, gravity, damping, and raycasting. The world
is immutable — `step_world` returns a new `PhysicsWorld2D` each frame.

## Quick start

```python
import numpy as np
from morphogen.stdlib.rigidbody import (
    create_circle_body, create_box_body,
    PhysicsWorld2D, step_world,
    detect_collisions, apply_force, apply_impulse,
)
```

---

## Recipe 1 — Bouncing ball

The minimal simulation: one ball, gravity, 60 Hz timestep.

```python
# World with Earth gravity
world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]), dt=1/60.0)

# Ball: 1 kg circle, radius 0.5 m, starts at y=5
ball = create_circle_body(
    position=np.array([0.0, 5.0]),
    radius=0.5,
    mass=1.0,
    restitution=0.8,    # bounce coefficient (0=dead, 1=elastic)
    friction=0.2,
)
ball_id = world.add_body(ball)

# Simulate 2 seconds (120 frames)
for frame in range(120):
    world = step_world(world)

body = world.get_body(ball_id)
print(f"position after 2s: {body.position}")
print(f"velocity after 2s: {body.velocity}")
```

**Static bodies** (floors, walls) have `mass=0` — they participate in collision
resolution but are never moved:

```python
floor = create_box_body(
    position=np.array([0.0, 0.0]),
    width=20.0, height=0.5,
    mass=0.0,           # static — infinite mass
)
world.add_body(floor)
```

---

## Recipe 2 — Collision detection

`detect_collisions` returns all active contacts in the current world state.
Use this to trigger sound, spawn particles, log events, etc.

```python
# Two balls on a collision course
world = PhysicsWorld2D(gravity=np.array([0.0, 0.0]), dt=1/60.0)

a = create_circle_body(np.array([-3.0, 0.0]), 0.5, mass=1.0, restitution=0.9)
b = create_circle_body(np.array([ 3.0, 0.0]), 0.5, mass=1.0, restitution=0.9)

# Give them opposing velocities
a.velocity = np.array([ 2.0, 0.0])
b.velocity = np.array([-2.0, 0.0])

id_a = world.add_body(a)
id_b = world.add_body(b)

collision_frames = []
for frame in range(300):
    contacts = detect_collisions(world)
    if contacts:
        # Contact carries: body_a_id, body_b_id, normal, penetration, point
        for c in contacts:
            collision_frames.append({
                "frame": frame,
                "penetration": c.penetration,
                "normal": c.normal.tolist(),
            })
    world = step_world(world)

print(f"collisions detected: {len(collision_frames)}")
if collision_frames:
    c = collision_frames[0]
    print(f"  first at frame {c['frame']}, penetration {c['penetration']:.4f}")
```

---

## Recipe 3 — Applying forces and impulses

```python
world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]), dt=1/120.0)

ball = create_circle_body(np.array([0.0, 1.0]), 0.3, mass=0.5)
bid = world.add_body(ball)

# Per-frame force (N) — accumulated until next step, then zeroed
body = world.get_body(bid)
body = apply_force(body, force=np.array([5.0, 0.0]))   # push right
world.bodies[0] = body

# Instantaneous impulse (kg·m/s) — use for explosions, kicks
body = world.get_body(bid)
body = apply_impulse(body, impulse=np.array([0.0, 3.0]))  # jump
world.bodies[0] = body

for _ in range(60):
    world = step_world(world)

print(f"final pos: {world.get_body(bid).position}")
```

`apply_force` and `apply_impulse` accept an optional `point` argument (world
coordinates). Off-centre application generates torque.

---

## Recipe 4 — Collision events to audio

This is the same pattern used by
[`examples/canonical/01_physics_to_audio.py`](/examples/canonical/01_physics_to_audio.py).
Each collision's impact velocity maps to a sound event.

```python
import numpy as np
from morphogen.stdlib.rigidbody import (
    PhysicsWorld2D, create_circle_body, create_box_body,
    step_world, detect_collisions,
)
from morphogen.stdlib.audio import AudioBuffer, save_wav
from morphogen.stdlib.audio import synthesize_tone

SR = 44100
world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]), dt=1/SR)

floor = create_box_body(np.array([0.0, -0.25]), 10.0, 0.5, mass=0.0)
ball  = create_circle_body(np.array([0.0, 3.0]), 0.2, mass=1.0, restitution=0.75)
world.add_body(floor)
world.add_body(ball)

audio_events = []
prev_vel = np.array([0.0, 0.0])

for frame in range(SR):     # 1 second at physics rate
    contacts = detect_collisions(world)
    body = world.bodies[1]  # ball
    if contacts:
        impact_speed = float(np.linalg.norm(prev_vel))
        if impact_speed > 0.1:
            audio_events.append({
                "sample": frame,
                "speed": impact_speed,
                "pitch": 220.0 * (1 + impact_speed * 0.5),
            })
    prev_vel = body.velocity.copy()
    world = step_world(world)

print(f"impact events: {len(audio_events)}")
for e in audio_events:
    print(f"  t={e['sample']/SR:.3f}s  speed={e['speed']:.2f}  pitch={e['pitch']:.1f} Hz")
```

---

## Recipe 5 — Raycasting

Fire a ray from an origin in a given direction; returns the first body hit.

```python
from morphogen.stdlib.rigidbody import raycast

world = PhysicsWorld2D(gravity=np.array([0.0, 0.0]), dt=1/60.0)
target = create_circle_body(np.array([5.0, 0.0]), 0.5, mass=1.0)
world.add_body(target)

result = raycast(
    world,
    origin=np.array([0.0, 0.0]),
    direction=np.array([1.0, 0.0]),   # rightward
    max_distance=20.0,
)

if result is not None:
    hit_body, hit_point, distance = result
    print(f"hit at {hit_point}, distance={distance:.3f}")
else:
    print("no hit")
```

---

## Full operator reference

| Operator | Returns | Notes |
|----------|---------|-------|
| `create_circle_body(position, radius, mass, restitution, friction)` | `RigidBody2D` | Circle shape |
| `create_box_body(position, width, height, mass, restitution, friction)` | `RigidBody2D` | AABB box |
| `step_world(world, dt)` | `PhysicsWorld2D` | Full physics step: integrate → collide → resolve |
| `detect_collisions(world)` | `List[Contact]` | All contacts this frame |
| `resolve_collision(body_a, body_b, contact)` | `(RigidBody2D, RigidBody2D)` | Impulse resolution |
| `apply_force(body, force, point)` | `RigidBody2D` | Accumulated per frame |
| `apply_impulse(body, impulse, point)` | `RigidBody2D` | Instantaneous Δv |
| `integrate_body(body, dt, damping)` | `RigidBody2D` | Semi-implicit Euler |
| `get_body_vertices(body)` | `np.ndarray` or `None` | Box corner positions |
| `raycast(world, origin, direction, max_distance)` | `(body, point, dist)` or `None` | First hit |

`Contact` fields: `body_a_id`, `body_b_id`, `normal` (unit), `penetration` (m), `point`.

`RigidBody2D` fields: `position`, `velocity`, `angle`, `angular_velocity`, `mass`,
`restitution`, `friction`, `shape`, `is_static()`.

`PhysicsWorld2D` constructor: `gravity` (default `[0, −9.81]`), `dt` (default `1/60`).

---

## See also

- [`examples/canonical/01_physics_to_audio.py`](/examples/canonical/01_physics_to_audio.py) — collision events → WAV
- [`examples/rigidbody_physics/`](/examples/rigidbody_physics/) — additional physics demos
- [`morphogen/stdlib/rigidbody.py`](/morphogen/stdlib/rigidbody.py) — source with full docstrings
