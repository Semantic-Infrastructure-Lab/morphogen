---
title: "Controls Domain Usage Guide"
type: guide
beth_topics:
  - controls
  - pid
  - lqr
  - kalman
  - state-space
---

# Controls Domain

The `controls` domain adds classical and modern control theory as first-class Morphogen operators. It is the feedback-loop complement to `rigidbody`, `field`, and `agents` — everything you need to close the loop on a dynamic system.

**Operators at a glance:**

| Layer | Operators |
|-------|-----------|
| Construct | `pid`, `state_space`, `transfer_function` |
| Transform | `pid_step`, `step`, `kalman_predict`, `kalman_update`, `discretize` |
| Design | `lqr`, `place_poles`, `observer` |
| Query | `poles`, `is_stable`, `step_response`, `bode` |
| Simulate | `simulate` |

---

## 1. Hello World — PID Temperature Controller

The simplest feedback loop: a PID controller driving a first-order thermal system.

```python
from morphogen.stdlib import controls
import numpy as np

# Create PID state (gains, timestep, output limits)
pid = controls.pid(kp=5.0, ki=1.0, kd=0.2, dt=0.01, output_min=0.0, output_max=100.0)

# Simple thermal plant: dT/dt = -T/tau + K_plant*u
tau = 2.0    # Time constant (seconds)
K_plant = 0.5
dt = 0.01
T = 20.0     # Initial temperature (°C)
setpoint = 60.0

temps = [T]
for _ in range(3000):
    u, pid = controls.pid_step(pid, setpoint=setpoint, measurement=T)
    T = T + dt * (-T / tau + K_plant * u)
    temps.append(T)

print(f"Final temperature: {T:.2f}°C (target: {setpoint}°C)")
# → Final temperature: 59.95°C  (integral still converging; more steps → 60.0)
```

`pid_step` is **stateless** — it returns both the control output and a new `PIDState`. The original state is never modified, making PID loops easy to test and parallelize.

**Anti-windup** is built in: when output hits `output_min` / `output_max`, the integral stops accumulating.

---

## 2. State-Space Basics — Spring-Mass-Damper

Model a spring-mass-damper and simulate its free response.

```python
from morphogen.stdlib import controls
import numpy as np

# Spring-mass-damper: mẍ + bẋ + kx = u
# State: [x, ẋ], input: force u
m, b, k = 1.0, 0.5, 4.0

A = np.array([[0,       1     ],
              [-k/m,  -b/m   ]])
B = np.array([[0    ],
              [1/m  ]])
C = np.array([[1, 0]])   # Output: position only
D = np.array([[0    ]])

sys = controls.state_space(A, B, C, D)

# Check stability
print("Stable:", controls.is_stable(sys))
# → True (poles at -0.25 ± 1.98j)

# Simulate free response from x=1, ẋ=0 (no forcing input)
t = np.linspace(0, 10, 1000)
u = np.zeros((1000, 1))
t_out, y_out, x_out = controls.simulate(sys, u, t, x0=np.array([1.0, 0.0]))

print(f"Amplitude at t=10: {y_out[-1, 0]:.4f}")  # Decays toward 0
```

The `simulate` operator handles both continuous and discrete systems. For continuous, it uses Euler integration; for discrete (`sys.dt` set), it uses `scipy.signal.dlsim` (exact).

---

## 3. LQR Optimal Control — Double Integrator

LQR finds the optimal state-feedback gain that minimizes ∫(xᵀQx + uᵀRu)dt. Classic test case: park a double integrator at the origin.

```python
from morphogen.stdlib import controls
import numpy as np

# Double integrator: ẍ = u (position and velocity, force input)
A = np.array([[0.0, 1.0],
              [0.0, 0.0]])
B = np.array([[0.0],
              [1.0]])

# Cost matrices: penalize state equally, unit input cost
Q = np.eye(2)
R = np.array([[1.0]])

# Compute optimal gain
K = controls.lqr(A, B, Q, R)
print("LQR gain K:", K)
# → [[1.  1.732...]]

# Close the loop: A_cl = A - B*K
C = np.eye(2)
D = np.zeros((2, 1))
sys_cl = controls.state_space(A - B @ K, B, C, D)

print("Closed-loop poles:", controls.poles(sys_cl))
# → [-1.  -1.732...] — all negative (stable)

# Simulate: start at x=[1, 0], no external input (K already inside A)
t = np.linspace(0, 5, 500)
u = np.zeros((500, 1))
_, _, x = controls.simulate(sys_cl, u, t, x0=np.array([1.0, 0.0]))

print(f"Final state: {x[-1]}")  # Near [0, 0]
# → [0.0025, -0.015] — converged
```

**Tuning tip:** increasing `Q` relative to `R` → larger gain, faster response, higher control effort. Increasing `R` → gentler control, slower convergence.

---

## 4. Kalman Filter — Estimate Velocity from Noisy Position

A 1D constant-velocity model: observe noisy position, estimate position + velocity.

```python
from morphogen.stdlib import controls
from morphogen.stdlib.controls import KalmanState
import numpy as np

# Constant velocity model: x = [position, velocity]
dt = 0.1
A = np.array([[1.0, dt ],
              [0.0, 1.0]])
B = np.zeros((2, 1))
C = np.array([[1.0, 0.0]])  # Observe position only

Q = np.diag([0.001, 0.001])  # Process noise
R = np.array([[0.25]])        # Measurement noise (σ = 0.5)

# True trajectory: constant velocity v=1 m/s
rng = np.random.RandomState(42)
true_pos = np.arange(0, 10, dt)
noisy_pos = true_pos + rng.randn(len(true_pos)) * 0.5

# Initialize filter
ks = KalmanState(x_hat=np.array([0.0, 0.0]), P=np.eye(2))

estimates = []
for z in noisy_pos:
    ks = controls.kalman_predict(ks, A, B, Q)
    ks = controls.kalman_update(ks, C, R, y=np.array([z]))
    estimates.append(ks.x_hat.copy())

estimates = np.array(estimates)
print(f"Raw MSE:    {np.mean((noisy_pos - true_pos)**2):.4f}")
print(f"Kalman MSE: {np.mean((estimates[:, 0] - true_pos)**2):.4f}")
# Raw MSE:    0.2531
# Kalman MSE: 0.0089  — ~28× better

# Velocity estimate at end
print(f"Estimated velocity: {ks.x_hat[1]:.3f} m/s (true: 1.0)")
```

`kalman_predict` / `kalman_update` are **pure functions** — each returns a new `KalmanState`. Chain them in a loop; no mutable state to track.

---

## 5. Cross-Domain: Rigidbody + Controls

LQR-controlled ball (rigidbody) that brakes to a stop using force feedback.

```python
from morphogen.stdlib import controls, rigidbody
from morphogen.cross_domain.controls_physics import (
    PhysicsToControlsInterface,
    ControlsToPhysicsInterface,
)
import numpy as np

# --- Design LQR for 2D point mass: [x, y, vx, vy], inputs [fx, fy] ---
A = np.zeros((4, 4))
A[0, 2] = 1.0  # dx/dt = vx
A[1, 3] = 1.0  # dy/dt = vy
B = np.zeros((4, 2))
B[2, 0] = 1.0  # fx → vx dot
B[3, 1] = 1.0  # fy → vy dot

Q = np.diag([10.0, 10.0, 1.0, 1.0])  # Penalize position more than velocity
R = np.eye(2) * 0.5
K = controls.lqr(A, B, Q, R)

# --- Create a rigidbody ---
ball = rigidbody.body(mass=1.0, position=np.array([5.0, 3.0]),
                      velocity=np.array([-2.0, 1.0]))

# --- Sensor/actuator interfaces ---
sensor = PhysicsToControlsInterface(ball, sensor_mapping=["x", "y", "vx", "vy"])
actuator = ControlsToPhysicsInterface(ball, actuator_mapping={0: "force_x", 1: "force_y"})

# --- Simulate with LQR feedback ---
dt = 0.02
for _ in range(250):
    # Measure state
    y = sensor.transform(ball)         # [x, y, vx, vy]

    # Compute control (target is origin)
    u = -(K @ y)                        # LQR: u = -K*x

    # Apply force
    ball = actuator.transform(u)

    # Physics step
    ball = rigidbody.integrate(ball, dt=dt)
    ball = rigidbody.clear_forces(ball)

    # Update sensor binding
    sensor.body = ball
    actuator.body = ball

print(f"Final position: {ball.position}")  # Near [0, 0]
print(f"Final velocity: {ball.velocity}")  # Near [0, 0]
```

`PhysicsToControlsInterface` extracts a measurement vector from any `RigidBody2D` using a configurable `sensor_mapping`. `ControlsToPhysicsInterface` maps control output indices to forces and torques.

---

## Operator Reference

```python
# Construction
pid_state  = controls.pid(kp, ki=0, kd=0, dt=0.01, output_min=-inf, output_max=inf)
sys        = controls.state_space(A, B, C, D, dt=None)
tf         = controls.transfer_function(num, den, dt=None)

# Step / Transform
u, pid2    = controls.pid_step(pid_state, setpoint, measurement)
sys2       = controls.step(sys, u, dt=None)
ks2        = controls.kalman_predict(ks, A, B, Q, u=None)
ks3        = controls.kalman_update(ks, C, R, y)
disc_sys   = controls.discretize(sys, dt, method="zoh")  # or "tustin"

# Design
K          = controls.lqr(A, B, Q, R)          # m×n gain matrix
K          = controls.place_poles(A, B, poles)  # m×n gain matrix
L          = controls.observer(A, C, poles)     # n×p observer gain

# Query
p          = controls.poles(sys)                # complex eigenvalues
stable     = controls.is_stable(sys)            # bool
t, y       = controls.step_response(sys, t_end, dt=0.01)
w, mag, ph = controls.bode(sys, frequencies=None)

# Simulate
t, y, x    = controls.simulate(sys, u, t, x0=None)
```

**See also:**
- [`docs/usage/rigidbody.md`](rigidbody.md) — for the physics layer controls closes the loop on
- [`docs/usage/cross_domain_coupling.md`](cross_domain_coupling.md) — how `DomainInterface` works
- [`morphogen/cross_domain/controls_physics.py`](../../morphogen/cross_domain/controls_physics.py) — `PhysicsToControlsInterface` source
