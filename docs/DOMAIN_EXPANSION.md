---
title: "Morphogen: Domain Expansion Plan (Post-v1.0)"
type: reference
beth_topics:
  - morphogen
  - strategy
  - domains
  - roadmap
  - controls
  - fem
  - orbital
  - epidemiology
  - photonics
  - quantum
  - robotics
  - economics
---

# Morphogen: Domain Expansion Plan (Post-v1.0)

**Date:** 2026-03-16
**Context:** v1.0 ships with 39 domains. This doc evaluates the highest-value additions for v1.1+.
**Related:** [STRATEGY.md](STRATEGY.md) · [ROADMAP.md](ROADMAP.md) · [STATUS.md](../STATUS.md)

---

## Evaluation Criteria

Each candidate domain is scored on three axes:

| Axis | Question |
|------|----------|
| **Usefulness** | How many researchers/engineers need this? Does it solve real problems? |
| **Cross-domain leverage** | How many compelling new compositions does it unlock? |
| **Implementation effort** | How much work to reach production quality (ops, tests, tutorial)? |

---

## Tier 1 — High Impact, Strong Cross-Domain Stories

These domains address large, underserved user populations and unlock cross-domain compositions that are genuinely compelling — the kind of thing no other platform makes easy.

---

### `controls` — Control Systems

**What it is:** State-space control theory. PID loops, LQR/LQE optimal control, Kalman filter state estimation, transfer functions, Bode/Nyquist plots, pole-zero analysis, discretization.

**Why it matters:** Every mechatronics, robotics, and process control engineer needs this. It's the missing link between simulation (`rigidbody`, `field`) and actuation — you can simulate a system but you can't close the loop without a controller.

**Key operators:**
```python
from morphogen.stdlib.controls import (
    pid,                 # PID(Kp, Ki, Kd) → controller
    lqr,                 # state matrix A, B + cost Q, R → gain K
    kalman_filter,       # process/measurement noise → state estimator
    state_space,         # (A, B, C, D) → LinearSystem
    transfer_function,   # num/den → TF
    bode,                # TF/SS → frequency response
    step_response,       # system → time-domain step response
    discretize,          # continuous SS → discrete SS (ZOH, Tustin)
    simulate,            # system + input signal → output signal
    pole_zero,           # TF/SS → poles and zeros
)
```

**Cross-domain compositions unlocked:**

| Pipeline | Story |
|----------|-------|
| `rigidbody` + `controls` | Simulated robot arm with feedback control — design the controller, watch it stabilize |
| `field` + `controls` | Flow control in a PDE field (active damping of fluid instabilities) |
| `agents` + `controls` | Multi-agent formation control — each agent runs its own LQR |
| `rigidbody` + `controls` + `audio` | Ball-on-beam demo: controller sound when it falls, silence when stable |
| `thermal_ode` + `controls` | Thermostat: model a room, design a PID, hear the convergence as audio |

**Effort:** Medium. SciPy's `control`-adjacent tools exist but have an inconsistent API. Morphogen needs clean operator semantics. ~150 operators, 200 tests.

**Dependencies:** `scipy.signal`, `numpy`

---

### `fem` — Finite Element Methods

**What it is:** Structural mechanics via FEM. Mesh generation, element assembly, boundary conditions, linear static analysis, modal analysis, heat conduction, beam/plate/shell elements.

**Why it matters:** Mechanical and civil engineers use FEM constantly. It's the most underserved domain on any Python platform — PyFEM and FEniCS exist but are heavyweight dependencies with steep learning curves. A Morphogen-style clean API would stand out.

**Key operators:**
```python
from morphogen.stdlib.fem import (
    mesh_from_geometry,  # geometry → FEMesh
    add_element,         # mesh + element type → meshed domain
    apply_bc,            # mesh + boundary condition → constrained mesh
    apply_load,          # mesh + force/pressure → loaded mesh
    assemble,            # mesh → K (stiffness), M (mass), F (force)
    solve_static,        # K, F → displacement field
    solve_modal,         # K, M → eigenvalues, mode shapes
    solve_transient,     # K, M, F, dt → time-domain response
    von_mises_stress,    # displacement field → stress field
    natural_frequency,   # K, M → Hz
)
```

**Cross-domain compositions unlocked:**

| Pipeline | Story |
|----------|-------|
| `geometry` + `fem` | Design a shape, mesh it, analyze stress — CAD to analysis in one pipeline |
| `fem` + `audio` | Modal synthesis: compute natural frequencies of a plate → synthesize its "ping" |
| `thermal_ode` + `fem` | Thermal stress: heat distribution → mechanical deformation |
| `optimization` + `fem` | Topology optimization: evolve the best beam shape for a given load |
| `fem` + `visual3d` | Visualize displacement and stress on a 3D mesh |

**Effort:** Large. FEM is genuinely complex — element formulations, sparse assembly, BC handling. Likely needs a focused sprint. ~200 operators, 250 tests.

**Dependencies:** `numpy`, `scipy.sparse`, optionally `meshio`

---

### `orbit` — Orbital Mechanics

**What it is:** Astrodynamics and orbital mechanics. Kepler's equations, two-body problem, N-body integration, orbital element conversions, maneuver planning (Hohmann, bi-elliptic), ground track computation, SGP4 propagation.

**Why it matters:** Space engineering is a growing field. No other Python platform treats orbital mechanics as a first-class composable domain alongside signal processing and audio. The cross-domain story (orbit + controls + rigidbody = spacecraft attitude control) is unique.

**Key operators:**
```python
from morphogen.stdlib.orbit import (
    keplerian,           # (a, e, i, Ω, ω, ν) → OrbitalState
    propagate,           # state + dt → state (Cowell / SGP4)
    hohmann_transfer,    # r1, r2 → delta-v burns
    period,              # semimajor axis → orbital period
    vis_viva,            # r, a, mu → velocity
    ground_track,        # state + time → (lat, lon) array
    eclipse_fraction,    # orbit → fraction in shadow
    n_body,              # [Body] + dt → integrated positions
    rv_to_elements,      # position/velocity → orbital elements
    elements_to_rv,      # orbital elements → position/velocity
)
```

**Cross-domain compositions unlocked:**

| Pipeline | Story |
|----------|-------|
| `orbit` + `controls` | Spacecraft attitude control: orbit dynamics + PID/LQR stabilization |
| `orbit` + `rigidbody` | Debris collision risk: propagate two objects, detect close approaches |
| `orbit` + `optimization` | Trajectory optimization: find minimum delta-v between orbits |
| `orbit` + `audio` | Sonify an orbit: map altitude to pitch, eclipse to silence |
| `orbit` + `visual3d` | 3D orbital visualization with ground tracks |

**Effort:** Medium. The math is well-defined; reference implementations exist (Astropy, Poliastro). Clean Morphogen operator wrappers are the work. ~100 operators, 150 tests.

**Dependencies:** `numpy`, `scipy.integrate`; optionally `astropy` for SGP4

---

### `epidemiology` — Population Dynamics

**What it is:** Compartmental epidemic models (SIR, SEIR, SIRD, SEIRD), Lotka-Volterra predator-prey, demographic models, stochastic disease spread, network-based transmission.

**Why it matters:** Post-2020 the audience for epidemic modeling is large — public health researchers, policy analysts, biology students. Lotka-Volterra and SIR are also the canonical examples in every nonlinear dynamics textbook. The `agents` + `epidemiology` + `graph` composition (network epidemiology) is genuinely novel.

**Key operators:**
```python
from morphogen.stdlib.epidemiology import (
    sir,                 # beta, gamma, N → SIRModel
    seir,                # beta, sigma, gamma, N → SEIRModel
    simulate_ode,        # model + initial conditions + t_span → trajectory
    lotka_volterra,      # alpha, beta, delta, gamma → LVModel
    basic_reproduction,  # model → R0
    herd_immunity,       # R0 → threshold fraction
    stochastic_sir,      # parameters + seed → Monte Carlo trajectories
    age_structured,      # contact matrix + age groups → structured model
    network_sir,         # graph + beta + gamma → network epidemic
    fit_to_data,         # observed + model → fitted parameters
)
```

**Cross-domain compositions unlocked:**

| Pipeline | Story |
|----------|-------|
| `epidemiology` + `agents` | Agent-based epidemic: each agent is a person, disease spreads on contact |
| `epidemiology` + `graph` | Network epidemiology: spread on a social network (scale-free, random, geographic) |
| `epidemiology` + `optimization` | Optimal intervention: find the vaccination strategy that minimizes deaths |
| `epidemiology` + `audio` | Sonify an outbreak: map infected count to drone pitch + reverb |
| `epidemiology` + `statemachine` | Policy automaton: lockdown triggers at threshold, relaxes at another |

**Effort:** Small-Medium. ODE integration is already solid via `integrators`. The core models are clean ODEs; the operators are thin wrappers with good naming. ~80 operators, 120 tests.

**Dependencies:** `scipy.integrate`, `numpy`; graph ops already in `graph`

---

## Tier 2 — Strong Value, More Specialized Audience

---

### `photonics` — Physical Optics

**What it is:** Physical (wave) optics: ray tracing, diffraction, Gaussian beams, waveguides, optical fiber, Fabry-Pérot cavities, spectral transmission.

**Why it matters:** Optics research and photonics engineering. Also enables a cross-domain story that no competitor has: `combustion_light` → `photonics` = model a flame and measure its emission spectrum through an optical system.

**Key operators:**
```python
from morphogen.stdlib.photonics import (
    ray,                 # origin + direction + wavelength → Ray
    trace,               # rays + scene → propagated rays
    gaussian_beam,       # waist + wavelength → GaussianBeam
    propagate_beam,      # beam + distance → beam
    diffract,            # aperture + beam → diffracted field
    fresnel,             # n1, n2, angle → (R, T) reflectance/transmittance
    spectrum,            # wavelengths + intensities → Spectrum
    thin_lens,           # focal_length → OpticalElement
    waveguide,           # n_core, n_clad, geometry → guided modes
    fabry_perot,         # mirror_R + cavity_length → transmission vs frequency
)
```

**Cross-domain coupling highlight:** `combustion_light` → `photonics` → `signal` = simulate a flame, compute its spectral emission, pass it through a spectrometer model, output detected spectrum. This is physically meaningful and unique.

**Effort:** Medium-Large. Physical optics has real complexity (diffraction integrals, mode solvers). ~120 operators, 160 tests.

---

### `quantum` — Quantum Computing

**What it is:** Quantum circuits and gate-level simulation. Qubits, unitary gates, measurement, state vector and density matrix evolution, entanglement metrics, variational circuits (VQE/QAOA structure).

**Note:** Distinct from `qchem` (quantum *chemistry*). This domain is quantum *computing* — qubits and gates, not molecular orbitals.

**Key operators:**
```python
from morphogen.stdlib.quantum import (
    qubit,               # n → zero-state n-qubit register
    apply_gate,          # register + gate + targets → register
    measure,             # register + basis → outcome + collapsed state
    hadamard, pauli_x, pauli_z, cnot, toffoli,  # standard gates
    bloch_vector,        # single-qubit state → (x, y, z) on Bloch sphere
    fidelity,            # state1, state2 → overlap
    entanglement_entropy,# bipartite state → von Neumann entropy
    circuit,             # list of (gate, targets) → Circuit
    simulate,            # circuit + initial state → final state
    variational,         # ansatz + params → parametrized circuit
)
```

**Cross-domain coupling highlight:** `quantum` + `optimization` = QAOA — quantum-inspired optimization where the optimizer (CMA-ES, DE) tunes variational circuit parameters to minimize a cost Hamiltonian.

**Effort:** Medium. State vector simulation is well-understood. ~100 operators, 150 tests. Can avoid the full density matrix path for v1 of this domain.

---

### `robotics` — Kinematics & Motion Planning

**What it is:** Robot kinematics (forward + inverse), Denavit-Hartenberg parameters, workspace analysis, motion planning (RRT, PRM), trajectory interpolation, URDF parsing.

**Why it matters:** Robotics is one of the highest-demand engineering domains. The `rigidbody` + `controls` + `robotics` stack would be the most complete robot simulation suite in Python outside of ROS.

**Key operators:**
```python
from morphogen.stdlib.robotics import (
    dh_chain,            # DH parameters → KinematicChain
    forward_kinematics,  # chain + joint_angles → end_effector_pose
    inverse_kinematics,  # chain + target_pose → joint_angles (numerical IK)
    jacobian,            # chain + angles → Jacobian matrix
    workspace,           # chain → reachable workspace (sampled)
    rrt,                 # start + goal + obstacles → path
    prm,                 # space + obstacles → roadmap
    trajectory_cubic,    # waypoints + timing → cubic spline trajectory
    load_urdf,           # urdf_file → KinematicChain
    collision_check,     # chain + geometry → bool
)
```

**Cross-domain coupling highlight:** `rigidbody` + `robotics` + `controls` = full robot arm simulation: physics for contact/collision, kinematics for arm configuration, LQR for joint control.

**Effort:** Medium-Large. IK is numerically finicky. URDF parsing is an optional heavy dependency. Core analytic IK for common arm types is tractable. ~130 operators, 180 tests.

---

### `economics` — Financial & Economic Modeling

**What it is:** Stochastic processes (GBM, mean-reversion, jump diffusion), option pricing (Black-Scholes, Monte Carlo, binomial trees), portfolio optimization, agent-based market models, time series decomposition.

**Key operators:**
```python
from morphogen.stdlib.economics import (
    gbm,                 # mu, sigma, S0, dt → price path
    mean_reversion,      # theta, mu, sigma → OU process
    black_scholes,       # S, K, T, r, sigma → option price
    monte_carlo_option,  # payoff_fn + paths → price with CI
    binomial_tree,       # S, K, T, r, sigma, N → American/European price
    efficient_frontier,  # returns + cov_matrix → Pareto frontier
    sharpe_ratio,        # returns + rf → ratio
    var,                 # returns + confidence → Value at Risk
    garch,               # returns → volatility model
    simulate_market,     # agents + rules + T → price series
)
```

**Cross-domain coupling highlight:** `agents` + `economics` = agent-based market: heterogeneous traders (trend-follower, fundamentalist, noise) each running their own logic, producing emergent price dynamics.

**Effort:** Medium. Finance math is well-defined. The GBM / Monte Carlo / Black-Scholes core is ~60 operators. Agent-based market is a stretch goal. ~100 operators, 130 tests.

---

## Tier 3 — Niche or Long-Horizon

These domains are scientifically interesting but serve narrower audiences or require significant infrastructure investment.

| Domain | Description | Main Blocker |
|--------|-------------|--------------|
| `atmosphere` | Standard atmosphere, boundary layer, turbulence, weather | Complex physics; CFD-adjacent |
| `seismic` | Elastic wave propagation, seismograms, earthquake source | Niche audience; field-adjacent |
| `plasma` | MHD, Vlasov equation, plasma instabilities | Very specialized; sparse Python tools |
| `biophysics` | Hodgkin-Huxley, calcium dynamics, membrane potentials | Overlap with neural; niche |
| `materials` | Crystal structure, phonons, band structure | Requires `ase`/VASP-adjacent tooling |
| `nlp` | Text ops, embeddings, similarity | Crowded space; not Morphogen's core identity |
| `cryptography` | Hash/entropy primitives, information theory | Narrow cross-domain coupling |

None of these are wrong to add; they're just lower leverage per unit of effort relative to Tiers 1 and 2.

---

## Implementation Order (Recommended)

If following this plan sequentially, suggested order based on impact × effort:

1. **`controls`** — unlocks the most cross-domain compositions; Medium effort; completes the physics → actuation → feedback loop
2. **`epidemiology`** — Small-Medium effort; immediate audience; `agents` + `graph` compositions are ready to ship
3. **`orbit`** — Medium effort; well-defined math; unique positioning in the Python ecosystem
4. **`fem`** — Large effort; high demand; may warrant an external contributor or focused sprint
5. **`photonics`** — Medium-Large; differentiating compositions with `combustion_light` and `signal`
6. **`quantum`** — Medium; growing audience; `optimization` + `quantum` is a compelling research story
7. **`economics`** — Medium; broad audience; `agents` + `economics` market simulation is ready
8. **`robotics`** — Medium-Large; high demand; best tackled after `controls` is stable

---

## Per-Domain Effort Summary

| Domain | Operators (est.) | Tests (est.) | Effort | New Cross-Domain Stories |
|--------|-----------------|--------------|--------|--------------------------|
| `controls` | ~150 | ~200 | Medium | 5+ |
| `epidemiology` | ~80 | ~120 | Small-Med | 5+ |
| `orbit` | ~100 | ~150 | Medium | 5+ |
| `fem` | ~200 | ~250 | Large | 5+ |
| `photonics` | ~120 | ~160 | Med-Large | 3+ |
| `quantum` | ~100 | ~150 | Medium | 2+ |
| `economics` | ~100 | ~130 | Medium | 3+ |
| `robotics` | ~130 | ~180 | Med-Large | 3+ |

---

## The Expansion Story

The 39 current domains give Morphogen credibility across simulation, audio, chemistry, and geometry. The Tier 1 expansions — `controls`, `epidemiology`, `orbit`, `fem` — extend that credibility into four new territory claims:

- **Controls:** "Morphogen is the only platform where you can design a controller and the physical plant in the same pipeline."
- **Epidemiology:** "Model an epidemic and sonify its dynamics in three imports."
- **Orbital mechanics:** "Design a spacecraft trajectory, simulate the attitude control, and hear it as data sonification."
- **FEM:** "From geometry to structural analysis to modal synthesis — no handoffs, no format conversions."

Each of these is a demonstration that the architecture generalizes — that the pattern established with the first 39 domains keeps working as the domain count grows.

---

## Related Documents

- **[STRATEGY.md](STRATEGY.md)** — v1.0 path and post-v1.0 vision
- **[ROADMAP.md](ROADMAP.md)** — v1.0 implementation tracking
- **[STATUS.md](../STATUS.md)** — current 39-domain status
- **[docs/usage/cross_domain_coupling.md](usage/cross_domain_coupling.md)** — how DomainInterface works
- **[docs/usage/chemistry_pipeline.md](usage/chemistry_pipeline.md)** — example of multi-domain pipeline
