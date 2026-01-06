# SPEC: Emergence Domain for Complex Systems and Artificial Life

> 🔬 **Implementation Status: PARTIAL**
> Core emergence operators exist. Full cross-domain integration is proposed for future versions.

**Version:** 1.0
**Status:** Partial Implementation
**Last Updated:** 2025-11-15

---

## Overview

This document specifies the **EmergenceDomain** — a unified layer for simulating emergent systems including Cellular Automata, Agent-Based Models, Swarms, Reaction-Diffusion, L-systems, and complex adaptive systems.

### Why EmergenceDomain?

Emergence is a natural fit for Morphogen because:

1. **Emergent systems are graph-friendly** — Everything is local, composable, and parallelizable
2. **Cross-domain potential** — EmergenceDomain integrates with Geometry, Physics, Acoustics, Optimization, ML
3. **Creative + Scientific applications** — Designers, artists, mathematicians, physicists all benefit
4. **GPU-accelerable** — CA + ABM are massively parallel
5. **Novel synthesis** — No existing tool unifies emergence + physics + audio + geometry

### Applications

**Scientific:**
- Biological morphogenesis modeling
- Predator-prey ecology simulation
- Reaction-diffusion pattern formation
- Active matter physics (Vicsek model)
- Distributed optimization (ACO, PSO)
- Complex adaptive systems research

**Creative:**
- Procedural texture generation
- Biological-looking geometry (corals, shells, trees)
- Generative art (fractal growth, cellular patterns)
- Animation (flocking, swarming, schooling)
- Interactive installations (responsive emergence)

**Engineering:**
- Network generation (roads, vascular systems, wiring)
- Structural optimization via growth algorithms
- Material microstructure simulation
- Swarm robotics algorithms

---

## Domain Classification

The EmergenceDomain encompasses 6 sub-domains:

1. **Cellular Automata (CA)** — Discrete grid-based evolution
2. **Agent-Based Models (ABM)** — Particle systems with behavior rules
3. **Reaction-Diffusion (RD)** — Continuous pattern-forming PDEs
4. **L-Systems** — Recursive growth and morphogenesis
5. **Swarm Intelligence** — Collective optimization (ACO, PSO, slime mold)
6. **Hybrid Systems** — Combinations of the above

Each sub-domain provides specific operators while sharing common infrastructure (grids, agents, fields).

---

## 1. Cellular Automata (CA)

**Purpose:** Discrete grid evolution with local rules

**Why:** Essential for:
- Conway's Game of Life and variants
- Texture generation (surfaces, materials)
- Physics approximations (lattice-gas)
- Electronic circuits (Wireworld)
- Procedural modeling

---

### 1.1 Core Types

```morphogen
// CA grid state
type CAGrid2D<T> {
    width: i32,
    height: i32,
    cells: Field2D<T>,      // Current state
    neighborhood: NeighborhoodType,
    boundary: BoundaryType
}

type CAGrid3D<T> {
    width: i32,
    height: i32,
    depth: i32,
    cells: Field3D<T>,
    neighborhood: NeighborhoodType,
    boundary: BoundaryType
}

enum NeighborhoodType {
    Moore,        // 8-neighbors (2D), 26-neighbors (3D)
    VonNeumann,   // 4-neighbors (2D), 6-neighbors (3D)
    Custom(kernel: Kernel)
}

enum BoundaryType {
    Periodic,     // Wrap-around (torus)
    Fixed(value: T),
    Mirror,       // Reflect at boundaries
    NoSlip        // Zero gradient
}

// Rule specification
type CARule {
    rule_number: Option<i32>,      // Wolfram rule number (1D)
    birth: Set<i32>,               // Birth conditions (Life-like)
    survival: Set<i32>,            // Survival conditions
    states: i32,                   // Number of states
    transition: Fn(neighborhood) -> State
}
```

---

### 1.2 Operators

#### ca.create

**Create CA grid**

```json
{
  "name": "ca.create",
  "domain": "emergence",
  "category": "cellular_automata",
  "layer": 4,
  "description": "Create cellular automaton grid",

  "params": [
    {"name": "width", "type": "i32", "description": "Grid width"},
    {"name": "height", "type": "i32", "description": "Grid height"},
    {"name": "states", "type": "i32", "default": 2, "description": "Number of cell states"},
    {"name": "neighborhood", "type": "NeighborhoodType", "default": "Moore"},
    {"name": "boundary", "type": "BoundaryType", "default": "Periodic"},
    {"name": "init", "type": "Enum", "default": "random", "enum": ["random", "empty", "seed"]}
  ],

  "outputs": [
    {"name": "grid", "type": "CAGrid2D<i32>"}
  ],

  "determinism": "strict",
  "rate": "control"
}
```

**Example:**
```morphogen
let grid = ca.create(
    width = 512,
    height = 512,
    neighborhood = "Moore",
    boundary = "Periodic",
    init = "random"
)
```

---

#### ca.step

**Evolve CA one timestep**

```json
{
  "name": "ca.step",
  "domain": "emergence",
  "category": "cellular_automata",
  "description": "Evolve cellular automaton one generation",

  "inputs": [
    {"name": "grid", "type": "CAGrid2D<T>"}
  ],

  "params": [
    {"name": "rule", "type": "CARule", "description": "Transition rule"}
  ],

  "outputs": [
    {"name": "grid", "type": "CAGrid2D<T>"}
  ],

  "determinism": "strict",
  "lowering_hints": {
    "parallelize": true,
    "tile_sizes": [16, 16],
    "memory_pattern": "stencil"
  }
}
```

**Example:**
```morphogen
// Conway's Game of Life rule
let life_rule = CARule {
    birth: {3},
    survival: {2, 3},
    states: 2
}

let evolved = ca.step(grid, rule=life_rule)
```

---

#### ca.rule_preset

**Create common CA rules**

```morphogen
ca.rule_preset(name: String) -> CARule
```

**Presets:**
- `"life"` — Conway's Game of Life (B3/S23)
- `"highlife"` — HighLife (B36/S23)
- `"day_and_night"` — Day & Night (B3678/S34678)
- `"brian's_brain"` — Brian's Brain (3-state)
- `"wireworld"` — Wireworld (electronic circuits, 4-state)
- `"rule30"` — Wolfram Rule 30 (1D)
- `"rule110"` — Wolfram Rule 110 (1D, Turing-complete)
- `"lenia"` — Lenia (continuous CA)

**Example:**
```morphogen
let rule = ca.rule_preset("life")
let grid = ca.step(grid, rule=rule)
```

---

#### ca.lenia

**Continuous cellular automaton (Lenia)**

Lenia is a continuous generalization of Conway's Life with smooth kernels and continuous states.

```json
{
  "name": "ca.lenia",
  "domain": "emergence",
  "category": "cellular_automata",
  "description": "Lenia continuous cellular automaton",

  "inputs": [
    {"name": "state", "type": "Field2D<f32>"}
  ],

  "params": [
    {"name": "kernel", "type": "String", "default": "gaussian"},
    {"name": "growth_func", "type": "String", "default": "gaussian"},
    {"name": "mu", "type": "f32", "default": 0.15},
    {"name": "sigma", "type": "f32", "default": 0.015},
    {"name": "dt", "type": "f32", "default": 0.1}
  ],

  "outputs": [
    {"name": "state", "type": "Field2D<f32>"}
  ],

  "determinism": "repro"
}
```

**Example:**
```morphogen
let state = field.create(256, 256, 0.0)
state = ca.lenia(state, mu=0.15, sigma=0.015)
```

---

#### ca.to_field

**Convert CA grid to field**

```morphogen
ca.to_field<T>(grid: CAGrid2D<T>) -> Field2D<f32>
```

Converts discrete CA states to continuous field (for visualization, geometry generation, etc.).

---

#### ca.diffuse

**Add diffusion to CA (continuous approximation)**

```morphogen
ca.diffuse(grid: CAGrid2D<f32>, coeff: f32, dt: Time) -> CAGrid2D<f32>
```

Applies diffusion kernel to continuous CA states.

---

### 1.3 Use Cases

**Texture Generation:**
```morphogen
// Generate biological texture
let ca = ca.create(512, 512, init="random")
let evolved = ca.step_n(ca, rule=ca.rule_preset("highlife"), steps=200)
let texture = ca.to_field(evolved)
let geometry = geom.extrude(texture, height=texture)
```

**Physics Approximation:**
```morphogen
// Lattice-gas hydrodynamics
let fluid_ca = ca.create(256, 256, states=16)  // 16 velocity directions
let evolved = ca.step(fluid_ca, rule=lattice_gas_rule)
```

**Electronic Circuits (Wireworld):**
```morphogen
let circuit = ca.create(128, 128, states=4, init="seed")
let evolved = ca.step(circuit, rule=ca.rule_preset("wireworld"))
```

---

## 2. Agent-Based Models (ABM)

**Purpose:** Particle systems with behavioral rules

**Why:** Essential for:
- Flocking / swarming / schooling
- Crowd simulation
- Social dynamics (Schelling segregation)
- Ecological modeling (predator-prey)
- Active matter physics

---

### 2.1 Core Types

```morphogen
// Agent attributes
type Agent<A> {
    id: u64,
    position: Vec2<m> | Vec3<m>,
    velocity: Vec2<m/s> | Vec3<m/s>,
    attributes: A  // Custom agent data
}

// Agent collection
type Agents<A> {
    agents: Array<Agent<A>>,
    count: i32,
    spatial_index: Option<SpatialIndex>
}

// Behavior rules
type BehaviorRule<A> {
    update_fn: Fn(agent: Agent<A>, neighbors: Array<Agent<A>>) -> Agent<A>
}
```

---

### 2.2 Operators

#### agent.create

**Create agent population**

```json
{
  "name": "agent.create",
  "domain": "emergence",
  "category": "agent_based_model",
  "description": "Create agent population",

  "params": [
    {"name": "count", "type": "i32"},
    {"name": "bounds", "type": "BoundingBox"},
    {"name": "distribution", "type": "String", "default": "random", "enum": ["random", "grid", "circle"]},
    {"name": "seed", "type": "u64", "required": true}
  ],

  "outputs": [
    {"name": "agents", "type": "Agents<A>"}
  ],

  "determinism": "strict"
}
```

---

#### agent.boids

**Flocking behavior (Reynolds boids)**

```json
{
  "name": "agent.boids",
  "domain": "emergence",
  "category": "agent_based_model",
  "description": "Apply boids flocking rules",

  "inputs": [
    {"name": "agents", "type": "Agents<BoidState>"}
  ],

  "params": [
    {"name": "separation", "type": "f32", "default": 1.0},
    {"name": "alignment", "type": "f32", "default": 1.0},
    {"name": "cohesion", "type": "f32", "default": 1.0},
    {"name": "perception_radius", "type": "f32<m>", "default": "10m"},
    {"name": "max_speed", "type": "f32<m/s>", "default": "5m/s"},
    {"name": "dt", "type": "Time", "default": "0.016s"}
  ],

  "outputs": [
    {"name": "agents", "type": "Agents<BoidState>"}
  ],

  "determinism": "strict"
}
```

**Example:**
```morphogen
let boids = agent.create(count=500, bounds=box(100m, 100m), seed=42)
boids = agent.boids(
    boids,
    separation = 1.5,
    alignment = 1.0,
    cohesion = 0.8,
    perception_radius = 15m
)
```

---

#### agent.vicsek

**Vicsek model (active matter physics)**

```json
{
  "name": "agent.vicsek",
  "domain": "emergence",
  "category": "agent_based_model",
  "description": "Vicsek model for collective motion",

  "inputs": [
    {"name": "agents", "type": "Agents<VicsekState>"}
  ],

  "params": [
    {"name": "speed", "type": "f32<m/s>"},
    {"name": "noise", "type": "f32", "description": "Angular noise amplitude"},
    {"name": "radius", "type": "f32<m>"},
    {"name": "dt", "type": "Time"}
  ],

  "outputs": [
    {"name": "agents", "type": "Agents<VicsekState>"}
  ],

  "determinism": "repro"
}
```

---

#### agent.schelling

**Schelling segregation model**

```json
{
  "name": "agent.schelling",
  "domain": "emergence",
  "category": "agent_based_model",
  "description": "Schelling segregation model for social dynamics",

  "inputs": [
    {"name": "agents", "type": "Agents<SchellingState>"}
  ],

  "params": [
    {"name": "threshold", "type": "f32", "default": 0.3, "description": "Similarity threshold"},
    {"name": "types", "type": "i32", "default": 2}
  ],

  "outputs": [
    {"name": "agents", "type": "Agents<SchellingState>"}
  ],

  "determinism": "repro"
}
```

---

#### agent.predator_prey

**Predator-prey ecology (Lotka-Volterra agent-based)**

```json
{
  "name": "agent.predator_prey",
  "domain": "emergence",
  "category": "agent_based_model",
  "description": "Agent-based predator-prey model",

  "inputs": [
    {"name": "prey", "type": "Agents<PreyState>"},
    {"name": "predators", "type": "Agents<PredatorState>"}
  ],

  "params": [
    {"name": "prey_birth_rate", "type": "f32", "default": 0.1},
    {"name": "predation_rate", "type": "f32", "default": 0.01},
    {"name": "predator_death_rate", "type": "f32", "default": 0.05},
    {"name": "dt", "type": "Time"}
  ],

  "outputs": [
    {"name": "prey", "type": "Agents<PreyState>"},
    {"name": "predators", "type": "Agents<PredatorState>"}
  ],

  "determinism": "repro"
}
```

---

#### agent.to_field

**Rasterize agents to field (particle-in-cell)**

```morphogen
agent.to_field<A>(
    agents: Agents<A>,
    property: String,
    resolution: (i32, i32)
) -> Field2D<f32>
```

Projects agent properties (density, velocity, etc.) to grid.

---

#### agent.from_field

**Sample field at agent positions**

```morphogen
agent.from_field<A>(
    agents: Agents<A>,
    field: Field2D<f32>,
    target_attr: String
) -> Agents<A>
```

Samples field values and updates agent attributes.

---

### 2.3 Use Cases

**Flocking Animation:**
```morphogen
let boids = agent.create(count=1000, bounds=box(200m, 200m), seed=42)

scene Flocking {
    step(dt: Time) {
        boids = agent.boids(boids, separation=1.5, cohesion=0.8, dt=dt)
        let density = agent.to_field(boids, property="density", resolution=(512, 512))
        out visual = visual.render_field(density, palette="viridis")
    }
}
```

**Crowd Simulation:**
```morphogen
let crowd = agent.create(count=5000, bounds=stadium_geometry, seed=42)
crowd = agent.crowd_dynamics(
    crowd,
    goals = exits,
    avoid_walls = true,
    personal_space = 0.5m
)
```

---

## 3. Reaction-Diffusion Systems

**Purpose:** Continuous pattern-forming PDEs

**Why:** Essential for:
- Biological patterns (zebra stripes, leopard spots)
- Texture generation
- Chemical simulation (Belousov-Zhabotinsky)
- Procedural design

---

### 3.1 Core Types

```morphogen
type RDSystem {
    u: Field2D<f32>,  // Activator
    v: Field2D<f32>,  // Inhibitor
    Du: f32,          // Diffusion rate for u
    Dv: f32,          // Diffusion rate for v
    f: f32,           // Feed rate
    k: f32            // Kill rate
}
```

---

### 3.2 Operators

#### rd.gray_scott

**Gray-Scott reaction-diffusion**

```json
{
  "name": "rd.gray_scott",
  "domain": "emergence",
  "category": "reaction_diffusion",
  "description": "Gray-Scott reaction-diffusion system",

  "inputs": [
    {"name": "u", "type": "Field2D<f32>"},
    {"name": "v", "type": "Field2D<f32>"}
  ],

  "params": [
    {"name": "Du", "type": "f32", "default": 0.16},
    {"name": "Dv", "type": "f32", "default": 0.08},
    {"name": "f", "type": "f32", "description": "Feed rate"},
    {"name": "k", "type": "f32", "description": "Kill rate"},
    {"name": "dt", "type": "f32", "default": 1.0}
  ],

  "outputs": [
    {"name": "u", "type": "Field2D<f32>"},
    {"name": "v", "type": "Field2D<f32>"}
  ],

  "determinism": "repro"
}
```

**Example:**
```morphogen
let u = field.create(256, 256, 1.0)
let v = field.create(256, 256, 0.0)

// Add perturbation
v = field.set_region(v, center=(128, 128), radius=10, value=1.0)

// Evolve
for i in 0..1000 {
    (u, v) = rd.gray_scott(u, v, f=0.055, k=0.062, dt=1.0)
}

out visual = visual.render_field(v, palette="magma")
```

---

#### rd.turing

**Turing pattern generator**

```json
{
  "name": "rd.turing",
  "domain": "emergence",
  "category": "reaction_diffusion",
  "description": "General Turing pattern formation system",

  "inputs": [
    {"name": "field", "type": "Field2D<f32>"}
  ],

  "params": [
    {"name": "a", "type": "f32", "description": "Activator strength"},
    {"name": "b", "type": "f32", "description": "Inhibitor strength"},
    {"name": "D1", "type": "f32", "description": "Activator diffusion"},
    {"name": "D2", "type": "f32", "description": "Inhibitor diffusion"},
    {"name": "dt", "type": "f32"}
  ],

  "outputs": [
    {"name": "field", "type": "Field2D<f32>"}
  ],

  "determinism": "repro"
}
```

---

#### rd.to_geometry

**Convert RD pattern to geometry**

```morphogen
rd.to_geometry(field: Field2D<f32>, threshold: f32) -> Mesh
```

Extracts isosurface from RD pattern (via marching squares/cubes).

---

### 3.3 Use Cases

**Texture Generation:**
```morphogen
(u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)
let texture = visual.colorize(v, palette="inferno")
let material = material.from_texture(texture)
```

**Geometry Generation:**
```morphogen
let pattern = rd.turing(field, a=1.0, b=2.0, D1=0.5, D2=1.0)
let mesh = rd.to_geometry(pattern, threshold=0.5)
let surface = geom.from_mesh(mesh)
```

---

## 4. L-Systems (Lindenmayer Systems)

**Purpose:** Recursive growth and morphogenesis

**Why:** Essential for:
- Tree/plant generation
- Fractal structures
- Coral-like forms
- Vascular systems
- Procedural architecture

---

### 4.1 Core Types

```morphogen
type LSystem {
    axiom: String,
    rules: Map<Char, String>,
    angle: f32<deg>,
    step_size: f32<m>
}

type LSystemState {
    string: String,
    iteration: i32
}
```

---

### 4.2 Operators

#### lsys.create

**Create L-system**

```morphogen
lsys.create(axiom: String, rules: Map<Char, String>) -> LSystem
```

---

#### lsys.evolve

**Evolve L-system n generations**

```json
{
  "name": "lsys.evolve",
  "domain": "emergence",
  "category": "lsystem",
  "description": "Evolve L-system string n generations",

  "inputs": [
    {"name": "lsys", "type": "LSystem"}
  ],

  "params": [
    {"name": "iterations", "type": "i32"}
  ],

  "outputs": [
    {"name": "state", "type": "LSystemState"}
  ],

  "determinism": "strict"
}
```

---

#### lsys.to_geometry

**Interpret L-system as turtle graphics → geometry**

```json
{
  "name": "lsys.to_geometry",
  "domain": "emergence",
  "category": "lsystem",
  "description": "Convert L-system to 3D geometry via turtle interpretation",

  "inputs": [
    {"name": "state", "type": "LSystemState"}
  ],

  "params": [
    {"name": "angle", "type": "f32<deg>", "default": "25deg"},
    {"name": "step_size", "type": "f32<m>", "default": "1m"},
    {"name": "diameter_scale", "type": "f32", "default": 0.9}
  ],

  "outputs": [
    {"name": "geometry", "type": "Mesh"}
  ],

  "determinism": "strict"
}
```

**Example:**
```morphogen
// Koch curve
let koch = lsys.create(
    axiom = "F",
    rules = {
        'F': "F+F-F-F+F"
    }
)

let evolved = lsys.evolve(koch, iterations=4)
let curve = lsys.to_geometry(evolved, angle=90deg, step_size=1m)
```

**Tree generation:**
```morphogen
let tree = lsys.create(
    axiom = "F",
    rules = {
        'F': "FF+[+F-F-F]-[-F+F+F]"
    }
)

let evolved = lsys.evolve(tree, iterations=5)
let geometry = lsys.to_geometry(
    evolved,
    angle = 25deg,
    step_size = 0.5m,
    diameter_scale = 0.7
)
```

---

### 4.3 Use Cases

**Procedural Trees:**
```morphogen
let tree = lsys.fractal_tree(iterations=6)
let mesh = lsys.to_geometry(tree, angle=22.5deg)
let solid = geom.from_mesh(mesh)
```

**Coral Structures:**
```morphogen
let coral = lsys.create(axiom="F", rules={'F': "F[+FF][-FF]F[-F][+F]F"})
let evolved = lsys.evolve(coral, iterations=4)
let structure = lsys.to_geometry(evolved)
```

---

## 5. Swarm Intelligence (Stigmergy)

**Purpose:** Distributed optimization via stigmergic communication

**Why:** Essential for:
- Pathfinding (ACO)
- Network generation (slime mold)
- Optimization (PSO - also in OptimizationDomain)
- Distributed algorithms

---

### 5.1 Operators

#### swarm.ants

**Ant Colony Optimization**

```json
{
  "name": "swarm.ants",
  "domain": "emergence",
  "category": "swarm_intelligence",
  "description": "Ant colony optimization via pheromone trails",

  "params": [
    {"name": "num_ants", "type": "i32"},
    {"name": "graph", "type": "Graph"},
    {"name": "start", "type": "NodeRef"},
    {"name": "goal", "type": "NodeRef"},
    {"name": "evaporation", "type": "f32", "default": 0.1},
    {"name": "alpha", "type": "f32", "default": 1.0, "description": "Pheromone importance"},
    {"name": "beta", "type": "f32", "default": 2.0, "description": "Heuristic importance"}
  ],

  "outputs": [
    {"name": "best_path", "type": "Path"},
    {"name": "pheromones", "type": "Field<f32>"}
  ],

  "determinism": "repro"
}
```

---

#### swarm.slime_mold

**Physarum polycephalum network formation**

```json
{
  "name": "swarm.slime_mold",
  "domain": "emergence",
  "category": "swarm_intelligence",
  "description": "Slime mold network optimization",

  "inputs": [
    {"name": "field", "type": "Field2D<f32>"},
    {"name": "food_sources", "type": "Array<Vec2<m>>"}
  ],

  "params": [
    {"name": "dt", "type": "f32"},
    {"name": "sensitivity", "type": "f32", "default": 1.0}
  ],

  "outputs": [
    {"name": "network", "type": "Field2D<f32>"}
  ],

  "determinism": "repro"
}
```

**Use Case:**
```morphogen
// Generate road network connecting cities
let field = field.create(512, 512, 0.0)
let cities = [
    vec2(100, 100),
    vec2(400, 100),
    vec2(250, 400)
]

let network = swarm.slime_mold(field, food_sources=cities, dt=0.1)
let roads = network.threshold(0.5).to_graph()
```

---

#### swarm.firefly

**Firefly algorithm**

```json
{
  "name": "swarm.firefly",
  "domain": "emergence",
  "category": "swarm_intelligence",
  "description": "Firefly optimization algorithm",

  "params": [
    {"name": "num_fireflies", "type": "i32"},
    {"name": "objective", "type": "Fn(Vec<f32>) -> f32"},
    {"name": "bounds", "type": "Array<(f32, f32)>"},
    {"name": "alpha", "type": "f32", "default": 0.5},
    {"name": "beta", "type": "f32", "default": 1.0},
    {"name": "gamma", "type": "f32", "default": 1.0},
    {"name": "iterations", "type": "i32"}
  ],

  "outputs": [
    {"name": "best_solution", "type": "Vec<f32>"},
    {"name": "best_fitness", "type": "f32"}
  ],

  "determinism": "strict"
}
```

---

## 6. Cross-Domain Integration

### EmergenceDomain → GeometryDomain

**Pattern → Surface:**
```morphogen
// Reaction-diffusion → displacement map
let (u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)
let heightmap = v
let mesh = geom.plane(100m, 100m, resolution=(256, 256))
let displaced = geom.displace(mesh, heightmap, scale=5m)
```

**L-system → Tree:**
```morphogen
let tree_string = lsys.evolve(tree_lsys, iterations=6)
let tree_geometry = lsys.to_geometry(tree_string, angle=25deg)
let solid_tree = geom.from_mesh(tree_geometry)
```

**CA → Extrusion:**
```morphogen
let ca = ca.step_n(grid, rule=ca.rule_preset("highlife"), steps=100)
let height_field = ca.to_field(ca)
let surface = geom.extrude(height_field, height=height_field * 10m)
```

---

### EmergenceDomain → PhysicsDomain

**Agents → Rigid Bodies:**
```morphogen
let boids = agent.boids(boids, ...)
let rigid_bodies = physics.from_agents(
    boids,
    mass = 1.0kg,
    shape = geom.sphere(0.5m)
)
```

**Slime Mold → Structural Optimization:**
```morphogen
let network = swarm.slime_mold(field, food_sources=anchor_points)
let structure = geom.from_network(network, diameter=0.1m)
let stress = physics.stress_analysis(structure, loads=loads)
```

---

### EmergenceDomain → AcousticsDomain

**Boids → Sound Scattering:**
```morphogen
let boids = agent.boids(boids, count=1000)
let positions = agent.positions(boids)
let wave = acoustic.propagate_with_scatterers(
    source = point_source,
    scatterers = positions,
    radius = 0.1m
)
```

---

### EmergenceDomain → OptimizationDomain

**PSO Integration:**
```morphogen
// PSO for chamber geometry
let result = opt.particle_swarm(
    objective = |params| simulate_chamber_efficiency(params),
    bounds = [(50mm, 200mm); 8],
    particles = 30,
    iterations = 100
)

// Visualize swarm evolution
let swarm_history = result.history
```

**ACO for Network Routing:**
```morphogen
let path = swarm.ants(
    graph = pcb_layout,
    start = pin_a,
    goal = pin_b,
    num_ants = 50
)
```

---

### EmergenceDomain → VisualizationDomain

**CA → Animation:**
```morphogen
scene CAAnimation {
    let ca = ca.create(512, 512, init="random")
    let rule = ca.rule_preset("life")

    step(dt: Time) {
        ca = ca.step(ca, rule=rule)
        let field = ca.to_field(ca)
        out visual = visual.render_field(field, palette="viridis")
    }
}
```

**RD → Real-time Texture:**
```morphogen
scene RDTexture {
    let (u, v) = rd.init(256, 256)

    step(dt: Time) {
        (u, v) = rd.gray_scott(u, v, f=0.055, k=0.062)
        out texture = visual.colorize(v, palette="plasma")
    }
}
```

---

## 7. Complete Pipeline Examples

### Example 1: Biological Morphogenesis

```morphogen
scene Morphogenesis {
    // Reaction-diffusion pattern
    let (u, v) = rd.init(512, 512)

    // Evolve pattern
    for i in 0..500 {
        (u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)
    }

    // Convert to geometry
    let heightmap = v
    let base = geom.plane(200mm, 200mm, resolution=(512, 512))
    let surface = geom.displace(base, heightmap, scale=20mm)

    // Export for 3D printing
    out mesh = surface
}
```

---

### Example 2: Swarm → Audio

```morphogen
scene SwarmSonification {
    let boids = agent.create(count=100, bounds=box(100m, 100m), seed=42)

    step(dt: Time) {
        // Update boids
        boids = agent.boids(boids, separation=1.5, cohesion=0.8, dt=dt)

        // Sonify density
        let density_field = agent.to_field(boids, property="density", resolution=(64, 64))
        let avg_density = field.reduce(density_field, "mean")

        // Audio output
        let freq = 200Hz + avg_density * 800Hz
        out audio = osc.sine(freq)

        // Visual output
        out visual = visual.render_agents(boids, color=density_field)
    }
}
```

---

### Example 3: CA → Geometry → Physics

```morphogen
scene CAPhysics {
    // Generate structure from CA
    let ca = ca.create(128, 128, init="random")
    let evolved = ca.step_n(ca, rule=ca.rule_preset("rule30"), steps=64)
    let pattern = ca.to_field(evolved)

    // Create geometry
    let structure = geom.extrude(pattern.threshold(0.5), height=10mm)

    // Physics simulation
    let body = physics.rigid_body(structure, mass=1.0kg)
    body = physics.apply_force(body, force=vec3(0, -9.8, 0) * 1.0kg)

    // Stress analysis
    let stress = physics.stress_test(structure, force=vec3(0, -100N, 0))
    out visual = visual.render_stress(stress)
}
```

---

### Example 4: Slime Mold Network → Optimization

```morphogen
scene SlimeMoldOptimization {
    // Cities to connect
    let cities = [
        vec2(50, 50),
        vec2(450, 50),
        vec2(250, 250),
        vec2(100, 450),
        vec2(400, 400)
    ]

    // Grow network
    let field = field.create(512, 512, 0.0)
    let network = swarm.slime_mold(field, food_sources=cities, dt=0.1)

    // Extract graph
    let road_graph = network.threshold(0.3).to_graph()

    // Optimize (minimize total length while maintaining connectivity)
    let optimized = opt.minimize_network_length(road_graph, constraints={
        "all_connected": true,
        "min_width": 2m
    })

    out geometry = geom.from_graph(optimized, diameter=2m)
}
```

---

## 8. Implementation Strategy

### Phase 1: Core Infrastructure (v0.10)
- [ ] CAGrid2D/3D types
- [ ] Agents<A> container
- [ ] Field2D/3D integration (already exists)
- [ ] Spatial indexing (grid, k-d tree)

### Phase 2: CA Operators (v0.10)
- [ ] ca.create
- [ ] ca.step
- [ ] ca.rule_preset (Life, Brian's Brain, Wireworld, Rule 30/110)
- [ ] ca.lenia
- [ ] ca.to_field

### Phase 3: ABM Operators (v0.10)
- [ ] agent.create
- [ ] agent.boids
- [ ] agent.vicsek
- [ ] agent.schelling
- [ ] agent.to_field / agent.from_field

### Phase 4: RD Operators (v0.11)
- [ ] rd.gray_scott
- [ ] rd.turing
- [ ] rd.to_geometry

### Phase 5: L-Systems (v0.11)
- [ ] lsys.create
- [ ] lsys.evolve
- [ ] lsys.to_geometry (turtle graphics interpreter)

### Phase 6: Swarm Intelligence (v1.0)
- [ ] swarm.ants
- [ ] swarm.slime_mold
- [ ] swarm.firefly

### Phase 7: Cross-Domain Integration (v1.0)
- [ ] Emergence → Geometry
- [ ] Emergence → Physics
- [ ] Emergence → Acoustics
- [ ] Emergence → Optimization

---

## 9. Testing Strategy

### Determinism Tests
```morphogen
// CA must be bit-exact
let ca1 = ca.create(128, 128, seed=42)
let ca2 = ca.create(128, 128, seed=42)
assert_eq!(ca1, ca2)

let evolved1 = ca.step_n(ca1, rule=life_rule, steps=100)
let evolved2 = ca.step_n(ca2, rule=life_rule, steps=100)
assert_eq!(evolved1, evolved2)
```

---

### Conservation Tests
```morphogen
// Boids: total momentum conservation
let boids = agent.boids(boids, ...)
let momentum_before = agent.total_momentum(boids)
boids = agent.boids(boids, ...)
let momentum_after = agent.total_momentum(boids)
assert_approx_eq!(momentum_before, momentum_after)
```

---

### Pattern Recognition Tests
```morphogen
// Life: Glider should move diagonally
let glider = ca.create_pattern("glider")
let evolved = ca.step_n(glider, rule=life_rule, steps=4)
let position = ca.find_pattern(evolved, "glider")
assert_eq!(position, vec2(1, 1))  // Moved 1 cell diagonally
```

---

## 10. GPU Acceleration

All emergence operators are designed for GPU execution:

**CA:**
- Each cell update is independent → perfect parallelism
- Stencil access patterns → shared memory optimization
- Double-buffering → no synchronization needed

**ABM:**
- Spatial indexing → parallel neighbor search
- Force accumulation → atomic operations or reduction
- Position updates → fully parallel

**RD:**
- Laplacian stencil → tile + shared memory
- Integration → parallel per-cell
- Same as field operations

**L-Systems:**
- String evolution → parallel character replacement (GPU string ops)
- Turtle interpretation → sequential (CPU) or batched (GPU)

---

## 11. Use Cases Summary

| Domain | Applications |
|--------|--------------|
| **CA** | Texture generation, physics approximation, circuits, procedural modeling |
| **ABM** | Flocking, crowds, ecology, social dynamics, active matter |
| **RD** | Biological patterns, textures, chemical simulation, design |
| **L-Systems** | Trees, plants, corals, vascular systems, fractals |
| **Swarm** | Pathfinding (ACO), network generation (slime mold), optimization |

---

## 12. References

### Cellular Automata
- **Wolfram, S.** "A New Kind of Science" (2002)
- **Gardner, M.** "Mathematical Games: The fantastic combinations of John Conway's new solitaire game 'life'" (1970)
- **Chan, B.** "Lenia: Biology of Artificial Life" (2019)

### Agent-Based Models
- **Reynolds, C.** "Flocks, Herds, and Schools: A Distributed Behavioral Model" (1987)
- **Vicsek, T. et al.** "Novel Type of Phase Transition in a System of Self-Driven Particles" (1995)
- **Schelling, T.** "Dynamic Models of Segregation" (1971)

### Reaction-Diffusion
- **Turing, A.** "The Chemical Basis of Morphogenesis" (1952)
- **Gray, P. & Scott, S.** "Autocatalytic reactions in the isothermal, continuous stirred tank reactor" (1983)

### L-Systems
- **Lindenmayer, A.** "Mathematical models for cellular interactions in development" (1968)
- **Prusinkiewicz, P. & Lindenmayer, A.** "The Algorithmic Beauty of Plants" (1990)

### Swarm Intelligence
- **Dorigo, M.** "Ant Colony Optimization" (2004)
- **Tero, A. et al.** "Rules for Biologically Inspired Adaptive Network Design" (2010) — Slime mold

---

## Summary

The **EmergenceDomain** provides:

✅ **Unified emergence platform** — CA, ABM, RD, L-systems, swarms in one framework
✅ **Cross-domain integration** — Seamless composition with Geometry, Physics, Audio, Optimization
✅ **GPU-accelerated** — All operators designed for parallel execution
✅ **Deterministic** — Strict/repro guarantees for reproducible research
✅ **Creative + Scientific** — Applications from generative art to biological modeling

This makes Morphogen a **true synthetic universe builder** — no other tool unifies emergence, physics, audio, geometry, and optimization in one coherent system.

---

**End of Document**
