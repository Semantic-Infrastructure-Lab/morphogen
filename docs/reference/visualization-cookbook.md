# Visualization Cookbook

A comprehensive catalog of visualization techniques that leverage Morphogen's unique cross-domain capabilities. This document organizes visualization patterns by computational domain and highlights powerful cross-domain compositions.

**Related Documentation:**
- [Mathematical Transformation Metaphors](./math-transformation-metaphors.md) - Intuitive frameworks for understanding the transforms behind these visualizations
- [Advanced Visualizations](./advanced-visualizations.md) - Implemented visualization techniques
- [Transform Specification](../specifications/transform.md) - Technical details of domain transformations
- [Sonification Cookbook](./sonification-cookbook.md) - Sonification patterns (making computation audible)
- [Visual Scene Domain](./visual-scene-domain.md) - Architecture for 3D scene visualization
- [Visual Domain Quick Reference](./visual-domain-quickref.md) - Quick reference for visual operations

**Status Legend:**
- ✅ **Fully Implemented** - Ready to use now
- 🚧 **Partially Implemented** - Some components available
- 📋 **Planned** - Documented but not yet implemented
- 💡 **Concept** - New idea for future consideration

---

## Table of Contents

1. [Audio & Signal Processing](#audio--signal-processing)
2. [Physics & Dynamics](#physics--dynamics)
3. [Fields & PDEs](#fields--pdes)
4. [Agents & Particle Systems](#agents--particle-systems)
5. [Optimization & Search](#optimization--search)
6. [Geometry & Spatial](#geometry--spatial)
7. [Cellular Automata & Emergence](#cellular-automata--emergence)
8. [Graph & Network](#graph--network)
9. [Terrain & Procedural](#terrain--procedural)
10. [Cross-Domain Compositions](#cross-domain-compositions)
11. [Scientific & Educational](#scientific--educational)
12. [Creative & Generative Art](#creative--generative-art)

---

## Audio & Signal Processing

### Waveform Visualizations

**✅ Time-Domain Waveform Display**
```python
# Currently available with matplotlib/visual.output
audio_signal = audio.oscillator(freq=440.0, shape="sine")
waveform = audio.render(audio_signal, duration=1.0)
# Visualize as line plot
```

**Status:** ✅ Available via matplotlib integration
**Domains:** Audio, Visual
**Use Cases:** Debugging audio synthesis, verifying signal generation

---

**💡 Real-Time Oscilloscope**
```python
# Interactive waveform display with scrolling
def audio_oscilloscope():
    buffer = audio.circular_buffer(size=2048)
    while True:
        signal = audio.stream_input()
        buffer.push(signal)
        yield visual.line_plot(buffer.data, color="green", thickness=2)

visual.display(audio_oscilloscope, target_fps=60)
```

**Status:** 💡 Concept - needs real-time audio input integration
**Domains:** Audio, Visual, Temporal
**Use Cases:** Live audio monitoring, debugging real-time audio systems

---

### Spectral Visualizations

**🚧 Spectrogram (Time-Frequency)**
```python
# STFT-based spectrogram
audio_signal = audio.load("sound.wav")
spec = signal.stft(audio_signal, window_size=2048, hop=512)
magnitude = signal.magnitude(spec)

# Visualize as colorized field
vis = visual.colorize(magnitude, palette="inferno", vmin=-80, vmax=0)
visual.output(vis, "spectrogram.png")
```

**Status:** 🚧 STFT exists, needs time-axis visualization
**Domains:** Audio, Signal, Visual
**Use Cases:** Audio analysis, music production, speech processing

---

**💡 3D Spectral Waterfall**
```python
# 3D surface of frequency spectrum over time
scene = scene.create("spectral_waterfall")

for t, frame in enumerate(audio_frames):
    spectrum = signal.fft(frame)
    magnitude = signal.magnitude(spectrum)

    # Add as row in 3D surface
    surface.add_row(t, magnitude)

# Visualize as 3D surface with color mapping
surface.color_by_field(magnitude, palette="plasma")
camera.orbit(angle=2*pi, duration=10.0)
```

**Status:** 📋 Needs 3D surface rendering (Phase 6)
**Domains:** Audio, Signal, Visual
**Use Cases:** Music analysis, harmonic content visualization, signal evolution

---

**✅ Circular Spectrum Analyzer**
```python
# Frequency spectrum arranged in polar coordinates
def spectrum_analyzer():
    while True:
        audio_frame = audio.get_frame()
        spectrum = signal.fft(audio_frame)
        magnitude = signal.magnitude(spectrum)

        # Convert to polar visualization
        angles = np.linspace(0, 2*np.pi, len(magnitude))
        points = polar_to_cartesian(angles, magnitude)

        yield visual.line_plot(points, color_by_value=magnitude,
                              palette="fire", closed=True)

visual.display(spectrum_analyzer, target_fps=30)
```

**Status:** ✅ Can be implemented with current tools
**Domains:** Audio, Signal, Visual, Geometry
**Use Cases:** Music visualizers, DJ software, creative displays

---

### Phase & Harmonic Visualizations

**💡 Lissajous Curves from Audio**
```python
# Stereo signal as 2D phase plot
left, right = audio.load_stereo("music.wav")

# Plot left channel vs right channel
scene = scene.create("lissajous")
curve = geo.parametric_curve(t -> [left(t), right(t)], (0, duration))
curve.set_color_by_parameter(t, palette="viridis")
anim.draw_curve(curve, duration=duration)
```

**Status:** 📋 Needs parametric curve animation
**Domains:** Audio, Geometry, Visual
**Use Cases:** Stereo field analysis, phase correlation, audio engineering

---

**💡 Harmonic Series Visualization**
```python
# Show harmonic partials as stacked waveforms
fundamental = audio.oscillator(freq=440.0, shape="sine")
harmonics = [audio.oscillator(freq=440.0*i, shape="sine")
             for i in range(1, 9)]

scene = scene.create("harmonics")
for i, harmonic in enumerate(harmonics):
    waveform = audio.render(harmonic, duration=0.01)
    line = geo.curve(t -> [t, waveform(t) + i*0.5], (0, 0.01))
    line.set_color(palette.get("rainbow", i/8))
    scene.add(line)
```

**Status:** 📋 Needs multi-curve scene composition
**Domains:** Audio, Visual, Geometry
**Use Cases:** Music theory education, additive synthesis design, harmonic analysis

---

### Filter & Effect Visualizations

**💡 Filter Frequency Response**
```python
# Visualize filter magnitude and phase response
filter_coeffs = audio.filter_design(type="lowpass", cutoff=1000, order=4)

# Sweep frequencies and measure response
freqs = np.logspace(1, 4, 1000)  # 10 Hz to 10 kHz
response = [audio.filter_response(filter_coeffs, f) for f in freqs]
magnitude = [abs(r) for r in response]
phase = [np.angle(r) for r in response]

# Dual plot: magnitude and phase
scene = scene.create("filter_response")
mag_curve = geo.curve(lambda f: [np.log10(f), 20*np.log10(magnitude(f))],
                      (1, 4))
phase_curve = geo.curve(lambda f: [np.log10(f), phase(f)], (1, 4))

scene.add([axes, mag_curve, phase_curve])
```

**Status:** 💡 Concept - needs filter design and response analysis
**Domains:** Audio, Signal, Visual
**Use Cases:** Filter design, audio effect development, DSP education

---

## Physics & Dynamics

### Rigid Body Physics

**✅ 2D Rigid Body Trajectories**
```python
# Visualize bouncing balls with trails
def physics_viz():
    bodies = rigidbody.create_system([
        rigidbody.circle(pos=[0, 10], radius=1, mass=1),
        rigidbody.circle(pos=[5, 15], radius=0.5, mass=0.5),
    ])

    while True:
        bodies = rigidbody.step(bodies, dt=0.016, gravity=[0, -9.8])

        # Render with trails
        vis = visual.agents(
            bodies,
            size_property='radius',
            color_property='velocity',
            palette='viridis',
            trail=True,
            trail_length=20
        )
        yield vis

visual.display(physics_viz, target_fps=60)
```

**Status:** ✅ Fully available (v0.8.2 + v0.6.0)
**Domains:** RigidBody, Agents, Visual
**Use Cases:** Physics simulation, game development, education

---

**💡 Collision Impact Visualization**
```python
# Show impulse vectors at collision points
def collision_viz():
    bodies = rigidbody.create_system(many_bodies)

    while True:
        bodies, collisions = rigidbody.step_with_collisions(bodies, dt=0.016)

        # Base visualization
        vis = visual.agents(bodies)

        # Add impulse arrows at collision points
        for collision in collisions:
            arrow = geo.arrow(
                start=collision.point,
                end=collision.point + collision.impulse,
                color="red",
                width=0.1
            )
            vis = visual.composite(vis, arrow, mode="over")

        yield vis
```

**Status:** 💡 Concept - needs collision event extraction and arrow rendering
**Domains:** RigidBody, Geometry, Visual
**Use Cases:** Physics debugging, collision analysis, game development

---

**📋 Force Field Overlay**
```python
# Show forces acting on rigid bodies
scene = scene.create("forces")

# Add rigid bodies
bodies = rigidbody.create_system(...)

# Compute force field (gravity + drag)
force_field = field.vector(lambda pos: gravity + drag(pos))

# Visualize
vfield_vis = geo.vector_field(force_field, bounds)
body_vis = visual.agents(bodies)

scene.add([vfield_vis, body_vis])
```

**Status:** 📋 Needs vector field rendering (planned)
**Domains:** RigidBody, Field, Geometry, Visual
**Use Cases:** Force analysis, physics education, simulation debugging

---

### Spring & Constraint Systems

**💡 Spring Network Visualization**
```python
# Cloth or soft body with visible springs
def spring_network():
    # Create mass-spring grid
    masses = [(i, j) for i in range(10) for j in range(10)]
    springs = create_spring_connections(masses)

    while True:
        # Physics step
        forces = compute_spring_forces(masses, springs)
        masses = integrate(masses, forces, dt=0.01)

        # Visualize springs colored by tension
        for spring in springs:
            tension = spring.current_length / spring.rest_length
            line = geo.line(spring.mass1.pos, spring.mass2.pos)
            line.set_color(palette.get("coolwarm", tension))
            scene.add(line)

        # Visualize masses
        dots = [geo.dot(m.pos, radius=0.2) for m in masses]
        scene.add(dots)

        yield scene.render()
```

**Status:** 💡 Concept - needs spring physics and line rendering
**Domains:** Physics, Geometry, Visual
**Use Cases:** Soft body simulation, cloth simulation, constraint visualization

---

## Fields & PDEs

### Scalar Field Visualizations

**✅ Heat Diffusion with Colormap**
```python
# Classic heat equation visualization
def heat_diffusion():
    temp = field.random((128, 128), seed=42, low=0.0, high=1.0)

    while True:
        temp = field.diffuse(temp, rate=0.2, dt=0.1, iterations=20)
        temp = field.boundary(temp, spec="reflect")
        yield visual.colorize(temp, palette="fire", vmin=0, vmax=1)

visual.display(heat_diffusion, target_fps=30)
```

**Status:** ✅ Fully implemented
**Domains:** Field, Visual
**Use Cases:** Heat transfer, diffusion processes, PDE education

---

**💡 3D Isosurface Rendering**
```python
# Extract and render isosurfaces from 3D scalar field
field3d = field.random_3d((64, 64, 64), seed=42)

# Extract isosurface at value=0.5
vertices, faces = field.isosurface(field3d, isovalue=0.5, method="marching_cubes")
mesh = geo.mesh(vertices, faces)

# Color by gradient magnitude
gradient = field.gradient_3d(field3d)
grad_mag = field.magnitude(gradient)
mesh.color_by_field(grad_mag, palette="viridis")

scene.add(mesh)
camera.orbit(2*pi, duration=10)
```

**Status:** 💡 Concept - needs 3D field operations and mesh rendering
**Domains:** Field, Geometry, Visual
**Use Cases:** Medical imaging (MRI/CT), volumetric data, scientific visualization

---

**💡 Contour Plot with Labels**
```python
# 2D contour lines with elevation labels
scalar_field = field.gaussian_bump((256, 256), center=(128, 128), sigma=30)

# Extract contour levels
contours = field.contours(scalar_field, levels=10)

scene = scene.create("contours")
for i, contour in enumerate(contours):
    curve = geo.curve_from_points(contour.points)
    curve.set_color(palette.get("terrain", i/10))
    label = text.create(f"{contour.value:.2f}", contour.center)
    scene.add([curve, label])
```

**Status:** 📋 Needs contour extraction and curve rendering
**Domains:** Field, Geometry, Visual
**Use Cases:** Topography, meteorology, scientific data visualization

---

### Vector Field Visualizations

**💡 Streamline Visualization**
```python
# Integral curves following vector field
velocity_field = field.vector(lambda pos: rotation_field(pos))

# Seed points in grid
seeds = [(i, j) for i in range(0, 10, 2) for j in range(0, 10, 2)]

scene = scene.create("streamlines")
for seed in seeds:
    # Compute integral curve
    curve = field.integral_curve(velocity_field, start=seed, steps=100)

    # Color by velocity magnitude
    speeds = [np.linalg.norm(velocity_field(p)) for p in curve.points]
    curve.color_by_values(speeds, palette="plasma")

    anim.draw_curve(curve, duration=2.0)
```

**Status:** 📋 Planned (Phase 2)
**Domains:** Field, Geometry, Visual
**Use Cases:** Fluid flow, vector field analysis, dynamical systems

---

**✅ Line Integral Convolution (LIC)**
```python
# Texture-based vector field visualization
velocity_field = field.vector(lambda pos: [sin(pos[1]), cos(pos[0])])

# Generate LIC texture
lic_texture = field.line_integral_convolution(
    velocity_field,
    noise_texture=field.random((256, 256)),
    kernel_length=20
)

vis = visual.colorize(lic_texture, palette="grayscale")
visual.output(vis, "lic_visualization.png")
```

**Status:** 💡 Concept - needs LIC algorithm implementation
**Domains:** Field, Visual
**Use Cases:** CFD visualization, flow analysis, scientific papers

---

**💡 Arrow Glyph Field**
```python
# Vector field shown as arrow glyphs
velocity_field = field.vector(lambda pos: vortex(pos))

# Sample at grid points
sample_points = grid_sample((0, 10), (0, 10), spacing=0.5)

scene = scene.create("vector_glyphs")
for point in sample_points:
    vector = velocity_field(point)
    magnitude = np.linalg.norm(vector)

    # Create arrow glyph
    arrow = geo.arrow(
        start=point,
        end=point + vector * 0.3,  # Scale for visibility
        color=palette.get("viridis", magnitude / max_magnitude),
        width=0.05
    )
    scene.add(arrow)
```

**Status:** 📋 Needs arrow rendering
**Domains:** Field, Geometry, Visual
**Use Cases:** Wind maps, electromagnetic fields, force visualization

---

### Multi-Field Composites

**🚧 Divergence/Curl Overlay**
```python
# Show vector field with divergence and curl as overlays
velocity = field.vector(lambda pos: flow_field(pos))

# Compute derivatives
divergence = field.divergence(velocity)
curl = field.curl(velocity)  # 2D: scalar vorticity

# Composite visualization
base_vis = visual.colorize(divergence, palette="coolwarm", vmin=-1, vmax=1)
streamlines = visual.streamlines(velocity, density=2)
vis = visual.composite(base_vis, streamlines, mode="over", opacity=[1.0, 0.7])
```

**Status:** 🚧 Needs divergence/curl operators and streamline rendering
**Domains:** Field, Visual
**Use Cases:** Fluid dynamics, electromagnetic analysis, differential geometry

---

## Agents & Particle Systems

### Particle Visualizations

**✅ Particle System with Property Mapping**
```python
# Particles colored by velocity, sized by energy
def particle_viz():
    particles = agents.create([
        agents.particle(pos=[random(), random()],
                       vel=[random(), random()],
                       energy=random())
        for _ in range(1000)
    ])

    while True:
        particles = agents.step(particles, dt=0.016)

        vis = visual.agents(
            particles,
            width=512, height=512,
            color_property='vel',
            size_property='energy',
            palette='plasma',
            blend_mode='additive'
        )
        yield vis

visual.display(particle_viz, target_fps=60)
```

**Status:** ✅ Fully implemented (v0.6.0)
**Domains:** Agents, Visual
**Use Cases:** Particle effects, n-body simulation, swarm behaviors

---

**✅ Particle Trails with Fade**
```python
# Motion blur effect with particle trails
vis = visual.agents(
    particles,
    color_property='type',
    trail=True,
    trail_length=30,
    alpha_property='energy',  # Fade based on energy
    palette='fire'
)
```

**Status:** ✅ Fully implemented
**Domains:** Agents, Visual
**Use Cases:** Motion visualization, trajectory analysis, artistic effects

---

**💡 Particle Flow Ribbons**
```python
# Connect particles with ribbons showing flow
def particle_ribbons():
    particles = agents.create_flow(n=100)

    while True:
        particles = agents.step(particles, dt=0.016)

        # Find nearest neighbors
        for particle in particles:
            neighbors = agents.query_neighbors(particle, radius=2.0, max_count=3)

            # Draw ribbons with alpha based on distance
            for neighbor in neighbors:
                dist = distance(particle.pos, neighbor.pos)
                ribbon = geo.line(particle.pos, neighbor.pos)
                ribbon.set_opacity(1.0 - dist/2.0)
                ribbon.set_color(palette.get("plasma", particle.speed))
                scene.add(ribbon)

        yield scene.render()
```

**Status:** 💡 Concept - needs neighbor queries and line rendering
**Domains:** Agents, Geometry, Visual
**Use Cases:** Flow visualization, network effects, generative art

---

### Swarm & Flocking Visualizations

**💡 Boid Orientation Indicators**
```python
# Show velocity direction with rotation indicators
def boid_visualization():
    boids = agents.create_boids(n=200)

    while True:
        boids = agents.step_boids(boids, dt=0.016)

        vis = visual.agents(
            boids,
            color_property='vel',
            size_property='speed',
            rotation_from_velocity=True,  # Rotate glyphs based on velocity
            glyph='triangle',  # Directional glyph
            palette='viridis'
        )
        yield vis
```

**Status:** 💡 Concept - needs rotation visualization and custom glyphs
**Domains:** Agents, Visual
**Use Cases:** Flocking simulation, swarm robotics, collective behavior

---

**💡 Flocking Force Vectors**
```python
# Visualize separation, alignment, cohesion forces
def flocking_forces():
    boids = agents.create_boids(n=100)

    while True:
        boids, forces = agents.step_boids_with_forces(boids, dt=0.016)

        # Base visualization
        vis = visual.agents(boids)

        # Overlay force arrows
        for boid, force_breakdown in zip(boids, forces):
            # Show three force components in different colors
            arrows = [
                geo.arrow(boid.pos, boid.pos + force_breakdown.separation,
                         color="red"),
                geo.arrow(boid.pos, boid.pos + force_breakdown.alignment,
                         color="green"),
                geo.arrow(boid.pos, boid.pos + force_breakdown.cohesion,
                         color="blue"),
            ]
            vis = visual.composite(vis, arrows, mode="over")

        yield vis
```

**Status:** 💡 Concept - needs force extraction and arrow rendering
**Domains:** Agents, Geometry, Visual
**Use Cases:** Algorithm debugging, behavior analysis, education

---

### Agent-Field Coupling

**🚧 Agents Depositing Pheromones**
```python
# Ant colony with pheromone field visualization
def ant_colony():
    ants = agents.create(n=100, type="ant")
    pheromone = field.zeros((256, 256))

    while True:
        # Ants deposit pheromones
        pheromone = agents.deposit_to_field(ants, pheromone,
                                           property='pheromone_amount')

        # Pheromones diffuse and decay
        pheromone = field.diffuse(pheromone, rate=0.1, dt=0.1)
        pheromone = pheromone * 0.99  # Decay

        # Ants move based on pheromone gradient
        grad = field.gradient(pheromone)
        ants = agents.move_by_field(ants, grad, strength=0.5, dt=0.016)

        # Composite visualization
        field_vis = visual.colorize(pheromone, palette="green")
        agent_vis = visual.agents(ants, color="white")
        yield visual.composite(field_vis, agent_vis, mode="over")
```

**Status:** 🚧 Has field and agents, needs coupling operators
**Domains:** Agents, Field, Visual
**Use Cases:** Swarm intelligence, ant colony optimization, stigmergy

---

## Optimization & Search

### Optimization Landscape

**💡 2D Cost Function Surface**
```python
# Visualize optimization landscape with search trajectory
cost_function = lambda x, y: rastrigin([x, y])

# Create 2D surface
x_range, y_range = (-5, 5), (-5, 5)
surface = field.from_function(cost_function, shape=(256, 256),
                             bounds=[x_range, y_range])

# Run optimization and track path
optimizer = optimization.gradient_descent(start=[4, 4], learning_rate=0.01)
path = []
for step in range(100):
    pos = optimizer.step(cost_function)
    path.append(pos)

# Visualize
scene = scene.create("optimization")
surface_vis = visual.colorize(surface, palette="viridis")
path_curve = geo.curve_from_points(path)
path_curve.set_color("red")
path_curve.set_thickness(3)

scene.add([surface_vis, path_curve])
anim.draw_curve(path_curve, duration=5.0)
```

**Status:** 📋 Has optimization (v0.9.0), needs 3D surface rendering
**Domains:** Optimization, Field, Geometry, Visual
**Use Cases:** Algorithm visualization, debugging optimization, education

---

**💡 Particle Swarm Optimization Visualization**
```python
# Show PSO particles exploring cost surface
def pso_viz():
    particles = optimization.pso_init(n=30, bounds=[(-5, 5), (-5, 5)])
    cost_surface = field.from_function(sphere_function, shape=(128, 128))

    while True:
        particles = optimization.pso_step(particles, cost_function=sphere_function)

        # Composite: cost surface + particles
        surface_vis = visual.colorize(cost_surface, palette="terrain")
        particle_vis = visual.agents(
            particles,
            color_property='best_cost',
            size_property='velocity',
            palette='plasma',
            trail=True,
            trail_length=10
        )

        yield visual.composite(surface_vis, particle_vis, mode="over")
```

**Status:** 🚧 Has PSO (v0.9.0), needs agent-field composite rendering
**Domains:** Optimization, Agents, Field, Visual
**Use Cases:** Algorithm comparison, parameter tuning, research

---

### Genetic Algorithm Visualizations

**💡 Population Fitness Distribution**
```python
# Show fitness evolution as histogram animation
def ga_fitness_viz():
    population = optimization.ga_init(size=100)

    while True:
        population = optimization.ga_step(population, fitness_function)

        # Extract fitness values
        fitnesses = [individual.fitness for individual in population]

        # Create histogram
        hist = visual.histogram(fitnesses, bins=20, color="blue")

        # Add statistics overlay
        mean_line = geo.line([0, np.mean(fitnesses)],
                            [hist.height, np.mean(fitnesses)],
                            color="red", width=2)

        yield visual.composite(hist, mean_line, mode="over")
```

**Status:** 💡 Concept - needs histogram rendering and GA state extraction
**Domains:** Optimization, Visual
**Use Cases:** Algorithm analysis, convergence monitoring, research

---

**💡 Gene Space Projection (t-SNE)**
```python
# Project high-dimensional individuals to 2D for visualization
def ga_projection():
    population = optimization.ga_init(size=200, genome_length=50)

    while True:
        population = optimization.ga_step(population, fitness_function)

        # Extract genomes and project to 2D
        genomes = [ind.genome for ind in population]
        projected = tsne_projection(genomes, dims=2)

        # Create scatter plot colored by fitness
        fitnesses = [ind.fitness for ind in population]
        scatter = visual.scatter(
            projected,
            color_values=fitnesses,
            palette="viridis",
            size=5
        )

        yield scatter
```

**Status:** 💡 Concept - needs dimensionality reduction and scatter plots
**Domains:** Optimization, Visual, Graph (for embeddings)
**Use Cases:** Population diversity analysis, convergence detection

---

## Geometry & Spatial

### Geometric Primitives

**📋 Parametric Curve Gallery**
```python
# Classic mathematical curves
curves = {
    "spiral": lambda t: [t*cos(t), t*sin(t)],
    "lissajous": lambda t: [sin(3*t), sin(2*t)],
    "rose": lambda t: [cos(5*t)*cos(t), cos(5*t)*sin(t)],
    "epicycloid": lambda t: [(R+r)*cos(t) - r*cos((R+r)*t/r),
                             (R+r)*sin(t) - r*sin((R+r)*t/r)],
}

scene = scene.create("curves")
for i, (name, f) in enumerate(curves.items()):
    curve = geo.curve(f, t_range=(0, 2*pi))
    curve.set_color(palette.get("rainbow", i/len(curves)))
    curve.translate([i*5, 0])  # Space them out
    scene.add(curve)

    # Animate drawing
    anim.draw_curve(curve, duration=3.0, delay=i*0.5)
```

**Status:** 📋 Needs parametric curve rendering (planned)
**Domains:** Geometry, Visual
**Use Cases:** Mathematics education, generative art, design

---

**💡 3D Parametric Surfaces**
```python
# Mathematical surfaces (torus, Klein bottle, etc.)
surfaces = {
    "torus": lambda u, v: [(2 + cos(v))*cos(u),
                           (2 + cos(v))*sin(u),
                           sin(v)],
    "sphere": lambda u, v: [sin(u)*cos(v), sin(u)*sin(v), cos(u)],
    "mobius": lambda u, v: [(1 + v*cos(u/2))*cos(u),
                            (1 + v*cos(u/2))*sin(u),
                            v*sin(u/2)],
}

for name, f in surfaces.items():
    surface = geo.parametric_surface(f, u=(0, 2*pi), v=(0, 2*pi))
    surface.color_by_uv(palette="viridis")

    scene.add(surface)
    camera.orbit(2*pi, duration=10)
```

**Status:** 📋 Needs 3D surface rendering (Phase 6)
**Domains:** Geometry, Visual
**Use Cases:** Differential geometry, topology, mathematical art

---

### Voronoi & Delaunay

**💡 Voronoi Diagram Animation**
```python
# Animate Voronoi cell formation
def voronoi_animation():
    points = []

    for frame in range(100):
        # Add new points gradually
        if frame % 5 == 0:
            points.append([random()*10, random()*10])

        # Compute Voronoi diagram
        voronoi = geometry.voronoi(points)

        # Visualize cells with different colors
        vis = visual.empty((512, 512))
        for i, cell in enumerate(voronoi.cells):
            color = palette.get("rainbow", i / len(voronoi.cells))
            vis = geometry.fill_polygon(vis, cell.vertices, color)

            # Draw cell edges
            for edge in cell.edges:
                vis = geometry.draw_line(vis, edge.start, edge.end,
                                        color="black", width=2)

        # Draw seed points
        for point in points:
            vis = geometry.draw_circle(vis, point, radius=3, color="red")

        yield vis
```

**Status:** 💡 Concept - needs Voronoi computation and polygon rendering
**Domains:** Geometry, Visual
**Use Cases:** Spatial analysis, procedural generation, computational geometry

---

## Cellular Automata & Emergence

### Cellular Automata Patterns

**✅ Conway's Game of Life**
```python
# Classic CA visualization
def game_of_life():
    ca = emergence.cellular_automaton(
        rules="conway",
        initial=field.random_binary((256, 256), density=0.3)
    )

    while True:
        ca = emergence.step(ca)
        yield visual.colorize(ca.state, palette="grayscale")

visual.display(game_of_life, target_fps=10)
```

**Status:** ✅ Fully available (v0.9.1)
**Domains:** Emergence, Visual
**Use Cases:** Complexity science, artificial life, education

---

**💡 Multi-State CA with Color**
```python
# Larger-than-life or multiple cell types
def multistate_ca():
    ca = emergence.cellular_automaton(
        rules=custom_multistate_rules,
        initial=field.random_int((256, 256), min=0, max=5),
        states=5
    )

    while True:
        ca = emergence.step(ca)
        # Each state gets a different color
        yield visual.colorize(ca.state, palette="rainbow",
                            vmin=0, vmax=4)
```

**Status:** 🚧 Has CA system, needs multi-state support
**Domains:** Emergence, Visual
**Use Cases:** Traffic models, ecosystem simulation, complex systems

---

**💡 CA Pattern Analysis Overlay**
```python
# Detect and highlight patterns (gliders, oscillators, still lifes)
def ca_pattern_detection():
    ca = emergence.cellular_automaton(rules="conway", initial=random_state)

    while True:
        ca = emergence.step(ca)

        # Detect patterns
        gliders = emergence.detect_gliders(ca)
        oscillators = emergence.detect_oscillators(ca)

        # Base visualization
        vis = visual.colorize(ca.state, palette="grayscale")

        # Overlay pattern highlights
        for glider in gliders:
            bbox = geo.rectangle(glider.bbox, color="red", fill=False)
            vis = visual.composite(vis, bbox, mode="over")

        for osc in oscillators:
            bbox = geo.rectangle(osc.bbox, color="blue", fill=False)
            vis = visual.composite(vis, bbox, mode="over")

        yield vis
```

**Status:** 💡 Concept - needs pattern detection algorithms
**Domains:** Emergence, Geometry, Visual
**Use Cases:** CA research, pattern discovery, complexity analysis

---

### Reaction-Diffusion

**🚧 Reaction-Diffusion Patterns**
```python
# Gray-Scott or other RD systems
def reaction_diffusion():
    u = field.ones((256, 256)) - field.random((256, 256), low=0, high=0.1)
    v = field.random((256, 256), low=0, high=0.1)

    # Parameters for different patterns
    F, k = 0.055, 0.062  # Coral growth

    while True:
        # RD equations
        laplacian_u = field.laplacian(u)
        laplacian_v = field.laplacian(v)

        u_new = u + (Du * laplacian_u - u*v*v + F*(1-u)) * dt
        v_new = v + (Dv * laplacian_v + u*v*v - (F+k)*v) * dt

        u, v = u_new, v_new

        # Visualize v concentration
        yield visual.colorize(v, palette="plasma", vmin=0, vmax=0.5)
```

**Status:** 🚧 Has field operations, needs RD-specific operators
**Domains:** Field, Emergence, Visual
**Use Cases:** Pattern formation, morphogenesis, generative art

---

## Graph & Network

### Graph Layouts

**💡 Force-Directed Graph Layout**
```python
# Visualize network with force-directed layout
def graph_viz():
    # Create graph
    G = graph.erdos_renyi(n=50, p=0.1)

    # Initialize random positions
    positions = {node: [random()*10, random()*10] for node in G.nodes}

    while True:
        # Apply force-directed layout step
        forces = graph.spring_forces(G, positions)
        positions = update_positions(positions, forces, dt=0.01)

        # Visualize
        scene = scene.create("graph")

        # Draw edges
        for edge in G.edges:
            line = geo.line(positions[edge.source], positions[edge.target])
            line.set_color("gray")
            scene.add(line)

        # Draw nodes
        for node in G.nodes:
            dot = geo.dot(positions[node], radius=0.3)
            dot.set_color(palette.get("viridis", node.degree / max_degree))
            scene.add(dot)

        yield scene.render()
```

**Status:** 💡 Concept - needs graph layout algorithms
**Domains:** Graph, Geometry, Visual
**Use Cases:** Network analysis, social networks, system architecture

---

**💡 Hierarchical Tree Layout**
```python
# Tree visualization with different layout algorithms
tree = graph.create_tree(branching_factor=3, depth=4)

# Layout options: radial, dendrogram, tidy tree
layout = graph.layout_tree(tree, method="radial")

scene = scene.create("tree")
# Recursive drawing with animation
def draw_tree(node, depth=0):
    if node.children:
        for child in node.children:
            # Edge from parent to child
            edge = geo.line(layout[node], layout[child])
            edge.set_color(palette.get("viridis", depth/max_depth))
            scene.add(edge)
            anim.fade_in(edge, duration=0.3, delay=depth*0.5)

            # Recurse
            draw_tree(child, depth+1)

    # Node circle
    circle = geo.circle(layout[node], radius=0.2)
    circle.set_fill(palette.get("plasma", depth/max_depth))
    scene.add(circle)
    anim.grow(circle, duration=0.3, delay=depth*0.5)

draw_tree(tree.root)
```

**Status:** 📋 Has graph domain (v0.10.0), needs layout algorithms
**Domains:** Graph, Geometry, Visual
**Use Cases:** Decision trees, taxonomy, file systems

---

### Network Analysis Visualization

**💡 Centrality Heatmap**
```python
# Show node importance via color and size
G = graph.load("social_network.graph")

# Compute centrality measures
betweenness = graph.betweenness_centrality(G)
pagerank = graph.pagerank(G)

# Layout
positions = graph.layout_force_directed(G)

# Visualize
scene = scene.create("centrality")
for node in G.nodes:
    dot = geo.dot(positions[node],
                  radius=0.2 + pagerank[node]*2)  # Size by PageRank
    dot.set_color(palette.get("hot", betweenness[node]))  # Color by betweenness
    scene.add(dot)

# Edges with transparency
for edge in G.edges:
    line = geo.line(positions[edge.source], positions[edge.target])
    line.set_opacity(0.3)
    scene.add(line)
```

**Status:** 🚧 Has centrality (v0.10.0), needs graph rendering
**Domains:** Graph, Geometry, Visual
**Use Cases:** Social network analysis, infrastructure networks, influence analysis

---

**💡 Community Detection Visualization**
```python
# Color nodes by detected community
G = graph.load("network.graph")
communities = graph.community_detection(G, method="louvain")

positions = graph.layout_force_directed(G)

# Assign colors to communities
community_colors = {i: palette.get("rainbow", i/len(communities))
                   for i in range(len(communities))}

scene = scene.create("communities")
for node in G.nodes:
    dot = geo.dot(positions[node], radius=0.3)
    dot.set_color(community_colors[communities[node]])
    scene.add(dot)
```

**Status:** 🚧 Has community detection (v0.10.0), needs rendering
**Domains:** Graph, Visual
**Use Cases:** Social networks, biological networks, modularity analysis

---

## Terrain & Procedural

### Terrain Visualization

**🚧 Heightmap with Shading**
```python
# Terrain generation with procedural features
terrain = terrain.generate(
    size=(512, 512),
    algorithm="perlin",
    octaves=6,
    erosion="hydraulic"
)

# Apply erosion
terrain = terrain.hydraulic_erosion(iterations=100, rain_rate=0.01)

# Visualize with hillshade
hillshade = terrain.hillshade(terrain, azimuth=315, altitude=45)
color_map = visual.colorize(terrain, palette="terrain")

# Composite shading with color
vis = visual.composite(
    color_map,
    hillshade,
    mode="multiply",
    opacity=[1.0, 0.5]
)
```

**Status:** 🚧 Has terrain generation (v0.10.0), needs hillshade and compositing
**Domains:** Terrain, Visual
**Use Cases:** Procedural landscape generation, game development, cartography

---

**💡 3D Terrain Mesh**
```python
# Render terrain as 3D surface
terrain_height = terrain.generate((256, 256), algorithm="diamond_square")

# Create mesh from heightmap
mesh = geo.mesh_from_heightmap(terrain_height)

# Apply texture based on elevation and slope
elevation = terrain_height
slope = terrain.calculate_slope(terrain_height)

# Multi-layered texture
texture = terrain.texture_blend([
    ("grass", 0.0, 0.3),      # Low elevation
    ("rock", 0.3, 0.7),       # Mid elevation
    ("snow", 0.7, 1.0),       # High elevation
], elevation)

# Modulate by slope (rock on steep slopes)
texture = terrain.slope_modulate(texture, slope, rock_threshold=0.5)

mesh.set_texture(texture)
scene.add(mesh)
camera.fly_over(mesh, duration=20, height=50)
```

**Status:** 📋 Needs 3D mesh rendering and camera paths
**Domains:** Terrain, Geometry, Visual
**Use Cases:** Game development, flight simulation, terrain analysis

---

### Biome Visualization

**💡 Biome Map with Temperature/Moisture**
```python
# Whittaker biome diagram applied to terrain
terrain_height = terrain.generate((512, 512))
temperature = terrain.temperature_from_latitude_elevation(terrain_height)
moisture = terrain.moisture_from_rainfall_drainage(terrain_height)

# Classify biomes
biomes = terrain.classify_biomes(temperature, moisture)

# Color map
biome_colors = {
    "tundra": [0.8, 0.8, 1.0],
    "taiga": [0.3, 0.5, 0.3],
    "grassland": [0.7, 0.9, 0.5],
    "desert": [0.9, 0.8, 0.6],
    "rainforest": [0.1, 0.5, 0.2],
    # ... more biomes
}

vis = visual.colorize_categorical(biomes, color_map=biome_colors)
visual.output(vis, "biome_map.png")
```

**Status:** 🚧 Has biome classification (v0.10.0), needs categorical coloring
**Domains:** Terrain, Visual
**Use Cases:** World generation, ecology simulation, game development

---

## Cross-Domain Compositions

These are the most powerful visualizations unique to Morphogen's multi-domain architecture.

### Audio-Visual Coupling

**💡 Frequency-Driven Particle System**
```python
# Particles react to audio spectrum
def audio_reactive_particles():
    # Load or stream audio
    audio_signal = audio.load("music.wav")
    particles = agents.create(n=1000)

    frame = 0
    while frame < len(audio_signal):
        # Extract audio frame
        audio_frame = audio_signal[frame:frame+2048]
        spectrum = signal.fft(audio_frame)
        magnitude = signal.magnitude(spectrum)

        # Map frequency bands to particle properties
        bass = np.mean(magnitude[0:4])
        mid = np.mean(magnitude[4:16])
        treble = np.mean(magnitude[16:64])

        # Update particles based on audio
        particles = agents.apply_force(
            particles,
            force=field.radial(strength=bass*100),
            dt=0.016
        )

        # Visualize
        vis = visual.agents(
            particles,
            color_property='speed',
            size_property='energy',
            palette='plasma',
            blend_mode='additive'
        )

        # Composite with waveform
        waveform_vis = visual.waveform(audio_frame, color="cyan")
        yield visual.composite(vis, waveform_vis, mode="over")

        frame += 2048

visual.display(audio_reactive_particles, target_fps=30)
```

**Status:** 🚧 Has audio and agents, needs coupling and composite rendering
**Domains:** Audio, Signal, Agents, Visual
**Use Cases:** Music visualizers, VJ software, creative coding

---

**💡 Sound-Driven Field Perturbation**
```python
# Audio modulates field evolution
def audio_field_coupling():
    audio_signal = audio.load("drums.wav")
    field_state = field.random((256, 256))

    frame = 0
    while frame < len(audio_signal):
        audio_frame = audio_signal[frame:frame+2048]

        # Compute beat energy
        energy = np.sum(audio_frame**2)

        # Modulate diffusion rate by audio energy
        diffusion_rate = 0.1 + energy * 2.0

        field_state = field.diffuse(field_state, rate=diffusion_rate, dt=0.1)

        # Add impulse at beat detection
        if signal.detect_onset(audio_frame):
            center = (random_int(256), random_int(256))
            impulse = field.gaussian_bump((256, 256), center, sigma=10)
            field_state = field_state + impulse

        yield visual.colorize(field_state, palette="fire")
        frame += 2048
```

**Status:** 🚧 Has field and audio, needs onset detection
**Domains:** Audio, Signal, Field, Visual
**Use Cases:** Audio-reactive art, synesthesia simulation, live performances

---

### Physics-Audio Pipeline

**💡 Physical String Synthesis Visualization**
```python
# Visualize and hear a vibrating string
def string_simulation():
    # Physical string as 1D wave equation
    string = field.zeros((256,))
    velocity = field.zeros((256,))

    # Pluck the string
    string = field.gaussian_bump((256,), center=64, sigma=5, amplitude=1.0)

    audio_buffer = []

    while True:
        # Wave equation step
        string, velocity = physics.wave_equation_1d(
            string, velocity,
            wave_speed=343.0,
            dt=1.0/44100.0,
            boundary="fixed"
        )

        # Sample audio at pickup position
        audio_sample = string[192]  # Pickup at 3/4 length
        audio_buffer.append(audio_sample)

        # Visualize string shape
        vis = visual.line_plot(string, color="blue", thickness=2)

        # Add pickup indicator
        pickup_marker = geo.dot([192, string[192]], radius=5, color="red")
        vis = visual.composite(vis, pickup_marker, mode="over")

        yield vis

    # Play audio
    audio.play(np.array(audio_buffer), sample_rate=44100)
```

**Status:** 💡 Concept - needs 1D wave equation and line plots
**Domains:** Physics, Audio, Visual
**Use Cases:** Musical instrument design, physics education, sound synthesis

---

**💡 Drum Impact Visualization**
```python
# 2D membrane with audio output
def drum_simulation():
    # 2D wave equation (drum head)
    membrane = field.zeros((128, 128))
    velocity = field.zeros((128, 128))

    while True:
        # Strike the drum at random location
        if should_strike():
            strike_pos = (random_int(128), random_int(128))
            impulse = field.gaussian_bump((128, 128), strike_pos, sigma=3)
            velocity = velocity + impulse * 10.0

        # 2D wave equation
        membrane, velocity = physics.wave_equation_2d(
            membrane, velocity,
            wave_speed=100.0,
            dt=1.0/44100.0,
            boundary="fixed"
        )

        # Sample audio from center
        audio_sample = membrane[64, 64]

        # Visualize membrane displacement
        vis = visual.colorize(membrane, palette="coolwarm", vmin=-1, vmax=1)

        yield vis, audio_sample
```

**Status:** 💡 Concept - needs 2D wave equation
**Domains:** Physics, Audio, Field, Visual
**Use Cases:** Percussion synthesis, modal synthesis, physical modeling

---

### Fluid-Audio Coupling

**💡 Fluid Acoustics Visualization**
```python
# Incompressible flow → acoustic pressure → audio
def fluid_acoustics():
    # Velocity field (incompressible Navier-Stokes)
    velocity = field.vector_zeros((128, 128))

    # Acoustic pressure (compressible wave equation)
    pressure = field.zeros((128, 128))
    pressure_vel = field.zeros((128, 128))

    while True:
        # Fluid step
        velocity = fluid.advect(velocity, velocity, dt=0.01)
        velocity = fluid.diffuse(velocity, viscosity=0.001, dt=0.01)
        velocity = fluid.project(velocity)  # Incompressibility

        # Couple to acoustics: div(v) → pressure source
        divergence = field.divergence(velocity)
        pressure_source = divergence * 100.0

        # Acoustic wave propagation
        pressure, pressure_vel = physics.wave_equation_2d(
            pressure, pressure_vel,
            source=pressure_source,
            wave_speed=343.0,
            dt=1.0/44100.0
        )

        # Sample audio
        audio_sample = pressure[64, 64]

        # Visualize: velocity field + pressure overlay
        vel_vis = visual.streamlines(velocity, density=10)
        pressure_vis = visual.colorize(pressure, palette="coolwarm",
                                       vmin=-1, vmax=1, alpha=0.7)

        yield visual.composite(vel_vis, pressure_vis, mode="over"), audio_sample
```

**Status:** 💡 Concept - needs fluid operators and wave equation
**Domains:** Field, Physics, Audio, Visual
**Use Cases:** Acoustic simulation, room acoustics, wind instruments

---

### Optimization-Driven Generative Art

**💡 Evolutionary Art Gallery**
```python
# Genetic algorithm evolves visual patterns
def evolutionary_art():
    # Population of image-generating programs
    population = optimization.ga_init(
        size=16,
        genome_type="expression_tree",
        output_type="image"
    )

    generation = 0
    while True:
        # Evaluate fitness (user selection or aesthetic measure)
        fitnesses = [aesthetic_score(render(ind)) for ind in population]

        # Visualize population as grid
        grid_vis = visual.empty((512, 512))
        for i, individual in enumerate(population):
            # Render individual's program
            img = render_program(individual, size=(128, 128))

            # Place in grid
            row, col = i // 4, i % 4
            grid_vis = visual.paste(grid_vis, img,
                                   position=(col*128, row*128))

            # Show fitness as border color
            border_color = palette.get("viridis", fitnesses[i])
            grid_vis = visual.draw_rectangle(grid_vis,
                                            (col*128, row*128, 128, 128),
                                            color=border_color, width=3)

        yield grid_vis

        # Evolve population
        population = optimization.ga_step(population, fitnesses)
        generation += 1
```

**Status:** 💡 Concept - needs expression tree GA and image grid layout
**Domains:** Optimization, Visual, Procedural
**Use Cases:** Generative art, procedural content, interactive evolution

---

### Terrain-Physics-Audio Chain

**💡 Landslide Sonification**
```python
# Terrain erosion with audio feedback
def landslide_audio():
    terrain = terrain.generate((256, 256), algorithm="perlin")

    while True:
        # Detect unstable regions (high slope)
        slope = terrain.calculate_slope(terrain)
        unstable = slope > 0.7

        # Simulate mass movement
        terrain_new, displaced_mass = terrain.simulate_erosion(
            terrain,
            method="gravitational",
            dt=0.1
        )

        # Sonify based on displacement
        total_displacement = np.sum(displaced_mass)
        frequency = 100 + total_displacement * 1000
        amplitude = np.clip(total_displacement * 10, 0, 1)

        audio_sample = audio.oscillator(freq=frequency, amp=amplitude,
                                       shape="noise") * 0.1

        # Visualize
        terrain_vis = visual.colorize(terrain, palette="terrain")
        unstable_overlay = visual.colorize(unstable, palette="red", alpha=0.5)
        vis = visual.composite(terrain_vis, unstable_overlay, mode="over")

        yield vis, audio_sample

        terrain = terrain_new
```

**Status:** 💡 Concept - needs erosion simulation and audio coupling
**Domains:** Terrain, Physics, Audio, Visual
**Use Cases:** Data sonification, terrain simulation, artistic installations

---

## Scientific & Educational

### Mathematical Demonstrations

**📋 Fourier Series Visualization**
```python
# Build up square wave from harmonics
def fourier_series():
    scene = scene.create("fourier")

    # Target function (square wave)
    target = geo.curve(lambda t: np.sign(np.sin(t)), (0, 4*pi))
    target.set_color("gray")
    target.set_opacity(0.3)
    scene.add(target)

    # Start with fundamental
    approximation = lambda t: (4/pi) * np.sin(t)

    # Add harmonics one by one
    for n in range(1, 20, 2):  # Odd harmonics
        # Add nth harmonic
        term = lambda t, n=n: (4/pi) * np.sin(n*t) / n
        approximation_prev = approximation
        approximation = lambda t: approximation_prev(t) + term(t)

        # Show individual harmonic
        harmonic_curve = geo.curve(term, (0, 4*pi))
        harmonic_curve.set_color(palette.get("rainbow", n/20))
        scene.add(harmonic_curve)
        anim.fade_in(harmonic_curve, duration=0.5)

        # Update sum
        sum_curve = geo.curve(approximation, (0, 4*pi))
        sum_curve.set_color("red")
        sum_curve.set_thickness(3)
        scene.add(sum_curve)
        anim.draw_curve(sum_curve, duration=1.0)

        yield scene.render()

        # Fade out individual harmonic
        anim.fade_out(harmonic_curve, duration=0.3)
```

**Status:** 📋 Needs curve animation system
**Domains:** Signal, Geometry, Visual
**Use Cases:** Signal processing education, Fourier analysis, mathematics

---

**💡 Gradient Descent on Surface**
```python
# Show optimization path on 3D cost surface
cost_function = lambda x, y: rosenbrock([x, y])

# Create 3D surface
surface = geo.parametric_surface(
    lambda u, v: [u, v, cost_function(u, v)],
    u=(-2, 2), v=(-2, 2)
)
surface.color_by_field(cost_function, palette="viridis")

# Run gradient descent
optimizer = optimization.gradient_descent(start=[1.5, 1.5], lr=0.01)
path_3d = []

for step in range(100):
    pos_2d = optimizer.step(cost_function)
    pos_3d = [pos_2d[0], pos_2d[1], cost_function(*pos_2d)]
    path_3d.append(pos_3d)

# Visualize path on surface
path_curve = geo.curve_from_points(path_3d)
path_curve.set_color("red")
path_curve.set_thickness(3)

scene.add([surface, path_curve])
anim.draw_curve(path_curve, duration=5.0)
camera.orbit(2*pi, duration=10.0)
```

**Status:** 📋 Needs 3D surface and path rendering
**Domains:** Optimization, Geometry, Visual
**Use Cases:** Machine learning education, optimization visualization, research

---

### Physics Demonstrations

**💡 Double Pendulum Chaos**
```python
# Show sensitive dependence on initial conditions
def double_pendulum():
    # Two pendulums with slightly different initial conditions
    pendulum1 = physics.double_pendulum(theta1=0.1, theta2=0.0)
    pendulum2 = physics.double_pendulum(theta1=0.10001, theta2=0.0)

    trail1, trail2 = [], []

    while True:
        # Step physics
        pendulum1 = physics.step(pendulum1, dt=0.01)
        pendulum2 = physics.step(pendulum2, dt=0.01)

        # Track end of second arm
        trail1.append(pendulum1.end_position)
        trail2.append(pendulum2.end_position)

        # Visualize both pendulums
        scene = scene.create("chaos")

        # Pendulum 1
        arm1_1 = geo.line([0, 0], pendulum1.joint1_pos)
        arm1_2 = geo.line(pendulum1.joint1_pos, pendulum1.end_pos)
        arm1_1.set_color("blue")
        arm1_2.set_color("blue")

        # Pendulum 2
        arm2_1 = geo.line([0, 0], pendulum2.joint1_pos)
        arm2_2 = geo.line(pendulum2.joint1_pos, pendulum2.end_pos)
        arm2_1.set_color("red")
        arm2_2.set_color("red")

        # Trails
        trail1_curve = geo.curve_from_points(trail1[-100:])
        trail2_curve = geo.curve_from_points(trail2[-100:])
        trail1_curve.set_color("cyan")
        trail2_curve.set_color("orange")
        trail1_curve.set_opacity(0.5)
        trail2_curve.set_opacity(0.5)

        scene.add([arm1_1, arm1_2, arm2_1, arm2_2,
                  trail1_curve, trail2_curve])

        yield scene.render()
```

**Status:** 💡 Concept - needs pendulum physics and line rendering
**Domains:** Physics, Geometry, Visual
**Use Cases:** Chaos theory education, nonlinear dynamics, physics demonstrations

---

## Creative & Generative Art

### Procedural Patterns

**✅ Noise-Based Patterns**
```python
# Layered noise for organic patterns
def procedural_pattern():
    # Combine multiple noise octaves
    pattern = field.zeros((512, 512))

    for octave in range(6):
        frequency = 2 ** octave
        amplitude = 1.0 / (2 ** octave)

        noise_layer = noise.perlin2d(
            x * frequency / 512,
            y * frequency / 512,
            seed=42 + octave
        )

        pattern = pattern + noise_layer * amplitude

    # Normalize and colorize
    pattern = field.normalize(pattern, vmin=0, vmax=1)
    vis = visual.colorize(pattern, palette="viridis")

    return vis

visual.output(procedural_pattern(), "pattern.png")
```

**Status:** ✅ Can be implemented with current noise and field domains
**Domains:** Noise, Field, Visual
**Use Cases:** Texture generation, background patterns, generative art

---

**💡 Reaction-Diffusion Art**
```python
# Use RD for organic pattern generation
def rd_art():
    u, v = field.ones((512, 512)), field.random((512, 512), low=0, high=0.1)

    # Try different parameter sets for different patterns
    params = [
        (0.055, 0.062),  # Coral
        (0.035, 0.065),  # Spots
        (0.012, 0.050),  # Waves
        (0.025, 0.055),  # Stripes
    ]

    F, k = params[0]

    for step in range(1000):
        u, v = reaction_diffusion_step(u, v, F, k, dt=1.0)

    # Composite multiple RD results with different blending
    vis = visual.colorize(v, palette="plasma")
    return vis

visual.output(rd_art(), "rd_pattern.png")
```

**Status:** 🚧 Needs RD operators
**Domains:** Field, Emergence, Visual
**Use Cases:** Generative art, texture synthesis, pattern design

---

### Animated Abstracts

**💡 Morphing Shapes**
```python
# Smooth transitions between geometric shapes
def shape_morph():
    shapes = [
        geo.circle(center=[0, 0], radius=5),
        geo.rectangle(center=[0, 0], width=8, height=8),
        geo.polygon(points=star_points(n=5, radius=5)),
        geo.ellipse(center=[0, 0], a=6, b=4),
    ]

    scene = scene.create("morph")

    # Start with first shape
    current = shapes[0]
    current.set_fill(palette.get("rainbow", 0.0))
    scene.add(current)

    # Morph through all shapes
    for i, next_shape in enumerate(shapes[1:]):
        next_shape.set_fill(palette.get("rainbow", (i+1)/len(shapes)))

        # Morph animation
        anim.morph(current, to=next_shape, duration=2.0)
        anim.wait(0.5)

        current = next_shape

    # Return to first
    anim.morph(current, to=shapes[0], duration=2.0)
```

**Status:** 📋 Needs shape morphing animation
**Domains:** Geometry, Visual
**Use Cases:** Motion graphics, logo animations, generative art

---

**💡 Particle Flow Fields**
```python
# Particles following curl noise
def flow_field_art():
    # Create curl noise field
    noise_field = noise.curl_noise_2d(
        x, y, t,
        frequency=0.01,
        octaves=3
    )

    # Initialize particles
    particles = agents.create([
        agents.particle(pos=[random()*512, random()*512])
        for _ in range(10000)
    ])

    while True:
        # Move particles along field
        particles = agents.move_by_field(
            particles,
            noise_field,
            strength=2.0,
            dt=0.016
        )

        # Wrap around boundaries
        particles = agents.wrap_boundaries(particles, (0, 512), (0, 512))

        # Visualize with trails and additive blending
        vis = visual.agents(
            particles,
            color_property='speed',
            size=2,
            trail=True,
            trail_length=50,
            palette='plasma',
            blend_mode='additive'
        )

        yield vis
```

**Status:** 🚧 Needs curl noise and agent-field coupling
**Domains:** Noise, Field, Agents, Visual
**Use Cases:** Generative art, flow visualization, creative coding

---

## Implementation Priority Suggestions

Based on impact and feasibility:

### High Priority (Immediate Impact)
1. **Audio waveform/spectrum plotting** - Essential for audio work
2. **Line/curve rendering** - Foundation for many visualizations
3. **Vector field streamlines** - Critical for field analysis
4. **Multi-layer compositing** - Enables rich visualizations
5. **Scatter plots** - Useful across many domains

### Medium Priority (Powerful Additions)
1. **3D surface rendering** - Opens up new visualization categories
2. **Arrow/glyph rendering** - Force visualization, vector fields
3. **Animation system** - Explanatory graphics, education
4. **Graph layout algorithms** - Network visualization
5. **Contour extraction** - Scientific visualization

### Long-Term (Advanced Features)
1. **Equation rendering (LaTeX)** - Mathematical visualization
2. **Volume rendering** - 3D field visualization
3. **Shader-based effects** - Performance and artistic effects
4. **Interactive widgets** - Parameter exploration
5. **VR/AR support** - Immersive visualization

---

## Contributing Ideas

When adding new visualization concepts to this document:

1. **Specify status** using the legend (✅ 🚧 📋 💡)
2. **List domains** involved in the visualization
3. **Provide code sketch** showing the API design
4. **Identify use cases** - who benefits?
5. **Note dependencies** - what needs to be implemented first?

---

*This document is a living catalog. Add ideas as they emerge, and update status as features are implemented.*
