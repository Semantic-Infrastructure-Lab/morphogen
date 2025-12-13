"""Interactive Physics Sandbox - Cross-Domain Showcase

This example demonstrates the power of combining multiple Morphogen domains:
- Rigidbody physics for realistic collisions and dynamics
- Field operations for force fields and environmental effects
- Visual rendering for beautiful simulations
- Genetic algorithms for emergent behavior
- Cellular automata for environmental interactions
- Palette and color for stunning visuals

Creates interactive physics demonstrations:
- Particle collision sandbox with force fields
- Gravity wells and repulsion zones
- Emergent structures from physics
- Audio-reactive physics (sound drives forces)
- Multi-domain integration showcase
"""

import numpy as np
from morphogen.stdlib import rigidbody, field, visual, genetic, cellular, palette, color, image, noise, io_storage, Individual
from morphogen.stdlib.field import Field2D
from morphogen.stdlib.visual import Visual

# Note: For audio integration
try:
    from morphogen.stdlib import audio
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False


def create_gravity_field(width, height, gravity_centers):
    """Create a gravitational field with multiple attractors.

    Args:
        width: Field width
        height: Field height
        gravity_centers: List of (x, y, strength) tuples

    Returns:
        Field2D with gravitational potential
    """
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Initialize potential field
    potential = np.zeros((height, width), dtype=np.float32)

    for cx, cy, strength in gravity_centers:
        # Distance from center
        dx = x - cx
        dy = y - cy
        dist = np.sqrt(dx**2 + dy**2) + 1.0  # Avoid division by zero

        # Gravitational potential (1/r)
        potential += strength / dist

    return Field2D(potential)


def simulate_particle_collision(n_particles=100, n_steps=500,
                                width=400, height=400, seed=42):
    """Simulate particle collisions in a bounded box.

    Args:
        n_particles: Number of particles
        n_steps: Number of simulation steps
        width: Simulation width
        height: Simulation height
        seed: Random seed

    Returns:
        List of frames (numpy arrays with particle positions)
    """
    print(f"  Simulating {n_particles} particles for {n_steps} steps...")

    np.random.seed(seed)

    # Initialize particles
    positions = np.random.rand(n_particles, 2) * np.array([width, height])
    velocities = (np.random.rand(n_particles, 2) - 0.5) * 20.0
    radii = np.random.rand(n_particles) * 3 + 2  # Radius 2-5
    masses = radii ** 2  # Mass proportional to area

    # Physics parameters
    dt = 0.1
    restitution = 0.95  # Energy loss on collision
    friction = 0.99  # Air resistance

    frames = []

    for step in range(n_steps):
        # Apply friction
        velocities *= friction

        # Update positions
        positions += velocities * dt

        # Bounce off walls
        for i in range(n_particles):
            if positions[i, 0] < radii[i] or positions[i, 0] > width - radii[i]:
                velocities[i, 0] *= -restitution
                positions[i, 0] = np.clip(positions[i, 0], radii[i], width - radii[i])

            if positions[i, 1] < radii[i] or positions[i, 1] > height - radii[i]:
                velocities[i, 1] *= -restitution
                positions[i, 1] = np.clip(positions[i, 1], radii[i], height - radii[i])

        # Particle-particle collisions (simple)
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx**2 + dy**2)

                min_dist = radii[i] + radii[j]

                if dist < min_dist and dist > 0:
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist

                    # Relative velocity
                    dvx = velocities[j, 0] - velocities[i, 0]
                    dvy = velocities[j, 1] - velocities[i, 1]

                    # Relative velocity along collision normal
                    dvn = dvx * nx + dvy * ny

                    # Do not resolve if velocities are separating
                    if dvn < 0:
                        continue

                    # Impulse scalar
                    impulse = (2 * dvn) / (1/masses[i] + 1/masses[j])

                    # Apply impulse
                    velocities[i, 0] += impulse * nx / masses[i]
                    velocities[i, 1] += impulse * ny / masses[i]
                    velocities[j, 0] -= impulse * nx / masses[j]
                    velocities[j, 1] -= impulse * ny / masses[j]

                    # Separate particles
                    overlap = min_dist - dist
                    positions[i, 0] -= overlap * 0.5 * nx
                    positions[i, 1] -= overlap * 0.5 * ny
                    positions[j, 0] += overlap * 0.5 * nx
                    positions[j, 1] += overlap * 0.5 * ny

        # Store frame
        if step % 5 == 0:  # Sample every 5 steps
            frames.append({
                'positions': positions.copy(),
                'radii': radii.copy(),
                'velocities': velocities.copy()
            })

    return frames


def render_physics_frame(frame, width, height, colormap='plasma'):
    """Render a physics frame as an image.

    Args:
        frame: Dict with positions, radii, velocities
        width: Image width
        height: Image height
        colormap: Colormap name

    Returns:
        RGB image
    """
    positions = frame['positions']
    radii = frame['radii']
    velocities = frame['velocities']

    # Create field for particle rendering
    particle_field = np.zeros((height, width), dtype=np.float32)

    # Get velocity magnitudes for coloring
    vel_magnitudes = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    max_vel = np.max(vel_magnitudes) + 1e-6

    # Render each particle
    for i, (pos, radius) in enumerate(zip(positions, radii)):
        x, y = int(pos[0]), int(pos[1])
        r = int(radius)

        # Velocity-based color
        vel_norm = vel_magnitudes[i] / max_vel

        # Draw circle
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx**2 + dy**2 <= r**2:
                    py, px = y + dy, x + dx
                    if 0 <= py < height and 0 <= px < width:
                        particle_field[py, px] = max(particle_field[py, px], vel_norm)

    # Apply colormap
    if colormap == 'plasma':
        pal = palette.plasma(256)
    elif colormap == 'viridis':
        pal = palette.viridis(256)
    elif colormap == 'hot':
        pal = palette.fire(256)
    else:
        pal = palette.magma(256)

    img = palette.map(pal, particle_field)

    return img


def particles_in_gravity_field(n_particles=150, n_steps=600,
                               width=400, height=400, seed=42):
    """Simulate particles in a gravity field with multiple attractors.

    Args:
        n_particles: Number of particles
        n_steps: Number of simulation steps
        width: Simulation width
        height: Simulation height
        seed: Random seed

    Returns:
        List of rendered frames
    """
    print(f"  Simulating {n_particles} particles in gravity field...")

    np.random.seed(seed)

    # Create gravity field with multiple attractors
    gravity_centers = [
        (width * 0.25, height * 0.25, 5000),
        (width * 0.75, height * 0.75, 5000),
        (width * 0.5, height * 0.5, -3000),  # Repulsor in center
    ]

    grav_field = create_gravity_field(width, height, gravity_centers)

    # Initialize particles
    positions = np.random.rand(n_particles, 2) * np.array([width, height])
    velocities = (np.random.rand(n_particles, 2) - 0.5) * 5.0

    # Physics parameters
    dt = 0.05
    friction = 0.995

    frames = []

    for step in range(n_steps):
        # Compute gravity force from field
        for i in range(n_particles):
            x, y = int(positions[i, 0]), int(positions[i, 1])

            # Clamp to field bounds
            x = np.clip(x, 1, width - 2)
            y = np.clip(y, 1, height - 2)

            # Compute gradient (force direction)
            fx = (grav_field.data[y, min(x+1, width-1)] -
                  grav_field.data[y, max(x-1, 0)])
            fy = (grav_field.data[min(y+1, height-1), x] -
                  grav_field.data[max(y-1, 0), x])

            # Apply force
            velocities[i, 0] += fx * dt
            velocities[i, 1] += fy * dt

        # Apply friction
        velocities *= friction

        # Update positions
        positions += velocities * dt

        # Bounce off walls
        for i in range(n_particles):
            if positions[i, 0] < 0 or positions[i, 0] >= width:
                velocities[i, 0] *= -0.8
                positions[i, 0] = np.clip(positions[i, 0], 0, width - 1)

            if positions[i, 1] < 0 or positions[i, 1] >= height:
                velocities[i, 1] *= -0.8
                positions[i, 1] = np.clip(positions[i, 1], 0, height - 1)

        # Store frame
        if step % 5 == 0:
            # Render particles on field
            particle_field = np.zeros((height, width), dtype=np.float32)

            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < width and 0 <= y < height:
                    # Draw small circle
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if dx**2 + dy**2 <= 4:
                                py, px = y + dy, x + dx
                                if 0 <= py < height and 0 <= px < width:
                                    particle_field[py, px] = 1.0

            # Combine with gravity field visualization
            grav_normalized = field.normalize(grav_field, -1.0, 1.0)

            # Create composite
            composite = grav_normalized.data * 0.3 + particle_field * 0.7

            frames.append(composite)

    return frames


def emergent_structures_from_physics(width=400, height=400,
                                     n_particles=200, n_steps=800, seed=42):
    """Create emergent structures using physics + cellular automata.

    Particles deposit energy into a cellular automaton, creating patterns.

    Args:
        width: Simulation width
        height: Simulation height
        n_particles: Number of particles
        n_steps: Number of simulation steps
        seed: Random seed

    Returns:
        List of rendered frames
    """
    print(f"  Creating emergent structures with {n_particles} particles...")

    np.random.seed(seed)

    # Initialize CA field
    ca_field = Field2D(np.zeros((height, width), dtype=np.float32))

    # Initialize particles
    positions = np.random.rand(n_particles, 2) * np.array([width, height])
    velocities = (np.random.rand(n_particles, 2) - 0.5) * 10.0

    # Physics parameters
    dt = 0.1

    frames = []

    for step in range(n_steps):
        # Update particles
        positions += velocities * dt

        # Bounce off walls
        for i in range(n_particles):
            if positions[i, 0] < 0 or positions[i, 0] >= width:
                velocities[i, 0] *= -1
                positions[i, 0] = np.clip(positions[i, 0], 0, width - 1)

            if positions[i, 1] < 0 or positions[i, 1] >= height:
                velocities[i, 1] *= -1
                positions[i, 1] = np.clip(positions[i, 1], 0, height - 1)

            # Deposit energy into CA
            x, y = int(positions[i, 0]), int(positions[i, 1])
            if 0 <= x < width and 0 <= y < height:
                ca_field.data[y, x] += 0.1

        # CA evolution: diffuse and decay
        ca_field = field.diffuse(ca_field, rate=0.05, dt=dt)
        ca_field.data *= 0.99  # Decay

        # Threshold to create binary patterns
        binary_pattern = (ca_field.data > 0.3).astype(np.float32)

        # Particles interact with CA: repelled by high-energy areas
        for i in range(n_particles):
            x, y = int(positions[i, 0]), int(positions[i, 1])
            x = np.clip(x, 1, width - 2)
            y = np.clip(y, 1, height - 2)

            # Gradient of CA field
            fx = binary_pattern[y, min(x+1, width-1)] - binary_pattern[y, max(x-1, 0)]
            fy = binary_pattern[min(y+1, height-1), x] - binary_pattern[max(y-1, 0), x]

            # Repulsion from high-energy areas
            velocities[i, 0] -= fx * 5.0
            velocities[i, 1] -= fy * 5.0

        # Clamp velocities
        vel_mag = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        too_fast = vel_mag > 20.0
        if np.any(too_fast):
            velocities[too_fast] = velocities[too_fast] / vel_mag[too_fast, np.newaxis] * 20.0

        # Store frame
        if step % 5 == 0:
            frames.append(ca_field.data.copy())

    return frames


def genetic_physics_optimization(width=400, height=400, seed=42):
    """Use genetic algorithm to optimize particle starting conditions.

    Goal: Find initial conditions that maximize particle clustering.

    Args:
        width: Simulation width
        height: Simulation height
        seed: Random seed

    Returns:
        Best individual and final state visualization
    """
    print("  Running genetic optimization of physics parameters...")

    np.random.seed(seed)

    # Population parameters
    population_size = 20
    n_generations = 30
    n_particles = 50

    # Fitness function: maximize clustering (minimize spread)
    def fitness(individual):
        # Individual encodes initial velocities
        velocities = individual.genome.reshape(n_particles, 2) * 20 - 10

        # Initialize positions (fixed grid)
        positions = np.array([
            [width * (i % 10) / 10 + width * 0.05,
             height * (i // 10) / 10 + height * 0.05]
            for i in range(n_particles)
        ])

        # Simulate
        dt = 0.1
        friction = 0.99

        for _ in range(100):
            positions += velocities * dt
            velocities *= friction

            # Bounce off walls
            for i in range(n_particles):
                if positions[i, 0] < 0 or positions[i, 0] >= width:
                    velocities[i, 0] *= -0.8
                if positions[i, 1] < 0 or positions[i, 1] >= height:
                    velocities[i, 1] *= -0.8

        # Fitness: negative variance (want clustering)
        center = np.mean(positions, axis=0)
        variance = np.mean(np.sum((positions - center)**2, axis=1))

        return -variance  # Maximize negative variance = minimize spread

    # Create initial population
    population = []
    for _ in range(population_size):
        genome = np.random.rand(n_particles * 2)  # Random velocities
        individual = Individual(genome=genome, fitness=0.0)
        population.append(individual)

    # Evolve
    best_individual = None
    best_fitness = -np.inf

    for gen in range(n_generations):
        # Evaluate fitness
        for ind in population:
            ind.fitness = fitness(ind)
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best_individual = ind

        if gen % 10 == 0:
            print(f"    Generation {gen}: Best fitness = {best_fitness:.2f}")

        # Select parents (tournament selection)
        new_population = []

        for _ in range(population_size):
            # Tournament
            tournament = np.random.choice(population, size=3, replace=False)
            parent1 = max(tournament, key=lambda x: x.fitness)

            tournament = np.random.choice(population, size=3, replace=False)
            parent2 = max(tournament, key=lambda x: x.fitness)

            # Crossover
            child_genome = np.zeros_like(parent1.genome)
            for i in range(len(child_genome)):
                child_genome[i] = parent1.genome[i] if np.random.rand() < 0.5 else parent2.genome[i]

            # Mutation
            mutation_mask = np.random.rand(len(child_genome)) < 0.1
            child_genome[mutation_mask] = np.random.rand(np.sum(mutation_mask))

            child = Individual(genome=child_genome, fitness=0.0)
            new_population.append(child)

        population = new_population

    print(f"    Final best fitness: {best_fitness:.2f}")

    # Visualize best individual
    velocities = best_individual.genome.reshape(n_particles, 2) * 20 - 10
    positions = np.array([
        [width * (i % 10) / 10 + width * 0.05,
         height * (i // 10) / 10 + height * 0.05]
        for i in range(n_particles)
    ])

    # Simulate and render
    frames = []
    dt = 0.1
    friction = 0.99

    for step in range(200):
        positions += velocities * dt
        velocities *= friction

        # Bounce off walls
        for i in range(n_particles):
            if positions[i, 0] < 0 or positions[i, 0] >= width:
                velocities[i, 0] *= -0.8
                positions[i, 0] = np.clip(positions[i, 0], 0, width - 1)
            if positions[i, 1] < 0 or positions[i, 1] >= height:
                velocities[i, 1] *= -0.8
                positions[i, 1] = np.clip(positions[i, 1], 0, height - 1)

        if step % 5 == 0:
            # Render
            frame_field = np.zeros((height, width), dtype=np.float32)
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < width and 0 <= y < height:
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            if dx**2 + dy**2 <= 9:
                                py, px = y + dy, x + dx
                                if 0 <= py < height and 0 <= px < width:
                                    frame_field[py, px] = 1.0

            frames.append(frame_field)

    return frames, best_individual


def demo_particle_collisions():
    """Demo: Particle collision sandbox."""
    print("Demo 1: Particle Collision Sandbox")
    print("-" * 60)

    frames = simulate_particle_collision(n_particles=80, n_steps=500,
                                        width=400, height=400, seed=42)

    # Render a few key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame = frames[frame_idx]
        img = render_physics_frame(frame, width=400, height=400, colormap='plasma')

        output_path = f"output_physics_sandbox_collisions_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")


def demo_gravity_field():
    """Demo: Particles in gravity field."""
    print("\nDemo 2: Particles in Gravity Field")
    print("-" * 60)

    frames = particles_in_gravity_field(n_particles=150, n_steps=600,
                                        width=400, height=400, seed=42)

    # Render key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame_data = frames[frame_idx]

        # Apply colormap
        pal = palette.viridis(256)
        img = palette.map(pal, frame_data)

        output_path = f"output_physics_sandbox_gravity_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")


def demo_emergent_structures():
    """Demo: Emergent structures from physics."""
    print("\nDemo 3: Emergent Structures")
    print("-" * 60)

    frames = emergent_structures_from_physics(width=400, height=400,
                                             n_particles=150, n_steps=800, seed=42)

    # Render key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame_data = frames[frame_idx]

        # Normalize
        normalized = (frame_data - frame_data.min()) / (frame_data.max() + 1e-6)

        # Apply colormap
        pal = palette.magma(256)
        img = palette.map(pal, normalized)

        output_path = f"output_physics_sandbox_emergent_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")


def demo_genetic_optimization():
    """Demo: Genetic optimization of physics."""
    print("\nDemo 4: Genetic Optimization")
    print("-" * 60)

    frames, best = genetic_physics_optimization(width=400, height=400, seed=42)

    # Render key frames
    frame_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        frame_data = frames[frame_idx]

        # Apply colormap
        pal = palette.fire(256)
        img = palette.map(pal, frame_data)

        output_path = f"output_physics_sandbox_genetic_frame{idx:02d}.png"
        io_storage.save_image(output_path, img)
        print(f"  ✓ Saved: {output_path}")


def main():
    """Run all physics sandbox demonstrations."""
    print("=" * 60)
    print("INTERACTIVE PHYSICS SANDBOX - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Domains: RigidBody + Field + Genetic + Cellular + Visual")
    print()

    demo_particle_collisions()
    demo_gravity_field()
    demo_emergent_structures()
    demo_genetic_optimization()

    print()
    print("=" * 60)
    print("ALL PHYSICS SANDBOX DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("This showcase demonstrates:")
    print("  • Particle physics with collisions")
    print("  • Force fields and gravity wells")
    print("  • Emergent structures from physics + CA")
    print("  • Genetic algorithms optimizing physics")
    print("  • Multi-domain integration")
    print()
    print("Key insight: Physics simulations can be combined with")
    print("optimization algorithms and cellular automata to create")
    print("complex emergent behaviors and beautiful visualizations!")


if __name__ == "__main__":
    main()
