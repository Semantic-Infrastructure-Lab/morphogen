"""Physics Simulation Visualizer - Cross-Domain Showcase Example

This example demonstrates the power of combining:
- Integrators for time evolution of physical systems
- Field operations for spatial computations
- Palette for physical quantity visualization
- Image for output and animation
- I/O for saving simulation results

Creates compelling visualizations of:
- Heat diffusion with sources and sinks
- Wave propagation and interference
- Reaction-diffusion patterns
- N-body gravitational systems
- Coupled oscillator networks
"""

import numpy as np
from morphogen.stdlib import integrators, field, palette, image, io_storage
from morphogen.stdlib.field import Field2D


def heat_diffusion_with_sources():
    """Demo 1: Heat diffusion with multiple heat sources."""
    print("Demo 1: Heat Diffusion with Multiple Sources")
    print("-" * 60)

    # Initialize temperature field
    width, height = 400, 300
    print(f"  - Initializing {width}x{height} temperature field...")
    temperature = np.zeros((height, width), dtype=np.float32)

    # Add heat sources (hot spots)
    temperature[75, 100] = 1.0   # Top-left source
    temperature[225, 300] = 1.0  # Bottom-right source
    temperature[150, 200] = 0.8  # Center source

    # Add cold sinks
    temperature[75, 300] = -0.5  # Top-right sink
    temperature[225, 100] = -0.5 # Bottom-left sink

    # Diffusion parameters
    dx = 1.0
    diffusivity = 0.5
    dt = 0.1

    def heat_equation(t, u):
        """Heat equation: du/dt = D * laplacian(u)"""
        u_field = Field2D(u.reshape(height, width, 1), dx=dx, dy=dx)
        laplacian = field.laplacian(u_field)
        dudt = diffusivity * laplacian.data.squeeze()
        return dudt.flatten()

    # Time integration
    print("  - Integrating heat equation (RK4, 200 steps)...")
    times = []
    states = []
    u0 = temperature.flatten()

    for step in range(200):
        t = step * dt
        u_next = integrators.rk4(t, u0, heat_equation, dt)
        u0 = u_next

        if step % 40 == 0:
            times.append(t)
            states.append(u_next.reshape(height, width))

    # Visualize evolution
    print("  - Creating temperature visualizations...")
    pal_temp = palette.from_gradient([
        (0.0, (0.0, 0.0, 0.5)),    # Cold (blue)
        (0.5, (0.0, 0.0, 0.0)),    # Neutral (black)
        (1.0, (1.0, 0.3, 0.0))     # Hot (red-orange)
    ], resolution=256, name="temperature")

    images = []
    for i, state in enumerate(states):
        # Normalize for visualization
        normalized = (state - state.min()) / (state.max() - state.min() + 1e-8)
        img = image.from_field(normalized, pal_temp)
        images.append(img)
        print(f"    Frame {i}: t={times[i]:.2f}, T_max={state.max():.3f}")

    print(f"  ✓ Generated {len(images)} temperature field frames")
    print()
    return images


def wave_interference_pattern():
    """Demo 2: Wave equation with multiple sources showing interference."""
    print("Demo 2: Wave Interference Pattern")
    print("-" * 60)

    # Initialize wave field (displacement and velocity)
    width, height = 512, 512
    print(f"  - Initializing {width}x{height} wave field...")

    u = np.zeros((height, width), dtype=np.float32)  # Displacement
    v = np.zeros((height, width), dtype=np.float32)  # Velocity

    # Add initial disturbances (wave sources)
    # Create Gaussian bumps at different locations
    def gaussian(x, y, cx, cy, sigma):
        return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Three wave sources
    u += 0.5 * gaussian(x, y, 128, 256, 20)
    u += 0.5 * gaussian(x, y, 384, 256, 20)
    u += 0.3 * gaussian(x, y, 256, 128, 15)

    # Wave equation parameters
    c = 50.0  # Wave speed
    dx = 1.0
    dt = 0.01

    def wave_equation_2d(t, state):
        """Wave equation: d²u/dt² = c² * laplacian(u)"""
        # State vector contains [u, v] flattened
        n = len(state) // 2
        u_flat = state[:n]
        v_flat = state[n:]

        u_curr = u_flat.reshape(height, width)
        u_field = Field2D(u_curr.reshape(height, width, 1), dx=dx, dy=dx)

        # Compute Laplacian
        laplacian = field.laplacian(u_field)
        lap_u = laplacian.data.squeeze()

        # Wave equation: dv/dt = c² * laplacian(u), du/dt = v
        dv_dt = (c**2) * lap_u
        du_dt = v_flat.reshape(height, width)

        return np.concatenate([du_dt.flatten(), dv_dt.flatten()])

    # Time integration
    print("  - Integrating wave equation (RK4)...")
    state0 = np.concatenate([u.flatten(), v.flatten()])

    frames = []
    for step in range(150):
        t = step * dt
        state_next = integrators.rk4(t, state0, wave_equation_2d, dt)
        state0 = state_next

        if step % 15 == 0:
            # Extract displacement
            n = len(state_next) // 2
            u_curr = state_next[:n].reshape(height, width)
            frames.append(u_curr.copy())

    # Visualize
    print("  - Creating wave visualizations...")
    pal_wave = palette.from_gradient([
        (0.0, (0.0, 0.2, 0.5)),    # Trough (blue)
        (0.5, (1.0, 1.0, 1.0)),    # Zero (white)
        (1.0, (0.5, 0.0, 0.0))     # Peak (red)
    ], resolution=256, name="wave")

    images = []
    for i, frame in enumerate(frames):
        # Normalize around zero
        vmax = max(abs(frame.min()), abs(frame.max()))
        if vmax > 0:
            normalized = (frame + vmax) / (2 * vmax)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(frame)

        img = image.from_field(normalized, pal_wave)
        images.append(img)
        print(f"    Frame {i}: t={i*15*dt:.3f}, amplitude={vmax:.4f}")

    print(f"  ✓ Generated {len(images)} wave interference frames")
    print()
    return images


def reaction_diffusion_simulation():
    """Demo 3: Gray-Scott reaction-diffusion system."""
    print("Demo 3: Reaction-Diffusion Pattern Formation")
    print("-" * 60)

    # Initialize chemical concentrations
    size = 256
    print(f"  - Initializing {size}x{size} chemical fields...")

    # U and V concentrations
    U = np.ones((size, size), dtype=np.float32)
    V = np.zeros((size, size), dtype=np.float32)

    # Add initial perturbation
    center = size // 2
    r = 20
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= r**2
    U[mask] = 0.5
    V[mask] = 0.25

    # Gray-Scott parameters (coral pattern)
    F = 0.055  # Feed rate
    k = 0.062  # Kill rate
    Du = 0.16  # U diffusion rate
    Dv = 0.08  # V diffusion rate
    dx = 1.0
    dt = 1.0

    def gray_scott(t, state):
        """Gray-Scott reaction-diffusion equations."""
        n = len(state) // 2
        U_flat = state[:n]
        V_flat = state[n:]

        U_curr = U_flat.reshape(size, size)
        V_curr = V_flat.reshape(size, size)

        # Compute Laplacians
        U_field = Field2D(U_curr.reshape(size, size, 1), dx=dx, dy=dx)
        V_field = Field2D(V_curr.reshape(size, size, 1), dx=dx, dy=dx)

        lap_U = field.laplacian(U_field).data.squeeze()
        lap_V = field.laplacian(V_field).data.squeeze()

        # Reaction terms
        UVV = U_curr * V_curr * V_curr

        dU_dt = Du * lap_U - UVV + F * (1.0 - U_curr)
        dV_dt = Dv * lap_V + UVV - (F + k) * V_curr

        return np.concatenate([dU_dt.flatten(), dV_dt.flatten()])

    # Time integration
    print("  - Simulating pattern formation (400 steps)...")
    state0 = np.concatenate([U.flatten(), V.flatten()])

    frames = []
    for step in range(400):
        t = step * dt
        state_next = integrators.rk4(t, state0, gray_scott, dt)
        state0 = state_next

        if step % 50 == 0:
            # Extract V concentration (shows pattern best)
            n = len(state_next) // 2
            V_curr = state_next[n:].reshape(size, size)
            frames.append(V_curr.copy())

    # Visualize
    print("  - Creating pattern visualizations...")
    pal_plasma = palette.plasma(resolution=256)

    images = []
    for i, frame in enumerate(frames):
        # Normalize V concentration
        normalized = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        img = image.from_field(normalized, pal_plasma)
        images.append(img)
        print(f"    Frame {i}: t={i*50*dt:.1f}, V_max={frame.max():.4f}")

    print(f"  ✓ Generated {len(images)} reaction-diffusion frames")
    print()
    return images


def coupled_oscillators_field():
    """Demo 4: Field of coupled harmonic oscillators."""
    print("Demo 4: Coupled Oscillator Network")
    print("-" * 60)

    # Create grid of oscillators
    grid_size = 64
    print(f"  - Initializing {grid_size}x{grid_size} oscillator grid...")

    # Position and velocity for each oscillator
    x = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.1
    v = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Oscillator parameters
    omega = 2.0  # Natural frequency
    coupling = 0.5  # Coupling strength
    dt = 0.05

    def coupled_oscillators(t, state):
        """Coupled oscillator network."""
        n = len(state) // 2
        x_flat = state[:n]
        v_flat = state[n:]

        x_curr = x_flat.reshape(grid_size, grid_size)
        v_curr = v_flat.reshape(grid_size, grid_size)

        # Compute Laplacian for coupling (spatial diffusion of displacement)
        x_field = Field2D(x_curr.reshape(grid_size, grid_size, 1), dx=1.0, dy=1.0)
        lap_x = field.laplacian(x_field).data.squeeze()

        # Oscillator dynamics with coupling
        dv_dt = -omega**2 * x_curr + coupling * lap_x
        dx_dt = v_curr

        return np.concatenate([dx_dt.flatten(), dv_dt.flatten()])

    # Time integration
    print("  - Simulating coupled oscillations (300 steps)...")
    state0 = np.concatenate([x.flatten(), v.flatten()])

    frames = []
    for step in range(300):
        t = step * dt
        state_next = integrators.rk4(t, state0, coupled_oscillators, dt)
        state0 = state_next

        if step % 10 == 0:
            n = len(state_next) // 2
            x_curr = state_next[:n].reshape(grid_size, grid_size)
            frames.append(x_curr.copy())

    # Visualize
    print("  - Creating oscillation visualizations...")
    pal_cool = palette.from_gradient([
        (0.0, (0.0, 0.0, 0.5)),
        (0.5, (0.5, 0.5, 0.5)),
        (1.0, (1.0, 0.5, 0.0))
    ], resolution=256, name="cool_warm")

    images = []
    for i, frame in enumerate(frames):
        # Normalize
        vmax = max(abs(frame.min()), abs(frame.max()))
        if vmax > 0:
            normalized = (frame + vmax) / (2 * vmax)
        else:
            normalized = np.ones_like(frame) * 0.5

        img = image.from_field(normalized, pal_cool)
        images.append(img)
        if i % 5 == 0:
            print(f"    Frame {i}: t={i*10*dt:.2f}")

    print(f"  ✓ Generated {len(images)} oscillator frames")
    print()
    return images


def nbody_visualization():
    """Demo 5: N-body gravitational system visualization."""
    print("Demo 5: N-Body Gravitational System")
    print("-" * 60)

    # Initialize particles
    n_particles = 100
    print(f"  - Initializing {n_particles} particles...")

    # Positions and velocities (2D)
    positions = np.random.randn(n_particles, 2).astype(np.float32) * 10
    velocities = np.random.randn(n_particles, 2).astype(np.float32) * 0.5
    masses = np.random.uniform(0.5, 2.0, n_particles).astype(np.float32)

    # Gravitational constant
    G = 1.0
    dt = 0.01
    softening = 0.1  # Softening to prevent singularities

    def nbody_gravity(t, state):
        """N-body gravitational dynamics."""
        # State: [x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ...]
        n = len(state) // 4
        pos = state[:2*n].reshape(n, 2)
        vel = state[2*n:].reshape(n, 2)

        # Compute accelerations
        acc = np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = pos[j] - pos[i]
                    r_mag = np.sqrt(np.sum(r**2) + softening**2)
                    acc[i] += G * masses[j] * r / (r_mag**3)

        return np.concatenate([vel.flatten(), acc.flatten()])

    # Time integration
    print("  - Simulating gravitational dynamics (500 steps)...")
    state0 = np.concatenate([positions.flatten(), velocities.flatten()])

    trajectories = []
    for step in range(500):
        t = step * dt
        state_next = integrators.rk4(t, state0, nbody_gravity, dt)
        state0 = state_next

        if step % 25 == 0:
            n = len(state_next) // 4
            pos = state_next[:2*n].reshape(n, 2)
            trajectories.append(pos.copy())

    # Create density field visualization
    print("  - Creating density field visualization...")
    size = 256
    frames = []

    for i, pos in enumerate(trajectories):
        # Create density field
        density = np.zeros((size, size), dtype=np.float32)

        # Map positions to grid
        pos_normalized = pos / 30.0  # Normalize to grid
        pos_grid = ((pos_normalized + 1.0) * size / 2).astype(int)
        pos_grid = np.clip(pos_grid, 0, size - 1)

        # Add Gaussians at particle positions
        for j in range(len(pos_grid)):
            x, y = pos_grid[j]
            y_idx, x_idx = np.ogrid[:size, :size]
            dist = np.sqrt((x_idx - x)**2 + (y_idx - y)**2)
            density += masses[j] * np.exp(-dist**2 / (2 * 5**2))

        frames.append(density)

    # Visualize
    print("  - Rendering frames...")
    pal_magma = palette.magma(resolution=256)

    images = []
    for i, frame in enumerate(frames):
        # Normalize
        normalized = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        img = image.from_field(normalized, pal_magma)
        images.append(img)
        print(f"    Frame {i}: t={i*25*dt:.2f}")

    print(f"  ✓ Generated {len(images)} n-body frames")
    print()
    return images


def main():
    """Run all physics visualization demos."""
    print("=" * 60)
    print("MORPHOGEN PHYSICS VISUALIZER - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Demonstrating integration of:")
    print("  • Integrators (time evolution)")
    print("  • Field operations (spatial computations)")
    print("  • Palette (physical quantity visualization)")
    print("  • Image (rendering & animation)")
    print()
    print("=" * 60)
    print()

    # Run all demos
    heat_diffusion_with_sources()
    wave_interference_pattern()
    reaction_diffusion_simulation()
    coupled_oscillators_field()
    nbody_visualization()

    print("=" * 60)
    print("PHYSICS VISUALIZER COMPLETED!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Heat diffusion equation")
    print("  ✓ Wave equation with interference")
    print("  ✓ Gray-Scott reaction-diffusion")
    print("  ✓ Coupled oscillator networks")
    print("  ✓ N-body gravitational dynamics")
    print()
    print("Cross-Domain Integration:")
    print("  • Integrators provide time evolution (RK4)")
    print("  • Field ops compute spatial derivatives (Laplacian)")
    print("  • Palette maps physical quantities to colors")
    print("  • Image creates visual output")
    print()
    print("Physical Systems Simulated:")
    print("  • Parabolic PDEs (heat equation)")
    print("  • Hyperbolic PDEs (wave equation)")
    print("  • Coupled PDEs (reaction-diffusion)")
    print("  • Particle systems (N-body)")
    print()


if __name__ == "__main__":
    main()
