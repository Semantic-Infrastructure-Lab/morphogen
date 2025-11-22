"""Test and generate visual outputs for new examples 04 and 05."""

import numpy as np
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual

print("=" * 60)
print("Testing Examples 04 and 05")
print("=" * 60)
print()

# ============================================================================
# Example 04: Random Walk
# ============================================================================

def generate_random_walk():
    """Generate output for 04_random_walk.kairo."""
    print("Example 04: Random Walk - Brownian Motion Patterns")
    print("-" * 60)

    NUM_WALKERS = 500
    GRID_SIZE = 128

    # Initialize density field
    density = field.alloc((GRID_SIZE, GRID_SIZE), fill_value=0.0)

    # Initialize walker positions at center
    walker_x = np.ones(NUM_WALKERS) * 64.0
    walker_y = np.ones(NUM_WALKERS) * 64.0

    seed = 42
    np.random.seed(seed)

    print(f"  Simulating {NUM_WALKERS} random walkers...")

    # Simulate random walks
    for step in range(2000):
        # Each walker takes a random step
        for i in range(NUM_WALKERS):
            # Random direction (-1, 0, or 1 for each axis)
            rand_x = np.random.randint(-1, 2)
            rand_y = np.random.randint(-1, 2)

            # Update walker position
            walker_x[i] += rand_x
            walker_y[i] += rand_y

            # Boundary conditions (wrap around)
            walker_x[i] = walker_x[i] % GRID_SIZE
            walker_y[i] = walker_y[i] % GRID_SIZE

            # Accumulate density at walker position
            ix = int(walker_x[i]) % GRID_SIZE
            iy = int(walker_y[i]) % GRID_SIZE
            density.data[iy, ix] += 1.0

        # Apply slight diffusion every 10 steps
        if step % 10 == 0:
            density = field.diffuse(density, rate=0.05, dt=1.0, iterations=5)

        # Save output at key frames
        if step in [199, 499, 999, 1499, 1999]:
            # Normalize
            max_density = np.max(density.data)
            normalized_data = density.data / max(max_density, 1.0)
            normalized = Field2D(normalized_data)

            vis = visual.colorize(normalized, palette="viridis", vmin=0.0, vmax=1.0)
            output_path = f"examples/output_04_random_walk_step{step:04d}.png"
            visual.output(vis, path=output_path)

            print(f"    Step {step:4d}: Max density = {max_density:.1f}")

    print("  ✓ Generated output_04_random_walk_step*.png")
    print()


# ============================================================================
# Example 05: Gradient Flow
# ============================================================================

def generate_gradient_flow():
    """Generate output for 05_gradient_flow.kairo."""
    print("Example 05: Gradient Flow - Advection and Mixing")
    print("-" * 60)

    GRID_SIZE = 128
    ROTATION_SPEED = 1.5

    # Initialize color field with diagonal gradient
    color_field = field.alloc((GRID_SIZE, GRID_SIZE), fill_value=0.0)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color_field.data[y, x] = (x + y) / 256.0

    # Create swirling velocity field
    velocity_data = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)

    cx, cy = 64.0, 64.0

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            dx = x - cx
            dy = y - cy

            # Tangential velocity (perpendicular to radius)
            velocity_data[y, x, 0] = -dy * ROTATION_SPEED * 0.1  # vx
            velocity_data[y, x, 1] = dx * ROTATION_SPEED * 0.1   # vy

    velocity = Field2D(velocity_data)

    print(f"  Simulating gradient flow with rotation speed {ROTATION_SPEED}...")

    dt = 0.1

    # Simulate advection
    for step in range(400):
        # Advect the color field
        color_field = field.advect(color_field, velocity, dt)

        # Add slight diffusion for smooth mixing
        if step % 5 == 0:
            color_field = field.diffuse(color_field, rate=0.02, dt=dt, iterations=5)

        # Save output at key frames
        if step in [0, 50, 100, 200, 300, 399]:
            vis = visual.colorize(color_field, palette="coolwarm", vmin=0.0, vmax=1.0)
            output_path = f"examples/output_05_gradient_flow_step{step:03d}.png"
            visual.output(vis, path=output_path)

            print(f"    Step {step:3d}: Min={np.min(color_field.data):.3f}, Max={np.max(color_field.data):.3f}")

    print("  ✓ Generated output_05_gradient_flow_step*.png")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    generate_random_walk()
    generate_gradient_flow()

    print("=" * 60)
    print("✓ All examples generated successfully!")
    print("=" * 60)
