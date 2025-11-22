"""Test proper physics simulation patterns like the portfolio examples."""

import numpy as np
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual

print("=" * 60)
print("Testing Kairo with Proper Physics Patterns")
print("=" * 60)
print()

# Test 1: Heat diffusion from hot spot (like 01_hello_heat.kairo)
print("Test 1: Heat diffusion from center hot spot...")
temp = field.alloc((128, 128), fill_value=0.0)

# Create hot spot in center (like the example)
cx, cy = 64, 64
radius = 10.0

for y in range(128):
    for x in range(128):
        dx = x - cx
        dy = y - cy
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < radius:
            temp.data[y, x] = 100.0

print(f"  Initial hot spot created")
print(f"  Hot pixels: {np.sum(temp.data > 0)}")

# Simulate diffusion for many steps (like the example: 200 steps)
for step in range(50):  # Just 50 for testing
    temp = field.diffuse(temp, rate=0.1, dt=0.1, iterations=10)

print(f"  After 50 diffusion steps:")
print(f"  Min: {np.min(temp.data):.3f}, Max: {np.max(temp.data):.3f}")
print(f"  Center value: {temp.data[cy, cx]:.3f}")

# Visualize
vis = visual.colorize(temp, palette="fire", vmin=0.0, vmax=100.0)
visual.output(vis, path="test_heat_diffusion.png")
print("  Saved: test_heat_diffusion.png")

# Test 2: Wave ripples (like 03_wave_ripples.kairo)
print("\nTest 2: Wave ripples simulation...")

u = field.alloc((128, 128), fill_value=0.0)
v = field.alloc((128, 128), fill_value=0.0)

# Initialize with Gaussian bump
cx, cy = 64, 64
sigma = 5.0

for y in range(128):
    for x in range(128):
        dx = x - cx
        dy = y - cy
        dist_sq = dx*dx + dy*dy
        u.data[y, x] = 1.0 * np.exp(-dist_sq / (2.0 * sigma * sigma))

print(f"  Initial Gaussian bump created")

# Simulate wave equation
c_squared = 0.5 * 0.5
damping = 0.995

for step in range(50):
    lap = field.laplacian(u)
    v.data = v.data + lap.data * c_squared * 0.1
    u.data = u.data + v.data * 0.1
    v.data = v.data * damping

print(f"  After 50 wave steps:")
print(f"  Min: {np.min(u.data):.3f}, Max: {np.max(u.data):.3f}")

# Visualize
vis_wave = visual.colorize(u, palette="coolwarm", vmin=-1.0, vmax=1.0)
visual.output(vis_wave, path="test_wave_ripples.png")
print("  Saved: test_wave_ripples.png")

# Test 3: Gray-Scott reaction-diffusion (like 11_gray_scott.kairo)
print("\nTest 3: Gray-Scott reaction-diffusion patterns...")

u_gs = field.alloc((128, 128), fill_value=1.0)
v_gs = field.alloc((128, 128), fill_value=0.0)

# Initialize with perturbation in center
cx, cy = 64, 64
radius = 10.0

for y in range(128):
    for x in range(128):
        dx = x - cx
        dy = y - cy
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < radius:
            u_gs.data[y, x] = 0.5
            v_gs.data[y, x] = 0.25

print(f"  Initial perturbation created")

# Gray-Scott parameters
Du, Dv = 0.16, 0.08
F, K = 0.060, 0.062

# Simulate (100 steps for testing, examples use 10000)
for step in range(100):
    uvv = u_gs.data * v_gs.data * v_gs.data
    du_dt = Du * field.laplacian(u_gs).data - uvv + F * (1.0 - u_gs.data)
    dv_dt = Dv * field.laplacian(v_gs).data + uvv - (F + K) * v_gs.data

    u_gs.data = u_gs.data + du_dt * 1.0
    v_gs.data = v_gs.data + dv_dt * 1.0

print(f"  After 100 reaction steps:")
print(f"  V field - Min: {np.min(v_gs.data):.3f}, Max: {np.max(v_gs.data):.3f}")

# Visualize
vis_gs = visual.colorize(v_gs, palette="viridis", vmin=0.0, vmax=1.0)
visual.output(vis_gs, path="test_gray_scott.png")
print("  Saved: test_gray_scott.png")

print("\n" + "=" * 60)
print("All tests complete! Check the generated PNGs.")
print("=" * 60)
