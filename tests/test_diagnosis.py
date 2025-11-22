"""Diagnostic test to check if Kairo operations are working correctly."""

import numpy as np
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual

print("=" * 60)
print("Kairo Diagnostic Test")
print("=" * 60)
print()

# Test 1: Simple field with hot spot
print("Test 1: Creating field with center hot spot...")
temp = field.alloc((128, 128), fill_value=0.0)

# Set center to 1.0
cx, cy = 64, 64
temp.data[cy, cx] = 1.0

print(f"  Initial - Center value: {temp.data[cy, cx]:.3f}")
print(f"  Initial - Corner value: {temp.data[0, 0]:.3f}")
print(f"  Initial - Min: {np.min(temp.data):.3f}, Max: {np.max(temp.data):.3f}")
print(f"  Initial - Mean: {np.mean(temp.data):.3f}")

# Apply diffusion
print("\nApplying diffusion (rate=0.5, dt=0.1, 10 iterations)...")
temp_diffused = field.diffuse(temp, rate=0.5, dt=0.1, iterations=10)

print(f"  After diffusion - Center value: {temp_diffused.data[cy, cx]:.3f}")
print(f"  After diffusion - Neighbor value: {temp_diffused.data[cy, cx+1]:.3f}")
print(f"  After diffusion - Corner value: {temp_diffused.data[0, 0]:.3f}")
print(f"  After diffusion - Min: {np.min(temp_diffused.data):.3f}, Max: {np.max(temp_diffused.data):.3f}")
print(f"  After diffusion - Mean: {np.mean(temp_diffused.data):.3f}")

# Visualize
vis = visual.colorize(temp_diffused, palette="fire", vmin=0.0, vmax=1.0)
visual.output(vis, path="diagnostic_hotspot.png")
print("\n  Saved: diagnostic_hotspot.png")

# Test 2: Random field diffusion
print("\nTest 2: Random field with diffusion...")
temp_random = field.random((128, 128), seed=42, low=0.0, high=1.0)

print(f"  Random initial - Min: {np.min(temp_random.data):.3f}, Max: {np.max(temp_random.data):.3f}")
print(f"  Random initial - Mean: {np.mean(temp_random.data):.3f}")
print(f"  Random initial - Std: {np.std(temp_random.data):.3f}")

# Apply strong diffusion
temp_random_diffused = field.diffuse(temp_random, rate=0.5, dt=0.1, iterations=10)

print(f"  After diffusion - Min: {np.min(temp_random_diffused.data):.3f}, Max: {np.max(temp_random_diffused.data):.3f}")
print(f"  After diffusion - Mean: {np.mean(temp_random_diffused.data):.3f}")
print(f"  After diffusion - Std: {np.std(temp_random_diffused.data):.3f}")

# Visualize
vis_random = visual.colorize(temp_random_diffused, palette="fire", vmin=0.0, vmax=1.0)
visual.output(vis_random, path="diagnostic_random_diffused.png")
print("  Saved: diagnostic_random_diffused.png")

# Test 3: Check if data is actually changing
print("\nTest 3: Checking if diffusion changes data...")
test_field = field.alloc((5, 5), fill_value=0.0)
test_field.data[2, 2] = 1.0

print("  Before diffusion:")
print(test_field.data)

test_diffused = field.diffuse(test_field, rate=0.5, dt=0.1, iterations=5)

print("\n  After diffusion:")
print(test_diffused.data)

print("\n" + "=" * 60)
print("Diagnostic test complete!")
print("=" * 60)
