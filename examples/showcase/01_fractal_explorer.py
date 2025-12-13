"""Fractal Explorer - Cross-Domain Showcase Example

This example demonstrates the power of combining multiple Morphogen domains:
- Field operations for fractal computation
- Noise for texture overlay and perturbation
- Palette for sophisticated color mapping
- Color for blend modes and adjustments
- Image for output and composition

Creates stunning visualizations of Mandelbrot and Julia sets with:
- Smooth iteration coloring
- Orbit trap coloring
- Noise-based texture overlays
- Multiple color palettes
- Advanced blend modes
"""

import numpy as np
from morphogen.stdlib import field, noise, palette, color, image
from morphogen.stdlib.field import Field2D


def compute_mandelbrot(width, height, x_min=-2.5, x_max=1.0,
                       y_min=-1.25, y_max=1.25, max_iter=256):
    """Compute the Mandelbrot set with smooth iteration count.

    Returns both escape iterations and smooth coloring data.
    """
    # Create coordinate grids
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Initialize
    Z = np.zeros_like(C)
    iterations = np.zeros(C.shape, dtype=np.float32)

    # Iterate
    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask]**2 + C[mask]
        iterations[mask] = i

    # Smooth coloring: add fractional iteration count
    # Based on distance from escape radius
    escaped = np.abs(Z) > 2.0
    if np.any(escaped):
        smooth = iterations[escaped] + 1 - np.log2(np.log2(np.abs(Z[escaped])))
        iterations[escaped] = smooth

    # Normalize to [0, 1]
    max_val = np.max(iterations)
    if max_val > 0:
        iterations = iterations / max_val

    return iterations


def compute_julia(width, height, c_real=-0.7, c_imag=0.27015,
                  x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, max_iter=256):
    """Compute a Julia set for a given complex parameter c.

    Returns smooth iteration data.
    """
    # Create coordinate grids
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    C = complex(c_real, c_imag)

    iterations = np.zeros(Z.shape, dtype=np.float32)

    # Iterate
    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask]**2 + C
        iterations[mask] = i

    # Smooth coloring
    escaped = np.abs(Z) > 2.0
    if np.any(escaped):
        smooth = iterations[escaped] + 1 - np.log2(np.log2(np.abs(Z[escaped])))
        iterations[escaped] = smooth

    # Normalize
    max_val = np.max(iterations)
    if max_val > 0:
        iterations = iterations / max_val

    return iterations


def mandelbrot_with_noise_overlay():
    """Demo 1: Mandelbrot set with noise texture overlay."""
    print("Demo 1: Mandelbrot Set with Noise Overlay")
    print("-" * 60)

    # Compute Mandelbrot
    print("  - Computing Mandelbrot set (800x600, 512 iterations)...")
    mandel = compute_mandelbrot(800, 600, max_iter=512)

    # Create base visualization with inferno palette
    print("  - Applying Inferno palette...")
    pal_inferno = palette.inferno(resolution=512)
    img_base = image.from_field(mandel, pal_inferno)

    # Generate subtle noise overlay for texture
    print("  - Generating noise texture overlay...")
    # Note: mandel is (height, width) = (600, 800), so noise should match
    texture = noise.perlin2d((600, 800), scale=0.05, octaves=3, seed=42)
    texture_normalized = field.normalize(Field2D(texture.data), 0.0, 1.0)

    # Map noise to grayscale
    gray_pal = palette.from_gradient([
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (1.0, 1.0, 1.0))
    ], resolution=256, name="grayscale")
    img_texture = image.from_field(texture_normalized.data, gray_pal)

    # Blend using overlay mode
    print("  - Blending with overlay mode...")
    result = image.blend(img_base, img_texture, mode="overlay", opacity=0.15)

    print(f"  ✓ Generated Mandelbrot with texture: {result}")
    print()
    return result


def julia_set_multicolor():
    """Demo 2: Julia set with multi-palette composition."""
    print("Demo 2: Julia Set Multi-Palette Composition")
    print("-" * 60)

    # Compute Julia set (creates 600x800 array)
    print("  - Computing Julia set (c = -0.7 + 0.27015i)...")
    julia = compute_julia(800, 600, c_real=-0.7, c_imag=0.27015, max_iter=512)
    # julia is (600, 800)

    # Create multiple colored versions
    print("  - Creating multiple palette variations...")

    # Fire palette version
    pal_fire = palette.fire(resolution=512)
    img_fire = image.from_field(julia, pal_fire)

    # Ocean palette version (custom cosine gradient)
    pal_ocean = palette.cosine(
        a=(0.5, 0.5, 0.5),
        b=(0.5, 0.5, 0.5),
        c=(1.0, 1.0, 1.0),
        d=(0.30, 0.20, 0.20),
        resolution=512
    )
    img_ocean = image.from_field(julia, pal_ocean)

    # Purple-green palette
    pal_purple = palette.from_gradient([
        (0.0, (0.05, 0.0, 0.1)),
        (0.3, (0.5, 0.0, 0.8)),
        (0.6, (0.2, 0.8, 0.3)),
        (1.0, (1.0, 1.0, 0.5))
    ], resolution=512, name="purple_green")
    img_purple = image.from_field(julia, pal_purple)

    # Blend all three with screen mode
    print("  - Compositing with screen blend mode...")
    temp = image.blend(img_fire, img_ocean, mode="screen", opacity=0.5)
    result = image.blend(temp, img_purple, mode="screen", opacity=0.4)

    print(f"  ✓ Generated multi-palette Julia set: {result}")
    print()
    return result


def mandelbrot_deep_zoom():
    """Demo 3: Deep zoom into Mandelbrot with dynamic palette."""
    print("Demo 3: Mandelbrot Deep Zoom with Dynamic Palette")
    print("-" * 60)

    # Zoom coordinates (near a interesting feature)
    print("  - Computing deep zoom (elephant valley)...")
    mandel_zoom = compute_mandelbrot(
        1024, 768,
        x_min=0.275, x_max=0.285,
        y_min=0.005, y_max=0.012,
        max_iter=1024
    )

    # Create custom palette with more color stops
    print("  - Creating rich gradient palette...")
    pal_rich = palette.from_gradient([
        (0.00, (0.0, 0.0, 0.0)),      # Black
        (0.16, (0.1, 0.0, 0.4)),      # Deep purple
        (0.32, (0.4, 0.0, 0.2)),      # Dark red
        (0.48, (0.8, 0.4, 0.0)),      # Orange
        (0.64, (1.0, 0.8, 0.0)),      # Yellow
        (0.80, (0.0, 0.6, 1.0)),      # Cyan
        (1.00, (1.0, 1.0, 1.0))       # White
    ], resolution=1024, name="rich_gradient")

    img_zoomed = image.from_field(mandel_zoom, pal_rich)

    # Apply slight sharpening to enhance details
    print("  - Sharpening details...")
    result = image.sharpen(img_zoomed, strength=0.3)

    print(f"  ✓ Generated deep zoom visualization: {result}")
    print()
    return result


def julia_with_orbit_trap():
    """Demo 4: Julia set with simulated orbit trap coloring."""
    print("Demo 4: Julia Set with Orbit Trap Coloring")
    print("-" * 60)

    # Compute Julia set with different parameters
    print("  - Computing Julia set (c = -0.4 + 0.6i)...")
    julia = compute_julia(800, 600, c_real=-0.4, c_imag=0.6, max_iter=256)

    # Simulate orbit trap by using field operations
    # In real orbit trap, we'd track minimum distance to a shape during iteration
    # Here we approximate with field gradient magnitude
    print("  - Computing field gradient for orbit trap effect...")
    julia_field = Field2D(julia)  # julia is already 2D (600, 800)
    grad_x, grad_y = field.gradient(julia_field)
    # Combine gradients into vector field
    grad_data = np.stack([grad_x.data, grad_y.data], axis=-1)
    grad_field = Field2D(grad_data)
    grad_mag = field.magnitude(grad_field)

    # Normalize gradient magnitude
    grad_normalized = field.normalize(grad_mag, 0.0, 1.0)

    # Combine original Julia with gradient using multiplication
    combined = julia * (1.0 - grad_normalized.data.squeeze() * 0.5)

    # Apply viridis palette
    print("  - Applying Viridis palette...")
    pal_viridis = palette.viridis(resolution=256)
    result = image.from_field(combined, pal_viridis)

    print(f"  ✓ Generated orbit trap Julia set: {result}")
    print()
    return result


def fractal_with_fbm_coloring():
    """Demo 5: Fractal colored by FBM noise for organic look."""
    print("Demo 5: Fractal with FBM-Based Organic Coloring")
    print("-" * 60)

    # Compute Mandelbrot (creates 768x1024 array)
    print("  - Computing Mandelbrot set...")
    mandel = compute_mandelbrot(1024, 768, max_iter=512)

    # Generate FBM noise at same resolution
    print("  - Generating fractal Brownian motion texture...")
    fbm = noise.fbm((768, 1024), scale=0.008, octaves=6,
                    persistence=0.5, lacunarity=2.0, seed=123)
    fbm_normalized = field.normalize(Field2D(fbm.data), 0.0, 1.0)

    # Modulate fractal iterations by FBM
    print("  - Modulating fractal with noise...")
    modulated = mandel * 0.7 + fbm_normalized.data.squeeze() * 0.3
    modulated = np.clip(modulated, 0, 1)

    # Apply custom organic palette
    print("  - Creating organic color palette...")
    pal_organic = palette.from_gradient([
        (0.0, (0.02, 0.02, 0.05)),    # Deep space
        (0.2, (0.1, 0.3, 0.2)),       # Dark teal
        (0.4, (0.3, 0.5, 0.1)),       # Moss
        (0.6, (0.7, 0.6, 0.2)),       # Gold
        (0.8, (0.9, 0.4, 0.3)),       # Coral
        (1.0, (1.0, 0.9, 0.7))        # Cream
    ], resolution=512, name="organic")

    result = image.from_field(modulated, pal_organic)

    print(f"  ✓ Generated organic fractal: {result}")
    print()
    return result


def animated_fractal_palette_cycle():
    """Demo 6: Simulate animated palette cycling."""
    print("Demo 6: Animated Fractal with Palette Cycling")
    print("-" * 60)

    # Compute Julia set
    print("  - Computing Julia set (dendrite fractal)...")
    julia = compute_julia(800, 600, c_real=0.0, c_imag=1.0, max_iter=256)

    # Create rainbow palette
    print("  - Creating rainbow palette...")
    pal_rainbow = palette.rainbow(resolution=256)

    # Simulate animation frames
    print("  - Generating animation frames with palette cycling...")
    frames = []
    for i in range(4):
        time = i * 0.25
        cycled_pal = palette.cycle(pal_rainbow, speed=1.0, time=time)
        frame = image.from_field(julia, cycled_pal)
        frames.append(frame)
        print(f"    Frame {i}: palette phase {time:.2f}")

    print(f"  ✓ Generated {len(frames)} animation frames")
    print()
    return frames


def main():
    """Run all fractal explorer demos."""
    print("=" * 60)
    print("MORPHOGEN FRACTAL EXPLORER - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Demonstrating integration of:")
    print("  • Field operations (fractal computation)")
    print("  • Noise (texture overlays & modulation)")
    print("  • Palette (sophisticated color mapping)")
    print("  • Color (blend modes & adjustments)")
    print("  • Image (composition & filters)")
    print()
    print("=" * 60)
    print()

    # Run all demos
    mandelbrot_with_noise_overlay()
    julia_set_multicolor()
    mandelbrot_deep_zoom()
    julia_with_orbit_trap()
    fractal_with_fbm_coloring()
    animated_fractal_palette_cycle()

    print("=" * 60)
    print("FRACTAL EXPLORER COMPLETED!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Mandelbrot & Julia set computation")
    print("  ✓ Smooth iteration coloring")
    print("  ✓ Noise texture overlays")
    print("  ✓ Multi-palette composition")
    print("  ✓ Advanced blend modes (overlay, screen)")
    print("  ✓ Deep zoom visualization")
    print("  ✓ Orbit trap effects")
    print("  ✓ FBM-based organic coloring")
    print("  ✓ Animated palette cycling")
    print()
    print("Cross-Domain Integration:")
    print("  • Field ops provide gradient computation")
    print("  • Noise adds organic texture & variation")
    print("  • Palette enables rich color mapping")
    print("  • Image supports composition & effects")
    print("  • Color manages blend modes")
    print()


if __name__ == "__main__":
    main()
