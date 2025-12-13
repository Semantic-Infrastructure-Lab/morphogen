"""Comprehensive demo of all new procedural graphics domains.

This example demonstrates:
- NoiseDomain: Generate Perlin noise, FBM, marble patterns
- PaletteDomain: Create gradients, scientific colormaps
- ColorDomain: Manipulate colors, blend modes
- ImageDomain: Create images from fields, apply effects
- FieldDomain: Compute gradients, smooth, normalize

Generates several procedural images showing the power of these domains.
"""

import numpy as np
from morphogen.stdlib import noise, palette, color, image, field
from morphogen.stdlib.field import Field2D


def demo_basic_noise_with_palette():
    """Demo 1: Basic Perlin noise with colormap."""
    print("Demo 1: Basic Perlin noise with palette")
    print("-" * 50)

    # Generate Perlin noise
    print("  - Generating Perlin noise (256x256)...")
    perlin = noise.perlin2d((256, 256), scale=0.02, octaves=4, seed=42)

    # Create a palette
    print("  - Creating Viridis palette...")
    pal = palette.viridis(resolution=256)

    # Map noise to colors
    print("  - Mapping noise to colors...")
    img = image.from_field(perlin.data, pal)

    print(f"  ✓ Generated image: {img}")
    print()


def demo_fractal_brownian_motion():
    """Demo 2: Fractal Brownian Motion with custom gradient."""
    print("Demo 2: Fractal Brownian Motion (fBm)")
    print("-" * 50)

    # Generate fBm with multiple octaves
    print("  - Generating fBm (6 octaves)...")
    fbm_noise = noise.fbm((512, 512), scale=0.01, octaves=6,
                          persistence=0.5, lacunarity=2.0, seed=123)

    # Create custom gradient
    print("  - Creating custom fire palette...")
    fire_pal = palette.fire(resolution=256)

    # Normalize the field
    print("  - Normalizing field to [0, 1]...")
    normalized = field.normalize(Field2D(fbm_noise.data), 0.0, 1.0)

    # Map to colors
    img = image.from_field(normalized.data, fire_pal)

    print(f"  ✓ Generated fBm image: {img}")
    print()


def demo_marble_with_processing():
    """Demo 3: Marble pattern with image processing."""
    print("Demo 3: Marble pattern with post-processing")
    print("-" * 50)

    # Generate marble pattern
    print("  - Generating marble noise...")
    marble_noise = noise.marble((256, 256), scale=0.05,
                                turbulence_power=3.0, seed=456)

    # Create palette from gradient
    print("  - Creating gradient palette...")
    gradient_pal = palette.from_gradient([
        (0.0, (0.1, 0.1, 0.1)),     # Dark grey
        (0.3, (0.4, 0.4, 0.45)),    # Medium grey
        (0.7, (0.8, 0.8, 0.85)),    # Light grey
        (1.0, (1.0, 1.0, 1.0))      # White
    ], resolution=256, name="marble_gradient")

    # Create base image
    img = image.from_field(marble_noise.data, gradient_pal)

    # Apply blur for smoothness
    print("  - Applying Gaussian blur...")
    blurred = image.blur(img, sigma=1.5)

    # Sharpen slightly
    print("  - Sharpening...")
    sharpened = image.sharpen(blurred, strength=0.5)

    print(f"  ✓ Generated marble image: {sharpened}")
    print()


def demo_procedural_terrain():
    """Demo 4: Procedural terrain with normal map."""
    print("Demo 4: Procedural terrain with normal map")
    print("-" * 50)

    # Generate terrain using ridged fBm
    print("  - Generating ridged multifractal terrain...")
    terrain = noise.ridged_fbm((256, 256), scale=0.02, octaves=6,
                               persistence=0.6, lacunarity=2.2, seed=789)

    # Create terrain palette (elevation-based)
    print("  - Creating terrain palette...")
    terrain_pal = palette.from_gradient([
        (0.0, (0.0, 0.2, 0.5)),     # Deep water
        (0.3, (0.0, 0.5, 0.8)),     # Shallow water
        (0.4, (0.8, 0.7, 0.4)),     # Beach
        (0.6, (0.2, 0.6, 0.2)),     # Grass
        (0.8, (0.5, 0.4, 0.3)),     # Mountain
        (1.0, (1.0, 1.0, 1.0))      # Snow
    ], resolution=256, name="terrain")

    # Map terrain to colors
    terrain_img = image.from_field(terrain.data, terrain_pal)

    # Generate normal map from heightfield
    print("  - Generating normal map...")
    normal_map = image.normal_map_from_heightfield(terrain.data, strength=2.0)

    print(f"  ✓ Terrain image: {terrain_img}")
    print(f"  ✓ Normal map: {normal_map}")
    print()


def demo_color_manipulation():
    """Demo 5: Color space conversions and manipulation."""
    print("Demo 5: Color manipulation and blending")
    print("-" * 50)

    # Create base noise
    print("  - Generating base noise...")
    base_noise = noise.perlin2d((128, 128), scale=0.05, octaves=3, seed=111)

    # Create two different colored versions
    print("  - Creating colored versions...")

    # Version 1: Hot colors
    hot_pal = palette.from_gradient([
        (0.0, (0.0, 0.0, 0.0)),
        (0.5, (1.0, 0.5, 0.0)),
        (1.0, (1.0, 1.0, 0.0))
    ], resolution=128)
    img_hot = image.from_field(base_noise.data, hot_pal)

    # Version 2: Cold colors
    cold_pal = palette.from_gradient([
        (0.0, (0.0, 0.0, 0.0)),
        (0.5, (0.0, 0.5, 1.0)),
        (1.0, (0.0, 1.0, 1.0))
    ], resolution=128)
    img_cold = image.from_field(base_noise.data, cold_pal)

    # Blend with different modes
    print("  - Blending with different modes...")
    blended_normal = image.blend(img_hot, img_cold, mode="normal", opacity=0.5)
    blended_screen = image.blend(img_hot, img_cold, mode="screen", opacity=0.7)
    blended_overlay = image.blend(img_hot, img_cold, mode="overlay", opacity=0.6)

    print(f"  ✓ Hot image: {img_hot}")
    print(f"  ✓ Cold image: {img_cold}")
    print(f"  ✓ Blended (normal): {blended_normal}")
    print(f"  ✓ Blended (screen): {blended_screen}")
    print(f"  ✓ Blended (overlay): {blended_overlay}")
    print()


def demo_field_operations():
    """Demo 6: Field operations (gradient, divergence, curl)."""
    print("Demo 6: Field operations and vector fields")
    print("-" * 50)

    # Generate vector field using noise
    print("  - Generating vector field from noise...")
    vx, vy = noise.vector_field((128, 128), scale=0.05, octaves=3, seed=222)

    # Create velocity field
    velocity_data = np.stack([vx.data, vy.data], axis=-1)
    velocity_field = Field2D(velocity_data)

    # Compute field operations
    print("  - Computing divergence...")
    div_field = field.divergence(velocity_field)

    print("  - Computing curl (vorticity)...")
    curl_field = field.curl(velocity_field)

    print("  - Computing magnitude...")
    mag_field = field.magnitude(velocity_field)

    # Visualize using palettes
    print("  - Visualizing field properties...")
    pal_div = palette.from_gradient([
        (0.0, (0.0, 0.0, 1.0)),
        (0.5, (1.0, 1.0, 1.0)),
        (1.0, (1.0, 0.0, 0.0))
    ], resolution=256, name="diverging")

    img_div = image.from_field(div_field.data, pal_div)
    img_curl = image.from_field(curl_field.data, pal_div)
    img_mag = image.from_field(mag_field.data, palette.viridis())

    print(f"  ✓ Divergence field: {div_field}")
    print(f"  ✓ Curl field: {curl_field}")
    print(f"  ✓ Magnitude field: {mag_field}")
    print()


def demo_animated_palette():
    """Demo 7: Animated palette cycling."""
    print("Demo 7: Animated palette cycling")
    print("-" * 50)

    # Generate base noise
    print("  - Generating base noise...")
    base = noise.fbm((256, 256), scale=0.02, octaves=4, seed=333)

    # Create base palette
    print("  - Creating rainbow palette...")
    base_pal = palette.rainbow(resolution=256)

    # Simulate animation frames by cycling palette
    print("  - Generating animation frames...")
    for frame in range(3):
        time = frame * 0.1
        cycled_pal = palette.cycle(base_pal, speed=1.0, time=time)
        img_frame = image.from_field(base.data, cycled_pal)
        print(f"    Frame {frame}: palette shifted by {time:.2f}")

    print("  ✓ Animation simulation complete")
    print()


def demo_cosine_gradient():
    """Demo 8: Procedural IQ-style cosine gradients."""
    print("Demo 8: Procedural cosine gradients (IQ style)")
    print("-" * 50)

    # Generate noise
    print("  - Generating turbulence noise...")
    turbulence = noise.turbulence((256, 256), scale=0.03, octaves=5, seed=444)

    # Create multiple cosine palettes
    print("  - Creating cosine gradient palettes...")

    # Warm sunset
    pal1 = palette.cosine(
        a=(0.5, 0.5, 0.5),
        b=(0.5, 0.5, 0.5),
        c=(1.0, 1.0, 1.0),
        d=(0.0, 0.10, 0.20),
        resolution=256
    )

    # Cool ocean
    pal2 = palette.cosine(
        a=(0.5, 0.5, 0.5),
        b=(0.5, 0.5, 0.5),
        c=(1.0, 1.0, 1.0),
        d=(0.30, 0.20, 0.20),
        resolution=256
    )

    # Psychedelic
    pal3 = palette.cosine(
        a=(0.5, 0.5, 0.5),
        b=(0.5, 0.5, 0.5),
        c=(2.0, 1.0, 0.0),
        d=(0.50, 0.20, 0.25),
        resolution=256
    )

    img1 = image.from_field(turbulence.data, pal1)
    img2 = image.from_field(turbulence.data, pal2)
    img3 = image.from_field(turbulence.data, pal3)

    print(f"  ✓ Sunset palette: {img1}")
    print(f"  ✓ Ocean palette: {img2}")
    print(f"  ✓ Psychedelic palette: {img3}")
    print()


def main():
    """Run all demos."""
    print("=" * 50)
    print("MORPHOGEN PROCEDURAL GRAPHICS DOMAINS DEMO")
    print("=" * 50)
    print()

    demo_basic_noise_with_palette()
    demo_fractal_brownian_motion()
    demo_marble_with_processing()
    demo_procedural_terrain()
    demo_color_manipulation()
    demo_field_operations()
    demo_animated_palette()
    demo_cosine_gradient()

    print("=" * 50)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print()
    print("Summary of new domains:")
    print("  ✓ NoiseDomain  - Perlin, Simplex, Worley, fBm, Marble, Plasma")
    print("  ✓ PaletteDomain - Gradients, Scientific colormaps, Cosine gradients")
    print("  ✓ ColorDomain  - RGB/HSV/HSL conversion, Blend modes, Temperature")
    print("  ✓ ImageDomain  - Creation, Transforms, Filters, Compositing")
    print("  ✓ FieldDomain  - Gradient, Divergence, Curl, Smoothing (extended)")
    print()


if __name__ == "__main__":
    main()
