#!/usr/bin/env python3
"""3D Terrain Visualization Example

Demonstrates Phase 1 of Morphogen's 3D visualization capabilities:
- Creating terrain mesh from heightmap field
- Camera configuration and orbit animation
- Rendering to image and video

This example creates a procedural terrain using noise-based heightmaps
and renders it with various camera angles.

Usage:
    python 01_terrain_mesh.py

Requirements:
    pip install morphogen[visual3d]
"""

import numpy as np
from pathlib import Path

# Import Morphogen stdlib
from morphogen.stdlib import field, visual, visual3d


def generate_terrain_heightmap(size: int = 128, seed: int = 42) -> np.ndarray:
    """Generate a procedural terrain heightmap using multiple noise octaves.

    Args:
        size: Heightmap resolution (size x size)
        seed: Random seed for reproducibility

    Returns:
        2D numpy array with height values in [0, 1]
    """
    np.random.seed(seed)

    # Start with base noise
    heightmap = np.zeros((size, size))

    # Add multiple octaves of noise for realistic terrain
    for octave in range(5):
        freq = 2 ** octave
        amplitude = 1.0 / (2 ** octave)

        # Generate noise at this frequency
        noise_size = max(2, size // freq)
        noise = np.random.rand(noise_size, noise_size)

        # Upsample to full resolution
        from scipy.ndimage import zoom
        if noise_size < size:
            scale = size / noise_size
            noise = zoom(noise, scale, order=1)
            # Ensure exact size match
            noise = noise[:size, :size]

        heightmap += noise * amplitude

    # Normalize to [0, 1]
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    return heightmap


def main():
    """Main example demonstrating 3D terrain visualization."""

    print("Morphogen 3D Terrain Visualization Example")
    print("=" * 50)

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # ==========================================================================
    # 1. Generate Terrain Heightmap
    # ==========================================================================
    print("\n1. Generating terrain heightmap...")

    size = 128
    heightmap_data = generate_terrain_heightmap(size=size, seed=42)

    # Create Field2D from heightmap
    heightmap = field.alloc((size, size))
    heightmap.data[:] = heightmap_data
    print(f"   Created {size}x{size} heightmap")

    # ==========================================================================
    # 2. Create 3D Mesh from Heightmap
    # ==========================================================================
    print("\n2. Creating 3D terrain mesh...")

    terrain_mesh = visual3d.mesh_from_field(
        heightmap,
        scale_x=1.0,
        scale_y=1.0,
        scale_z=20.0,  # Exaggerate height for drama
        colormap="terrain",
        color_by_height=True
    )

    print(f"   Created mesh with {terrain_mesh.n_vertices} vertices, {terrain_mesh.n_faces} faces")

    # ==========================================================================
    # 3. Configure Camera
    # ==========================================================================
    print("\n3. Configuring camera...")

    # Center of terrain
    center_x = size / 2
    center_y = size / 2
    center_z = 10.0

    # Camera positioned to view terrain at an angle
    camera = visual3d.camera(
        position=(size * 1.5, size * 1.5, size * 0.8),
        focal_point=(center_x, center_y, center_z),
        up_vector=(0.0, 0.0, 1.0),
        fov=35.0
    )
    print(f"   Camera: position={camera.position}, focal={camera.focal_point}")

    # ==========================================================================
    # 4. Render Static Image
    # ==========================================================================
    print("\n4. Rendering static image...")

    image = visual3d.render_3d(
        terrain_mesh,
        camera=camera,
        background=(0.1, 0.1, 0.15),
        width=1280,
        height=720,
        anti_aliasing=True
    )

    # Save using visual.output
    output_path = output_dir / "terrain_static.png"
    visual.output(image, str(output_path))
    print(f"   Saved: {output_path}")

    # ==========================================================================
    # 5. Create Orbital Camera Animation
    # ==========================================================================
    print("\n5. Creating orbital camera animation...")

    orbit_cameras = visual3d.orbit_camera(
        center=(center_x, center_y, center_z),
        radius=size * 1.2,
        elevation=30.0,
        frames=60,  # 2 seconds at 30fps
        orbits=1.0,
        start_azimuth=45.0
    )
    print(f"   Generated {len(orbit_cameras)} camera positions for orbit")

    # ==========================================================================
    # 6. Render Animation to Video
    # ==========================================================================
    print("\n6. Rendering orbit video...")

    video_path = output_dir / "terrain_orbit.mp4"
    visual3d.video_3d(
        terrain_mesh,
        camera_path=orbit_cameras,
        path=str(video_path),
        fps=30,
        width=1280,
        height=720,
        background=(0.1, 0.1, 0.15)
    )
    print(f"   Saved: {video_path}")

    # ==========================================================================
    # 7. Multiple Views Gallery
    # ==========================================================================
    print("\n7. Rendering multiple view angles...")

    views = [
        ("top_down", (center_x, center_y, size * 2), (center_x, center_y, 0)),
        ("side_view", (center_x - size, center_y, center_z), (center_x, center_y, center_z)),
        ("corner_low", (0, 0, 5), (center_x, center_y, center_z)),
        ("corner_high", (size * 1.5, 0, size), (center_x, center_y, center_z)),
    ]

    for name, position, focal in views:
        cam = visual3d.camera(position=position, focal_point=focal)
        img = visual3d.render_3d(
            terrain_mesh,
            camera=cam,
            background=(0.1, 0.1, 0.15),
            width=800,
            height=600
        )
        view_path = output_dir / f"terrain_{name}.png"
        visual.output(img, str(view_path))
        print(f"   Saved: {view_path}")

    # ==========================================================================
    # 8. Demonstrate Primitive Shapes
    # ==========================================================================
    print("\n8. Demonstrating primitive shapes...")

    # Create a sphere
    sphere = visual3d.sphere(
        center=(center_x, center_y, 25.0),
        radius=5.0,
        color=(0.8, 0.2, 0.2)  # Red
    )

    # Render terrain with sphere
    combined_img = visual3d.render_3d(
        terrain_mesh,
        sphere,
        camera=camera,
        background=(0.1, 0.1, 0.15),
        width=1280,
        height=720
    )
    combined_path = output_dir / "terrain_with_sphere.png"
    visual.output(combined_img, str(combined_path))
    print(f"   Saved: {combined_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 50)
    print("3D Terrain Visualization Complete!")
    print(f"Output files saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("terrain_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
