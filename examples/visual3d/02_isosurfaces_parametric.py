#!/usr/bin/env python3
"""Isosurfaces and Parametric Surfaces Example

Demonstrates Phase 2 of Morphogen's 3D visualization capabilities:
- Isosurface extraction from 3D scalar volumes (marching cubes)
- Parametric surface generation from f(u,v) → (x,y,z)
- Custom lighting configuration
- Multiple surface compositions

This example creates mathematical surfaces like spheres, tori, and
implicit surfaces extracted from 3D scalar fields.

Usage:
    python 02_isosurfaces_parametric.py

Requirements:
    pip install morphogen[visual3d]
"""

import numpy as np
from pathlib import Path

from morphogen.stdlib import visual, visual3d


# =============================================================================
# Output Setup
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Parametric Surface Definitions
# =============================================================================

def sphere(u, v, radius=1.0):
    """Parametric sphere: maps (u,v) in [0,1]² to unit sphere."""
    theta = u * 2 * np.pi  # azimuth
    phi = v * np.pi        # elevation (0 to pi)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z


def torus(u, v, R=2.0, r=0.5):
    """Parametric torus with major radius R and minor radius r."""
    theta = u * 2 * np.pi
    phi = v * 2 * np.pi
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def klein_bottle(u, v):
    """Klein bottle - a non-orientable surface (figure-8 immersion)."""
    theta = u * 2 * np.pi
    phi = v * 2 * np.pi

    r = 4 * (1 - np.cos(theta) / 2)

    if hasattr(theta, '__iter__'):
        # Vectorized version
        x = np.where(
            theta < np.pi,
            6 * np.cos(theta) * (1 + np.sin(theta)) + r * np.cos(theta) * np.cos(phi),
            6 * np.cos(theta) * (1 + np.sin(theta)) + r * np.cos(phi + np.pi)
        )
        y = np.where(
            theta < np.pi,
            16 * np.sin(theta) + r * np.sin(theta) * np.cos(phi),
            16 * np.sin(theta)
        )
    else:
        if theta < np.pi:
            x = 6 * np.cos(theta) * (1 + np.sin(theta)) + r * np.cos(theta) * np.cos(phi)
            y = 16 * np.sin(theta) + r * np.sin(theta) * np.cos(phi)
        else:
            x = 6 * np.cos(theta) * (1 + np.sin(theta)) + r * np.cos(phi + np.pi)
            y = 16 * np.sin(theta)

    z = r * np.sin(phi)
    return x, y, z


def mobius_strip(u, v, R=1.0, w=0.5):
    """Mobius strip - a non-orientable surface with single boundary."""
    theta = u * 2 * np.pi
    s = v - 0.5  # Center the strip width

    x = (R + s * w * np.cos(theta / 2)) * np.cos(theta)
    y = (R + s * w * np.cos(theta / 2)) * np.sin(theta)
    z = s * w * np.sin(theta / 2)
    return x, y, z


def seashell(u, v, a=1.0, b=0.2, c=0.1, n=2):
    """Seashell (conical spiral) surface."""
    theta = u * 4 * np.pi  # Multiple turns
    phi = v * 2 * np.pi

    r = a * np.exp(b * theta)
    x = r * np.cos(theta) * (1 + c * np.cos(phi))
    y = r * np.sin(theta) * (1 + c * np.cos(phi))
    z = r * (b * theta + c * np.sin(phi))
    return x, y, z


# =============================================================================
# Examples
# =============================================================================

def example_parametric_gallery():
    """Create a gallery of parametric surfaces."""
    print("Creating parametric surface gallery...")

    # 1. Torus
    torus_mesh = visual3d.parametric_surface(
        torus,
        u_resolution=64,
        v_resolution=32,
        colormap="plasma",
        color_by="u"
    )

    cam = visual3d.camera((6, 6, 4), (0, 0, 0))
    img = visual3d.render_3d(torus_mesh, camera=cam, width=800, height=600)
    visual.output(img, str(OUTPUT_DIR / "parametric_torus.png"))
    print(f"  Saved: {OUTPUT_DIR / 'parametric_torus.png'}")

    # 2. Mobius Strip
    mobius_mesh = visual3d.parametric_surface(
        mobius_strip,
        u_resolution=128,
        v_resolution=16,
        colormap="coolwarm",
        color_by="u"
    )

    cam = visual3d.camera((3, 3, 2), (0, 0, 0))
    img = visual3d.render_3d(mobius_mesh, camera=cam, width=800, height=600)
    visual.output(img, str(OUTPUT_DIR / "parametric_mobius.png"))
    print(f"  Saved: {OUTPUT_DIR / 'parametric_mobius.png'}")

    # 3. Seashell
    shell_mesh = visual3d.parametric_surface(
        seashell,
        u_resolution=128,
        v_resolution=32,
        colormap="YlOrBr",
        color_by="z"
    )

    cam = visual3d.camera((8, 8, 6), (2, 2, 3))
    img = visual3d.render_3d(shell_mesh, camera=cam, width=800, height=600)
    visual.output(img, str(OUTPUT_DIR / "parametric_seashell.png"))
    print(f"  Saved: {OUTPUT_DIR / 'parametric_seashell.png'}")


def example_isosurface_sphere():
    """Extract a sphere isosurface from a distance field."""
    print("Creating sphere isosurface from SDF...")

    # Create signed distance field for sphere
    # Values < 0 are inside, = 0 is surface, > 0 is outside
    x, y, z = np.mgrid[-2:2:64j, -2:2:64j, -2:2:64j]
    sdf = np.sqrt(x**2 + y**2 + z**2) - 1.5  # Sphere at origin, radius 1.5

    # Extract isosurface at distance = 0 (the surface)
    sphere_mesh = visual3d.isosurface(
        sdf,
        isovalue=0.0,
        spacing=(4/64, 4/64, 4/64),  # Match grid spacing
        color=(0.3, 0.6, 1.0),  # Blue
        opacity=1.0
    )

    print(f"  Extracted mesh: {sphere_mesh.n_vertices} vertices, {sphere_mesh.n_faces} faces")

    # Render with custom lighting
    lights = [
        visual3d.light((5, 5, 5), color=(1.0, 0.95, 0.9), intensity=1.0, light_type="point"),
        visual3d.light(color=(0.2, 0.2, 0.3), intensity=0.4, light_type="ambient"),
    ]

    cam = visual3d.camera((4, 4, 3), (0, 0, 0))
    img = visual3d.render_3d(
        sphere_mesh,
        camera=cam,
        lights=lights,
        background=(0.05, 0.05, 0.1),
        width=800,
        height=600
    )

    visual.output(img, str(OUTPUT_DIR / "isosurface_sphere.png"))
    print(f"  Saved: {OUTPUT_DIR / 'isosurface_sphere.png'}")


def example_isosurface_gyroid():
    """Extract a gyroid minimal surface (TPMS)."""
    print("Creating gyroid isosurface...")

    # Gyroid implicit surface: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    resolution = 64
    x, y, z = np.mgrid[0:2*np.pi:complex(0, resolution),
                       0:2*np.pi:complex(0, resolution),
                       0:2*np.pi:complex(0, resolution)]

    gyroid = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)

    # Extract isosurface
    mesh = visual3d.isosurface(
        gyroid,
        isovalue=0.0,
        spacing=(2*np.pi/resolution,) * 3,
        origin=(0, 0, 0),  # Grid starts at origin
        colormap="viridis",
        color_by_value=True
    )

    print(f"  Extracted mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

    cam = visual3d.camera((12, 12, 8), (np.pi, np.pi, np.pi))
    img = visual3d.render_3d(
        mesh,
        camera=cam,
        background=(0.02, 0.02, 0.05),
        width=800,
        height=600
    )

    visual.output(img, str(OUTPUT_DIR / "isosurface_gyroid.png"))
    print(f"  Saved: {OUTPUT_DIR / 'isosurface_gyroid.png'}")


def example_isosurface_metaballs():
    """Create metaballs (blobby surfaces) using implicit functions."""
    print("Creating metaballs isosurface...")

    # Define metaball centers and radii
    balls = [
        ((0, 0, 0), 1.0),
        ((1.2, 0, 0), 0.8),
        ((0.5, 1.0, 0.3), 0.7),
        ((-0.8, 0.5, 0.5), 0.6),
    ]

    # Create grid
    resolution = 64
    x, y, z = np.mgrid[-2:2:complex(0, resolution),
                       -2:2:complex(0, resolution),
                       -2:2:complex(0, resolution)]

    # Compute metaball field (sum of 1/distance² from each ball)
    field = np.zeros_like(x)
    for (cx, cy, cz), radius in balls:
        dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 0.001)
        field += radius**2 / dist_sq

    # Extract isosurface where field = 1.0 (threshold)
    mesh = visual3d.isosurface(
        field,
        isovalue=1.0,
        spacing=(4/resolution,) * 3,
        origin=(-2, -2, -2),  # Match the grid origin
        colormap="magma",
        color_by_value=True
    )

    print(f"  Extracted mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

    # Warm lighting for organic look
    lights = [
        visual3d.light((3, 3, 3), color=(1.0, 0.9, 0.7), intensity=1.2, light_type="point"),
        visual3d.light((-2, 1, 2), color=(0.3, 0.4, 0.6), intensity=0.5, light_type="point"),
    ]

    cam = visual3d.camera((4, 4, 3), (0, 0, 0))
    img = visual3d.render_3d(
        mesh,
        camera=cam,
        lights=lights,
        background=(0.02, 0.02, 0.03),
        width=800,
        height=600
    )

    visual.output(img, str(OUTPUT_DIR / "isosurface_metaballs.png"))
    print(f"  Saved: {OUTPUT_DIR / 'isosurface_metaballs.png'}")


def example_combined_scene():
    """Combine parametric and isosurface meshes in one scene."""
    print("Creating combined scene...")

    # Parametric torus (outer ring)
    torus_mesh = visual3d.parametric_surface(
        lambda u, v: torus(u, v, R=3.0, r=0.3),
        u_resolution=64,
        v_resolution=24,
        color=(0.8, 0.6, 0.2),  # Gold
    )

    # Isosurface sphere (center)
    x, y, z = np.mgrid[-2:2:48j, -2:2:48j, -2:2:48j]
    sdf = np.sqrt(x**2 + y**2 + z**2)

    sphere_mesh = visual3d.isosurface(
        sdf,
        isovalue=1.2,
        spacing=(4/48, 4/48, 4/48),
        color=(0.2, 0.5, 0.9),  # Blue
    )

    # Studio lighting
    lights = [
        visual3d.light((10, 10, 8), color=(1.0, 1.0, 0.95), intensity=1.0, light_type="point"),
        visual3d.light((-5, 5, 3), color=(0.4, 0.5, 0.7), intensity=0.6, light_type="point"),
        visual3d.light(color=(0.15, 0.15, 0.2), intensity=0.3, light_type="ambient"),
    ]

    cam = visual3d.camera((6, 6, 4), (0, 0, 0))
    img = visual3d.render_3d(
        torus_mesh,
        sphere_mesh,
        camera=cam,
        lights=lights,
        background=(0.08, 0.08, 0.12),
        width=1200,
        height=800
    )

    visual.output(img, str(OUTPUT_DIR / "combined_torus_sphere.png"))
    print(f"  Saved: {OUTPUT_DIR / 'combined_torus_sphere.png'}")

    # Orbit video
    print("  Rendering orbit animation...")
    cam_path = visual3d.orbit_camera(
        center=(0, 0, 0),
        radius=8,
        elevation=25,
        frames=60,
        orbits=1.0
    )

    visual3d.video_3d(
        torus_mesh,
        sphere_mesh,
        camera_path=cam_path,
        path=str(OUTPUT_DIR / "combined_orbit.mp4"),
        fps=30,
        width=800,
        height=600,
        background=(0.08, 0.08, 0.12)
    )
    print(f"  Saved: {OUTPUT_DIR / 'combined_orbit.mp4'}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Morphogen 3D Visualization - Phase 2 Examples")
    print("Isosurfaces and Parametric Surfaces")
    print("=" * 60)
    print()

    example_parametric_gallery()
    print()

    example_isosurface_sphere()
    print()

    example_isosurface_gyroid()
    print()

    example_isosurface_metaballs()
    print()

    example_combined_scene()
    print()

    print("=" * 60)
    print("All examples complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
