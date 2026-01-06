# 3D Visualization Examples

Examples demonstrating Morphogen's 3D visualization capabilities using PyVista.

## Installation

```bash
pip install morphogen[visual3d]
```

## Examples

### 01_terrain_mesh.py (Phase 1)

Demonstrates core 3D visualization features:
- Creating terrain mesh from 2D heightmap
- Camera configuration and positioning
- Rendering to static images
- Orbital camera animation
- Video export
- Primitive shapes (sphere, box, cylinder)
- Compositing multiple 3D objects

```bash
python 01_terrain_mesh.py
```

Outputs:
- `output/terrain_static.png` - Static terrain view
- `output/terrain_orbit.mp4` - Orbital camera animation
- `output/terrain_*.png` - Multiple camera angles
- `output/terrain_with_sphere.png` - Terrain with sphere overlay

### 02_isosurfaces_parametric.py (Phase 2)

Demonstrates advanced surface generation:
- Isosurface extraction from 3D scalar volumes (marching cubes)
- Parametric surfaces: torus, Mobius strip, seashell
- Custom lighting configuration
- Combined scenes with multiple surface types
- Mathematical surfaces (gyroid, metaballs)

```bash
python 02_isosurfaces_parametric.py
```

Outputs:
- `output/parametric_torus.png` - Parametric torus
- `output/parametric_mobius.png` - Mobius strip
- `output/parametric_seashell.png` - Seashell spiral
- `output/isosurface_sphere.png` - Sphere from SDF
- `output/isosurface_gyroid.png` - Gyroid minimal surface
- `output/isosurface_metaballs.png` - Blobby metaballs
- `output/combined_torus_sphere.png` - Combined scene
- `output/combined_orbit.mp4` - Orbit animation

## API Quick Reference

```python
from morphogen.stdlib import field, visual, visual3d

# Create mesh from heightmap
heightmap = field.random((128, 128))
mesh = visual3d.mesh_from_field(heightmap, scale_z=10.0)

# Configure camera
camera = visual3d.camera(
    position=(100, 100, 50),
    focal_point=(64, 64, 0)
)

# Render to image
image = visual3d.render_3d(mesh, camera=camera)
visual.output(image, "scene.png")

# Create orbit animation
cam_path = visual3d.orbit_camera(
    center=(64, 64, 0),
    radius=100,
    frames=60
)
visual3d.video_3d(mesh, camera_path=cam_path, path="orbit.mp4")
```

## Implemented Features

### Phase 1 - Foundation
- [x] `mesh()` - Create mesh from vertices/faces
- [x] `mesh_from_field()` - Heightmap to 3D terrain
- [x] `camera()` - Camera configuration
- [x] `orbit_camera()` - Orbital camera animation
- [x] `render_3d()` - Scene to image
- [x] `video_3d()` - Animation to video
- [x] `display_3d()` - Interactive display
- [x] `sphere()`, `box()`, `cylinder()` - Primitive shapes

### Phase 2 - Advanced Surfaces
- [x] `isosurface()` - Marching cubes extraction from 3D volumes
- [x] `parametric_surface()` - f(u,v) → (x,y,z) mesh generation
- [x] `light()` - Custom lighting configuration
- [x] `clim` - Scalar range control for coloring

## Coming Soon (Future Phases)

- Phase 3: Molecular visualization (ball-and-stick, orbitals)
- Phase 4: Camera path animation, keyframes
- Phase 5: Streamlines, volume rendering
