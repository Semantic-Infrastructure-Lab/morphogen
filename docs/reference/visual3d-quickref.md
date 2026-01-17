# Visual3D Quick Reference

Quick reference for Morphogen's 3D visualization module (`visual3d.py`).

See [3D_VISUALIZATION_SYSTEM_PLAN.md](../planning/3D_VISUALIZATION_SYSTEM_PLAN.md) for architecture documentation.

---

## Core Types

```python
from morphogen.stdlib.visual3d import Visual3D, Camera3D, Light3D, Scene3D

# Visual3D - 3D mesh representation
Visual3D(
    vertices: np.ndarray,      # (N, 3) vertex positions
    faces: np.ndarray,         # Face indices
    normals: np.ndarray,       # Vertex normals
    scalars: np.ndarray,       # Scalar values for coloring
    vertex_colors: np.ndarray, # Per-vertex RGB colors
    colormap: str,             # Colormap name
    color: Tuple[float, ...],  # Solid color
    opacity: float             # 0.0-1.0
)

# Camera3D - Camera configuration
Camera3D(
    position: Tuple[float, float, float],
    focal_point: Tuple[float, float, float] = (0, 0, 0),
    up_vector: Tuple[float, float, float] = (0, 0, 1),
    fov: float = 30.0
)

# Light3D - Light source
Light3D(
    position: Tuple[float, float, float],
    color: Tuple[float, float, float] = (1, 1, 1),
    intensity: float = 1.0,
    light_type: str = "directional"  # "point", "directional"
)
```

---

## Mesh Creation

### From Data
```python
# Basic mesh from vertices and faces
vis = visual3d.mesh(
    vertices,              # (N, 3) array
    faces=None,            # (M, 3) or (M, 4) indices, optional
    normals=None,          # (N, 3) vertex normals
    scalars=None,          # (N,) values for colormap
    colormap="viridis",
    color=None,            # (r, g, b) solid color
    opacity=1.0
)

# Terrain mesh from 2D heightmap
vis = visual3d.mesh_from_field(
    heightmap,             # 2D Field or array
    scale_x=1.0,
    scale_y=1.0,
    scale_z=1.0,
    colormap="terrain",
    color_by_height=True
)
```

### Primitives
```python
# Sphere
vis = visual3d.sphere(
    center=(0, 0, 0),
    radius=1.0,
    resolution=32,
    colormap="viridis",
    color=None
)

# Box
vis = visual3d.box(
    bounds=(-1, 1, -1, 1, -1, 1),  # xmin, xmax, ymin, ymax, zmin, zmax
    colormap="viridis",
    color=None
)

# Cylinder
vis = visual3d.cylinder(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    radius=1.0,
    height=2.0,
    resolution=32,
    colormap="viridis",
    color=None
)
```

---

## Volumetric Visualization

### Isosurfaces
```python
# Single isosurface (marching cubes)
vis = visual3d.isosurface(
    volume,                # 3D array
    isovalue=0.5,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    colormap="viridis",
    color=None,
    opacity=1.0,
    color_by_value=True
)

# Multiple contour surfaces
vis = visual3d.contour_3d(
    volume,                # 3D array
    isovalues=None,        # List of values, or auto
    n_contours=5,          # If isovalues=None
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    colormap="viridis",
    opacity=0.7
)
```

### Volume Rendering
```python
# Direct volume rendering
vis = visual3d.volume_render(
    volume,                # 3D array
    opacity_map="linear",  # "linear", "sigmoid", "geom", "geom_r"
    colormap="viridis",
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    shade=True,
    ambient=0.3,
    diffuse=0.6,
    specular=0.5
)

# Planar slice through volume
vis = visual3d.slice_volume(
    volume,                # 3D array
    normal=(0, 0, 1),      # Slice plane normal
    origin=None,           # Slice origin (default: center)
    spacing=(1.0, 1.0, 1.0),
    volume_origin=(0.0, 0.0, 0.0),
    colormap="viridis",
    opacity=1.0
)
```

---

## Parametric Surfaces

```python
# Create surface from parametric function
def torus(u, v):
    R, r = 2.0, 0.5
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

vis = visual3d.parametric_surface(
    func=torus,            # f(u, v) -> (x, y, z)
    u_range=(0, 2*np.pi),
    v_range=(0, 2*np.pi),
    u_resolution=64,
    v_resolution=64,
    colormap="viridis",
    color=None,
    color_by="v"           # "u", "v", "z", or "none"
)
```

---

## Vector Field Visualization

### Streamlines
```python
# 3D streamlines from vector field
vis = visual3d.streamlines_3d(
    vector_field,          # (nx, ny, nz, 3) array
    seed_points=None,      # (N, 3) explicit seeds
    n_seeds=100,           # Random seeds if seed_points=None
    max_length=100.0,
    integration_step=0.5,
    colormap="plasma",
    color_by="magnitude",  # "magnitude" or "velocity"
    tube_radius=0.0,       # 0 = lines, >0 = tubes
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0)
)
```

### Glyph Field
```python
# Render vectors as glyphs (arrows, cones, etc.)
vis = visual3d.glyph_field(
    positions,             # (N, 3) glyph positions
    vectors,               # (N, 3) vector directions
    glyph_type="arrow",    # "arrow", "cone", "sphere", "cylinder"
    scale=1.0,
    scale_by_magnitude=True,
    colormap="viridis",
    color_by="magnitude"   # "magnitude" or "direction"
)
```

---

## Chemistry Visualization

### Molecular Structures
```python
# Render molecule
vis = visual3d.molecule(
    mol,                   # Molecule object
    style="ball_and_stick", # "ball_and_stick", "spacefill", "stick"
    color_by="element",    # "element", "charge", "residue"
    atom_scale=0.3,        # Atom sphere radius multiplier
    bond_radius=0.1,       # Bond cylinder radius
    resolution=16          # Sphere/cylinder resolution
)
```

### Orbital Visualization
```python
# Render molecular orbital (positive/negative lobes)
vis = visual3d.orbital(
    orbital_field,         # 3D array of orbital values
    isovalues=(0.02, -0.02),
    positive_color=(0.0, 0.0, 0.8),  # Blue
    negative_color=(0.8, 0.0, 0.0),  # Red
    opacity=0.7,
    spacing=(0.2, 0.2, 0.2),
    origin=(0.0, 0.0, 0.0)
)
```

---

## Camera System

### Basic Camera
```python
# Create camera
cam = visual3d.camera(
    position=(10, 10, 10),
    focal_point=(0, 0, 0),
    up_vector=(0, 0, 1),
    fov=30.0
)
```

### Camera Paths
```python
# Orbital camera animation
camera_path = visual3d.orbit_camera(
    center=(0, 0, 0),
    radius=10.0,
    elevation=30.0,        # Degrees above horizon
    frames=60,
    orbits=1.0,            # Number of full rotations
    start_azimuth=0.0
)

# Fly-through path
camera_path = visual3d.fly_path(
    waypoints=[            # List of (x, y, z) positions
        (0, 0, 10),
        (10, 0, 5),
        (10, 10, 5),
        (0, 10, 10)
    ],
    frames=120,
    look_at=(5, 5, 0),     # Fixed target point
    look_ahead=False,      # Or look in direction of motion
    interpolation="spline" # "linear" or "spline"
)
```

---

## Lighting

```python
# Create light source
light = visual3d.light(
    position=(1, 1, 1),
    color=(1.0, 1.0, 1.0),
    intensity=1.0,
    light_type="directional"  # or "point"
)

# Use in rendering
img = visual3d.render_3d(mesh, lights=[light])
```

---

## Rendering & Output

### Static Rendering
```python
# Render to image (returns Visual for output)
img = visual3d.render_3d(
    mesh1, mesh2, mesh3,   # Multiple Visual3D objects
    camera=cam,            # Optional camera
    lights=[light1],       # Optional lights
    background=(0, 0, 0),  # RGB background
    width=800,
    height=600,
    anti_aliasing=True
)

# Save to file
visual.output(img, "scene.png")
```

### Video Export
```python
# Render animation to video
visual3d.video_3d(
    mesh1, mesh2,          # Visual3D objects
    camera_path=orbit_path, # List[Camera3D]
    path="animation.mp4",
    fps=30,
    width=800,
    height=600,
    background=(0, 0, 0)
)
```

### Interactive Display
```python
# Open interactive PyVista window
visual3d.display_3d(
    mesh1, mesh2,
    camera=cam,
    background=(0.1, 0.1, 0.1),
    title="Morphogen 3D"
)
```

---

## Common Patterns

### Terrain Flyover
```python
# Generate terrain
heightmap = field.random((256, 256))
terrain = visual3d.mesh_from_field(heightmap, scale_z=20.0)

# Create flyover path
path = visual3d.fly_path(
    waypoints=[(0, 0, 50), (128, 128, 80), (256, 256, 50)],
    frames=180,
    look_ahead=True
)

# Render video
visual3d.video_3d(terrain, camera_path=path, path="flyover.mp4")
```

### Molecular Visualization
```python
# Load and visualize molecule
mol = molecular.load_smiles("CCO")
mol = molecular.generate_3d(mol)

vis = visual3d.molecule(mol, style="ball_and_stick")

# Orbit animation
orbit = visual3d.orbit_camera(center=(0, 0, 0), radius=5, frames=60)
visual3d.video_3d(vis, camera_path=orbit, path="ethanol.mp4")
```

### Volume Slice Gallery
```python
# Create slices through volume
slices = []
for z in [0.25, 0.5, 0.75]:
    origin = (nx/2, ny/2, z * nz)
    s = visual3d.slice_volume(volume, origin=origin)
    slices.append(s)

# Render all slices
visual3d.display_3d(*slices)
```

### Streamlines with Isosurface
```python
# Vector field streamlines
streams = visual3d.streamlines_3d(velocity_field, n_seeds=200)

# Isosurface of vorticity
vortex = visual3d.isosurface(vorticity_magnitude, isovalue=0.5, opacity=0.5)

# Composite scene
visual3d.render_3d(streams, vortex, background=(0.05, 0.05, 0.1))
```

---

## Colormaps

All operators accept standard matplotlib/PyVista colormaps:

| Category | Colormaps |
|----------|-----------|
| Sequential | `viridis`, `plasma`, `inferno`, `magma`, `cividis` |
| Diverging | `coolwarm`, `RdBu`, `seismic`, `PiYG` |
| Qualitative | `tab10`, `Set1`, `Paired` |
| Scientific | `terrain`, `ocean`, `gist_earth`, `hot`, `bone` |

---

## Dependencies

The visual3d module requires:
- `pyvista>=0.44` - 3D rendering backend
- `vtk>=9.3` - VTK library (installed with pyvista)
- `imageio` - Video encoding (for `video_3d`)
- `scipy` - Spline interpolation (for `fly_path`)

Install with:
```bash
pip install pyvista imageio scipy
```

---

## See Also

- [3D Visualization System Plan](../planning/3D_VISUALIZATION_SYSTEM_PLAN.md) - Architecture documentation
- [Visual Domain Quick Reference](./visual-domain-quickref.md) - 2D visualization/animation
- [Advanced Visualizations](./advanced-visualizations.md) - Spectrogram, graph, phase space
- [Chemistry Specification](../specifications/chemistry.md) - Molecular domain

---

*Module: `morphogen/stdlib/visual3d.py` (1,938 lines, 26 operators)*
*Tests: `tests/test_visual3d.py` (55 tests)*
