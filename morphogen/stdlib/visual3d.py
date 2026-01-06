"""3D Visualization operations for Morphogen.

This module provides 3D visualization capabilities including mesh rendering,
isosurfaces, camera systems, and scene management using PyVista as the
primary backend.

Phase 1 - Foundation:
- Basic mesh rendering
- Camera system (orbit, position)
- Scene to image rendering
- PNG/video export

Phase 2 - Advanced Surfaces:
- Isosurface extraction (marching cubes)
- Parametric surfaces f(u,v) → (x,y,z)
- Enhanced lighting support
- Scalar range control
"""

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Union, Callable, Tuple, List
import numpy as np

from morphogen.core.operator import operator, OpCategory


# =============================================================================
# Core Types
# =============================================================================

@dataclass
class Camera3D:
    """3D camera configuration.

    Defines the viewpoint for 3D scene rendering.
    """
    position: Tuple[float, float, float]
    focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 30.0  # degrees

    def __repr__(self) -> str:
        return f"Camera3D(pos={self.position}, focal={self.focal_point})"


@dataclass
class Light3D:
    """3D light source configuration.

    Supports directional, point, and ambient light types.
    """
    position: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    light_type: str = "directional"  # "point", "directional", "ambient"

    def __repr__(self) -> str:
        return f"Light3D(type={self.light_type}, intensity={self.intensity})"


@dataclass
class Visual3D:
    """3D visual representation.

    Stores mesh data, scalars, and rendering properties for 3D visualization.
    Uses PyVista PolyData internally when available.
    """
    # Mesh data (stored as numpy for serialization, converted to PyVista on render)
    vertices: np.ndarray  # (N, 3) vertex positions
    faces: Optional[np.ndarray] = None  # (M, 3) or (M, 4) face indices

    # Optional data
    normals: Optional[np.ndarray] = None  # (N, 3) vertex normals
    scalars: Optional[np.ndarray] = None  # (N,) scalar values for coloring
    vectors: Optional[np.ndarray] = None  # (N, 3) vector data

    # Rendering properties
    colormap: str = "viridis"
    opacity: float = 1.0
    color: Optional[Tuple[float, float, float]] = None  # Solid color override
    clim: Optional[Tuple[float, float]] = None  # Scalar range (min, max)
    point_colors: Optional[np.ndarray] = None  # (N, 3) per-vertex RGB colors

    # Volume rendering data (for volume_render operator)
    _volume_data: Optional[dict] = dataclass_field(default=None, repr=False)

    # Cached PyVista mesh (created on demand)
    _pv_mesh: Optional[object] = dataclass_field(default=None, repr=False)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh."""
        return len(self.faces) if self.faces is not None else 0

    def to_pyvista(self):
        """Convert to PyVista PolyData mesh.

        Returns:
            pyvista.PolyData: The mesh in PyVista format.
        """
        if self._pv_mesh is not None:
            return self._pv_mesh

        import pyvista as pv

        if self.faces is not None:
            # Create mesh with faces
            # PyVista expects faces in format: [n_verts, v1, v2, v3, ...]
            n_face_verts = self.faces.shape[1] if len(self.faces.shape) > 1 else 3
            pv_faces = np.zeros((len(self.faces), n_face_verts + 1), dtype=np.int64)
            pv_faces[:, 0] = n_face_verts
            pv_faces[:, 1:] = self.faces

            mesh = pv.PolyData(self.vertices, pv_faces.ravel())
        else:
            # Point cloud only
            mesh = pv.PolyData(self.vertices)

        # Add scalars if present
        if self.scalars is not None:
            mesh.point_data["scalars"] = self.scalars

        # Add normals if present
        if self.normals is not None:
            mesh.point_data["Normals"] = self.normals
        elif self.faces is not None:
            # Compute normals if we have faces
            mesh.compute_normals(inplace=True)

        # Add vectors if present
        if self.vectors is not None:
            mesh.point_data["vectors"] = self.vectors

        self._pv_mesh = mesh
        return mesh

    def __repr__(self) -> str:
        return f"Visual3D(vertices={self.n_vertices}, faces={self.n_faces})"


@dataclass
class Scene3D:
    """3D scene container.

    Holds multiple Visual3D objects with camera and lighting configuration.
    """
    objects: List[Visual3D] = dataclass_field(default_factory=list)
    camera: Optional[Camera3D] = None
    lights: List[Light3D] = dataclass_field(default_factory=list)
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def add(self, visual: Visual3D) -> 'Scene3D':
        """Add a visual object to the scene."""
        self.objects.append(visual)
        return self

    def __repr__(self) -> str:
        return f"Scene3D(objects={len(self.objects)}, camera={self.camera is not None})"


# =============================================================================
# Operations Class
# =============================================================================

class Visual3DOperations:
    """Namespace for 3D visual operations (accessed as 'visual3d' in DSL).

    Provides operators for:
    - Mesh creation (from vertices/faces, heightmaps, etc.)
    - Camera configuration
    - Scene rendering to images/video
    - Isosurface extraction (Phase 2)
    - Molecular visualization (Phase 3)
    """

    # Common colormaps for 3D visualization
    COLORMAPS = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "terrain", "coolwarm", "RdBu", "RdYlBu",
        "hot", "fire", "ice", "bone", "gray",
    ]

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    @staticmethod
    def _generate_grid_faces(rows: int, cols: int) -> np.ndarray:
        """Generate triangle faces for a grid mesh.

        Creates two triangles per grid cell for a rows×cols vertex grid.
        Vertices are assumed to be in row-major order.

        Args:
            rows: Number of rows in the vertex grid
            cols: Number of columns in the vertex grid

        Returns:
            (M, 3) array of triangle face indices where M = 2*(rows-1)*(cols-1)
        """
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v00 = i * cols + j
                v01 = i * cols + (j + 1)
                v10 = (i + 1) * cols + j
                v11 = (i + 1) * cols + (j + 1)
                faces.append([v00, v01, v11])
                faces.append([v00, v11, v10])
        return np.array(faces, dtype=np.int64)

    @staticmethod
    def _add_mesh_to_plotter(plotter, obj: Visual3D) -> None:
        """Add a Visual3D object to a PyVista plotter with appropriate styling.

        Handles solid color, scalar coloring, per-vertex colors, volume rendering,
        and default white styling.

        Args:
            plotter: PyVista Plotter instance
            obj: Visual3D object to add
        """
        import pyvista as pv

        # Handle volume rendering specially
        if obj._volume_data is not None:
            vol_data = obj._volume_data
            volume = vol_data["volume"]
            nx, ny, nz = volume.shape

            # Create ImageData for volume
            grid = pv.ImageData(
                dimensions=(nx, ny, nz),
                spacing=vol_data["spacing"],
                origin=vol_data["origin"]
            )
            grid.point_data["values"] = volume.flatten(order='F')

            plotter.add_volume(
                grid,
                scalars="values",
                cmap=obj.colormap,
                opacity=vol_data["opacity_map"],
                shade=vol_data["shade"],
                ambient=vol_data["ambient"],
                diffuse=vol_data["diffuse"],
                specular=vol_data["specular"],
                show_scalar_bar=False,
            )
            return

        mesh = obj.to_pyvista()

        if obj.point_colors is not None:
            # Per-vertex RGB colors (e.g., molecular visualization)
            mesh['RGB'] = (obj.point_colors * 255).astype(np.uint8)
            plotter.add_mesh(
                mesh,
                scalars='RGB',
                rgb=True,
                opacity=obj.opacity,
                smooth_shading=True,
            )
        elif obj.color is not None:
            plotter.add_mesh(
                mesh,
                color=obj.color,
                opacity=obj.opacity,
                smooth_shading=True,
            )
        elif obj.scalars is not None:
            plotter.add_mesh(
                mesh,
                scalars="scalars",
                cmap=obj.colormap,
                clim=obj.clim,
                opacity=obj.opacity,
                smooth_shading=True,
                show_scalar_bar=False,
            )
        else:
            plotter.add_mesh(
                mesh,
                color="white",
                opacity=obj.opacity,
                smooth_shading=True,
            )

    @staticmethod
    def _configure_lights(plotter, lights: List[Light3D]) -> None:
        """Configure custom lighting on a PyVista plotter.

        Args:
            plotter: PyVista Plotter instance
            lights: List of Light3D configurations to add
        """
        import pyvista as pv

        plotter.remove_all_lights()
        for light_cfg in lights:
            if light_cfg.light_type == "point":
                pv_light = pv.Light(
                    position=light_cfg.position,
                    color=light_cfg.color,
                    intensity=light_cfg.intensity,
                    positional=True,
                )
            elif light_cfg.light_type == "ambient":
                pv_light = pv.Light(
                    position=light_cfg.position,
                    color=light_cfg.color,
                    intensity=light_cfg.intensity * 0.5,
                    positional=False,
                )
            else:  # directional
                pv_light = pv.Light(
                    position=light_cfg.position,
                    color=light_cfg.color,
                    intensity=light_cfg.intensity,
                    positional=False,
                )
            plotter.add_light(pv_light)

    # ==========================================================================
    # Mesh Creation Operators
    # ==========================================================================

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(vertices: np.ndarray, faces: np.ndarray, ...) -> Visual3D",
        deterministic=True,
        doc="Create 3D mesh from vertices and faces"
    )
    def mesh(
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        scalars: Optional[np.ndarray] = None,
        colormap: str = "viridis",
        color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 1.0
    ) -> Visual3D:
        """Create 3D mesh from vertices and faces.

        Args:
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) or (M, 4) array of face indices (triangles or quads)
            normals: Optional (N, 3) array of vertex normals
            scalars: Optional (N,) array of scalar values for coloring
            colormap: Colormap name for scalar coloring
            color: Optional solid color (r, g, b) in range [0, 1]
            opacity: Mesh opacity (0-1)

        Returns:
            Visual3D object ready for rendering

        Example:
            >>> # Create a simple triangle
            >>> verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
            >>> faces = np.array([[0, 1, 2]])
            >>> mesh = visual3d.mesh(verts, faces)
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {vertices.shape}")

        if faces is not None:
            faces = np.asarray(faces, dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1] not in (3, 4):
                raise ValueError(f"faces must be (M, 3) or (M, 4), got {faces.shape}")

        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
            if normals.shape != vertices.shape:
                raise ValueError(f"normals shape {normals.shape} must match vertices {vertices.shape}")

        if scalars is not None:
            scalars = np.asarray(scalars, dtype=np.float32)
            if scalars.shape[0] != vertices.shape[0]:
                raise ValueError(f"scalars length {len(scalars)} must match vertices {len(vertices)}")

        return Visual3D(
            vertices=vertices,
            faces=faces,
            normals=normals,
            scalars=scalars,
            colormap=colormap,
            color=color,
            opacity=opacity,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.TRANSFORM,
        signature="(heightmap: Field2D, ...) -> Visual3D",
        deterministic=True,
        doc="Create 3D terrain mesh from 2D heightmap field"
    )
    def mesh_from_field(
        heightmap,  # Field2D - avoid circular import
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        colormap: str = "terrain",
        color_by_height: bool = True
    ) -> Visual3D:
        """Create 3D terrain mesh from 2D heightmap field.

        Converts a 2D scalar field into a 3D surface mesh where the field
        values become the Z (height) coordinates.

        Args:
            heightmap: Field2D containing height values
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
            scale_z: Z-axis scale factor (height multiplier)
            colormap: Colormap for height-based coloring
            color_by_height: If True, color vertices by height value

        Returns:
            Visual3D mesh representing the terrain surface

        Example:
            >>> terrain = field.random((128, 128))
            >>> mesh = visual3d.mesh_from_field(terrain, scale_z=10.0)
        """
        # Get heightmap data
        if hasattr(heightmap, 'data'):
            data = heightmap.data
        else:
            data = np.asarray(heightmap)

        if data.ndim == 3:
            # Handle velocity fields - use magnitude
            data = np.sqrt(np.sum(data**2, axis=-1))

        height, width = data.shape[:2]

        # Create grid of vertices
        x = np.linspace(0, (width - 1) * scale_x, width)
        y = np.linspace(0, (height - 1) * scale_y, height)
        xx, yy = np.meshgrid(x, y)
        zz = data * scale_z

        # Flatten to vertex array
        vertices = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            zz.ravel()
        ]).astype(np.float32)

        # Create triangle faces (2 triangles per grid cell)
        faces = Visual3DOperations._generate_grid_faces(height, width)

        # Scalars for coloring
        scalars = zz.ravel() if color_by_height else None

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars,
            colormap=colormap,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.TRANSFORM,
        signature="(volume: np.ndarray, isovalue: float, ...) -> Visual3D",
        deterministic=True,
        doc="Extract isosurface from 3D scalar volume using marching cubes"
    )
    def isosurface(
        volume: np.ndarray,
        isovalue: float = 0.5,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        colormap: str = "viridis",
        color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 1.0,
        color_by_value: bool = True
    ) -> Visual3D:
        """Extract isosurface from 3D scalar volume using marching cubes.

        Creates a mesh surface where the scalar field equals the isovalue.
        This is the foundation for molecular orbital visualization, fluid
        boundaries, medical imaging, and any implicit surface rendering.

        Args:
            volume: 3D numpy array of scalar values (shape: D, H, W)
            isovalue: Scalar value at which to extract the surface
            spacing: Grid spacing (dx, dy, dz) for proper scaling
            origin: Grid origin (x, y, z) - where index [0,0,0] is located
            colormap: Colormap for scalar coloring
            color: Optional solid color (r, g, b) overrides scalar coloring
            opacity: Surface opacity (0-1)
            color_by_value: If True, color vertices by their scalar value

        Returns:
            Visual3D mesh representing the isosurface

        Example:
            >>> # Create 3D scalar field (sphere SDF)
            >>> x, y, z = np.mgrid[-2:2:64j, -2:2:64j, -2:2:64j]
            >>> volume = np.sqrt(x**2 + y**2 + z**2)
            >>> # Extract sphere surface at radius 1.5
            >>> sphere = visual3d.isosurface(volume, isovalue=1.5, origin=(-2, -2, -2))
        """
        import pyvista as pv

        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got {volume.ndim}D")

        # Create uniform grid
        grid = pv.ImageData(
            dimensions=volume.shape,
            spacing=spacing,
            origin=origin,
        )
        grid.point_data["scalars"] = volume.ravel(order="F")

        # Extract isosurface using marching cubes
        surface = grid.contour([isovalue], scalars="scalars")

        if surface.n_points == 0:
            raise ValueError(f"No isosurface found at isovalue={isovalue}. "
                           f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")

        # Get vertices and faces
        vertices = np.array(surface.points, dtype=np.float32)

        # Convert faces (PyVista format: [n, v0, v1, v2, ...])
        faces_raw = surface.faces.reshape(-1, 4)  # Triangles
        faces = faces_raw[:, 1:].astype(np.int64)

        # Scalars for coloring
        scalars = None
        if color_by_value and "scalars" in surface.point_data:
            scalars = np.array(surface.point_data["scalars"], dtype=np.float32)

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars,
            colormap=colormap,
            color=color,
            opacity=opacity,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(func: Callable, u_range: Tuple, v_range: Tuple, ...) -> Visual3D",
        deterministic=True,
        doc="Create mesh from parametric surface function f(u,v) → (x,y,z)"
    )
    def parametric_surface(
        func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        u_range: Tuple[float, float] = (0.0, 1.0),
        v_range: Tuple[float, float] = (0.0, 1.0),
        u_resolution: int = 64,
        v_resolution: int = 64,
        colormap: str = "viridis",
        color: Optional[Tuple[float, float, float]] = None,
        color_by: str = "v"
    ) -> Visual3D:
        """Create mesh from parametric surface function f(u,v) → (x,y,z).

        Evaluates a parametric function over a UV grid to generate a 3D surface.
        Useful for mathematical surfaces like spheres, tori, Klein bottles,
        Möbius strips, and any surface defined by parametric equations.

        Args:
            func: Function that takes (u, v) arrays and returns (x, y, z) arrays.
                  Each output should have same shape as input arrays.
            u_range: (u_min, u_max) parameter range
            v_range: (v_min, v_max) parameter range
            u_resolution: Number of samples in u direction
            v_resolution: Number of samples in v direction
            colormap: Colormap for scalar coloring
            color: Optional solid color (r, g, b)
            color_by: What to use for coloring: "u", "v", "z", or "none"

        Returns:
            Visual3D mesh of the parametric surface

        Example:
            >>> # Sphere via parametric equations
            >>> def sphere(u, v):
            ...     theta = u * 2 * np.pi  # azimuth
            ...     phi = v * np.pi        # elevation
            ...     x = np.sin(phi) * np.cos(theta)
            ...     y = np.sin(phi) * np.sin(theta)
            ...     z = np.cos(phi)
            ...     return x, y, z
            >>> mesh = visual3d.parametric_surface(sphere)

            >>> # Torus
            >>> def torus(u, v, R=2, r=0.5):
            ...     theta = u * 2 * np.pi
            ...     phi = v * 2 * np.pi
            ...     x = (R + r * np.cos(phi)) * np.cos(theta)
            ...     y = (R + r * np.cos(phi)) * np.sin(theta)
            ...     z = r * np.sin(phi)
            ...     return x, y, z
            >>> mesh = visual3d.parametric_surface(torus)
        """
        # Generate UV grid
        u = np.linspace(u_range[0], u_range[1], u_resolution)
        v = np.linspace(v_range[0], v_range[1], v_resolution)
        uu, vv = np.meshgrid(u, v)

        # Evaluate parametric function
        xx, yy, zz = func(uu, vv)

        # Flatten to vertex array
        vertices = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            zz.ravel()
        ]).astype(np.float32)

        # Create triangle faces
        faces = Visual3DOperations._generate_grid_faces(v_resolution, u_resolution)

        # Determine scalar coloring
        scalars = None
        if color is None:
            if color_by == "u":
                scalars = uu.ravel().astype(np.float32)
            elif color_by == "v":
                scalars = vv.ravel().astype(np.float32)
            elif color_by == "z":
                scalars = zz.ravel().astype(np.float32)

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars,
            colormap=colormap,
            color=color,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(position: Tuple, color: Tuple, intensity: float, ...) -> Light3D",
        deterministic=True,
        doc="Create light source configuration"
    )
    def light(
        position: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        light_type: str = "directional"
    ) -> Light3D:
        """Create light source configuration.

        Args:
            position: Light position (for point lights) or direction (for directional)
            color: Light color as RGB tuple (0-1 range)
            intensity: Light intensity multiplier
            light_type: "point", "directional", or "ambient"

        Returns:
            Light3D configuration object

        Example:
            >>> # Warm point light
            >>> warm = visual3d.light((10, 10, 10), color=(1.0, 0.9, 0.8), light_type="point")
            >>> # Blue ambient fill
            >>> ambient = visual3d.light(color=(0.1, 0.1, 0.2), intensity=0.3, light_type="ambient")
        """
        return Light3D(
            position=tuple(position),
            color=tuple(color),
            intensity=float(intensity),
            light_type=light_type,
        )

    # ==========================================================================
    # Camera Operators
    # ==========================================================================

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(position: Tuple, ...) -> Camera3D",
        deterministic=True,
        doc="Create camera configuration"
    )
    def camera(
        position: Tuple[float, float, float],
        focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        fov: float = 30.0
    ) -> Camera3D:
        """Create camera configuration.

        Args:
            position: (x, y, z) camera position in world coordinates
            focal_point: (x, y, z) point the camera looks at
            up_vector: (x, y, z) camera up direction
            fov: Field of view in degrees

        Returns:
            Camera3D configuration object

        Example:
            >>> cam = visual3d.camera(
            ...     position=(100, 100, 50),
            ...     focal_point=(64, 64, 0)
            ... )
        """
        return Camera3D(
            position=tuple(position),
            focal_point=tuple(focal_point),
            up_vector=tuple(up_vector),
            fov=float(fov),
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(center: Tuple, radius: float, ...) -> List[Camera3D]",
        deterministic=True,
        doc="Generate orbital camera path"
    )
    def orbit_camera(
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 10.0,
        elevation: float = 30.0,
        frames: int = 60,
        orbits: float = 1.0,
        start_azimuth: float = 0.0
    ) -> List[Camera3D]:
        """Generate orbital camera path around center point.

        Creates a list of camera positions that orbit around a center point,
        suitable for creating rotating 3D animations.

        Args:
            center: Point to orbit around
            radius: Distance from center
            elevation: Camera elevation angle in degrees
            frames: Number of camera positions to generate
            orbits: Number of complete orbits
            start_azimuth: Starting angle in degrees

        Returns:
            List of Camera3D objects for each frame

        Example:
            >>> cam_path = visual3d.orbit_camera(
            ...     center=(0, 0, 0),
            ...     radius=100,
            ...     frames=120
            ... )
        """
        cameras = []
        elev_rad = np.radians(elevation)

        for i in range(frames):
            # Azimuth angle for this frame
            t = i / frames
            azimuth = np.radians(start_azimuth + t * orbits * 360)

            # Calculate camera position
            x = center[0] + radius * np.cos(azimuth) * np.cos(elev_rad)
            y = center[1] + radius * np.sin(azimuth) * np.cos(elev_rad)
            z = center[2] + radius * np.sin(elev_rad)

            cameras.append(Camera3D(
                position=(x, y, z),
                focal_point=center,
                up_vector=(0.0, 0.0, 1.0),
            ))

        return cameras

    # ==========================================================================
    # Rendering Operators
    # ==========================================================================

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.TRANSFORM,
        signature="(*objects: Visual3D, camera: Camera3D, ...) -> Visual",
        deterministic=True,
        doc="Render 3D scene to 2D image"
    )
    def render_3d(
        *objects: Visual3D,
        camera: Optional[Camera3D] = None,
        lights: Optional[List[Light3D]] = None,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        width: int = 800,
        height: int = 600,
        anti_aliasing: bool = True
    ):
        """Render 3D scene to 2D image.

        Takes one or more Visual3D objects and renders them from the specified
        camera viewpoint, returning a Visual (2D image) that can be saved
        or displayed.

        Args:
            *objects: Visual3D objects to render
            camera: Camera configuration (auto-generated if None)
            lights: List of Light3D objects (default lighting if None)
            background: RGB background color
            width: Output image width in pixels
            height: Output image height in pixels
            anti_aliasing: Enable anti-aliasing for smoother edges

        Returns:
            Visual (2D image) of the rendered scene

        Example:
            >>> mesh = visual3d.mesh_from_field(terrain)
            >>> cam = visual3d.camera((100, 100, 50), (64, 64, 0))
            >>> img = visual3d.render_3d(mesh, camera=cam)
            >>> visual.output(img, "scene.png")
        """
        import pyvista as pv
        from morphogen.stdlib.visual import Visual

        # Configure for headless rendering if no display available
        import os
        if 'DISPLAY' not in os.environ:
            pv.start_xvfb()

        plotter = pv.Plotter(
            off_screen=True,
            window_size=(width, height),
        )

        # Set background
        plotter.set_background(background)

        # Configure custom lighting if provided
        if lights is not None:
            Visual3DOperations._configure_lights(plotter, lights)

        # Add all objects to the scene
        for obj in objects:
            Visual3DOperations._add_mesh_to_plotter(plotter, obj)

        # Configure camera
        if camera is not None:
            plotter.camera.position = camera.position
            plotter.camera.focal_point = camera.focal_point
            plotter.camera.up = camera.up_vector
            plotter.camera.view_angle = camera.fov

        # Configure anti-aliasing
        if anti_aliasing:
            plotter.enable_anti_aliasing('ssaa')

        # Render to image array
        img = plotter.screenshot(return_img=True)
        plotter.close()

        # Convert to Visual (normalize to [0, 1])
        img_float = img.astype(np.float32) / 255.0

        # Ensure RGB (drop alpha if present)
        if img_float.shape[2] == 4:
            img_float = img_float[:, :, :3]

        return Visual(img_float)

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(*objects: Visual3D, camera_path: List[Camera3D], path: str, ...) -> None",
        deterministic=True,
        doc="Render animated 3D scene to video"
    )
    def video_3d(
        *objects: Visual3D,
        camera_path: List[Camera3D],
        path: str,
        fps: int = 30,
        width: int = 800,
        height: int = 600,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> None:
        """Render animated 3D scene to video file.

        Creates a video by rendering the scene from each camera position
        in the camera path.

        Args:
            *objects: Visual3D objects to render
            camera_path: List of Camera3D positions for each frame
            path: Output video file path (.mp4, .gif)
            fps: Frames per second
            width: Output video width
            height: Output video height
            background: RGB background color

        Example:
            >>> mesh = visual3d.mesh_from_field(terrain)
            >>> cam_path = visual3d.orbit_camera(frames=120)
            >>> visual3d.video_3d(mesh, camera_path=cam_path, path="orbit.mp4")
        """
        import imageio

        # Render each frame
        frames = []
        for i, cam in enumerate(camera_path):
            # Render this frame
            img = Visual3DOperations.render_3d(
                *objects,
                camera=cam,
                background=background,
                width=width,
                height=height,
            )

            # Convert to uint8 for video
            frame = (img.data * 255).astype(np.uint8)
            frames.append(frame)

        # Write video
        if path.endswith('.gif'):
            imageio.mimsave(path, frames, duration=1000/fps)
        else:
            imageio.mimsave(path, frames, fps=fps)

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(*objects: Visual3D, camera: Camera3D, ...) -> None",
        deterministic=True,
        doc="Display 3D scene in interactive window"
    )
    def display_3d(
        *objects: Visual3D,
        camera: Optional[Camera3D] = None,
        background: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        title: str = "Morphogen 3D"
    ) -> None:
        """Display 3D scene in interactive window.

        Opens an interactive window for viewing and manipulating the 3D scene.
        Supports mouse rotation, zoom, and pan.

        Args:
            *objects: Visual3D objects to display
            camera: Initial camera position (optional)
            background: RGB background color
            title: Window title

        Example:
            >>> mesh = visual3d.mesh_from_field(terrain)
            >>> visual3d.display_3d(mesh)  # Opens interactive window
        """
        import pyvista as pv

        # Create interactive plotter
        plotter = pv.Plotter(title=title)
        plotter.set_background(background)

        # Add all objects
        for obj in objects:
            Visual3DOperations._add_mesh_to_plotter(plotter, obj)

        # Configure camera if provided
        if camera is not None:
            plotter.camera.position = camera.position
            plotter.camera.focal_point = camera.focal_point
            plotter.camera.up = camera.up_vector

        # Show interactive window
        plotter.show()

    # ==========================================================================
    # Primitive Generators
    # ==========================================================================

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(center: Tuple, radius: float, ...) -> Visual3D",
        deterministic=True,
        doc="Create a sphere mesh"
    )
    def sphere(
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        resolution: int = 32,
        colormap: str = "viridis",
        color: Optional[Tuple[float, float, float]] = None
    ) -> Visual3D:
        """Create a sphere mesh.

        Args:
            center: Center position (x, y, z)
            radius: Sphere radius
            resolution: Number of subdivisions (higher = smoother)
            colormap: Colormap for scalar coloring
            color: Optional solid color

        Returns:
            Visual3D sphere mesh
        """
        import pyvista as pv

        mesh = pv.Sphere(radius=radius, center=center, theta_resolution=resolution, phi_resolution=resolution)

        return Visual3D(
            vertices=np.array(mesh.points, dtype=np.float32),
            faces=mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64),
            colormap=colormap,
            color=color,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(bounds: Tuple, ...) -> Visual3D",
        deterministic=True,
        doc="Create a box mesh"
    )
    def box(
        bounds: Tuple[float, float, float, float, float, float] = (-1, 1, -1, 1, -1, 1),
        colormap: str = "viridis",
        color: Optional[Tuple[float, float, float]] = None
    ) -> Visual3D:
        """Create a box (cuboid) mesh.

        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            colormap: Colormap for scalar coloring
            color: Optional solid color

        Returns:
            Visual3D box mesh
        """
        import pyvista as pv

        mesh = pv.Box(bounds=bounds)

        return Visual3D(
            vertices=np.array(mesh.points, dtype=np.float32),
            faces=mesh.faces.reshape(-1, 5)[:, 1:].astype(np.int64),  # Quads
            colormap=colormap,
            color=color,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(center: Tuple, direction: Tuple, radius: float, height: float, ...) -> Visual3D",
        deterministic=True,
        doc="Create a cylinder mesh"
    )
    def cylinder(
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        radius: float = 1.0,
        height: float = 2.0,
        resolution: int = 32,
        colormap: str = "viridis",
        color: Optional[Tuple[float, float, float]] = None
    ) -> Visual3D:
        """Create a cylinder mesh.

        Args:
            center: Center position
            direction: Axis direction
            radius: Cylinder radius
            height: Cylinder height
            resolution: Number of sides
            colormap: Colormap for scalar coloring
            color: Optional solid color

        Returns:
            Visual3D cylinder mesh
        """
        import pyvista as pv

        mesh = pv.Cylinder(center=center, direction=direction, radius=radius, height=height, resolution=resolution)

        return Visual3D(
            vertices=np.array(mesh.points, dtype=np.float32),
            faces=mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64),
            colormap=colormap,
            color=color,
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(mol: Molecule, style: str, ...) -> Visual3D",
        deterministic=True,
        doc="Render molecular structure as 3D visualization"
    )
    def molecule(
        mol,  # Molecule type (avoid circular import)
        style: str = "ball_and_stick",
        color_by: str = "element",
        atom_scale: float = 0.3,
        bond_radius: float = 0.1,
        resolution: int = 16
    ) -> Visual3D:
        """Render molecular structure as 3D visualization.

        Args:
            mol: Molecule object with atoms, bonds, and positions
            style: Rendering style - "ball_and_stick", "spacefill", "stick", "wireframe"
            color_by: Coloring mode - "element", "charge", "index"
            atom_scale: Scale factor for atom spheres (relative to vdW radius)
            bond_radius: Radius of bond cylinders
            resolution: Mesh resolution for spheres/cylinders

        Returns:
            Visual3D containing combined molecular mesh
        """
        import pyvista as pv

        # CPK/Jmol element colors (RGB, normalized 0-1)
        element_colors = {
            'H': (1.0, 1.0, 1.0),      # White
            'C': (0.5, 0.5, 0.5),      # Gray
            'N': (0.0, 0.0, 1.0),      # Blue
            'O': (1.0, 0.0, 0.0),      # Red
            'F': (0.0, 1.0, 0.0),      # Green
            'Cl': (0.0, 1.0, 0.0),     # Green
            'Br': (0.6, 0.1, 0.1),     # Dark red
            'S': (1.0, 1.0, 0.0),      # Yellow
            'P': (1.0, 0.5, 0.0),      # Orange
            'Na': (0.0, 0.0, 1.0),     # Blue
            'default': (1.0, 0.0, 1.0)  # Magenta for unknown
        }

        # Van der Waals radii in Angstroms
        vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
            'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'S': 1.80,
            'P': 1.80, 'Na': 2.27, 'default': 1.50
        }

        # Collect all mesh components
        meshes = []

        # Determine atom radii based on style
        if style == "spacefill":
            base_scale = 1.0
        elif style == "ball_and_stick":
            base_scale = atom_scale
        elif style == "stick":
            base_scale = 0.15  # Small atoms for stick style
        elif style == "wireframe":
            base_scale = 0.0  # No atom spheres
        else:
            base_scale = atom_scale

        # Create atom spheres
        if style != "wireframe":
            for i, atom in enumerate(mol.atoms):
                pos = mol.positions[i]
                element = atom.element

                # Get color
                if color_by == "element":
                    color = element_colors.get(element, element_colors['default'])
                elif color_by == "charge":
                    # Map charge to color: negative=red, neutral=white, positive=blue
                    charge = atom.charge
                    if charge < 0:
                        t = min(1.0, abs(charge))
                        color = (1.0, 1.0 - t, 1.0 - t)  # Toward red
                    elif charge > 0:
                        t = min(1.0, charge)
                        color = (1.0 - t, 1.0 - t, 1.0)  # Toward blue
                    else:
                        color = (1.0, 1.0, 1.0)  # White
                elif color_by == "index":
                    # Rainbow by atom index
                    t = i / max(1, mol.n_atoms - 1)
                    color = (
                        0.5 + 0.5 * np.sin(2 * np.pi * t),
                        0.5 + 0.5 * np.sin(2 * np.pi * (t + 0.33)),
                        0.5 + 0.5 * np.sin(2 * np.pi * (t + 0.67))
                    )
                else:
                    color = element_colors.get(element, element_colors['default'])

                # Get radius
                radius = vdw_radii.get(element, vdw_radii['default']) * base_scale

                if radius > 0:
                    sphere = pv.Sphere(
                        center=tuple(pos),
                        radius=radius,
                        theta_resolution=resolution,
                        phi_resolution=resolution
                    )
                    # Store color as point data for rendering
                    sphere['color'] = np.array([color] * sphere.n_points)
                    meshes.append(sphere)

        # Create bond cylinders
        if style in ("ball_and_stick", "stick", "wireframe"):
            for bond in mol.bonds:
                pos1 = mol.positions[bond.atom1]
                pos2 = mol.positions[bond.atom2]

                # Bond midpoint and direction
                center = (pos1 + pos2) / 2
                direction = pos2 - pos1
                length = np.linalg.norm(direction)

                if length > 0.01:  # Skip if atoms overlap
                    # Create cylinder along bond
                    if style == "wireframe":
                        # Use lines for wireframe (thin cylinder approximation)
                        cyl_radius = 0.02
                    else:
                        cyl_radius = bond_radius

                    # Split bond into two halves colored by atom
                    for half in [0, 1]:
                        if half == 0:
                            c = (pos1 + center) / 2
                            atom_idx = bond.atom1
                        else:
                            c = (pos2 + center) / 2
                            atom_idx = bond.atom2

                        element = mol.atoms[atom_idx].element
                        color = element_colors.get(element, element_colors['default'])

                        cylinder = pv.Cylinder(
                            center=tuple(c),
                            direction=tuple(direction),
                            radius=cyl_radius,
                            height=length / 2,
                            resolution=max(8, resolution // 2)
                        )
                        cylinder['color'] = np.array([color] * cylinder.n_points)
                        meshes.append(cylinder)

        # Combine all meshes
        if len(meshes) == 0:
            # Return empty mesh if nothing to render
            return Visual3D(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64)
            )

        combined = meshes[0]
        for m in meshes[1:]:
            combined = combined.merge(m)

        # Extract colors from point data if available
        colors = None
        if 'color' in combined.point_data:
            colors = combined.point_data['color']

        return Visual3D(
            vertices=np.array(combined.points, dtype=np.float32),
            faces=combined.faces.reshape(-1, 4)[:, 1:].astype(np.int64),
            color=(0.5, 0.5, 0.5),  # Default gray, overridden by per-vertex colors
            point_colors=colors
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(orbital_field: np.ndarray, isovalues: Tuple, ...) -> Visual3D",
        deterministic=True,
        doc="Render molecular orbital as positive/negative isosurfaces"
    )
    def orbital(
        orbital_field: np.ndarray,
        isovalues: Tuple[float, float] = (0.02, -0.02),
        positive_color: Tuple[float, float, float] = (0.0, 0.0, 0.8),
        negative_color: Tuple[float, float, float] = (0.8, 0.0, 0.0),
        opacity: float = 0.7,
        spacing: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Visual3D:
        """Render molecular orbital as positive/negative isosurfaces.

        Args:
            orbital_field: 3D array of orbital values
            isovalues: (positive_isovalue, negative_isovalue)
            positive_color: Color for positive lobe (blue)
            negative_color: Color for negative lobe (red)
            opacity: Surface transparency
            spacing: Grid spacing in each dimension
            origin: Grid origin

        Returns:
            Visual3D with two-colored orbital lobes
        """
        import pyvista as pv

        # Create uniform grid for orbital data (use point data for contour)
        grid = pv.ImageData()
        grid.dimensions = orbital_field.shape
        grid.spacing = spacing
        grid.origin = origin
        grid.point_data['orbital'] = orbital_field.flatten(order='F')

        meshes = []

        # Positive isosurface
        if isovalues[0] != 0 and np.max(orbital_field) >= isovalues[0]:
            pos_surface = grid.contour([isovalues[0]], scalars='orbital')
            if pos_surface.n_points > 0:
                pos_surface['color'] = np.array([positive_color] * pos_surface.n_points)
                meshes.append(pos_surface)

        # Negative isosurface
        if isovalues[1] != 0 and np.min(orbital_field) <= isovalues[1]:
            neg_surface = grid.contour([isovalues[1]], scalars='orbital')
            if neg_surface.n_points > 0:
                neg_surface['color'] = np.array([negative_color] * neg_surface.n_points)
                meshes.append(neg_surface)

        if len(meshes) == 0:
            return Visual3D(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64)
            )

        combined = meshes[0]
        for m in meshes[1:]:
            combined = combined.merge(m)

        colors = None
        if 'color' in combined.point_data:
            colors = combined.point_data['color']

        return Visual3D(
            vertices=np.array(combined.points, dtype=np.float32),
            faces=combined.faces.reshape(-1, 4)[:, 1:].astype(np.int64),
            opacity=opacity,
            point_colors=colors
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.CONSTRUCT,
        signature="(waypoints: List, frames: int, ...) -> List[Camera3D]",
        deterministic=True,
        doc="Generate camera path through waypoints"
    )
    def fly_path(
        waypoints: List[Tuple[float, float, float]],
        frames: int,
        look_at: Optional[Tuple[float, float, float]] = None,
        look_ahead: bool = False,
        interpolation: str = "spline"
    ) -> List[Camera3D]:
        """Generate camera path through waypoints.

        Args:
            waypoints: List of (x, y, z) camera positions
            frames: Total number of frames
            look_at: Fixed point to look at (if None, uses look_ahead)
            look_ahead: If True, camera looks in direction of travel
            interpolation: "linear" or "spline" interpolation

        Returns:
            List of Camera3D configurations for each frame
        """
        from scipy import interpolate

        waypoints = np.array(waypoints)
        n_waypoints = len(waypoints)

        if n_waypoints < 2:
            raise ValueError("Need at least 2 waypoints")

        # Parameter along path
        t_waypoints = np.linspace(0, 1, n_waypoints)
        t_frames = np.linspace(0, 1, frames)

        # Interpolate positions
        if interpolation == "spline" and n_waypoints >= 4:
            # Cubic spline interpolation
            tck, u = interpolate.splprep([waypoints[:, i] for i in range(3)], s=0)
            positions = np.array(interpolate.splev(t_frames, tck)).T
        else:
            # Linear interpolation
            positions = np.array([
                np.interp(t_frames, t_waypoints, waypoints[:, i])
                for i in range(3)
            ]).T

        cameras = []
        for i in range(frames):
            pos = positions[i]

            if look_at is not None:
                focal = np.array(look_at)
            elif look_ahead and i < frames - 1:
                # Look toward next position
                focal = positions[min(i + 1, frames - 1)]
            else:
                focal = np.array([0.0, 0.0, 0.0])

            cameras.append(Camera3D(
                position=tuple(pos),
                focal_point=tuple(focal),
                up_vector=(0.0, 0.0, 1.0),
                fov=30.0
            ))

        return cameras

    # =========================================================================
    # Phase 5: Advanced Scientific Visualization
    # =========================================================================

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(vector_field: np.ndarray, ...) -> Visual3D",
        deterministic=True,
        doc="Generate streamlines from a 3D vector field"
    )
    def streamlines_3d(
        vector_field: np.ndarray,
        seed_points: Optional[np.ndarray] = None,
        n_seeds: int = 100,
        max_length: float = 100.0,
        integration_step: float = 0.5,
        colormap: str = "plasma",
        color_by: str = "magnitude",
        tube_radius: float = 0.0,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Visual3D:
        """Generate streamlines from a 3D vector field.

        Streamlines trace the path of massless particles through a vector field,
        providing intuitive visualization of flow patterns, magnetic fields, etc.

        Args:
            vector_field: 4D array of shape (nx, ny, nz, 3) containing vector components
            seed_points: Optional (N, 3) array of seed positions. If None, randomly generated.
            n_seeds: Number of random seed points if seed_points not provided
            max_length: Maximum streamline integration length
            integration_step: Step size for streamline integration
            colormap: Color map for streamlines
            color_by: What to color by - "magnitude", "x", "y", "z", or "arc_length"
            tube_radius: If > 0, render as tubes instead of lines
            spacing: Grid spacing in (x, y, z)
            origin: Origin offset in (x, y, z)

        Returns:
            Visual3D containing the streamlines mesh
        """
        import pyvista as pv

        if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
            raise ValueError(f"vector_field must have shape (nx, ny, nz, 3), got {vector_field.shape}")

        nx, ny, nz, _ = vector_field.shape

        # Create structured grid
        x = np.linspace(origin[0], origin[0] + (nx - 1) * spacing[0], nx)
        y = np.linspace(origin[1], origin[1] + (ny - 1) * spacing[1], ny)
        z = np.linspace(origin[2], origin[2] + (nz - 1) * spacing[2], nz)

        grid = pv.RectilinearGrid(x, y, z)

        # Reshape vectors for VTK (needs fortran order)
        vectors = vector_field.reshape(-1, 3, order='F')
        grid["vectors"] = vectors
        grid.set_active_vectors("vectors")

        # Generate seed points
        if seed_points is None:
            seed_points = np.column_stack([
                np.random.uniform(x.min(), x.max(), n_seeds),
                np.random.uniform(y.min(), y.max(), n_seeds),
                np.random.uniform(z.min(), z.max(), n_seeds)
            ])

        seed_cloud = pv.PolyData(seed_points)

        # Generate streamlines
        streamlines = grid.streamlines_from_source(
            seed_cloud,
            vectors="vectors",
            max_time=max_length,
            integration_direction="both",
            initial_step_length=integration_step
        )

        if streamlines.n_points == 0:
            return Visual3D(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=np.int32),
                colormap=colormap
            )

        # Calculate scalars for coloring
        if color_by == "magnitude":
            if "vectors" in streamlines.point_data:
                vecs = streamlines.point_data["vectors"]
                scalars = np.linalg.norm(vecs, axis=1)
            else:
                scalars = np.ones(streamlines.n_points)
        elif color_by in ["x", "y", "z"]:
            idx = {"x": 0, "y": 1, "z": 2}[color_by]
            scalars = streamlines.points[:, idx]
        else:
            scalars = np.ones(streamlines.n_points)

        # Convert to tubes if requested
        if tube_radius > 0:
            streamlines = streamlines.tube(radius=tube_radius)

        # Extract mesh data
        vertices = np.array(streamlines.points)
        if streamlines.n_cells > 0 and hasattr(streamlines, 'faces'):
            faces_flat = np.array(streamlines.faces)
            faces = []
            i = 0
            while i < len(faces_flat):
                n = faces_flat[i]
                if n == 3:
                    faces.append(faces_flat[i + 1:i + 4])
                i += n + 1
            faces = np.array(faces) if faces else np.zeros((0, 3), dtype=np.int32)
        else:
            faces = np.zeros((0, 3), dtype=np.int32)

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars if len(scalars) == len(vertices) else None,
            colormap=colormap
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(positions: np.ndarray, vectors: np.ndarray, ...) -> Visual3D",
        deterministic=True,
        doc="Render a vector field using glyphs (arrows, cones, etc.)"
    )
    def glyph_field(
        positions: np.ndarray,
        vectors: np.ndarray,
        glyph_type: str = "arrow",
        scale: float = 1.0,
        scale_by_magnitude: bool = True,
        colormap: str = "viridis",
        color_by: str = "magnitude"
    ) -> Visual3D:
        """Render a vector field using glyphs (arrows, cones, etc.).

        Places oriented glyphs at each position to visualize vector direction
        and optionally magnitude.

        Args:
            positions: (N, 3) array of glyph positions
            vectors: (N, 3) array of vector directions at each position
            glyph_type: Type of glyph - "arrow", "cone", "sphere", "cylinder"
            scale: Base scale factor for glyphs
            scale_by_magnitude: If True, scale glyph size by vector magnitude
            colormap: Color map for glyphs
            color_by: What to color by - "magnitude", "x", "y", "z"

        Returns:
            Visual3D containing the glyph mesh
        """
        import pyvista as pv

        positions = np.asarray(positions)
        vectors = np.asarray(vectors)

        if positions.shape[0] != vectors.shape[0]:
            raise ValueError(f"positions and vectors must have same length, got {positions.shape[0]} and {vectors.shape[0]}")

        # Create point cloud with vectors
        cloud = pv.PolyData(positions)
        cloud["vectors"] = vectors

        # Calculate magnitudes
        magnitudes = np.linalg.norm(vectors, axis=1)
        cloud["magnitude"] = magnitudes

        # Create glyph source
        if glyph_type == "arrow":
            source = pv.Arrow(tip_length=0.25, tip_radius=0.1, shaft_radius=0.05)
        elif glyph_type == "cone":
            source = pv.Cone(radius=0.3, height=1.0, resolution=16)
        elif glyph_type == "sphere":
            source = pv.Sphere(radius=0.5, theta_resolution=8, phi_resolution=8)
        elif glyph_type == "cylinder":
            source = pv.Cylinder(radius=0.1, height=1.0, resolution=16)
        else:
            raise ValueError(f"Unknown glyph_type: {glyph_type}. Use 'arrow', 'cone', 'sphere', or 'cylinder'")

        # Generate glyphs
        glyphs = cloud.glyph(
            orient="vectors",
            scale="magnitude" if scale_by_magnitude else False,
            factor=scale,
            geom=source
        )

        # Calculate scalars for coloring
        if color_by == "magnitude":
            scalars = glyphs.point_data.get("magnitude", magnitudes)
        elif color_by in ["x", "y", "z"]:
            idx = {"x": 0, "y": 1, "z": 2}[color_by]
            scalars = glyphs.points[:, idx]
        else:
            scalars = None

        # Extract mesh
        vertices = np.array(glyphs.points)
        faces_flat = np.array(glyphs.faces)

        # Parse VTK face format
        faces = []
        i = 0
        while i < len(faces_flat):
            n = faces_flat[i]
            if n == 3:
                faces.append(faces_flat[i + 1:i + 4])
            elif n == 4:
                idx = faces_flat[i + 1:i + 5]
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])
            i += n + 1

        faces = np.array(faces) if faces else np.zeros((0, 3), dtype=np.int32)

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars if scalars is not None and len(scalars) == len(vertices) else None,
            colormap=colormap
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(volume: np.ndarray, normal: Tuple, ...) -> Visual3D",
        deterministic=True,
        doc="Extract a planar slice through a 3D volume"
    )
    def slice_volume(
        volume: np.ndarray,
        normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        origin: Optional[Tuple[float, float, float]] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        volume_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        colormap: str = "viridis",
        opacity: float = 1.0
    ) -> Visual3D:
        """Extract a planar slice through a 3D volume.

        Creates a 2D cross-section of volumetric data along an arbitrary plane,
        useful for examining internal structure of 3D datasets.

        Args:
            volume: 3D array of scalar values
            normal: Normal vector defining the slice plane orientation
            origin: Point on the slice plane. If None, uses volume center.
            spacing: Grid spacing in (x, y, z)
            volume_origin: Origin offset of the volume in (x, y, z)
            colormap: Color map for the slice
            opacity: Opacity of the slice (0-1)

        Returns:
            Visual3D containing the slice mesh
        """
        import pyvista as pv

        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {volume.shape}")

        nx, ny, nz = volume.shape

        # Create image data (uniform grid)
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=spacing,
            origin=volume_origin
        )
        grid.point_data["values"] = volume.flatten(order='F')

        # Determine slice origin
        if origin is None:
            origin = (
                volume_origin[0] + (nx - 1) * spacing[0] / 2,
                volume_origin[1] + (ny - 1) * spacing[1] / 2,
                volume_origin[2] + (nz - 1) * spacing[2] / 2
            )

        # Extract slice
        sliced = grid.slice(normal=normal, origin=origin)

        if sliced.n_points == 0:
            return Visual3D(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=np.int32),
                colormap=colormap,
                opacity=opacity
            )

        # Extract mesh data
        vertices = np.array(sliced.points)
        scalars = sliced.point_data.get("values", None)

        # Extract faces
        if hasattr(sliced, 'faces') and len(sliced.faces) > 0:
            faces_flat = np.array(sliced.faces)
            faces = []
            i = 0
            while i < len(faces_flat):
                n = faces_flat[i]
                if n == 3:
                    faces.append(faces_flat[i + 1:i + 4])
                elif n == 4:
                    idx = faces_flat[i + 1:i + 5]
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
                i += n + 1
            faces = np.array(faces) if faces else np.zeros((0, 3), dtype=np.int32)
        else:
            faces = np.zeros((0, 3), dtype=np.int32)

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars,
            colormap=colormap,
            opacity=opacity
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(volume: np.ndarray, ...) -> Visual3D",
        deterministic=True,
        doc="Perform direct volume rendering of 3D scalar data"
    )
    def volume_render(
        volume: np.ndarray,
        opacity_map: str = "linear",
        colormap: str = "viridis",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        shade: bool = True,
        ambient: float = 0.3,
        diffuse: float = 0.6,
        specular: float = 0.5
    ) -> Visual3D:
        """Perform direct volume rendering of 3D scalar data.

        Uses ray casting to render semi-transparent volumetric data,
        ideal for medical imaging, fluid density fields, etc.

        Args:
            volume: 3D array of scalar values (will be normalized to 0-1)
            opacity_map: Opacity transfer function - "linear", "sigmoid", "geom", or "geom_r"
            colormap: Color map for volume values
            spacing: Grid spacing in (x, y, z)
            origin: Origin offset in (x, y, z)
            shade: Enable shading based on gradients
            ambient: Ambient lighting coefficient
            diffuse: Diffuse lighting coefficient
            specular: Specular lighting coefficient

        Returns:
            Visual3D containing the volume data (special handling in render)
        """
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {volume.shape}")

        # Normalize volume to 0-1
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            normalized = (volume - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(volume)

        return Visual3D(
            vertices=np.array([origin]),  # Dummy vertex at origin
            faces=np.zeros((0, 3), dtype=np.int32),
            scalars=normalized.flatten(order='F'),
            colormap=colormap,
            opacity=1.0,
            _volume_data={
                "volume": normalized,
                "spacing": spacing,
                "origin": origin,
                "opacity_map": opacity_map,
                "shade": shade,
                "ambient": ambient,
                "diffuse": diffuse,
                "specular": specular
            }
        )

    @staticmethod
    @operator(
        domain="visual3d",
        category=OpCategory.RENDER,
        signature="(volume: np.ndarray, ...) -> Visual3D",
        deterministic=True,
        doc="Generate multiple isosurface contours from a 3D scalar field"
    )
    def contour_3d(
        volume: np.ndarray,
        isovalues: Union[float, List[float]] = None,
        n_contours: int = 5,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        colormap: str = "viridis",
        opacity: float = 0.7
    ) -> Visual3D:
        """Generate multiple isosurface contours from a 3D scalar field.

        Creates a set of nested isosurfaces at specified values, useful for
        visualizing density distributions, potential fields, etc.

        Args:
            volume: 3D array of scalar values
            isovalues: Specific isovalue(s) to extract. If None, uses n_contours evenly spaced.
            n_contours: Number of contours if isovalues not specified
            spacing: Grid spacing in (x, y, z)
            origin: Origin offset in (x, y, z)
            colormap: Color map for contour surfaces
            opacity: Opacity of contour surfaces (0-1)

        Returns:
            Visual3D containing all contour surfaces
        """
        import pyvista as pv

        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {volume.shape}")

        nx, ny, nz = volume.shape

        # Create image data
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=spacing,
            origin=origin
        )
        grid.point_data["values"] = volume.flatten(order='F')

        # Determine isovalues
        if isovalues is None:
            vmin, vmax = volume.min(), volume.max()
            margin = (vmax - vmin) * 0.1
            isovalues = np.linspace(vmin + margin, vmax - margin, n_contours)
        elif isinstance(isovalues, (int, float)):
            isovalues = [isovalues]

        # Extract contours
        contours = grid.contour(isovalues, scalars="values")

        if contours.n_points == 0:
            return Visual3D(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=np.int32),
                colormap=colormap,
                opacity=opacity
            )

        # Extract mesh data
        vertices = np.array(contours.points)
        scalars = contours.point_data.get("values", None)

        # Extract faces
        faces_flat = np.array(contours.faces)
        faces = []
        i = 0
        while i < len(faces_flat):
            n = faces_flat[i]
            if n == 3:
                faces.append(faces_flat[i + 1:i + 4])
            elif n == 4:
                idx = faces_flat[i + 1:i + 5]
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])
            i += n + 1

        faces = np.array(faces) if faces else np.zeros((0, 3), dtype=np.int32)

        return Visual3D(
            vertices=vertices,
            faces=faces,
            scalars=scalars,
            colormap=colormap,
            opacity=opacity
        )


# Module-level instance for DSL access
visual3d = Visual3DOperations
