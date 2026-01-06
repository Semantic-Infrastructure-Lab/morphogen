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

        Handles solid color, scalar coloring, and default white styling.

        Args:
            plotter: PyVista Plotter instance
            obj: Visual3D object to add
        """
        mesh = obj.to_pyvista()

        if obj.color is not None:
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


# Module-level instance for DSL access
visual3d = Visual3DOperations
