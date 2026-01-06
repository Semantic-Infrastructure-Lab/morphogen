"""Geometry Domain for 2D/3D geometric primitives and spatial operations.

This module provides geometric primitives, coordinate transformations,
spatial queries, and advanced computational geometry algorithms for Morphogen.

Features:
- 2D/3D geometric primitives (point, line, circle, rectangle, polygon, box, sphere, mesh)
- Coordinate system conversions (Cartesian, polar, spherical)
- Frame-aware transformations (translate, rotate, scale, transform)
- Spatial queries (distance, intersection, containment, closest point)
- Geometric properties (area, perimeter, centroid, bounding box, volume, surface area)
- Advanced algorithms (convex hull, Delaunay triangulation, Voronoi diagrams, mesh booleans)
- Field domain integration (spatial field sampling and region queries)
- Rigidbody domain integration (collision shape conversion)

Architecture:
- Layer 1: Primitive construction (point, line, circle, rectangle, polygon, box3d, sphere, mesh)
- Layer 2: Transformations (translate, rotate, scale, transform)
- Layer 3: Spatial queries (distance, intersection, contains)
- Layer 4: Coordinate conversions
- Layer 5: Geometric properties (area, perimeter, centroid, bounding box)
- Layer 6: Advanced algorithms (convex hull, Delaunay, Voronoi, mesh booleans)
- Layer 7: Field domain integration (field sampling, region queries)
- Layer 8: Rigidbody domain integration (collision shapes)
"""

from typing import Tuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# CORE TYPES
# ============================================================================


class CoordinateFrame(Enum):
    """Coordinate frame types."""

    CARTESIAN = "cartesian"
    POLAR = "polar"
    SPHERICAL = "spherical"


@dataclass
class Point2D:
    """2D point in Cartesian coordinates.

    Attributes:
        x: X coordinate
        y: Y coordinate
        frame: Coordinate frame (default: Cartesian)
    """

    x: float
    y: float
    frame: CoordinateFrame = CoordinateFrame.CARTESIAN

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array([self.x, self.y])

    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.3f}, y={self.y:.3f})"


@dataclass
class Point3D:
    """3D point in Cartesian coordinates.

    Attributes:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        frame: Coordinate frame (default: Cartesian)
    """

    x: float
    y: float
    z: float
    frame: CoordinateFrame = CoordinateFrame.CARTESIAN

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array([self.x, self.y, self.z])

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


@dataclass
class Line2D:
    """2D line segment defined by two endpoints.

    Attributes:
        start: Start point
        end: End point
    """

    start: Point2D
    end: Point2D

    @property
    def length(self) -> float:
        """Calculate line length."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.sqrt(dx * dx + dy * dy)

    @property
    def direction(self) -> np.ndarray:
        """Get normalized direction vector."""
        vec = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        length = np.linalg.norm(vec)
        return vec / length if length > 0 else np.array([0.0, 0.0])

    def __repr__(self) -> str:
        return f"Line2D({self.start} -> {self.end})"


@dataclass
class Circle:
    """2D circle defined by center and radius.

    Attributes:
        center: Center point
        radius: Circle radius
    """

    center: Point2D
    radius: float

    @property
    def area(self) -> float:
        """Calculate circle area."""
        return np.pi * self.radius * self.radius

    @property
    def circumference(self) -> float:
        """Calculate circle circumference."""
        return 2 * np.pi * self.radius

    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius:.3f})"


@dataclass
class Rectangle:
    """2D axis-aligned rectangle.

    Attributes:
        center: Center point
        width: Rectangle width
        height: Rectangle height
        rotation: Rotation angle in radians (default: 0)
    """

    center: Point2D
    width: float
    height: float
    rotation: float = 0.0

    @property
    def area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)

    def get_vertices(self) -> np.ndarray:
        """Get rectangle vertices in world space.

        Returns:
            Array of 4 vertices [4, 2]
        """
        hw, hh = self.width / 2, self.height / 2

        # Local vertices (centered at origin)
        local_verts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

        # Rotate if needed
        if self.rotation != 0:
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            local_verts = local_verts @ rot_matrix.T

        # Translate to center
        return local_verts + np.array([self.center.x, self.center.y])

    def __repr__(self) -> str:
        return f"Rectangle(center={self.center}, w={self.width:.3f}, h={self.height:.3f})"


@dataclass
class Polygon:
    """2D polygon defined by vertices.

    Attributes:
        vertices: List of vertices (counter-clockwise winding)
    """

    vertices: np.ndarray  # Shape: [N, 2]

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)

    def _compute_area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2.0

    @property
    def area(self) -> float:
        """Polygon area (positive for CCW, negative for CW winding)."""
        return self._compute_area()

    @property
    def centroid(self) -> Point2D:
        """Calculate polygon centroid."""
        return Point2D(x=float(np.mean(self.vertices[:, 0])), y=float(np.mean(self.vertices[:, 1])))

    def _compute_perimeter(self) -> float:
        """Calculate polygon perimeter."""
        n = len(self.vertices)
        if n < 2:
            return 0.0
        perim = 0.0
        for i in range(n):
            j = (i + 1) % n
            dx = self.vertices[j][0] - self.vertices[i][0]
            dy = self.vertices[j][1] - self.vertices[i][1]
            perim += np.sqrt(dx * dx + dy * dy)
        return perim

    @property
    def perimeter(self) -> float:
        """Polygon perimeter."""
        return self._compute_perimeter()

    def __repr__(self) -> str:
        return f"Polygon(vertices={self.num_vertices})"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box.

    Attributes:
        min_point: Minimum corner
        max_point: Maximum corner
    """

    min_point: Point2D
    max_point: Point2D

    @property
    def width(self) -> float:
        """Get box width."""
        return self.max_point.x - self.min_point.x

    @property
    def height(self) -> float:
        """Get box height."""
        return self.max_point.y - self.min_point.y

    @property
    def center(self) -> Point2D:
        """Get box center."""
        return Point2D(
            x=(self.min_point.x + self.max_point.x) / 2, y=(self.min_point.y + self.max_point.y) / 2
        )

    def __repr__(self) -> str:
        return f"BoundingBox({self.min_point} to {self.max_point})"


@dataclass
class Box3D:
    """3D axis-aligned box.

    Attributes:
        center: Center point
        width: Box width (x dimension)
        height: Box height (y dimension)
        depth: Box depth (z dimension)
        rotation: Rotation as Euler angles (rx, ry, rz) in radians
    """

    center: Point3D
    width: float
    height: float
    depth: float
    rotation: np.ndarray = None

    def __post_init__(self):
        """Initialize rotation to zero if not provided."""
        if self.rotation is None:
            self.rotation = np.zeros(3)

    @property
    def volume(self) -> float:
        """Calculate box volume."""
        return self.width * self.height * self.depth

    @property
    def surface_area(self) -> float:
        """Calculate box surface area."""
        return 2 * (self.width * self.height + self.width * self.depth + self.height * self.depth)

    def get_vertices(self) -> np.ndarray:
        """Get box vertices in world space.

        Returns:
            Array of 8 vertices [8, 3]
        """
        hw, hh, hd = self.width / 2, self.height / 2, self.depth / 2

        # Local vertices (centered at origin)
        local_verts = np.array([
            [-hw, -hh, -hd], [hw, -hh, -hd], [hw, hh, -hd], [-hw, hh, -hd],
            [-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]
        ])

        # Apply rotation if needed (simplified - full rotation matrix would be better)
        if np.any(self.rotation != 0):
            # Apply Euler rotations (XYZ order)
            rx, ry, rz = self.rotation

            # Rotation around X
            if rx != 0:
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]
                ])
                local_verts = local_verts @ Rx.T

            # Rotation around Y
            if ry != 0:
                Ry = np.array([
                    [np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]
                ])
                local_verts = local_verts @ Ry.T

            # Rotation around Z
            if rz != 0:
                Rz = np.array([
                    [np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]
                ])
                local_verts = local_verts @ Rz.T

        # Translate to center
        return local_verts + np.array([self.center.x, self.center.y, self.center.z])

    def __repr__(self) -> str:
        return f"Box3D(center={self.center}, w={self.width:.3f}, h={self.height:.3f}, d={self.depth:.3f})"


@dataclass
class Sphere:
    """3D sphere defined by center and radius.

    Attributes:
        center: Center point
        radius: Sphere radius
    """

    center: Point3D
    radius: float

    @property
    def volume(self) -> float:
        """Calculate sphere volume."""
        return (4.0 / 3.0) * np.pi * self.radius ** 3

    @property
    def surface_area(self) -> float:
        """Calculate sphere surface area."""
        return 4 * np.pi * self.radius ** 2

    def __repr__(self) -> str:
        return f"Sphere(center={self.center}, radius={self.radius:.3f})"


@dataclass
class Mesh:
    """3D triangular mesh defined by vertices and faces.

    Attributes:
        vertices: Array of vertices [N, 3]
        faces: Array of face indices [M, 3] (triangles)
        normals: Optional array of vertex normals [N, 3]
    """

    vertices: np.ndarray  # Shape: [N, 3]
    faces: np.ndarray  # Shape: [M, 3]
    normals: Optional[np.ndarray] = None  # Shape: [N, 3]

    def __post_init__(self):
        """Validate mesh data."""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must have shape [N, 3], got {self.vertices.shape}")

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must have shape [M, 3], got {self.faces.shape}")

        if self.normals is not None:
            if self.normals.shape != self.vertices.shape:
                raise ValueError(f"Normals shape {self.normals.shape} must match vertices shape {self.vertices.shape}")

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Get number of faces."""
        return len(self.faces)

    def compute_normals(self) -> np.ndarray:
        """Compute vertex normals from face normals.

        Returns:
            Array of vertex normals [N, 3]
        """
        vertex_normals = np.zeros_like(self.vertices)

        # Compute face normals and accumulate to vertices
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            # Cross product of two edges
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Accumulate to vertices
            vertex_normals[face[0]] += face_normal
            vertex_normals[face[1]] += face_normal
            vertex_normals[face[2]] += face_normal

        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        vertex_normals = vertex_normals / norms

        return vertex_normals

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.num_vertices}, faces={self.num_faces})"


# ============================================================================
# LAYER 1: PRIMITIVE CONSTRUCTION
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(x: float, y: float) -> Point2D",
    deterministic=True,
    doc="Create a 2D point",
)
def point2d(x: float, y: float) -> Point2D:
    """Create a 2D point in Cartesian coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Point2D instance

    Example:
        p = point2d(x=3.0, y=4.0)
    """
    return Point2D(x=x, y=y)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(x: float, y: float, z: float) -> Point3D",
    deterministic=True,
    doc="Create a 3D point",
)
def point3d(x: float, y: float, z: float) -> Point3D:
    """Create a 3D point in Cartesian coordinates.

    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        Point3D instance

    Example:
        p = point3d(x=1.0, y=2.0, z=3.0)
    """
    return Point3D(x=x, y=y, z=z)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(start: Point2D, end: Point2D) -> Line2D",
    deterministic=True,
    doc="Create a 2D line segment",
)
def line2d(start: Point2D, end: Point2D) -> Line2D:
    """Create a 2D line segment from two points.

    Args:
        start: Start point
        end: End point

    Returns:
        Line2D instance

    Example:
        line = line2d(
            start=point2d(0.0, 0.0),
            end=point2d(1.0, 1.0)
        )
    """
    return Line2D(start=start, end=end)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point2D, radius: float) -> Circle",
    deterministic=True,
    doc="Create a circle",
)
def circle(center: Point2D, radius: float) -> Circle:
    """Create a circle from center point and radius.

    Args:
        center: Center point
        radius: Circle radius (must be positive)

    Returns:
        Circle instance

    Example:
        circ = circle(
            center=point2d(0.0, 0.0),
            radius=5.0
        )
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    return Circle(center=center, radius=radius)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point2D, width: float, height: float, rotation: float) -> Rectangle",
    deterministic=True,
    doc="Create a rectangle",
)
def rectangle(center: Point2D, width: float, height: float, rotation: float = 0.0) -> Rectangle:
    """Create a rectangle from center, dimensions, and rotation.

    Args:
        center: Center point
        width: Rectangle width (must be positive)
        height: Rectangle height (must be positive)
        rotation: Rotation angle in radians (default: 0)

    Returns:
        Rectangle instance

    Example:
        rect = rectangle(
            center=point2d(0.0, 0.0),
            width=10.0,
            height=5.0,
            rotation=np.pi / 4
        )
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")

    return Rectangle(center=center, width=width, height=height, rotation=rotation)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(vertices: ndarray) -> Polygon",
    deterministic=True,
    doc="Create a polygon from vertices",
)
def polygon(vertices: np.ndarray) -> Polygon:
    """Create a polygon from an array of vertices.

    Args:
        vertices: Array of vertices with shape [N, 2] where N >= 3

    Returns:
        Polygon instance

    Example:
        # Triangle
        tri = polygon(vertices=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ]))
    """
    vertices = np.asarray(vertices)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f"Vertices must have shape [N, 2], got {vertices.shape}")

    if len(vertices) < 3:
        raise ValueError(f"Polygon must have at least 3 vertices, got {len(vertices)}")

    return Polygon(vertices=vertices)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point2D, radius: float, num_sides: int) -> Polygon",
    deterministic=True,
    doc="Create a regular polygon",
)
def regular_polygon(center: Point2D, radius: float, num_sides: int) -> Polygon:
    """Create a regular polygon (equal sides and angles).

    Args:
        center: Center point
        radius: Radius (distance from center to vertices)
        num_sides: Number of sides (must be >= 3)

    Returns:
        Polygon instance

    Example:
        # Regular hexagon
        hex = regular_polygon(
            center=point2d(0.0, 0.0),
            radius=1.0,
            num_sides=6
        )
    """
    if num_sides < 3:
        raise ValueError(f"Regular polygon must have at least 3 sides, got {num_sides}")

    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    # Generate vertices in counter-clockwise order
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertices = np.zeros((num_sides, 2))
    vertices[:, 0] = center.x + radius * np.cos(angles)
    vertices[:, 1] = center.y + radius * np.sin(angles)

    return Polygon(vertices=vertices)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point3D, width: float, height: float, depth: float, rotation: ndarray) -> Box3D",
    deterministic=True,
    doc="Create a 3D box",
)
def box3d(center: Point3D, width: float, height: float, depth: float, rotation: Optional[np.ndarray] = None) -> Box3D:
    """Create a 3D box from center, dimensions, and rotation.

    Args:
        center: Center point
        width: Box width (x dimension, must be positive)
        height: Box height (y dimension, must be positive)
        depth: Box depth (z dimension, must be positive)
        rotation: Rotation as Euler angles [rx, ry, rz] in radians (default: no rotation)

    Returns:
        Box3D instance

    Example:
        box = box3d(
            center=point3d(0.0, 0.0, 0.0),
            width=2.0,
            height=3.0,
            depth=4.0,
            rotation=np.array([0, 0, np.pi/4])
        )
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"Dimensions must be positive, got {width}x{height}x{depth}")

    if rotation is None:
        rotation = np.zeros(3)

    return Box3D(center=center, width=width, height=height, depth=depth, rotation=rotation)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point3D, radius: float) -> Sphere",
    deterministic=True,
    doc="Create a 3D sphere",
)
def sphere(center: Point3D, radius: float) -> Sphere:
    """Create a 3D sphere from center point and radius.

    Args:
        center: Center point
        radius: Sphere radius (must be positive)

    Returns:
        Sphere instance

    Example:
        s = sphere(
            center=point3d(0.0, 0.0, 0.0),
            radius=5.0
        )
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    return Sphere(center=center, radius=radius)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(vertices: ndarray, faces: ndarray, normals: Optional[ndarray]) -> Mesh",
    deterministic=True,
    doc="Create a triangular mesh",
)
def mesh(vertices: np.ndarray, faces: np.ndarray, normals: Optional[np.ndarray] = None) -> Mesh:
    """Create a triangular mesh from vertices and faces.

    Args:
        vertices: Array of vertices with shape [N, 3]
        faces: Array of face indices with shape [M, 3] (triangular faces)
        normals: Optional array of vertex normals with shape [N, 3]

    Returns:
        Mesh instance

    Example:
        # Simple tetrahedron
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3]
        ])
        m = mesh(vertices=verts, faces=faces)
    """
    return Mesh(vertices=vertices, faces=faces, normals=normals)


# ============================================================================
# LAYER 2: TRANSFORMATIONS
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point2D, dx: float, dy: float) -> Point2D",
    deterministic=True,
    doc="Translate a 2D point",
)
def translate_point2d(point: Point2D, dx: float, dy: float) -> Point2D:
    """Translate a 2D point by offset.

    Args:
        point: Point to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated point

    Example:
        p2 = translate_point2d(p1, dx=5.0, dy=3.0)
    """
    return Point2D(x=point.x + dx, y=point.y + dy, frame=point.frame)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(circle: Circle, dx: float, dy: float) -> Circle",
    deterministic=True,
    doc="Translate a circle",
)
def translate_circle(circle: Circle, dx: float, dy: float) -> Circle:
    """Translate a circle by offset.

    Args:
        circle: Circle to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated circle

    Example:
        c2 = translate_circle(c1, dx=10.0, dy=0.0)
    """
    new_center = translate_point2d(circle.center, dx, dy)
    return Circle(center=new_center, radius=circle.radius)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(rect: Rectangle, dx: float, dy: float) -> Rectangle",
    deterministic=True,
    doc="Translate a rectangle",
)
def translate_rectangle(rect: Rectangle, dx: float, dy: float) -> Rectangle:
    """Translate a rectangle by offset.

    Args:
        rect: Rectangle to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated rectangle

    Example:
        r2 = translate_rectangle(r1, dx=5.0, dy=5.0)
    """
    new_center = translate_point2d(rect.center, dx, dy)
    return Rectangle(
        center=new_center, width=rect.width, height=rect.height, rotation=rect.rotation
    )


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(poly: Polygon, dx: float, dy: float) -> Polygon",
    deterministic=True,
    doc="Translate a polygon",
)
def translate_polygon(poly: Polygon, dx: float, dy: float) -> Polygon:
    """Translate a polygon by offset.

    Args:
        poly: Polygon to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated polygon

    Example:
        p2 = translate_polygon(p1, dx=2.0, dy=-3.0)
    """
    new_vertices = poly.vertices + np.array([dx, dy])
    return Polygon(vertices=new_vertices)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point2D, center: Point2D, angle: float) -> Point2D",
    deterministic=True,
    doc="Rotate a 2D point around a center",
)
def rotate_point2d(point: Point2D, center: Point2D, angle: float) -> Point2D:
    """Rotate a 2D point around a center point.

    Args:
        point: Point to rotate
        center: Center of rotation
        angle: Rotation angle in radians (counter-clockwise)

    Returns:
        Rotated point

    Example:
        # Rotate 90 degrees counter-clockwise
        p2 = rotate_point2d(p1, center=point2d(0, 0), angle=np.pi/2)
    """
    # Translate to origin
    dx = point.x - center.x
    dy = point.y - center.y

    # Rotate
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    new_x = dx * cos_a - dy * sin_a
    new_y = dx * sin_a + dy * cos_a

    # Translate back
    return Point2D(x=new_x + center.x, y=new_y + center.y, frame=point.frame)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(circle: Circle, center: Point2D, angle: float) -> Circle",
    deterministic=True,
    doc="Rotate a circle around a center",
)
def rotate_circle(circle: Circle, center: Point2D, angle: float) -> Circle:
    """Rotate a circle around a center point.

    Args:
        circle: Circle to rotate
        center: Center of rotation
        angle: Rotation angle in radians

    Returns:
        Rotated circle

    Note:
        Only the circle's center is rotated; radius remains unchanged.
    """
    new_center = rotate_point2d(circle.center, center, angle)
    return Circle(center=new_center, radius=circle.radius)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(rect: Rectangle, angle: float) -> Rectangle",
    deterministic=True,
    doc="Rotate a rectangle around its center",
)
def rotate_rectangle(rect: Rectangle, angle: float) -> Rectangle:
    """Rotate a rectangle around its center.

    Args:
        rect: Rectangle to rotate
        angle: Rotation angle in radians

    Returns:
        Rotated rectangle

    Example:
        # Rotate 45 degrees
        r2 = rotate_rectangle(r1, angle=np.pi/4)
    """
    return Rectangle(
        center=rect.center, width=rect.width, height=rect.height, rotation=rect.rotation + angle
    )


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(poly: Polygon, center: Point2D, angle: float) -> Polygon",
    deterministic=True,
    doc="Rotate a polygon around a center",
)
def rotate_polygon(poly: Polygon, center: Point2D, angle: float) -> Polygon:
    """Rotate a polygon around a center point.

    Args:
        poly: Polygon to rotate
        center: Center of rotation
        angle: Rotation angle in radians

    Returns:
        Rotated polygon

    Example:
        p2 = rotate_polygon(p1, center=p1.centroid, angle=np.pi/6)
    """
    # Translate to origin
    vertices = poly.vertices - np.array([center.x, center.y])

    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate
    rotated = vertices @ rot_matrix.T

    # Translate back
    new_vertices = rotated + np.array([center.x, center.y])

    return Polygon(vertices=new_vertices)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(circle: Circle, center: Point2D, scale: float) -> Circle",
    deterministic=True,
    doc="Scale a circle from a center point",
)
def scale_circle(circle: Circle, center: Point2D, scale: float) -> Circle:
    """Scale a circle from a center point.

    Args:
        circle: Circle to scale
        center: Center of scaling
        scale: Scale factor (must be positive)

    Returns:
        Scaled circle

    Example:
        # Double the size
        c2 = scale_circle(c1, center=c1.center, scale=2.0)
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")

    # Scale center position
    dx = (circle.center.x - center.x) * scale
    dy = (circle.center.y - center.y) * scale

    new_center = Point2D(x=center.x + dx, y=center.y + dy)
    new_radius = circle.radius * scale

    return Circle(center=new_center, radius=new_radius)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(rect: Rectangle, center: Point2D, scale_x: float, scale_y: float) -> Rectangle",
    deterministic=True,
    doc="Scale a rectangle from a center point",
)
def scale_rectangle(rect: Rectangle, center: Point2D, scale_x: float, scale_y: float) -> Rectangle:
    """Scale a rectangle from a center point.

    Args:
        rect: Rectangle to scale
        center: Center of scaling
        scale_x: X scale factor
        scale_y: Y scale factor

    Returns:
        Scaled rectangle

    Example:
        # Scale width by 2, height by 1.5
        r2 = scale_rectangle(r1, center=r1.center, scale_x=2.0, scale_y=1.5)
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Scale factors must be positive, got {scale_x}, {scale_y}")

    # Scale center position
    dx = (rect.center.x - center.x) * scale_x
    dy = (rect.center.y - center.y) * scale_y

    new_center = Point2D(x=center.x + dx, y=center.y + dy)

    return Rectangle(
        center=new_center,
        width=rect.width * scale_x,
        height=rect.height * scale_y,
        rotation=rect.rotation,
    )


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(poly: Polygon, center: Point2D, scale_x: float, scale_y: float) -> Polygon",
    deterministic=True,
    doc="Scale a polygon from a center point",
)
def scale_polygon(poly: Polygon, center: Point2D, scale_x: float, scale_y: float) -> Polygon:
    """Scale a polygon from a center point.

    Args:
        poly: Polygon to scale
        center: Center of scaling
        scale_x: X scale factor
        scale_y: Y scale factor

    Returns:
        Scaled polygon

    Example:
        p2 = scale_polygon(p1, center=p1.centroid, scale_x=2.0, scale_y=2.0)
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Scale factors must be positive, got {scale_x}, {scale_y}")

    # Translate to origin (convert to float to avoid casting issues)
    vertices = poly.vertices.astype(np.float64) - np.array([center.x, center.y])

    # Scale
    vertices[:, 0] *= scale_x
    vertices[:, 1] *= scale_y

    # Translate back
    new_vertices = vertices + np.array([center.x, center.y])

    return Polygon(vertices=new_vertices)


# ============================================================================
# LAYER 3: SPATIAL QUERIES
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(p1: Point2D, p2: Point2D) -> float",
    deterministic=True,
    doc="Calculate Euclidean distance between two 2D points",
)
def distance_point_point(p1: Point2D, p2: Point2D) -> float:
    """Calculate Euclidean distance between two 2D points.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Distance between points

    Example:
        dist = distance_point_point(
            point2d(0, 0),
            point2d(3, 4)
        )  # Returns 5.0
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return float(np.sqrt(dx * dx + dy * dy))


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, line: Line2D) -> float",
    deterministic=True,
    doc="Calculate distance from point to line segment",
)
def distance_point_line(point: Point2D, line: Line2D) -> float:
    """Calculate shortest distance from point to line segment.

    Args:
        point: Point to measure from
        line: Line segment

    Returns:
        Shortest distance

    Example:
        dist = distance_point_line(
            point=point2d(0, 1),
            line=line2d(point2d(-1, 0), point2d(1, 0))
        )  # Returns 1.0
    """
    # Vector from line start to point
    px = point.x - line.start.x
    py = point.y - line.start.y

    # Vector from line start to end
    lx = line.end.x - line.start.x
    ly = line.end.y - line.start.y

    # Line length squared
    line_len_sq = lx * lx + ly * ly

    if line_len_sq == 0:
        # Line is a point
        return distance_point_point(point, line.start)

    # Project point onto line (clamped to [0, 1])
    t = max(0, min(1, (px * lx + py * ly) / line_len_sq))

    # Closest point on line
    closest_x = line.start.x + t * lx
    closest_y = line.start.y + t * ly

    # Distance to closest point
    dx = point.x - closest_x
    dy = point.y - closest_y

    return float(np.sqrt(dx * dx + dy * dy))


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, circle: Circle) -> float",
    deterministic=True,
    doc="Calculate distance from point to circle perimeter",
)
def distance_point_circle(point: Point2D, circle: Circle) -> float:
    """Calculate distance from point to circle perimeter.

    Args:
        point: Point to measure from
        circle: Circle

    Returns:
        Distance to circle perimeter (negative if inside)

    Example:
        dist = distance_point_circle(
            point=point2d(5, 0),
            circle=circle(center=point2d(0, 0), radius=3)
        )  # Returns 2.0
    """
    dist_to_center = distance_point_point(point, circle.center)
    return dist_to_center - circle.radius


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(c1: Circle, c2: Circle) -> Optional[Tuple[Point2D, Point2D]]",
    deterministic=True,
    doc="Find intersection points between two circles",
)
def intersect_circle_circle(c1: Circle, c2: Circle) -> Optional[Tuple[Point2D, Point2D]]:
    """Find intersection points between two circles.

    Args:
        c1: First circle
        c2: Second circle

    Returns:
        Tuple of two intersection points, or None if no intersection

    Example:
        points = intersect_circle_circle(circle1, circle2)
        if points:
            p1, p2 = points
    """
    # Distance between centers
    d = distance_point_point(c1.center, c2.center)

    # Check if circles intersect
    if d > c1.radius + c2.radius or d < abs(c1.radius - c2.radius) or d == 0:
        return None

    # Find intersection points using analytical solution
    a = (c1.radius * c1.radius - c2.radius * c2.radius + d * d) / (2 * d)
    h = np.sqrt(c1.radius * c1.radius - a * a)

    # Point on line between centers
    dx = c2.center.x - c1.center.x
    dy = c2.center.y - c1.center.y

    px = c1.center.x + a * dx / d
    py = c1.center.y + a * dy / d

    # Perpendicular offset
    offset_x = h * dy / d
    offset_y = -h * dx / d

    p1 = Point2D(x=px + offset_x, y=py + offset_y)
    p2 = Point2D(x=px - offset_x, y=py - offset_y)

    return (p1, p2)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, circle: Circle) -> bool",
    deterministic=True,
    doc="Check if point is inside circle",
)
def contains_circle_point(circle: Circle, point: Point2D) -> bool:
    """Check if a circle contains a point.

    Args:
        circle: Circle
        point: Point to test

    Returns:
        True if point is inside or on circle boundary

    Example:
        inside = contains_circle_point(
            circle=circle(center=point2d(0, 0), radius=5),
            point=point2d(3, 4)
        )  # True (distance is 5.0)
    """
    dist = distance_point_point(circle.center, point)
    return dist <= circle.radius


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, poly: Polygon) -> bool",
    deterministic=True,
    doc="Check if point is inside polygon (ray casting)",
)
def contains_polygon_point(poly: Polygon, point: Point2D) -> bool:
    """Check if a polygon contains a point using ray casting algorithm.

    Args:
        poly: Polygon
        point: Point to test

    Returns:
        True if point is inside polygon

    Algorithm:
        Casts a ray from the point to infinity and counts edge crossings.
        Odd number of crossings = inside, even = outside.

    Example:
        inside = contains_polygon_point(triangle, point2d(0.5, 0.5))
    """
    n = len(poly.vertices)
    inside = False

    x, y = point.x, point.y

    j = n - 1
    for i in range(n):
        xi, yi = poly.vertices[i]
        xj, yj = poly.vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, rect: Rectangle) -> bool",
    deterministic=True,
    doc="Check if point is inside rectangle",
)
def contains_rectangle_point(rect: Rectangle, point: Point2D) -> bool:
    """Check if a rectangle contains a point.

    Args:
        rect: Rectangle
        point: Point to test

    Returns:
        True if point is inside rectangle

    Example:
        inside = contains_rectangle_point(rect, point2d(1, 1))
    """
    # If rectangle is not rotated, use simple bounds check
    if rect.rotation == 0:
        hw, hh = rect.width / 2, rect.height / 2
        dx = abs(point.x - rect.center.x)
        dy = abs(point.y - rect.center.y)
        return bool(dx <= hw and dy <= hh)

    # For rotated rectangle, transform point to local space
    # Translate to origin
    dx = point.x - rect.center.x
    dy = point.y - rect.center.y

    # Rotate by -rotation
    cos_r = np.cos(-rect.rotation)
    sin_r = np.sin(-rect.rotation)

    local_x = dx * cos_r - dy * sin_r
    local_y = dx * sin_r + dy * cos_r

    # Check bounds in local space
    hw, hh = rect.width / 2, rect.height / 2
    return bool(abs(local_x) <= hw and abs(local_y) <= hh)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon], point: Point2D) -> bool",
    deterministic=True,
    doc="Check if shape contains point (generic dispatch)",
)
def contains(shape: Union[Circle, Rectangle, Polygon], point: Point2D) -> bool:
    """Check if a geometric shape contains a point.

    Args:
        shape: Circle, Rectangle, or Polygon
        point: Point to test

    Returns:
        True if point is inside shape

    Example:
        >>> circ = circle(point2d(0, 0), 5)
        >>> contains(circ, point2d(1, 1))
        True
    """
    if isinstance(shape, Circle):
        return contains_circle_point(shape, point)
    elif isinstance(shape, Rectangle):
        return contains_rectangle_point(shape, point)
    elif isinstance(shape, Polygon):
        return contains_polygon_point(shape, point)
    else:
        raise TypeError(f"contains() not implemented for type {type(shape)}")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, circle: Circle) -> Point2D",
    deterministic=True,
    doc="Find closest point on circle to given point",
)
def closest_point_circle(circle: Circle, point: Point2D) -> Point2D:
    """Find closest point on circle perimeter to a given point.

    Args:
        circle: Circle
        point: Point to find closest point to

    Returns:
        Closest point on circle

    Example:
        closest = closest_point_circle(
            circle=circle(center=point2d(0, 0), radius=5),
            point=point2d(10, 0)
        )  # Returns point2d(5, 0)
    """
    # Direction from center to point
    dx = point.x - circle.center.x
    dy = point.y - circle.center.y

    dist = np.sqrt(dx * dx + dy * dy)

    if dist == 0:
        # Point is at center, return any point on circle
        return Point2D(x=circle.center.x + circle.radius, y=circle.center.y)

    # Normalize and scale by radius
    scale = circle.radius / dist

    return Point2D(x=circle.center.x + dx * scale, y=circle.center.y + dy * scale)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, line: Line2D) -> Point2D",
    deterministic=True,
    doc="Find closest point on line segment to given point",
)
def closest_point_line(line: Line2D, point: Point2D) -> Point2D:
    """Find closest point on line segment to a given point.

    Args:
        line: Line segment
        point: Point to find closest point to

    Returns:
        Closest point on line

    Example:
        closest = closest_point_line(
            line=line2d(point2d(0, 0), point2d(10, 0)),
            point=point2d(5, 5)
        )  # Returns point2d(5, 0)
    """
    # Vector from line start to point
    px = point.x - line.start.x
    py = point.y - line.start.y

    # Vector from line start to end
    lx = line.end.x - line.start.x
    ly = line.end.y - line.start.y

    # Line length squared
    line_len_sq = lx * lx + ly * ly

    if line_len_sq == 0:
        # Line is a point
        return line.start

    # Project point onto line (clamped to [0, 1])
    t = max(0, min(1, (px * lx + py * ly) / line_len_sq))

    # Closest point on line
    return Point2D(x=line.start.x + t * lx, y=line.start.y + t * ly)


# ============================================================================
# LAYER 4: COORDINATE CONVERSIONS
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point2D) -> Tuple[float, float]",
    deterministic=True,
    doc="Convert Cartesian coordinates to polar (r, theta)",
)
def cartesian_to_polar(point: Point2D) -> Tuple[float, float]:
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        point: Point in Cartesian coordinates

    Returns:
        Tuple of (radius, angle) where angle is in radians

    Example:
        r, theta = cartesian_to_polar(point2d(3, 4))
        # r = 5.0, theta ≈ 0.927 radians (53.13 degrees)
    """
    r = np.sqrt(point.x * point.x + point.y * point.y)
    theta = np.arctan2(point.y, point.x)
    return (float(r), float(theta))


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(r: float, theta: float) -> Point2D",
    deterministic=True,
    doc="Convert polar coordinates to Cartesian (x, y)",
)
def polar_to_cartesian(r: float, theta: float) -> Point2D:
    """Convert polar coordinates to Cartesian coordinates.

    Args:
        r: Radius
        theta: Angle in radians

    Returns:
        Point in Cartesian coordinates

    Example:
        point = polar_to_cartesian(r=5.0, theta=np.pi/4)
        # Returns point2d(3.536, 3.536)
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return Point2D(x=float(x), y=float(y), frame=CoordinateFrame.POLAR)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point3D) -> Tuple[float, float, float]",
    deterministic=True,
    doc="Convert Cartesian 3D to spherical (r, theta, phi)",
)
def cartesian_to_spherical(point: Point3D) -> Tuple[float, float, float]:
    """Convert Cartesian 3D coordinates to spherical coordinates.

    Args:
        point: Point in Cartesian coordinates

    Returns:
        Tuple of (r, theta, phi) where:
        - r: Radial distance
        - theta: Azimuthal angle (0 to 2π)
        - phi: Polar angle from z-axis (0 to π)

    Example:
        r, theta, phi = cartesian_to_spherical(point3d(1, 1, 1))
    """
    r = np.sqrt(point.x * point.x + point.y * point.y + point.z * point.z)
    theta = np.arctan2(point.y, point.x)
    phi = np.arccos(point.z / r) if r > 0 else 0.0

    return (float(r), float(theta), float(phi))


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(r: float, theta: float, phi: float) -> Point3D",
    deterministic=True,
    doc="Convert spherical coordinates to Cartesian 3D",
)
def spherical_to_cartesian(r: float, theta: float, phi: float) -> Point3D:
    """Convert spherical coordinates to Cartesian 3D coordinates.

    Args:
        r: Radial distance
        theta: Azimuthal angle (radians)
        phi: Polar angle from z-axis (radians)

    Returns:
        Point in Cartesian coordinates

    Example:
        point = spherical_to_cartesian(r=1.0, theta=0, phi=np.pi/2)
        # Returns point on unit sphere
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return Point3D(x=float(x), y=float(y), z=float(z), frame=CoordinateFrame.SPHERICAL)


# ============================================================================
# LAYER 5: GEOMETRIC PROPERTIES
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon]) -> float",
    deterministic=True,
    doc="Calculate area of a 2D shape",
)
def area(shape: Union[Circle, Rectangle, Polygon]) -> float:
    """Calculate area of a 2D shape.

    Args:
        shape: Circle, Rectangle, or Polygon

    Returns:
        Area of the shape

    Example:
        a = area(circle(center=point2d(0, 0), radius=5))  # 78.54
    """
    return shape.area


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon]) -> float",
    deterministic=True,
    doc="Calculate perimeter of a 2D shape",
)
def perimeter(shape: Union[Circle, Rectangle, Polygon]) -> float:
    """Calculate perimeter of a 2D shape.

    Args:
        shape: Circle, Rectangle, or Polygon

    Returns:
        Perimeter (or circumference for circles)

    Example:
        p = perimeter(rectangle(point2d(0,0), width=4, height=3))  # 14
    """
    if isinstance(shape, Circle):
        return shape.circumference
    elif isinstance(shape, (Rectangle, Polygon)):
        return shape.perimeter
    else:
        raise TypeError(f"Cannot calculate perimeter for {type(shape)}")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Rectangle, Polygon]) -> Point2D",
    deterministic=True,
    doc="Calculate centroid of a shape",
)
def centroid(shape: Union[Rectangle, Polygon]) -> Point2D:
    """Calculate centroid (center of mass) of a shape.

    Args:
        shape: Rectangle or Polygon

    Returns:
        Centroid point

    Example:
        c = centroid(triangle)
    """
    if isinstance(shape, Rectangle):
        return shape.center
    elif isinstance(shape, Polygon):
        return shape.centroid
    else:
        raise TypeError(f"Cannot calculate centroid for {type(shape)}")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon]) -> BoundingBox",
    deterministic=True,
    doc="Calculate axis-aligned bounding box of a shape",
)
def bounding_box(shape: Union[Circle, Rectangle, Polygon]) -> BoundingBox:
    """Calculate axis-aligned bounding box of a shape.

    Args:
        shape: Circle, Rectangle, or Polygon

    Returns:
        Bounding box

    Example:
        bbox = bounding_box(circle(center=point2d(0, 0), radius=5))
    """
    if isinstance(shape, Circle):
        return BoundingBox(
            min_point=Point2D(x=shape.center.x - shape.radius, y=shape.center.y - shape.radius),
            max_point=Point2D(x=shape.center.x + shape.radius, y=shape.center.y + shape.radius),
        )
    elif isinstance(shape, Rectangle):
        verts = shape.get_vertices()
        return BoundingBox(
            min_point=Point2D(x=float(np.min(verts[:, 0])), y=float(np.min(verts[:, 1]))),
            max_point=Point2D(x=float(np.max(verts[:, 0])), y=float(np.max(verts[:, 1]))),
        )
    elif isinstance(shape, Polygon):
        return BoundingBox(
            min_point=Point2D(
                x=float(np.min(shape.vertices[:, 0])), y=float(np.min(shape.vertices[:, 1]))
            ),
            max_point=Point2D(
                x=float(np.max(shape.vertices[:, 0])), y=float(np.max(shape.vertices[:, 1]))
            ),
        )
    else:
        raise TypeError(f"Cannot calculate bounding box for {type(shape)}")


# ============================================================================
# LAYER 6: ADVANCED ALGORITHMS
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(points: ndarray, dim: int) -> Union[Polygon, Mesh]",
    deterministic=True,
    doc="Compute convex hull of points",
)
def convex_hull(points: np.ndarray, dim: int = 2) -> Union[Polygon, Mesh]:
    """Compute convex hull of a set of points.

    Uses scipy.spatial.ConvexHull for robust computation.

    Args:
        points: Array of points with shape [N, 2] for 2D or [N, 3] for 3D
        dim: Dimension (2 for 2D, 3 for 3D)

    Returns:
        Polygon for 2D hull, Mesh for 3D hull

    Example:
        # 2D convex hull
        points = np.random.rand(100, 2)
        hull = convex_hull(points, dim=2)

        # 3D convex hull
        points_3d = np.random.rand(100, 3)
        hull_3d = convex_hull(points_3d, dim=3)
    """
    try:
        from scipy.spatial import ConvexHull as ScipyConvexHull
    except ImportError:
        raise ImportError("scipy is required for convex_hull. Install with: pip install scipy")

    points = np.asarray(points)

    # Check for NaN or infinity values
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        raise ValueError("Points contain NaN or infinity values")

    if dim == 2:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"For 2D hull, points must have shape [N, 2], got {points.shape}")

        if len(points) < 3:
            raise ValueError(f"2D convex hull requires at least 3 points, got {len(points)}")

        hull = ScipyConvexHull(points)
        # Extract vertices in order
        hull_vertices = points[hull.vertices]
        return Polygon(vertices=hull_vertices)

    elif dim == 3:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"For 3D hull, points must have shape [N, 3], got {points.shape}")

        if len(points) < 4:
            raise ValueError(f"3D convex hull requires at least 4 points, got {len(points)}")

        hull = ScipyConvexHull(points)
        return Mesh(vertices=points, faces=hull.simplices.astype(np.int32))

    else:
        raise ValueError(f"Dimension must be 2 or 3, got {dim}")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(points: ndarray) -> Mesh",
    deterministic=True,
    doc="Compute Delaunay triangulation of 2D points",
)
def delaunay_triangulation(points: np.ndarray) -> Mesh:
    """Compute Delaunay triangulation of 2D points.

    Uses scipy.spatial.Delaunay for robust computation.

    Args:
        points: Array of 2D points with shape [N, 2]

    Returns:
        Mesh with triangulated faces (embedded in 3D with z=0)

    Example:
        points = np.random.rand(100, 2)
        tri = delaunay_triangulation(points)
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError("scipy is required for delaunay_triangulation. Install with: pip install scipy")

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Points must have shape [N, 2], got {points.shape}")

    if len(points) < 3:
        raise ValueError(f"Delaunay triangulation requires at least 3 points, got {len(points)}")

    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        raise ValueError("Points contain NaN or infinity values")

    # Compute Delaunay triangulation
    tri = Delaunay(points)

    # Convert to 3D vertices (z=0)
    vertices_3d = np.zeros((len(points), 3))
    vertices_3d[:, :2] = points

    return Mesh(vertices=vertices_3d, faces=tri.simplices.astype(np.int32))


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(points: ndarray) -> Tuple[ndarray, ndarray, ndarray]",
    deterministic=True,
    doc="Compute Voronoi diagram of 2D points",
)
def voronoi(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Compute Voronoi diagram of 2D points.

    Uses scipy.spatial.Voronoi for robust computation.

    Args:
        points: Array of 2D points with shape [N, 2]

    Returns:
        Tuple of (vertices, ridge_points, ridge_vertices) where:
        - vertices: Voronoi vertices [M, 2]
        - ridge_points: Indices of input points forming each ridge [K, 2]
        - ridge_vertices: Indices of Voronoi vertices forming each ridge (list of lists)

    Example:
        points = np.random.rand(50, 2)
        vertices, ridge_points, ridge_vertices = voronoi(points)
    """
    try:
        from scipy.spatial import Voronoi as ScipyVoronoi
    except ImportError:
        raise ImportError("scipy is required for voronoi. Install with: pip install scipy")

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Points must have shape [N, 2], got {points.shape}")

    if len(points) < 4:
        raise ValueError(f"Voronoi diagram requires at least 4 points, got {len(points)}")

    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        raise ValueError("Points contain NaN or infinity values")

    vor = ScipyVoronoi(points)

    return (vor.vertices, vor.ridge_points, vor.ridge_vertices)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(points: ndarray) -> Dict[str, Any]",
    deterministic=True,
    doc="Compute Voronoi diagram of 2D points (dict format)",
)
def voronoi_diagram(points: np.ndarray) -> Dict[str, Any]:
    """Compute Voronoi diagram of 2D points, returning dict format.

    Convenience wrapper around voronoi() that returns results as a dictionary.

    Args:
        points: Array of 2D points with shape [N, 2]

    Returns:
        Dictionary with keys:
        - 'vertices': Voronoi vertices [M, 2]
        - 'ridge_points': Indices of input points forming each ridge [K, 2]
        - 'regions': List of vertex indices for each region

    Example:
        points = np.random.rand(50, 2)
        vor = voronoi_diagram(points)
        vertices = vor['vertices']
    """
    vertices, ridge_points, ridge_vertices = voronoi(points)

    # Convert ridge_vertices to regions
    # Note: scipy's voronoi doesn't directly give us regions per point,
    # we use ridge_vertices as a proxy
    return {
        'vertices': vertices,
        'ridge_points': ridge_points,
        'regions': ridge_vertices
    }


@operator(
    domain="geometry",
    category=OpCategory.COMPOSE,
    signature="(mesh_a: Mesh, mesh_b: Mesh) -> Mesh",
    deterministic=True,
    doc="Compute union of two meshes (boolean operation)",
)
def mesh_union(mesh_a: Mesh, mesh_b: Mesh) -> Mesh:
    """Compute union of two meshes using boolean operations.

    Note: This is a placeholder implementation that requires an external
    library like trimesh or PyMesh for robust mesh boolean operations.
    For MVP, this simply combines the meshes without intersection handling.

    Args:
        mesh_a: First mesh
        mesh_b: Second mesh

    Returns:
        Union mesh

    Example:
        box = box3d(point3d(0, 0, 0), 1, 1, 1)
        sphere_mesh = sphere(point3d(0.5, 0.5, 0.5), 0.7)
        # Note: would need mesh conversion for sphere
        # union = mesh_union(box_mesh, sphere_mesh)
    """
    # Simple implementation: combine vertices and faces
    # Offset face indices for second mesh
    combined_vertices = np.vstack([mesh_a.vertices, mesh_b.vertices])
    offset_faces_b = mesh_b.faces + len(mesh_a.vertices)
    combined_faces = np.vstack([mesh_a.faces, offset_faces_b])

    # Combine normals if both have them
    combined_normals = None
    if mesh_a.normals is not None and mesh_b.normals is not None:
        combined_normals = np.vstack([mesh_a.normals, mesh_b.normals])

    return Mesh(vertices=combined_vertices, faces=combined_faces, normals=combined_normals)


@operator(
    domain="geometry",
    category=OpCategory.COMPOSE,
    signature="(mesh_a: Mesh, mesh_b: Mesh) -> Mesh",
    deterministic=True,
    doc="Compute intersection of two meshes (boolean operation)",
)
def mesh_intersection(mesh_a: Mesh, mesh_b: Mesh) -> Mesh:
    """Compute intersection of two meshes using boolean operations.

    Note: This requires external libraries like trimesh for robust implementation.
    This is a placeholder that raises NotImplementedError.

    Args:
        mesh_a: First mesh
        mesh_b: Second mesh

    Returns:
        Intersection mesh

    Raises:
        NotImplementedError: Robust mesh intersection requires external libraries
    """
    raise NotImplementedError(
        "Mesh intersection requires external libraries like trimesh. "
        "Install with: pip install trimesh"
    )


@operator(
    domain="geometry",
    category=OpCategory.COMPOSE,
    signature="(mesh_a: Mesh, mesh_b: Mesh) -> Mesh",
    deterministic=True,
    doc="Compute difference of two meshes (boolean operation)",
)
def mesh_difference(mesh_a: Mesh, mesh_b: Mesh) -> Mesh:
    """Compute difference of two meshes (A - B) using boolean operations.

    Note: This requires external libraries like trimesh for robust implementation.
    This is a placeholder that raises NotImplementedError.

    Args:
        mesh_a: First mesh
        mesh_b: Second mesh

    Returns:
        Difference mesh (A - B)

    Raises:
        NotImplementedError: Robust mesh difference requires external libraries
    """
    raise NotImplementedError(
        "Mesh difference requires external libraries like trimesh. "
        "Install with: pip install trimesh"
    )


# ============================================================================
# LAYER 7: FIELD DOMAIN INTEGRATION
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(field: Any, point: Union[Point2D, Point3D]) -> float",
    deterministic=True,
    doc="Sample field value at a geometric point",
)
def sample_field_at_point(field: Any, point: Union[Point2D, Point3D]) -> float:
    """Sample a field value at a geometric point.

    Integrates with the field domain to query spatial fields at geometric locations.

    Args:
        field: Field2D object from field domain
        point: Point2D or Point3D to sample at

    Returns:
        Interpolated field value at point

    Example:
        from morphogen.stdlib.field import Field2D
        import numpy as np

        # Create a field
        data = np.random.rand(100, 100)
        field = Field2D(data, dx=0.1, dy=0.1)

        # Sample at a point
        value = sample_field_at_point(field, point2d(5.0, 5.0))
    """
    # Import field module to avoid circular dependency
    try:
        from morphogen.stdlib.field import Field2D
    except ImportError:
        raise ImportError("Field domain is required for field sampling")

    if not isinstance(field, Field2D):
        raise TypeError(f"Expected Field2D, got {type(field)}")

    # Convert point to grid coordinates
    if isinstance(point, Point2D):
        # Map point to grid indices
        x_idx = point.x / field.dx
        y_idx = point.y / field.dy

        # Clamp to field bounds
        x_idx = np.clip(x_idx, 0, field.width - 1)
        y_idx = np.clip(y_idx, 0, field.height - 1)

        # Bilinear interpolation
        x0, x1 = int(np.floor(x_idx)), int(np.ceil(x_idx))
        y0, y1 = int(np.floor(y_idx)), int(np.ceil(y_idx))

        # Handle edge case where point is exactly on boundary
        if x0 == x1:
            x1 = min(x0 + 1, field.width - 1)
        if y0 == y1:
            y1 = min(y0 + 1, field.height - 1)

        # Interpolation weights
        wx = x_idx - x0
        wy = y_idx - y0

        # Bilinear interpolation
        v00 = field.data[y0, x0] if field.data.ndim == 2 else field.data[y0, x0, 0]
        v10 = field.data[y0, x1] if field.data.ndim == 2 else field.data[y0, x1, 0]
        v01 = field.data[y1, x0] if field.data.ndim == 2 else field.data[y1, x0, 0]
        v11 = field.data[y1, x1] if field.data.ndim == 2 else field.data[y1, x1, 0]

        value = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11

        return float(value)

    else:
        raise NotImplementedError("3D field sampling not yet implemented")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(field: Any, shape: Union[Circle, Rectangle, Polygon]) -> ndarray",
    deterministic=True,
    doc="Query field values within a geometric region",
)
def query_field_in_region(field: Any, shape: Union[Circle, Rectangle, Polygon]) -> np.ndarray:
    """Query all field values within a geometric region.

    Args:
        field: Field2D object from field domain
        shape: Geometric shape defining the query region

    Returns:
        Array of (x, y, value) tuples for points inside the shape

    Example:
        # Query field values inside a circle
        circ = circle(center=point2d(5.0, 5.0), radius=2.0)
        values = query_field_in_region(field, circ)
    """
    try:
        from morphogen.stdlib.field import Field2D
    except ImportError:
        raise ImportError("Field domain is required for field queries")

    if not isinstance(field, Field2D):
        raise TypeError(f"Expected Field2D, got {type(field)}")

    # Get bounding box of shape
    bbox = bounding_box(shape)

    # Convert bounding box to grid coordinates
    x_min = int(np.floor(bbox.min_point.x / field.dx))
    x_max = int(np.ceil(bbox.max_point.x / field.dx))
    y_min = int(np.floor(bbox.min_point.y / field.dy))
    y_max = int(np.ceil(bbox.max_point.y / field.dy))

    # Clamp to field bounds
    x_min = max(0, x_min)
    x_max = min(field.width, x_max)
    y_min = max(0, y_min)
    y_max = min(field.height, y_max)

    # Collect values inside shape
    results = []

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            # Convert grid coordinate to world coordinate
            world_x = x * field.dx
            world_y = y * field.dy
            point = Point2D(x=world_x, y=world_y)

            # Check if point is inside shape
            is_inside = False
            if isinstance(shape, Circle):
                is_inside = contains_circle_point(shape, point)
            elif isinstance(shape, Rectangle):
                is_inside = contains_rectangle_point(shape, point)
            elif isinstance(shape, Polygon):
                is_inside = contains_polygon_point(shape, point)

            if is_inside:
                value = field.data[y, x] if field.data.ndim == 2 else field.data[y, x, 0]
                results.append([world_x, world_y, float(value)])

    return np.array(results) if results else np.empty((0, 3))


# ============================================================================
# LAYER 8: RIGIDBODY DOMAIN INTEGRATION
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(shape: Union[Circle, Rectangle, Polygon, Box3D, Sphere]) -> Dict",
    deterministic=True,
    doc="Convert geometric shape to rigidbody collision shape parameters",
)
def shape_to_rigidbody(shape: Union[Circle, Rectangle, Polygon, Box3D, Sphere]) -> Dict[str, Any]:
    """Convert a geometric shape to rigidbody collision shape parameters.

    Integrates with the rigidbody domain to create physics collision shapes
    from geometric primitives.

    Args:
        shape: Geometric shape (Circle, Rectangle, Polygon, Box3D, or Sphere)

    Returns:
        Dictionary with 'shape_type' and 'shape_params' for rigidbody creation

    Example:
        from morphogen.stdlib.rigidbody import RigidBody2D, ShapeType

        # Create geometric circle
        circ = circle(center=point2d(0, 0), radius=5.0)

        # Convert to rigidbody shape
        shape_data = shape_to_rigidbody(circ)

        # Create rigidbody with this shape
        body = RigidBody2D(
            position=np.array([0.0, 0.0]),
            mass=1.0,
            shape_type=shape_data['shape_type'],
            shape_params=shape_data['shape_params']
        )
    """
    try:
        from morphogen.stdlib.rigidbody import ShapeType
    except ImportError:
        raise ImportError("Rigidbody domain is required for shape conversion")

    if isinstance(shape, Circle):
        return {"shape_type": ShapeType.CIRCLE, "shape_params": {"radius": shape.radius}}

    elif isinstance(shape, Rectangle):
        return {
            "shape_type": ShapeType.BOX,
            "shape_params": {"width": shape.width, "height": shape.height, "rotation": shape.rotation},
        }

    elif isinstance(shape, Polygon):
        return {"shape_type": ShapeType.POLYGON, "shape_params": {"vertices": shape.vertices.copy()}}

    elif isinstance(shape, (Box3D, Sphere)):
        raise NotImplementedError("3D rigidbody shapes not yet supported in 2D rigidbody domain")

    else:
        raise TypeError(f"Cannot convert {type(shape)} to rigidbody shape")


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(mesh: Mesh) -> Mesh",
    deterministic=True,
    doc="Generate collision mesh from high-poly mesh",
)
def collision_mesh(mesh: Mesh, target_faces: int = 100) -> Mesh:
    """Generate a simplified collision mesh from a high-polygon mesh.

    Creates a lower-polygon version suitable for physics collision detection.

    Note: This is a placeholder. Robust mesh simplification requires external
    libraries like trimesh or Open3D.

    Args:
        mesh: High-polygon input mesh (or Box3D primitive)
        target_faces: Target number of faces for collision mesh

    Returns:
        Simplified collision mesh

    Example:
        # High-poly mesh
        high_poly = mesh(vertices=..., faces=...)

        # Generate collision mesh
        collision = collision_mesh(high_poly, target_faces=50)
    """
    # Handle Box3D primitives by converting to vertices
    if isinstance(mesh, Box3D):
        vertices = mesh.get_vertices()
    elif hasattr(mesh, 'vertices'):
        vertices = mesh.vertices
    else:
        raise TypeError(f"Expected Mesh or Box3D, got {type(mesh)}")

    # Simple implementation: compute convex hull as collision mesh
    # This is a safe, conservative collision shape
    hull_mesh = convex_hull(vertices, dim=3)

    # If hull has fewer faces than target, return it
    if isinstance(hull_mesh, Mesh) and hull_mesh.num_faces <= target_faces:
        return hull_mesh

    # Otherwise, return convex hull (proper decimation requires external libraries)
    return hull_mesh if isinstance(hull_mesh, Mesh) else mesh


# ============================================================================
# EXPORTS (for registry discovery)
# ============================================================================

# Primitives
__all__ = [
    # Types
    "Point2D",
    "Point3D",
    "Line2D",
    "Circle",
    "Rectangle",
    "Polygon",
    "BoundingBox",
    "Box3D",
    "Sphere",
    "Mesh",
    "CoordinateFrame",
    # Construction
    "point2d",
    "point3d",
    "line2d",
    "circle",
    "rectangle",
    "polygon",
    "regular_polygon",
    "box3d",
    "sphere",
    "mesh",
    # Transformations
    "translate_point2d",
    "translate_circle",
    "translate_rectangle",
    "translate_polygon",
    "rotate_point2d",
    "rotate_circle",
    "rotate_rectangle",
    "rotate_polygon",
    "scale_circle",
    "scale_rectangle",
    "scale_polygon",
    # Spatial queries
    "distance_point_point",
    "distance_point_line",
    "distance_point_circle",
    "intersect_circle_circle",
    "contains",
    "contains_circle_point",
    "contains_polygon_point",
    "contains_rectangle_point",
    "closest_point_circle",
    "closest_point_line",
    # Coordinate conversions
    "cartesian_to_polar",
    "polar_to_cartesian",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    # Properties
    "area",
    "perimeter",
    "centroid",
    "bounding_box",
    # Advanced algorithms
    "convex_hull",
    "delaunay_triangulation",
    "voronoi",
    "voronoi_diagram",
    "mesh_union",
    "mesh_intersection",
    "mesh_difference",
    # Field integration
    "sample_field_at_point",
    "query_field_in_region",
    # Rigidbody integration
    "shape_to_rigidbody",
    "collision_mesh",
]
