"""
Spatial Domain Transform Interfaces

Interfaces for affine transformations and coordinate system conversions.
"""

from typing import Any, Dict, Optional, Tuple, Type
import numpy as np

from .base import DomainInterface


class SpatialAffineInterface(DomainInterface):
    """
    Spatial → Spatial via Affine transformations (translate, rotate, scale, shear).

    Affine transformations preserve:
    - Points, straight lines, and planes
    - Parallel lines remain parallel
    - Ratios of distances along lines

    Useful for:
    - Image/geometry registration
    - Data augmentation
    - Coordinate system alignment
    - Field transformations
    """

    source_domain = "spatial"
    target_domain = "spatial"

    def __init__(self, data: np.ndarray,
                 translate: Optional[Tuple[float, float]] = None,
                 rotate: Optional[float] = None,
                 scale: Optional[Tuple[float, float]] = None,
                 shear: Optional[float] = None,
                 order: int = 1,
                 metadata: Optional[Dict] = None):
        """
        Initialize affine transform.

        Args:
            data: 2D spatial data (image or field)
            translate: Translation (dx, dy) in pixels
            rotate: Rotation angle in degrees (counter-clockwise)
            scale: Scale factors (sx, sy)
            shear: Shear angle in degrees
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            metadata: Optional metadata dict
        """
        super().__init__(data, metadata)
        self.translate = translate or (0, 0)
        self.rotate = rotate or 0
        self.scale = scale or (1, 1)
        self.shear = shear or 0
        self.order = order

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to spatial data.

        Args:
            data: 2D spatial data

        Returns:
            Transformed spatial data (same shape as input)
        """
        from scipy.ndimage import affine_transform

        if data.ndim not in [2, 3]:
            raise ValueError(f"Data must be 2D or 3D (2D + channels), got shape {data.shape}")

        # Get image center for rotation/scaling
        height, width = data.shape[:2]
        center_y, center_x = height / 2.0, width / 2.0

        # Build transformation matrix (applies transformations in order: scale, rotate, shear, translate)
        # We use homogeneous coordinates for easier composition

        # Start with identity
        matrix = np.eye(3)

        # Translate to origin (for rotation/scaling around center)
        T1 = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ])

        # Apply scale
        sx, sy = self.scale
        S = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])

        # Apply rotation (counter-clockwise)
        angle_rad = np.radians(self.rotate)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

        # Apply shear
        shear_rad = np.radians(self.shear)
        SH = np.array([
            [1, np.tan(shear_rad), 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Translate back from origin
        T2 = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]
        ])

        # Apply user translation
        dx, dy = self.translate
        T3 = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])

        # Compose transformations: T3 * T2 * SH * R * S * T1
        matrix = T3 @ T2 @ SH @ R @ S @ T1

        # Extract 2x2 matrix and offset for scipy
        # Note: scipy uses (y, x) indexing, so we need to swap
        M = matrix[:2, :2]
        offset = matrix[:2, 2]

        # scipy.ndimage.affine_transform uses inverse mapping (output -> input)
        # So we need to invert the matrix
        M_inv = np.linalg.inv(M)
        # Compute inverse offset: -M_inv @ offset
        offset_inv = -M_inv @ offset

        # Apply transformation
        if data.ndim == 2:
            transformed = affine_transform(data, M_inv, offset=offset_inv,
                                          order=self.order, mode='constant', cval=0)
        else:
            # Handle multi-channel data
            channels = []
            for i in range(data.shape[2]):
                channel = affine_transform(data[:, :, i], M_inv, offset=offset_inv,
                                          order=self.order, mode='constant', cval=0)
                channels.append(channel)
            transformed = np.stack(channels, axis=2)

        return transformed.astype(data.dtype)

    def validate(self) -> bool:
        """Validate data is 2D or 3D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Data must be numpy array")

        if self.source_data.ndim not in [2, 3]:
            raise ValueError("Data must be 2D or 3D (2D + channels)")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'spatial_data': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'transformed_spatial_data': np.ndarray}


class CartesianToPolarInterface(DomainInterface):
    """
    Cartesian → Polar coordinate conversion.

    Converts (x, y) Cartesian coordinates to (r, theta) polar coordinates.

    Useful for:
    - Radial pattern analysis
    - Rotational symmetry detection
    - Circular/angular data visualization
    - Fourier-Bessel transforms
    """

    source_domain = "cartesian"
    target_domain = "polar"

    def __init__(self, data: np.ndarray, center: Optional[Tuple[float, float]] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize Cartesian to Polar transform.

        Args:
            data: 2D field in Cartesian coordinates
            center: Origin for polar conversion (cx, cy). If None, uses image center.
            metadata: Optional metadata dict
        """
        super().__init__(data, metadata)
        self.center = center

    def transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Cartesian field to polar coordinates.

        Args:
            data: 2D field in Cartesian coordinates

        Returns:
            Tuple of (radius_array, angle_array) in polar coordinates
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        height, width = data.shape

        # Determine center
        if self.center is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = self.center

        # Create coordinate grids
        y, x = np.indices(data.shape, dtype=np.float32)

        # Convert to polar
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)  # Range: [-pi, pi]

        return r, theta

    def validate(self) -> bool:
        """Validate data is 2D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Data must be numpy array")

        if self.source_data.ndim != 2:
            raise ValueError("Data must be 2D array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'cartesian_data': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'radius': np.ndarray, 'angle': np.ndarray}


class PolarToCartesianInterface(DomainInterface):
    """
    Polar → Cartesian coordinate conversion.

    Converts (r, theta) polar coordinates back to (x, y) Cartesian coordinates.

    Useful for:
    - Reconstructing fields after radial processing
    - Visualization of polar data
    - Inverse transforms after polar filtering
    """

    source_domain = "polar"
    target_domain = "cartesian"

    def __init__(self, radius: np.ndarray, angle: np.ndarray,
                 output_shape: Tuple[int, int],
                 center: Optional[Tuple[float, float]] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize Polar to Cartesian transform.

        Args:
            radius: Radius values (2D array)
            angle: Angle values in radians (2D array)
            output_shape: Desired output shape (height, width)
            center: Origin for conversion (cx, cy). If None, uses image center.
            metadata: Optional metadata dict
        """
        super().__init__(radius, metadata)
        self.angle = angle
        self.output_shape = output_shape
        self.center = center

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convert polar field back to Cartesian coordinates.

        Args:
            data: Values in polar space (typically same shape as radius/angle)

        Returns:
            2D field in Cartesian coordinates
        """
        from scipy.interpolate import griddata

        # Use source_data (radius) for coordinate conversion
        radius = self.source_data if self.source_data is not None else data

        if data.shape != radius.shape:
            raise ValueError(f"Data shape {data.shape} must match radius shape {radius.shape}")

        height, width = self.output_shape

        # Determine center
        if self.center is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = self.center

        # Convert polar to Cartesian coordinates
        x_polar = radius * np.cos(self.angle) + cx
        y_polar = radius * np.sin(self.angle) + cy

        # Flatten for interpolation
        points = np.column_stack([x_polar.ravel(), y_polar.ravel()])
        values = data.ravel()

        # Create output grid
        y_out, x_out = np.indices((height, width), dtype=np.float32)
        grid_points = np.column_stack([x_out.ravel(), y_out.ravel()])

        # Interpolate
        cartesian = griddata(points, values, grid_points, method='linear', fill_value=0)
        cartesian = cartesian.reshape(height, width)

        return cartesian.astype(np.float32)

    def validate(self) -> bool:
        """Validate radius and angle arrays."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Radius must be numpy array")

        if not isinstance(self.angle, np.ndarray):
            raise TypeError("Angle must be numpy array")

        if self.source_data.shape != self.angle.shape:
            raise ValueError("Radius and angle must have same shape")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'radius': np.ndarray, 'angle': np.ndarray, 'values': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'cartesian_data': np.ndarray}
