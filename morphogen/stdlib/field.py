"""Field operations implementation using NumPy backend.

This module provides NumPy-based implementations of all core field operations
for the MVP, including advection, diffusion, projection, and boundary conditions.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np

from morphogen.core.operator import operator, OpCategory


class Field2D:
    """2D field with NumPy backend.

    Represents a dense 2D grid with scalar or vector values.
    """

    def __init__(self, data: np.ndarray, dx: float = 1.0, dy: float = 1.0):
        """Initialize field.

        Args:
            data: NumPy array of field values (shape: (height, width) or (height, width, channels))
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
        """
        self.data = data
        self.dx = dx
        self.dy = dy
        self.shape = data.shape[:2]  # (height, width)

    @property
    def height(self) -> int:
        """Get field height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get field width."""
        return self.shape[1]

    def copy(self) -> 'Field2D':
        """Create a copy of this field."""
        return Field2D(self.data.copy(), self.dx, self.dy)

    def map(self, func: Callable) -> 'Field2D':
        """Apply function to each element of field, optionally with coordinates.

        Supports two calling conventions:
        - func(value) -> value: Simple per-element transformation
        - func(value, x, y) -> value: Position-aware transformation

        Args:
            func: Function to apply. If it has 3 parameters (value, x, y),
                  coordinates will be passed. Otherwise, just the value.

        Returns:
            New Field2D with transformed values
        """
        result = self.copy()

        # Detect if function expects position arguments
        # Check for LambdaFunction from runtime or use inspect for Python callables
        is_position_aware = False

        if hasattr(func, 'params'):
            # LambdaFunction from Morphogen runtime
            is_position_aware = len(func.params) >= 3
        else:
            # Regular Python callable - check signature
            import inspect
            try:
                sig = inspect.signature(func)
                is_position_aware = len(sig.parameters) >= 3
            except (ValueError, TypeError):
                # Can't inspect, assume value-only
                is_position_aware = False

        if is_position_aware:
            # Create coordinate grids
            height, width = self.shape
            y_coords, x_coords = np.meshgrid(
                np.arange(height),
                np.arange(width),
                indexing='ij'
            )

            # Apply function element-wise with coordinates
            # Use vectorize for runtime lambdas, direct call for numpy-compatible funcs
            if hasattr(func, 'params'):
                # Runtime lambda - must call per-element
                new_data = np.zeros_like(self.data)
                for y in range(height):
                    for x in range(width):
                        new_data[y, x] = func(self.data[y, x], x, y)
                result.data = new_data
            else:
                # Try vectorized first, fall back to element-wise
                try:
                    result.data = func(self.data, x_coords, y_coords)
                except (TypeError, ValueError):
                    new_data = np.zeros_like(self.data)
                    for y in range(height):
                        for x in range(width):
                            new_data[y, x] = func(self.data[y, x], x, y)
                    result.data = new_data
        else:
            # Simple value-only mapping
            if callable(func):
                try:
                    result.data = func(self.data)
                except TypeError:
                    # Might need element-wise application
                    result.data = np.vectorize(func)(self.data)

        return result

    # Arithmetic operators for Field2D
    def __add__(self, other) -> 'Field2D':
        """Element-wise addition."""
        if isinstance(other, Field2D):
            return Field2D(self.data + other.data, self.dx, self.dy)
        return Field2D(self.data + other, self.dx, self.dy)

    def __radd__(self, other) -> 'Field2D':
        """Right addition."""
        return Field2D(other + self.data, self.dx, self.dy)

    def __sub__(self, other) -> 'Field2D':
        """Element-wise subtraction."""
        if isinstance(other, Field2D):
            return Field2D(self.data - other.data, self.dx, self.dy)
        return Field2D(self.data - other, self.dx, self.dy)

    def __rsub__(self, other) -> 'Field2D':
        """Right subtraction."""
        return Field2D(other - self.data, self.dx, self.dy)

    def __mul__(self, other) -> 'Field2D':
        """Element-wise multiplication."""
        if isinstance(other, Field2D):
            return Field2D(self.data * other.data, self.dx, self.dy)
        return Field2D(self.data * other, self.dx, self.dy)

    def __rmul__(self, other) -> 'Field2D':
        """Right multiplication."""
        return Field2D(other * self.data, self.dx, self.dy)

    def __truediv__(self, other) -> 'Field2D':
        """Element-wise division."""
        if isinstance(other, Field2D):
            return Field2D(self.data / other.data, self.dx, self.dy)
        return Field2D(self.data / other, self.dx, self.dy)

    def __rtruediv__(self, other) -> 'Field2D':
        """Right division."""
        return Field2D(other / self.data, self.dx, self.dy)

    def __neg__(self) -> 'Field2D':
        """Negation."""
        return Field2D(-self.data, self.dx, self.dy)

    def __repr__(self) -> str:
        return f"Field2D(shape={self.shape}, dtype={self.data.dtype})"


class FieldOperations:
    """Namespace for field operations (accessed as 'field' in DSL)."""

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], dtype: type, fill_value: float, dx: float, dy: float) -> Field2D",
        deterministic=True,
        doc="Allocate a new field"
    )
    def alloc(shape: Tuple[int, int], dtype: type = np.float32,
              fill_value: float = 0.0, dx: float = 1.0, dy: float = 1.0) -> Field2D:
        """Allocate a new field.

        Args:
            shape: Field shape (height, width)
            dtype: Data type
            fill_value: Initial value
            dx: Grid spacing in x
            dy: Grid spacing in y

        Returns:
            New field filled with fill_value
        """
        data = np.full(shape, fill_value, dtype=dtype)
        return Field2D(data, dx, dy)

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, velocity: Field2D, dt: float, method: str) -> Field2D",
        deterministic=True,
        doc="Advect field by velocity field"
    )
    def advect(field: Field2D, velocity: Field2D, dt: float,
               method: str = "semi_lagrangian") -> Field2D:
        """Advect field by velocity field.

        Uses semi-Lagrangian advection (backward trace + interpolation).

        Args:
            field: Field to advect
            velocity: Velocity field (2-channel: vx, vy)
            dt: Timestep
            method: Advection method ("semi_lagrangian" only for MVP)

        Returns:
            Advected field
        """
        if method != "semi_lagrangian":
            raise NotImplementedError(f"Advection method '{method}' not implemented in MVP")

        h, w = field.shape
        result = field.copy()

        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Backward trace
        if len(velocity.data.shape) == 3 and velocity.data.shape[2] == 2:
            # Vector velocity field
            vx = velocity.data[:, :, 0]
            vy = velocity.data[:, :, 1]
        else:
            raise ValueError("Velocity field must be 2-channel (vx, vy)")

        # Trace back to source positions
        src_x = x - vx * dt / field.dx
        src_y = y - vy * dt / field.dy

        # Clamp to field boundaries
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        # Bilinear interpolation
        # Get integer and fractional parts
        x0 = np.floor(src_x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(src_y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = src_x - x0
        fy = src_y - y0

        # Interpolate
        if len(field.data.shape) == 2:
            # Scalar field
            result.data = (
                field.data[y0, x0] * (1 - fx) * (1 - fy) +
                field.data[y0, x1] * fx * (1 - fy) +
                field.data[y1, x0] * (1 - fx) * fy +
                field.data[y1, x1] * fx * fy
            )
        else:
            # Vector field
            for c in range(field.data.shape[2]):
                result.data[:, :, c] = (
                    field.data[y0, x0, c] * (1 - fx) * (1 - fy) +
                    field.data[y0, x1, c] * fx * (1 - fy) +
                    field.data[y1, x0, c] * (1 - fx) * fy +
                    field.data[y1, x1, c] * fx * fy
                )

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, rate: float, dt: float, method: str, iterations: int) -> Field2D",
        deterministic=True,
        doc="Diffuse field using implicit solver"
    )
    def diffuse(field: Field2D, rate: float, dt: float,
                method: str = "jacobi", iterations: int = 20) -> Field2D:
        """Diffuse field using implicit solver.

        Solves: (I - α∇²) x = x₀
        where α = rate * dt

        Args:
            field: Field to diffuse
            rate: Diffusion rate
            dt: Timestep
            method: Solver method ("jacobi" only for MVP)
            iterations: Number of solver iterations

        Returns:
            Diffused field
        """
        if method != "jacobi":
            raise NotImplementedError(f"Diffusion method '{method}' not implemented in MVP")

        alpha = rate * dt
        h, w = field.shape

        # Jacobi iteration: x^(k+1) = (x₀ + α * neighbors) / (1 + 4α)
        result = field.copy()
        x0 = field.data.copy()

        for _ in range(iterations):
            # Get neighbors (with boundary handling)
            left = np.roll(result.data, 1, axis=1)
            right = np.roll(result.data, -1, axis=1)
            up = np.roll(result.data, 1, axis=0)
            down = np.roll(result.data, -1, axis=0)

            # Jacobi update
            result.data = (x0 + alpha * (left + right + up + down)) / (1 + 4 * alpha)

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D) -> Field2D",
        deterministic=True,
        doc="Compute Laplacian of field using 5-point stencil"
    )
    def laplacian(field: Field2D) -> Field2D:
        """Compute Laplacian of field using 5-point stencil.

        Computes ∇²f = ∂²f/∂x² + ∂²f/∂y² using finite differences.
        Uses central differences: ∇²f ≈ (f_left + f_right + f_up + f_down - 4*f_center)

        Args:
            field: Field to compute Laplacian of

        Returns:
            Laplacian field (same shape as input)

        Notes:
            - Uses reflective boundary conditions (duplicates edge values)
            - For periodic boundaries, call field.boundary(result, spec="periodic") after
            - Laplacian is scale-dependent on grid spacing (dx, dy)
        """
        h, w = field.shape
        result = field.copy()

        # Get shifted versions (pad with edge values for reflective boundaries)
        if len(field.data.shape) == 2:
            # Scalar field
            data = field.data
            left = np.pad(data, ((0, 0), (1, 0)), mode='edge')[:, :-1]
            right = np.pad(data, ((0, 0), (0, 1)), mode='edge')[:, 1:]
            up = np.pad(data, ((1, 0), (0, 0)), mode='edge')[:-1, :]
            down = np.pad(data, ((0, 1), (0, 0)), mode='edge')[1:, :]

            # 5-point stencil: ∇²f = (neighbors - 4*center)
            result.data = (left + right + up + down - 4 * data)

        else:
            # Vector field - compute Laplacian per channel
            for c in range(field.data.shape[2]):
                data = field.data[:, :, c]
                left = np.pad(data, ((0, 0), (1, 0)), mode='edge')[:, :-1]
                right = np.pad(data, ((0, 0), (0, 1)), mode='edge')[:, 1:]
                up = np.pad(data, ((1, 0), (0, 0)), mode='edge')[:-1, :]
                down = np.pad(data, ((0, 1), (0, 0)), mode='edge')[1:, :]

                result.data[:, :, c] = (left + right + up + down - 4 * data)

        # Scale by grid spacing (for accurate derivatives)
        dx2 = field.dx ** 2
        dy2 = field.dy ** 2
        result.data = result.data / ((dx2 + dy2) / 2)  # Average spacing

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(velocity: Field2D, method: str, iterations: int, tolerance: float) -> Field2D",
        deterministic=True,
        doc="Make velocity field divergence-free (pressure projection)"
    )
    def project(velocity: Field2D, method: str = "jacobi",
                iterations: int = 20, tolerance: float = 1e-4) -> Field2D:
        """Make velocity field divergence-free (pressure projection).

        Solves for pressure p: ∇²p = ∇·v
        Then updates velocity: v = v - ∇p

        Args:
            velocity: Velocity field to project
            method: Solver method ("jacobi" only for MVP)
            iterations: Number of solver iterations
            tolerance: Convergence tolerance (not used in MVP)

        Returns:
            Divergence-free velocity field
        """
        if method != "jacobi":
            raise NotImplementedError(f"Projection method '{method}' not implemented in MVP")

        if velocity.data.shape[2] != 2:
            raise ValueError("Projection requires 2-channel velocity field")

        h, w = velocity.shape
        vx = velocity.data[:, :, 0]
        vy = velocity.data[:, :, 1]

        # Compute divergence
        div = np.zeros((h, w), dtype=np.float32)
        div[1:-1, 1:-1] = (
            (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * velocity.dx) +
            (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * velocity.dy)
        )

        # Solve for pressure: ∇²p = div
        pressure = np.zeros((h, w), dtype=np.float32)

        for _ in range(iterations):
            # Jacobi iteration for Poisson equation
            left = np.roll(pressure, 1, axis=1)
            right = np.roll(pressure, -1, axis=1)
            up = np.roll(pressure, 1, axis=0)
            down = np.roll(pressure, -1, axis=0)

            pressure[1:-1, 1:-1] = (left[1:-1, 1:-1] + right[1:-1, 1:-1] +
                                    up[1:-1, 1:-1] + down[1:-1, 1:-1] - div[1:-1, 1:-1]) / 4

        # Compute pressure gradient
        grad_px = np.zeros((h, w), dtype=np.float32)
        grad_py = np.zeros((h, w), dtype=np.float32)

        grad_px[:, 1:-1] = (pressure[:, 2:] - pressure[:, :-2]) / (2 * velocity.dx)
        grad_py[1:-1, :] = (pressure[2:, :] - pressure[:-2, :]) / (2 * velocity.dy)

        # Subtract gradient from velocity
        result = velocity.copy()
        result.data[:, :, 0] = vx - grad_px
        result.data[:, :, 1] = vy - grad_py

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field_a: Field2D, field_b: Field2D, operation: Union[str, Callable]) -> Field2D",
        deterministic=True,
        doc="Combine two fields element-wise"
    )
    def combine(field_a: Field2D, field_b: Field2D,
                operation: Union[str, Callable] = "add") -> Field2D:
        """Combine two fields element-wise.

        Args:
            field_a: First field
            field_b: Second field
            operation: Operation ("add", "sub", "mul", "div", "min", "max") or callable

        Returns:
            Combined field
        """
        if field_a.shape != field_b.shape:
            raise ValueError(f"Field shapes must match: {field_a.shape} vs {field_b.shape}")

        result = field_a.copy()

        if callable(operation):
            result.data = operation(field_a.data, field_b.data)
        elif operation == "add":
            result.data = field_a.data + field_b.data
        elif operation == "sub":
            result.data = field_a.data - field_b.data
        elif operation == "mul":
            result.data = field_a.data * field_b.data
        elif operation == "div":
            result.data = field_a.data / (field_b.data + 1e-10)  # avoid division by zero
        elif operation == "min":
            result.data = np.minimum(field_a.data, field_b.data)
        elif operation == "max":
            result.data = np.maximum(field_a.data, field_b.data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, func: Union[str, Callable]) -> Field2D",
        deterministic=True,
        doc="Apply function to each element of field"
    )
    def map(field: Field2D, func: Union[str, Callable]) -> Field2D:
        """Apply function to each element of field.

        Args:
            field: Input field
            func: Function to apply (callable or string name like "abs", "sin", "cos")

        Returns:
            Mapped field
        """
        result = field.copy()

        if callable(func):
            result.data = func(field.data)
        elif func == "abs":
            result.data = np.abs(field.data)
        elif func == "sin":
            result.data = np.sin(field.data)
        elif func == "cos":
            result.data = np.cos(field.data)
        elif func == "sqrt":
            result.data = np.sqrt(np.maximum(field.data, 0))
        elif func == "square":
            result.data = field.data ** 2
        elif func == "exp":
            result.data = np.exp(field.data)
        elif func == "log":
            result.data = np.log(np.maximum(field.data, 1e-10))
        else:
            raise ValueError(f"Unknown function: {func}")

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, spec: str) -> Field2D",
        deterministic=True,
        doc="Apply boundary conditions"
    )
    def boundary(field: Field2D, spec: str = "reflect") -> Field2D:
        """Apply boundary conditions.

        Args:
            field: Field to apply boundaries to
            spec: Boundary specification ("reflect" or "periodic")

        Returns:
            Field with boundaries applied
        """
        result = field.copy()

        if spec == "reflect":
            # Mirror boundaries (Neumann)
            result.data[0, :] = result.data[1, :]     # Top
            result.data[-1, :] = result.data[-2, :]   # Bottom
            result.data[:, 0] = result.data[:, 1]     # Left
            result.data[:, -1] = result.data[:, -2]   # Right

        elif spec == "periodic":
            # Wrap boundaries
            result.data[0, :] = result.data[-2, :]    # Top = Bottom-1
            result.data[-1, :] = result.data[1, :]    # Bottom = Top+1
            result.data[:, 0] = result.data[:, -2]    # Left = Right-1
            result.data[:, -1] = result.data[:, 1]    # Right = Left+1

        else:
            raise ValueError(f"Unknown boundary spec: {spec}")

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], seed: int, low: float, high: float) -> Field2D",
        deterministic=False,
        doc="Create field with random values"
    )
    def random(shape: Tuple[int, int], seed: int = 0,
               low: float = 0.0, high: float = 1.0) -> Field2D:
        """Create field with random values.

        Args:
            shape: Field shape (height, width)
            seed: Random seed for determinism
            low: Minimum value
            high: Maximum value

        Returns:
            Field with random values
        """
        rng = np.random.RandomState(seed)
        data = rng.uniform(low, high, size=shape).astype(np.float32)
        return Field2D(data)

    # ============================================================================
    # EXTENDED FIELD OPERATIONS (for graphics & procedural work)
    # ============================================================================

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.QUERY,
        signature="(field: Field2D) -> Tuple[Field2D, Field2D]",
        deterministic=True,
        doc="Compute gradient of scalar field"
    )
    def gradient(field: Field2D) -> Tuple[Field2D, Field2D]:
        """Compute gradient of scalar field.

        Returns spatial derivatives (∂f/∂x, ∂f/∂y).

        Args:
            field: Scalar field

        Returns:
            Tuple of (grad_x, grad_y) fields

        Example:
            >>> gx, gy = field.gradient(scalar_field)
        """
        # Compute gradients using numpy
        grad_y, grad_x = np.gradient(field.data, field.dy, field.dx)

        return Field2D(grad_x, field.dx, field.dy), Field2D(grad_y, field.dx, field.dy)

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.QUERY,
        signature="(velocity: Field2D) -> Field2D",
        deterministic=True,
        doc="Compute divergence of vector field"
    )
    def divergence(velocity: Field2D) -> Field2D:
        """Compute divergence of vector field.

        Divergence measures "outflow" at each point: ∇·v = ∂vx/∂x + ∂vy/∂y

        Args:
            velocity: Vector field (2-channel: vx, vy)

        Returns:
            Scalar divergence field

        Example:
            >>> div = field.divergence(velocity_field)
        """
        if velocity.data.shape[2] != 2:
            raise ValueError("Divergence requires 2-channel velocity field")

        vx = velocity.data[:, :, 0]
        vy = velocity.data[:, :, 1]

        # Compute partial derivatives
        dvx_dx = np.gradient(vx, velocity.dx, axis=1)
        dvy_dy = np.gradient(vy, velocity.dy, axis=0)

        # Divergence
        div = dvx_dx + dvy_dy

        return Field2D(div, velocity.dx, velocity.dy)

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.QUERY,
        signature="(velocity: Field2D) -> Field2D",
        deterministic=True,
        doc="Compute curl (vorticity) of 2D vector field"
    )
    def curl(velocity: Field2D) -> Field2D:
        """Compute curl (vorticity) of 2D vector field.

        Curl measures rotation at each point: ∇×v = ∂vy/∂x - ∂vx/∂y

        Args:
            velocity: Vector field (2-channel: vx, vy)

        Returns:
            Scalar curl field (z-component of 3D curl)

        Example:
            >>> vorticity = field.curl(velocity_field)
        """
        if velocity.data.shape[2] != 2:
            raise ValueError("Curl requires 2-channel velocity field")

        vx = velocity.data[:, :, 0]
        vy = velocity.data[:, :, 1]

        # Compute partial derivatives
        dvy_dx = np.gradient(vy, velocity.dx, axis=1)
        dvx_dy = np.gradient(vx, velocity.dy, axis=0)

        # Curl (z-component in 2D)
        curl_z = dvy_dx - dvx_dy

        return Field2D(curl_z, velocity.dx, velocity.dy)

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, iterations: int, method: str) -> Field2D",
        deterministic=True,
        doc="Smooth field using filtering"
    )
    def smooth(field: Field2D, iterations: int = 1, method: str = "gaussian") -> Field2D:
        """Smooth field using filtering.

        Args:
            field: Field to smooth
            iterations: Number of smoothing passes
            method: Smoothing method ("gaussian" or "box")

        Returns:
            Smoothed field

        Example:
            >>> smoothed = field.smooth(noisy_field, iterations=3)
        """
        from scipy import ndimage

        result = field.copy()

        for _ in range(iterations):
            if len(field.data.shape) == 2:
                # Scalar field
                if method == "gaussian":
                    result.data = ndimage.gaussian_filter(result.data, sigma=1.0)
                elif method == "box":
                    result.data = ndimage.uniform_filter(result.data, size=3)
                else:
                    raise ValueError(f"Unknown smoothing method: {method}")
            else:
                # Vector field - smooth each channel
                for c in range(field.data.shape[2]):
                    if method == "gaussian":
                        result.data[:, :, c] = ndimage.gaussian_filter(result.data[:, :, c], sigma=1.0)
                    elif method == "box":
                        result.data[:, :, c] = ndimage.uniform_filter(result.data[:, :, c], size=3)

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, target_min: float, target_max: float) -> Field2D",
        deterministic=True,
        doc="Normalize field values to target range"
    )
    def normalize(field: Field2D, target_min: float = 0.0, target_max: float = 1.0) -> Field2D:
        """Normalize field values to target range.

        Args:
            field: Input field
            target_min: Target minimum value
            target_max: Target maximum value

        Returns:
            Normalized field

        Example:
            >>> normalized = field.normalize(field, 0, 1)
        """
        result = field.copy()

        # Find current min/max
        current_min = np.min(result.data)
        current_max = np.max(result.data)

        # Normalize
        if current_max > current_min:
            result.data = (result.data - current_min) / (current_max - current_min)
            result.data = result.data * (target_max - target_min) + target_min
        else:
            result.data = np.full_like(result.data, target_min)

        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, threshold_value: float, low_value: float, high_value: float) -> Field2D",
        deterministic=True,
        doc="Threshold field values"
    )
    def threshold(field: Field2D, threshold_value: float,
                 low_value: float = 0.0, high_value: float = 1.0) -> Field2D:
        """Threshold field values.

        Args:
            field: Input field
            threshold_value: Threshold value
            low_value: Value for pixels below threshold
            high_value: Value for pixels above threshold

        Returns:
            Thresholded field

        Example:
            >>> binary = field.threshold(field, 0.5, 0.0, 1.0)
        """
        result = field.copy()
        result.data = np.where(field.data > threshold_value, high_value, low_value)
        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.QUERY,
        signature="(field: Field2D, positions: ndarray, method: str) -> ndarray",
        deterministic=True,
        doc="Sample field at arbitrary positions"
    )
    def sample(field: Field2D, positions: np.ndarray, method: str = "bilinear") -> np.ndarray:
        """Sample field at arbitrary positions.

        Args:
            field: Field to sample
            positions: Array of (y, x) positions (shape: (N, 2) or (H, W, 2))
            method: Interpolation method ("nearest" or "bilinear")

        Returns:
            Sampled values (shape matches positions minus last dimension)

        Example:
            >>> # Sample at specific points
            >>> positions = np.array([[10.5, 20.3], [50.1, 60.7]])
            >>> values = field.sample(field, positions)
        """
        from scipy import ndimage

        original_shape = positions.shape[:-1]
        positions_flat = positions.reshape(-1, 2)

        # Extract y, x coordinates
        y_coords = positions_flat[:, 0]
        x_coords = positions_flat[:, 1]

        # Determine interpolation order
        order = 0 if method == "nearest" else 1

        # Sample field
        if len(field.data.shape) == 2:
            # Scalar field
            sampled = ndimage.map_coordinates(field.data, [y_coords, x_coords],
                                             order=order, mode='reflect')
        else:
            # Vector field - sample each channel
            sampled = np.zeros((len(positions_flat), field.data.shape[2]), dtype=np.float32)
            for c in range(field.data.shape[2]):
                sampled[:, c] = ndimage.map_coordinates(field.data[:, :, c],
                                                       [y_coords, x_coords],
                                                       order=order, mode='reflect')

        # Reshape to original dimensions
        if len(field.data.shape) == 2:
            return sampled.reshape(original_shape)
        else:
            return sampled.reshape(*original_shape, field.data.shape[2])

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D, min_value: float, max_value: float) -> Field2D",
        deterministic=True,
        doc="Clamp field values to range"
    )
    def clamp(field: Field2D, min_value: float, max_value: float) -> Field2D:
        """Clamp field values to range.

        Args:
            field: Input field
            min_value: Minimum value
            max_value: Maximum value

        Returns:
            Clamped field
        """
        result = field.copy()
        result.data = np.clip(field.data, min_value, max_value)
        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.TRANSFORM,
        signature="(field: Field2D) -> Field2D",
        deterministic=True,
        doc="Compute absolute value of field"
    )
    def abs(field: Field2D) -> Field2D:
        """Compute absolute value of field.

        Args:
            field: Input field

        Returns:
            Field with absolute values
        """
        result = field.copy()
        result.data = np.abs(field.data)
        return result

    @staticmethod
    @operator(
        domain="field",
        category=OpCategory.QUERY,
        signature="(velocity: Field2D) -> Field2D",
        deterministic=True,
        doc="Compute magnitude of vector field"
    )
    def magnitude(velocity: Field2D) -> Field2D:
        """Compute magnitude of vector field.

        Args:
            velocity: Vector field (2-channel: vx, vy)

        Returns:
            Scalar magnitude field

        Example:
            >>> speed = field.magnitude(velocity_field)
        """
        if velocity.data.shape[2] != 2:
            raise ValueError("Magnitude requires 2-channel velocity field")

        vx = velocity.data[:, :, 0]
        vy = velocity.data[:, :, 1]

        mag = np.sqrt(vx**2 + vy**2)

        return Field2D(mag, velocity.dx, velocity.dy)


# Create singleton instance for use as 'field' namespace
field = FieldOperations()

# Define public API
__all__ = [
    'Field2D', 'FieldOperations', 'field',
    'alloc', 'random', 'map', 'combine', 'threshold', 'clamp', 'normalize',
    'abs', 'magnitude', 'gradient', 'divergence', 'curl', 'laplacian',
    'smooth', 'diffuse', 'advect', 'project', 'boundary', 'sample',
]

# Export operators for domain registry discovery
alloc = FieldOperations.alloc
random = FieldOperations.random
map = FieldOperations.map
combine = FieldOperations.combine
threshold = FieldOperations.threshold
clamp = FieldOperations.clamp
normalize = FieldOperations.normalize
abs = FieldOperations.abs
magnitude = FieldOperations.magnitude
gradient = FieldOperations.gradient
divergence = FieldOperations.divergence
curl = FieldOperations.curl
laplacian = FieldOperations.laplacian
smooth = FieldOperations.smooth
diffuse = FieldOperations.diffuse
advect = FieldOperations.advect
project = FieldOperations.project
boundary = FieldOperations.boundary
sample = FieldOperations.sample
