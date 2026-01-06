"""
Field ↔ Agent Domain Interfaces

Interfaces for bidirectional data flow between field and agent domains.
"""

from typing import Any, Dict, Tuple, Type
import numpy as np

from .base import DomainInterface


class FieldToAgentInterface(DomainInterface):
    """
    Field → Agent: Sample field values at agent positions.

    Use cases:
    - Flow field → force on particles
    - Temperature field → agent color/behavior
    - Density field → agent sensing
    """

    source_domain = "field"
    target_domain = "agent"

    def __init__(self, field, positions, property_name="value"):
        super().__init__(source_data=field)
        self.field = field
        self.positions = positions
        self.property_name = property_name

    def transform(self, source_data: Any) -> np.ndarray:
        """Sample field at agent positions."""
        field = source_data if source_data is not None else self.field

        # Handle different field types
        if hasattr(field, 'data'):
            field_data = field.data
        elif isinstance(field, np.ndarray):
            field_data = field
        else:
            raise TypeError(f"Unknown field type: {type(field)}")

        # Sample using bilinear interpolation
        sampled = self._sample_field(field_data, self.positions)
        return sampled

    def validate(self) -> bool:
        """Check field and positions are compatible."""
        if self.field is None or self.positions is None:
            return False

        # Check positions are 2D (Nx2)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError(
                f"Agent positions must be Nx2, got shape {self.positions.shape}"
            )

        return True

    def _sample_field(self, field_data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Sample field at positions using bilinear interpolation.

        Args:
            field_data: 2D or 3D array (H, W) or (H, W, C)
            positions: Nx2 array of (y, x) coordinates

        Returns:
            N-length array of sampled values (or NxC for vector fields)
        """
        from scipy.ndimage import map_coordinates

        # Ensure field_data is a numpy array (not memoryview)
        field_data = np.asarray(field_data)

        # Normalize positions to grid coordinates
        h, w = field_data.shape[:2]
        coords = positions.copy()

        # Clamp to valid range
        coords[:, 0] = np.clip(coords[:, 0], 0, h - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, w - 1)

        # Sample using scipy map_coordinates
        if field_data.ndim == 2:
            # Scalar field
            sampled = map_coordinates(
                field_data,
                [coords[:, 0], coords[:, 1]],
                order=1,  # Bilinear
                mode='nearest'
            )
        else:
            # Vector field - sample each component
            sampled = np.zeros((len(positions), field_data.shape[2]), dtype=field_data.dtype)
            for c in range(field_data.shape[2]):
                component_data = np.asarray(field_data[:, :, c])
                sampled[:, c] = map_coordinates(
                    component_data,
                    [coords[:, 0], coords[:, 1]],
                    order=1,
                    mode='nearest'
                )

        return sampled

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
            'positions': np.ndarray,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'sampled_values': np.ndarray,
        }


class AgentToFieldInterface(DomainInterface):
    """
    Agent → Field: Deposit agent properties onto field grid.

    Use cases:
    - Particle positions → density field
    - Agent velocities → velocity field
    - Agent properties → heat sources
    """

    source_domain = "agent"
    target_domain = "field"

    def __init__(
        self,
        positions,
        values,
        field_shape: Tuple[int, int],
        method: str = "accumulate"
    ):
        super().__init__(source_data=(positions, values))
        self.positions = positions
        self.values = values
        self.field_shape = field_shape
        self.method = method  # "accumulate", "average", "max"

    def transform(self, source_data: Any) -> np.ndarray:
        """Deposit agent values onto field."""
        if source_data is not None:
            positions, values = source_data
        else:
            positions, values = self.positions, self.values

        field = np.zeros(self.field_shape, dtype=np.float32)

        # Convert positions to grid coordinates
        coords = positions.astype(int)

        # Clamp to valid range
        coords[:, 0] = np.clip(coords[:, 0], 0, self.field_shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, self.field_shape[1] - 1)

        if self.method == "accumulate":
            # Sum all values at each grid cell
            for i, (y, x) in enumerate(coords):
                field[y, x] += values[i]

        elif self.method == "average":
            # Average values at each grid cell
            counts = np.zeros(self.field_shape, dtype=int)
            for i, (y, x) in enumerate(coords):
                field[y, x] += values[i]
                counts[y, x] += 1

            # Avoid division by zero
            mask = counts > 0
            field[mask] /= counts[mask]

        elif self.method == "max":
            # Take maximum value at each grid cell
            field.fill(-np.inf)
            for i, (y, x) in enumerate(coords):
                field[y, x] = max(field[y, x], values[i])
            field[field == -np.inf] = 0

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return field

    def validate(self) -> bool:
        """Check positions and values are compatible."""
        if self.positions is None or self.values is None:
            return False

        if len(self.positions) != len(self.values):
            raise ValueError(
                f"Positions ({len(self.positions)}) and values ({len(self.values)}) "
                f"must have same length"
            )

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'positions': np.ndarray,
            'values': np.ndarray,
            'field_shape': Tuple[int, int],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
        }
