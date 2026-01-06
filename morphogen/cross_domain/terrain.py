"""
Terrain ↔ Field Domain Interfaces

Interfaces for bidirectional data flow between terrain and field domains.
"""

from typing import Any, Dict, Type
import numpy as np

from .base import DomainInterface


class TerrainToFieldInterface(DomainInterface):
    """
    Terrain → Field: Convert terrain heightmap to scalar field.

    Use cases:
    - Heightmap → diffusion initial conditions
    - Elevation → potential field
    - Terrain features → field patterns
    """

    source_domain = "terrain"
    target_domain = "field"

    def __init__(self, heightmap: np.ndarray, normalize: bool = True):
        """
        Args:
            heightmap: 2D terrain heightmap
            normalize: If True, normalize to [0, 1] range
        """
        super().__init__(source_data=heightmap)
        self.heightmap = heightmap
        self.normalize = normalize

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert heightmap to field."""
        heightmap = source_data if source_data is not None else self.heightmap

        # Extract height data if wrapped in object
        if hasattr(heightmap, 'data'):
            field_data = heightmap.data.copy()
        else:
            field_data = heightmap.copy()

        if self.normalize:
            # Normalize to [0, 1]
            field_min = field_data.min()
            field_max = field_data.max()
            if field_max > field_min:
                field_data = (field_data - field_min) / (field_max - field_min)

        return field_data

    def validate(self) -> bool:
        """Check heightmap is valid."""
        if self.heightmap is None:
            return False

        # Extract array
        if hasattr(self.heightmap, 'data'):
            arr = self.heightmap.data
        else:
            arr = self.heightmap

        if not isinstance(arr, np.ndarray):
            raise TypeError("Heightmap must be numpy array")

        if arr.ndim != 2:
            raise ValueError(f"Heightmap must be 2D, got shape {arr.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'heightmap': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}


class FieldToTerrainInterface(DomainInterface):
    """
    Field → Terrain: Convert scalar field to terrain heightmap.

    Use cases:
    - Procedural field → terrain generation
    - Simulation result → landscape
    """

    source_domain = "field"
    target_domain = "terrain"

    def __init__(self, field: np.ndarray, height_scale: float = 100.0):
        """
        Args:
            field: 2D scalar field
            height_scale: Scaling factor for height values
        """
        super().__init__(source_data=field)
        self.field = field
        self.height_scale = height_scale

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """Convert field to heightmap."""
        field = source_data if source_data is not None else self.field

        # Normalize field to [0, 1]
        field_min = field.min()
        field_max = field.max()
        normalized = (field - field_min) / (field_max - field_min + 1e-10)

        # Scale to height range
        heightmap = normalized * self.height_scale

        return {
            'heightmap': heightmap,
            'min_height': 0.0,
            'max_height': self.height_scale,
        }

    def validate(self) -> bool:
        """Check field is valid."""
        if self.field is None:
            return False

        if not isinstance(self.field, np.ndarray):
            raise TypeError("Field must be numpy array")

        if self.field.ndim != 2:
            raise ValueError(f"Field must be 2D, got shape {self.field.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'terrain_data': Dict[str, np.ndarray]}
