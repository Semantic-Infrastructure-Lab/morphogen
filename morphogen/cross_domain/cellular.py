"""
Cellular → Field Domain Interface

Interface for converting cellular automata state to fields.
"""

from typing import Any, Dict, Type
import numpy as np

from .base import DomainInterface


class CellularToFieldInterface(DomainInterface):
    """
    Cellular → Field: Convert cellular automata state to field.

    Use cases:
    - CA state → initial conditions for PDEs
    - Game of Life → density field
    - Pattern state → field patterns
    """

    source_domain = "cellular"
    target_domain = "field"

    def __init__(self, ca_state: np.ndarray, normalize: bool = True):
        """
        Args:
            ca_state: Cellular automata state array
            normalize: If True, normalize to [0, 1]
        """
        super().__init__(source_data=ca_state)
        self.ca_state = ca_state
        self.normalize = normalize

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert CA state to field."""
        ca_state = source_data if source_data is not None else self.ca_state

        field = ca_state.astype(np.float32)

        if self.normalize:
            field_min = field.min()
            field_max = field.max()
            if field_max > field_min:
                field = (field - field_min) / (field_max - field_min)

        return field

    def validate(self) -> bool:
        """Check CA state is valid."""
        if self.ca_state is None:
            return False

        if not isinstance(self.ca_state, np.ndarray):
            raise TypeError("CA state must be numpy array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'ca_state': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}
