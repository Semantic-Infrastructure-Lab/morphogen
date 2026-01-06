"""
Vision → Field Domain Interface

Interface for converting computer vision features to fields.
"""

from typing import Any, Dict, Type
import numpy as np

from .base import DomainInterface


class VisionToFieldInterface(DomainInterface):
    """
    Vision → Field: Convert computer vision features to fields.

    Use cases:
    - Edge map → scalar field
    - Optical flow → vector field
    - Feature map → field initialization
    """

    source_domain = "vision"
    target_domain = "field"

    def __init__(self, image: np.ndarray, mode: str = "edges"):
        """
        Args:
            image: Input image (grayscale or RGB)
            mode: Conversion mode ("edges", "gradient", "intensity")
        """
        super().__init__(source_data=image)
        self.image = image
        self.mode = mode

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert vision data to field."""
        image = source_data if source_data is not None else self.image

        # Convert to grayscale if RGB
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        if self.mode == "edges":
            # Edge detection produces scalar field
            from scipy.ndimage import sobel
            sx = sobel(image, axis=1)
            sy = sobel(image, axis=0)
            edge_mag = np.sqrt(sx**2 + sy**2)
            return edge_mag

        elif self.mode == "gradient":
            # Gradient field (vector)
            gy, gx = np.gradient(image)
            # Return as vector field (H, W, 2)
            return np.stack([gy, gx], axis=2)

        elif self.mode == "intensity":
            # Direct intensity mapping
            return image.astype(np.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def validate(self) -> bool:
        """Check image is valid."""
        if self.image is None:
            return False

        if not isinstance(self.image, np.ndarray):
            raise TypeError("Image must be numpy array")

        if self.image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got shape {self.image.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'image': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}
