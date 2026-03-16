"""Image operations for procedural graphics and visualization.

This module provides comprehensive image handling for:
- Procedural texture generation
- Fractal visualization (Mandelbrot, Julia sets)
- Simulation visualization (CA, fluids, physics)
- Post-processing effects
- Compositing and blending
- Image transformations

Supports:
- RGB/RGBA images
- Image creation from fields/noise
- Warping and distortion
- Filtering (blur, sharpen, edge detection)
- Morphological operations (erode, dilate)
- Compositing with blend modes
"""

from typing import Tuple, Optional, Union, Callable
import numpy as np

from morphogen.core.operator import operator, OpCategory
from scipy import ndimage


class Image:
    """RGB/RGBA image with NumPy backend.

    Represents a 2D image with color channels.
    """

    def __init__(self, data: np.ndarray):
        """Initialize image.

        Args:
            data: NumPy array (shape: (height, width, channels))
                 channels = 3 for RGB, 4 for RGBA
                 values in [0, 1]
        """
        if data.ndim != 3:
            raise ValueError(f"Image data must be 3D (H, W, C), got shape {data.shape}")

        if data.shape[2] not in (3, 4):
            raise ValueError(f"Image must have 3 (RGB) or 4 (RGBA) channels, got {data.shape[2]}")

        self.data = data.astype(np.float32)
        self.shape = data.shape[:2]  # (height, width)
        self.channels = data.shape[2]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get image width."""
        return self.shape[1]

    def copy(self) -> 'Image':
        """Create a copy of this image."""
        return Image(self.data.copy())

    def __repr__(self) -> str:
        return f"Image(shape={self.shape}, channels={self.channels})"


class ImageOperations:
    """Namespace for image operations (accessed as 'image' in DSL)."""

    # ============================================================================
    # LAYER 1: IMAGE CREATION
    # ============================================================================

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.CONSTRUCT,
        signature="(width: int, height: int, channels: int, fill_value: float) -> Image",
        deterministic=True,
        doc="Create blank image"
    )
    def blank(width: int, height: int, channels: int = 3,
             fill_value: float = 0.0) -> Image:
        """Create blank image.

        Args:
            width: Image width
            height: Image height
            channels: Number of channels (3 for RGB, 4 for RGBA)
            fill_value: Initial color value

        Returns:
            Blank image
        """
        data = np.full((height, width, channels), fill_value, dtype=np.float32)
        return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.CONSTRUCT,
        signature="(r: float, g: float, b: float, width: int, height: int) -> Image",
        deterministic=True,
        doc="Create solid color RGB image"
    )
    def rgb(r: float, g: float, b: float, width: int, height: int) -> Image:
        """Create solid color RGB image.

        Args:
            r: Red value [0, 1]
            g: Green value [0, 1]
            b: Blue value [0, 1]
            width: Image width
            height: Image height

        Returns:
            Solid color image
        """
        data = np.zeros((height, width, 3), dtype=np.float32)
        data[:, :, 0] = r
        data[:, :, 1] = g
        data[:, :, 2] = b
        return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.CONSTRUCT,
        signature="(field: ndarray, palette: Optional[object]) -> Image",
        deterministic=True,
        doc="Create image from scalar field"
    )
    def from_field(field: np.ndarray, palette: Optional[object] = None) -> Image:
        """Create image from scalar field.

        Args:
            field: 2D scalar field (height, width)
            palette: Optional palette for coloring (defaults to greyscale)

        Returns:
            RGB image

        Example:
            >>> from morphogen.stdlib.noise import noise
            >>> from morphogen.stdlib.palette import palette
            >>> field = noise.perlin2d((256, 256))
            >>> pal = palette.viridis()
            >>> img = image.from_field(field.data, pal)
        """
        if field.ndim != 2:
            raise ValueError(f"Field must be 2D, got shape {field.shape}")

        # Normalize field to [0, 1]
        field_min = np.min(field)
        field_max = np.max(field)

        if field_max > field_min:
            normalized = (field - field_min) / (field_max - field_min)
        else:
            normalized = np.zeros_like(field)

        # Apply palette or greyscale
        if palette is not None:
            rgb = palette.sample(normalized)
        else:
            # Greyscale
            rgb = np.stack([normalized, normalized, normalized], axis=-1)

        return Image(rgb)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.CONSTRUCT,
        signature="(r_channel: ndarray, g_channel: ndarray, b_channel: ndarray, a_channel: Optional[ndarray]) -> Image",
        deterministic=True,
        doc="Compose image from separate channel arrays"
    )
    def compose(r_channel: np.ndarray, g_channel: np.ndarray, b_channel: np.ndarray,
                a_channel: Optional[np.ndarray] = None) -> Image:
        """Compose image from separate channel arrays.

        Args:
            r_channel: Red channel (height, width)
            g_channel: Green channel (height, width)
            b_channel: Blue channel (height, width)
            a_channel: Optional alpha channel (height, width)

        Returns:
            RGB or RGBA image

        Example:
            >>> r = noise.perlin2d((256, 256), seed=0).data
            >>> g = noise.perlin2d((256, 256), seed=1).data
            >>> b = noise.perlin2d((256, 256), seed=2).data
            >>> img = image.compose(r, g, b)
        """
        if a_channel is not None:
            channels = [r_channel, g_channel, b_channel, a_channel]
        else:
            channels = [r_channel, g_channel, b_channel]

        # Normalize each channel to [0, 1]
        normalized_channels = []
        for ch in channels:
            ch_min = np.min(ch)
            ch_max = np.max(ch)
            if ch_max > ch_min:
                normalized = (ch - ch_min) / (ch_max - ch_min)
            else:
                normalized = np.zeros_like(ch)
            normalized_channels.append(normalized)

        data = np.stack(normalized_channels, axis=-1)
        return Image(data)

    # ============================================================================
    # LAYER 2: TRANSFORMATIONS
    # ============================================================================

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, factor: float, method: str) -> Image",
        deterministic=True,
        doc="Scale image by factor"
    )
    def scale(img: Image, factor: float, method: str = "bilinear") -> Image:
        """Scale image by factor.

        Args:
            img: Input image
            factor: Scale factor (>1 enlarges, <1 shrinks)
            method: Interpolation method ("nearest", "bilinear")

        Returns:
            Scaled image
        """
        if method == "nearest":
            order = 0
        elif method == "bilinear":
            order = 1
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        new_height = int(img.height * factor)
        new_width = int(img.width * factor)

        # Zoom each channel
        data = np.zeros((new_height, new_width, img.channels), dtype=np.float32)
        for c in range(img.channels):
            data[:, :, c] = ndimage.zoom(img.data[:, :, c], factor, order=order)

        return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, angle: float, reshape: bool) -> Image",
        deterministic=True,
        doc="Rotate image by angle (degrees)"
    )
    def rotate(img: Image, angle: float, reshape: bool = False) -> Image:
        """Rotate image by angle (degrees).

        Args:
            img: Input image
            angle: Rotation angle in degrees (counterclockwise)
            reshape: If True, expand image to fit rotated content

        Returns:
            Rotated image
        """
        data = np.zeros_like(img.data)

        for c in range(img.channels):
            data[:, :, c] = ndimage.rotate(img.data[:, :, c], angle,
                                          reshape=reshape, order=1)

        if reshape:
            return Image(data)
        else:
            return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, displacement_field: Tuple[ndarray, ndarray]) -> Image",
        deterministic=True,
        doc="Warp image using displacement field"
    )
    def warp(img: Image, displacement_field: Tuple[np.ndarray, np.ndarray]) -> Image:
        """Warp image using displacement field.

        Args:
            img: Input image
            displacement_field: Tuple of (dy, dx) displacement arrays
                               Same shape as image (height, width)

        Returns:
            Warped image

        Example:
            >>> # Warp using noise field
            >>> vx, vy = noise.vector_field((256, 256))
            >>> warped = image.warp(img, (vy.data * 10, vx.data * 10))
        """
        dy, dx = displacement_field

        # Create coordinate grids
        y, x = np.mgrid[0:img.height, 0:img.width].astype(np.float32)

        # Apply displacement
        coords_y = y + dy
        coords_x = x + dx

        # Warp each channel
        data = np.zeros_like(img.data)
        for c in range(img.channels):
            data[:, :, c] = ndimage.map_coordinates(
                img.data[:, :, c],
                [coords_y, coords_x],
                order=1,
                mode='reflect'
            )

        return Image(data)

    # ============================================================================
    # LAYER 3: FILTERS
    # ============================================================================

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, sigma: float) -> Image",
        deterministic=True,
        doc="Apply Gaussian blur"
    )
    def blur(img: Image, sigma: float = 1.0) -> Image:
        """Apply Gaussian blur.

        Args:
            img: Input image
            sigma: Blur radius (standard deviation)

        Returns:
            Blurred image
        """
        data = np.zeros_like(img.data)

        for c in range(img.channels):
            data[:, :, c] = ndimage.gaussian_filter(img.data[:, :, c], sigma)

        return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, strength: float) -> Image",
        deterministic=True,
        doc="Sharpen image using unsharp mask"
    )
    def sharpen(img: Image, strength: float = 1.0) -> Image:
        """Sharpen image using unsharp mask.

        Args:
            img: Input image
            strength: Sharpening strength

        Returns:
            Sharpened image
        """
        # Sharpen = original + strength * (original - blurred)
        blurred = ImageOperations.blur(img, sigma=1.0)
        data = img.data + strength * (img.data - blurred.data)

        return Image(np.clip(data, 0, 1))

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.QUERY,
        signature="(img: Image, method: str) -> Image",
        deterministic=True,
        doc="Detect edges in image"
    )
    def edge_detect(img: Image, method: str = "sobel") -> Image:
        """Detect edges in image.

        Args:
            img: Input image
            method: Edge detection method ("sobel", "prewitt", "laplacian")

        Returns:
            Edge-detected image (greyscale for RGB input)
        """
        # Convert to greyscale if RGB
        if img.channels >= 3:
            grey = 0.299 * img.data[:, :, 0] + 0.587 * img.data[:, :, 1] + 0.114 * img.data[:, :, 2]
        else:
            grey = img.data[:, :, 0]

        if method == "sobel":
            edges = ndimage.sobel(grey)
        elif method == "prewitt":
            edges = ndimage.prewitt(grey)
        elif method == "laplacian":
            edges = ndimage.laplace(grey)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")

        # Normalize
        edges = np.abs(edges)
        edges = edges / (np.max(edges) + 1e-10)

        # Return as RGB
        data = np.stack([edges, edges, edges], axis=-1)
        return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, iterations: int) -> Image",
        deterministic=True,
        doc="Morphological erosion"
    )
    def erode(img: Image, iterations: int = 1) -> Image:
        """Morphological erosion.

        Args:
            img: Input image
            iterations: Number of erosion iterations

        Returns:
            Eroded image
        """
        data = np.zeros_like(img.data)

        for c in range(img.channels):
            data[:, :, c] = ndimage.grey_erosion(img.data[:, :, c],
                                                 size=(3, 3),
                                                 iterations=iterations)

        return Image(data)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, iterations: int) -> Image",
        deterministic=True,
        doc="Morphological dilation"
    )
    def dilate(img: Image, iterations: int = 1) -> Image:
        """Morphological dilation.

        Args:
            img: Input image
            iterations: Number of dilation iterations

        Returns:
            Dilated image
        """
        data = np.zeros_like(img.data)

        for c in range(img.channels):
            data[:, :, c] = ndimage.grey_dilation(img.data[:, :, c],
                                                  size=(3, 3),
                                                  iterations=iterations)

        return Image(data)

    # ============================================================================
    # LAYER 4: COMPOSITING
    # ============================================================================

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img_a: Image, img_b: Image, mode: str, opacity: float) -> Image",
        deterministic=True,
        doc="Blend two images using blend mode"
    )
    def blend(img_a: Image, img_b: Image, mode: str = "normal", opacity: float = 1.0) -> Image:
        """Blend two images using blend mode.

        Args:
            img_a: Base image
            img_b: Blend image
            mode: Blend mode ("normal", "multiply", "screen", "overlay", "difference")
            opacity: Blend opacity [0, 1]

        Returns:
            Blended image

        Note:
            Images must have the same dimensions.
        """
        if img_a.shape != img_b.shape:
            raise ValueError(f"Images must have same dimensions: {img_a.shape} vs {img_b.shape}")

        # Import color operations for blend modes (sys.modules avoids package __init__ cycle)
        import sys
        color = sys.modules["morphogen.stdlib.color"]

        # Extract RGB channels (ignore alpha for blending logic)
        rgb_a = img_a.data[:, :, :3]
        rgb_b = img_b.data[:, :, :3]

        # Apply blend mode
        if mode == "normal":
            result_rgb = rgb_b
        elif mode == "multiply":
            result_rgb = color.blend_multiply(rgb_a, rgb_b)
        elif mode == "screen":
            result_rgb = color.blend_screen(rgb_a, rgb_b)
        elif mode == "overlay":
            result_rgb = color.blend_overlay(rgb_a, rgb_b)
        elif mode == "difference":
            result_rgb = color.blend_difference(rgb_a, rgb_b)
        elif mode == "soft_light":
            result_rgb = color.blend_soft_light(rgb_a, rgb_b)
        else:
            raise ValueError(f"Unknown blend mode: {mode}")

        # Mix with base using opacity
        result_rgb = rgb_a * (1 - opacity) + result_rgb * opacity

        # Handle alpha channel if present
        if img_a.channels == 4:
            alpha = img_a.data[:, :, 3:4]
            result = np.concatenate([result_rgb, alpha], axis=2)
        else:
            result = result_rgb

        return Image(np.clip(result, 0, 1))

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, overlay_img: Image, mask: Optional[ndarray]) -> Image",
        deterministic=True,
        doc="Overlay image with optional mask"
    )
    def overlay(img: Image, overlay_img: Image, mask: Optional[np.ndarray] = None) -> Image:
        """Overlay image with optional mask.

        Args:
            img: Base image
            overlay_img: Overlay image
            mask: Optional mask (height, width), values in [0, 1]

        Returns:
            Composited image
        """
        if img.shape != overlay_img.shape:
            raise ValueError(f"Images must have same dimensions: {img.shape} vs {overlay_img.shape}")

        if mask is None:
            # If overlay has alpha, use it as mask
            if overlay_img.channels == 4:
                mask = overlay_img.data[:, :, 3]
            else:
                mask = np.ones((img.height, img.width), dtype=np.float32)

        # Expand mask to match channels
        mask = mask[:, :, np.newaxis]

        # Composite
        result = img.data * (1 - mask) + overlay_img.data[:, :, :img.channels] * mask

        return Image(np.clip(result, 0, 1))

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(background: Image, foreground: Image) -> Image",
        deterministic=True,
        doc="Alpha composite foreground over background"
    )
    def alpha_composite(background: Image, foreground: Image) -> Image:
        """Alpha composite foreground over background.

        Args:
            background: Background image (RGB or RGBA)
            foreground: Foreground image (must have alpha channel)

        Returns:
            Composited image

        Note:
            Uses standard alpha compositing formula:
            result = fg * alpha + bg * (1 - alpha)
        """
        if foreground.channels != 4:
            raise ValueError("Foreground must have alpha channel (RGBA)")

        if background.shape != foreground.shape:
            raise ValueError(f"Images must have same dimensions: {background.shape} vs {foreground.shape}")

        # Extract alpha
        alpha = foreground.data[:, :, 3:4]

        # Composite RGB
        fg_rgb = foreground.data[:, :, :3]
        bg_rgb = background.data[:, :, :3]

        result_rgb = fg_rgb * alpha + bg_rgb * (1 - alpha)

        # Handle background alpha if present
        if background.channels == 4:
            bg_alpha = background.data[:, :, 3:4]
            result_alpha = alpha + bg_alpha * (1 - alpha)
            result = np.concatenate([result_rgb, result_alpha], axis=2)
        else:
            result = result_rgb

        return Image(np.clip(result, 0, 1))

    # ============================================================================
    # LAYER 5: PROCEDURAL EFFECTS
    # ============================================================================

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, palette: object, channel: str) -> Image",
        deterministic=True,
        doc="Apply palette to image based on channel"
    )
    def apply_palette(img: Image, palette: object, channel: str = "luminance") -> Image:
        """Apply palette to image based on channel.

        Args:
            img: Input image
            palette: Palette object
            channel: Channel to use ("luminance", "r", "g", "b", "saturation")

        Returns:
            Palette-mapped image

        Example:
            >>> from morphogen.stdlib.palette import palette
            >>> pal = palette.viridis()
            >>> colored = image.apply_palette(img, pal, "luminance")
        """
        # Extract channel values
        if channel == "luminance":
            values = 0.299 * img.data[:, :, 0] + 0.587 * img.data[:, :, 1] + 0.114 * img.data[:, :, 2]
        elif channel == "r":
            values = img.data[:, :, 0]
        elif channel == "g":
            values = img.data[:, :, 1]
        elif channel == "b":
            values = img.data[:, :, 2]
        elif channel == "saturation":
            import sys
            color = sys.modules["morphogen.stdlib.color"]
            hsv = color.rgb_to_hsv(img.data[:, :, :3])
            values = hsv[:, :, 1]
        else:
            raise ValueError(f"Unknown channel: {channel}")

        # Normalize to [0, 1]
        vmin = np.min(values)
        vmax = np.max(values)
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(values)

        # Sample palette
        rgb = palette.sample(normalized)

        return Image(rgb)

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.CONSTRUCT,
        signature="(heightfield: ndarray, strength: float) -> Image",
        deterministic=True,
        doc="Generate normal map from height field"
    )
    def normal_map_from_heightfield(heightfield: np.ndarray, strength: float = 1.0) -> Image:
        """Generate normal map from height field.

        Args:
            heightfield: 2D height field
            strength: Normal strength multiplier

        Returns:
            RGB normal map (normals encoded in RGB)

        Example:
            >>> from morphogen.stdlib.noise import noise
            >>> terrain = noise.fbm((256, 256), octaves=6)
            >>> normals = image.normal_map_from_heightfield(terrain.data)
        """
        # Compute gradients
        gy, gx = np.gradient(heightfield)

        # Scale by strength
        gx = gx * strength
        gy = gy * strength

        # Compute normals (pointing up)
        normal_x = -gx
        normal_y = -gy
        normal_z = np.ones_like(gx)

        # Normalize
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x = normal_x / (length + 1e-10)
        normal_y = normal_y / (length + 1e-10)
        normal_z = normal_z / (length + 1e-10)

        # Encode as RGB (map from [-1, 1] to [0, 1])
        r = (normal_x + 1) * 0.5
        g = (normal_y + 1) * 0.5
        b = (normal_z + 1) * 0.5

        data = np.stack([r, g, b], axis=-1)
        return Image(np.clip(data, 0, 1))

    @staticmethod
    @operator(
        domain="image",
        category=OpCategory.TRANSFORM,
        signature="(img: Image, gradient_palette: object) -> Image",
        deterministic=True,
        doc="Apply gradient map (like Photoshop gradient map)"
    )
    def gradient_map(img: Image, gradient_palette: object) -> Image:
        """Apply gradient map (like Photoshop gradient map).

        Args:
            img: Input image
            gradient_palette: Palette to map to

        Returns:
            Gradient-mapped image
        """
        return ImageOperations.apply_palette(img, gradient_palette, "luminance")


# Create singleton instance for use as 'image' namespace
image = ImageOperations()

# Export operators for domain registry discovery
blank = ImageOperations.blank
rgb = ImageOperations.rgb
from_field = ImageOperations.from_field
scale = ImageOperations.scale
rotate = ImageOperations.rotate
blur = ImageOperations.blur
sharpen = ImageOperations.sharpen
edge_detect = ImageOperations.edge_detect
dilate = ImageOperations.dilate
erode = ImageOperations.erode
blend = ImageOperations.blend
compose = ImageOperations.compose
overlay = ImageOperations.overlay
alpha_composite = ImageOperations.alpha_composite
warp = ImageOperations.warp
apply_palette = ImageOperations.apply_palette
gradient_map = ImageOperations.gradient_map
normal_map_from_heightfield = ImageOperations.normal_map_from_heightfield
