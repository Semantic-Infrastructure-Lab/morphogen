"""
Computer Vision Domain

Provides computer vision and image analysis operations:
- Edge detection (Sobel, Canny, Laplacian)
- Feature detection (corners, blobs, lines)
- Image segmentation
- Morphological operations
- Optical flow
- Template matching
- Contour detection and analysis

Version: v0.10.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import scipy.ndimage as ndimage
from scipy.signal import convolve2d

from morphogen.core.operator import operator, OpCategory


class EdgeDetector(Enum):
    """Edge detection algorithms"""
    SOBEL = "sobel"
    PREWITT = "prewitt"
    SCHARR = "scharr"
    LAPLACIAN = "laplacian"
    CANNY = "canny"


class MorphOp(Enum):
    """Morphological operations"""
    ERODE = "erode"
    DILATE = "dilate"
    OPEN = "open"
    CLOSE = "close"
    GRADIENT = "gradient"
    TOPHAT = "tophat"
    BLACKHAT = "blackhat"


@dataclass
class ImageGray:
    """Grayscale image

    Attributes:
        data: 2D array of intensity values (0-1)
    """
    data: np.ndarray

    def copy(self) -> 'ImageGray':
        """Create a deep copy"""
        return ImageGray(data=self.data.copy())

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape (height, width)"""
        return self.data.shape


@dataclass
class EdgeMap:
    """Edge detection result

    Attributes:
        magnitude: Edge strength
        direction: Edge direction (radians)
    """
    magnitude: np.ndarray
    direction: np.ndarray

    def copy(self) -> 'EdgeMap':
        """Create a deep copy"""
        return EdgeMap(
            magnitude=self.magnitude.copy(),
            direction=self.direction.copy()
        )


@dataclass
class Keypoint:
    """Detected keypoint/feature

    Attributes:
        x: X coordinate
        y: Y coordinate
        response: Feature response strength
        scale: Feature scale
        orientation: Feature orientation (radians)
    """
    x: float
    y: float
    response: float
    scale: float = 1.0
    orientation: float = 0.0


@dataclass
class Contour:
    """Detected contour

    Attributes:
        points: List of (x, y) points defining the contour
        area: Contour area
        perimeter: Contour perimeter
        centroid: Centroid (cx, cy)
    """
    points: np.ndarray
    area: float
    perimeter: float
    centroid: Tuple[float, float]


class VisionOperations:
    """Computer vision operations"""

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.CONSTRUCT,
        signature="(data: np.ndarray) -> ImageGray",
        deterministic=True,
        doc="Create grayscale image from array"
    )
    def create_image(data: np.ndarray) -> ImageGray:
        """Create grayscale image from array

        Args:
            data: 2D array (will be normalized to 0-1)

        Returns:
            Grayscale image
        """
        data = np.asarray(data, dtype=np.float64)

        # Normalize to [0, 1]
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())

        return ImageGray(data=data)

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img: ImageGray) -> EdgeMap",
        deterministic=True,
        doc="Sobel edge detection"
    )
    def sobel(img: ImageGray) -> EdgeMap:
        """Sobel edge detection

        Args:
            img: Input grayscale image

        Returns:
            Edge map with magnitude and direction
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float64)

        # Convolve
        grad_x = convolve2d(img.data, sobel_x, mode='same', boundary='symm')
        grad_y = convolve2d(img.data, sobel_y, mode='same', boundary='symm')

        # Calculate magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        return EdgeMap(magnitude=magnitude, direction=direction)

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img: ImageGray) -> ImageGray",
        deterministic=True,
        doc="Laplacian edge detection"
    )
    def laplacian(img: ImageGray) -> ImageGray:
        """Laplacian edge detection

        Args:
            img: Input image

        Returns:
            Edge-detected image
        """
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float64)

        result = convolve2d(img.data, laplacian_kernel, mode='same', boundary='symm')

        return ImageGray(data=np.abs(result))

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img: ImageGray, low_threshold: float, high_threshold: float) -> ImageGray",
        deterministic=True,
        doc="Canny edge detection (simplified)"
    )
    def canny(img: ImageGray, low_threshold: float = 0.1,
             high_threshold: float = 0.3) -> ImageGray:
        """Canny edge detection (simplified)

        Args:
            img: Input image
            low_threshold: Low threshold for hysteresis
            high_threshold: High threshold for hysteresis

        Returns:
            Binary edge map
        """
        # 1. Gaussian smoothing
        smoothed = VisionOperations.gaussian_blur(img, sigma=1.4)

        # 2. Gradient calculation
        edge_map = VisionOperations.sobel(smoothed)

        # 3. Non-maximum suppression (simplified)
        magnitude = edge_map.magnitude
        direction = edge_map.direction

        # Quantize direction to 4 bins (0, 45, 90, 135 degrees)
        angle = np.rad2deg(direction) % 180
        angle_quant = np.round(angle / 45) * 45

        nms = np.zeros_like(magnitude)
        h, w = magnitude.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                q = angle_quant[i, j]

                # Check neighbors along gradient direction
                if q == 0 or q == 180:
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif q == 45:
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif q == 90:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 135
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

                if magnitude[i, j] >= max(neighbors):
                    nms[i, j] = magnitude[i, j]

        # 4. Double thresholding
        strong_edges = nms > high_threshold
        weak_edges = (nms >= low_threshold) & (nms <= high_threshold)

        # 5. Edge tracking by hysteresis (simplified)
        edges = strong_edges.copy()

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak_edges[i, j]:
                    # Check if connected to strong edge
                    if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                        edges[i, j] = True

        return ImageGray(data=edges.astype(np.float64))

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img: ImageGray, k: float, threshold: float) -> List[Keypoint]",
        deterministic=True,
        doc="Harris corner detection"
    )
    def harris_corners(img: ImageGray, k: float = 0.04,
                      threshold: float = 0.01) -> List[Keypoint]:
        """Harris corner detection

        Args:
            img: Input image
            k: Harris detector free parameter (typically 0.04-0.06)
            threshold: Corner response threshold

        Returns:
            List of detected corners
        """
        # Calculate gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        Ix = convolve2d(img.data, sobel_x, mode='same', boundary='symm')
        Iy = convolve2d(img.data, sobel_y, mode='same', boundary='symm')

        # Products of derivatives
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Gaussian window (simple box filter)
        window_size = 5
        Sxx = ndimage.uniform_filter(Ixx, window_size)
        Syy = ndimage.uniform_filter(Iyy, window_size)
        Sxy = ndimage.uniform_filter(Ixy, window_size)

        # Harris response
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        response = det - k * (trace ** 2)

        # Find local maxima above threshold
        keypoints = []
        h, w = response.shape

        for i in range(2, h - 2):
            for j in range(2, w - 2):
                if response[i, j] > threshold:
                    # Check if local maximum
                    local_region = response[i-2:i+3, j-2:j+3]
                    if response[i, j] == local_region.max():
                        keypoints.append(Keypoint(
                            x=float(j),
                            y=float(i),
                            response=float(response[i, j])
                        ))

        return keypoints

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.TRANSFORM,
        signature="(img: ImageGray, sigma: float) -> ImageGray",
        deterministic=True,
        doc="Apply Gaussian blur"
    )
    def gaussian_blur(img: ImageGray, sigma: float = 1.0) -> ImageGray:
        """Apply Gaussian blur

        Args:
            img: Input image
            sigma: Gaussian standard deviation

        Returns:
            Blurred image
        """
        blurred = ndimage.gaussian_filter(img.data, sigma=sigma)
        return ImageGray(data=blurred)

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.TRANSFORM,
        signature="(img: ImageGray, operation: MorphOp, kernel_size: int) -> ImageGray",
        deterministic=True,
        doc="Apply morphological operation"
    )
    def morphological(img: ImageGray, operation: MorphOp,
                     kernel_size: int = 3) -> ImageGray:
        """Apply morphological operation

        Args:
            img: Input binary image
            operation: Morphological operation type
            kernel_size: Structuring element size

        Returns:
            Result image
        """
        # Create structuring element
        structure = np.ones((kernel_size, kernel_size))

        # Convert to binary
        binary = img.data > 0.5

        if operation == MorphOp.ERODE:
            result = ndimage.binary_erosion(binary, structure=structure)
        elif operation == MorphOp.DILATE:
            result = ndimage.binary_dilation(binary, structure=structure)
        elif operation == MorphOp.OPEN:
            result = ndimage.binary_opening(binary, structure=structure)
        elif operation == MorphOp.CLOSE:
            result = ndimage.binary_closing(binary, structure=structure)
        elif operation == MorphOp.GRADIENT:
            dilated = ndimage.binary_dilation(binary, structure=structure)
            eroded = ndimage.binary_erosion(binary, structure=structure)
            result = dilated ^ eroded
        elif operation == MorphOp.TOPHAT:
            opened = ndimage.binary_opening(binary, structure=structure)
            result = binary & ~opened
        elif operation == MorphOp.BLACKHAT:
            closed = ndimage.binary_closing(binary, structure=structure)
            result = closed & ~binary
        else:
            result = binary

        return ImageGray(data=result.astype(np.float64))

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.TRANSFORM,
        signature="(img: ImageGray, threshold: float) -> ImageGray",
        deterministic=True,
        doc="Apply binary threshold"
    )
    def threshold(img: ImageGray, threshold: float = 0.5) -> ImageGray:
        """Apply binary threshold

        Args:
            img: Input image
            threshold: Threshold value

        Returns:
            Binary image
        """
        binary = (img.data > threshold).astype(np.float64)
        return ImageGray(data=binary)

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.TRANSFORM,
        signature="(img: ImageGray, block_size: int, c: float) -> ImageGray",
        deterministic=True,
        doc="Apply adaptive threshold"
    )
    def adaptive_threshold(img: ImageGray, block_size: int = 11,
                          c: float = 2.0) -> ImageGray:
        """Apply adaptive threshold

        Args:
            img: Input image
            block_size: Size of local neighborhood
            c: Constant subtracted from mean

        Returns:
            Binary image
        """
        # Calculate local mean
        local_mean = ndimage.uniform_filter(img.data, size=block_size)

        # Threshold
        binary = (img.data > (local_mean - c / 255.0)).astype(np.float64)

        return ImageGray(data=binary)

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img: ImageGray, min_area: float) -> List[Contour]",
        deterministic=True,
        doc="Find contours in binary image (simplified)"
    )
    def find_contours(img: ImageGray, min_area: float = 10.0) -> List[Contour]:
        """Find contours in binary image (simplified)

        Args:
            img: Input binary image
            min_area: Minimum contour area

        Returns:
            List of detected contours
        """
        # Label connected components
        labeled, num_features = ndimage.label(img.data > 0.5)

        contours = []

        for label_id in range(1, num_features + 1):
            # Get mask for this component
            mask = (labeled == label_id)

            # Calculate properties
            area = float(np.sum(mask))

            if area < min_area:
                continue

            # Get boundary points
            boundary = mask & ~ndimage.binary_erosion(mask)
            y_coords, x_coords = np.where(boundary)

            if len(x_coords) < 3:
                continue

            points = np.column_stack((x_coords, y_coords))

            # Calculate centroid
            cy, cx = ndimage.center_of_mass(mask)

            # Calculate perimeter (approximate)
            perimeter = float(len(points))

            contours.append(Contour(
                points=points,
                area=area,
                perimeter=perimeter,
                centroid=(float(cx), float(cy))
            ))

        return contours

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img: ImageGray, template: ImageGray) -> np.ndarray",
        deterministic=True,
        doc="Template matching using normalized cross-correlation"
    )
    def template_match(img: ImageGray, template: ImageGray) -> np.ndarray:
        """Template matching using normalized cross-correlation

        Args:
            img: Input image
            template: Template image

        Returns:
            Correlation map (high values = good match)
        """
        from scipy.signal import correlate2d

        # Normalize template
        template_norm = template.data - template.data.mean()
        template_norm = template_norm / (np.std(template_norm) + 1e-8)

        # Normalize image locally
        img_mean = ndimage.uniform_filter(img.data, template.shape)
        img_sq_mean = ndimage.uniform_filter(img.data ** 2, template.shape)
        img_std = np.sqrt(img_sq_mean - img_mean ** 2 + 1e-8)

        # Cross-correlation
        correlation = correlate2d(img.data, template_norm, mode='same')

        # Normalize
        correlation = correlation / (img_std * template.shape[0] * template.shape[1])

        return correlation

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(edges: ImageGray, threshold: int) -> List[Tuple[float, float]]",
        deterministic=True,
        doc="Hough transform for line detection (simplified)"
    )
    def hough_lines(edges: ImageGray, threshold: int = 50) -> List[Tuple[float, float]]:
        """Hough transform for line detection (simplified)

        Args:
            edges: Binary edge map
            threshold: Minimum number of votes

        Returns:
            List of (rho, theta) line parameters
        """
        # Get edge points
        y_coords, x_coords = np.where(edges.data > 0.5)

        if len(x_coords) == 0:
            return []

        # Hough space parameters
        diagonal = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
        rhos = np.arange(-diagonal, diagonal + 1)
        thetas = np.deg2rad(np.arange(0, 180))

        # Accumulator
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)

        # Vote
        for x, y in zip(x_coords, y_coords):
            for theta_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = rho + diagonal
                if 0 <= rho_idx < len(rhos):
                    accumulator[rho_idx, theta_idx] += 1

        # Find peaks
        lines = []
        peaks = accumulator > threshold

        for rho_idx, theta_idx in zip(*np.where(peaks)):
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            lines.append((float(rho), float(theta)))

        return lines

    @staticmethod
    @operator(
        domain="vision",
        category=OpCategory.QUERY,
        signature="(img1: ImageGray, img2: ImageGray, window_size: int) -> Tuple[np.ndarray, np.ndarray]",
        deterministic=True,
        doc="Lucas-Kanade optical flow (simplified)"
    )
    def optical_flow_lucas_kanade(img1: ImageGray, img2: ImageGray,
                                  window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Lucas-Kanade optical flow (simplified)

        Args:
            img1: First image
            img2: Second image
            window_size: Window size for flow calculation

        Returns:
            Tuple of (flow_x, flow_y) arrays
        """
        # Temporal derivative
        It = img2.data - img1.data

        # Spatial derivatives
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = sobel_x.T

        Ix = convolve2d(img1.data, sobel_x, mode='same', boundary='symm')
        Iy = convolve2d(img1.data, sobel_y, mode='same', boundary='symm')

        # Initialize flow
        h, w = img1.shape
        flow_x = np.zeros((h, w))
        flow_y = np.zeros((h, w))

        half_win = window_size // 2

        for i in range(half_win, h - half_win):
            for j in range(half_win, w - half_win):
                # Extract window
                Ix_win = Ix[i-half_win:i+half_win+1, j-half_win:j+half_win+1].flatten()
                Iy_win = Iy[i-half_win:i+half_win+1, j-half_win:j+half_win+1].flatten()
                It_win = It[i-half_win:i+half_win+1, j-half_win:j+half_win+1].flatten()

                # Build matrix A and vector b
                A = np.column_stack((Ix_win, Iy_win))
                b = -It_win

                # Solve least squares
                try:
                    flow = np.linalg.lstsq(A, b, rcond=None)[0]
                    flow_x[i, j] = flow[0]
                    flow_y[i, j] = flow[1]
                except np.linalg.LinAlgError:
                    pass  # singular window (no texture) -> leave flow at 0

        return flow_x, flow_y


# Export singleton instance for DSL access
vision = VisionOperations()

# Export operators for domain registry discovery
create_image = VisionOperations.create_image
sobel = VisionOperations.sobel
laplacian = VisionOperations.laplacian
canny = VisionOperations.canny
harris_corners = VisionOperations.harris_corners
gaussian_blur = VisionOperations.gaussian_blur
morphological = VisionOperations.morphological
threshold = VisionOperations.threshold
adaptive_threshold = VisionOperations.adaptive_threshold
find_contours = VisionOperations.find_contours
template_match = VisionOperations.template_match
hough_lines = VisionOperations.hough_lines
optical_flow_lucas_kanade = VisionOperations.optical_flow_lucas_kanade
