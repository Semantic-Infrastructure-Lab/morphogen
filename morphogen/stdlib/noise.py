"""Noise operations for procedural generation.

This module provides comprehensive noise generation capabilities for:
- Procedural textures and terrain
- Fractal generation (Mandelbrot, Julia sets)
- Turbulence and wind fields
- Randomized simulation parameters
- Audio-reactive visual effects
- Scientific visualization

Supports multiple noise algorithms:
- Perlin noise (smooth gradient noise)
- Simplex noise (improved Perlin, fewer artifacts)
- Value noise (interpolated random values)
- Worley/Voronoi (cellular patterns)
- Fractal Brownian Motion (layered noise)
- Ridged/Turbulence variants
"""

from typing import Tuple, Optional, Callable
import numpy as np
from morphogen.core.operator import operator, OpCategory


class NoiseField2D:
    """2D noise field with NumPy backend.

    Represents a procedurally generated 2D noise pattern.
    """

    def __init__(self, data: np.ndarray, scale: float = 1.0):
        """Initialize noise field.

        Args:
            data: NumPy array of noise values (shape: (height, width))
            scale: Spatial scale factor
        """
        self.data = data
        self.scale = scale
        self.shape = data.shape

    def __repr__(self) -> str:
        return f"NoiseField2D(shape={self.shape}, scale={self.scale})"


class NoiseField3D:
    """3D noise field with NumPy backend.

    Represents a procedurally generated 3D noise volume.
    """

    def __init__(self, data: np.ndarray, scale: float = 1.0):
        """Initialize 3D noise field.

        Args:
            data: NumPy array of noise values (shape: (depth, height, width))
            scale: Spatial scale factor
        """
        self.data = data
        self.scale = scale
        self.shape = data.shape

    def __repr__(self) -> str:
        return f"NoiseField3D(shape={self.shape}, scale={self.scale})"


def _worley_compute_distances(coords, points, num_points, h, w, distance_metric):
    """Compute per-point distances for Worley noise."""
    distances = np.zeros((h, w, num_points), dtype=np.float32)
    for i, point in enumerate(points):
        dy = coords[:, :, 0] - point[0]
        dx = coords[:, :, 1] - point[1]
        if distance_metric == "euclidean":
            distances[:, :, i] = np.sqrt(dx**2 + dy**2)
        elif distance_metric == "manhattan":
            distances[:, :, i] = np.abs(dx) + np.abs(dy)
        elif distance_metric == "chebyshev":
            distances[:, :, i] = np.maximum(np.abs(dx), np.abs(dy))
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    return distances


def _worley_extract_feature(distances_sorted, feature, num_points):
    """Extract F1, F2, or F2-F1 feature from sorted Worley distances."""
    f1 = distances_sorted[:, :, 0]
    f2 = distances_sorted[:, :, 1] if num_points > 1 else f1
    if feature == "F1":
        return f1
    elif feature == "F2":
        return f2
    elif feature == "F2-F1":
        return f2 - f1
    else:
        raise ValueError(f"Unknown feature: {feature}")


class NoiseOperations:
    """Namespace for noise operations (accessed as 'noise' in DSL)."""

    # ============================================================================
    # LAYER 1: ATOMIC NOISE OPERATIONS
    # ============================================================================

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        """Perlin fade function: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return a + t * (b - a)

    @staticmethod
    def _gradient_2d(hash_val: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """2D gradient vector based on hash value."""
        h = hash_val & 3
        u = np.where(h < 2, x, -x)
        v = np.where((h == 0) | (h == 2), y, -y)
        return u + v

    @staticmethod
    def _gradient_3d(hash_val: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """3D gradient vector based on hash value."""
        h = hash_val & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    @staticmethod
    def _generate_permutation(seed: int = 0) -> np.ndarray:
        """Generate permutation table for noise."""
        rng = np.random.RandomState(seed)
        p = np.arange(256, dtype=np.int32)
        rng.shuffle(p)
        return np.concatenate([p, p])  # Double to avoid overflow

    # ============================================================================
    # LAYER 2: BASIC NOISE TYPES
    # ============================================================================

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, persistence: float, lacunarity: float, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate 2D Perlin noise"
    )
    def perlin2d(shape: Tuple[int, int],
                 scale: float = 1.0,
                 octaves: int = 1,
                 persistence: float = 0.5,
                 lacunarity: float = 2.0,
                 seed: int = 0) -> NoiseField2D:
        """Generate 2D Perlin noise.

        Perlin noise is smooth gradient noise widely used for:
        - Natural textures (clouds, wood, marble)
        - Terrain heightmaps
        - Turbulence effects

        Args:
            shape: Output shape (height, width)
            scale: Spatial frequency (smaller = larger features)
            octaves: Number of noise layers to combine
            persistence: Amplitude decay per octave (typically 0.5)
            lacunarity: Frequency increase per octave (typically 2.0)
            seed: Random seed for determinism

        Returns:
            NoiseField2D with values approximately in [-1, 1]
        """
        h, w = shape
        result = np.zeros((h, w), dtype=np.float32)

        # Generate permutation table
        perm = NoiseOperations._generate_permutation(seed)

        # Accumulate octaves
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            # Generate coordinate grids
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            x = x * frequency / (w * scale)
            y = y * frequency / (h * scale)

            # Integer and fractional parts
            xi = x.astype(np.int32) & 255
            yi = y.astype(np.int32) & 255
            xf = x - np.floor(x)
            yf = y - np.floor(y)

            # Fade curves
            u = NoiseOperations._fade(xf)
            v = NoiseOperations._fade(yf)

            # Hash coordinates
            aa = perm[perm[xi] + yi]
            ab = perm[perm[xi] + yi + 1]
            ba = perm[perm[xi + 1] + yi]
            bb = perm[perm[xi + 1] + yi + 1]

            # Gradient contributions
            g_aa = NoiseOperations._gradient_2d(aa, xf, yf)
            g_ba = NoiseOperations._gradient_2d(ba, xf - 1, yf)
            g_ab = NoiseOperations._gradient_2d(ab, xf, yf - 1)
            g_bb = NoiseOperations._gradient_2d(bb, xf - 1, yf - 1)

            # Bilinear interpolation
            x1 = NoiseOperations._lerp(g_aa, g_ba, u)
            x2 = NoiseOperations._lerp(g_ab, g_bb, u)
            noise = NoiseOperations._lerp(x1, x2, v)

            result += noise * amplitude
            max_value += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        # Normalize
        if max_value > 0:
            result /= max_value

        return NoiseField2D(result, scale)

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, persistence: float, lacunarity: float, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate 2D Simplex noise"
    )
    def simplex2d(shape: Tuple[int, int],
                  scale: float = 1.0,
                  octaves: int = 1,
                  persistence: float = 0.5,
                  lacunarity: float = 2.0,
                  seed: int = 0) -> NoiseField2D:
        """Generate 2D Simplex noise.

        Simplex noise is an improved version of Perlin with:
        - Fewer directional artifacts
        - Better visual isotropy
        - Lower computational complexity in higher dimensions

        Args:
            shape: Output shape (height, width)
            scale: Spatial frequency
            octaves: Number of noise layers
            persistence: Amplitude decay per octave
            lacunarity: Frequency increase per octave
            seed: Random seed

        Returns:
            NoiseField2D with values approximately in [-1, 1]
        """
        # Simplex implementation (simplified - using Perlin as fallback for MVP)
        # Full simplex requires skewing/unskewing of 2D space
        # For MVP, we use Perlin as a placeholder
        return NoiseOperations.perlin2d(shape, scale, octaves, persistence, lacunarity, seed)

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, persistence: float, lacunarity: float, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate 2D value noise"
    )
    def value2d(shape: Tuple[int, int],
                scale: float = 1.0,
                octaves: int = 1,
                persistence: float = 0.5,
                lacunarity: float = 2.0,
                seed: int = 0) -> NoiseField2D:
        """Generate 2D value noise.

        Value noise interpolates random values at grid points.
        Simpler than Perlin but can show grid artifacts.

        Args:
            shape: Output shape (height, width)
            scale: Spatial frequency
            octaves: Number of noise layers
            persistence: Amplitude decay per octave
            lacunarity: Frequency increase per octave
            seed: Random seed

        Returns:
            NoiseField2D with values in [0, 1]
        """
        h, w = shape
        result = np.zeros((h, w), dtype=np.float32)

        rng = np.random.RandomState(seed)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            # Generate coordinate grids
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            x = x * frequency / (w * scale)
            y = y * frequency / (h * scale)

            # Integer and fractional parts
            xi = np.floor(x).astype(np.int32)
            yi = np.floor(y).astype(np.int32)
            xf = x - xi
            yf = y - yi

            # Smooth interpolation
            u = NoiseOperations._fade(xf)
            v = NoiseOperations._fade(yf)

            # Random values at grid corners
            aa = rng.rand()
            ab = rng.rand()
            ba = rng.rand()
            bb = rng.rand()

            # Bilinear interpolation
            x1 = NoiseOperations._lerp(aa, ba, u)
            x2 = NoiseOperations._lerp(ab, bb, u)
            noise = NoiseOperations._lerp(x1, x2, v)

            result += noise * amplitude
            max_value += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        # Normalize to [0, 1]
        if max_value > 0:
            result /= max_value

        return NoiseField2D(result, scale)

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], num_points: int, distance_metric: str, feature: str, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate 2D Worley (Voronoi/cellular) noise"
    )
    def worley(shape: Tuple[int, int],
               num_points: int = 20,
               distance_metric: str = "euclidean",
               feature: str = "F1",
               seed: int = 0) -> NoiseField2D:
        """Generate 2D Worley (Voronoi/cellular) noise.

        Creates cellular patterns useful for:
        - Stone/rock textures
        - Water caustics
        - Organic cell patterns
        - Procedural cracks

        Args:
            shape: Output shape (height, width)
            num_points: Number of feature points
            distance_metric: "euclidean", "manhattan", or "chebyshev"
            feature: Which feature to return ("F1" = closest, "F2" = second closest, "F2-F1" = difference)
            seed: Random seed

        Returns:
            NoiseField2D with distance values
        """
        h, w = shape
        rng = np.random.RandomState(seed)

        # Generate random feature points
        points = rng.rand(num_points, 2) * np.array([h, w])

        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        coords = np.stack([y, x], axis=-1)

        distances_sorted = np.sort(
            _worley_compute_distances(coords, points, num_points, h, w, distance_metric),
            axis=2
        )
        result = _worley_extract_feature(distances_sorted, feature, num_points)

        # Normalize to approximately [0, 1]
        result = result / np.max(result) if np.max(result) > 0 else result

        return NoiseField2D(result, 1.0)

    # ============================================================================
    # LAYER 3: FRACTAL NOISE PATTERNS
    # ============================================================================

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, persistence: float, lacunarity: float, noise_type: str, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate Fractional Brownian Motion (fBm)"
    )
    def fbm(shape: Tuple[int, int],
            scale: float = 1.0,
            octaves: int = 6,
            persistence: float = 0.5,
            lacunarity: float = 2.0,
            noise_type: str = "perlin",
            seed: int = 0) -> NoiseField2D:
        """Generate Fractional Brownian Motion (fBm).

        fBm is layered noise with decreasing amplitude and increasing frequency.
        Essential for:
        - Natural terrain
        - Cloud formations
        - Procedural textures

        Args:
            shape: Output shape (height, width)
            scale: Base spatial frequency
            octaves: Number of noise layers (typically 4-8)
            persistence: Amplitude multiplier per octave (typically 0.5)
            lacunarity: Frequency multiplier per octave (typically 2.0)
            noise_type: "perlin", "simplex", or "value"
            seed: Random seed

        Returns:
            NoiseField2D with accumulated noise
        """
        if noise_type == "perlin":
            return NoiseOperations.perlin2d(shape, scale, octaves, persistence, lacunarity, seed)
        elif noise_type == "simplex":
            return NoiseOperations.simplex2d(shape, scale, octaves, persistence, lacunarity, seed)
        elif noise_type == "value":
            return NoiseOperations.value2d(shape, scale, octaves, persistence, lacunarity, seed)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, persistence: float, lacunarity: float, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate ridged multifractal noise"
    )
    def ridged_fbm(shape: Tuple[int, int],
                   scale: float = 1.0,
                   octaves: int = 6,
                   persistence: float = 0.5,
                   lacunarity: float = 2.0,
                   seed: int = 0) -> NoiseField2D:
        """Generate ridged multifractal noise.

        Creates sharp ridges by inverting and squaring noise.
        Perfect for:
        - Mountain ranges
        - Rock formations
        - Erosion patterns

        Args:
            shape: Output shape
            scale: Spatial frequency
            octaves: Number of layers
            persistence: Amplitude decay
            lacunarity: Frequency increase
            seed: Random seed

        Returns:
            NoiseField2D with ridged patterns
        """
        # Generate base noise
        noise = NoiseOperations.perlin2d(shape, scale, octaves, persistence, lacunarity, seed)

        # Ridge transformation: 1 - |2*noise - 1|
        result = 1.0 - np.abs(noise.data)
        result = result ** 2  # Sharpen ridges

        return NoiseField2D(result, scale)

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, persistence: float, lacunarity: float, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate turbulence noise"
    )
    def turbulence(shape: Tuple[int, int],
                   scale: float = 1.0,
                   octaves: int = 6,
                   persistence: float = 0.5,
                   lacunarity: float = 2.0,
                   seed: int = 0) -> NoiseField2D:
        """Generate turbulence noise.

        Uses absolute values of noise for swirling patterns.
        Useful for:
        - Marble textures
        - Fire/smoke effects
        - Wood grain

        Args:
            shape: Output shape
            scale: Spatial frequency
            octaves: Number of layers
            persistence: Amplitude decay
            lacunarity: Frequency increase
            seed: Random seed

        Returns:
            NoiseField2D with turbulent patterns
        """
        # Generate base noise
        noise = NoiseOperations.perlin2d(shape, scale, octaves, persistence, lacunarity, seed)

        # Turbulence: sum of absolute values
        result = np.abs(noise.data)

        return NoiseField2D(result, scale)

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, turbulence_power: float, seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate marble-like patterns"
    )
    def marble(shape: Tuple[int, int],
               scale: float = 1.0,
               turbulence_power: float = 5.0,
               seed: int = 0) -> NoiseField2D:
        """Generate marble-like patterns.

        Combines sine waves with turbulence for marble veining.

        Args:
            shape: Output shape
            scale: Spatial frequency
            turbulence_power: Strength of turbulence distortion
            seed: Random seed

        Returns:
            NoiseField2D with marble patterns
        """
        h, w = shape

        # Generate turbulence
        turb = NoiseOperations.turbulence(shape, scale, octaves=4, seed=seed)

        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Sine wave pattern with turbulence distortion
        xy_value = (x + y) / (w + h) * 20.0 + turb.data * turbulence_power
        result = (np.sin(xy_value) + 1.0) * 0.5

        return NoiseField2D(result, scale)

    # ============================================================================
    # LAYER 4: VECTOR FIELDS & ADVANCED
    # ============================================================================

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, seed: int) -> Tuple[NoiseField2D, NoiseField2D]",
        deterministic=True,
        doc="Generate 2D vector field using noise"
    )
    def vector_field(shape: Tuple[int, int],
                     scale: float = 1.0,
                     octaves: int = 4,
                     seed: int = 0) -> Tuple[NoiseField2D, NoiseField2D]:
        """Generate 2D vector field using noise.

        Creates two noise fields for x and y components.
        Useful for:
        - Flow fields
        - Particle advection
        - Wind patterns

        Args:
            shape: Output shape
            scale: Spatial frequency
            octaves: Number of noise layers
            seed: Random seed

        Returns:
            Tuple of (vx_field, vy_field)
        """
        # Generate independent noise for each component
        vx = NoiseOperations.perlin2d(shape, scale, octaves, seed=seed)
        vy = NoiseOperations.perlin2d(shape, scale, octaves, seed=seed + 1000)

        return vx, vy

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], scale: float, octaves: int, seed: int) -> Tuple[NoiseField2D, NoiseField2D]",
        deterministic=True,
        doc="Generate gradient field from noise"
    )
    def gradient_field(shape: Tuple[int, int],
                       scale: float = 1.0,
                       octaves: int = 4,
                       seed: int = 0) -> Tuple[NoiseField2D, NoiseField2D]:
        """Generate gradient field from noise.

        Computes spatial derivatives of noise for flow patterns.

        Args:
            shape: Output shape
            scale: Spatial frequency
            octaves: Number of noise layers
            seed: Random seed

        Returns:
            Tuple of (grad_x, grad_y) fields
        """
        # Generate base noise
        noise = NoiseOperations.perlin2d(shape, scale, octaves, seed=seed)

        # Compute gradients
        grad_y, grad_x = np.gradient(noise.data)

        return NoiseField2D(grad_x, scale), NoiseField2D(grad_y, scale)

    @staticmethod
    @operator(
        domain="noise",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], seed: int) -> NoiseField2D",
        deterministic=True,
        doc="Generate plasma effect using diamond-square algorithm"
    )
    def plasma(shape: Tuple[int, int],
               seed: int = 0) -> NoiseField2D:
        """Generate plasma effect using diamond-square algorithm.

        Classic fractal noise technique for:
        - Retro plasma effects
        - Height maps
        - Procedural terrain

        Args:
            shape: Output shape (must be power of 2 + 1, e.g., 257x257)
            seed: Random seed

        Returns:
            NoiseField2D with plasma pattern
        """
        h, w = shape
        rng = np.random.RandomState(seed)

        # Initialize with corners
        data = np.zeros(shape, dtype=np.float32)
        data[0, 0] = rng.rand()
        data[0, -1] = rng.rand()
        data[-1, 0] = rng.rand()
        data[-1, -1] = rng.rand()

        # Diamond-square iterations
        step = max(h, w) - 1
        scale = 1.0

        while step > 1:
            half_step = step // 2

            # Diamond step
            for y in range(half_step, h, step):
                for x in range(half_step, w, step):
                    avg = (data[y - half_step, x - half_step] +
                          data[y - half_step, min(x + half_step, w-1)] +
                          data[min(y + half_step, h-1), x - half_step] +
                          data[min(y + half_step, h-1), min(x + half_step, w-1)]) / 4.0
                    data[y, x] = avg + rng.uniform(-scale, scale)

            # Square step
            for y in range(0, h, half_step):
                for x in range((y + half_step) % step, w, step):
                    count = 0
                    total = 0.0
                    if y >= half_step:
                        total += data[y - half_step, x]
                        count += 1
                    if y + half_step < h:
                        total += data[y + half_step, x]
                        count += 1
                    if x >= half_step:
                        total += data[y, x - half_step]
                        count += 1
                    if x + half_step < w:
                        total += data[y, x + half_step]
                        count += 1

                    if count > 0:
                        data[y, x] = total / count + rng.uniform(-scale, scale)

            step = half_step
            scale *= 0.5

        # Normalize
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

        return NoiseField2D(data, 1.0)


# Create singleton instance for use as 'noise' namespace
noise = NoiseOperations()

# Export operators for domain registry discovery
value2d = NoiseOperations.value2d
perlin2d = NoiseOperations.perlin2d
simplex2d = NoiseOperations.simplex2d
worley = NoiseOperations.worley
fbm = NoiseOperations.fbm
ridged_fbm = NoiseOperations.ridged_fbm
turbulence = NoiseOperations.turbulence
marble = NoiseOperations.marble
plasma = NoiseOperations.plasma
vector_field = NoiseOperations.vector_field
gradient_field = NoiseOperations.gradient_field
