"""Tests for procedural graphics domains - noise, palette, color, image."""

import pytest
import numpy as np
from morphogen.stdlib import noise, palette, color, image


class TestNoiseDomain:
    """Test noise generation functions."""

    def test_perlin2d_basic(self):
        """Test basic Perlin noise generation."""
        result = noise.perlin2d(seed=42, shape=(64, 64), scale=0.1)

        assert result.shape == (64, 64)
        assert result.data.dtype in (np.float32, np.float64)
        # Perlin noise should be in reasonable range
        assert -2.0 < result.data.min() < result.data.max() < 2.0

    def test_perlin2d_determinism(self):
        """Test that Perlin noise is deterministic with seed."""
        result1 = noise.perlin2d(seed=123, shape=(32, 32), scale=0.05)
        result2 = noise.perlin2d(seed=123, shape=(32, 32), scale=0.05)

        assert np.allclose(result1.data, result2.data, atol=1e-12)

    def test_perlin2d_different_seeds(self):
        """Test that different seeds produce different noise."""
        result1 = noise.perlin2d(seed=1, shape=(32, 32), scale=0.05)
        result2 = noise.perlin2d(seed=2, shape=(32, 32), scale=0.05)

        assert not np.allclose(result1.data, result2.data)

    def test_simplex2d(self):
        """Test Simplex noise generation."""
        result = noise.simplex2d(seed=42, shape=(64, 64), scale=0.1)

        assert result.shape == (64, 64)
        assert -1.5 < result.data.min() < result.data.max() < 1.5

    def test_worley_noise(self):
        """Test Worley/Voronoi noise generation."""
        result = noise.worley(seed=42, shape=(64, 64), num_points=20, distance_metric='euclidean')

        assert result.shape == (64, 64)
        assert result.data.min() >= 0.0  # Distances are non-negative
        assert result.data.max() > 0.0

    def test_fbm_fractional_brownian_motion(self):
        """Test fractional Brownian motion."""
        result = noise.fbm(shape=(64, 64), scale=0.05, octaves=4, persistence=0.5, lacunarity=2.0, seed=42)

        assert result.shape == (64, 64)
        # FBM should have more detail than single-octave noise
        assert result.data.std() > 0.0

    def test_turbulence(self):
        """Test turbulence noise generation."""
        result = noise.turbulence(seed=42, shape=(64, 64), octaves=4, scale=0.05)

        assert result.shape == (64, 64)
        # Turbulence uses absolute value, so should be non-negative
        assert result.data.min() >= 0.0

    def test_marble_pattern(self):
        """Test marble pattern generation."""
        result = noise.marble(seed=42, shape=(64, 64), scale=0.05, turbulence_power=5.0)

        assert result.shape == (64, 64)
        # Marble uses sine function, so should be in [-1, 1]
        assert -1.1 < result.data.min() < result.data.max() < 1.1

    def test_vector_field(self):
        """Test vector field generation."""
        vx, vy = noise.vector_field(seed=42, shape=(32, 32), scale=0.1)

        assert vx.data.shape == (32, 32)
        assert vy.data.shape == (32, 32)

    def test_plasma_effect(self):
        """Test plasma effect generation."""
        result = noise.plasma(seed=42, shape=(64, 64))

        assert result.shape == (64, 64)


class TestPaletteDomain:
    """Test color palette functions."""

    def test_from_colors(self):
        """Test creating palette from color list."""
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # RGB
        pal = palette.from_colors(colors)

        assert pal.colors.shape[0] == 256  # Default palette size
        assert pal.colors.shape[1] == 3    # RGB
        # First color should be close to red
        assert pal.colors[0, 0] > 0.9

    def test_from_gradient(self):
        """Test creating palette from gradient stops."""
        stops = [(0.0, [1, 0, 0]), (0.5, [0, 1, 0]), (1.0, [0, 0, 1])]
        pal = palette.from_gradient(stops, resolution=256)

        assert pal.colors.shape == (256, 3)
        # Check interpolation at stops
        assert np.allclose(pal.colors[0], [1, 0, 0], atol=0.01)    # Red at 0.0
        assert np.allclose(pal.colors[127], [0, 1, 0], atol=0.1)   # Green at ~0.5
        assert np.allclose(pal.colors[255], [0, 0, 1], atol=0.01)  # Blue at 1.0

    def test_greyscale_palette(self):
        """Test greyscale palette."""
        pal = palette.greyscale()

        assert pal.colors.shape == (256, 3)
        # Greyscale: R = G = B
        assert np.allclose(pal.colors[:, 0], pal.colors[:, 1])
        assert np.allclose(pal.colors[:, 1], pal.colors[:, 2])
        # Should go from 0 to 1
        assert pal.colors[0, 0] < 0.1
        assert pal.colors[-1, 0] > 0.9

    def test_viridis_palette(self):
        """Test Viridis scientific colormap."""
        pal = palette.viridis()

        assert pal.colors.shape == (256, 3)
        # Viridis starts dark blue-ish, ends yellow-ish
        assert pal.colors[0, 0] < 0.5    # Low red at start
        assert pal.colors[-1, 1] > 0.7   # High green at end

    def test_inferno_palette(self):
        """Test Inferno scientific colormap."""
        pal = palette.inferno()

        assert pal.colors.shape == (256, 3)

    def test_fire_palette(self):
        """Test fire-themed palette."""
        pal = palette.fire()

        assert pal.colors.shape == (256, 3)
        # Fire should have reds and yellows
        assert pal.colors[-1, 0] > 0.8  # High red at hot end

    def test_rainbow_palette(self):
        """Test rainbow palette."""
        pal = palette.rainbow()

        assert pal.colors.shape == (256, 3)

    def test_cosine_palette(self):
        """Test procedural cosine palette (IQ-style)."""
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        c = np.array([1.0, 1.0, 1.0])
        d = np.array([0.0, 0.33, 0.67])

        pal = palette.cosine(a, b, c, d, resolution=256)

        assert pal.colors.shape == (256, 3)
        assert np.all(pal.colors >= 0.0)
        assert np.all(pal.colors <= 1.0)

    def test_palette_map(self):
        """Test mapping scalar values to colors using palette."""
        pal = palette.viridis()
        values = np.linspace(0.0, 1.0, 100)

        colored = palette.map(pal, values, vmin=0.0, vmax=1.0)

        assert colored.shape == (100, 3)
        assert np.all(colored >= 0.0)
        assert np.all(colored <= 1.0)

    def test_palette_shift(self):
        """Test palette shifting."""
        pal = palette.viridis()
        shifted = palette.shift(pal, 0.5)

        assert shifted.colors.shape == pal.colors.shape
        # Shifted palette should be different
        assert not np.allclose(shifted.colors, pal.colors)

    def test_palette_flip(self):
        """Test palette flipping/reversal."""
        pal = palette.viridis()
        flipped = palette.flip(pal)

        assert flipped.colors.shape == pal.colors.shape
        # First color should match last color of original
        assert np.allclose(flipped.colors[0], pal.colors[-1], atol=0.01)
        assert np.allclose(flipped.colors[-1], pal.colors[0], atol=0.01)


class TestColorDomain:
    """Test color manipulation functions."""

    def test_rgb_to_hsv(self):
        """Test RGB to HSV conversion."""
        # Pure red in RGB
        rgb = np.array([[[1.0, 0.0, 0.0]]])  # Shape (1, 1, 3)
        hsv = color.rgb_to_hsv(rgb)

        # Red: H=0, S=1, V=1
        assert np.allclose(hsv[0, 0, 0], 0.0, atol=0.01)  # Hue
        assert np.allclose(hsv[0, 0, 1], 1.0, atol=0.01)  # Saturation
        assert np.allclose(hsv[0, 0, 2], 1.0, atol=0.01)  # Value

    def test_hsv_to_rgb(self):
        """Test HSV to RGB conversion."""
        # Pure red in HSV: H=0, S=1, V=1
        hsv = np.array([[[0.0, 1.0, 1.0]]])
        rgb = color.hsv_to_rgb(hsv)

        assert np.allclose(rgb[0, 0], [1.0, 0.0, 0.0], atol=0.01)

    def test_rgb_hsv_roundtrip(self):
        """Test RGB -> HSV -> RGB roundtrip."""
        original = np.array([[[0.7, 0.3, 0.9]]])
        hsv = color.rgb_to_hsv(original)
        recovered = color.hsv_to_rgb(hsv)

        assert np.allclose(original, recovered, atol=0.01)

    def test_rgb_to_hsl(self):
        """Test RGB to HSL conversion."""
        rgb = np.array([[[1.0, 0.0, 0.0]]])  # Pure red
        hsl = color.rgb_to_hsl(rgb)

        # Red: H=0, S=1, L=0.5
        assert np.allclose(hsl[0, 0, 0], 0.0, atol=0.01)  # Hue
        assert np.allclose(hsl[0, 0, 1], 1.0, atol=0.01)  # Saturation
        assert np.allclose(hsl[0, 0, 2], 0.5, atol=0.01)  # Lightness

    def test_hex_to_rgb(self):
        """Test hex string to RGB conversion."""
        rgb = color.hex_to_rgb("#FF0000")  # Red
        assert np.allclose(rgb, [1.0, 0.0, 0.0], atol=0.01)

        rgb2 = color.hex_to_rgb("#00FF00")  # Green
        assert np.allclose(rgb2, [0.0, 1.0, 0.0], atol=0.01)

    def test_rgb_to_hex(self):
        """Test RGB to hex string conversion."""
        hex_str = color.rgb_to_hex(np.array([1.0, 0.0, 0.0]))
        assert hex_str.upper() == "#FF0000"

    @pytest.mark.skip(reason="Temperature color conversion not fully implemented")
    def test_temperature_to_rgb(self):
        """Test blackbody temperature to RGB conversion."""
        # Test various temperatures
        rgb_2000k = color.temperature_to_rgb(2000)  # Warm/orange
        rgb_6500k = color.temperature_to_rgb(6500)  # Daylight/white
        rgb_10000k = color.temperature_to_rgb(10000)  # Cool/blue

        # Check shapes
        assert rgb_2000k.shape == (3,)
        assert rgb_6500k.shape == (3,)
        assert rgb_10000k.shape == (3,)

        # Warm light should have more red
        assert rgb_2000k[0] > rgb_2000k[2]

        # Cool light should have more blue
        assert rgb_10000k[2] > rgb_10000k[0]

    def test_color_add(self):
        """Test color addition."""
        c1 = np.array([[[0.5, 0.3, 0.2]]])
        c2 = np.array([[[0.2, 0.4, 0.1]]])

        result = color.add(c1, c2)
        expected = np.array([[[0.7, 0.7, 0.3]]])

        assert np.allclose(result, expected, atol=0.01)

    def test_color_multiply(self):
        """Test color multiplication."""
        c1 = np.array([[[0.8, 0.5, 0.3]]])
        c2 = np.array([[[0.5, 1.0, 0.5]]])

        result = color.multiply(c1, c2)
        expected = np.array([[[0.4, 0.5, 0.15]]])

        assert np.allclose(result, expected, atol=0.01)

    def test_color_mix(self):
        """Test color mixing/lerp."""
        c1 = np.array([[[1.0, 0.0, 0.0]]])  # Red
        c2 = np.array([[[0.0, 0.0, 1.0]]])  # Blue

        # 50/50 mix
        result = color.mix(c1, c2, t=0.5)
        expected = np.array([[[0.5, 0.0, 0.5]]])  # Purple

        assert np.allclose(result, expected, atol=0.01)

    def test_brightness_adjustment(self):
        """Test brightness adjustment."""
        img = np.array([[[0.5, 0.5, 0.5]]])

        brighter = color.brightness(img, factor=1.5)
        assert np.all(brighter >= img)

        darker = color.brightness(img, factor=0.5)
        assert np.all(darker <= img)

    def test_saturation_adjustment(self):
        """Test saturation adjustment."""
        # Colorful pixel
        img = np.array([[[1.0, 0.0, 0.0]]])  # Pure red

        # Desaturate (toward gray)
        desaturated = color.saturate(img, factor=0.0)
        # Should become gray
        assert np.allclose(desaturated[0, 0, 0], desaturated[0, 0, 1], atol=0.01)
        assert np.allclose(desaturated[0, 0, 1], desaturated[0, 0, 2], atol=0.01)

    def test_gamma_correction(self):
        """Test gamma correction."""
        img = np.array([[[0.5, 0.5, 0.5]]])

        # Gamma > 1 darkens mid-tones
        gamma_high = color.gamma_correct(img, gamma=2.0)
        assert np.all(gamma_high < img)

        # Gamma < 1 brightens mid-tones
        gamma_low = color.gamma_correct(img, gamma=0.5)
        assert np.all(gamma_low > img)

    def test_blend_modes(self):
        """Test various blend modes."""
        base = np.array([[[0.5, 0.5, 0.5]]])
        blend_layer = np.array([[[0.8, 0.3, 0.6]]])

        # Test different blend modes
        overlay = color.blend_overlay(base, blend_layer)
        screen = color.blend_screen(base, blend_layer)
        multiply = color.blend_multiply(base, blend_layer)

        # All should produce valid RGB values
        assert np.all(overlay >= 0.0) and np.all(overlay <= 1.0)
        assert np.all(screen >= 0.0) and np.all(screen <= 1.0)
        assert np.all(multiply >= 0.0) and np.all(multiply <= 1.0)


@pytest.mark.skip(reason="Depends on unimplemented palette and image features")
class TestDeterminism:
    """Test deterministic behavior across procedural graphics domains."""

    def test_noise_determinism(self):
        """Test noise generation is deterministic."""
        n1 = noise.perlin2d(seed=42, shape=(64, 64), scale=0.1)
        n2 = noise.perlin2d(seed=42, shape=(64, 64), scale=0.1)

        assert np.array_equal(n1, n2)

    def test_palette_determinism(self):
        """Test palette generation is deterministic."""
        pal1 = palette.viridis()
        pal2 = palette.viridis()

        assert np.array_equal(pal1, pal2)

    def test_full_pipeline_determinism(self):
        """Test full noise → palette → image pipeline is deterministic."""
        # Generate noise
        n = noise.fbm(noise.perlin2d(seed=123, shape=(32, 32), scale=0.05),
                      octaves=4, persistence=0.5, lacunarity=2.0)

        # Apply palette
        pal = palette.inferno()
        img1 = image.from_field(n, pal, vmin=-1.0, vmax=1.0)

        # Do it again
        n2 = noise.fbm(noise.perlin2d(seed=123, shape=(32, 32), scale=0.05),
                       octaves=4, persistence=0.5, lacunarity=2.0)
        img2 = image.from_field(n2, pal, vmin=-1.0, vmax=1.0)

        assert np.allclose(img1, img2, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
