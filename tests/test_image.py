"""Comprehensive tests for morphogen.stdlib.image.

Tests the Image class and all ImageOperations methods,
checking both correctness of output values and error handling.
"""

import pytest
import numpy as np
import tempfile
import os

# Import through stdlib to ensure color module is available
# (blend() uses sys.modules["morphogen.stdlib.color"])
from morphogen.stdlib import image, Image


# ============================================================================
# Helpers
# ============================================================================

def make_rgb(h=32, w=32, fill=0.5):
    """Make a solid RGB Image."""
    data = np.full((h, w, 3), fill, dtype=np.float32)
    return Image(data)


def make_rgba(h=32, w=32, rgb=0.5, alpha=1.0):
    """Make a solid RGBA Image."""
    data = np.full((h, w, 4), rgb, dtype=np.float32)
    data[:, :, 3] = alpha
    return Image(data)


# ============================================================================
# Image class
# ============================================================================

class TestImageClass:
    def test_init_rgb(self):
        data = np.zeros((10, 20, 3), dtype=np.float32)
        img = Image(data)
        assert img.shape == (10, 20)
        assert img.channels == 3
        assert img.height == 10
        assert img.width == 20

    def test_init_rgba(self):
        data = np.zeros((8, 16, 4), dtype=np.float32)
        img = Image(data)
        assert img.channels == 4

    def test_init_converts_to_float32(self):
        data = np.ones((4, 4, 3), dtype=np.float64)
        img = Image(data)
        assert img.data.dtype == np.float32

    def test_init_rejects_2d(self):
        with pytest.raises(ValueError, match="3D"):
            Image(np.zeros((10, 10)))

    def test_init_rejects_4d(self):
        with pytest.raises(ValueError, match="3D"):
            Image(np.zeros((10, 10, 3, 1)))

    def test_init_rejects_2_channels(self):
        with pytest.raises(ValueError, match="3.*4.*channels"):
            Image(np.zeros((10, 10, 2)))

    def test_init_rejects_5_channels(self):
        with pytest.raises(ValueError, match="3.*4.*channels"):
            Image(np.zeros((10, 10, 5)))

    def test_copy_is_independent(self):
        img = make_rgb(fill=0.5)
        copy = img.copy()
        copy.data[0, 0, 0] = 0.9
        assert img.data[0, 0, 0] == pytest.approx(0.5)

    def test_repr(self):
        img = make_rgb(h=10, w=20)
        r = repr(img)
        assert "Image" in r
        assert "10" in r
        assert "20" in r


# ============================================================================
# Creation operators
# ============================================================================

class TestCreation:
    def test_blank_defaults(self):
        img = image.blank(width=64, height=48)
        assert isinstance(img, Image)
        assert img.width == 64
        assert img.height == 48
        assert img.channels == 3
        assert np.all(img.data == 0.0)

    def test_blank_fill_value(self):
        img = image.blank(width=10, height=10, fill_value=0.7)
        assert np.allclose(img.data, 0.7)

    def test_blank_rgba(self):
        img = image.blank(width=10, height=10, channels=4)
        assert img.channels == 4

    def test_rgb_values(self):
        img = image.rgb(r=1.0, g=0.5, b=0.0, width=10, height=8)
        assert img.shape == (8, 10)
        assert img.channels == 3
        assert np.allclose(img.data[:, :, 0], 1.0)
        assert np.allclose(img.data[:, :, 1], 0.5)
        assert np.allclose(img.data[:, :, 2], 0.0)

    def test_from_field_greyscale(self):
        field = np.linspace(0, 1, 64 * 64).reshape(64, 64)
        img = image.from_field(field)
        assert isinstance(img, Image)
        assert img.shape == (64, 64)
        assert img.channels == 3
        # All channels equal (greyscale)
        assert np.allclose(img.data[:, :, 0], img.data[:, :, 1])
        assert np.allclose(img.data[:, :, 1], img.data[:, :, 2])

    def test_from_field_normalized_to_01(self):
        # Uniform field → all zeros (max == min branch)
        field = np.ones((16, 16))
        img = image.from_field(field)
        assert np.all(img.data == 0.0)

    def test_from_field_with_palette(self):
        from morphogen.stdlib import palette
        field = np.random.default_rng(0).random((32, 32))
        pal = palette.viridis()
        img = image.from_field(field, palette=pal)
        assert img.shape == (32, 32)
        assert np.all(img.data >= 0.0)
        assert np.all(img.data <= 1.0)

    def test_from_field_rejects_non_2d(self):
        with pytest.raises(ValueError):
            image.from_field(np.ones((4, 4, 3)))

    def test_compose_rgb(self):
        rng = np.random.default_rng(42)
        r = rng.random((16, 16))
        g = rng.random((16, 16))
        b = rng.random((16, 16))
        img = image.compose(r, g, b)
        assert isinstance(img, Image)
        assert img.channels == 3
        assert np.all(img.data >= 0.0)
        assert np.all(img.data <= 1.0)

    def test_compose_rgba(self):
        rng = np.random.default_rng(0)
        r, g, b = rng.random((3, 16, 16))
        a = rng.random((16, 16))
        img = image.compose(r, g, b, a_channel=a)
        assert img.channels == 4

    def test_compose_normalizes_channels_independently(self):
        # R channel in [0, 2], G channel in [0, 1]
        # Both should be normalized to [0, 1]
        r = np.linspace(0, 2, 16).reshape(4, 4)
        g = np.linspace(0, 1, 16).reshape(4, 4)
        b = np.linspace(0, 1, 16).reshape(4, 4)
        img = image.compose(r, g, b)
        assert img.data[:, :, 0].max() == pytest.approx(1.0, abs=1e-5)
        assert img.data[:, :, 1].max() == pytest.approx(1.0, abs=1e-5)


# ============================================================================
# Transformations
# ============================================================================

class TestTransformations:
    def test_scale_up(self):
        img = make_rgb(h=16, w=16)
        scaled = image.scale(img, factor=2.0)
        assert scaled.height == 32
        assert scaled.width == 32
        assert scaled.channels == 3

    def test_scale_down(self):
        img = make_rgb(h=32, w=32)
        scaled = image.scale(img, factor=0.5)
        assert scaled.height == 16
        assert scaled.width == 16

    def test_scale_nearest(self):
        img = make_rgb(h=10, w=10, fill=0.3)
        scaled = image.scale(img, factor=2.0, method="nearest")
        assert scaled.shape == (20, 20)

    def test_scale_invalid_method(self):
        img = make_rgb()
        with pytest.raises(ValueError, match="interpolation"):
            image.scale(img, factor=1.0, method="cubic")

    def test_scale_preserves_values(self):
        # Solid image — scaling should not change pixel values
        img = image.rgb(r=0.3, g=0.6, b=0.9, width=8, height=8)
        scaled = image.scale(img, factor=2.0)
        assert np.allclose(scaled.data[:, :, 0], 0.3, atol=0.05)
        assert np.allclose(scaled.data[:, :, 1], 0.6, atol=0.05)

    def test_rotate_preserves_shape(self):
        img = make_rgb(h=16, w=32)
        rotated = image.rotate(img, angle=45.0, reshape=False)
        assert rotated.shape == (16, 32)

    def test_rotate_90_same_shape(self):
        img = make_rgb(h=20, w=20)
        rotated = image.rotate(img, angle=90.0)
        assert rotated.shape == (20, 20)

    def test_rotate_preserves_channels(self):
        img = make_rgba(h=16, w=16)
        rotated = image.rotate(img, angle=30.0)
        assert rotated.channels == 4

    def test_warp_zero_displacement(self):
        img = make_rgb(h=16, w=16, fill=0.5)
        dy = np.zeros((16, 16))
        dx = np.zeros((16, 16))
        warped = image.warp(img, (dy, dx))
        assert warped.shape == img.shape
        # Zero displacement → same values
        assert np.allclose(warped.data, img.data, atol=0.01)

    def test_warp_preserves_shape(self):
        img = make_rgb(h=32, w=32)
        dy = np.random.default_rng(0).random((32, 32)) * 2
        dx = np.random.default_rng(1).random((32, 32)) * 2
        warped = image.warp(img, (dy, dx))
        assert warped.shape == (32, 32)
        assert warped.channels == 3


# ============================================================================
# Filters
# ============================================================================

class TestFilters:
    def test_blur_reduces_edge_sharpness(self):
        # Sharp edge: left half black, right half white
        data = np.zeros((32, 32, 3), dtype=np.float32)
        data[:, 16:, :] = 1.0
        img = Image(data)
        blurred = image.blur(img, sigma=2.0)
        # The edge pixel at column 16 should be intermediate (not 0 or 1)
        edge_val = blurred.data[16, 16, 0]
        assert 0.2 < edge_val < 0.8

    def test_blur_preserves_shape(self):
        img = make_rgb(h=24, w=24)
        blurred = image.blur(img, sigma=1.0)
        assert blurred.shape == (24, 24)
        assert blurred.channels == 3

    def test_blur_uniform_unchanged(self):
        img = image.blank(width=16, height=16, fill_value=0.5)
        blurred = image.blur(img, sigma=2.0)
        # Uniform image stays uniform after blur
        assert np.allclose(blurred.data, 0.5, atol=0.01)

    def test_sharpen_output_in_range(self):
        img = make_rgb(h=16, w=16, fill=0.5)
        sharpened = image.sharpen(img, strength=2.0)
        assert np.all(sharpened.data >= 0.0)
        assert np.all(sharpened.data <= 1.0)

    def test_sharpen_preserves_shape(self):
        img = make_rgba(h=16, w=16)
        sharpened = image.sharpen(img, strength=1.0)
        assert sharpened.shape == (16, 16)

    def test_edge_detect_sobel(self):
        # Vertical edge at center
        data = np.zeros((32, 32, 3), dtype=np.float32)
        data[:, 16:, :] = 1.0
        img = Image(data)
        edges = image.edge_detect(img, method="sobel")
        # Edge columns should have high values
        edge_peak = edges.data[:, 15:17, 0].max()
        interior = edges.data[:, :8, 0].max()
        assert edge_peak > interior + 0.1

    def test_edge_detect_prewitt(self):
        data = np.zeros((16, 16, 3), dtype=np.float32)
        data[:, 8:, :] = 1.0
        img = Image(data)
        edges = image.edge_detect(img, method="prewitt")
        assert edges.shape == (16, 16)
        assert np.all(edges.data >= 0.0)
        assert np.all(edges.data <= 1.0)

    def test_edge_detect_laplacian(self):
        data = np.zeros((16, 16, 3), dtype=np.float32)
        data[4:12, 4:12, :] = 1.0
        img = Image(data)
        edges = image.edge_detect(img, method="laplacian")
        assert edges.shape == (16, 16)

    def test_edge_detect_uniform_is_near_zero(self):
        img = image.blank(width=16, height=16, fill_value=0.5)
        edges = image.edge_detect(img)
        # Uniform image → no edges
        assert np.all(edges.data < 0.01)

    def test_edge_detect_invalid_method(self):
        img = make_rgb()
        with pytest.raises(ValueError, match="method"):
            image.edge_detect(img, method="canny")

    def test_erode_shrinks_white_region(self):
        data = np.zeros((32, 32, 3), dtype=np.float32)
        data[8:24, 8:24, :] = 1.0
        img = Image(data)
        eroded = image.erode(img, iterations=2)
        assert eroded.data.sum() < img.data.sum()

    def test_dilate_grows_white_region(self):
        data = np.zeros((32, 32, 3), dtype=np.float32)
        data[8:24, 8:24, :] = 1.0
        img = Image(data)
        dilated = image.dilate(img, iterations=2)
        assert dilated.data.sum() > img.data.sum()

    def test_erode_dilate_preserve_shape(self):
        img = make_rgb(h=20, w=20)
        assert image.erode(img).shape == (20, 20)
        assert image.dilate(img).shape == (20, 20)


# ============================================================================
# Compositing / blending
# ============================================================================

class TestCompositing:
    def test_blend_normal_opacity_1(self):
        # opacity=1 → result is img_b
        a = image.rgb(0.2, 0.2, 0.2, 16, 16)
        b = image.rgb(0.8, 0.8, 0.8, 16, 16)
        result = image.blend(a, b, mode="normal", opacity=1.0)
        assert np.allclose(result.data, b.data, atol=1e-4)

    def test_blend_normal_opacity_0(self):
        # opacity=0 → result is img_a
        a = image.rgb(0.2, 0.2, 0.2, 16, 16)
        b = image.rgb(0.8, 0.8, 0.8, 16, 16)
        result = image.blend(a, b, mode="normal", opacity=0.0)
        assert np.allclose(result.data, a.data, atol=1e-4)

    def test_blend_multiply_correctness(self):
        a = image.rgb(0.5, 0.5, 0.5, 4, 4)
        b = image.rgb(0.5, 0.5, 0.5, 4, 4)
        result = image.blend(a, b, mode="multiply", opacity=1.0)
        # multiply: 0.5 * 0.5 = 0.25
        assert np.allclose(result.data, 0.25, atol=0.01)

    def test_blend_add_clips(self):
        a = image.rgb(0.8, 0.8, 0.8, 4, 4)
        b = image.rgb(0.8, 0.8, 0.8, 4, 4)
        result = image.blend(a, b, mode="add", opacity=1.0)
        # 0.8 + 0.8 = 1.6 → clipped to 1.0
        assert np.allclose(result.data, 1.0, atol=0.01)

    def test_blend_subtract(self):
        a = image.rgb(0.8, 0.8, 0.8, 4, 4)
        b = image.rgb(0.3, 0.3, 0.3, 4, 4)
        result = image.blend(a, b, mode="subtract", opacity=1.0)
        # 0.8 - 0.3 = 0.5
        assert np.allclose(result.data, 0.5, atol=0.02)

    def test_blend_subtract_clips_to_zero(self):
        a = image.rgb(0.2, 0.2, 0.2, 4, 4)
        b = image.rgb(0.9, 0.9, 0.9, 4, 4)
        result = image.blend(a, b, mode="subtract", opacity=1.0)
        # 0.2 - 0.9 = -0.7 → clipped to 0
        assert np.allclose(result.data, 0.0, atol=0.01)

    def test_blend_screen(self):
        a = image.rgb(0.5, 0.5, 0.5, 4, 4)
        b = image.rgb(0.5, 0.5, 0.5, 4, 4)
        result = image.blend(a, b, mode="screen", opacity=1.0)
        # screen: 1 - (1-a)(1-b) = 1 - 0.25 = 0.75
        assert np.allclose(result.data, 0.75, atol=0.01)

    def test_blend_difference(self):
        a = image.rgb(0.7, 0.7, 0.7, 4, 4)
        b = image.rgb(0.3, 0.3, 0.3, 4, 4)
        result = image.blend(a, b, mode="difference", opacity=1.0)
        # difference: |0.7 - 0.3| = 0.4
        assert np.allclose(result.data, 0.4, atol=0.02)

    def test_blend_overlay(self):
        a = make_rgb(fill=0.5)
        b = make_rgb(fill=0.5)
        result = image.blend(a, b, mode="overlay")
        assert result.shape == (32, 32)
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)

    def test_blend_soft_light(self):
        a = make_rgb(fill=0.4)
        b = make_rgb(fill=0.6)
        result = image.blend(a, b, mode="soft_light")
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)

    def test_blend_mismatched_shapes_raises(self):
        a = make_rgb(h=16, w=16)
        b = make_rgb(h=8, w=8)
        with pytest.raises(ValueError, match="dimensions"):
            image.blend(a, b)

    def test_blend_unknown_mode_raises(self):
        a = make_rgb()
        b = make_rgb()
        with pytest.raises(ValueError, match="blend mode"):
            image.blend(a, b, mode="dissolve")

    def test_blend_preserves_alpha(self):
        a = make_rgba(h=8, w=8, rgb=0.3, alpha=0.7)
        b = make_rgba(h=8, w=8, rgb=0.7, alpha=0.7)
        result = image.blend(a, b, mode="normal", opacity=1.0)
        assert result.channels == 4
        # Alpha should be carried through from img_a
        assert np.allclose(result.data[:, :, 3], 0.7, atol=0.01)

    def test_overlay_with_full_mask(self):
        base = image.rgb(0.1, 0.1, 0.1, 16, 16)
        over = image.rgb(0.9, 0.9, 0.9, 16, 16)
        mask = np.ones((16, 16), dtype=np.float32)
        result = image.overlay(base, over, mask=mask)
        # Full mask → entirely overlay
        assert np.allclose(result.data[:, :, :3], over.data, atol=0.01)

    def test_overlay_with_zero_mask(self):
        base = image.rgb(0.1, 0.1, 0.1, 16, 16)
        over = image.rgb(0.9, 0.9, 0.9, 16, 16)
        mask = np.zeros((16, 16), dtype=np.float32)
        result = image.overlay(base, over, mask=mask)
        # Zero mask → entirely base
        assert np.allclose(result.data[:, :, :3], base.data, atol=0.01)

    def test_overlay_uses_alpha_when_no_mask(self):
        base = image.rgb(0.0, 0.0, 0.0, 8, 8)
        # RGBA overlay with 50% alpha
        data = np.ones((8, 8, 4), dtype=np.float32) * 1.0
        data[:, :, 3] = 0.5
        over = Image(data)
        result = image.overlay(base, over)
        # Result should be 50% between 0 and 1
        assert np.allclose(result.data[:, :, 0], 0.5, atol=0.01)

    def test_overlay_mismatched_shapes_raises(self):
        base = make_rgb(h=16, w=16)
        over = make_rgb(h=8, w=8)
        with pytest.raises(ValueError):
            image.overlay(base, over)

    def test_alpha_composite_full_alpha(self):
        bg = image.rgb(0.0, 0.0, 0.0, 8, 8)
        fg = make_rgba(h=8, w=8, rgb=1.0, alpha=1.0)
        result = image.alpha_composite(bg, fg)
        # alpha=1 → fully foreground
        assert np.allclose(result.data[:, :, :3], 1.0, atol=0.01)

    def test_alpha_composite_zero_alpha(self):
        bg = image.rgb(0.3, 0.3, 0.3, 8, 8)
        fg = make_rgba(h=8, w=8, rgb=1.0, alpha=0.0)
        result = image.alpha_composite(bg, fg)
        # alpha=0 → fully background
        assert np.allclose(result.data[:, :, :3], 0.3, atol=0.01)

    def test_alpha_composite_half_alpha(self):
        bg = image.rgb(0.0, 0.0, 0.0, 8, 8)
        fg = make_rgba(h=8, w=8, rgb=1.0, alpha=0.5)
        result = image.alpha_composite(bg, fg)
        # 50% alpha blend
        assert np.allclose(result.data[:, :, 0], 0.5, atol=0.01)

    def test_alpha_composite_requires_rgba_foreground(self):
        bg = make_rgb()
        fg = make_rgb()  # RGB, not RGBA
        with pytest.raises(ValueError, match="alpha"):
            image.alpha_composite(bg, fg)

    def test_alpha_composite_mismatched_shapes_raises(self):
        bg = make_rgb(h=16, w=16)
        fg = make_rgba(h=8, w=8)
        with pytest.raises(ValueError):
            image.alpha_composite(bg, fg)

    def test_alpha_composite_rgba_background_preserves_alpha(self):
        bg = make_rgba(h=8, w=8, rgb=0.0, alpha=1.0)
        fg = make_rgba(h=8, w=8, rgb=1.0, alpha=0.5)
        result = image.alpha_composite(bg, fg)
        assert result.channels == 4


# ============================================================================
# Procedural effects
# ============================================================================

class TestProceduralEffects:
    def test_apply_palette_luminance(self):
        from morphogen.stdlib import palette
        img = make_rgb(h=16, w=16, fill=0.5)
        pal = palette.viridis()
        result = image.apply_palette(img, pal, channel="luminance")
        assert isinstance(result, Image)
        assert result.shape == (16, 16)

    def test_apply_palette_r_channel(self):
        from morphogen.stdlib import palette
        img = image.rgb(0.8, 0.2, 0.5, 16, 16)
        pal = palette.inferno()
        result = image.apply_palette(img, pal, channel="r")
        assert result.shape == (16, 16)

    def test_apply_palette_g_channel(self):
        from morphogen.stdlib import palette
        img = make_rgb(fill=0.5)
        pal = palette.viridis()
        result = image.apply_palette(img, pal, channel="g")
        assert result.shape == img.shape

    def test_apply_palette_b_channel(self):
        from morphogen.stdlib import palette
        img = make_rgb(fill=0.5)
        pal = palette.viridis()
        result = image.apply_palette(img, pal, channel="b")
        assert result.shape == img.shape

    def test_apply_palette_saturation(self):
        from morphogen.stdlib import palette
        img = image.rgb(0.8, 0.2, 0.2, 16, 16)
        pal = palette.viridis()
        result = image.apply_palette(img, pal, channel="saturation")
        assert result.shape == (16, 16)

    def test_apply_palette_invalid_channel(self):
        from morphogen.stdlib import palette
        img = make_rgb()
        pal = palette.viridis()
        with pytest.raises(ValueError, match="channel"):
            image.apply_palette(img, pal, channel="hue")

    def test_apply_palette_output_in_range(self):
        from morphogen.stdlib import palette
        rng = np.random.default_rng(42)
        data = rng.random((16, 16, 3)).astype(np.float32)
        img = Image(data)
        pal = palette.viridis()
        result = image.apply_palette(img, pal)
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)

    def test_normal_map_shape(self):
        heightfield = np.random.default_rng(0).random((32, 32))
        normals = image.normal_map_from_heightfield(heightfield)
        assert isinstance(normals, Image)
        assert normals.shape == (32, 32)
        assert normals.channels == 3

    def test_normal_map_flat_surface(self):
        # Flat heightfield → normals all pointing up (0, 0, 1) → RGB ≈ (0.5, 0.5, 1.0)
        heightfield = np.ones((16, 16))
        normals = image.normal_map_from_heightfield(heightfield)
        # B channel (Z normal) should be ≈ 1.0 → maps to ≈ 1.0 in RGB encoding
        assert np.allclose(normals.data[:, :, 2], 1.0, atol=0.01)

    def test_normal_map_in_range(self):
        heightfield = np.random.default_rng(0).random((32, 32))
        normals = image.normal_map_from_heightfield(heightfield)
        assert np.all(normals.data >= 0.0)
        assert np.all(normals.data <= 1.0)

    def test_gradient_map_aliases_apply_palette(self):
        from morphogen.stdlib import palette
        img = make_rgb(h=16, w=16, fill=0.5)
        pal = palette.viridis()
        result_gradient = image.gradient_map(img, pal)
        result_apply = image.apply_palette(img, pal, channel="luminance")
        assert np.allclose(result_gradient.data, result_apply.data)


# ============================================================================
# Save operation
# ============================================================================

class TestSave:
    def test_save_png_image_object(self, tmp_path):
        img = image.rgb(0.5, 0.3, 0.8, 32, 32)
        path = str(tmp_path / "test.png")
        image.save(img, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_save_png_numpy_array(self, tmp_path):
        data = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
        path = str(tmp_path / "array.png")
        image.save(data, path)
        assert os.path.exists(path)

    def test_save_jpeg(self, tmp_path):
        img = image.rgb(0.2, 0.5, 0.9, 16, 16)
        path = str(tmp_path / "test.jpg")
        image.save(img, path)
        assert os.path.exists(path)

    def test_save_roundtrip(self, tmp_path):
        from morphogen.stdlib.io_storage import load_image
        img = image.rgb(0.0, 1.0, 0.5, 32, 32)
        path = str(tmp_path / "roundtrip.png")
        image.save(img, path)
        loaded = load_image(path)
        # Allow tolerance for uint8 quantization
        assert np.allclose(img.data, loaded, atol=0.01)
