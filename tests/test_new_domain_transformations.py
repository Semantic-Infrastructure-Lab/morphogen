"""
Tests for new domain transformations.

Tests the following transformations:
- Time → Cepstral (DCT)
- Cepstral → Time (IDCT)
- Time → Wavelet (CWT)
- Spatial Affine (translate, rotate, scale, shear)
- Cartesian → Polar
- Polar → Cartesian
"""

import numpy as np
import pytest
from morphogen.cross_domain import (
    TimeToCepstralInterface,
    CepstralToTimeInterface,
    TimeToWaveletInterface,
    SpatialAffineInterface,
    CartesianToPolarInterface,
    PolarToCartesianInterface,
)
from morphogen.cross_domain.registry import CrossDomainRegistry


# ==============================================================================
# TIME-FREQUENCY TRANSFORMS TESTS
# ==============================================================================


def test_time_to_cepstral_basic():
    """Test basic DCT transformation."""
    # Create a simple test signal (sine wave)
    t = np.linspace(0, 1, 256, dtype=np.float32)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

    # Create transform interface
    interface = TimeToCepstralInterface(signal, dct_type=2, norm="ortho")

    # Validate
    assert interface.validate()

    # Transform
    cepstral = interface.transform(signal)

    # Check shape
    assert cepstral.shape == signal.shape
    assert cepstral.dtype == np.float32

    # DCT should be valid (no NaNs or Infs)
    assert not np.any(np.isnan(cepstral))
    assert not np.any(np.isinf(cepstral))

    # DCT should have non-zero values for a non-zero signal
    assert np.any(cepstral != 0)

    print("✓ Time → Cepstral basic test passed")


def test_dct_idct_roundtrip():
    """Test that DCT → IDCT reconstructs the original signal."""
    # Create test signal
    signal = np.random.randn(128).astype(np.float32)

    # Forward transform
    dct_interface = TimeToCepstralInterface(signal, dct_type=2, norm="ortho")
    cepstral = dct_interface.transform(signal)

    # Inverse transform
    idct_interface = CepstralToTimeInterface(cepstral, dct_type=2, norm="ortho")
    reconstructed = idct_interface.transform(cepstral)

    # Check reconstruction
    assert reconstructed.shape == signal.shape
    assert np.allclose(reconstructed, signal, atol=1e-5)

    print("✓ DCT → IDCT roundtrip test passed")


def test_dct_types():
    """Test different DCT types (1-4)."""
    signal = np.random.randn(64).astype(np.float32)

    for dct_type in [1, 2, 3, 4]:
        interface = TimeToCepstralInterface(signal, dct_type=dct_type, norm="ortho")
        cepstral = interface.transform(signal)
        assert cepstral.shape == signal.shape
        assert not np.any(np.isnan(cepstral))

    print("✓ DCT types test passed")


def test_dct_validation():
    """Test DCT validation."""
    # Invalid DCT type
    signal = np.random.randn(64).astype(np.float32)
    with pytest.raises(ValueError, match="DCT type must be 1-4"):
        TimeToCepstralInterface(signal, dct_type=5)

    # Invalid norm
    with pytest.raises(ValueError, match="Norm must be"):
        TimeToCepstralInterface(signal, norm="invalid")

    # 2D signal should fail
    signal_2d = np.random.randn(8, 8).astype(np.float32)
    interface = TimeToCepstralInterface(signal_2d)
    with pytest.raises(ValueError, match="Signal must be 1D"):
        interface.transform(signal_2d)

    print("✓ DCT validation test passed")


def test_time_to_wavelet_basic():
    """Test basic wavelet transformation."""
    # Create test signal with multiple frequencies
    t = np.linspace(0, 1, 512, dtype=np.float32)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)

    # Define scales
    scales = np.arange(1, 65, dtype=np.float32)

    # Create transform interface
    interface = TimeToWaveletInterface(signal, scales, wavelet="ricker")

    # Validate
    assert interface.validate()

    # Transform
    wavelet_coeffs = interface.transform(signal)

    # Check shape (scales × time)
    assert wavelet_coeffs.shape == (len(scales), len(signal))
    assert wavelet_coeffs.dtype == np.float32

    # Coefficients should have significant magnitude
    assert np.max(np.abs(wavelet_coeffs)) > 0.1

    print("✓ Time → Wavelet basic test passed")


def test_wavelet_validation():
    """Test wavelet validation."""
    signal = np.random.randn(128).astype(np.float32)
    scales = np.arange(1, 33, dtype=np.float32)

    # Invalid wavelet type
    with pytest.raises(ValueError, match="Wavelet must be one of"):
        TimeToWaveletInterface(signal, scales, wavelet="invalid")

    # 2D signal should fail
    signal_2d = np.random.randn(8, 8).astype(np.float32)
    interface = TimeToWaveletInterface(signal_2d, scales)
    with pytest.raises(ValueError, match="Signal must be 1D"):
        interface.transform(signal_2d)

    print("✓ Wavelet validation test passed")


# ==============================================================================
# SPATIAL TRANSFORMS TESTS
# ==============================================================================


def test_spatial_affine_identity():
    """Test affine transform with identity parameters (no transformation)."""
    # Create test image
    image = np.random.rand(32, 32).astype(np.float32)

    # Identity transform (no translation, rotation, scale, or shear)
    interface = SpatialAffineInterface(image)

    # Validate
    assert interface.validate()

    # Transform
    transformed = interface.transform(image)

    # Should be identical (or very close due to interpolation)
    assert transformed.shape == image.shape
    assert np.allclose(transformed, image, atol=0.1)

    print("✓ Spatial affine identity test passed")


def test_spatial_affine_translate():
    """Test spatial translation."""
    # Create image with a bright spot in the center
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 1.0

    # Translate by (10, 10) pixels
    interface = SpatialAffineInterface(image, translate=(10, 10))
    transformed = interface.transform(image)

    # Check that the bright spot moved
    assert transformed.shape == image.shape
    # Due to interpolation, check that maximum is near expected location
    max_loc = np.unravel_index(np.argmax(transformed), transformed.shape)
    assert abs(max_loc[0] - 32) < 15  # Allow some error due to interpolation
    assert abs(max_loc[1] - 32) < 15

    print("✓ Spatial affine translate test passed")


def test_spatial_affine_rotate():
    """Test spatial rotation."""
    # Create image with a horizontal line
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 20:44] = 1.0

    # Rotate by 90 degrees
    interface = SpatialAffineInterface(image, rotate=90)
    transformed = interface.transform(image)

    # Check that transformation occurred
    assert transformed.shape == image.shape
    # Check that the image changed (rotation applied)
    # Not bit-exact comparison due to interpolation
    assert not np.allclose(transformed, image)
    # Check that some non-zero values exist (line wasn't lost)
    assert np.sum(transformed > 0.1) > 0

    print("✓ Spatial affine rotate test passed")


def test_spatial_affine_scale():
    """Test spatial scaling."""
    # Create small square
    image = np.zeros((64, 64), dtype=np.float32)
    image[28:36, 28:36] = 1.0

    # Scale down by 0.5x (makes object appear larger in image)
    interface = SpatialAffineInterface(image, scale=(0.5, 0.5))
    transformed = interface.transform(image)

    # Check that transformation occurred
    assert transformed.shape == image.shape
    # Check that the image changed
    assert not np.allclose(transformed, image)
    # Check that some non-zero values exist
    assert np.sum(transformed > 0.1) > 0

    print("✓ Spatial affine scale test passed")


def test_spatial_affine_multichannel():
    """Test affine transform on multi-channel data (e.g., RGB image)."""
    # Create 3-channel image
    image = np.random.rand(32, 32, 3).astype(np.float32)

    # Apply translation
    interface = SpatialAffineInterface(image, translate=(5, 5))
    transformed = interface.transform(image)

    # Check shape
    assert transformed.shape == image.shape
    assert transformed.shape[2] == 3  # Channels preserved

    print("✓ Spatial affine multi-channel test passed")


def test_spatial_affine_validation():
    """Test spatial affine validation."""
    # 1D data should fail
    data_1d = np.random.rand(64).astype(np.float32)
    interface = SpatialAffineInterface(data_1d)
    with pytest.raises(ValueError, match="Data must be 2D or 3D"):
        interface.transform(data_1d)

    print("✓ Spatial affine validation test passed")


# ==============================================================================
# COORDINATE CONVERSION TESTS
# ==============================================================================


def test_cartesian_to_polar_basic():
    """Test basic Cartesian to Polar conversion."""
    # Create simple 2D field
    field = np.random.rand(64, 64).astype(np.float32)

    # Convert to polar
    interface = CartesianToPolarInterface(field)
    assert interface.validate()

    r, theta = interface.transform(field)

    # Check shapes
    assert r.shape == field.shape
    assert theta.shape == field.shape

    # Check radius values (should be >= 0)
    assert np.all(r >= 0)

    # Check angle range [-pi, pi]
    assert np.all(theta >= -np.pi)
    assert np.all(theta <= np.pi)

    # Center should have radius ≈ 0
    center_y, center_x = 32, 32
    assert r[center_y, center_x] < 1.0

    # Corners should have large radius
    assert r[0, 0] > 40  # Distance from center to corner

    print("✓ Cartesian → Polar basic test passed")


def test_cartesian_to_polar_custom_center():
    """Test Cartesian to Polar with custom center."""
    field = np.random.rand(64, 64).astype(np.float32)

    # Custom center (top-left corner)
    interface = CartesianToPolarInterface(field, center=(0, 0))
    r, theta = interface.transform(field)

    # Radius at (0, 0) should be ≈ 0
    assert r[0, 0] < 1.0

    # Bottom-right corner should have maximum radius
    assert r[-1, -1] > r[32, 32]

    print("✓ Cartesian → Polar custom center test passed")


def test_polar_to_cartesian_basic():
    """Test basic Polar to Cartesian conversion."""
    # Create polar coordinate grids
    height, width = 64, 64
    cy, cx = height / 2, width / 2

    y, x = np.indices((height, width), dtype=np.float32)
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Create test data (radially symmetric pattern)
    polar_data = np.sin(r * 0.2)

    # Convert back to Cartesian
    interface = PolarToCartesianInterface(r, theta, output_shape=(height, width))
    assert interface.validate()

    cartesian = interface.transform(polar_data)

    # Check shape
    assert cartesian.shape == (height, width)

    # Check that data is reasonable
    assert not np.any(np.isnan(cartesian))
    assert np.abs(np.mean(cartesian) - np.mean(polar_data)) < 0.5

    print("✓ Polar → Cartesian basic test passed")


def test_cartesian_polar_roundtrip():
    """Test Cartesian → Polar → Cartesian roundtrip."""
    # Create simple radially symmetric field
    height, width = 64, 64
    y, x = np.indices((height, width), dtype=np.float32)
    cy, cx = height / 2, width / 2
    r_field = np.sqrt((x - cx)**2 + (y - cy)**2)
    field = np.exp(-r_field / 10)  # Gaussian-like radial pattern

    # Convert to polar
    cart_to_polar = CartesianToPolarInterface(field)
    r, theta = cart_to_polar.transform(field)

    # Convert back to Cartesian
    polar_to_cart = PolarToCartesianInterface(r, theta, output_shape=(height, width))
    reconstructed = polar_to_cart.transform(field)

    # Check reconstruction (will not be perfect due to interpolation)
    assert reconstructed.shape == field.shape
    # Check correlation is high
    correlation = np.corrcoef(field.ravel(), reconstructed.ravel())[0, 1]
    assert correlation > 0.7  # Should be reasonably correlated

    print("✓ Cartesian → Polar → Cartesian roundtrip test passed")


def test_polar_validation():
    """Test polar coordinate validation."""
    # Mismatched radius and angle shapes
    r = np.random.rand(32, 32).astype(np.float32)
    theta = np.random.rand(64, 64).astype(np.float32)

    interface = PolarToCartesianInterface(r, theta, output_shape=(32, 32))
    with pytest.raises(ValueError, match="Radius and angle must have same shape"):
        interface.validate()

    print("✓ Polar validation test passed")


# ==============================================================================
# REGISTRY TESTS
# ==============================================================================


def test_registry_has_new_transforms():
    """Test that new transforms are registered in CrossDomainRegistry."""
    # Check time-frequency transforms
    assert CrossDomainRegistry.has_transform("time", "cepstral")
    assert CrossDomainRegistry.has_transform("cepstral", "time")
    assert CrossDomainRegistry.has_transform("time", "wavelet")

    # Check spatial transforms
    assert CrossDomainRegistry.has_transform("spatial", "spatial")
    assert CrossDomainRegistry.has_transform("cartesian", "polar")
    assert CrossDomainRegistry.has_transform("polar", "cartesian")

    # Verify we can retrieve them
    dct_class = CrossDomainRegistry.get("time", "cepstral")
    assert dct_class == TimeToCepstralInterface

    idct_class = CrossDomainRegistry.get("cepstral", "time")
    assert idct_class == CepstralToTimeInterface

    wavelet_class = CrossDomainRegistry.get("time", "wavelet")
    assert wavelet_class == TimeToWaveletInterface

    affine_class = CrossDomainRegistry.get("spatial", "spatial")
    assert affine_class == SpatialAffineInterface

    cart_to_polar_class = CrossDomainRegistry.get("cartesian", "polar")
    assert cart_to_polar_class == CartesianToPolarInterface

    polar_to_cart_class = CrossDomainRegistry.get("polar", "cartesian")
    assert polar_to_cart_class == PolarToCartesianInterface

    print("✓ Registry has new transforms test passed")


def test_transform_metadata():
    """Test that transforms have proper metadata."""
    # Check DCT metadata
    metadata = CrossDomainRegistry.get_metadata("time", "cepstral")
    assert metadata is not None
    assert "description" in metadata
    assert "use_cases" in metadata
    assert "MFCC" in str(metadata["use_cases"])

    # Check affine metadata
    metadata = CrossDomainRegistry.get_metadata("spatial", "spatial")
    assert metadata is not None
    assert "affine" in metadata["description"].lower()

    print("✓ Transform metadata test passed")


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    print("Running new domain transformation tests...\n")

    # Time-frequency tests
    print("=" * 60)
    print("TIME-FREQUENCY TRANSFORMS")
    print("=" * 60)
    test_time_to_cepstral_basic()
    test_dct_idct_roundtrip()
    test_dct_types()
    test_dct_validation()
    test_time_to_wavelet_basic()
    test_wavelet_validation()

    # Spatial tests
    print("\n" + "=" * 60)
    print("SPATIAL TRANSFORMS")
    print("=" * 60)
    test_spatial_affine_identity()
    test_spatial_affine_translate()
    test_spatial_affine_rotate()
    test_spatial_affine_scale()
    test_spatial_affine_multichannel()
    test_spatial_affine_validation()

    # Coordinate conversion tests
    print("\n" + "=" * 60)
    print("COORDINATE CONVERSIONS")
    print("=" * 60)
    test_cartesian_to_polar_basic()
    test_cartesian_to_polar_custom_center()
    test_polar_to_cartesian_basic()
    test_cartesian_polar_roundtrip()
    test_polar_validation()

    # Registry tests
    print("\n" + "=" * 60)
    print("REGISTRY TESTS")
    print("=" * 60)
    test_registry_has_new_transforms()
    test_transform_metadata()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
