"""Example: Image I/O operations.

Demonstrates loading, saving, and manipulating images using the I/O domain.
"""

import numpy as np
import sys
from pathlib import Path

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from morphogen.stdlib import io_storage as io


def example_1_generate_and_save_gradient():
    """Generate a gradient image and save it."""
    print("=" * 60)
    print("Example 1: Generate and save gradient image")
    print("=" * 60)

    # Create gradient (256x256 RGB)
    width, height = 256, 256
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Red gradient (left to right)
    img[:, :, 0] = np.linspace(0, 1, width)[None, :]

    # Green gradient (top to bottom)
    img[:, :, 1] = np.linspace(0, 1, height)[:, None]

    # Blue constant
    img[:, :, 2] = 0.5

    # Save as PNG
    io.save_image("gradient.png", img)
    print("  ✓ Saved gradient.png")

    # Save as JPEG (lossy compression)
    io.save_image("gradient.jpg", img, quality=95)
    print("  ✓ Saved gradient.jpg (95% quality)")

    print()


def example_2_load_and_convert_grayscale():
    """Load image and convert to grayscale."""
    print("=" * 60)
    print("Example 2: Load and convert to grayscale")
    print("=" * 60)

    # Load RGB image
    img_rgb = io.load_image("gradient.png")
    print(f"  Loaded RGB image: shape={img_rgb.shape}, dtype={img_rgb.dtype}")

    # Load as grayscale
    img_gray = io.load_image("gradient.png", grayscale=True)
    print(f"  Loaded grayscale: shape={img_gray.shape}, dtype={img_gray.dtype}")

    # Convert RGB to grayscale manually (luminance formula)
    img_gray_manual = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]

    # Save grayscale
    io.save_image("gradient_gray.png", img_gray)
    print("  ✓ Saved gradient_gray.png")

    print()


def example_3_procedural_texture():
    """Generate procedural texture and save."""
    print("=" * 60)
    print("Example 3: Procedural texture generation")
    print("=" * 60)

    # Create checkerboard pattern
    size = 256
    checker = np.zeros((size, size, 3), dtype=np.float32)

    # 8x8 checkerboard
    square_size = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                checker[i*square_size:(i+1)*square_size,
                       j*square_size:(j+1)*square_size] = 1.0

    io.save_image("checkerboard.png", checker)
    print("  ✓ Saved checkerboard.png (8x8 grid)")

    # Create circular pattern
    circular = np.zeros((size, size, 3), dtype=np.float32)
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    dist = np.sqrt(x**2 + y**2)

    # Rainbow circles
    circular[:, :, 0] = np.sin(dist * 10) * 0.5 + 0.5  # Red
    circular[:, :, 1] = np.sin(dist * 10 + 2*np.pi/3) * 0.5 + 0.5  # Green
    circular[:, :, 2] = np.sin(dist * 10 + 4*np.pi/3) * 0.5 + 0.5  # Blue

    io.save_image("circular_pattern.png", circular)
    print("  ✓ Saved circular_pattern.png (rainbow circles)")

    print()


def example_4_data_analysis_workflow():
    """Combine image I/O with data analysis."""
    print("=" * 60)
    print("Example 4: Data analysis workflow")
    print("=" * 60)

    # Generate "simulation data" (2D field)
    size = 128
    field = np.random.rand(size, size).astype(np.float32)

    # Smooth the field (simple box blur)
    kernel_size = 5
    for _ in range(3):
        smoothed = np.zeros_like(field)
        for i in range(size):
            for j in range(size):
                i_min = max(0, i - kernel_size // 2)
                i_max = min(size, i + kernel_size // 2 + 1)
                j_min = max(0, j - kernel_size // 2)
                j_max = min(size, j + kernel_size // 2 + 1)
                smoothed[i, j] = field[i_min:i_max, j_min:j_max].mean()
        field = smoothed

    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min())

    # Convert to RGB (heatmap - red=high, blue=low)
    heatmap = np.zeros((size, size, 3), dtype=np.float32)
    heatmap[:, :, 0] = field  # Red channel
    heatmap[:, :, 2] = 1.0 - field  # Blue channel (inverted)

    io.save_image("heatmap.png", heatmap)
    print("  ✓ Saved heatmap.png (red=high, blue=low)")

    print(f"  Field statistics:")
    print(f"    Min: {field.min():.4f}")
    print(f"    Max: {field.max():.4f}")
    print(f"    Mean: {field.mean():.4f}")
    print(f"    Std: {field.std():.4f}")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("IMAGE I/O EXAMPLES")
    print("=" * 60)
    print()

    example_1_generate_and_save_gradient()
    example_2_load_and_convert_grayscale()
    example_3_procedural_texture()
    example_4_data_analysis_workflow()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - gradient.png")
    print("  - gradient.jpg")
    print("  - gradient_gray.png")
    print("  - checkerboard.png")
    print("  - circular_pattern.png")
    print("  - heatmap.png")
    print()


if __name__ == "__main__":
    main()
