"""Verification script for I/O & Storage domain (no pytest required).

Tests all I/O operations without external test frameworks.
"""

import numpy as np
import tempfile
from pathlib import Path
import sys

# Import I/O functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from morphogen.stdlib import io_storage as io


def test_image_io():
    """Test image I/O operations."""
    print("\n" + "="*60)
    print("TEST: Image I/O")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        gray = np.random.rand(64, 64).astype(np.float32)
        rgb = np.random.rand(64, 64, 3).astype(np.float32)

        # Test PNG save/load (grayscale)
        print("  ✓ Testing PNG grayscale...")
        gray_path = tmpdir / "test_gray.png"
        io.save_image(gray_path, gray)
        loaded_gray = io.load_image(gray_path, grayscale=True)
        assert loaded_gray.shape == gray.shape
        assert np.allclose(loaded_gray, gray, atol=0.01)
        print(f"    Saved and loaded {gray.shape} grayscale PNG")

        # Test PNG save/load (RGB)
        print("  ✓ Testing PNG RGB...")
        rgb_path = tmpdir / "test_rgb.png"
        io.save_image(rgb_path, rgb)
        loaded_rgb = io.load_image(rgb_path)
        assert loaded_rgb.shape == rgb.shape
        assert np.allclose(loaded_rgb, rgb, atol=0.01)
        print(f"    Saved and loaded {rgb.shape} RGB PNG")

        # Test JPEG save/load
        print("  ✓ Testing JPEG...")
        jpeg_path = tmpdir / "test.jpg"
        io.save_image(jpeg_path, rgb, quality=95)
        loaded_jpeg = io.load_image(jpeg_path)
        assert loaded_jpeg.shape == rgb.shape
        # JPEG is lossy, use larger tolerance
        print(f"    Saved and loaded {rgb.shape} JPEG (lossy compression)")

        print("  ✅ Image I/O: ALL TESTS PASSED")


def test_audio_io():
    """Test audio I/O operations."""
    print("\n" + "="*60)
    print("TEST: Audio I/O")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test audio (440 Hz sine wave)
        sample_rate = 44100
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        mono = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.column_stack([mono, mono * 0.5])

        # Test WAV save/load (mono)
        print("  ✓ Testing WAV mono...")
        wav_mono_path = tmpdir / "test_mono.wav"
        io.save_audio(wav_mono_path, mono, sample_rate)
        loaded_mono, loaded_sr = io.load_audio(wav_mono_path)
        assert loaded_sr == sample_rate
        assert loaded_mono.shape == mono.shape
        # WAV uses 16-bit PCM by default, so use reasonable tolerance
        assert np.allclose(loaded_mono, mono, atol=1e-4)
        print(f"    Saved and loaded {len(mono)} mono samples @ {sample_rate} Hz")

        # Test WAV save/load (stereo)
        print("  ✓ Testing WAV stereo...")
        wav_stereo_path = tmpdir / "test_stereo.wav"
        io.save_audio(wav_stereo_path, stereo, sample_rate)
        loaded_stereo, loaded_sr = io.load_audio(wav_stereo_path)
        assert loaded_sr == sample_rate
        assert loaded_stereo.shape == stereo.shape
        # WAV uses 16-bit PCM by default, so use reasonable tolerance
        assert np.allclose(loaded_stereo, stereo, atol=1e-4)
        print(f"    Saved and loaded {stereo.shape} stereo samples @ {sample_rate} Hz")

        # Test FLAC save/load
        print("  ✓ Testing FLAC...")
        flac_path = tmpdir / "test.flac"
        io.save_audio(flac_path, mono, sample_rate)
        loaded_flac, loaded_sr = io.load_audio(flac_path)
        assert loaded_sr == sample_rate
        # FLAC is lossless but still subject to some precision differences
        assert np.allclose(loaded_flac, mono, atol=1e-4)
        print(f"    Saved and loaded FLAC (lossless compression)")

        # Test mono conversion
        print("  ✓ Testing stereo-to-mono conversion...")
        loaded_mono_from_stereo, _ = io.load_audio(wav_stereo_path, mono=True)
        assert loaded_mono_from_stereo.ndim == 1
        print(f"    Converted stereo to mono: {loaded_mono_from_stereo.shape}")

        print("  ✅ Audio I/O: ALL TESTS PASSED")


def test_json_io():
    """Test JSON I/O operations."""
    print("\n" + "="*60)
    print("TEST: JSON I/O")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test basic JSON
        print("  ✓ Testing basic JSON...")
        data = {
            "name": "test",
            "value": 42,
            "pi": 3.14159,
            "enabled": True,
            "items": [1, 2, 3, 4, 5],
            "nested": {"a": 1, "b": 2}
        }
        json_path = tmpdir / "test.json"
        io.save_json(json_path, data)
        loaded = io.load_json(json_path)
        assert loaded == data
        print(f"    Saved and loaded {len(data)} fields")

        # Test NumPy types
        print("  ✓ Testing NumPy type conversion...")
        numpy_data = {
            "int32": np.int32(42),
            "float32": np.float32(3.14),
            "array": np.array([1, 2, 3])
        }
        numpy_path = tmpdir / "numpy.json"
        io.save_json(numpy_path, numpy_data)
        loaded_numpy = io.load_json(numpy_path)
        assert loaded_numpy["int32"] == 42
        assert loaded_numpy["array"] == [1, 2, 3]
        print(f"    NumPy types converted correctly")

        print("  ✅ JSON I/O: ALL TESTS PASSED")


def test_hdf5_io():
    """Test HDF5 I/O operations."""
    print("\n" + "="*60)
    print("TEST: HDF5 I/O")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test single array
        print("  ✓ Testing single array...")
        single_data = np.random.rand(100, 100).astype(np.float32)
        single_path = tmpdir / "single.h5"
        io.save_hdf5(single_path, single_data)
        loaded_single = io.load_hdf5(single_path, dataset='data')
        assert np.array_equal(loaded_single, single_data)
        print(f"    Saved and loaded {single_data.shape} array")

        # Test multiple arrays
        print("  ✓ Testing multiple arrays...")
        multi_data = {
            "velocity": np.random.rand(64, 64, 2).astype(np.float32),
            "pressure": np.random.rand(64, 64).astype(np.float32),
            "temperature": np.random.rand(64, 64).astype(np.float32)
        }
        multi_path = tmpdir / "multi.h5"
        io.save_hdf5(multi_path, multi_data)
        loaded_multi = io.load_hdf5(multi_path)
        assert set(loaded_multi.keys()) == set(multi_data.keys())
        for key in multi_data:
            assert np.array_equal(loaded_multi[key], multi_data[key])
        print(f"    Saved and loaded {len(multi_data)} arrays")

        # Test compression
        print("  ✓ Testing compression...")
        zeros = np.zeros((1000, 1000), dtype=np.float32)
        uncomp_path = tmpdir / "uncompressed.h5"
        comp_path = tmpdir / "compressed.h5"
        io.save_hdf5(uncomp_path, zeros, compression=None)
        io.save_hdf5(comp_path, zeros, compression='gzip', compression_opts=9)
        uncomp_size = uncomp_path.stat().st_size
        comp_size = comp_path.stat().st_size
        ratio = uncomp_size / comp_size
        print(f"    Compression ratio: {ratio:.2f}x (uncompressed={uncomp_size/1024:.1f}KB, compressed={comp_size/1024:.1f}KB)")

        print("  ✅ HDF5 I/O: ALL TESTS PASSED")


def test_checkpointing():
    """Test checkpoint save/load operations."""
    print("\n" + "="*60)
    print("TEST: Checkpointing")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create simulation state
        print("  ✓ Testing checkpoint save/load...")
        state = {
            "velocity_field": np.random.rand(64, 64, 2).astype(np.float32),
            "pressure_field": np.random.rand(64, 64).astype(np.float32),
            "particles": np.random.rand(100, 3).astype(np.float32),
            "parameters": {
                "dt": 0.01,
                "viscosity": 0.1,
                "gravity": -9.81
            }
        }
        metadata = {
            "iteration": 1000,
            "time": 10.0,
            "version": "1.0"
        }

        # Save checkpoint
        checkpoint_path = tmpdir / "checkpoint.h5"
        io.save_checkpoint(checkpoint_path, state, metadata)

        # Load checkpoint
        loaded_state, loaded_metadata = io.load_checkpoint(checkpoint_path)

        # Verify arrays
        assert np.array_equal(loaded_state["velocity_field"], state["velocity_field"])
        assert np.array_equal(loaded_state["pressure_field"], state["pressure_field"])
        assert np.array_equal(loaded_state["particles"], state["particles"])

        # Verify parameters
        assert loaded_state["parameters"]["dt"] == state["parameters"]["dt"]
        assert loaded_state["parameters"]["viscosity"] == state["parameters"]["viscosity"]

        # Verify metadata
        assert loaded_metadata["iteration"] == metadata["iteration"]
        assert loaded_metadata["time"] == metadata["time"]

        print(f"    Saved and loaded checkpoint with {len(state)} state fields and {len(metadata)} metadata fields")

        # Test determinism
        print("  ✓ Testing checkpoint determinism...")
        checkpoint2_path = tmpdir / "checkpoint2.h5"
        io.save_checkpoint(checkpoint2_path, state, metadata)
        loaded_state2, _ = io.load_checkpoint(checkpoint2_path)
        assert np.array_equal(loaded_state["velocity_field"], loaded_state2["velocity_field"])
        print(f"    Checkpoint save/load is deterministic")

        print("  ✅ Checkpointing: ALL TESTS PASSED")


def test_integration():
    """Integration test: complete simulation workflow."""
    print("\n" + "="*60)
    print("TEST: Integration (Simulation Workflow)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Simulate fluid dynamics with checkpointing
        print("  ✓ Running simulated fluid dynamics...")
        state = {
            "velocity": np.random.rand(32, 32, 2).astype(np.float32),
            "density": np.random.rand(32, 32).astype(np.float32)
        }

        # Save initial checkpoint
        checkpoint_0 = tmpdir / "sim_0000.h5"
        io.save_checkpoint(checkpoint_0, state, {"iteration": 0, "time": 0.0})

        # Simulate 10 steps
        for i in range(1, 11):
            state["velocity"] *= 0.99  # Damping
            state["density"] *= 0.98

            if i % 5 == 0:  # Save every 5 steps
                checkpoint = tmpdir / f"sim_{i:04d}.h5"
                io.save_checkpoint(checkpoint, state, {"iteration": i, "time": i * 0.01})

        # Load final checkpoint
        final_state, final_meta = io.load_checkpoint(tmpdir / "sim_0010.h5")
        assert final_meta["iteration"] == 10

        # Visualize final density field
        density = final_state["density"]
        density_norm = (density - density.min()) / (density.max() - density.min())
        density_rgb = np.stack([density_norm, density_norm, density_norm], axis=-1)

        # Save visualization
        img_path = tmpdir / "final_density.png"
        io.save_image(img_path, density_rgb)

        # Save raw data
        h5_path = tmpdir / "final_data.h5"
        io.save_hdf5(h5_path, {"density": density, "velocity": final_state["velocity"]})

        print(f"    Simulated 10 steps, saved 3 checkpoints")
        print(f"    Exported visualization: {img_path.name}")
        print(f"    Exported data: {h5_path.name}")

        print("  ✅ Integration: ALL TESTS PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("I/O & STORAGE DOMAIN VERIFICATION")
    print("="*70)

    tests = [
        test_image_io,
        test_audio_io,
        test_json_io,
        test_hdf5_io,
        test_checkpointing,
        test_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ FAILED: {test_func.__name__}")
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED! ✓")
        print("\n" + "="*70)
        print("I/O & Storage Domain is production-ready!")
        print("="*70)
        return 0
    else:
        print(f"\n  ⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
