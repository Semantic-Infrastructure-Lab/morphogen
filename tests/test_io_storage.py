"""Tests for morphogen.stdlib.io_storage.

Covers image, audio, JSON, and HDF5 I/O.
"""

import pytest
import numpy as np
import json
import os

from morphogen.stdlib.io_storage import (
    load_image, save_image,
    load_json, save_json,
)


# ============================================================================
# Image I/O
# ============================================================================

class TestSaveImage:
    def test_save_rgb_float(self, tmp_path):
        data = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        path = str(tmp_path / "rgb.png")
        save_image(path, data)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_save_rgba_float(self, tmp_path):
        data = np.random.default_rng(0).random((16, 16, 4)).astype(np.float32)
        path = str(tmp_path / "rgba.png")
        save_image(path, data)
        assert os.path.exists(path)

    def test_save_grayscale_2d(self, tmp_path):
        data = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        path = str(tmp_path / "gray.png")
        save_image(path, data)
        assert os.path.exists(path)

    def test_save_grayscale_3d_single_channel(self, tmp_path):
        data = np.random.default_rng(0).random((16, 16, 1)).astype(np.float32)
        path = str(tmp_path / "gray1ch.png")
        save_image(path, data)
        assert os.path.exists(path)

    def test_save_jpeg(self, tmp_path):
        data = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        path = str(tmp_path / "test.jpg")
        save_image(path, data, quality=85)
        assert os.path.exists(path)

    def test_save_no_denormalize(self, tmp_path):
        data = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
        path = str(tmp_path / "uint8.png")
        save_image(path, data, denormalize=False)
        assert os.path.exists(path)

    def test_save_clips_values(self, tmp_path):
        # Values outside [0, 1] should be clipped without error
        data = np.ones((8, 8, 3), dtype=np.float32) * 1.5
        path = str(tmp_path / "clipped.png")
        save_image(path, data)
        assert os.path.exists(path)

    def test_save_invalid_channels_raises(self, tmp_path):
        data = np.ones((8, 8, 5), dtype=np.float32)
        path = str(tmp_path / "bad.png")
        with pytest.raises(ValueError, match="channels"):
            save_image(path, data)


class TestLoadImage:
    def _make_png(self, tmp_path, shape=(32, 32, 3), name="test.png"):
        data = np.random.default_rng(42).random(shape).astype(np.float32)
        path = str(tmp_path / name)
        save_image(path, data)
        return path, data

    def test_load_rgb_as_float(self, tmp_path):
        path, original = self._make_png(tmp_path)
        loaded = load_image(path)
        assert loaded.dtype == np.float32
        assert loaded.shape == (32, 32, 3)
        assert np.all(loaded >= 0.0)
        assert np.all(loaded <= 1.0)

    def test_load_roundtrip(self, tmp_path):
        data = (np.ones((16, 16, 3)) * 0.5).astype(np.float32)
        path = str(tmp_path / "solid.png")
        save_image(path, data)
        loaded = load_image(path)
        # Allow uint8 quantization tolerance
        assert np.allclose(data, loaded, atol=0.01)

    def test_load_as_grayscale(self, tmp_path):
        path, _ = self._make_png(tmp_path)
        loaded = load_image(path, grayscale=True)
        assert loaded.ndim == 2
        assert loaded.dtype == np.float32

    def test_load_not_normalized(self, tmp_path):
        path, _ = self._make_png(tmp_path)
        loaded = load_image(path, normalize=False)
        # Without normalization, uint8 values in [0, 255]
        assert loaded.max() > 1.0

    def test_load_as_uint8(self, tmp_path):
        path, _ = self._make_png(tmp_path)
        loaded = load_image(path, as_float=False)
        assert loaded.dtype == np.uint8

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/tmp/definitely_does_not_exist_xyz123.png")


# ============================================================================
# JSON I/O
# ============================================================================

class TestJsonIO:
    def test_save_load_roundtrip(self, tmp_path):
        data = {"key": "value", "count": 42, "values": [1.0, 2.0, 3.0]}
        path = str(tmp_path / "data.json")
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "out.json")
        save_json(path, {"x": 1})
        assert os.path.exists(path)

    def test_save_is_valid_json(self, tmp_path):
        path = str(tmp_path / "out.json")
        data = {"nested": {"a": 1, "b": [1, 2, 3]}}
        save_json(path, data)
        with open(path) as f:
            parsed = json.load(f)
        assert parsed == data

    def test_save_with_indent(self, tmp_path):
        path = str(tmp_path / "pretty.json")
        save_json(path, {"a": 1}, indent=4)
        with open(path) as f:
            content = f.read()
        # Indented JSON has newlines
        assert "\n" in content

    def test_save_numpy_scalars(self, tmp_path):
        # NumpyEncoder should handle np types
        data = {"val": np.float32(0.5), "arr": np.array([1, 2, 3])}
        path = str(tmp_path / "numpy.json")
        save_json(path, data)
        loaded = load_json(path)
        assert abs(loaded["val"] - 0.5) < 0.001
        assert loaded["arr"] == [1, 2, 3]

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json("/tmp/definitely_does_not_exist_xyz123.json")

    def test_save_sort_keys(self, tmp_path):
        path = str(tmp_path / "sorted.json")
        save_json(path, {"z": 1, "a": 2, "m": 3}, sort_keys=True)
        with open(path) as f:
            content = f.read()
        # With sort_keys, "a" comes before "z"
        assert content.index('"a"') < content.index('"z"')


# ============================================================================
# Audio I/O (conditional on soundfile)
# ============================================================================

class TestAudioIO:
    @pytest.fixture(autouse=True)
    def check_soundfile(self):
        try:
            import soundfile
        except ImportError:
            pytest.skip("soundfile not installed")

    def test_save_load_wav_roundtrip(self, tmp_path):
        from morphogen.stdlib.io_storage import save_audio, load_audio
        data = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32)
        data = data[:, np.newaxis]  # (samples, channels)
        path = str(tmp_path / "sine.wav")
        save_audio(path, data, sample_rate=22050)
        loaded, sr = load_audio(path)
        assert sr == 22050
        assert loaded.shape[0] == data.shape[0]

    def test_save_stereo(self, tmp_path):
        from morphogen.stdlib.io_storage import save_audio, load_audio
        data = np.random.default_rng(0).random((4410, 2)).astype(np.float32)
        path = str(tmp_path / "stereo.wav")
        save_audio(path, data, sample_rate=44100)
        loaded, sr = load_audio(path)
        assert sr == 44100
        assert loaded.shape[1] == 2


# ============================================================================
# HDF5 I/O (conditional on h5py)
# ============================================================================

class TestHDF5IO:
    @pytest.fixture(autouse=True)
    def check_h5py(self):
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")

    def test_save_load_array_roundtrip(self, tmp_path):
        from morphogen.stdlib.io_storage import save_hdf5, load_hdf5
        data = np.random.default_rng(0).random((32, 32)).astype(np.float64)
        path = str(tmp_path / "data.h5")
        save_hdf5(path, data)
        loaded = load_hdf5(path)
        # Bare array is stored under key "data"
        if isinstance(loaded, dict):
            loaded = loaded["data"]
        assert np.allclose(data, loaded)

    def test_save_load_dict_roundtrip(self, tmp_path):
        from morphogen.stdlib.io_storage import save_hdf5, load_hdf5
        data = {
            "field": np.random.default_rng(0).random((16, 16)),
            "weights": np.linspace(0, 1, 10),
        }
        path = str(tmp_path / "dict.h5")
        save_hdf5(path, data)
        loaded = load_hdf5(path)
        assert isinstance(loaded, dict)
        assert np.allclose(data["field"], loaded["field"])
        assert np.allclose(data["weights"], loaded["weights"])

    def test_save_creates_file(self, tmp_path):
        from morphogen.stdlib.io_storage import save_hdf5
        path = str(tmp_path / "check.h5")
        save_hdf5(path, np.ones((4, 4)))
        assert os.path.exists(path)


# ============================================================================
# Checkpoint I/O (conditional on h5py)
# ============================================================================

class TestCheckpointIO:
    @pytest.fixture(autouse=True)
    def check_h5py(self):
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")

    def test_save_load_checkpoint_roundtrip(self, tmp_path):
        from morphogen.stdlib.io_storage import save_checkpoint, load_checkpoint
        state = {
            "positions": np.random.default_rng(0).random((10, 3)),
            "velocities": np.random.default_rng(1).random((10, 3)),
        }
        metadata = {"step": 100, "dt": 0.01}
        path = str(tmp_path / "checkpoint.h5")
        save_checkpoint(path, state, metadata=metadata)
        loaded_state, loaded_meta = load_checkpoint(path)
        assert np.allclose(state["positions"], loaded_state["positions"])
        assert np.allclose(state["velocities"], loaded_state["velocities"])
        assert loaded_meta["step"] == 100

    def test_checkpoint_without_metadata(self, tmp_path):
        from morphogen.stdlib.io_storage import save_checkpoint, load_checkpoint
        state = {"data": np.ones((4, 4))}
        path = str(tmp_path / "no_meta.h5")
        save_checkpoint(path, state)
        loaded_state, loaded_meta = load_checkpoint(path)
        assert np.allclose(state["data"], loaded_state["data"])
