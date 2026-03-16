"""I/O & Storage domain implementation.

This module provides file I/O operations for loading and saving various data formats
including images, audio, JSON, HDF5, and simulation checkpoints. All operations are
designed for deterministic behavior and error handling.

Supported Formats:
- Images: PNG, JPEG, BMP (via Pillow)
- Audio: WAV, FLAC, MP3 (via soundfile)
- Data: JSON, HDF5
- Checkpoints: Full simulation state with metadata
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import json
from pathlib import Path

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# IMAGE I/O
# ============================================================================

def _detect_image_mode(data: "np.ndarray"):
    """Return (data, PIL mode) for a 2D or 3D numpy array."""
    if data.ndim == 2:
        return data, 'L'
    if data.ndim == 3:
        ch = data.shape[2]
        if ch == 1:
            return data.squeeze(axis=2), 'L'
        if ch == 3:
            return data, 'RGB'
        if ch == 4:
            return data, 'RGBA'
        raise ValueError(f"Unsupported number of channels: {ch}")
    raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

@operator(
    domain="io_storage",
    category=OpCategory.CONSTRUCT,
    signature="(path: Union[str, Path], as_float: bool, normalize: bool, grayscale: bool) -> ndarray",
    deterministic=True,
    doc="Load image from file as NumPy array"
)
def load_image(
    path: Union[str, Path],
    as_float: bool = True,
    normalize: bool = True,
    grayscale: bool = False
) -> np.ndarray:
    """Load image from file as NumPy array.

    Supports PNG, JPEG, BMP, and other PIL-supported formats.

    Args:
        path: Path to image file
        as_float: Convert to float (default True)
        normalize: Normalize to [0, 1] range (default True, requires as_float=True)
        grayscale: Convert to grayscale (default False)

    Returns:
        NumPy array of shape:
        - (H, W) if grayscale=True
        - (H, W, 3) for RGB images
        - (H, W, 4) for RGBA images

    Example:
        img = io.load_image("texture.png")  # Shape: (512, 512, 3), range [0, 1]
        img_gray = io.load_image("photo.jpg", grayscale=True)  # Shape: (1024, 768)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image I/O. "
            "Install with: pip install pillow"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Load image
    img = Image.open(path)

    # Convert to grayscale if requested
    if grayscale:
        img = img.convert('L')
    else:
        # Convert to RGB or RGBA
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')

    # Convert to NumPy array
    arr = np.array(img)

    # Convert to float if requested
    if as_float:
        arr = arr.astype(np.float32)
        if normalize:
            arr = arr / 255.0

    return arr


@operator(
    domain="io_storage",
    category=OpCategory.TRANSFORM,
    signature="(path: Union[str, Path], data: ndarray, denormalize: bool, quality: int, format: Optional[str]) -> None",
    deterministic=True,
    doc="Save NumPy array as image file"
)
def save_image(
    path: Union[str, Path],
    data: np.ndarray,
    denormalize: bool = True,
    quality: int = 95,
    format: Optional[str] = None
) -> None:
    """Save NumPy array as image file.

    Args:
        path: Output path (extension determines format)
        data: NumPy array of shape (H, W) or (H, W, C) where C=1,3,4
        denormalize: If True, assumes data is in [0, 1] and scales to [0, 255]
        quality: JPEG quality (1-100, default 95)
        format: Force specific format (e.g., 'PNG', 'JPEG'). Auto-detected from extension if None.

    Example:
        field = np.random.rand(512, 512, 3)  # RGB in [0, 1]
        io.save_image("output.png", field)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image I/O. "
            "Install with: pip install pillow"
        )

    path = Path(path)
    data, mode = _detect_image_mode(data)

    # Denormalize if needed
    if denormalize and data.dtype in [np.float32, np.float64]:
        data = np.clip(data * 255.0, 0, 255).astype(np.uint8)
    elif data.dtype not in [np.uint8]:
        data = np.clip(data, 0, 255).astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(data, mode=mode)

    # Save with appropriate options
    save_kwargs = {}
    if format:
        save_kwargs['format'] = format
    if path.suffix.lower() in ['.jpg', '.jpeg'] or format == 'JPEG':
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True

    img.save(path, **save_kwargs)


# ============================================================================
# AUDIO I/O
# ============================================================================

@operator(
    domain="io_storage",
    category=OpCategory.CONSTRUCT,
    signature="(path: Union[str, Path], sample_rate: Optional[int], mono: bool) -> Tuple[ndarray, int]",
    deterministic=True,
    doc="Load audio file as NumPy array"
)
def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = False
) -> Tuple[np.ndarray, int]:
    """Load audio file as NumPy array.

    Supports WAV, FLAC, OGG, and other formats via soundfile.

    Args:
        path: Path to audio file
        sample_rate: Resample to this rate (None = keep original)
        mono: Convert to mono (default False)

    Returns:
        Tuple of (audio_data, sample_rate) where:
        - audio_data: Shape (n_samples,) for mono or (n_samples, n_channels) for stereo
        - sample_rate: Sampling rate in Hz

    Example:
        audio, sr = io.load_audio("music.wav")
        audio_mono, sr = io.load_audio("stereo.flac", mono=True)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio I/O. "
            "Install with: pip install soundfile"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load audio
    data, sr = sf.read(path, dtype='float32')

    # Resample if requested
    if sample_rate is not None and sample_rate != sr:
        # Simple resampling (use scipy.signal.resample for better quality)
        try:
            from scipy.signal import resample
            n_samples = int(len(data) * sample_rate / sr)
            data = resample(data, n_samples)
            sr = sample_rate
        except ImportError:
            raise ImportError(
                "scipy is required for audio resampling. "
                "Install with: pip install scipy"
            )

    # Convert to mono if requested
    if mono and data.ndim == 2:
        data = np.mean(data, axis=1)

    return data, sr


@operator(
    domain="io_storage",
    category=OpCategory.TRANSFORM,
    signature="(path: Union[str, Path], data: ndarray, sample_rate: int, format: Optional[str], subtype: Optional[str]) -> None",
    deterministic=True,
    doc="Save NumPy array as audio file"
)
def save_audio(
    path: Union[str, Path],
    data: np.ndarray,
    sample_rate: int,
    format: Optional[str] = None,
    subtype: Optional[str] = None
) -> None:
    """Save NumPy array as audio file.

    Args:
        path: Output path (extension determines format)
        data: Audio data, shape (n_samples,) or (n_samples, n_channels)
        sample_rate: Sampling rate in Hz
        format: Audio format ('WAV', 'FLAC', 'OGG', etc.). Auto-detected from extension if None.
        subtype: Audio subtype ('PCM_16', 'PCM_24', 'FLOAT', etc.)

    Example:
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # 440 Hz sine
        io.save_audio("tone.wav", audio, 44100)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio I/O. "
            "Install with: pip install soundfile"
        )

    path = Path(path)

    # Validate data
    if data.ndim > 2:
        raise ValueError(f"Audio data must be 1D or 2D, got shape {data.shape}")

    # Clip to valid range [-1, 1]
    if np.max(np.abs(data)) > 1.0:
        import warnings
        warnings.warn(f"Audio data out of range [-1, 1], clipping...")
        data = np.clip(data, -1.0, 1.0)

    # Save audio
    sf.write(path, data, sample_rate, format=format, subtype=subtype)


# ============================================================================
# JSON I/O
# ============================================================================

@operator(
    domain="io_storage",
    category=OpCategory.CONSTRUCT,
    signature="(path: Union[str, Path]) -> Dict[str, Any]",
    deterministic=True,
    doc="Load JSON file as Python dict"
)
def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file as Python dict.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary containing JSON data

    Example:
        config = io.load_json("config.json")
        params = config["parameters"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, 'r') as f:
        return json.load(f)


@operator(
    domain="io_storage",
    category=OpCategory.TRANSFORM,
    signature="(path: Union[str, Path], data: Dict[str, Any], indent: int, sort_keys: bool) -> None",
    deterministic=True,
    doc="Save Python dict as JSON file"
)
def save_json(
    path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2,
    sort_keys: bool = False
) -> None:
    """Save Python dict as JSON file.

    Args:
        path: Output path
        data: Dictionary to save
        indent: Indentation level (default 2, use None for compact)
        sort_keys: Sort dictionary keys alphabetically

    Example:
        params = {"learning_rate": 0.01, "epochs": 100}
        io.save_json("params.json", params)
    """
    path = Path(path)

    # Custom JSON encoder for NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, cls=NumpyEncoder)


# ============================================================================
# HDF5 I/O
# ============================================================================

@operator(
    domain="io_storage",
    category=OpCategory.CONSTRUCT,
    signature="(path: Union[str, Path], dataset: Optional[str]) -> Union[ndarray, Dict[str, ndarray]]",
    deterministic=True,
    doc="Load HDF5 file"
)
def load_hdf5(
    path: Union[str, Path],
    dataset: Optional[str] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Load HDF5 file.

    Args:
        path: Path to HDF5 file
        dataset: Specific dataset to load (None = load all datasets)

    Returns:
        - If dataset is specified: NumPy array
        - If dataset is None: Dictionary mapping dataset names to arrays

    Example:
        # Load specific dataset
        field = io.load_hdf5("sim_output.h5", "velocity_field")

        # Load all datasets
        data = io.load_hdf5("results.h5")
        velocity = data["velocity"]
        pressure = data["pressure"]
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O. "
            "Install with: pip install h5py"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, 'r') as f:
        if dataset is not None:
            if dataset not in f:
                raise KeyError(f"Dataset '{dataset}' not found in {path}")
            return np.array(f[dataset])
        else:
            # Load all datasets
            data = {}
            def load_recursive(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = np.array(obj)
            f.visititems(load_recursive)
            return data


@operator(
    domain="io_storage",
    category=OpCategory.TRANSFORM,
    signature="(path: Union[str, Path], data: Union[ndarray, Dict[str, ndarray]], compression: Optional[str], compression_opts: Optional[int]) -> None",
    deterministic=True,
    doc="Save NumPy arrays to HDF5 file"
)
def save_hdf5(
    path: Union[str, Path],
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    compression: Optional[str] = 'gzip',
    compression_opts: Optional[int] = 4
) -> None:
    """Save NumPy arrays to HDF5 file.

    Args:
        path: Output path
        data: Single array or dict mapping names to arrays
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)

    Example:
        # Save single array
        io.save_hdf5("field.h5", velocity_field)

        # Save multiple arrays
        io.save_hdf5("results.h5", {
            "velocity": vel_field,
            "pressure": press_field,
            "temperature": temp_field
        })
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O. "
            "Install with: pip install h5py"
        )

    path = Path(path)

    with h5py.File(path, 'w') as f:
        if isinstance(data, dict):
            for name, arr in data.items():
                # Build kwargs conditionally
                kwargs = {}
                if compression is not None:
                    kwargs['compression'] = compression
                    if compression_opts is not None:
                        kwargs['compression_opts'] = compression_opts
                f.create_dataset(name, data=arr, **kwargs)
        else:
            # Build kwargs conditionally
            kwargs = {}
            if compression is not None:
                kwargs['compression'] = compression
                if compression_opts is not None:
                    kwargs['compression_opts'] = compression_opts
            f.create_dataset('data', data=data, **kwargs)


# ============================================================================
# CHECKPOINT/RESUME
# ============================================================================

@operator(
    domain="io_storage",
    category=OpCategory.TRANSFORM,
    signature="(path: Union[str, Path], state: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> None",
    deterministic=True,
    doc="Save simulation checkpoint with metadata"
)
def save_checkpoint(
    path: Union[str, Path],
    state: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save simulation checkpoint with metadata.

    Args:
        path: Output path (HDF5 file)
        state: Dictionary containing simulation state (arrays, parameters, etc.)
        metadata: Optional metadata (timestep, iteration, timestamp, etc.)

    Example:
        state = {
            "velocity_field": vel,
            "pressure_field": press,
            "particles": particle_positions,
            "parameters": {"dt": 0.01, "viscosity": 0.1}
        }
        metadata = {
            "iteration": 1000,
            "time": 10.0,
            "timestamp": "2025-11-15T10:30:00"
        }
        io.save_checkpoint("sim_checkpoint_1000.h5", state, metadata)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for checkpointing. "
            "Install with: pip install h5py"
        )

    path = Path(path)

    with h5py.File(path, 'w') as f:
        # Save state data
        state_group = f.create_group('state')
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                state_group.create_dataset(name, data=value, compression='gzip')
            elif isinstance(value, dict):
                # Nested dict → subgroup
                subgroup = state_group.create_group(name)
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        subgroup.create_dataset(k, data=v, compression='gzip')
                    else:
                        subgroup.attrs[k] = v
            else:
                # Scalar → attribute
                state_group.attrs[name] = value

        # Save metadata
        if metadata is not None:
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                meta_group.attrs[key] = value


@operator(
    domain="io_storage",
    category=OpCategory.CONSTRUCT,
    signature="(path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]",
    deterministic=True,
    doc="Load simulation checkpoint"
)
def load_checkpoint(path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load simulation checkpoint.

    Args:
        path: Path to checkpoint file (HDF5)

    Returns:
        Tuple of (state, metadata)

    Example:
        state, metadata = io.load_checkpoint("sim_checkpoint_1000.h5")
        vel = state["velocity_field"]
        iteration = metadata["iteration"]
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for checkpointing. "
            "Install with: pip install h5py"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    state = {}
    metadata = {}

    with h5py.File(path, 'r') as f:
        # Load state
        if 'state' in f:
            state_group = f['state']
            for name in state_group.keys():
                item = state_group[name]
                if isinstance(item, h5py.Dataset):
                    state[name] = np.array(item)
                elif isinstance(item, h5py.Group):
                    # Nested group → dict
                    state[name] = {}
                    for k in item.keys():
                        state[name][k] = np.array(item[k])
                    # Also load attributes
                    for k, v in item.attrs.items():
                        state[name][k] = v

            # Load state attributes (scalars)
            for key, value in state_group.attrs.items():
                state[key] = value

        # Load metadata
        if 'metadata' in f:
            meta_group = f['metadata']
            for key, value in meta_group.attrs.items():
                metadata[key] = value

    return state, metadata


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Image I/O
    'load_image',
    'save_image',

    # Audio I/O
    'load_audio',
    'save_audio',

    # JSON I/O
    'load_json',
    'save_json',

    # HDF5 I/O
    'load_hdf5',
    'save_hdf5',

    # Checkpointing
    'save_checkpoint',
    'load_checkpoint',
]
