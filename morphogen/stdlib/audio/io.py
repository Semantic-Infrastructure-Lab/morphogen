"""Audio input/output operations.

This module provides audio file loading, saving, playback, and recording.
"""

from typing import Optional
import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer, DEFAULT_SAMPLE_RATE


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(buffer: AudioBuffer, blocking: bool) -> None",
    deterministic=False,
    doc="Play audio buffer in real-time"
)
def play(buffer: AudioBuffer, blocking: bool = True) -> None:
    """Play audio buffer in real-time.

    Args:
        buffer: Audio buffer to play
        blocking: If True, wait for playback to complete (default: True)

    Raises:
        ImportError: If sounddevice is not installed
    """
    if not isinstance(buffer, AudioBuffer):
        raise TypeError(f"Expected AudioBuffer, got {type(buffer)}")

    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "sounddevice is required for audio playback. "
            "Install with: pip install sounddevice"
        )

    # Prepare data for playback
    if buffer.is_stereo:
        data = buffer.data
    else:
        data = buffer.data.reshape(-1, 1)

    sd.play(data, samplerate=buffer.sample_rate, blocking=blocking)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(buffer: AudioBuffer, path: str, format: str) -> None",
    deterministic=True,
    doc="Save audio buffer to file"
)
def save(buffer: AudioBuffer, path: str, format: str = "auto") -> None:
    """Save audio buffer to file.

    Supports WAV and FLAC formats with automatic format detection from file extension.

    Args:
        buffer: Audio buffer to save
        path: Output file path
        format: Output format ("auto", "wav", "flac") - auto infers from extension

    Raises:
        ImportError: If soundfile is not installed (for FLAC) or scipy (for WAV fallback)
        ValueError: If format is unsupported
    """
    if not isinstance(buffer, AudioBuffer):
        raise TypeError(f"Expected AudioBuffer, got {type(buffer)}")

    # Infer format from path if auto
    if format == "auto":
        if path.endswith(".wav"):
            format = "wav"
        elif path.endswith(".flac"):
            format = "flac"
        else:
            format = "wav"

    format = format.lower()

    # Prepare data
    data = np.clip(buffer.data, -1.0, 1.0).astype(np.float32)

    if format == "flac":
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for FLAC export. "
                "Install with: pip install soundfile"
            )

        if buffer.is_stereo:
            sf.write(path, data, buffer.sample_rate, format='FLAC')
        else:
            sf.write(path, data.reshape(-1, 1), buffer.sample_rate, format='FLAC')

    elif format == "wav":
        try:
            import soundfile as sf
            if buffer.is_stereo:
                sf.write(path, data, buffer.sample_rate, format='WAV')
            else:
                sf.write(path, data.reshape(-1, 1), buffer.sample_rate, format='WAV')
        except ImportError:
            try:
                from scipy.io import wavfile
            except ImportError:
                raise ImportError(
                    "Either soundfile or scipy is required for WAV export. "
                    "Install with: pip install soundfile  OR  pip install scipy"
                )

            data_int16 = (data * 32767).astype(np.int16)
            wavfile.write(path, buffer.sample_rate, data_int16)

    else:
        raise ValueError(
            f"Unsupported format: {format}. Supported: 'wav', 'flac'"
        )

    print(f"Saved audio to: {path}")


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(path: str) -> AudioBuffer",
    deterministic=True,
    doc="Load audio buffer from file"
)
def load(path: str) -> AudioBuffer:
    """Load audio buffer from file.

    Supports WAV and FLAC formats with automatic format detection.

    Args:
        path: Input file path

    Returns:
        Loaded audio buffer

    Raises:
        ImportError: If soundfile is not installed
        FileNotFoundError: If file doesn't exist
    """
    try:
        import soundfile as sf
    except ImportError:
        if path.endswith('.wav'):
            try:
                from scipy.io import wavfile
                sample_rate, data = wavfile.read(path)

                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0

                return AudioBuffer(data=data, sample_rate=sample_rate)
            except ImportError:
                raise ImportError(
                    "Either soundfile or scipy is required for audio loading. "
                    "Install with: pip install soundfile  OR  pip install scipy"
                )
        else:
            raise ImportError(
                "soundfile is required for loading non-WAV audio. "
                "Install with: pip install soundfile"
            )

    data, sample_rate = sf.read(path, dtype='float32')
    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(duration: float, sample_rate: int, channels: int) -> AudioBuffer",
    deterministic=False,
    doc="Record audio from microphone"
)
def record(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
           channels: int = 1) -> AudioBuffer:
    """Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels (1=mono, 2=stereo)

    Returns:
        Recorded audio buffer

    Raises:
        ImportError: If sounddevice is not installed
    """
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 (mono) or 2 (stereo), got {channels}")

    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "sounddevice is required for audio recording. "
            "Install with: pip install sounddevice"
        )

    print(f"Recording {duration}s of audio...")

    data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='float32'
    )
    sd.wait()

    print("Recording complete!")

    if channels == 1:
        data = data.reshape(-1)

    return AudioBuffer(data=data, sample_rate=sample_rate)
