"""Audio buffer class for sample storage.

This module provides the AudioBuffer class which represents audio signals
as NumPy arrays with associated sample rate metadata.
"""

import numpy as np


# Default audio parameters
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_CONTROL_RATE = 1000  # Hz


class AudioBuffer:
    """Audio buffer representing a stream of samples.

    Represents audio-rate (Sig) or control-rate (Ctl) signals as NumPy arrays
    with associated sample rate and metadata.

    Example:
        # Create a 1-second buffer at 44.1kHz
        buf = AudioBuffer(
            data=np.zeros(44100),
            sample_rate=44100
        )
    """

    def __init__(self, data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """Initialize audio buffer.

        Args:
            data: NumPy array of samples (1D for mono, 2D for multi-channel)
            sample_rate: Sample rate in Hz
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        return len(self.data)

    @property
    def is_stereo(self) -> bool:
        """Check if buffer is stereo."""
        return len(self.data.shape) > 1 and self.data.shape[1] == 2

    def copy(self) -> 'AudioBuffer':
        """Create a deep copy of this buffer."""
        return AudioBuffer(data=self.data.copy(), sample_rate=self.sample_rate)

    def __repr__(self) -> str:
        """String representation."""
        channels = "stereo" if self.is_stereo else "mono"
        return f"AudioBuffer({channels}, {self.num_samples} samples, {self.sample_rate}Hz)"
