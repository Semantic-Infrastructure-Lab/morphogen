"""Audio buffer transformation operations.

This module provides buffer manipulation functions including slicing,
concatenation, resampling, reversing, and fade effects.
"""

from typing import Optional
import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, start: float, end: Optional[float]) -> AudioBuffer",
    deterministic=True,
    doc="Extract a portion of an audio buffer"
)
def slice(signal: AudioBuffer, start: float = 0.0, end: Optional[float] = None) -> AudioBuffer:
    """Extract a portion of an audio buffer.

    Args:
        signal: Input audio buffer
        start: Start time in seconds
        end: End time in seconds (None = end of buffer)

    Returns:
        Sliced audio buffer

    Example:
        # Extract middle second
        sliced = audio.slice(signal, start=1.0, end=2.0)
    """
    start_sample = int(start * signal.sample_rate)
    end_sample = int(end * signal.sample_rate) if end is not None else signal.num_samples

    # Clamp to valid range
    start_sample = max(0, min(start_sample, signal.num_samples))
    end_sample = max(start_sample, min(end_sample, signal.num_samples))

    return AudioBuffer(data=signal.data[start_sample:end_sample].copy(),
                      sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(*signals: AudioBuffer) -> AudioBuffer",
    deterministic=True,
    doc="Concatenate multiple audio buffers"
)
def concat(*signals: AudioBuffer) -> AudioBuffer:
    """Concatenate multiple audio buffers.

    Args:
        *signals: Audio buffers to concatenate

    Returns:
        Concatenated audio buffer

    Example:
        # Join three sounds
        combined = audio.concat(intro, middle, outro)
    """
    if not signals:
        raise ValueError("At least one signal required")

    # Ensure all have same sample rate
    sample_rate = signals[0].sample_rate
    for sig in signals:
        if sig.sample_rate != sample_rate:
            raise ValueError("All signals must have same sample rate")

    # Concatenate data
    data = np.concatenate([sig.data for sig in signals])
    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, new_sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Resample audio buffer to a different sample rate"
)
def resample(signal: AudioBuffer, new_sample_rate: int) -> AudioBuffer:
    """Resample audio buffer to a different sample rate.

    Args:
        signal: Input audio buffer
        new_sample_rate: Target sample rate in Hz

    Returns:
        Resampled audio buffer

    Example:
        # Convert 44.1kHz to 48kHz
        resampled = audio.resample(signal, new_sample_rate=48000)
    """
    if signal.sample_rate == new_sample_rate:
        return signal.copy()

    # Calculate new length
    ratio = new_sample_rate / signal.sample_rate
    new_length = int(signal.num_samples * ratio)

    # Linear interpolation resampling
    old_indices = np.arange(signal.num_samples)
    new_indices = np.linspace(0, signal.num_samples - 1, new_length)

    # Handle stereo
    if signal.is_stereo:
        left = np.interp(new_indices, old_indices, signal.data[:, 0])
        right = np.interp(new_indices, old_indices, signal.data[:, 1])
        data = np.column_stack([left, right])
    else:
        data = np.interp(new_indices, old_indices, signal.data)

    return AudioBuffer(data=data, sample_rate=new_sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer) -> AudioBuffer",
    deterministic=True,
    doc="Reverse an audio buffer"
)
def reverse(signal: AudioBuffer) -> AudioBuffer:
    """Reverse an audio buffer.

    Args:
        signal: Input audio buffer

    Returns:
        Reversed audio buffer

    Example:
        # Reverse audio
        backwards = audio.reverse(signal)
    """
    return AudioBuffer(data=signal.data[::-1].copy(), sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, duration: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply fade-in to audio buffer"
)
def fade_in(signal: AudioBuffer, duration: float = 0.05) -> AudioBuffer:
    """Apply fade-in envelope.

    Args:
        signal: Input audio buffer
        duration: Fade duration in seconds

    Returns:
        Audio buffer with fade-in
    """
    fade_samples = int(duration * signal.sample_rate)
    fade_samples = min(fade_samples, signal.num_samples)

    envelope = np.ones(signal.num_samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)

    if signal.is_stereo:
        data = signal.data.copy()
        data[:, 0] *= envelope
        data[:, 1] *= envelope
    else:
        data = signal.data * envelope

    return AudioBuffer(data=data, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, duration: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply fade-out to audio buffer"
)
def fade_out(signal: AudioBuffer, duration: float = 0.05) -> AudioBuffer:
    """Apply fade-out envelope.

    Args:
        signal: Input audio buffer
        duration: Fade duration in seconds

    Returns:
        Audio buffer with fade-out
    """
    fade_samples = int(duration * signal.sample_rate)
    fade_samples = min(fade_samples, signal.num_samples)

    envelope = np.ones(signal.num_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    if signal.is_stereo:
        data = signal.data.copy()
        data[:, 0] *= envelope
        data[:, 1] *= envelope
    else:
        data = signal.data * envelope

    return AudioBuffer(data=data, sample_rate=signal.sample_rate)
