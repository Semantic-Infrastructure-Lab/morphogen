"""Audio filter functions.

This module provides digital audio filters including lowpass, highpass,
bandpass, notch, equalizers, and voltage-controlled filters (VCF).
"""

from typing import Optional, Tuple
import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer, DEFAULT_SAMPLE_RATE
from .helpers import apply_iir_filter


# =============================================================================
# Biquad Coefficient Calculators
# =============================================================================

def biquad_lowpass(cutoff: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad lowpass filter coefficients."""
    w0 = 2.0 * np.pi * cutoff / sample_rate
    alpha = np.sin(w0) / (2.0 * q)

    b0 = (1.0 - np.cos(w0)) / 2.0
    b1 = 1.0 - np.cos(w0)
    b2 = (1.0 - np.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def biquad_highpass(cutoff: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad highpass filter coefficients."""
    w0 = 2.0 * np.pi * cutoff / sample_rate
    alpha = np.sin(w0) / (2.0 * q)

    b0 = (1.0 + np.cos(w0)) / 2.0
    b1 = -(1.0 + np.cos(w0))
    b2 = (1.0 + np.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def biquad_bandpass(center: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad bandpass filter coefficients."""
    w0 = 2.0 * np.pi * center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def biquad_notch(center: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad notch filter coefficients."""
    w0 = 2.0 * np.pi * center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def biquad_low_shelf(cutoff: float, gain_db: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad low shelf filter coefficients."""
    A = np.sqrt(10.0 ** (gain_db / 20.0))
    w0 = 2.0 * np.pi * cutoff / sample_rate
    alpha = np.sin(w0) / 2.0

    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def biquad_high_shelf(cutoff: float, gain_db: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad high shelf filter coefficients."""
    A = np.sqrt(10.0 ** (gain_db / 20.0))
    w0 = 2.0 * np.pi * cutoff / sample_rate
    alpha = np.sin(w0) / 2.0

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def biquad_peaking(center: float, gain_db: float, q: float,
                   sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate biquad peaking filter coefficients."""
    A = np.sqrt(10.0 ** (gain_db / 20.0))
    w0 = 2.0 * np.pi * center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


# =============================================================================
# Time-Varying Filter Functions
# =============================================================================

def apply_time_varying_lowpass(
    signal: np.ndarray,
    cutoff_array: np.ndarray,
    q: float,
    sample_rate: int,
    initial_state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply biquad lowpass filter with time-varying cutoff.

    Args:
        signal: Input signal array
        cutoff_array: Array of cutoff frequencies (one per sample) in Hz
        q: Quality factor (resonance)
        sample_rate: Sample rate in Hz
        initial_state: Optional initial state [z1, z2] for continuity

    Returns:
        Tuple of (filtered signal array, final state [z1, z2])
    """
    output = np.zeros_like(signal)
    state = initial_state.copy() if initial_state is not None else np.zeros(2)

    for i in range(len(signal)):
        cutoff = cutoff_array[i]
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = (1.0 - np.cos(w0)) / 2.0
        b1 = 1.0 - np.cos(w0)
        b2 = (1.0 - np.cos(w0)) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0

        w = signal[i] - a1*state[0] - a2*state[1]
        output[i] = b0*w + b1*state[0] + b2*state[1]
        state[1] = state[0]
        state[0] = w

    return output, state


def apply_time_varying_highpass(
    signal: np.ndarray,
    cutoff_array: np.ndarray,
    q: float,
    sample_rate: int,
    initial_state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply biquad highpass filter with time-varying cutoff.

    Args:
        signal: Input signal array
        cutoff_array: Array of cutoff frequencies (one per sample) in Hz
        q: Quality factor (resonance)
        sample_rate: Sample rate in Hz
        initial_state: Optional initial state [z1, z2] for continuity

    Returns:
        Tuple of (filtered signal array, final state [z1, z2])
    """
    output = np.zeros_like(signal)
    state = initial_state.copy() if initial_state is not None else np.zeros(2)

    for i in range(len(signal)):
        cutoff = cutoff_array[i]
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = (1.0 + np.cos(w0)) / 2.0
        b1 = -(1.0 + np.cos(w0))
        b2 = (1.0 + np.cos(w0)) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0

        w = signal[i] - a1*state[0] - a2*state[1]
        output[i] = b0*w + b1*state[0] + b2*state[1]
        state[1] = state[0]
        state[0] = w

    return output, state


def apply_time_varying_bandpass(
    signal: np.ndarray,
    cutoff_array: np.ndarray,
    q: float,
    sample_rate: int,
    initial_state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply biquad bandpass filter with time-varying center frequency.

    Args:
        signal: Input signal array
        cutoff_array: Array of center frequencies (one per sample) in Hz
        q: Quality factor (bandwidth control)
        sample_rate: Sample rate in Hz
        initial_state: Optional initial state [z1, z2] for continuity

    Returns:
        Tuple of (filtered signal array, final state [z1, z2])
    """
    output = np.zeros_like(signal)
    state = initial_state.copy() if initial_state is not None else np.zeros(2)

    for i in range(len(signal)):
        cutoff = cutoff_array[i]
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0

        w = signal[i] - a1*state[0] - a2*state[1]
        output[i] = b0*w + b1*state[0] + b2*state[1]
        state[1] = state[0]
        state[0] = w

    return output, state


# =============================================================================
# Filter Operators
# =============================================================================

@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, cutoff: float, q: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply lowpass filter"
)
def lowpass(signal: AudioBuffer, cutoff: float = 2000.0, q: float = 0.707) -> AudioBuffer:
    """Apply lowpass filter.

    Args:
        signal: Input audio buffer
        cutoff: Cutoff frequency in Hz
        q: Quality factor (resonance)

    Returns:
        Filtered audio buffer

    Example:
        # Remove high frequencies above 2kHz
        filtered = audio.lowpass(signal, cutoff=2000.0)
    """
    b, a = biquad_lowpass(cutoff, q, signal.sample_rate)
    filtered = apply_iir_filter(signal.data, b, a)
    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, cutoff: float, q: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply highpass filter"
)
def highpass(signal: AudioBuffer, cutoff: float = 120.0, q: float = 0.707) -> AudioBuffer:
    """Apply highpass filter.

    Args:
        signal: Input audio buffer
        cutoff: Cutoff frequency in Hz
        q: Quality factor (resonance)

    Returns:
        Filtered audio buffer
    """
    b, a = biquad_highpass(cutoff, q, signal.sample_rate)
    filtered = apply_iir_filter(signal.data, b, a)
    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, center: float, q: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply bandpass filter"
)
def bandpass(signal: AudioBuffer, center: float = 1000.0, q: float = 1.0) -> AudioBuffer:
    """Apply bandpass filter.

    Args:
        signal: Input audio buffer
        center: Center frequency in Hz
        q: Quality factor (bandwidth)

    Returns:
        Filtered audio buffer
    """
    b, a = biquad_bandpass(center, q, signal.sample_rate)
    filtered = apply_iir_filter(signal.data, b, a)
    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, center: float, q: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply notch (band-stop) filter"
)
def notch(signal: AudioBuffer, center: float = 1000.0, q: float = 1.0) -> AudioBuffer:
    """Apply notch (band-stop) filter.

    Args:
        signal: Input audio buffer
        center: Center frequency in Hz
        q: Quality factor (bandwidth)

    Returns:
        Filtered audio buffer
    """
    b, a = biquad_notch(center, q, signal.sample_rate)
    filtered = apply_iir_filter(signal.data, b, a)
    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, bass: float, mid: float, treble: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply 3-band equalizer"
)
def eq3(signal: AudioBuffer, bass: float = 0.0, mid: float = 0.0,
        treble: float = 0.0) -> AudioBuffer:
    """Apply 3-band equalizer.

    Args:
        signal: Input audio buffer
        bass: Bass gain in dB (-12 to +12)
        mid: Mid gain in dB (-12 to +12)
        treble: Treble gain in dB (-12 to +12)

    Returns:
        Equalized audio buffer
    """
    # Apply low shelf for bass
    if abs(bass) > 0.01:
        b, a = biquad_low_shelf(100.0, bass, signal.sample_rate)
        signal = AudioBuffer(
            data=apply_iir_filter(signal.data, b, a),
            sample_rate=signal.sample_rate
        )

    # Apply peaking filter for mids
    if abs(mid) > 0.01:
        b, a = biquad_peaking(1000.0, mid, 1.0, signal.sample_rate)
        signal = AudioBuffer(
            data=apply_iir_filter(signal.data, b, a),
            sample_rate=signal.sample_rate
        )

    # Apply high shelf for treble
    if abs(treble) > 0.01:
        b, a = biquad_high_shelf(8000.0, treble, signal.sample_rate)
        signal = AudioBuffer(
            data=apply_iir_filter(signal.data, b, a),
            sample_rate=signal.sample_rate
        )

    return signal


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, cutoff: AudioBuffer, q: float, filter_state: Optional[AudioBuffer]) -> AudioBuffer",
    deterministic=True,
    doc="Voltage-controlled lowpass filter with time-varying cutoff"
)
def vcf_lowpass(signal: AudioBuffer, cutoff: AudioBuffer, q: float = 0.707,
                filter_state: Optional[AudioBuffer] = None) -> AudioBuffer:
    """Apply voltage-controlled lowpass filter with modulated cutoff.

    Args:
        signal: Input audio buffer
        cutoff: Cutoff frequency modulation as AudioBuffer (in Hz)
        q: Quality factor (resonance), typically 0.5 to 10.0
        filter_state: Optional filter state for continuity across hops

    Returns:
        Filtered audio buffer
    """
    # Ensure cutoff buffer length matches signal
    if len(cutoff.data) != len(signal.data):
        cutoff_resampled = np.interp(
            np.linspace(0, len(cutoff.data) - 1, len(signal.data)),
            np.arange(len(cutoff.data)),
            cutoff.data
        )
    else:
        cutoff_resampled = cutoff.data

    # Clamp cutoff to valid range
    nyquist = signal.sample_rate / 2.0
    cutoff_clamped = np.clip(cutoff_resampled, 20.0, nyquist * 0.95)

    # Extract initial state if provided
    initial_state = filter_state.data[:2] if filter_state is not None else None

    # Apply time-varying biquad filter
    filtered, final_state = apply_time_varying_lowpass(
        signal.data, cutoff_clamped, q, signal.sample_rate, initial_state
    )

    # Update filter state with final state
    if filter_state is not None:
        filter_state.data[:2] = final_state

    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, cutoff: AudioBuffer, q: float, filter_state: Optional[AudioBuffer]) -> AudioBuffer",
    deterministic=True,
    doc="Voltage-controlled highpass filter with time-varying cutoff"
)
def vcf_highpass(signal: AudioBuffer, cutoff: AudioBuffer, q: float = 0.707,
                 filter_state: Optional[AudioBuffer] = None) -> AudioBuffer:
    """Apply voltage-controlled highpass filter with modulated cutoff.

    Args:
        signal: Input audio buffer
        cutoff: Cutoff frequency modulation as AudioBuffer (in Hz)
        q: Quality factor (resonance), typically 0.5 to 10.0
        filter_state: Optional filter state for continuity across hops

    Returns:
        Filtered audio buffer
    """
    # Ensure cutoff buffer length matches signal
    if len(cutoff.data) != len(signal.data):
        cutoff_resampled = np.interp(
            np.linspace(0, len(cutoff.data) - 1, len(signal.data)),
            np.arange(len(cutoff.data)),
            cutoff.data
        )
    else:
        cutoff_resampled = cutoff.data

    # Clamp cutoff to valid range
    nyquist = signal.sample_rate / 2.0
    cutoff_clamped = np.clip(cutoff_resampled, 20.0, nyquist * 0.95)

    # Extract initial state if provided
    initial_state = filter_state.data[:2] if filter_state is not None else None

    # Apply time-varying biquad filter
    filtered, final_state = apply_time_varying_highpass(
        signal.data, cutoff_clamped, q, signal.sample_rate, initial_state
    )

    # Update filter state with final state
    if filter_state is not None:
        filter_state.data[:2] = final_state

    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, center_freq: AudioBuffer, q: float, filter_state: Optional[AudioBuffer]) -> AudioBuffer",
    deterministic=True,
    doc="Voltage-controlled bandpass filter with time-varying center frequency"
)
def vcf_bandpass(signal: AudioBuffer, center_freq: AudioBuffer, q: float = 1.0,
                 filter_state: Optional[AudioBuffer] = None) -> AudioBuffer:
    """Apply voltage-controlled bandpass filter with modulated center frequency.

    Args:
        signal: Input audio buffer
        center_freq: Center frequency modulation as AudioBuffer (in Hz)
        q: Quality factor (bandwidth control), typically 0.5 to 10.0
        filter_state: Optional filter state for continuity across hops

    Returns:
        Filtered audio buffer
    """
    # Ensure center_freq buffer length matches signal
    if len(center_freq.data) != len(signal.data):
        center_resampled = np.interp(
            np.linspace(0, len(center_freq.data) - 1, len(signal.data)),
            np.arange(len(center_freq.data)),
            center_freq.data
        )
    else:
        center_resampled = center_freq.data

    # Clamp center frequency to valid range
    nyquist = signal.sample_rate / 2.0
    center_clamped = np.clip(center_resampled, 20.0, nyquist * 0.95)

    # Extract initial state if provided
    initial_state = filter_state.data[:2] if filter_state is not None else None

    # Apply time-varying biquad filter
    filtered, final_state = apply_time_varying_bandpass(
        signal.data, center_clamped, q, signal.sample_rate, initial_state
    )

    # Update filter state with final state
    if filter_state is not None:
        filter_state.data[:2] = final_state

    return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)
