"""Audio mixing and utility functions.

This module provides mixing, gain control, panning, and utility functions
for audio signal processing.
"""

import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(*signals: AudioBuffer) -> AudioBuffer",
    deterministic=True,
    doc="Mix multiple audio signals with gain compensation"
)
def mix(*signals: AudioBuffer) -> AudioBuffer:
    """Mix multiple audio signals with gain compensation.

    Args:
        *signals: Audio buffers to mix

    Returns:
        Mixed audio buffer

    Example:
        # Mix three signals
        mixed = audio.mix(bass, lead, pad)
    """
    if not signals:
        raise ValueError("At least one signal required")

    # Ensure all signals have same length and sample rate
    sample_rate = signals[0].sample_rate
    max_len = max(s.num_samples for s in signals)

    # Sum with gain compensation
    output = np.zeros(max_len)
    for signal in signals:
        # Pad if needed
        if signal.num_samples < max_len:
            padded = np.pad(signal.data, (0, max_len - signal.num_samples))
            output += padded
        else:
            output += signal.data[:max_len]

    # Gain compensate by sqrt(N) to prevent clipping
    output = output / np.sqrt(len(signals))

    return AudioBuffer(data=output, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, amount_db: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply gain in dB"
)
def gain(signal: AudioBuffer, amount_db: float) -> AudioBuffer:
    """Apply gain in dB.

    Args:
        signal: Input audio buffer
        amount_db: Gain in decibels

    Returns:
        Audio buffer with gain applied
    """
    gain_lin = db2lin(amount_db)
    return AudioBuffer(data=signal.data * gain_lin, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal1: AudioBuffer, signal2: AudioBuffer, gain: float) -> AudioBuffer",
    deterministic=True,
    doc="Multiply two audio signals (ring modulation, AM synthesis)"
)
def multiply(signal1: AudioBuffer, signal2: AudioBuffer, gain_val: float = 1.0) -> AudioBuffer:
    """Multiply two audio signals element-wise.

    Used for ring modulation, amplitude modulation, and other
    modulation synthesis techniques.

    Args:
        signal1: First input signal (carrier)
        signal2: Second input signal (modulator)
        gain_val: Output gain multiplier (default 1.0)

    Returns:
        Product of the two signals with gain applied

    Example:
        # Ring modulation: multiply two oscillators
        carrier = audio.sine(freq=440.0, duration=1.0)
        modulator = audio.sine(freq=220.0, duration=1.0)
        ring_mod = audio.multiply(carrier, modulator)
    """
    # Use the sample rate from signal1 (scheduler handles rate conversion)
    sample_rate = signal1.sample_rate

    # Get lengths
    len1 = signal1.num_samples
    len2 = signal2.num_samples
    max_len = max(len1, len2)

    # Pad shorter signal with zeros if needed
    data1 = signal1.data
    data2 = signal2.data

    if len1 < max_len:
        data1 = np.pad(data1, (0, max_len - len1))
    elif len1 > max_len:
        data1 = data1[:max_len]

    if len2 < max_len:
        data2 = np.pad(data2, (0, max_len - len2))
    elif len2 > max_len:
        data2 = data2[:max_len]

    # Element-wise multiplication with gain
    result = data1 * data2 * gain_val

    return AudioBuffer(data=result, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, cv: AudioBuffer, curve: str) -> AudioBuffer",
    deterministic=True,
    doc="Voltage-controlled amplifier for amplitude control"
)
def vca(signal: AudioBuffer, cv: AudioBuffer, curve: str = "linear") -> AudioBuffer:
    """Voltage-controlled amplifier for amplitude control.

    Classic synthesizer VCA module - controls signal amplitude via
    control voltage (CV). Typically used with envelope generators
    to shape note dynamics.

    Args:
        signal: Input audio signal
        cv: Control voltage (0.0 to 1.0 range, auto-normalized)
        curve: Response curve - "linear" or "exponential" (default: "linear")

    Returns:
        Signal with CV-controlled amplitude

    Example:
        # Classic ADSR envelope shaping
        audio_sig = audio.saw(freq=110.0, duration=2.0, sample_rate=48000)
        envelope = audio.adsr(attack=0.1, decay=0.3, sustain=0.6, release=0.5,
                            duration=2.0, sample_rate=1000)
        shaped = audio.vca(audio_sig, envelope)
    """
    # Use the sample rate from signal (scheduler handles rate conversion)
    sample_rate = signal.sample_rate

    # Get lengths
    signal_len = signal.num_samples
    cv_len = cv.num_samples
    max_len = max(signal_len, cv_len)

    # Pad shorter buffer with zeros if needed
    signal_data = signal.data
    cv_data = cv.data

    if signal_len < max_len:
        signal_data = np.pad(signal_data, (0, max_len - signal_len))
    elif signal_len > max_len:
        signal_data = signal_data[:max_len]

    if cv_len < max_len:
        cv_data = np.pad(cv_data, (0, max_len - cv_len))
    elif cv_len > max_len:
        cv_data = cv_data[:max_len]

    # Normalize CV to 0-1 range (handle bipolar CVs)
    cv_normalized = (cv_data - cv_data.min()) / (cv_data.max() - cv_data.min() + 1e-10)

    # Apply curve
    if curve == "exponential":
        # Exponential curve: more natural-sounding amplitude response
        cv_normalized = np.sqrt(cv_normalized)
    elif curve != "linear":
        raise ValueError(f"Invalid curve type: {curve}. Use 'linear' or 'exponential'")

    # Apply CV to signal amplitude
    result = signal_data * cv_normalized

    return AudioBuffer(data=result, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, position: float) -> AudioBuffer",
    deterministic=True,
    doc="Pan mono signal to stereo"
)
def pan(signal: AudioBuffer, position: float = 0.0) -> AudioBuffer:
    """Pan mono signal to stereo.

    Args:
        signal: Input audio buffer (mono)
        position: Pan position (-1.0 = left, 0.0 = center, 1.0 = right)

    Returns:
        Stereo audio buffer
    """
    # Constant power panning
    position = np.clip(position, -1.0, 1.0)
    angle = (position + 1.0) * np.pi / 4.0  # -1..1 -> 0..pi/2

    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    stereo = np.zeros((signal.num_samples, 2))
    stereo[:, 0] = signal.data * left_gain
    stereo[:, 1] = signal.data * right_gain

    return AudioBuffer(data=stereo, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, limit: float) -> AudioBuffer",
    deterministic=True,
    doc="Hard clip signal"
)
def clip(signal: AudioBuffer, limit: float = 0.98) -> AudioBuffer:
    """Hard clip signal.

    Args:
        signal: Input audio buffer
        limit: Clipping threshold (0.0 to 1.0)

    Returns:
        Clipped audio buffer
    """
    clipped = np.clip(signal.data, -limit, limit)
    return AudioBuffer(data=clipped, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, target: float) -> AudioBuffer",
    deterministic=True,
    doc="Normalize signal to target peak level"
)
def normalize(signal: AudioBuffer, target: float = 0.98) -> AudioBuffer:
    """Normalize signal to target peak level.

    Args:
        signal: Input audio buffer
        target: Target peak level (0.0 to 1.0)

    Returns:
        Normalized audio buffer
    """
    peak = np.max(np.abs(signal.data))
    if peak > 1e-6:
        gain_val = target / peak
        return AudioBuffer(data=signal.data * gain_val, sample_rate=signal.sample_rate)
    return signal.copy()


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(db: float) -> float",
    deterministic=True,
    doc="Convert decibels to linear gain"
)
def db2lin(db: float) -> float:
    """Convert decibels to linear gain.

    Args:
        db: Value in decibels

    Returns:
        Linear gain value
    """
    return 10.0 ** (db / 20.0)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(linear: float) -> float",
    deterministic=True,
    doc="Convert linear gain to decibels"
)
def lin2db(linear: float) -> float:
    """Convert linear gain to decibels.

    Args:
        linear: Linear gain value

    Returns:
        Value in decibels
    """
    return 20.0 * np.log10(linear + 1e-10)
