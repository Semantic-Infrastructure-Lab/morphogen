"""Physical modeling synthesis.

This module provides physical modeling synthesis including Karplus-Strong
string synthesis and modal synthesis.
"""

from typing import Optional, List
import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(excitation: AudioBuffer, freq: float, t60: float, damping: float) -> AudioBuffer",
    deterministic=True,
    doc="Karplus-Strong string physical model"
)
def string(excitation: AudioBuffer, freq: float, t60: float = 1.5,
           damping: float = 0.3) -> AudioBuffer:
    """Karplus-Strong string physical model.

    Args:
        excitation: Excitation signal (noise burst, pluck, etc.)
        freq: Fundamental frequency in Hz
        t60: Decay time (time to -60dB) in seconds
        damping: High-frequency damping (0.0 to 1.0)

    Returns:
        String resonance output

    Example:
        # Plucked string
        exc = audio.noise(seed=1, duration=0.01)
        exc = audio.lowpass(exc, cutoff=6000.0)
        string_sound = audio.string(exc, freq=220.0, t60=1.5)
    """
    # Handle invalid frequencies
    if freq <= 0:
        return excitation.copy()

    delay_samples = int(excitation.sample_rate / freq)

    if delay_samples <= 0:
        return excitation.copy()

    # Output should be long enough for full decay (t60 + excitation)
    output_duration = excitation.duration + t60
    output_samples = int(output_duration * excitation.sample_rate)
    output = np.zeros(output_samples)

    # Karplus-Strong algorithm
    delay_line = np.zeros(delay_samples)

    # Calculate feedback gain for desired T60
    feedback = 0.001 ** (1.0 / (t60 * freq))

    for i in range(output_samples):
        # Read from delay line
        delayed = delay_line[0]

        # Add excitation (if within excitation duration)
        if i < excitation.num_samples:
            output[i] = excitation.data[i] + delayed
        else:
            output[i] = delayed

        # Lowpass filter for damping (averaging filter)
        if damping > 0:
            if i > 0:
                filtered = (output[i] + output[i-1]) * 0.5
            else:
                filtered = output[i]
            filtered = output[i] * (1.0 - damping) + filtered * damping
        else:
            filtered = output[i]

        # Write to delay line
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = filtered * feedback

    return AudioBuffer(data=output, sample_rate=excitation.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(excitation: AudioBuffer, frequencies: list, decays: list, amplitudes: Optional[list]) -> AudioBuffer",
    deterministic=True,
    doc="Modal synthesis (resonant body)"
)
def modal(excitation: AudioBuffer, frequencies: List[float], decays: List[float],
          amplitudes: Optional[List[float]] = None) -> AudioBuffer:
    """Modal synthesis (resonant body).

    Args:
        excitation: Excitation signal
        frequencies: List of modal frequencies in Hz
        decays: List of decay times in seconds for each mode
        amplitudes: Optional list of relative amplitudes (default: all 1.0)

    Returns:
        Modal synthesis output

    Example:
        # Bell-like sound
        exc = audio.impulse(rate=1.0, duration=0.001)
        bell = audio.modal(exc,
                          frequencies=[200, 370, 550, 720],
                          decays=[2.0, 1.5, 1.0, 0.8])
    """
    from .envelopes import envexp
    from .oscillators import sine

    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)

    if len(frequencies) != len(decays) or len(frequencies) != len(amplitudes):
        raise ValueError("frequencies, decays, and amplitudes must have same length")

    # Output duration should be long enough for longest decay
    max_decay = max(decays)
    output_duration = max_decay * 5.0
    output_samples = int(output_duration * excitation.sample_rate)
    output = np.zeros(output_samples)

    # Each mode is a decaying sinusoid
    for freq, decay, amp in zip(frequencies, decays, amplitudes):
        # Exponential decay envelope
        env = envexp(time_constant=decay / 5.0,
                     duration=output_duration,
                     sample_rate=excitation.sample_rate)

        # Sinusoidal oscillator
        osc = sine(freq=freq, duration=output_duration,
                   sample_rate=excitation.sample_rate)

        # Apply envelope and amplitude
        mode_output = osc.data * env.data * amp

        # Convolve with excitation (simplified - just multiply by excitation energy)
        output += mode_output * np.mean(np.abs(excitation.data))

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak

    return AudioBuffer(data=output, sample_rate=excitation.sample_rate)
