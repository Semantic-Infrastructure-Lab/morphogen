"""Audio oscillator functions.

This module provides basic waveform generators including sine, sawtooth,
square, triangle, noise, impulse train, and constant value generators.
"""

import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer, DEFAULT_SAMPLE_RATE
from .helpers import apply_iir_filter


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(freq: float, phase: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate sine wave oscillator"
)
def sine(freq: float = 440.0, phase: float = 0.0, duration: float = 1.0,
         sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate sine wave oscillator.

    Args:
        freq: Frequency in Hz
        phase: Initial phase in radians (0 to 2pi)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with sine wave

    Example:
        # A440 tone for 1 second
        tone = audio.sine(freq=440.0, duration=1.0)
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    data = np.sin(2.0 * np.pi * freq * t + phase)
    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(freq: float, phase: float, duration: float, blep: bool, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate sawtooth wave oscillator"
)
def saw(freq: float = 440.0, phase: float = 0.0, duration: float = 1.0, blep: bool = True,
        sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate sawtooth wave oscillator.

    Args:
        freq: Frequency in Hz
        phase: Initial phase in radians (0 to 2pi)
        duration: Duration in seconds
        blep: Enable band-limiting (PolyBLEP)
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with sawtooth wave
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Convert phase from radians to normalized (0-1)
    phase_norm = phase / (2.0 * np.pi)

    if blep:
        # PolyBLEP sawtooth (band-limited)
        phase_t = (freq * t + phase_norm) % 1.0
        data = 2.0 * phase_t - 1.0

        # Simple PolyBLEP residual
        dt = freq / sample_rate
        for i in range(num_samples):
            t_norm = phase_t[i]
            if t_norm < dt:
                t_norm = t_norm / dt
                data[i] += t_norm + t_norm - t_norm * t_norm - 1.0
            elif t_norm > 1.0 - dt:
                t_norm = (t_norm - 1.0) / dt
                data[i] += t_norm * t_norm + t_norm + t_norm + 1.0
    else:
        # Naive sawtooth (aliased)
        phase_t = (freq * t + phase_norm) % 1.0
        data = 2.0 * phase_t - 1.0

    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(freq: float, phase: float, pwm: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate square wave oscillator"
)
def square(freq: float = 440.0, phase: float = 0.0, pwm: float = 0.5, duration: float = 1.0,
           sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate square wave oscillator.

    Args:
        freq: Frequency in Hz
        phase: Initial phase in radians (0 to 2pi)
        pwm: Pulse width modulation (0.0 to 1.0, 0.5 = 50% duty cycle)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with square wave
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Convert phase from radians to normalized (0-1)
    phase_norm = phase / (2.0 * np.pi)
    phase_t = (freq * t + phase_norm) % 1.0

    data = np.where(phase_t < pwm, 1.0, -1.0)
    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(freq: float, phase: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate triangle wave oscillator"
)
def triangle(freq: float = 440.0, phase: float = 0.0, duration: float = 1.0,
             sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate triangle wave oscillator.

    Args:
        freq: Frequency in Hz
        phase: Initial phase in radians (0 to 2pi)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with triangle wave
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Convert phase from radians to normalized (0-1)
    phase_norm = phase / (2.0 * np.pi)
    phase_t = (freq * t + phase_norm) % 1.0

    # Triangle: ramp up from -1 to 1, then down from 1 to -1
    data = np.where(phase_t < 0.5,
                   4.0 * phase_t - 1.0,  # Rising edge
                   3.0 - 4.0 * phase_t)   # Falling edge
    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(noise_type: str, seed: int, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=False,
    doc="Generate noise oscillator"
)
def noise(noise_type: str = "white", seed: int = 0, duration: float = 1.0,
          sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate noise oscillator.

    Args:
        noise_type: Type of noise ("white", "pink", "brown")
        seed: Random seed for deterministic output
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with noise

    Example:
        # White noise, deterministic
        noise = audio.noise(noise_type="white", seed=42, duration=1.0)
    """
    rng = np.random.RandomState(seed)
    num_samples = int(duration * sample_rate)

    if noise_type == "white":
        data = rng.randn(num_samples)
    elif noise_type == "pink":
        # Simple pink noise approximation (1/f)
        white = rng.randn(num_samples)
        # Apply moving average filter for 1/f characteristic
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        # Simple IIR filter implementation
        data = apply_iir_filter(white, b, a)
    elif noise_type == "brown":
        # Brownian noise (integrated white noise)
        white = rng.randn(num_samples)
        data = np.cumsum(white)
        # Normalize to prevent drift
        data = data / (np.max(np.abs(data)) + 1e-6)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize to [-1, 1]
    data = data / (np.max(np.abs(data)) + 1e-6)
    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(rate: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate impulse train"
)
def impulse(rate: float = 1.0, duration: float = 1.0,
            sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate impulse train.

    Args:
        rate: Impulse rate in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with impulse train
    """
    num_samples = int(duration * sample_rate)
    data = np.zeros(num_samples)

    # Place impulses at regular intervals
    interval = int(sample_rate / rate)
    if interval > 0:
        data[::interval] = 1.0

    return AudioBuffer(data=data, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(value: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate constant-value AudioBuffer"
)
def constant(value: float = 0.0, duration: float = 1.0,
             sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate constant-value AudioBuffer.

    Creates an AudioBuffer where all samples have the same value.
    Useful for fixed cutoff frequencies, DC offsets, bias voltages,
    and any scenario requiring a non-varying signal.

    Args:
        value: Constant value for all samples
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with constant value

    Example:
        # Fixed 2kHz cutoff for VCF filter
        cutoff = audio.constant(value=2000.0, duration=1.0)

        # DC offset
        dc_offset = audio.constant(value=0.5, duration=2.0)
    """
    num_samples = int(duration * sample_rate)
    data = np.full(num_samples, value)
    return AudioBuffer(data=data, sample_rate=sample_rate)
