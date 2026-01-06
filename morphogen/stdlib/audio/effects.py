"""Audio effects processing.

This module provides audio effects including delay, reverb, chorus, flanger,
distortion, and limiting.
"""

import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, time: float, feedback: float, mix: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply delay effect"
)
def delay(signal: AudioBuffer, time: float = 0.3, feedback: float = 0.3,
          mix: float = 0.25) -> AudioBuffer:
    """Apply delay effect.

    Args:
        signal: Input audio buffer
        time: Delay time in seconds
        feedback: Feedback amount (0.0 to <1.0)
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet)

    Returns:
        Audio buffer with delay

    Example:
        # Classic slapback delay
        delayed = audio.delay(signal, time=0.125, feedback=0.3, mix=0.3)
    """
    delay_samples = int(time * signal.sample_rate)

    if delay_samples <= 0:
        return signal.copy()

    # Create delay buffer
    delayed = np.zeros_like(signal.data)

    # Simple delay: shift signal and add feedback
    for i in range(len(signal.data)):
        if i < delay_samples:
            delayed[i] = 0.0
        else:
            # Delayed input + feedback from previous delayed output
            delayed[i] = signal.data[i - delay_samples] + feedback * delayed[i - delay_samples]

    # Mix dry and wet
    output = (1.0 - mix) * signal.data + mix * delayed
    return AudioBuffer(data=output, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, mix: float, size: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply reverb effect (Schroeder reverberator)"
)
def reverb(signal: AudioBuffer, mix: float = 0.12, size: float = 0.8) -> AudioBuffer:
    """Apply reverb effect (Schroeder reverberator).

    Args:
        signal: Input audio buffer
        mix: Dry/wet mix (0.0 to 1.0)
        size: Room size (0.0 to 1.0)

    Returns:
        Audio buffer with reverb
    """
    # Simple Schroeder reverb with 4 comb filters and 2 allpass
    sr = signal.sample_rate

    # Comb filter delays (scaled by room size)
    comb_delays = [int(size * d) for d in [1557, 1617, 1491, 1422]]
    comb_gains = [0.805, 0.827, 0.783, 0.764]

    # Allpass delays
    allpass_delays = [int(size * d) for d in [225, 556]]
    allpass_gains = [0.7, 0.7]

    # Process through comb filters
    wet = np.zeros_like(signal.data)
    for delay_val, gain in zip(comb_delays, comb_gains):
        comb_out = np.zeros_like(signal.data)
        for i in range(len(signal.data)):
            comb_out[i] = signal.data[i]
            if i >= delay_val:
                comb_out[i] += gain * comb_out[i - delay_val]
        wet += comb_out

    wet = wet / len(comb_delays)

    # Process through allpass filters
    for delay_val, gain in zip(allpass_delays, allpass_gains):
        allpass_out = np.zeros_like(wet)
        for i in range(len(wet)):
            if i >= delay_val:
                allpass_out[i] = -gain * wet[i] + wet[i - delay_val] + gain * allpass_out[i - delay_val]
            else:
                allpass_out[i] = -gain * wet[i]
        wet = allpass_out

    # Mix dry and wet
    output = (1.0 - mix) * signal.data + mix * wet
    return AudioBuffer(data=output, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, rate: float, depth: float, mix: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply chorus effect"
)
def chorus(signal: AudioBuffer, rate: float = 0.3, depth: float = 0.008,
           mix: float = 0.25) -> AudioBuffer:
    """Apply chorus effect.

    Args:
        signal: Input audio buffer
        rate: LFO rate in Hz
        depth: Modulation depth in seconds
        mix: Dry/wet mix

    Returns:
        Audio buffer with chorus
    """
    from .oscillators import sine

    # Generate LFO
    lfo = sine(freq=rate, duration=signal.duration, sample_rate=signal.sample_rate)

    # Modulated delay line
    base_delay = 0.02  # 20ms base delay
    depth_samples = depth * signal.sample_rate
    base_samples = int(base_delay * signal.sample_rate)

    wet = np.zeros_like(signal.data)
    for i in range(len(signal.data)):
        # Calculate modulated delay
        mod_delay = int(base_samples + depth_samples * lfo.data[i])
        mod_delay = max(0, min(mod_delay, i))

        if i >= mod_delay:
            wet[i] = signal.data[i - mod_delay]

    # Mix dry and wet
    output = (1.0 - mix) * signal.data + mix * wet
    return AudioBuffer(data=output, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, rate: float, depth: float, feedback: float, mix: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply flanger effect"
)
def flanger(signal: AudioBuffer, rate: float = 0.2, depth: float = 0.003,
            feedback: float = 0.25, mix: float = 0.5) -> AudioBuffer:
    """Apply flanger effect.

    Args:
        signal: Input audio buffer
        rate: LFO rate in Hz
        depth: Modulation depth in seconds
        feedback: Feedback amount
        mix: Dry/wet mix

    Returns:
        Audio buffer with flanger
    """
    from .oscillators import sine

    # Similar to chorus but with shorter delay and feedback
    lfo = sine(freq=rate, duration=signal.duration, sample_rate=signal.sample_rate)

    base_delay = 0.001  # 1ms base delay
    depth_samples = depth * signal.sample_rate
    base_samples = int(base_delay * signal.sample_rate)

    wet = np.zeros_like(signal.data)
    for i in range(len(signal.data)):
        mod_delay = int(base_samples + depth_samples * lfo.data[i])
        mod_delay = max(0, min(mod_delay, i))

        if i >= mod_delay:
            wet[i] = signal.data[i - mod_delay]
            if i >= mod_delay and mod_delay > 0:
                wet[i] += feedback * wet[i - mod_delay]

    output = (1.0 - mix) * signal.data + mix * wet
    return AudioBuffer(data=output, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, amount: float, shape: str) -> AudioBuffer",
    deterministic=True,
    doc="Apply distortion/drive"
)
def drive(signal: AudioBuffer, amount: float = 0.5, shape: str = "tanh") -> AudioBuffer:
    """Apply distortion/drive.

    Args:
        signal: Input audio buffer
        amount: Drive amount (0.0 to 1.0)
        shape: Distortion shape ("tanh", "hard", "soft")

    Returns:
        Distorted audio buffer
    """
    gain = 1.0 + amount * 10.0
    driven = signal.data * gain

    if shape == "tanh":
        # Smooth saturation
        output = np.tanh(driven)
    elif shape == "hard":
        # Hard clipping
        output = np.clip(driven, -1.0, 1.0)
    elif shape == "soft":
        # Soft clipping (cubic)
        output = np.where(np.abs(driven) < 1.0,
                        driven - (driven ** 3) / 3.0,
                        np.sign(driven))
    else:
        raise ValueError(f"Unknown distortion shape: {shape}")

    # Compensate for gain
    output = output / (1.0 + amount)

    return AudioBuffer(data=output, sample_rate=signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, threshold: float, release: float) -> AudioBuffer",
    deterministic=True,
    doc="Apply limiter/compressor"
)
def limiter(signal: AudioBuffer, threshold: float = -1.0,
            release: float = 0.05) -> AudioBuffer:
    """Apply limiter/compressor.

    Args:
        signal: Input audio buffer
        threshold: Threshold in dB
        release: Release time in seconds

    Returns:
        Limited audio buffer
    """
    from .mixing import db2lin

    threshold_lin = db2lin(threshold)

    # Simple peak limiter
    output = signal.data.copy()
    gain = 1.0
    release_coef = np.exp(-1.0 / (release * signal.sample_rate))

    for i in range(len(output)):
        # Detect peak
        peak = abs(output[i])

        # Calculate required gain reduction
        if peak > threshold_lin:
            target_gain = threshold_lin / (peak + 1e-6)
        else:
            target_gain = 1.0

        # Smooth gain changes
        if target_gain < gain:
            gain = target_gain  # Fast attack
        else:
            gain = target_gain + (gain - target_gain) * release_coef  # Slow release

        output[i] *= gain

    return AudioBuffer(data=output, sample_rate=signal.sample_rate)
