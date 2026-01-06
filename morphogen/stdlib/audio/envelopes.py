"""Audio envelope generators.

This module provides envelope generators including ADSR, AR, and exponential decay.
"""

import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer, DEFAULT_SAMPLE_RATE


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(attack: float, decay: float, sustain: float, release: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate ADSR envelope"
)
def adsr(attack: float = 0.005, decay: float = 0.08, sustain: float = 0.7,
         release: float = 0.2, duration: float = 1.0,
         sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate ADSR envelope.

    Args:
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain: Sustain level (0.0 to 1.0)
        release: Release time in seconds
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with ADSR envelope

    Example:
        # Classic synth envelope
        env = audio.adsr(attack=0.01, decay=0.1, sustain=0.6, release=0.3, duration=1.0)
    """
    num_samples = int(duration * sample_rate)
    envelope = np.zeros(num_samples)

    # Calculate sample counts for each stage
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    # Ensure we don't overflow
    attack_samples = min(attack_samples, num_samples)
    decay_samples = min(decay_samples, num_samples - attack_samples)
    release_samples = min(release_samples, num_samples)

    sustain_samples = num_samples - attack_samples - decay_samples - release_samples
    sustain_samples = max(0, sustain_samples)

    idx = 0

    # Attack: 0 -> 1
    if attack_samples > 0:
        envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
        idx += attack_samples

    # Decay: 1 -> sustain
    if decay_samples > 0:
        envelope[idx:idx+decay_samples] = np.linspace(1, sustain, decay_samples)
        idx += decay_samples

    # Sustain: hold at sustain level
    if sustain_samples > 0:
        envelope[idx:idx+sustain_samples] = sustain
        idx += sustain_samples

    # Release: sustain -> 0
    if release_samples > 0 and idx < num_samples:
        actual_release = min(release_samples, num_samples - idx)
        envelope[idx:idx+actual_release] = np.linspace(sustain, 0, actual_release)

    return AudioBuffer(data=envelope, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(attack: float, release: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate AR (Attack-Release) envelope"
)
def ar(attack: float = 0.005, release: float = 0.3, duration: float = 1.0,
       sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate AR (Attack-Release) envelope.

    Args:
        attack: Attack time in seconds
        release: Release time in seconds
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with AR envelope
    """
    num_samples = int(duration * sample_rate)
    envelope = np.zeros(num_samples)

    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)

    attack_samples = min(attack_samples, num_samples)
    release_samples = min(release_samples, num_samples - attack_samples)

    idx = 0

    # Attack: 0 -> 1
    if attack_samples > 0:
        envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
        idx += attack_samples

    # Release: 1 -> 0
    if release_samples > 0 and idx < num_samples:
        actual_release = min(release_samples, num_samples - idx)
        envelope[idx:idx+actual_release] = np.linspace(1, 0, actual_release)

    return AudioBuffer(data=envelope, sample_rate=sample_rate)


@operator(
    domain="audio",
    category=OpCategory.CONSTRUCT,
    signature="(time_constant: float, duration: float, sample_rate: int) -> AudioBuffer",
    deterministic=True,
    doc="Generate exponential decay envelope"
)
def envexp(time_constant: float = 0.05, duration: float = 1.0,
           sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
    """Generate exponential decay envelope.

    Args:
        time_constant: Time constant (63% decay time) in seconds
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        AudioBuffer with exponential envelope
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    envelope = np.exp(-t / time_constant)
    return AudioBuffer(data=envelope, sample_rate=sample_rate)
