"""Audio operations implementation using NumPy backend.

This module provides NumPy-based implementations of all core audio operations
for deterministic audio synthesis, including oscillators, filters, envelopes,
effects, and physical modeling primitives.

All operations follow the audio-rate model (44.1kHz default) with deterministic
semantics ensuring same seed = same output.
"""

from typing import Callable, Optional, Dict, Any, Tuple, Union
import numpy as np

from morphogen.core.operator import operator, OpCategory


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


class AudioOperations:
    """Namespace for audio operations (accessed as 'audio' in DSL)."""

    # ========================================================================
    # OSCILLATORS (Section 5.1)
    # ========================================================================

    @staticmethod
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
            phase: Initial phase in radians (0 to 2π)
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

    @staticmethod
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
            phase: Initial phase in radians (0 to 2π)
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

    @staticmethod
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
            phase: Initial phase in radians (0 to 2π)
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

    @staticmethod
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
            phase: Initial phase in radians (0 to 2π)
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

    @staticmethod
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
            data = AudioOperations._apply_iir_filter(white, b, a)
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

    @staticmethod
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

    @staticmethod
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

    # ========================================================================
    # FILTERS (Section 5.2)
    # ========================================================================

    @staticmethod
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
        # Biquad lowpass filter
        b, a = AudioOperations._biquad_lowpass(cutoff, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
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
        # Biquad highpass filter
        b, a = AudioOperations._biquad_highpass(cutoff, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
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
        # Biquad bandpass filter
        b, a = AudioOperations._biquad_bandpass(center, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
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
        # Biquad notch filter
        b, a = AudioOperations._biquad_notch(center, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
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
            b, a = AudioOperations._biquad_low_shelf(100.0, bass, signal.sample_rate)
            signal = AudioBuffer(
                data=AudioOperations._apply_iir_filter(signal.data, b, a),
                sample_rate=signal.sample_rate
            )

        # Apply peaking filter for mids
        if abs(mid) > 0.01:
            b, a = AudioOperations._biquad_peaking(1000.0, mid, 1.0, signal.sample_rate)
            signal = AudioBuffer(
                data=AudioOperations._apply_iir_filter(signal.data, b, a),
                sample_rate=signal.sample_rate
            )

        # Apply high shelf for treble
        if abs(treble) > 0.01:
            b, a = AudioOperations._biquad_high_shelf(8000.0, treble, signal.sample_rate)
            signal = AudioBuffer(
                data=AudioOperations._apply_iir_filter(signal.data, b, a),
                sample_rate=signal.sample_rate
            )

        return signal

    @staticmethod
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

        This filter accepts cutoff frequency as an AudioBuffer, enabling
        control-rate modulation (e.g., ADSR envelope controlling filter cutoff).
        The cutoff buffer is automatically resampled to match the signal rate
        by the scheduler.

        Args:
            signal: Input audio buffer
            cutoff: Cutoff frequency modulation as AudioBuffer (in Hz)
            q: Quality factor (resonance), typically 0.5 to 10.0
            filter_state: Optional filter state for continuity across hops

        Returns:
            Filtered audio buffer

        Example:
            # Classic subtractive synth: envelope-controlled filter
            saw = audio.sawtooth(freq=110.0, duration=2.0, sample_rate=48000)
            env = audio.adsr(attack=0.01, decay=0.5, sustain=0.3, release=0.8,
                           duration=2.0, sample_rate=1000)
            # Scale envelope to cutoff range: 200Hz to 4000Hz
            cutoff_mod = AudioBuffer(
                data=200.0 + env.data * 3800.0,
                sample_rate=env.sample_rate
            )
            filtered = audio.vcf_lowpass(saw, cutoff_mod, q=2.0)

        Notes:
            - Cutoff buffer is linearly resampled to signal rate by scheduler
            - Filter coefficients recomputed per-sample for smooth modulation
            - State maintained across coefficient changes for continuity
            - Cutoff values clamped to valid range: [20Hz, nyquist/2]
            - filter_state parameter enables seamless continuation across buffer hops
        """
        # Ensure cutoff buffer length matches signal
        if len(cutoff.data) != len(signal.data):
            # Resample cutoff to match signal length
            # (This should already be done by scheduler, but handle edge case)
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
        filtered, final_state = AudioOperations._apply_time_varying_lowpass(
            signal.data, cutoff_clamped, q, signal.sample_rate, initial_state
        )

        # Update filter state with final state
        if filter_state is not None:
            filter_state.data[:2] = final_state

        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
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

        This filter accepts cutoff frequency as an AudioBuffer, enabling
        control-rate modulation (e.g., ADSR envelope controlling filter cutoff).
        The cutoff buffer is automatically resampled to match the signal rate
        by the scheduler.

        Args:
            signal: Input audio buffer
            cutoff: Cutoff frequency modulation as AudioBuffer (in Hz)
            q: Quality factor (resonance), typically 0.5 to 10.0
            filter_state: Optional filter state for continuity across hops

        Returns:
            Filtered audio buffer

        Example:
            # Highpass filter with envelope sweep
            signal = audio.saw(freq=110.0, duration=2.0, sample_rate=48000)
            env = audio.adsr(attack=0.01, decay=0.5, sustain=0.3, release=0.8,
                           duration=2.0, sample_rate=1000)
            # Scale envelope to cutoff range: 100Hz to 2000Hz
            cutoff_mod = AudioBuffer(
                data=100.0 + env.data * 1900.0,
                sample_rate=env.sample_rate
            )
            filtered = audio.vcf_highpass(signal, cutoff_mod, q=2.0)

        Notes:
            - Cutoff buffer is linearly resampled to signal rate by scheduler
            - Filter coefficients recomputed per-sample for smooth modulation
            - State maintained across coefficient changes for continuity
            - Cutoff values clamped to valid range: [20Hz, nyquist/2]
            - Attenuates frequencies below cutoff, passes above
            - filter_state parameter enables seamless continuation across buffer hops
        """
        # Ensure cutoff buffer length matches signal
        if len(cutoff.data) != len(signal.data):
            # Resample cutoff to match signal length
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
        filtered, final_state = AudioOperations._apply_time_varying_highpass(
            signal.data, cutoff_clamped, q, signal.sample_rate, initial_state
        )

        # Update filter state with final state
        if filter_state is not None:
            filter_state.data[:2] = final_state

        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
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

        This filter accepts center frequency as an AudioBuffer, enabling
        control-rate modulation (e.g., LFO controlling filter center frequency).
        The center_freq buffer is automatically resampled to match the signal rate
        by the scheduler.

        Args:
            signal: Input audio buffer
            center_freq: Center frequency modulation as AudioBuffer (in Hz)
            q: Quality factor (bandwidth control), typically 0.5 to 10.0
               Higher Q = narrower bandwidth, lower Q = wider bandwidth
            filter_state: Optional filter state for continuity across hops

        Returns:
            Filtered audio buffer

        Example:
            # Bandpass filter with LFO sweep (vowel formant effect)
            signal = audio.saw(freq=110.0, duration=2.0, sample_rate=48000)
            lfo = audio.sine(freq=0.5, duration=2.0, sample_rate=1000)
            # Scale LFO to center freq range: 500Hz to 1500Hz
            center_mod = AudioBuffer(
                data=1000.0 + lfo.data * 500.0,
                sample_rate=lfo.sample_rate
            )
            filtered = audio.vcf_bandpass(signal, center_mod, q=5.0)

        Notes:
            - Center_freq buffer is linearly resampled to signal rate by scheduler
            - Filter coefficients recomputed per-sample for smooth modulation
            - State maintained across coefficient changes for continuity
            - Center frequency values clamped to valid range: [20Hz, nyquist/2]
            - Passes frequencies near center, attenuates above and below
            - Q controls bandwidth: Q=1 is wide, Q=10 is narrow
            - Useful for formant synthesis, vowel sounds, wah effects
            - filter_state parameter enables seamless continuation across buffer hops
        """
        # Ensure center_freq buffer length matches signal
        if len(center_freq.data) != len(signal.data):
            # Resample center_freq to match signal length
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
        filtered, final_state = AudioOperations._apply_time_varying_bandpass(
            signal.data, center_clamped, q, signal.sample_rate, initial_state
        )

        # Update filter state with final state
        if filter_state is not None:
            filter_state.data[:2] = final_state

        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    # ========================================================================
    # ENVELOPES (Section 5.3)
    # ========================================================================

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    # ========================================================================
    # EFFECTS (Section 5.4)
    # ========================================================================

    @staticmethod
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

    @staticmethod
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
        for delay, gain in zip(comb_delays, comb_gains):
            comb_out = np.zeros_like(signal.data)
            for i in range(len(signal.data)):
                comb_out[i] = signal.data[i]
                if i >= delay:
                    comb_out[i] += gain * comb_out[i - delay]
            wet += comb_out

        wet = wet / len(comb_delays)

        # Process through allpass filters
        for delay, gain in zip(allpass_delays, allpass_gains):
            allpass_out = np.zeros_like(wet)
            for i in range(len(wet)):
                if i >= delay:
                    allpass_out[i] = -gain * wet[i] + wet[i - delay] + gain * allpass_out[i - delay]
                else:
                    allpass_out[i] = -gain * wet[i]
            wet = allpass_out

        # Mix dry and wet
        output = (1.0 - mix) * signal.data + mix * wet
        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    @staticmethod
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
        # Generate LFO
        lfo = AudioOperations.sine(freq=rate, duration=signal.duration,
                                   sample_rate=signal.sample_rate)

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

    @staticmethod
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
        # Similar to chorus but with shorter delay and feedback
        lfo = AudioOperations.sine(freq=rate, duration=signal.duration,
                                   sample_rate=signal.sample_rate)

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

    @staticmethod
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

    @staticmethod
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
        threshold_lin = AudioOperations.db2lin(threshold)

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

    # ========================================================================
    # UTILITIES (Section 5.5)
    # ========================================================================

    @staticmethod
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

    @staticmethod
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
        gain_lin = AudioOperations.db2lin(amount_db)
        return AudioBuffer(data=signal.data * gain_lin, sample_rate=signal.sample_rate)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(signal1: AudioBuffer, signal2: AudioBuffer, gain: float) -> AudioBuffer",
        deterministic=True,
        doc="Multiply two audio signals (ring modulation, AM synthesis)"
    )
    def multiply(signal1: AudioBuffer, signal2: AudioBuffer, gain: float = 1.0) -> AudioBuffer:
        """Multiply two audio signals element-wise.

        Used for ring modulation, amplitude modulation, and other
        modulation synthesis techniques.

        Args:
            signal1: First input signal (carrier)
            signal2: Second input signal (modulator)
            gain: Output gain multiplier (default 1.0)

        Returns:
            Product of the two signals with gain applied

        Example:
            # Ring modulation: multiply two oscillators
            carrier = audio.sine(freq=440.0, duration=1.0)
            modulator = audio.sine(freq=220.0, duration=1.0)
            ring_mod = audio.multiply(carrier, modulator)

            # Amplitude modulation with envelope
            audio_sig = audio.saw(freq=110.0, duration=2.0, sample_rate=48000)
            envelope = audio.adsr(attack=0.1, decay=0.3, sustain=0.6, release=0.5,
                                duration=2.0, sample_rate=1000)
            # Scheduler will auto-resample envelope from 1kHz to 48kHz
            shaped = audio.multiply(audio_sig, envelope)
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
        result = data1 * data2 * gain

        return AudioBuffer(data=result, sample_rate=sample_rate)

    @staticmethod
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
            # Scheduler will auto-resample envelope from 1kHz to 48kHz
            shaped = audio.vca(audio_sig, envelope)

            # Exponential response (more natural-sounding amplitude curves)
            shaped_exp = audio.vca(audio_sig, envelope, curve="exponential")

            # Tremolo effect with LFO
            audio_sig = audio.sine(freq=440.0, duration=4.0)
            lfo = audio.sine(freq=6.0, duration=4.0)  # 6 Hz tremolo
            tremolo = audio.vca(audio_sig, lfo, curve="linear")
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
            # Square root curve (inverse of quadratic) - resists changes at low values
            # Provides smoother fade-out and more pronounced initial amplitude
            cv_normalized = np.sqrt(cv_normalized)
        elif curve != "linear":
            raise ValueError(f"Invalid curve type: {curve}. Use 'linear' or 'exponential'")

        # Apply CV to signal amplitude
        result = signal_data * cv_normalized

        return AudioBuffer(data=result, sample_rate=sample_rate)

    @staticmethod
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
        angle = (position + 1.0) * np.pi / 4.0  # -1..1 -> 0..π/2

        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        stereo = np.zeros((signal.num_samples, 2))
        stereo[:, 0] = signal.data * left_gain
        stereo[:, 1] = signal.data * right_gain

        return AudioBuffer(data=stereo, sample_rate=signal.sample_rate)

    @staticmethod
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

    @staticmethod
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
            gain = target / peak
            return AudioBuffer(data=signal.data * gain, sample_rate=signal.sample_rate)
        return signal.copy()

    @staticmethod
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

    @staticmethod
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

    # ========================================================================
    # PHYSICAL MODELING (Section 7)
    # ========================================================================

    @staticmethod
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
        # T60 is time to decay to -60dB (amplitude factor of 0.001)
        # feedback^N = 0.001 where N = t60 * freq (number of delay line cycles in t60 seconds)
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
            # Filter the output (excitation + delayed) before feedback
            if damping > 0:
                # Simple averaging filter for damping
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

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(excitation: AudioBuffer, frequencies: list, decays: list, amplitudes: Optional[list]) -> AudioBuffer",
        deterministic=True,
        doc="Modal synthesis (resonant body)"
    )
    def modal(excitation: AudioBuffer, frequencies: list, decays: list,
              amplitudes: Optional[list] = None) -> AudioBuffer:
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
        if amplitudes is None:
            amplitudes = [1.0] * len(frequencies)

        if len(frequencies) != len(decays) or len(frequencies) != len(amplitudes):
            raise ValueError("frequencies, decays, and amplitudes must have same length")

        # Output duration should be long enough for longest decay
        # Use 5 time constants for full decay
        max_decay = max(decays)
        output_duration = max_decay * 5.0
        output_samples = int(output_duration * excitation.sample_rate)
        output = np.zeros(output_samples)

        # Each mode is a decaying sinusoid
        for freq, decay, amp in zip(frequencies, decays, amplitudes):
            # Exponential decay envelope
            env = AudioOperations.envexp(time_constant=decay / 5.0,
                                        duration=output_duration,
                                        sample_rate=excitation.sample_rate)

            # Sinusoidal oscillator
            osc = AudioOperations.sine(freq=freq, duration=output_duration,
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

    # ========================================================================
    # HELPER FUNCTIONS (Internal)
    # ========================================================================

    @staticmethod
    def _apply_iir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Apply IIR filter using Direct Form II."""
        # Normalize coefficients
        a0 = a[0]
        if abs(a0) < 1e-10:
            return signal.copy()

        b = b / a0
        a = a / a0

        # Initialize state
        n_b = len(b)
        n_a = len(a)
        max_order = max(n_b, n_a) - 1

        # Pad coefficients
        if n_b < max_order + 1:
            b = np.pad(b, (0, max_order + 1 - n_b))
        if n_a < max_order + 1:
            a = np.pad(a, (0, max_order + 1 - n_a))

        # Apply filter
        output = np.zeros_like(signal)
        state = np.zeros(max_order)

        for i in range(len(signal)):
            # Direct Form II
            w = signal[i]
            for j in range(1, len(a)):
                w -= a[j] * state[j - 1] if j - 1 < len(state) else 0

            output[i] = b[0] * w
            for j in range(1, len(b)):
                output[i] += b[j] * state[j - 1] if j - 1 < len(state) else 0

            # Update state
            state = np.roll(state, 1)
            state[0] = w

        return output

    @staticmethod
    def _biquad_lowpass(cutoff: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _biquad_highpass(cutoff: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _biquad_bandpass(center: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _biquad_notch(center: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _biquad_low_shelf(cutoff: float, gain_db: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _biquad_high_shelf(cutoff: float, gain_db: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _biquad_peaking(center: float, gain_db: float, q: float,
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

    @staticmethod
    def _apply_time_varying_lowpass(
        signal: np.ndarray,
        cutoff_array: np.ndarray,
        q: float,
        sample_rate: int,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply biquad lowpass filter with time-varying cutoff.

        This implements a sample-by-sample biquad filter where the cutoff
        frequency changes over time. Filter coefficients are recomputed
        for each sample, and the filter state is maintained for continuity.

        Args:
            signal: Input signal array
            cutoff_array: Array of cutoff frequencies (one per sample) in Hz
            q: Quality factor (resonance)
            sample_rate: Sample rate in Hz
            initial_state: Optional initial state [z1, z2] for continuity across hops

        Returns:
            Tuple of (filtered signal array, final state [z1, z2])

        Implementation notes:
            - Uses Direct Form II for state continuity
            - Coefficients recomputed per-sample for smooth modulation
            - State vector preserved across coefficient changes
            - Normalized coefficients to avoid numerical issues
            - State can be preserved across buffer hops for seamless continuation
        """
        output = np.zeros_like(signal)
        # Biquad state (2 elements for 2nd-order filter)
        state = initial_state.copy() if initial_state is not None else np.zeros(2)

        for i in range(len(signal)):
            # Compute filter coefficients for current cutoff
            cutoff = cutoff_array[i]
            w0 = 2.0 * np.pi * cutoff / sample_rate
            alpha = np.sin(w0) / (2.0 * q)

            # Biquad lowpass coefficients
            b0 = (1.0 - np.cos(w0)) / 2.0
            b1 = 1.0 - np.cos(w0)
            b2 = (1.0 - np.cos(w0)) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * np.cos(w0)
            a2 = 1.0 - alpha

            # Normalize by a0
            b0, b1, b2 = b0/a0, b1/a0, b2/a0
            a1, a2 = a1/a0, a2/a0

            # Direct Form II implementation
            # w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
            w = signal[i] - a1*state[0] - a2*state[1]

            # y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
            output[i] = b0*w + b1*state[0] + b2*state[1]

            # Update state: shift w into state buffer
            state[1] = state[0]
            state[0] = w

        return output, state

    @staticmethod
    def _apply_time_varying_highpass(
        signal: np.ndarray,
        cutoff_array: np.ndarray,
        q: float,
        sample_rate: int,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply biquad highpass filter with time-varying cutoff.

        This implements a sample-by-sample biquad filter where the cutoff
        frequency changes over time. Filter coefficients are recomputed
        for each sample, and the filter state is maintained for continuity.

        Args:
            signal: Input signal array
            cutoff_array: Array of cutoff frequencies (one per sample) in Hz
            q: Quality factor (resonance)
            sample_rate: Sample rate in Hz
            initial_state: Optional initial state [z1, z2] for continuity across hops

        Returns:
            Tuple of (filtered signal array, final state [z1, z2])

        Implementation notes:
            - Uses Direct Form II for state continuity
            - Coefficients recomputed per-sample for smooth modulation
            - State vector preserved across coefficient changes
            - Normalized coefficients to avoid numerical issues
            - State can be preserved across buffer hops for seamless continuation
        """
        output = np.zeros_like(signal)
        # Biquad state (2 elements for 2nd-order filter)
        state = initial_state.copy() if initial_state is not None else np.zeros(2)

        for i in range(len(signal)):
            # Compute filter coefficients for current cutoff
            cutoff = cutoff_array[i]
            w0 = 2.0 * np.pi * cutoff / sample_rate
            alpha = np.sin(w0) / (2.0 * q)

            # Biquad highpass coefficients
            b0 = (1.0 + np.cos(w0)) / 2.0
            b1 = -(1.0 + np.cos(w0))
            b2 = (1.0 + np.cos(w0)) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * np.cos(w0)
            a2 = 1.0 - alpha

            # Normalize by a0
            b0, b1, b2 = b0/a0, b1/a0, b2/a0
            a1, a2 = a1/a0, a2/a0

            # Direct Form II implementation
            # w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
            w = signal[i] - a1*state[0] - a2*state[1]

            # y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
            output[i] = b0*w + b1*state[0] + b2*state[1]

            # Update state: shift w into state buffer
            state[1] = state[0]
            state[0] = w

        return output, state

    @staticmethod
    def _apply_time_varying_bandpass(
        signal: np.ndarray,
        cutoff_array: np.ndarray,
        q: float,
        sample_rate: int,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply biquad bandpass filter with time-varying center frequency.

        This implements a sample-by-sample biquad filter where the center
        frequency changes over time. Filter coefficients are recomputed
        for each sample, and the filter state is maintained for continuity.

        Args:
            signal: Input signal array
            cutoff_array: Array of center frequencies (one per sample) in Hz
            q: Quality factor (bandwidth control, higher Q = narrower band)
            sample_rate: Sample rate in Hz
            initial_state: Optional initial state [z1, z2] for continuity across hops

        Returns:
            Tuple of (filtered signal array, final state [z1, z2])

        Implementation notes:
            - Uses Direct Form II for state continuity
            - Coefficients recomputed per-sample for smooth modulation
            - State vector preserved across coefficient changes
            - Normalized coefficients to avoid numerical issues
            - Q controls bandwidth: Q=1 is wide, Q=10 is narrow
            - State can be preserved across buffer hops for seamless continuation
        """
        output = np.zeros_like(signal)
        # Biquad state (2 elements for 2nd-order filter)
        state = initial_state.copy() if initial_state is not None else np.zeros(2)

        for i in range(len(signal)):
            # Compute filter coefficients for current center frequency
            cutoff = cutoff_array[i]
            w0 = 2.0 * np.pi * cutoff / sample_rate
            alpha = np.sin(w0) / (2.0 * q)

            # Biquad bandpass coefficients (constant 0 dB peak gain)
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * np.cos(w0)
            a2 = 1.0 - alpha

            # Normalize by a0
            b0, b1, b2 = b0/a0, b1/a0, b2/a0
            a1, a2 = a1/a0, a2/a0

            # Direct Form II implementation
            # w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
            w = signal[i] - a1*state[0] - a2*state[1]

            # y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
            output[i] = b0*w + b1*state[0] + b2*state[1]

            # Update state: shift w into state buffer
            state[1] = state[0]
            state[0] = w

        return output, state

    # ========================================================================
    # AUDIO I/O (v0.6.0)
    # ========================================================================

    @staticmethod
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

        Example:
            # Generate and play a tone
            tone = audio.sine(freq=440.0, duration=1.0)
            audio.play(tone)
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
        # sounddevice expects shape (samples, channels) for stereo
        if buffer.is_stereo:
            data = buffer.data  # Already (samples, 2)
        else:
            data = buffer.data.reshape(-1, 1)  # Make (samples, 1) for mono

        # Play audio
        sd.play(data, samplerate=buffer.sample_rate, blocking=blocking)

    @staticmethod
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

        Example:
            # Generate and save audio
            tone = audio.sine(freq=440.0, duration=1.0)
            audio.save(tone, "output.wav")
            audio.save(tone, "output.flac")
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
                format = "wav"  # Default to WAV

        format = format.lower()

        # Prepare data
        # Ensure data is in the correct format (float32, clipped to [-1, 1])
        data = np.clip(buffer.data, -1.0, 1.0).astype(np.float32)

        if format == "flac":
            # FLAC requires soundfile
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError(
                    "soundfile is required for FLAC export. "
                    "Install with: pip install soundfile"
                )

            # soundfile expects (samples, channels) for stereo
            if buffer.is_stereo:
                sf.write(path, data, buffer.sample_rate, format='FLAC')
            else:
                sf.write(path, data.reshape(-1, 1), buffer.sample_rate, format='FLAC')

        elif format == "wav":
            # Try soundfile first (better quality), fall back to scipy
            try:
                import soundfile as sf
                if buffer.is_stereo:
                    sf.write(path, data, buffer.sample_rate, format='WAV')
                else:
                    sf.write(path, data.reshape(-1, 1), buffer.sample_rate, format='WAV')
            except ImportError:
                # Fall back to scipy.io.wavfile
                try:
                    from scipy.io import wavfile
                except ImportError:
                    raise ImportError(
                        "Either soundfile or scipy is required for WAV export. "
                        "Install with: pip install soundfile  OR  pip install scipy"
                    )

                # scipy.io.wavfile expects int16 format
                # Convert float32 [-1, 1] to int16 [-32768, 32767]
                data_int16 = (data * 32767).astype(np.int16)
                wavfile.write(path, buffer.sample_rate, data_int16)

        else:
            raise ValueError(
                f"Unsupported format: {format}. Supported: 'wav', 'flac'"
            )

        print(f"Saved audio to: {path}")

    @staticmethod
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

        Example:
            # Load audio file
            loaded = audio.load("input.wav")
            print(f"Loaded {loaded.duration:.2f}s of audio")
        """
        try:
            import soundfile as sf
        except ImportError:
            # Try scipy fallback for WAV
            if path.endswith('.wav'):
                try:
                    from scipy.io import wavfile
                    sample_rate, data = wavfile.read(path)

                    # Convert to float32 [-1, 1]
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.int32:
                        data = data.astype(np.float32) / 2147483648.0
                    elif data.dtype == np.uint8:
                        data = (data.astype(np.float32) - 128.0) / 128.0

                    # Handle stereo: scipy returns (samples, channels)
                    # We need to keep it that way
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

        # Load with soundfile
        data, sample_rate = sf.read(path, dtype='float32')

        # soundfile returns (samples,) for mono, (samples, channels) for stereo
        # This matches our AudioBuffer expectations
        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
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

        Example:
            # Record 3 seconds from microphone
            recording = audio.record(duration=3.0)
            audio.save(recording, "recording.wav")
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

        # Record audio
        data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete

        print("Recording complete!")

        # sounddevice returns (samples, channels) even for mono
        # For mono, we want (samples,) to match our convention
        if channels == 1:
            data = data.reshape(-1)

        return AudioBuffer(data=data, sample_rate=sample_rate)

    # ========================================================================
    # BUFFER OPERATIONS (Section 5.6)
    # ========================================================================

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

        Example:
            # Smooth fade-in
            faded = audio.fade_in(signal, duration=0.1)
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

    @staticmethod
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

        Example:
            # Smooth fade-out
            faded = audio.fade_out(signal, duration=0.1)
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

    # ========================================================================
    # FFT / SPECTRAL TRANSFORMS (Section 5.7)
    # ========================================================================

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(signal: AudioBuffer) -> Tuple[ndarray, ndarray]",
        deterministic=True,
        doc="Compute FFT of audio buffer"
    )
    def fft(signal: AudioBuffer) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fast Fourier Transform.

        Args:
            signal: Input audio buffer (must be mono)

        Returns:
            Tuple of (frequencies, complex_spectrum)
            - frequencies: Frequency bins in Hz
            - complex_spectrum: Complex FFT coefficients

        Example:
            # Analyze frequency content
            freqs, spectrum = audio.fft(signal)
            magnitude = np.abs(spectrum)
        """
        if signal.is_stereo:
            raise ValueError("FFT requires mono signal (use left channel)")

        # Compute FFT
        spectrum = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(signal.num_samples, 1.0 / signal.sample_rate)

        return freqs, spectrum

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(spectrum: ndarray, sample_rate: int) -> AudioBuffer",
        deterministic=True,
        doc="Compute inverse FFT to audio buffer"
    )
    def ifft(spectrum: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Compute Inverse Fast Fourier Transform.

        Args:
            spectrum: Complex FFT coefficients (from rfft)
            sample_rate: Sample rate in Hz

        Returns:
            Audio buffer reconstructed from spectrum

        Example:
            # Modify and reconstruct
            freqs, spectrum = audio.fft(signal)
            # ... modify spectrum ...
            reconstructed = audio.ifft(spectrum, signal.sample_rate)
        """
        data = np.fft.irfft(spectrum)
        return AudioBuffer(data=data.real, sample_rate=sample_rate)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(signal: AudioBuffer, window_size: int, hop_size: Optional[int], window: str) -> ndarray",
        deterministic=True,
        doc="Compute Short-Time Fourier Transform"
    )
    def stft(signal: AudioBuffer, window_size: int = 2048,
             hop_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Short-Time Fourier Transform.

        Args:
            signal: Input audio buffer (must be mono)
            window_size: FFT window size in samples
            hop_size: Hop size in samples

        Returns:
            Tuple of (times, frequencies, stft_matrix)
            - times: Time bins in seconds
            - frequencies: Frequency bins in Hz
            - stft_matrix: Complex STFT matrix (freq × time)

        Example:
            # Compute spectrogram
            times, freqs, stft_data = audio.stft(signal)
            spectrogram = np.abs(stft_data)
        """
        if signal.is_stereo:
            raise ValueError("STFT requires mono signal")

        # Hann window
        window = np.hanning(window_size)

        # Calculate number of frames
        num_frames = 1 + (signal.num_samples - window_size) // hop_size

        # Allocate STFT matrix
        num_freqs = window_size // 2 + 1
        stft_matrix = np.zeros((num_freqs, num_frames), dtype=np.complex128)

        # Compute STFT
        for i in range(num_frames):
            start = i * hop_size
            frame = signal.data[start:start + window_size]

            # Apply window
            windowed = frame * window

            # FFT
            stft_matrix[:, i] = np.fft.rfft(windowed)

        # Time and frequency axes
        times = np.arange(num_frames) * hop_size / signal.sample_rate
        freqs = np.fft.rfftfreq(window_size, 1.0 / signal.sample_rate)

        return times, freqs, stft_matrix

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(stft_matrix: ndarray, hop_size: int, window_size: int) -> AudioBuffer",
        deterministic=True,
        doc="Compute inverse STFT to audio buffer"
    )
    def istft(stft_matrix: np.ndarray, hop_size: int = 512,
              sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Compute Inverse Short-Time Fourier Transform.

        Args:
            stft_matrix: Complex STFT matrix (freq × time)
            hop_size: Hop size in samples
            sample_rate: Sample rate in Hz

        Returns:
            Reconstructed audio buffer

        Example:
            # Modify and reconstruct
            times, freqs, stft_data = audio.stft(signal)
            # ... modify stft_data ...
            reconstructed = audio.istft(stft_data, hop_size=512)
        """
        num_freqs, num_frames = stft_matrix.shape
        window_size = (num_freqs - 1) * 2

        # Hann window
        window = np.hanning(window_size)

        # Output length
        output_length = (num_frames - 1) * hop_size + window_size
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)

        # Overlap-add
        for i in range(num_frames):
            start = i * hop_size

            # Inverse FFT
            frame = np.fft.irfft(stft_matrix[:, i])

            # Apply window and accumulate
            output[start:start + window_size] += frame * window
            window_sum[start:start + window_size] += window ** 2

        # Normalize by window overlap
        # Avoid division by zero
        window_sum[window_sum < 1e-10] = 1.0
        output = output / window_sum

        return AudioBuffer(data=output, sample_rate=sample_rate)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer) -> Tuple[ndarray, ndarray]",
        deterministic=True,
        doc="Compute magnitude spectrum"
    )
    def spectrum(signal: AudioBuffer) -> Tuple[np.ndarray, np.ndarray]:
        """Get magnitude spectrum.

        Args:
            signal: Input audio buffer (must be mono)

        Returns:
            Tuple of (frequencies, magnitudes)

        Example:
            # Get magnitude spectrum
            freqs, mags = audio.spectrum(signal)
        """
        freqs, complex_spec = AudioOperations.fft(signal)
        magnitudes = np.abs(complex_spec)
        return freqs, magnitudes

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer) -> Tuple[ndarray, ndarray]",
        deterministic=True,
        doc="Compute phase spectrum"
    )
    def phase_spectrum(signal: AudioBuffer) -> Tuple[np.ndarray, np.ndarray]:
        """Get phase spectrum.

        Args:
            signal: Input audio buffer (must be mono)

        Returns:
            Tuple of (frequencies, phases)

        Example:
            # Get phase spectrum
            freqs, phases = audio.phase_spectrum(signal)
        """
        freqs, complex_spec = AudioOperations.fft(signal)
        phases = np.angle(complex_spec)
        return freqs, phases

    # ========================================================================
    # SPECTRAL ANALYSIS (Section 5.8)
    # ========================================================================

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer) -> float",
        deterministic=True,
        doc="Compute spectral centroid"
    )
    def spectral_centroid(signal: AudioBuffer) -> float:
        """Calculate spectral centroid (brightness measure).

        The spectral centroid is the center of mass of the spectrum,
        indicating where the "center" of the sound's frequency content is.

        Args:
            signal: Input audio buffer (must be mono)

        Returns:
            Spectral centroid in Hz

        Example:
            # Measure brightness
            brightness = audio.spectral_centroid(signal)
            print(f"Spectral centroid: {brightness:.1f} Hz")
        """
        freqs, magnitudes = AudioOperations.spectrum(signal)

        # Weighted average of frequencies by magnitude
        centroid = np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-10)
        return float(centroid)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer, threshold: float) -> float",
        deterministic=True,
        doc="Compute spectral rolloff frequency"
    )
    def spectral_rolloff(signal: AudioBuffer, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency.

        The frequency below which the given threshold of spectral energy is contained.

        Args:
            signal: Input audio buffer (must be mono)
            threshold: Energy threshold (0.0 to 1.0, default 0.85 = 85%)

        Returns:
            Rolloff frequency in Hz

        Example:
            # Find high-frequency content
            rolloff = audio.spectral_rolloff(signal, threshold=0.85)
        """
        freqs, magnitudes = AudioOperations.spectrum(signal)

        # Calculate cumulative energy
        energy = magnitudes ** 2
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1]

        # Find rolloff point
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]

        if len(rolloff_idx) > 0:
            return float(freqs[rolloff_idx[0]])
        else:
            return float(freqs[-1])

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer, hop_size: int) -> ndarray",
        deterministic=True,
        doc="Compute spectral flux over time"
    )
    def spectral_flux(signal: AudioBuffer, hop_size: int = 512) -> np.ndarray:
        """Calculate spectral flux (change in spectrum over time).

        Spectral flux measures how quickly the spectrum is changing,
        useful for onset detection and rhythm analysis.

        Args:
            signal: Input audio buffer (must be mono)
            hop_size: Hop size in samples

        Returns:
            Array of spectral flux values over time

        Example:
            # Detect onsets
            flux = audio.spectral_flux(signal)
            onsets = np.where(flux > np.mean(flux) * 3)[0]
        """
        times, freqs, stft_matrix = AudioOperations.stft(signal, hop_size=hop_size)

        # Calculate magnitude spectrum
        mag_spectrum = np.abs(stft_matrix)

        # Spectral flux = sum of squared differences between adjacent frames
        flux = np.zeros(mag_spectrum.shape[1])
        for i in range(1, mag_spectrum.shape[1]):
            diff = mag_spectrum[:, i] - mag_spectrum[:, i-1]
            # Only positive changes (rectified)
            diff = np.maximum(0, diff)
            flux[i] = np.sum(diff ** 2)

        return flux

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer, num_peaks: int, threshold_db: float) -> Tuple[ndarray, ndarray]",
        deterministic=True,
        doc="Find spectral peaks"
    )
    def spectral_peaks(signal: AudioBuffer, num_peaks: int = 5,
                       min_freq: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """Find spectral peaks (dominant frequencies).

        Args:
            signal: Input audio buffer (must be mono)
            num_peaks: Number of peaks to return
            min_freq: Minimum frequency in Hz

        Returns:
            Tuple of (peak_frequencies, peak_magnitudes)

        Example:
            # Find dominant frequencies
            peak_freqs, peak_mags = audio.spectral_peaks(signal, num_peaks=5)
            print(f"Strongest frequency: {peak_freqs[0]:.1f} Hz")
        """
        freqs, magnitudes = AudioOperations.spectrum(signal)

        # Filter by minimum frequency
        valid_idx = freqs >= min_freq
        freqs = freqs[valid_idx]
        magnitudes = magnitudes[valid_idx]

        # Find peaks using simple maximum finding
        # Look for local maxima
        peaks_idx = []
        for i in range(1, len(magnitudes) - 1):
            if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                peaks_idx.append(i)

        if len(peaks_idx) == 0:
            # No peaks found, return highest magnitudes
            peaks_idx = np.argsort(magnitudes)[::-1][:num_peaks]
        else:
            # Sort peaks by magnitude
            peaks_idx = np.array(peaks_idx)
            sorted_idx = np.argsort(magnitudes[peaks_idx])[::-1]
            peaks_idx = peaks_idx[sorted_idx][:num_peaks]

        peak_freqs = freqs[peaks_idx]
        peak_mags = magnitudes[peaks_idx]

        return peak_freqs, peak_mags

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer) -> float",
        deterministic=True,
        doc="Compute RMS level"
    )
    def rms(signal: AudioBuffer) -> float:
        """Calculate RMS (Root Mean Square) level.

        RMS is a measure of the average power/loudness of the signal.

        Args:
            signal: Input audio buffer

        Returns:
            RMS value (0.0 to 1.0)

        Example:
            # Measure loudness
            loudness = audio.rms(signal)
            loudness_db = audio.lin2db(loudness)
        """
        if signal.is_stereo:
            # Average both channels
            rms_val = np.sqrt(np.mean(signal.data ** 2))
        else:
            rms_val = np.sqrt(np.mean(signal.data ** 2))

        return float(rms_val)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.QUERY,
        signature="(signal: AudioBuffer) -> int",
        deterministic=True,
        doc="Count zero crossings"
    )
    def zero_crossings(signal: AudioBuffer) -> int:
        """Count zero crossings (sign changes).

        Zero crossing rate is correlated with the noisiness/pitch of a signal.

        Args:
            signal: Input audio buffer (must be mono)

        Returns:
            Number of zero crossings

        Example:
            # Measure noisiness
            zcr = audio.zero_crossings(signal)
            zcr_rate = zcr / signal.duration
        """
        if signal.is_stereo:
            raise ValueError("Zero crossings requires mono signal")

        # Count sign changes
        signs = np.sign(signal.data)
        crossings = np.sum(np.abs(np.diff(signs))) / 2

        return int(crossings)

    # ========================================================================
    # SPECTRAL PROCESSING (Section 5.9)
    # ========================================================================

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(signal: AudioBuffer, threshold_db: float, attack: float, release: float) -> AudioBuffer",
        deterministic=True,
        doc="Apply spectral noise gate"
    )
    def spectral_gate(signal: AudioBuffer, threshold_db: float = -40.0,
                      window_size: int = 2048, hop_size: int = 512) -> AudioBuffer:
        """Apply spectral noise gate.

        Removes frequency bins below threshold, useful for noise reduction.

        Args:
            signal: Input audio buffer (must be mono)
            threshold_db: Threshold in dB
            window_size: FFT window size
            hop_size: Hop size in samples

        Returns:
            Noise-gated audio buffer

        Example:
            # Remove quiet frequencies
            cleaned = audio.spectral_gate(signal, threshold_db=-40.0)
        """
        if signal.is_stereo:
            raise ValueError("Spectral gate requires mono signal")

        # Compute STFT
        times, freqs, stft_matrix = AudioOperations.stft(signal, window_size, hop_size)

        # Convert threshold to linear
        threshold_lin = AudioOperations.db2lin(threshold_db)

        # Get magnitude and phase
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # Apply gate
        mask = magnitude > threshold_lin
        magnitude_gated = magnitude * mask

        # Reconstruct complex spectrum
        stft_gated = magnitude_gated * np.exp(1j * phase)

        # Inverse STFT
        return AudioOperations.istft(stft_gated, hop_size, signal.sample_rate)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(signal: AudioBuffer, freq_mask: ndarray) -> AudioBuffer",
        deterministic=True,
        doc="Apply spectral filtering with frequency mask"
    )
    def spectral_filter(signal: AudioBuffer, freq_mask: np.ndarray) -> AudioBuffer:
        """Apply arbitrary frequency-domain filter.

        Args:
            signal: Input audio buffer (must be mono)
            freq_mask: Frequency mask (same length as FFT bins)

        Returns:
            Filtered audio buffer

        Example:
            # Custom spectral filter
            freqs, _ = audio.fft(signal)
            mask = (freqs > 200) & (freqs < 2000)  # Bandpass
            filtered = audio.spectral_filter(signal, mask.astype(float))
        """
        if signal.is_stereo:
            raise ValueError("Spectral filter requires mono signal")

        # Compute FFT
        freqs, spectrum = AudioOperations.fft(signal)

        # Apply mask
        spectrum_filtered = spectrum * freq_mask

        # Inverse FFT
        return AudioOperations.ifft(spectrum_filtered, signal.sample_rate)

    @staticmethod
    @operator(
        domain="audio",
        category=OpCategory.TRANSFORM,
        signature="(signal: AudioBuffer, impulse: AudioBuffer) -> AudioBuffer",
        deterministic=True,
        doc="Convolve signal with impulse response"
    )
    def convolution(signal: AudioBuffer, impulse: AudioBuffer) -> AudioBuffer:
        """Apply convolution (for reverb, filtering, etc.).

        Convolution in time domain = multiplication in frequency domain.
        Useful for convolution reverb with impulse responses.

        Args:
            signal: Input audio buffer (must be mono)
            impulse: Impulse response (must be mono)

        Returns:
            Convolved audio buffer

        Example:
            # Convolution reverb
            # impulse_response = audio.load("hall_ir.wav")
            # reverb_sound = audio.convolution(dry_signal, impulse_response)
        """
        if signal.is_stereo or impulse.is_stereo:
            raise ValueError("Convolution requires mono signals")

        # FFT-based convolution (faster for long impulses)
        conv_length = signal.num_samples + impulse.num_samples - 1

        # Pad to power of 2 for efficiency
        fft_length = 2 ** int(np.ceil(np.log2(conv_length)))

        # FFT
        signal_fft = np.fft.rfft(signal.data, n=fft_length)
        impulse_fft = np.fft.rfft(impulse.data, n=fft_length)

        # Multiply in frequency domain
        result_fft = signal_fft * impulse_fft

        # Inverse FFT
        result = np.fft.irfft(result_fft)

        # Trim to correct length
        result = result[:conv_length]

        # Normalize
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak

        return AudioBuffer(data=result, sample_rate=signal.sample_rate)


# Create singleton instance for use as 'audio' namespace
audio = AudioOperations()

# Export operators for domain registry discovery
sine = AudioOperations.sine
saw = AudioOperations.saw
square = AudioOperations.square
triangle = AudioOperations.triangle
noise = AudioOperations.noise
impulse = AudioOperations.impulse
constant = AudioOperations.constant
lowpass = AudioOperations.lowpass
highpass = AudioOperations.highpass
bandpass = AudioOperations.bandpass
notch = AudioOperations.notch
vcf_lowpass = AudioOperations.vcf_lowpass
vcf_highpass = AudioOperations.vcf_highpass
vcf_bandpass = AudioOperations.vcf_bandpass
eq3 = AudioOperations.eq3
adsr = AudioOperations.adsr
ar = AudioOperations.ar
envexp = AudioOperations.envexp
delay = AudioOperations.delay
reverb = AudioOperations.reverb
chorus = AudioOperations.chorus
flanger = AudioOperations.flanger
drive = AudioOperations.drive
limiter = AudioOperations.limiter
mix = AudioOperations.mix
gain = AudioOperations.gain
multiply = AudioOperations.multiply
vca = AudioOperations.vca
pan = AudioOperations.pan
clip = AudioOperations.clip
normalize = AudioOperations.normalize
db2lin = AudioOperations.db2lin
lin2db = AudioOperations.lin2db
string = AudioOperations.string
modal = AudioOperations.modal
play = AudioOperations.play
save = AudioOperations.save
load = AudioOperations.load
record = AudioOperations.record
slice = AudioOperations.slice
concat = AudioOperations.concat
resample = AudioOperations.resample
reverse = AudioOperations.reverse
fade_in = AudioOperations.fade_in
fade_out = AudioOperations.fade_out
fft = AudioOperations.fft
ifft = AudioOperations.ifft
stft = AudioOperations.stft
istft = AudioOperations.istft
spectrum = AudioOperations.spectrum
phase_spectrum = AudioOperations.phase_spectrum
spectral_centroid = AudioOperations.spectral_centroid
spectral_rolloff = AudioOperations.spectral_rolloff
spectral_flux = AudioOperations.spectral_flux
spectral_peaks = AudioOperations.spectral_peaks
rms = AudioOperations.rms
zero_crossings = AudioOperations.zero_crossings
spectral_gate = AudioOperations.spectral_gate
spectral_filter = AudioOperations.spectral_filter
convolution = AudioOperations.convolution
