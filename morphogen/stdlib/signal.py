"""
Signal Processing Domain

Provides signal analysis and transformation operations:
- Fourier transforms (FFT, STFT, inverse FFT)
- Wavelet transforms
- Filtering (FIR, IIR)
- Spectral analysis
- Time-frequency representations
- Signal generation and windowing

Follows Morphogen's immutability pattern: all operations return new instances.

Version: v0.10.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from enum import Enum
import scipy.signal as signal
import scipy.fft as fft

from morphogen.core.operator import operator, OpCategory


class WindowType(Enum):
    """Window function types"""
    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    BARTLETT = "bartlett"
    KAISER = "kaiser"


class WaveletType(Enum):
    """Wavelet types"""
    MORLET = "morlet"
    RICKER = "ricker"
    GAUSSIAN = "gaussian"


@dataclass
class Signal1D:
    """1D signal data structure

    Attributes:
        data: Signal samples (1D array)
        sample_rate: Samples per second (Hz)
        time_offset: Start time in seconds (default 0.0)
    """
    data: np.ndarray
    sample_rate: float
    time_offset: float = 0.0

    def copy(self) -> 'Signal1D':
        """Create a deep copy"""
        return Signal1D(
            data=self.data.copy(),
            sample_rate=self.sample_rate,
            time_offset=self.time_offset
        )

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return len(self.data) / self.sample_rate

    @property
    def time_axis(self) -> np.ndarray:
        """Time axis in seconds"""
        return np.arange(len(self.data)) / self.sample_rate + self.time_offset


@dataclass
class Spectrum:
    """Frequency domain representation

    Attributes:
        data: Complex frequency bins
        frequencies: Frequency values (Hz)
        sample_rate: Original signal sample rate
    """
    data: np.ndarray
    frequencies: np.ndarray
    sample_rate: float

    def copy(self) -> 'Spectrum':
        """Create a deep copy"""
        return Spectrum(
            data=self.data.copy(),
            frequencies=self.frequencies.copy(),
            sample_rate=self.sample_rate
        )

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude spectrum"""
        return np.abs(self.data)

    @property
    def phase(self) -> np.ndarray:
        """Phase spectrum in radians"""
        return np.angle(self.data)

    @property
    def power(self) -> np.ndarray:
        """Power spectrum"""
        return np.abs(self.data) ** 2


@dataclass
class Spectrogram:
    """Time-frequency representation

    Attributes:
        data: 2D array (time x frequency)
        times: Time values
        frequencies: Frequency values
        sample_rate: Original signal sample rate
    """
    data: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    sample_rate: float

    def copy(self) -> 'Spectrogram':
        """Create a deep copy"""
        return Spectrogram(
            data=self.data.copy(),
            times=self.times.copy(),
            frequencies=self.frequencies.copy(),
            sample_rate=self.sample_rate
        )

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude spectrogram"""
        return np.abs(self.data)

    @property
    def power(self) -> np.ndarray:
        """Power spectrogram (dB scale)"""
        power_linear = np.abs(self.data) ** 2
        return 10 * np.log10(power_linear + 1e-10)


class SignalOperations:
    """Signal processing operations"""

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.CONSTRUCT,
        signature="(data: ndarray, sample_rate: float, time_offset: float) -> Signal1D",
        deterministic=True,
        doc="Create a signal from data array"
    )
    def create_signal(data: np.ndarray, sample_rate: float,
                     time_offset: float = 0.0) -> Signal1D:
        """Create a signal from data array

        Args:
            data: Signal samples
            sample_rate: Samples per second
            time_offset: Start time in seconds

        Returns:
            Signal1D instance
        """
        return Signal1D(
            data=np.asarray(data, dtype=np.float64),
            sample_rate=float(sample_rate),
            time_offset=float(time_offset)
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.CONSTRUCT,
        signature="(frequency: float, duration: float, sample_rate: float, amplitude: float, phase: float) -> Signal1D",
        deterministic=True,
        doc="Generate sine wave"
    )
    def sine_wave(frequency: float, duration: float, sample_rate: float,
                 amplitude: float = 1.0, phase: float = 0.0) -> Signal1D:
        """Generate sine wave

        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Samples per second
            amplitude: Wave amplitude
            phase: Initial phase in radians

        Returns:
            Sine wave signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        data = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return Signal1D(data=data, sample_rate=sample_rate)

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.CONSTRUCT,
        signature="(f0: float, f1: float, duration: float, sample_rate: float, method: str) -> Signal1D",
        deterministic=True,
        doc="Generate chirp signal (frequency sweep)"
    )
    def chirp(f0: float, f1: float, duration: float, sample_rate: float,
             method: str = 'linear') -> Signal1D:
        """Generate chirp signal (frequency sweep)

        Args:
            f0: Starting frequency (Hz)
            f1: Ending frequency (Hz)
            duration: Duration in seconds
            sample_rate: Samples per second
            method: 'linear', 'quadratic', 'logarithmic', or 'hyperbolic'

        Returns:
            Chirp signal
        """
        t = np.arange(0, duration, 1.0 / sample_rate)
        data = signal.chirp(t, f0, duration, f1, method=method)
        return Signal1D(data=data, sample_rate=sample_rate)

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.CONSTRUCT,
        signature="(duration: float, sample_rate: float, amplitude: float, seed: Optional[int]) -> Signal1D",
        deterministic=False,
        doc="Generate white noise"
    )
    def white_noise(duration: float, sample_rate: float,
                   amplitude: float = 1.0, seed: Optional[int] = None) -> Signal1D:
        """Generate white noise

        Args:
            duration: Duration in seconds
            sample_rate: Samples per second
            amplitude: Noise amplitude
            seed: Random seed

        Returns:
            White noise signal
        """
        if seed is not None:
            np.random.seed(seed)

        num_samples = int(duration * sample_rate)
        data = amplitude * np.random.randn(num_samples)
        return Signal1D(data=data, sample_rate=sample_rate)

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, window_type: WindowType, beta: float) -> Signal1D",
        deterministic=True,
        doc="Apply window function to signal"
    )
    def window(sig: Signal1D, window_type: WindowType,
              beta: float = 8.6) -> Signal1D:
        """Apply window function to signal

        Args:
            sig: Input signal
            window_type: Window type
            beta: Kaiser window beta parameter (if using Kaiser window)

        Returns:
            Windowed signal
        """
        n = len(sig.data)

        if window_type == WindowType.RECTANGULAR:
            win = np.ones(n)
        elif window_type == WindowType.HANN:
            win = np.hanning(n)
        elif window_type == WindowType.HAMMING:
            win = np.hamming(n)
        elif window_type == WindowType.BLACKMAN:
            win = np.blackman(n)
        elif window_type == WindowType.BARTLETT:
            win = np.bartlett(n)
        elif window_type == WindowType.KAISER:
            win = np.kaiser(n, beta)
        else:
            win = np.ones(n)

        result_data = sig.data * win
        return Signal1D(
            data=result_data,
            sample_rate=sig.sample_rate,
            time_offset=sig.time_offset
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D) -> Spectrum",
        deterministic=True,
        doc="Compute Fast Fourier Transform"
    )
    def fft(sig: Signal1D) -> Spectrum:
        """Compute Fast Fourier Transform

        Args:
            sig: Input signal

        Returns:
            Frequency domain representation
        """
        # Compute FFT
        fft_data = fft.fft(sig.data)

        # Compute frequency bins
        n = len(sig.data)
        frequencies = fft.fftfreq(n, 1.0 / sig.sample_rate)

        return Spectrum(
            data=fft_data,
            frequencies=frequencies,
            sample_rate=sig.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(spectrum: Spectrum) -> Signal1D",
        deterministic=True,
        doc="Compute Inverse Fast Fourier Transform"
    )
    def ifft(spectrum: Spectrum) -> Signal1D:
        """Compute Inverse Fast Fourier Transform

        Args:
            spectrum: Frequency domain representation

        Returns:
            Time domain signal
        """
        # Compute inverse FFT
        time_data = fft.ifft(spectrum.data)

        # Take real part (imaginary should be ~0 for real signals)
        result_data = np.real(time_data)

        return Signal1D(
            data=result_data,
            sample_rate=spectrum.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D) -> Spectrum",
        deterministic=True,
        doc="Compute Real FFT (only positive frequencies)"
    )
    def rfft(sig: Signal1D) -> Spectrum:
        """Compute Real FFT (only positive frequencies)

        Args:
            sig: Input signal

        Returns:
            Spectrum with only positive frequencies
        """
        # Compute real FFT
        rfft_data = fft.rfft(sig.data)

        # Compute frequency bins
        n = len(sig.data)
        frequencies = fft.rfftfreq(n, 1.0 / sig.sample_rate)

        return Spectrum(
            data=rfft_data,
            frequencies=frequencies,
            sample_rate=sig.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, window_size: int, hop_size: Optional[int], window_type: WindowType) -> Spectrogram",
        deterministic=True,
        doc="Compute Short-Time Fourier Transform"
    )
    def stft(sig: Signal1D, window_size: int = 256,
            hop_size: Optional[int] = None,
            window_type: WindowType = WindowType.HANN) -> Spectrogram:
        """Compute Short-Time Fourier Transform

        Args:
            sig: Input signal
            window_size: Window size in samples
            hop_size: Hop size in samples (default: window_size // 4)
            window_type: Window function type

        Returns:
            Time-frequency representation
        """
        if hop_size is None:
            hop_size = window_size // 4

        # Get window
        if window_type == WindowType.HANN:
            win = signal.windows.hann(window_size, sym=False)
        elif window_type == WindowType.HAMMING:
            win = signal.windows.hamming(window_size, sym=False)
        elif window_type == WindowType.BLACKMAN:
            win = signal.windows.blackman(window_size, sym=False)
        else:
            win = np.ones(window_size)

        # Compute STFT
        frequencies, times, stft_data = signal.stft(
            sig.data,
            fs=sig.sample_rate,
            window=win,
            nperseg=window_size,
            noverlap=window_size - hop_size
        )

        return Spectrogram(
            data=stft_data.T,  # Transpose to (time x frequency)
            times=times + sig.time_offset,
            frequencies=frequencies,
            sample_rate=sig.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(spec: Spectrogram, window_size: int, hop_size: Optional[int], window_type: WindowType) -> Signal1D",
        deterministic=True,
        doc="Compute Inverse Short-Time Fourier Transform"
    )
    def istft(spec: Spectrogram, window_size: int = 256,
             hop_size: Optional[int] = None,
             window_type: WindowType = WindowType.HANN) -> Signal1D:
        """Compute Inverse Short-Time Fourier Transform

        Args:
            spec: Input spectrogram
            window_size: Window size in samples
            hop_size: Hop size in samples
            window_type: Window function type

        Returns:
            Reconstructed signal
        """
        if hop_size is None:
            hop_size = window_size // 4

        # Get window
        if window_type == WindowType.HANN:
            win = signal.windows.hann(window_size, sym=False)
        elif window_type == WindowType.HAMMING:
            win = signal.windows.hamming(window_size, sym=False)
        elif window_type == WindowType.BLACKMAN:
            win = signal.windows.blackman(window_size, sym=False)
        else:
            win = np.ones(window_size)

        # Compute inverse STFT
        _, reconstructed = signal.istft(
            spec.data.T,  # Transpose back to (frequency x time)
            fs=spec.sample_rate,
            window=win,
            nperseg=window_size,
            noverlap=window_size - hop_size
        )

        return Signal1D(
            data=reconstructed,
            sample_rate=spec.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, cutoff: float, order: int) -> Signal1D",
        deterministic=True,
        doc="Apply lowpass filter"
    )
    def lowpass(sig: Signal1D, cutoff: float, order: int = 5) -> Signal1D:
        """Apply lowpass filter

        Args:
            sig: Input signal
            cutoff: Cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = sig.sample_rate / 2.0
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
        filtered_data = signal.filtfilt(b, a, sig.data)

        return Signal1D(
            data=filtered_data,
            sample_rate=sig.sample_rate,
            time_offset=sig.time_offset
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, cutoff: float, order: int) -> Signal1D",
        deterministic=True,
        doc="Apply highpass filter"
    )
    def highpass(sig: Signal1D, cutoff: float, order: int = 5) -> Signal1D:
        """Apply highpass filter

        Args:
            sig: Input signal
            cutoff: Cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = sig.sample_rate / 2.0
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high')
        filtered_data = signal.filtfilt(b, a, sig.data)

        return Signal1D(
            data=filtered_data,
            sample_rate=sig.sample_rate,
            time_offset=sig.time_offset
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, low: float, high: float, order: int) -> Signal1D",
        deterministic=True,
        doc="Apply bandpass filter"
    )
    def bandpass(sig: Signal1D, low: float, high: float, order: int = 5) -> Signal1D:
        """Apply bandpass filter

        Args:
            sig: Input signal
            low: Low cutoff frequency (Hz)
            high: High cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = sig.sample_rate / 2.0
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered_data = signal.filtfilt(b, a, sig.data)

        return Signal1D(
            data=filtered_data,
            sample_rate=sig.sample_rate,
            time_offset=sig.time_offset
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, new_sample_rate: float) -> Signal1D",
        deterministic=True,
        doc="Resample signal to new sample rate"
    )
    def resample(sig: Signal1D, new_sample_rate: float) -> Signal1D:
        """Resample signal to new sample rate

        Args:
            sig: Input signal
            new_sample_rate: Target sample rate

        Returns:
            Resampled signal
        """
        num_samples = int(len(sig.data) * new_sample_rate / sig.sample_rate)
        resampled_data = signal.resample(sig.data, num_samples)

        return Signal1D(
            data=resampled_data,
            sample_rate=new_sample_rate,
            time_offset=sig.time_offset
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D) -> Signal1D",
        deterministic=True,
        doc="Extract signal envelope using Hilbert transform"
    )
    def envelope(sig: Signal1D) -> Signal1D:
        """Extract signal envelope using Hilbert transform

        Args:
            sig: Input signal

        Returns:
            Envelope signal
        """
        analytic_signal = signal.hilbert(sig.data)
        envelope_data = np.abs(analytic_signal)

        return Signal1D(
            data=envelope_data,
            sample_rate=sig.sample_rate,
            time_offset=sig.time_offset
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.QUERY,
        signature="(sig1: Signal1D, sig2: Signal1D, mode: str) -> Signal1D",
        deterministic=True,
        doc="Cross-correlation of two signals"
    )
    def correlate(sig1: Signal1D, sig2: Signal1D, mode: str = 'full') -> Signal1D:
        """Cross-correlation of two signals

        Args:
            sig1: First signal
            sig2: Second signal
            mode: 'full', 'valid', or 'same'

        Returns:
            Cross-correlation signal
        """
        assert sig1.sample_rate == sig2.sample_rate, "Signals must have same sample rate"

        corr_data = np.correlate(sig1.data, sig2.data, mode=mode)

        return Signal1D(
            data=corr_data,
            sample_rate=sig1.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.QUERY,
        signature="(sig: Signal1D, height: Optional[float], distance: Optional[int]) -> ndarray",
        deterministic=True,
        doc="Detect peaks in signal"
    )
    def peak_detection(sig: Signal1D, height: Optional[float] = None,
                      distance: Optional[int] = None) -> np.ndarray:
        """Detect peaks in signal

        Args:
            sig: Input signal
            height: Minimum peak height
            distance: Minimum peak distance in samples

        Returns:
            Array of peak indices
        """
        peaks, _ = signal.find_peaks(sig.data, height=height, distance=distance)
        return peaks

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.QUERY,
        signature="(sig: Signal1D, window_size: int, hop_size: Optional[int]) -> Spectrogram",
        deterministic=True,
        doc="Compute power spectrogram"
    )
    def spectrogram_power(sig: Signal1D, window_size: int = 256,
                         hop_size: Optional[int] = None) -> Spectrogram:
        """Compute power spectrogram

        Args:
            sig: Input signal
            window_size: Window size in samples
            hop_size: Hop size in samples

        Returns:
            Power spectrogram in dB
        """
        if hop_size is None:
            hop_size = window_size // 4

        frequencies, times, power_data = signal.spectrogram(
            sig.data,
            fs=sig.sample_rate,
            nperseg=window_size,
            noverlap=window_size - hop_size
        )

        # Convert to dB
        power_db = 10 * np.log10(power_data.T + 1e-10)

        return Spectrogram(
            data=power_db,
            times=times + sig.time_offset,
            frequencies=frequencies,
            sample_rate=sig.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.QUERY,
        signature="(sig: Signal1D, window_size: int) -> Spectrum",
        deterministic=True,
        doc="Estimate power spectral density using Welch's method"
    )
    def welch_psd(sig: Signal1D, window_size: int = 256) -> Spectrum:
        """Estimate power spectral density using Welch's method

        Args:
            sig: Input signal
            window_size: Window size for averaging

        Returns:
            Power spectral density
        """
        frequencies, psd = signal.welch(
            sig.data,
            fs=sig.sample_rate,
            nperseg=window_size
        )

        return Spectrum(
            data=psd,
            frequencies=frequencies,
            sample_rate=sig.sample_rate
        )

    @staticmethod
    @operator(
        domain="signal",
        category=OpCategory.TRANSFORM,
        signature="(sig: Signal1D, method: str) -> Signal1D",
        deterministic=True,
        doc="Normalize signal"
    )
    def normalize(sig: Signal1D, method: str = 'peak') -> Signal1D:
        """Normalize signal

        Args:
            sig: Input signal
            method: 'peak' (to ±1.0) or 'rms' (to RMS = 1.0)

        Returns:
            Normalized signal
        """
        if method == 'peak':
            max_val = np.max(np.abs(sig.data))
            if max_val > 0:
                normalized_data = sig.data / max_val
            else:
                normalized_data = sig.data.copy()
        elif method == 'rms':
            rms = np.sqrt(np.mean(sig.data ** 2))
            if rms > 0:
                normalized_data = sig.data / rms
            else:
                normalized_data = sig.data.copy()
        else:
            normalized_data = sig.data.copy()

        return Signal1D(
            data=normalized_data,
            sample_rate=sig.sample_rate,
            time_offset=sig.time_offset
        )


# Export singleton instance for DSL access
sig = SignalOperations()

# Export operators for domain registry discovery
create_signal = SignalOperations.create_signal
sine_wave = SignalOperations.sine_wave
chirp = SignalOperations.chirp
white_noise = SignalOperations.white_noise
window = SignalOperations.window
fft = SignalOperations.fft
ifft = SignalOperations.ifft
rfft = SignalOperations.rfft
stft = SignalOperations.stft
istft = SignalOperations.istft
lowpass = SignalOperations.lowpass
highpass = SignalOperations.highpass
bandpass = SignalOperations.bandpass
resample = SignalOperations.resample
envelope = SignalOperations.envelope
correlate = SignalOperations.correlate
peak_detection = SignalOperations.peak_detection
spectrogram_power = SignalOperations.spectrogram_power
welch_psd = SignalOperations.welch_psd
normalize = SignalOperations.normalize
