"""AudioAnalysisDomain - Extract timbre features from audio recordings.

This module implements timbre extraction and analysis operators for converting
acoustic recordings into reusable synthesis models. Essential for instrument
modeling, luthier analysis, and physical modeling synthesis.

Specification: docs/specifications/timbre-extraction.md
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, rfft, rfftfreq

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class ModalModel:
    """Damped sinusoid modal model."""
    frequencies: np.ndarray  # Hz
    amplitudes: np.ndarray  # Linear amplitude
    decay_rates: np.ndarray  # 1/s
    phases: np.ndarray  # Radians

    @property
    def num_modes(self) -> int:
        return len(self.frequencies)


@dataclass
class NoiseModel:
    """Broadband noise signature."""
    spectral_envelope: np.ndarray  # Frequency bins
    temporal_envelope: np.ndarray  # Time samples
    sample_rate: float  # Hz


@dataclass
class ExcitationModel:
    """Pluck/attack transient model."""
    waveform: np.ndarray  # Time-domain signal
    duration: float  # Seconds
    sample_rate: float  # Hz


# ============================================================================
# Operators - Pitch Tracking
# ============================================================================

@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, method: str = 'autocorrelation', frame_size: int = 2048, hop_size: int = 512) -> np.ndarray",
    deterministic=True,
    doc="Track fundamental frequency (f0) over time using pitch detection"
)
def track_fundamental(
    signal_data: np.ndarray,
    sample_rate: float,
    method: str = "autocorrelation",
    frame_size: int = 2048,
    hop_size: int = 512
) -> np.ndarray:
    """Track fundamental frequency over time.

    Args:
        signal_data: Audio signal
        sample_rate: Sample rate (Hz)
        method: "autocorrelation", "yin", or "hps" (harmonic product spectrum)
        frame_size: Analysis frame size (samples)
        hop_size: Hop size between frames (samples)

    Returns:
        f0_trajectory: Fundamental frequency over time (Hz)

    Determinism: repro
    """
    n_frames = (len(signal_data) - frame_size) // hop_size + 1
    f0_trajectory = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal_data[start:end]

        if method == "autocorrelation":
            f0 = _autocorrelation_pitch(frame, sample_rate)
        elif method == "yin":
            f0 = _yin_pitch(frame, sample_rate)
        elif method == "hps":
            f0 = _hps_pitch(frame, sample_rate)
        else:
            raise ValueError(f"Unknown method: {method}")

        f0_trajectory[i] = f0

    return f0_trajectory


def _autocorrelation_pitch(frame: np.ndarray, sample_rate: float) -> float:
    """Pitch detection via autocorrelation."""
    # Apply window
    windowed = frame * np.hanning(len(frame))

    # Autocorrelation
    corr = np.correlate(windowed, windowed, mode='full')
    corr = corr[len(corr)//2:]

    # Find first peak after lag=0
    # Look in range 50Hz to 2000Hz
    min_lag = int(sample_rate / 2000.0)
    max_lag = int(sample_rate / 50.0)

    if max_lag >= len(corr):
        return 0.0

    search_range = corr[min_lag:max_lag]
    if len(search_range) == 0:
        return 0.0

    peak_lag = np.argmax(search_range) + min_lag

    if peak_lag == 0:
        return 0.0

    f0 = sample_rate / peak_lag
    return f0


def _yin_pitch(frame: np.ndarray, sample_rate: float, threshold: float = 0.1) -> float:
    """YIN pitch detection algorithm."""
    frame_size = len(frame)
    tau_max = frame_size // 2

    # Difference function
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff[tau] = np.sum((frame[:frame_size-tau] - frame[tau:frame_size])**2)

    # Cumulative mean normalized difference
    cmnd = np.zeros(tau_max)
    cmnd[0] = 1.0
    running_sum = 0.0

    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmnd[tau] = diff[tau] / (running_sum / tau) if running_sum > 0 else 1.0

    # Find first minimum below threshold
    for tau in range(2, tau_max):
        if cmnd[tau] < threshold:
            # Parabolic interpolation
            if tau < tau_max - 1:
                x0, x1, x2 = cmnd[tau-1], cmnd[tau], cmnd[tau+1]
                tau_interp = tau + (x2 - x0) / (2 * (2*x1 - x0 - x2))
            else:
                tau_interp = tau

            f0 = sample_rate / tau_interp
            return f0

    return 0.0


def _hps_pitch(frame: np.ndarray, sample_rate: float) -> float:
    """Harmonic product spectrum pitch detection."""
    # FFT
    spectrum = np.abs(rfft(frame * np.hanning(len(frame))))
    freqs = rfftfreq(len(frame), 1.0/sample_rate)

    # Harmonic product spectrum (multiply downsampled spectra)
    hps = spectrum.copy()
    for h in range(2, 6):  # Harmonics 2-5
        decimated = scipy_signal.decimate(spectrum, h, ftype='fir', zero_phase=True)
        hps[:len(decimated)] *= decimated

    # Find peak
    min_idx = int(50.0 * len(frame) / sample_rate)  # 50 Hz
    max_idx = int(2000.0 * len(frame) / sample_rate)  # 2000 Hz

    if max_idx >= len(hps):
        max_idx = len(hps) - 1

    peak_idx = np.argmax(hps[min_idx:max_idx]) + min_idx
    f0 = freqs[peak_idx]

    return f0


# ============================================================================
# Operators - Harmonic Analysis
# ============================================================================

@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, f0_trajectory: np.ndarray, num_partials: int = 20, frame_size: int = 4096, hop_size: int = 512) -> np.ndarray",
    deterministic=True,
    doc="Track harmonic partials over time given f0 trajectory"
)
def track_partials(
    signal_data: np.ndarray,
    sample_rate: float,
    f0_trajectory: np.ndarray,
    num_partials: int = 20,
    frame_size: int = 4096,
    hop_size: int = 512
) -> np.ndarray:
    """Track harmonic amplitudes over time.

    Args:
        signal_data: Audio signal
        sample_rate: Sample rate (Hz)
        f0_trajectory: Fundamental frequency trajectory (Hz)
        num_partials: Number of harmonics to track
        frame_size: Analysis frame size (samples)
        hop_size: Hop size between frames (samples)

    Returns:
        harmonics: (n_frames, num_partials) array of harmonic amplitudes

    Determinism: repro
    """
    n_frames = len(f0_trajectory)
    harmonics = np.zeros((n_frames, num_partials))

    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size

        if end > len(signal_data):
            break

        frame = signal_data[start:end]
        f0 = f0_trajectory[i]

        if f0 > 0:
            # FFT
            spectrum = rfft(frame * np.hanning(len(frame)))
            freqs = rfftfreq(len(frame), 1.0/sample_rate)

            # Extract harmonic amplitudes
            for h in range(num_partials):
                target_freq = f0 * (h + 1)

                # Find closest bin
                idx = np.argmin(np.abs(freqs - target_freq))
                harmonics[i, h] = np.abs(spectrum[idx])

    return harmonics


@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(stft_result: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray",
    deterministic=True,
    doc="Extract smooth spectral envelope from STFT"
)
def spectral_envelope(
    stft_result: np.ndarray,
    smoothing_factor: float = 0.1
) -> np.ndarray:
    """Extract smooth spectral envelope.

    Args:
        stft_result: STFT result (freq_bins, time_frames)
        smoothing_factor: Smoothing factor (0-1)

    Returns:
        envelope: Spectral envelope (freq_bins,)

    Determinism: strict
    """
    # Average across time
    magnitude = np.abs(stft_result)
    envelope = np.mean(magnitude, axis=1)

    # Smooth with moving average
    window_size = max(3, int(len(envelope) * smoothing_factor))
    if window_size % 2 == 0:
        window_size += 1

    envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')

    return envelope_smooth


# ============================================================================
# Operators - Modal Analysis
# ============================================================================

@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, num_modes: int = 20, freq_range: Tuple[float, float] = (20.0, 8000.0)) -> ModalModel",
    deterministic=True,
    doc="Analyze signal into modal components (frequencies, decay rates, amplitudes)"
)
def analyze_modes(
    signal_data: np.ndarray,
    sample_rate: float,
    num_modes: int = 20,
    freq_range: Tuple[float, float] = (20.0, 8000.0)
) -> ModalModel:
    """Fit damped sinusoid modes to signal.

    Model: sum of A_i * exp(-d_i*t) * sin(2*pi*f_i*t + phi_i)

    Args:
        signal_data: Audio signal
        sample_rate: Sample rate (Hz)
        num_modes: Number of modes to extract
        freq_range: (min_freq, max_freq) in Hz

    Returns:
        ModalModel with fitted parameters

    Determinism: repro
    """
    # FFT to find resonant peaks
    spectrum = rfft(signal_data)
    freqs = rfftfreq(len(signal_data), 1.0/sample_rate)

    # Find peaks in spectrum
    magnitude = np.abs(spectrum)

    # Restrict to frequency range
    min_idx = np.argmin(np.abs(freqs - freq_range[0]))
    max_idx = np.argmin(np.abs(freqs - freq_range[1]))

    # Find peaks
    peaks, properties = scipy_signal.find_peaks(
        magnitude[min_idx:max_idx],
        height=np.max(magnitude) * 0.01,
        distance=int(sample_rate / 4000.0)
    )

    peaks = peaks + min_idx

    # Sort by magnitude and take top num_modes
    peak_magnitudes = magnitude[peaks]
    sorted_indices = np.argsort(peak_magnitudes)[::-1]
    top_peaks = peaks[sorted_indices[:num_modes]]

    # Extract mode parameters
    mode_freqs = freqs[top_peaks]
    mode_amps = magnitude[top_peaks]
    mode_phases = np.angle(spectrum[top_peaks])

    # Estimate decay rates from signal envelope
    mode_decays = np.zeros(len(mode_freqs))
    for i in range(len(mode_freqs)):
        # Simple estimate: assume T60 ~ 1 second
        mode_decays[i] = 6.91 / 1.0  # -60 dB decay

    return ModalModel(
        frequencies=mode_freqs,
        amplitudes=mode_amps,
        decay_rates=mode_decays,
        phases=mode_phases
    )


# ============================================================================
# Operators - Decay Analysis
# ============================================================================

@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(harmonics: np.ndarray, sample_rate: float, hop_size: int = 512) -> np.ndarray",
    deterministic=True,
    doc="Fit exponential decay envelope to each harmonic partial"
)
def fit_exponential_decay(
    harmonics: np.ndarray,
    sample_rate: float,
    hop_size: int = 512
) -> np.ndarray:
    """Fit exponential decay envelope per partial.

    Args:
        harmonics: (n_frames, num_partials) harmonic amplitudes
        sample_rate: Sample rate (Hz)
        hop_size: Hop size (samples)

    Returns:
        decay_rates: (num_partials,) decay rates (1/s)

    Determinism: repro
    """
    n_frames, num_partials = harmonics.shape
    decay_rates = np.zeros(num_partials)

    frame_time = hop_size / sample_rate

    for p in range(num_partials):
        amp_trajectory = harmonics[:, p]

        # Find peak
        peak_idx = np.argmax(amp_trajectory)

        if peak_idx >= n_frames - 10:
            continue

        # Fit exponential to decay portion
        decay_portion = amp_trajectory[peak_idx:]

        if len(decay_portion) < 10:
            continue

        # Log-linear fit: log(A) = log(A0) - d*t
        times = np.arange(len(decay_portion)) * frame_time
        log_amps = np.log(decay_portion + 1e-10)

        # Linear regression
        coeffs = np.polyfit(times, log_amps, 1)
        decay_rates[p] = -coeffs[0]

    return decay_rates


@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(decay_rate: float) -> float",
    deterministic=True,
    doc="Compute T60 (time to decay 60dB) from decay rate"
)
def measure_t60(decay_rate: float) -> float:
    """Compute T60 (time to decay 60dB) from decay rate.

    Args:
        decay_rate: Decay rate (1/s)

    Returns:
        t60: Time to decay 60dB (s)

    Determinism: strict
    """
    # A(t) = A0 * exp(-d*t)
    # -60 dB = 20*log10(A/A0) = -60
    # A/A0 = 10^(-60/20) = 10^(-3) = 0.001
    # exp(-d*t60) = 0.001
    # -d*t60 = ln(0.001)
    # t60 = -ln(0.001) / d = 6.91 / d

    if decay_rate <= 0:
        return float('inf')

    t60 = 6.91 / decay_rate
    return t60


@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, f0: float, num_partials: int = 10) -> float",
    deterministic=True,
    doc="Measure inharmonicity coefficient for string-like signals"
)
def measure_inharmonicity(
    signal_data: np.ndarray,
    sample_rate: float,
    f0: float,
    num_partials: int = 10
) -> float:
    """Measure inharmonicity coefficient (piano strings, guitar).

    Inharmonicity: f_n = n*f0 * sqrt(1 + B*n^2)

    Args:
        signal_data: Audio signal
        sample_rate: Sample rate (Hz)
        f0: Fundamental frequency (Hz)
        num_partials: Number of partials to analyze

    Returns:
        B: Inharmonicity coefficient

    Determinism: repro
    """
    # FFT
    spectrum = rfft(signal_data)
    freqs = rfftfreq(len(signal_data), 1.0/sample_rate)

    # Measure actual partial frequencies
    measured_freqs = []

    for n in range(1, num_partials + 1):
        target_freq = n * f0

        # Search window around expected frequency
        search_width = f0 * 0.1
        min_idx = np.argmin(np.abs(freqs - (target_freq - search_width)))
        max_idx = np.argmin(np.abs(freqs - (target_freq + search_width)))

        # Find peak in window
        window_spectrum = np.abs(spectrum[min_idx:max_idx])
        peak_idx = np.argmax(window_spectrum) + min_idx

        measured_freqs.append(freqs[peak_idx])

    # Fit inharmonicity model: f_n = n*f0 * sqrt(1 + B*n^2)
    # Rearrange: (f_n / (n*f0))^2 = 1 + B*n^2
    # Linear fit: y = 1 + B*x where y = (f_n / (n*f0))^2, x = n^2

    n_values = np.arange(1, num_partials + 1)
    y_values = np.array([(f / (n * f0))**2 for n, f in zip(n_values, measured_freqs)])
    x_values = n_values**2

    # Linear regression
    coeffs = np.polyfit(x_values, y_values, 1)
    B = coeffs[0]

    return B


# ============================================================================
# Operators - Deconvolution
# ============================================================================

@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, cepstral_lifter: int = 50) -> Tuple[np.ndarray, np.ndarray]",
    deterministic=True,
    doc="Separate excitation and resonator via homomorphic deconvolution"
)
def deconvolve(
    signal_data: np.ndarray,
    sample_rate: float,
    cepstral_lifter: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate excitation and resonator via homomorphic deconvolution.

    Uses cepstral domain:
        cepstrum(signal) = cepstrum(excitation) + cepstrum(resonator)

    Args:
        signal_data: Audio signal
        sample_rate: Sample rate (Hz)
        cepstral_lifter: Lifter length (samples)

    Returns:
        excitation: Excitation signal
        resonator: Impulse response of resonator

    Determinism: repro
    """
    # FFT
    spectrum = fft(signal_data)

    # Cepstrum (log spectrum → IFFT)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = ifft(log_spectrum).real

    # Separate excitation (high quefrency) and resonator (low quefrency)
    # Excitation has fast variations → high quefrency
    # Resonator has slow variations → low quefrency

    cepstrum_resonator = cepstrum.copy()
    cepstrum_resonator[cepstral_lifter:-cepstral_lifter] = 0  # Keep low quefrency

    cepstrum_excitation = cepstrum.copy()
    cepstrum_excitation[:cepstral_lifter] = 0  # Keep high quefrency
    cepstrum_excitation[-cepstral_lifter:] = 0

    # Back to frequency domain
    log_spectrum_resonator = fft(cepstrum_resonator).real
    log_spectrum_excitation = fft(cepstrum_excitation).real

    # Reconstruct signals
    spectrum_resonator = np.exp(log_spectrum_resonator + 1j * np.angle(spectrum))
    spectrum_excitation = np.exp(log_spectrum_excitation + 1j * np.angle(spectrum))

    resonator = ifft(spectrum_resonator).real
    excitation = ifft(spectrum_excitation).real

    return excitation, resonator


# ============================================================================
# Operators - Noise Modeling
# ============================================================================

@operator(
    domain="audio_analysis",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, num_bands: int = 32) -> NoiseModel",
    deterministic=True,
    doc="Capture broadband noise signature in frequency bands"
)
def model_noise(
    signal_data: np.ndarray,
    sample_rate: float,
    num_bands: int = 32
) -> NoiseModel:
    """Capture broadband noise signature.

    Args:
        signal_data: Audio signal
        sample_rate: Sample rate (Hz)
        num_bands: Number of frequency bands

    Returns:
        NoiseModel

    Determinism: repro
    """
    # STFT
    f, t, Zxx = scipy_signal.stft(signal_data, sample_rate, nperseg=1024)

    # Extract noise floor (minimum magnitude across time)
    noise_floor = np.min(np.abs(Zxx), axis=1)

    # Bin into bands
    band_edges = np.logspace(np.log10(20), np.log10(sample_rate/2), num_bands + 1)
    spectral_envelope = np.zeros(num_bands)

    for i in range(num_bands):
        band_mask = (f >= band_edges[i]) & (f < band_edges[i+1])
        if band_mask.any():
            spectral_envelope[i] = np.mean(noise_floor[band_mask])
        # else leave as 0.0 — band spans no STFT bins at this resolution

    # Temporal envelope (RMS over time)
    temporal_envelope = np.sqrt(np.mean(np.abs(Zxx)**2, axis=0))

    return NoiseModel(
        spectral_envelope=spectral_envelope,
        temporal_envelope=temporal_envelope,
        sample_rate=sample_rate
    )


# ============================================================================
# Domain Registration
# ============================================================================

class AudioAnalysisOperations:
    """Audio analysis domain operations."""

    @staticmethod
    def track_fundamental(signal_data, sample_rate, method="autocorrelation", frame_size=2048, hop_size=512):
        return track_fundamental(signal_data, sample_rate, method, frame_size, hop_size)

    @staticmethod
    def track_partials(signal_data, sample_rate, f0_trajectory, num_partials=20, frame_size=4096, hop_size=512):
        return track_partials(signal_data, sample_rate, f0_trajectory, num_partials, frame_size, hop_size)

    @staticmethod
    def spectral_envelope(stft_result, smoothing_factor=0.1):
        return spectral_envelope(stft_result, smoothing_factor)

    @staticmethod
    def analyze_modes(signal_data, sample_rate, num_modes=20, freq_range=(20.0, 8000.0)):
        return analyze_modes(signal_data, sample_rate, num_modes, freq_range)

    @staticmethod
    def fit_exponential_decay(harmonics, sample_rate, hop_size=512):
        return fit_exponential_decay(harmonics, sample_rate, hop_size)

    @staticmethod
    def measure_t60(decay_rate):
        return measure_t60(decay_rate)

    @staticmethod
    def measure_inharmonicity(signal_data, sample_rate, f0, num_partials=10):
        return measure_inharmonicity(signal_data, sample_rate, f0, num_partials)

    @staticmethod
    def deconvolve(signal_data, sample_rate, cepstral_lifter=50):
        return deconvolve(signal_data, sample_rate, cepstral_lifter)

    @staticmethod
    def model_noise(signal_data, sample_rate, num_bands=32):
        return model_noise(signal_data, sample_rate, num_bands)


# Create domain instance
audio_analysis = AudioAnalysisOperations()


__all__ = [
    'ModalModel', 'NoiseModel', 'ExcitationModel',
    'track_fundamental', 'track_partials', 'spectral_envelope',
    'analyze_modes', 'fit_exponential_decay', 'measure_t60', 'measure_inharmonicity',
    'deconvolve', 'model_noise',
    'audio_analysis', 'AudioAnalysisOperations'
]
