"""Audio analysis and spectral processing.

This module provides FFT/STFT operations, spectral analysis functions,
and spectral processing tools.
"""

from typing import Tuple
import numpy as np

from morphogen.core.operator import operator, OpCategory
from .buffer import AudioBuffer, DEFAULT_SAMPLE_RATE
from .mixing import db2lin


# =============================================================================
# FFT / STFT Operations
# =============================================================================

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
    """
    if signal.is_stereo:
        raise ValueError("FFT requires mono signal (use left channel)")

    spectrum = np.fft.rfft(signal.data)
    freqs = np.fft.rfftfreq(signal.num_samples, 1.0 / signal.sample_rate)

    return freqs, spectrum


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
    """
    data = np.fft.irfft(spectrum)
    return AudioBuffer(data=data.real, sample_rate=sample_rate)


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
    """
    if signal.is_stereo:
        raise ValueError("STFT requires mono signal")

    window = np.hanning(window_size)
    num_frames = 1 + (signal.num_samples - window_size) // hop_size
    num_freqs = window_size // 2 + 1
    stft_matrix = np.zeros((num_freqs, num_frames), dtype=np.complex128)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal.data[start:start + window_size]
        windowed = frame * window
        stft_matrix[:, i] = np.fft.rfft(windowed)

    times = np.arange(num_frames) * hop_size / signal.sample_rate
    freqs = np.fft.rfftfreq(window_size, 1.0 / signal.sample_rate)

    return times, freqs, stft_matrix


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
        stft_matrix: Complex STFT matrix (freq x time)
        hop_size: Hop size in samples
        sample_rate: Sample rate in Hz

    Returns:
        Reconstructed audio buffer
    """
    num_freqs, num_frames = stft_matrix.shape
    window_size = (num_freqs - 1) * 2

    window = np.hanning(window_size)
    output_length = (num_frames - 1) * hop_size + window_size
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i in range(num_frames):
        start = i * hop_size
        frame = np.fft.irfft(stft_matrix[:, i])
        output[start:start + window_size] += frame * window
        window_sum[start:start + window_size] += window ** 2

    window_sum[window_sum < 1e-10] = 1.0
    output = output / window_sum

    return AudioBuffer(data=output, sample_rate=sample_rate)


# =============================================================================
# Spectrum Analysis
# =============================================================================

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
    """
    freqs, complex_spec = fft(signal)
    magnitudes = np.abs(complex_spec)
    return freqs, magnitudes


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
    """
    freqs, complex_spec = fft(signal)
    phases = np.angle(complex_spec)
    return freqs, phases


@operator(
    domain="audio",
    category=OpCategory.QUERY,
    signature="(signal: AudioBuffer) -> float",
    deterministic=True,
    doc="Compute spectral centroid"
)
def spectral_centroid(signal: AudioBuffer) -> float:
    """Calculate spectral centroid (brightness measure).

    Args:
        signal: Input audio buffer (must be mono)

    Returns:
        Spectral centroid in Hz
    """
    freqs, magnitudes = spectrum(signal)
    centroid = np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-10)
    return float(centroid)


@operator(
    domain="audio",
    category=OpCategory.QUERY,
    signature="(signal: AudioBuffer, threshold: float) -> float",
    deterministic=True,
    doc="Compute spectral rolloff frequency"
)
def spectral_rolloff(signal: AudioBuffer, threshold: float = 0.85) -> float:
    """Calculate spectral rolloff frequency.

    Args:
        signal: Input audio buffer (must be mono)
        threshold: Energy threshold (0.0 to 1.0, default 0.85)

    Returns:
        Rolloff frequency in Hz
    """
    freqs, magnitudes = spectrum(signal)
    energy = magnitudes ** 2
    cumulative_energy = np.cumsum(energy)
    total_energy = cumulative_energy[-1]

    rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]

    if len(rolloff_idx) > 0:
        return float(freqs[rolloff_idx[0]])
    else:
        return float(freqs[-1])


@operator(
    domain="audio",
    category=OpCategory.QUERY,
    signature="(signal: AudioBuffer, hop_size: int) -> ndarray",
    deterministic=True,
    doc="Compute spectral flux over time"
)
def spectral_flux(signal: AudioBuffer, hop_size: int = 512) -> np.ndarray:
    """Calculate spectral flux (change in spectrum over time).

    Args:
        signal: Input audio buffer (must be mono)
        hop_size: Hop size in samples

    Returns:
        Array of spectral flux values over time
    """
    times, freqs, stft_matrix = stft(signal, hop_size=hop_size)
    mag_spectrum = np.abs(stft_matrix)

    flux = np.zeros(mag_spectrum.shape[1])
    for i in range(1, mag_spectrum.shape[1]):
        diff = mag_spectrum[:, i] - mag_spectrum[:, i-1]
        diff = np.maximum(0, diff)
        flux[i] = np.sum(diff ** 2)

    return flux


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
    """
    freqs, magnitudes = spectrum(signal)

    valid_idx = freqs >= min_freq
    freqs = freqs[valid_idx]
    magnitudes = magnitudes[valid_idx]

    peaks_idx = []
    for i in range(1, len(magnitudes) - 1):
        if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
            peaks_idx.append(i)

    if len(peaks_idx) == 0:
        peaks_idx = np.argsort(magnitudes)[::-1][:num_peaks]
    else:
        peaks_idx = np.array(peaks_idx)
        sorted_idx = np.argsort(magnitudes[peaks_idx])[::-1]
        peaks_idx = peaks_idx[sorted_idx][:num_peaks]

    peak_freqs = freqs[peaks_idx]
    peak_mags = magnitudes[peaks_idx]

    return peak_freqs, peak_mags


@operator(
    domain="audio",
    category=OpCategory.QUERY,
    signature="(signal: AudioBuffer) -> float",
    deterministic=True,
    doc="Compute RMS level"
)
def rms(signal: AudioBuffer) -> float:
    """Calculate RMS (Root Mean Square) level.

    Args:
        signal: Input audio buffer

    Returns:
        RMS value (0.0 to 1.0)
    """
    rms_val = np.sqrt(np.mean(signal.data ** 2))
    return float(rms_val)


@operator(
    domain="audio",
    category=OpCategory.QUERY,
    signature="(signal: AudioBuffer) -> int",
    deterministic=True,
    doc="Count zero crossings"
)
def zero_crossings(signal: AudioBuffer) -> int:
    """Count zero crossings (sign changes).

    Args:
        signal: Input audio buffer (must be mono)

    Returns:
        Number of zero crossings
    """
    if signal.is_stereo:
        raise ValueError("Zero crossings requires mono signal")

    signs = np.sign(signal.data)
    crossings = np.sum(np.abs(np.diff(signs))) / 2

    return int(crossings)


# =============================================================================
# Spectral Processing
# =============================================================================

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

    Args:
        signal: Input audio buffer (must be mono)
        threshold_db: Threshold in dB
        window_size: FFT window size
        hop_size: Hop size in samples

    Returns:
        Noise-gated audio buffer
    """
    if signal.is_stereo:
        raise ValueError("Spectral gate requires mono signal")

    times, freqs, stft_matrix = stft(signal, window_size, hop_size)
    threshold_lin = db2lin(threshold_db)

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    mask = magnitude > threshold_lin
    magnitude_gated = magnitude * mask

    stft_gated = magnitude_gated * np.exp(1j * phase)

    return istft(stft_gated, hop_size, signal.sample_rate)


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
    """
    if signal.is_stereo:
        raise ValueError("Spectral filter requires mono signal")

    freqs, spec = fft(signal)
    spectrum_filtered = spec * freq_mask

    return ifft(spectrum_filtered, signal.sample_rate)


@operator(
    domain="audio",
    category=OpCategory.TRANSFORM,
    signature="(signal: AudioBuffer, impulse: AudioBuffer) -> AudioBuffer",
    deterministic=True,
    doc="Convolve signal with impulse response"
)
def convolution(signal: AudioBuffer, impulse: AudioBuffer) -> AudioBuffer:
    """Apply convolution (for reverb, filtering, etc.).

    Args:
        signal: Input audio buffer (must be mono)
        impulse: Impulse response (must be mono)

    Returns:
        Convolved audio buffer
    """
    if signal.is_stereo or impulse.is_stereo:
        raise ValueError("Convolution requires mono signals")

    conv_length = signal.num_samples + impulse.num_samples - 1
    fft_length = 2 ** int(np.ceil(np.log2(conv_length)))

    signal_fft = np.fft.rfft(signal.data, n=fft_length)
    impulse_fft = np.fft.rfft(impulse.data, n=fft_length)

    result_fft = signal_fft * impulse_fft
    result = np.fft.irfft(result_fft)
    result = result[:conv_length]

    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak

    return AudioBuffer(data=result, sample_rate=signal.sample_rate)
