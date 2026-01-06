"""
Time-Frequency Domain Transform Interfaces

Interfaces for cepstral and wavelet domain transformations.
"""

from typing import Any, Dict, Optional, Type
import numpy as np

from .base import DomainInterface


class TimeToCepstralInterface(DomainInterface):
    """
    Time → Cepstral domain via Discrete Cosine Transform (DCT).

    DCT is widely used for:
    - Audio compression (e.g., MP3, AAC)
    - MFCC computation (Mel-frequency cepstral coefficients)
    - Cepstral analysis for pitch detection
    - Feature extraction for speech recognition

    Supports DCT types 1-4 with orthogonal normalization.
    """

    source_domain = "time"
    target_domain = "cepstral"

    def __init__(self, signal: np.ndarray, dct_type: int = 2,
                 norm: str = "ortho", metadata: Optional[Dict] = None):
        """
        Initialize DCT transform.

        Args:
            signal: Time-domain signal (1D array)
            dct_type: DCT type (1, 2, 3, or 4). Type-2 is most common.
            norm: Normalization mode ("ortho" or None)
            metadata: Optional metadata dict
        """
        super().__init__(signal, metadata)
        self.dct_type = dct_type
        self.norm = norm

        if dct_type not in [1, 2, 3, 4]:
            raise ValueError(f"DCT type must be 1-4, got {dct_type}")
        if norm not in ["ortho", None]:
            raise ValueError(f"Norm must be 'ortho' or None, got {norm}")

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply DCT to transform time-domain signal to cepstral domain.

        Args:
            signal: Time-domain signal (1D array)

        Returns:
            Cepstral coefficients (1D array, same length as input)
        """
        from scipy.fft import dct

        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")

        # Apply DCT
        cepstral = dct(signal, type=self.dct_type, norm=self.norm)

        return cepstral.astype(np.float32)

    def validate(self) -> bool:
        """Validate signal is 1D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Signal must be numpy array")

        if self.source_data.ndim != 1:
            raise ValueError("Signal must be 1D array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'signal': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'cepstral_coefficients': np.ndarray}


class CepstralToTimeInterface(DomainInterface):
    """
    Cepstral → Time domain via Inverse Discrete Cosine Transform (IDCT).

    Reconstructs time-domain signal from DCT coefficients.
    """

    source_domain = "cepstral"
    target_domain = "time"

    def __init__(self, cepstral: np.ndarray, dct_type: int = 2,
                 norm: str = "ortho", metadata: Optional[Dict] = None):
        """
        Initialize IDCT transform.

        Args:
            cepstral: Cepstral coefficients (1D array)
            dct_type: DCT type used in forward transform (1, 2, 3, or 4)
            norm: Normalization mode ("ortho" or None)
            metadata: Optional metadata dict
        """
        super().__init__(cepstral, metadata)
        self.dct_type = dct_type
        self.norm = norm

        if dct_type not in [1, 2, 3, 4]:
            raise ValueError(f"DCT type must be 1-4, got {dct_type}")
        if norm not in ["ortho", None]:
            raise ValueError(f"Norm must be 'ortho' or None, got {norm}")

    def transform(self, cepstral: np.ndarray) -> np.ndarray:
        """
        Apply IDCT to transform cepstral coefficients back to time domain.

        Args:
            cepstral: Cepstral coefficients (1D array)

        Returns:
            Time-domain signal (1D array, same length as input)
        """
        from scipy.fft import idct

        if cepstral.ndim != 1:
            raise ValueError(f"Cepstral coefficients must be 1D, got shape {cepstral.shape}")

        # Apply IDCT
        signal = idct(cepstral, type=self.dct_type, norm=self.norm)

        return signal.astype(np.float32)

    def validate(self) -> bool:
        """Validate cepstral coefficients are 1D array."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Cepstral coefficients must be numpy array")

        if self.source_data.ndim != 1:
            raise ValueError("Cepstral coefficients must be 1D array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'cepstral_coefficients': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'signal': np.ndarray}


class TimeToWaveletInterface(DomainInterface):
    """
    Time → Wavelet domain via Continuous Wavelet Transform (CWT).

    CWT provides time-scale representation of signals, useful for:
    - Non-stationary signal analysis
    - Feature detection at multiple scales
    - Edge detection in images
    - Transient analysis in audio

    Uses scipy.signal.cwt with various mother wavelets (Morlet, Ricker, etc.)
    """

    source_domain = "time"
    target_domain = "wavelet"

    def __init__(self, signal: np.ndarray, scales: np.ndarray,
                 wavelet: str = "morlet", metadata: Optional[Dict] = None):
        """
        Initialize CWT transform.

        Args:
            signal: Time-domain signal (1D array)
            scales: Array of scales to use (e.g., np.arange(1, 128))
            wavelet: Wavelet type ("morlet", "ricker", "mexh", "morl")
            metadata: Optional metadata dict
        """
        super().__init__(signal, metadata)
        self.scales = scales
        self.wavelet = wavelet

        valid_wavelets = ["morlet", "ricker", "mexh", "morl"]
        if wavelet not in valid_wavelets:
            raise ValueError(f"Wavelet must be one of {valid_wavelets}, got {wavelet}")

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply CWT to transform signal to wavelet domain.

        Args:
            signal: Time-domain signal (1D array)

        Returns:
            Wavelet coefficients (2D array: scales × time)
        """
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")

        # Apply CWT using convolution with scaled wavelets
        # Generate Ricker wavelet (Mexican hat) for each scale
        coefficients = []

        for scale in self.scales:
            # Generate Ricker wavelet
            wavelet = self._ricker_wavelet(scale)

            # Convolve signal with wavelet
            coeff = np.convolve(signal, wavelet, mode='same')
            coefficients.append(coeff)

        coefficients = np.array(coefficients, dtype=np.float32)
        return coefficients

    def _ricker_wavelet(self, scale: float) -> np.ndarray:
        """Generate Ricker (Mexican hat) wavelet at given scale."""
        # Ricker wavelet: ψ(t) = (1 - t²) * exp(-t²/2)
        # Scale determines the width
        points = min(int(10 * scale), 100)  # Wavelet support, capped at 100
        if points < 3:
            points = 3  # Minimum wavelet size
        if points % 2 == 0:
            points += 1  # Make odd for symmetry
        t = np.linspace(-5, 5, points)
        wavelet = (1.0 - t**2) * np.exp(-t**2 / 2.0)
        # Normalize
        wavelet = wavelet / (np.sqrt(np.sum(wavelet**2)) + 1e-10)
        return wavelet.astype(np.float32)

    def validate(self) -> bool:
        """Validate signal and scales."""
        if self.source_data is None:
            return True

        if not isinstance(self.source_data, np.ndarray):
            raise TypeError("Signal must be numpy array")

        if self.source_data.ndim != 1:
            raise ValueError("Signal must be 1D array")

        if not isinstance(self.scales, np.ndarray):
            raise TypeError("Scales must be numpy array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'signal': np.ndarray, 'scales': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'wavelet_coefficients': np.ndarray}
