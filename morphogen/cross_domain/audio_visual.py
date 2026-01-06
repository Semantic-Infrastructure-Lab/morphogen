"""
Audio ↔ Visual Domain Interfaces

Interfaces for data flow between audio and visual domains,
including field-to-audio transformations.
"""

from typing import Any, Dict, List, Type
import numpy as np

from .base import DomainInterface


class AudioToVisualInterface(DomainInterface):
    """
    Audio → Visual: Audio-reactive visual generation.

    Use cases:
    - FFT spectrum → color palette
    - Amplitude → particle emission
    - Beat detection → visual effects
    - Frequency analysis → color shifts
    """

    source_domain = "audio"
    target_domain = "visual"

    def __init__(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        mode: str = "spectrum"
    ):
        """
        Args:
            audio_signal: Audio samples (mono or stereo)
            sample_rate: Audio sample rate
            fft_size: FFT window size for spectral analysis
            mode: Analysis mode ("spectrum", "waveform", "energy", "beat")
        """
        super().__init__(source_data=audio_signal)
        self.audio_signal = audio_signal
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.mode = mode

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """
        Convert audio to visual parameters.

        Returns:
            Dict with keys: 'colors', 'intensities', 'frequencies', 'energy'
        """
        audio = source_data if source_data is not None else self.audio_signal

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        result = {}

        if self.mode == "spectrum":
            # FFT analysis
            fft = np.fft.rfft(audio[:self.fft_size])
            spectrum = np.abs(fft)

            # Normalize spectrum
            spectrum = spectrum / (np.max(spectrum) + 1e-10)

            result['spectrum'] = spectrum
            result['frequencies'] = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

            # Spectral centroid (brightness)
            spectral_centroid = np.sum(result['frequencies'] * spectrum) / (np.sum(spectrum) + 1e-10)
            result['brightness'] = spectral_centroid / (self.sample_rate / 2)  # Normalize

        elif self.mode == "waveform":
            # Raw waveform for oscilloscope-style visuals
            result['waveform'] = audio[:self.fft_size]
            result['amplitude'] = np.abs(audio[:self.fft_size])

        elif self.mode == "energy":
            # RMS energy
            energy = np.sqrt(np.mean(audio[:self.fft_size] ** 2))
            result['energy'] = energy
            result['intensity'] = np.clip(energy * 10.0, 0.0, 1.0)

        elif self.mode == "beat":
            # Simple beat detection (onset strength)
            hop_length = 512
            n_frames = len(audio) // hop_length

            onset_strength = []
            for i in range(n_frames):
                start = i * hop_length
                end = start + hop_length
                chunk = audio[start:end]
                energy = np.sqrt(np.mean(chunk ** 2))
                onset_strength.append(energy)

            onset_strength = np.array(onset_strength)

            # Detect peaks
            threshold = np.mean(onset_strength) + np.std(onset_strength)
            beats = onset_strength > threshold

            result['onset_strength'] = onset_strength
            result['beats'] = beats

        return result

    def validate(self) -> bool:
        """Check audio signal is valid."""
        if self.audio_signal is None:
            return False

        if not isinstance(self.audio_signal, np.ndarray):
            raise TypeError("Audio signal must be numpy array")

        if len(self.audio_signal) < self.fft_size:
            raise ValueError(f"Audio signal too short (need at least {self.fft_size} samples)")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'audio_signal': np.ndarray,
            'sample_rate': int,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'visual_params': Dict[str, np.ndarray],
        }


class FieldToAudioInterface(DomainInterface):
    """
    Field → Audio: Field-driven audio synthesis.

    Use cases:
    - Temperature field → synthesis parameters
    - Vorticity → frequency modulation
    - Density patterns → rhythm generation
    - Field evolution → audio sequences
    """

    source_domain = "field"
    target_domain = "audio"

    def __init__(
        self,
        field: np.ndarray,
        mapping: Dict[str, str],
        sample_rate: int = 44100,
        duration: float = 1.0
    ):
        """
        Args:
            field: 2D field array
            mapping: Dict mapping field properties to audio parameters
                     e.g., {"mean": "frequency", "std": "amplitude"}
            sample_rate: Audio sample rate
            duration: Duration of generated audio (seconds)
        """
        super().__init__(source_data=field)
        self.field = field
        self.mapping = mapping
        self.sample_rate = sample_rate
        self.duration = duration

    def transform(self, source_data: Any) -> Dict[str, Any]:
        """
        Convert field to audio synthesis parameters.

        Returns:
            Dict with synthesis parameters
        """
        field = source_data if source_data is not None else self.field

        # Extract field statistics
        stats = {
            'mean': np.mean(field),
            'std': np.std(field),
            'min': np.min(field),
            'max': np.max(field),
            'range': np.ptp(field),
        }

        # Compute spatial statistics
        if field.ndim >= 2:
            # Gradient magnitude (activity/turbulence)
            gy, gx = np.gradient(field)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            stats['gradient_mean'] = np.mean(gradient_mag)
            stats['gradient_max'] = np.max(gradient_mag)

        audio_params = {}

        # Map field properties to audio parameters
        for field_prop, audio_param in self.mapping.items():
            value = stats.get(field_prop, 0.0)

            if audio_param == "frequency":
                # Map to musical frequency range (100-1000 Hz)
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['frequency'] = 100.0 + normalized * 900.0

            elif audio_param == "amplitude":
                # Map to amplitude (0-1)
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['amplitude'] = np.clip(normalized, 0.0, 1.0)

            elif audio_param == "modulation":
                # Modulation depth
                audio_params['modulation_depth'] = np.clip(value / 10.0, 0.0, 1.0)

            elif audio_param == "filter_cutoff":
                # Filter cutoff frequency
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['filter_cutoff'] = 200.0 + normalized * 3800.0

        # Add timing info
        audio_params['sample_rate'] = self.sample_rate
        audio_params['duration'] = self.duration
        audio_params['n_samples'] = int(self.sample_rate * self.duration)

        return audio_params

    def validate(self) -> bool:
        """Check field and mapping are valid."""
        if self.field is None or self.mapping is None:
            return False

        valid_field_props = ['mean', 'std', 'min', 'max', 'range', 'gradient_mean', 'gradient_max']
        valid_audio_params = ['frequency', 'amplitude', 'modulation', 'filter_cutoff']

        for field_prop, audio_param in self.mapping.items():
            if field_prop not in valid_field_props:
                raise ValueError(f"Unknown field property: {field_prop}")
            if audio_param not in valid_audio_params:
                raise ValueError(f"Unknown audio parameter: {audio_param}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
            'mapping': Dict[str, str],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'audio_params': Dict[str, Any],
        }
