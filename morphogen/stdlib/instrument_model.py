"""InstrumentModelDomain - High-level instrument modeling (Layer 7).

This module implements reusable, parameterized instrument models that store
extracted features and synthesize new notes. Enables MIDI instrument creation,
timbre morphing, and luthier analysis tools.

Specification: docs/specifications/timbre-extraction.md
"""

import numpy as np
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from enum import Enum

# Import from audio_analysis
from .audio_analysis import (
    ModalModel, NoiseModel, ExcitationModel,
    track_fundamental, track_partials, analyze_modes,
    fit_exponential_decay, measure_inharmonicity, deconvolve, model_noise
)

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Core Types
# ============================================================================

class InstrumentType(Enum):
    """Instrument model type."""
    MODAL_STRING = "modal_string"
    MODAL_MEMBRANE = "modal_membrane"
    ADDITIVE = "additive"
    WAVEGUIDE = "waveguide"
    HYBRID = "hybrid"


@dataclass
class SynthParams:
    """Synthesis parameters."""
    pluck_position: float = 0.18  # 0.0 = bridge, 1.0 = neck
    pluck_stiffness: float = 0.97  # 0.0 = soft, 1.0 = hard
    body_coupling: float = 0.9  # How much body resonance to apply
    noise_level: float = -60.0  # Broadband noise mix (dB)


@dataclass
class InstrumentModel:
    """Reusable, parameterized instrument model."""
    id: str
    instrument_type: InstrumentType

    # Analysis results
    fundamental: float  # Base pitch of analyzed note (Hz)
    harmonics: np.ndarray  # (n_frames, num_partials) harmonic amplitudes
    modes: ModalModel  # Resonant modes
    body_ir: np.ndarray  # Body impulse response
    noise: NoiseModel  # Noise layer
    excitation: np.ndarray  # Pluck/attack model
    decay_rates: np.ndarray  # Decay per partial (1/s)
    inharmonicity: float  # Inharmonicity coefficient

    # Metadata
    sample_rate: float  # Hz
    synth_params: SynthParams

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = {
            'id': self.id,
            'instrument_type': self.instrument_type.value,
            'fundamental': float(self.fundamental),
            'harmonics': self.harmonics.tolist(),
            'modes': {
                'frequencies': self.modes.frequencies.tolist(),
                'amplitudes': self.modes.amplitudes.tolist(),
                'decay_rates': self.modes.decay_rates.tolist(),
                'phases': self.modes.phases.tolist()
            },
            'body_ir': self.body_ir.tolist(),
            'noise': {
                'spectral_envelope': self.noise.spectral_envelope.tolist(),
                'temporal_envelope': self.noise.temporal_envelope.tolist(),
                'sample_rate': float(self.noise.sample_rate)
            },
            'excitation': self.excitation.tolist(),
            'decay_rates': self.decay_rates.tolist(),
            'inharmonicity': float(self.inharmonicity),
            'sample_rate': float(self.sample_rate),
            'synth_params': asdict(self.synth_params)
        }
        return data

    @staticmethod
    def from_dict(data: dict) -> 'InstrumentModel':
        """Load from dictionary."""
        modes = ModalModel(
            frequencies=np.array(data['modes']['frequencies']),
            amplitudes=np.array(data['modes']['amplitudes']),
            decay_rates=np.array(data['modes']['decay_rates']),
            phases=np.array(data['modes']['phases'])
        )

        noise = NoiseModel(
            spectral_envelope=np.array(data['noise']['spectral_envelope']),
            temporal_envelope=np.array(data['noise']['temporal_envelope']),
            sample_rate=data['noise']['sample_rate']
        )

        synth_params = SynthParams(**data['synth_params'])

        return InstrumentModel(
            id=data['id'],
            instrument_type=InstrumentType(data['instrument_type']),
            fundamental=data['fundamental'],
            harmonics=np.array(data['harmonics']),
            modes=modes,
            body_ir=np.array(data['body_ir']),
            noise=noise,
            excitation=np.array(data['excitation']),
            decay_rates=np.array(data['decay_rates']),
            inharmonicity=data['inharmonicity'],
            sample_rate=data['sample_rate'],
            synth_params=synth_params
        )


# ============================================================================
# Operators
# ============================================================================

@operator(
    domain="instrument_model",
    category=OpCategory.QUERY,
    signature="(signal_data: np.ndarray, sample_rate: float, instrument_id: str = 'instrument', instrument_type: InstrumentType = InstrumentType.MODAL_STRING, num_partials: int = 20, num_modes: int = 20) -> InstrumentModel",
    deterministic=True,
    doc="Full analysis pipeline: extract all timbre features from audio recording"
)
def analyze_instrument(
    signal_data: np.ndarray,
    sample_rate: float,
    instrument_id: str = "instrument",
    instrument_type: InstrumentType = InstrumentType.MODAL_STRING,
    num_partials: int = 20,
    num_modes: int = 20
) -> InstrumentModel:
    """Full analysis pipeline: extract all timbre features.

    Args:
        signal_data: Audio recording of single note
        sample_rate: Sample rate (Hz)
        instrument_id: Unique identifier
        instrument_type: Type of instrument model
        num_partials: Number of harmonics to track
        num_modes: Number of modal resonances to extract

    Returns:
        InstrumentModel with extracted features

    Determinism: repro
    """
    # 1. Track fundamental frequency
    f0_trajectory = track_fundamental(signal_data, sample_rate, method="yin")
    fundamental = np.median(f0_trajectory[f0_trajectory > 0])

    # 2. Track harmonic amplitudes
    harmonics = track_partials(
        signal_data, sample_rate, f0_trajectory,
        num_partials=num_partials
    )

    # 3. Fit resonant modes
    modes = analyze_modes(signal_data, sample_rate, num_modes=num_modes)

    # 4. Separate excitation and body IR
    excitation, body_ir = deconvolve(signal_data, sample_rate)

    # 5. Extract noise model
    noise = model_noise(signal_data, sample_rate, num_bands=32)

    # 6. Fit decay rates
    decay_rates = fit_exponential_decay(harmonics, sample_rate)

    # 7. Measure inharmonicity
    inharmonicity = measure_inharmonicity(signal_data, sample_rate, fundamental)

    # Create model
    model = InstrumentModel(
        id=instrument_id,
        instrument_type=instrument_type,
        fundamental=fundamental,
        harmonics=harmonics,
        modes=modes,
        body_ir=body_ir,
        noise=noise,
        excitation=excitation,
        decay_rates=decay_rates,
        inharmonicity=inharmonicity,
        sample_rate=sample_rate,
        synth_params=SynthParams()
    )

    return model


@operator(
    domain="instrument_model",
    category=OpCategory.QUERY,
    signature="(model: InstrumentModel, pitch: float, velocity: float = 1.0, duration: float = 2.0) -> np.ndarray",
    deterministic=True,
    doc="Generate new note from instrument model at specified pitch and velocity"
)
def synthesize_note(
    model: InstrumentModel,
    pitch: float,
    velocity: float = 1.0,
    duration: float = 2.0
) -> np.ndarray:
    """Generate new note from instrument model.

    Args:
        model: Instrument model
        pitch: Target pitch (Hz)
        velocity: Note velocity (0-1)
        duration: Note duration (seconds)

    Returns:
        audio: Synthesized audio signal

    Determinism: strict
    """
    sample_rate = model.sample_rate
    n_samples = int(duration * sample_rate)

    # Pitch ratio (for transposition)
    pitch_ratio = pitch / model.fundamental

    # Method depends on instrument type
    if model.instrument_type in [InstrumentType.MODAL_STRING, InstrumentType.MODAL_MEMBRANE]:
        audio = _synthesize_modal(model, pitch_ratio, velocity, n_samples)

    elif model.instrument_type == InstrumentType.ADDITIVE:
        audio = _synthesize_additive(model, pitch_ratio, velocity, n_samples)

    else:
        # Default: additive synthesis
        audio = _synthesize_additive(model, pitch_ratio, velocity, n_samples)

    # Apply body resonance (convolution)
    if model.synth_params.body_coupling > 0:
        audio = _apply_body_ir(audio, model.body_ir, model.synth_params.body_coupling)

    # Normalize to 0.9 peak first, then scale by velocity so velocity is preserved
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    audio = audio * velocity

    return audio


def _synthesize_modal(
    model: InstrumentModel,
    pitch_ratio: float,
    velocity: float,
    n_samples: int
) -> np.ndarray:
    """Synthesize using modal synthesis."""
    sample_rate = model.sample_rate
    t = np.arange(n_samples) / sample_rate

    # Sum of damped sinusoids
    audio = np.zeros(n_samples)

    for i in range(model.modes.num_modes):
        f = model.modes.frequencies[i] * pitch_ratio
        A = model.modes.amplitudes[i] * velocity
        d = model.modes.decay_rates[i]
        phi = model.modes.phases[i]

        # Damped sinusoid: A * exp(-d*t) * sin(2*pi*f*t + phi)
        mode_signal = A * np.exp(-d * t) * np.sin(2 * np.pi * f * t + phi)
        audio += mode_signal

    return audio


def _synthesize_additive(
    model: InstrumentModel,
    pitch_ratio: float,
    velocity: float,
    n_samples: int
) -> np.ndarray:
    """Synthesize using additive synthesis."""
    sample_rate = model.sample_rate
    t = np.arange(n_samples) / sample_rate

    # Average harmonic amplitudes over time
    avg_harmonics = np.mean(model.harmonics, axis=0)

    # Sum of harmonics with exponential decay
    audio = np.zeros(n_samples)

    for h, amp in enumerate(avg_harmonics):
        if amp > 1e-6:
            f = model.fundamental * pitch_ratio * (h + 1) * (1 + model.inharmonicity * (h + 1)**2)**0.5
            d = model.decay_rates[h] if h < len(model.decay_rates) else 5.0

            # Harmonic: A * exp(-d*t) * sin(2*pi*f*t)
            harmonic_signal = amp * np.exp(-d * t) * np.sin(2 * np.pi * f * t)
            audio += harmonic_signal

    return audio


def _apply_body_ir(audio: np.ndarray, body_ir: np.ndarray, coupling: float) -> np.ndarray:
    """Apply body impulse response via convolution."""
    # Convolve and truncate to original length (body_ir may be longer than audio)
    body_response = np.convolve(audio, body_ir, mode='full')[:len(audio)]

    # Mix with dry signal
    audio_with_body = audio * (1 - coupling) + body_response * coupling

    return audio_with_body


@operator(
    domain="instrument_model",
    category=OpCategory.TRANSFORM,
    signature="(model_a: InstrumentModel, model_b: InstrumentModel, blend: float) -> InstrumentModel",
    deterministic=True,
    doc="Morph between two instrument models with blend parameter [0,1]"
)
def morph_instruments(
    model_a: InstrumentModel,
    model_b: InstrumentModel,
    blend: float
) -> InstrumentModel:
    """Morph between two instrument models.

    Args:
        model_a: First instrument model
        model_b: Second instrument model
        blend: Blend factor (0 = all A, 1 = all B)

    Returns:
        Morphed instrument model

    Determinism: strict
    """
    # Interpolate fundamental
    fundamental = model_a.fundamental * (1 - blend) + model_b.fundamental * blend

    # Interpolate harmonics (match dimensions)
    min_frames = min(model_a.harmonics.shape[0], model_b.harmonics.shape[0])
    min_partials = min(model_a.harmonics.shape[1], model_b.harmonics.shape[1])

    harmonics_a = model_a.harmonics[:min_frames, :min_partials]
    harmonics_b = model_b.harmonics[:min_frames, :min_partials]
    harmonics = harmonics_a * (1 - blend) + harmonics_b * blend

    # Interpolate modes
    min_modes = min(model_a.modes.num_modes, model_b.modes.num_modes)
    modes = ModalModel(
        frequencies=model_a.modes.frequencies[:min_modes] * (1 - blend) + model_b.modes.frequencies[:min_modes] * blend,
        amplitudes=model_a.modes.amplitudes[:min_modes] * (1 - blend) + model_b.modes.amplitudes[:min_modes] * blend,
        decay_rates=model_a.modes.decay_rates[:min_modes] * (1 - blend) + model_b.modes.decay_rates[:min_modes] * blend,
        phases=model_a.modes.phases[:min_modes] * (1 - blend) + model_b.modes.phases[:min_modes] * blend
    )

    # Interpolate body IR (match lengths)
    min_ir_len = min(len(model_a.body_ir), len(model_b.body_ir))
    body_ir = model_a.body_ir[:min_ir_len] * (1 - blend) + model_b.body_ir[:min_ir_len] * blend

    # Interpolate noise spectral envelope
    min_bands = min(len(model_a.noise.spectral_envelope), len(model_b.noise.spectral_envelope))
    noise_spec = model_a.noise.spectral_envelope[:min_bands] * (1 - blend) + model_b.noise.spectral_envelope[:min_bands] * blend

    min_time = min(len(model_a.noise.temporal_envelope), len(model_b.noise.temporal_envelope))
    noise_temp = model_a.noise.temporal_envelope[:min_time] * (1 - blend) + model_b.noise.temporal_envelope[:min_time] * blend

    noise = NoiseModel(
        spectral_envelope=noise_spec,
        temporal_envelope=noise_temp,
        sample_rate=model_a.noise.sample_rate
    )

    # Mix excitations
    min_exc_len = min(len(model_a.excitation), len(model_b.excitation))
    excitation = model_a.excitation[:min_exc_len] * (1 - blend) + model_b.excitation[:min_exc_len] * blend

    # Interpolate decay rates
    min_decays = min(len(model_a.decay_rates), len(model_b.decay_rates))
    decay_rates = model_a.decay_rates[:min_decays] * (1 - blend) + model_b.decay_rates[:min_decays] * blend

    # Interpolate inharmonicity
    inharmonicity = model_a.inharmonicity * (1 - blend) + model_b.inharmonicity * blend

    # Create morphed model
    morphed = InstrumentModel(
        id=f"{model_a.id}_{model_b.id}_morph_{blend:.2f}",
        instrument_type=model_a.instrument_type,
        fundamental=fundamental,
        harmonics=harmonics,
        modes=modes,
        body_ir=body_ir,
        noise=noise,
        excitation=excitation,
        decay_rates=decay_rates,
        inharmonicity=inharmonicity,
        sample_rate=model_a.sample_rate,
        synth_params=model_a.synth_params
    )

    return morphed


@operator(
    domain="instrument_model",
    category=OpCategory.MUTATE,
    signature="(model: InstrumentModel, path: str) -> None",
    deterministic=False,
    doc="Serialize instrument model to disk (.pkl or .npz)"
)
def save_instrument(model: InstrumentModel, path: str):
    """Serialize instrument model to disk.

    Args:
        model: Instrument model
        path: File path (.pkl or .npz)

    Determinism: N/A (I/O)
    """
    data = model.to_dict()

    if path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif path.endswith('.npz'):
        # Convert to numpy arrays for npz format
        np.savez_compressed(path, **{k: np.array(v) for k, v in data.items()})
    else:
        raise ValueError(f"Unsupported format: {path}. Use .pkl or .npz")


@operator(
    domain="instrument_model",
    category=OpCategory.CONSTRUCT,
    signature="(path: str) -> InstrumentModel",
    deterministic=False,
    doc="Load instrument model from disk (.pkl or .npz)"
)
def load_instrument(path: str) -> InstrumentModel:
    """Load instrument model from disk.

    Args:
        path: File path (.pkl or .npz)

    Returns:
        InstrumentModel

    Determinism: N/A (I/O)
    """
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif path.endswith('.npz'):
        npz_data = np.load(path, allow_pickle=True)
        data = {k: npz_data[k].item() for k in npz_data.files}
    else:
        raise ValueError(f"Unsupported format: {path}. Use .pkl or .npz")

    return InstrumentModel.from_dict(data)


# ============================================================================
# Domain Registration
# ============================================================================

class InstrumentModelOperations:
    """Instrument model domain operations."""

    @staticmethod
    def analyze(signal_data, sample_rate, instrument_id="instrument", instrument_type=InstrumentType.MODAL_STRING, num_partials=20, num_modes=20):
        return analyze_instrument(signal_data, sample_rate, instrument_id, instrument_type, num_partials, num_modes)

    @staticmethod
    def synthesize(model, pitch, velocity=1.0, duration=2.0):
        return synthesize_note(model, pitch, velocity, duration)

    @staticmethod
    def morph(model_a, model_b, blend):
        return morph_instruments(model_a, model_b, blend)

    @staticmethod
    def save(model, path):
        return save_instrument(model, path)

    @staticmethod
    def load(path):
        return load_instrument(path)


# Create domain instance
instrument = InstrumentModelOperations()


__all__ = [
    'InstrumentType', 'SynthParams', 'InstrumentModel',
    'analyze_instrument', 'synthesize_note', 'morph_instruments',
    'save_instrument', 'load_instrument',
    'instrument', 'InstrumentModelOperations'
]
