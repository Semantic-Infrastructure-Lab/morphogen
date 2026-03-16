"""Functional tests for audio_analysis domain operators.

Tests verify operators produce correct output types, shapes, and values
for known inputs — not just that they're callable.
"""

import pytest
import numpy as np
from morphogen.core.domain_registry import DomainRegistry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SR = 48000  # sample rate used throughout


@pytest.fixture
def sine_440():
    """1-second 440 Hz pure tone."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float64), SR


@pytest.fixture
def decaying_tone():
    """1-second 440 Hz tone with exponential decay (instrument-like)."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    envelope = np.exp(-t * 5.0)  # decay_rate ~5 → T60 ~1.38s
    return (envelope * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64), SR


@pytest.fixture
def harmonic_tone():
    """1-second signal with harmonics at 200, 400, 600, 800 Hz."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 200.0 * t)
        + 0.5 * np.sin(2 * np.pi * 400.0 * t)
        + 0.25 * np.sin(2 * np.pi * 600.0 * t)
        + 0.125 * np.sin(2 * np.pi * 800.0 * t)
    ).astype(np.float64)
    return signal, SR


# ---------------------------------------------------------------------------
# Domain registration
# ---------------------------------------------------------------------------

class TestAudioAnalysisDomainRegistration:
    """Verify audio_analysis domain is properly registered."""

    def test_domain_registered(self):
        DomainRegistry.initialize()
        domain = DomainRegistry.get('audio_analysis')
        assert domain is not None

    def test_all_operators_registered(self):
        DomainRegistry.initialize()
        domain = DomainRegistry.get('audio_analysis')
        operators = domain.list_operators()

        expected_ops = [
            'track_fundamental',
            'track_partials',
            'spectral_envelope',
            'analyze_modes',
            'fit_exponential_decay',
            'measure_t60',
            'measure_inharmonicity',
            'deconvolve',
            'model_noise'
        ]

        for op in expected_ops:
            assert op in operators, f"Operator {op} not registered"

        assert len(operators) == 9


# ---------------------------------------------------------------------------
# track_fundamental
# ---------------------------------------------------------------------------

class TestTrackFundamental:
    """Tests for fundamental frequency tracking."""

    def test_returns_ndarray(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        result = audio_analysis.track_fundamental(signal, sr)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        frame_size, hop_size = 2048, 512
        result = audio_analysis.track_fundamental(signal, sr, frame_size=frame_size, hop_size=hop_size)
        expected_frames = (len(signal) - frame_size) // hop_size + 1
        assert result.shape == (expected_frames,)

    def test_detects_440hz_autocorrelation(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        f0 = audio_analysis.track_fundamental(signal, sr, method="autocorrelation")
        # Median of non-zero frames should be near 440 Hz (±10%)
        active = f0[f0 > 0]
        assert len(active) > 0
        assert abs(np.median(active) - 440.0) < 44.0

    def test_detects_440hz_yin(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        f0 = audio_analysis.track_fundamental(signal, sr, method="yin")
        active = f0[f0 > 0]
        assert len(active) > 0
        assert abs(np.median(active) - 440.0) < 44.0

    def test_unknown_method_raises(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        with pytest.raises(ValueError, match="Unknown method"):
            audio_analysis.track_fundamental(signal, sr, method="bad_method")


# ---------------------------------------------------------------------------
# measure_t60
# ---------------------------------------------------------------------------

class TestMeasureT60:
    """Tests for T60 computation (pure math — strict determinism)."""

    def test_known_value(self):
        from morphogen.stdlib import audio_analysis
        # T60 = 6.91 / decay_rate
        t60 = audio_analysis.measure_t60(6.91)
        assert abs(t60 - 1.0) < 1e-6

    def test_zero_decay_returns_inf(self):
        from morphogen.stdlib import audio_analysis
        assert audio_analysis.measure_t60(0.0) == float('inf')

    def test_negative_decay_returns_inf(self):
        from morphogen.stdlib import audio_analysis
        assert audio_analysis.measure_t60(-1.0) == float('inf')

    def test_larger_decay_shorter_t60(self):
        from morphogen.stdlib import audio_analysis
        t60_slow = audio_analysis.measure_t60(1.0)
        t60_fast = audio_analysis.measure_t60(10.0)
        assert t60_slow > t60_fast

    def test_returns_float(self):
        from morphogen.stdlib import audio_analysis
        result = audio_analysis.measure_t60(5.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# deconvolve
# ---------------------------------------------------------------------------

class TestDeconvolve:
    """Tests for homomorphic deconvolution."""

    def test_returns_tuple(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        result = audio_analysis.deconvolve(signal, sr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_arrays_are_ndarray(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        excitation, resonator = audio_analysis.deconvolve(signal, sr)
        assert isinstance(excitation, np.ndarray)
        assert isinstance(resonator, np.ndarray)

    def test_output_shapes_match_input(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        excitation, resonator = audio_analysis.deconvolve(signal, sr)
        assert excitation.shape == signal.shape
        assert resonator.shape == signal.shape

    def test_is_deterministic(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        r1 = audio_analysis.deconvolve(signal, sr)
        r2 = audio_analysis.deconvolve(signal, sr)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])


# ---------------------------------------------------------------------------
# model_noise
# ---------------------------------------------------------------------------

class TestModelNoise:
    """Tests for broadband noise modeling."""

    def test_returns_noise_model(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        from morphogen.stdlib.audio_analysis import NoiseModel
        signal, sr = decaying_tone
        result = audio_analysis.model_noise(signal, sr)
        assert isinstance(result, NoiseModel)

    def test_noise_model_fields(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        result = audio_analysis.model_noise(signal, sr, num_bands=16)
        assert isinstance(result.spectral_envelope, np.ndarray)
        assert isinstance(result.temporal_envelope, np.ndarray)
        assert result.sample_rate == sr

    def test_spectral_envelope_length(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        num_bands = 16
        result = audio_analysis.model_noise(signal, sr, num_bands=num_bands)
        assert len(result.spectral_envelope) == num_bands

    def test_spectral_envelope_non_negative(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = decaying_tone
        result = audio_analysis.model_noise(signal, sr)
        assert np.all(result.spectral_envelope >= 0)


# ---------------------------------------------------------------------------
# analyze_modes
# ---------------------------------------------------------------------------

class TestAnalyzeModes:
    """Tests for modal resonance analysis."""

    def test_returns_modal_model(self, harmonic_tone):
        from morphogen.stdlib import audio_analysis
        from morphogen.stdlib.audio_analysis import ModalModel
        signal, sr = harmonic_tone
        result = audio_analysis.analyze_modes(signal, sr, num_modes=10)
        assert isinstance(result, ModalModel)

    def test_modal_model_fields(self, harmonic_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = harmonic_tone
        num_modes = 8
        result = audio_analysis.analyze_modes(signal, sr, num_modes=num_modes)
        assert isinstance(result.frequencies, np.ndarray)
        assert isinstance(result.amplitudes, np.ndarray)
        assert isinstance(result.decay_rates, np.ndarray)
        assert isinstance(result.phases, np.ndarray)

    def test_num_modes_is_upper_bound(self, harmonic_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = harmonic_tone
        num_modes = 10
        result = audio_analysis.analyze_modes(signal, sr, num_modes=num_modes)
        # Returns at most num_modes — may return fewer if fewer peaks found
        assert result.num_modes <= num_modes
        assert len(result.frequencies) == result.num_modes

    def test_frequencies_in_audible_range(self, harmonic_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = harmonic_tone
        result = audio_analysis.analyze_modes(signal, sr, num_modes=10)
        # All detected mode frequencies should be within freq_range (20-8000 Hz)
        assert np.all(result.frequencies >= 20.0)
        assert np.all(result.frequencies <= 8000.0)

    def test_amplitudes_non_negative(self, harmonic_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = harmonic_tone
        result = audio_analysis.analyze_modes(signal, sr, num_modes=10)
        assert np.all(result.amplitudes >= 0)


# ---------------------------------------------------------------------------
# spectral_envelope
# ---------------------------------------------------------------------------

class TestSpectralEnvelope:
    """Tests for spectral envelope smoothing.

    spectral_envelope takes (freq_bins, time_frames) and returns (freq_bins,)
    by averaging across time then smoothing.
    """

    @pytest.fixture
    def stft_input(self):
        """Synthetic STFT: 100 freq bins × 20 time frames."""
        rng = np.random.default_rng(42)
        return rng.random((100, 20)).astype(np.float64)

    def test_returns_ndarray(self, stft_input):
        from morphogen.stdlib import audio_analysis
        result = audio_analysis.spectral_envelope(stft_input)
        assert isinstance(result, np.ndarray)

    def test_output_shape_is_freq_bins(self, stft_input):
        from morphogen.stdlib import audio_analysis
        result = audio_analysis.spectral_envelope(stft_input)
        assert result.shape == (stft_input.shape[0],)

    def test_output_non_negative(self, stft_input):
        from morphogen.stdlib import audio_analysis
        result = audio_analysis.spectral_envelope(stft_input)
        assert np.all(result >= 0)

    def test_smoothing_reduces_spectral_variance(self):
        """Smoothed envelope should have lower variance than raw mean."""
        from morphogen.stdlib import audio_analysis
        # Noisy spectrum (high variance)
        rng = np.random.default_rng(0)
        stft = rng.random((200, 10))
        stft[100, :] = 50.0  # Sharp spike in one bin, all frames
        raw_mean = np.mean(stft, axis=1)
        smoothed = audio_analysis.spectral_envelope(stft, smoothing_factor=0.2)
        assert smoothed.var() <= raw_mean.var()


# ---------------------------------------------------------------------------
# measure_inharmonicity
# ---------------------------------------------------------------------------

class TestMeasureInharmonicity:
    """Tests for inharmonicity coefficient measurement."""

    def test_returns_float(self, harmonic_tone):
        from morphogen.stdlib import audio_analysis
        signal, sr = harmonic_tone
        result = audio_analysis.measure_inharmonicity(signal, sr, f0=200.0)
        assert isinstance(result, float)

    def test_pure_harmonic_near_zero(self, harmonic_tone):
        """A perfectly harmonic signal should have near-zero inharmonicity."""
        from morphogen.stdlib import audio_analysis
        signal, sr = harmonic_tone
        B = audio_analysis.measure_inharmonicity(signal, sr, f0=200.0, num_partials=4)
        # B should be close to 0 for ideal harmonics (within numerical tolerance)
        assert abs(B) < 0.01
