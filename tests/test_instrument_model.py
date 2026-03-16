"""Functional tests for instrument_model domain operators.

Tests verify operators produce correct output types, shapes, and values
for known inputs — not just that they're callable.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from morphogen.core.domain_registry import DomainRegistry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SR = 48000


@pytest.fixture
def decaying_tone():
    """1-second 440 Hz tone with exponential decay — realistic for instrument analysis."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    envelope = np.exp(-t * 5.0)
    return (envelope * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64), SR


@pytest.fixture
def decaying_tone_880():
    """1-second 880 Hz decaying tone — different pitch for morph tests."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    envelope = np.exp(-t * 3.0)
    return (envelope * np.sin(2 * np.pi * 880.0 * t)).astype(np.float64), SR


@pytest.fixture
def analyzed_model(decaying_tone):
    """Pre-analyzed InstrumentModel for reuse across tests."""
    from morphogen.stdlib import instrument_model
    signal, sr = decaying_tone
    return instrument_model.analyze_instrument(signal, sr, instrument_id="test_440")


@pytest.fixture
def analyzed_model_880(decaying_tone_880):
    """Pre-analyzed InstrumentModel at 880 Hz."""
    from morphogen.stdlib import instrument_model
    signal, sr = decaying_tone_880
    return instrument_model.analyze_instrument(signal, sr, instrument_id="test_880")


# ---------------------------------------------------------------------------
# Domain registration
# ---------------------------------------------------------------------------

class TestInstrumentModelDomainRegistration:
    """Verify instrument_model domain is properly registered."""

    def test_domain_registered(self):
        DomainRegistry.initialize()
        domain = DomainRegistry.get('instrument_model')
        assert domain is not None

    def test_all_operators_registered(self):
        DomainRegistry.initialize()
        domain = DomainRegistry.get('instrument_model')
        operators = domain.list_operators()

        core_ops = [
            'analyze_instrument',
            'synthesize_note',
            'morph_instruments',
            'save_instrument',
            'load_instrument'
        ]
        reexported_ops = [
            'track_fundamental',
            'track_partials',
            'analyze_modes',
            'fit_exponential_decay',
            'measure_inharmonicity',
            'deconvolve',
            'model_noise'
        ]
        for op in core_ops + reexported_ops:
            assert op in operators, f"Operator {op} not registered"

        assert len(operators) == 12


# ---------------------------------------------------------------------------
# analyze_instrument
# ---------------------------------------------------------------------------

class TestAnalyzeInstrument:
    """Tests for the full analysis pipeline."""

    def test_returns_instrument_model(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        from morphogen.stdlib.instrument_model import InstrumentModel
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        assert isinstance(result, InstrumentModel)

    def test_instrument_id_stored(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr, instrument_id="my_guitar")
        assert result.id == "my_guitar"

    def test_sample_rate_stored(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        assert result.sample_rate == sr

    def test_fundamental_near_440(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        # Should detect ~440 Hz (within ±20%)
        assert abs(result.fundamental - 440.0) < 88.0

    def test_harmonics_shape(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        signal, sr = decaying_tone
        num_partials = 10
        result = instrument_model.analyze_instrument(signal, sr, num_partials=num_partials)
        assert result.harmonics.ndim == 2
        assert result.harmonics.shape[1] == num_partials

    def test_modes_is_modal_model(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        from morphogen.stdlib.audio_analysis import ModalModel
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        assert isinstance(result.modes, ModalModel)

    def test_noise_is_noise_model(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        from morphogen.stdlib.audio_analysis import NoiseModel
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        assert isinstance(result.noise, NoiseModel)

    def test_body_ir_is_ndarray(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        assert isinstance(result.body_ir, np.ndarray)
        assert len(result.body_ir) > 0

    def test_decay_rates_positive(self, decaying_tone):
        from morphogen.stdlib import instrument_model
        signal, sr = decaying_tone
        result = instrument_model.analyze_instrument(signal, sr)
        # Physical decay rates should be positive
        assert np.all(result.decay_rates >= 0)


# ---------------------------------------------------------------------------
# synthesize_note
# ---------------------------------------------------------------------------

class TestSynthesizeNote:
    """Tests for note synthesis from instrument model."""

    def test_returns_ndarray(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0)
        assert isinstance(audio, np.ndarray)

    def test_correct_length(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        duration = 0.5
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, duration=duration)
        expected_samples = int(duration * analyzed_model.sample_rate)
        assert len(audio) == expected_samples

    def test_normalized_amplitude(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=1.0)
        # Normalized to ~0.9 peak
        assert np.max(np.abs(audio)) <= 1.0

    def test_velocity_scales_amplitude(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        # synthesize_note normalizes to 0.9 peak then multiplies by velocity;
        # so peak(velocity=1.0) = 0.9, peak(velocity=0.5) = 0.45
        loud = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=1.0, duration=0.5)
        quiet = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=0.5, duration=0.5)
        assert np.max(np.abs(quiet)) < np.max(np.abs(loud))

    def test_is_deterministic(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        a1 = instrument_model.synthesize_note(analyzed_model, pitch=440.0, duration=0.2)
        a2 = instrument_model.synthesize_note(analyzed_model, pitch=440.0, duration=0.2)
        np.testing.assert_array_equal(a1, a2)

    def test_different_pitches_produce_different_audio(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        a440 = instrument_model.synthesize_note(analyzed_model, pitch=440.0, duration=0.2)
        a880 = instrument_model.synthesize_note(analyzed_model, pitch=880.0, duration=0.2)
        assert not np.allclose(a440, a880)


# ---------------------------------------------------------------------------
# morph_instruments
# ---------------------------------------------------------------------------

class TestMorphInstruments:
    """Tests for instrument morphing."""

    def test_returns_instrument_model(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        from morphogen.stdlib.instrument_model import InstrumentModel
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.5)
        assert isinstance(morphed, InstrumentModel)

    def test_blend_zero_close_to_model_a(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.0)
        assert abs(morphed.fundamental - analyzed_model.fundamental) < 1.0

    def test_blend_one_close_to_model_b(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=1.0)
        assert abs(morphed.fundamental - analyzed_model_880.fundamental) < 1.0

    def test_blend_half_is_interpolated(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.5)
        expected = analyzed_model.fundamental * 0.5 + analyzed_model_880.fundamental * 0.5
        assert abs(morphed.fundamental - expected) < 1.0

    def test_morphed_id_encodes_blend(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.5)
        assert "0.50" in morphed.id


# ---------------------------------------------------------------------------
# save_instrument / load_instrument (round-trip)
# ---------------------------------------------------------------------------

class TestSaveLoadInstrument:
    """Tests for instrument serialization round-trip."""

    def test_save_creates_file(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            instrument_model.save_instrument(analyzed_model, path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_returns_instrument_model(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        from morphogen.stdlib.instrument_model import InstrumentModel
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            instrument_model.save_instrument(analyzed_model, path)
            loaded = instrument_model.load_instrument(path)
            assert isinstance(loaded, InstrumentModel)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip_preserves_id(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            instrument_model.save_instrument(analyzed_model, path)
            loaded = instrument_model.load_instrument(path)
            assert loaded.id == analyzed_model.id
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip_preserves_fundamental(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            instrument_model.save_instrument(analyzed_model, path)
            loaded = instrument_model.load_instrument(path)
            assert abs(loaded.fundamental - analyzed_model.fundamental) < 1e-6
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip_preserves_harmonics(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            instrument_model.save_instrument(analyzed_model, path)
            loaded = instrument_model.load_instrument(path)
            np.testing.assert_allclose(loaded.harmonics, analyzed_model.harmonics, rtol=1e-5)
        finally:
            Path(path).unlink(missing_ok=True)
