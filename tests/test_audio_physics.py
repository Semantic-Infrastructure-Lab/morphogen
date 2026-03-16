"""Physics-level tests for audio_analysis and instrument_model domains.

Tests encode hard physical/mathematical invariants — exact formulas, known
input→output relationships — not just type or shape checks.
"""

import pytest
import numpy as np


SR = 48000


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_440():
    """1-second 440 Hz pure tone."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float64), SR


@pytest.fixture
def sine_880():
    """1-second 880 Hz pure tone."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    return np.sin(2 * np.pi * 880.0 * t).astype(np.float64), SR


@pytest.fixture
def decaying_tone():
    """1-second 440 Hz tone with exponential decay, rate=5."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    return (np.exp(-5.0 * t) * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64), SR


@pytest.fixture
def analyzed_model(decaying_tone):
    from morphogen.stdlib import instrument_model
    signal, sr = decaying_tone
    return instrument_model.analyze_instrument(signal, sr, instrument_id="test_440")


@pytest.fixture
def analyzed_model_880(sine_880):
    """Model from 880 Hz decaying tone."""
    from morphogen.stdlib import instrument_model
    signal, sr = sine_880
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (np.exp(-3.0 * t) * np.sin(2 * np.pi * 880.0 * t)).astype(np.float64)
    return instrument_model.analyze_instrument(sig, sr, instrument_id="test_880")


# ---------------------------------------------------------------------------
# T60 physics — exact mathematical invariants
# ---------------------------------------------------------------------------

class TestT60Physics:
    """T60 = 6.91 / decay_rate — exact formula, no approximation."""

    def test_t60_times_rate_equals_691_for_rate_1(self):
        from morphogen.stdlib import audio_analysis
        d = 1.0
        assert abs(audio_analysis.measure_t60(d) * d - 6.91) < 1e-9

    def test_t60_times_rate_equals_691_for_rate_691(self):
        from morphogen.stdlib import audio_analysis
        d = 6.91
        assert abs(audio_analysis.measure_t60(d) * d - 6.91) < 1e-9

    def test_t60_times_rate_equals_691_for_rate_100(self):
        from morphogen.stdlib import audio_analysis
        d = 100.0
        assert abs(audio_analysis.measure_t60(d) * d - 6.91) < 1e-9

    def test_t60_for_rate_1_is_691(self):
        """measure_t60(1.0) == 6.91 exactly (inverse of the d=6.91 test)."""
        from morphogen.stdlib import audio_analysis
        assert abs(audio_analysis.measure_t60(1.0) - 6.91) < 1e-9

    def test_amplitude_at_t60_is_0001(self):
        """The amplitude of an exponential decay at exactly T60 must equal 0.001.

        Definition: T60 is the time for the signal to decay by 60 dB.
        60 dB = factor of 0.001 in amplitude.
        """
        from morphogen.stdlib import audio_analysis
        decay_rate = 5.0
        t60 = audio_analysis.measure_t60(decay_rate)
        amplitude_at_t60 = np.exp(-decay_rate * t60)
        # 6.91 is rounded (exact: ln(1000)≈6.9078), so tolerance ~3e-6
        assert abs(amplitude_at_t60 - 0.001) < 1e-5


# ---------------------------------------------------------------------------
# Pitch detection physics
# ---------------------------------------------------------------------------

class TestPitchDetectionPhysics:
    """Pitch estimators must detect fundamental, not octave aliases."""

    def test_autocorr_detects_440(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        f0 = audio_analysis.track_fundamental(signal, sr, method="autocorrelation")
        active = f0[f0 > 0]
        assert abs(np.median(active) - 440.0) < 22.0  # within 5%

    def test_yin_detects_440(self, sine_440):
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        f0 = audio_analysis.track_fundamental(signal, sr, method="yin")
        active = f0[f0 > 0]
        assert abs(np.median(active) - 440.0) < 22.0

    def test_autocorr_detects_880_not_440(self, sine_880):
        """880 Hz must not be detected as its sub-octave 440 Hz."""
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_880
        f0 = audio_analysis.track_fundamental(signal, sr, method="autocorrelation")
        active = f0[f0 > 0]
        median_f0 = np.median(active)
        # Must be above 660 Hz (midpoint between 440 and 880 octave)
        assert median_f0 > 660.0

    def test_autocorr_and_yin_agree_on_440(self, sine_440):
        """Autocorrelation and YIN must agree within 5% on a pure tone."""
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        f0_ac = audio_analysis.track_fundamental(signal, sr, method="autocorrelation")
        f0_yin = audio_analysis.track_fundamental(signal, sr, method="yin")
        ac_median = np.median(f0_ac[f0_ac > 0])
        yin_median = np.median(f0_yin[f0_yin > 0])
        # Both agree within 5% of each other
        assert abs(ac_median - yin_median) / ac_median < 0.05

    def test_f0_frames_are_consistent_on_pure_tone(self, sine_440):
        """On a stationary pure tone, all active frames should agree within 10%."""
        from morphogen.stdlib import audio_analysis
        signal, sr = sine_440
        f0 = audio_analysis.track_fundamental(signal, sr, method="autocorrelation")
        active = f0[f0 > 0]
        assert np.std(active) / np.mean(active) < 0.10


# ---------------------------------------------------------------------------
# fit_exponential_decay physics
# ---------------------------------------------------------------------------

class TestFitExponentialDecayPhysics:
    """fit_exponential_decay must recover the true rate from synthetic data."""

    def _make_harmonics(self, decay_rate, n_frames=100, hop_size=512, sr=48000):
        """Build a (n_frames, 1) harmonics array with known exponential decay."""
        frame_time = hop_size / sr
        t = np.arange(n_frames) * frame_time
        amplitude = np.exp(-decay_rate * t)
        return amplitude[:, np.newaxis]

    def test_recovers_known_decay_rate_within_5pct(self):
        from morphogen.stdlib import audio_analysis
        d_true = 3.0
        harmonics = self._make_harmonics(d_true)
        rates = audio_analysis.fit_exponential_decay(harmonics, SR, hop_size=512)
        assert abs(rates[0] - d_true) / d_true < 0.05

    def test_faster_decay_gives_larger_rate(self):
        from morphogen.stdlib import audio_analysis
        harmonics_slow = self._make_harmonics(2.0)
        harmonics_fast = self._make_harmonics(10.0)
        rate_slow = audio_analysis.fit_exponential_decay(harmonics_slow, SR, hop_size=512)[0]
        rate_fast = audio_analysis.fit_exponential_decay(harmonics_fast, SR, hop_size=512)[0]
        assert rate_fast > rate_slow

    def test_recovery_accurate_for_range_of_rates(self):
        """Fits must be within 10% for rates spanning two decades."""
        from morphogen.stdlib import audio_analysis
        for d_true in [1.0, 5.0, 20.0]:
            harmonics = self._make_harmonics(d_true)
            rates = audio_analysis.fit_exponential_decay(harmonics, SR, hop_size=512)
            assert abs(rates[0] - d_true) / d_true < 0.10, f"Failed for d={d_true}"


# ---------------------------------------------------------------------------
# deconvolve physics
# ---------------------------------------------------------------------------

class TestDeconvolvePhysics:
    """Homomorphic deconvolution: resonator captures spectral envelope."""

    def test_resonator_log_spectrum_smoother_than_signal(self, decaying_tone):
        """Resonator log-spectrum must have lower variance than input log-spectrum.

        cepstral liftering keeps low-quefrency components (slowly varying
        spectral envelope) in the resonator — removing the high-quefrency
        excitation structure reduces variance.
        """
        from morphogen.stdlib import audio_analysis
        from scipy.fft import rfft
        signal, sr = decaying_tone
        _, resonator = audio_analysis.deconvolve(signal, sr)

        log_sig = np.log(np.abs(rfft(signal)) + 1e-10)
        log_res = np.log(np.abs(rfft(resonator)) + 1e-10)

        assert log_res.var() < log_sig.var()

    def test_excitation_plus_resonator_spans_full_log_spectrum(self, decaying_tone):
        """Excitation and resonator log-spectra should sum close to input log-spectrum."""
        from morphogen.stdlib import audio_analysis
        from scipy.fft import rfft
        signal, sr = decaying_tone
        excitation, resonator = audio_analysis.deconvolve(signal, sr)

        log_sig = np.log(np.abs(rfft(signal)) + 1e-10)
        log_exc = np.log(np.abs(rfft(excitation)) + 1e-10)
        log_res = np.log(np.abs(rfft(resonator)) + 1e-10)

        # The sum approximates the original; correlation > 0.5 (loose)
        residual = log_sig - (log_exc + log_res)
        assert np.std(residual) < np.std(log_sig)


# ---------------------------------------------------------------------------
# model_noise physics
# ---------------------------------------------------------------------------

class TestModelNoisePhysics:
    """Noise model must faithfully represent the signal's noise floor."""

    def test_temporal_envelope_length_matches_stft_time_bins(self, decaying_tone):
        """temporal_envelope length must equal the number of STFT time frames."""
        from morphogen.stdlib import audio_analysis
        from scipy import signal as scipy_signal
        sig, sr = decaying_tone
        noise = audio_analysis.model_noise(sig, sr)
        _, t, _ = scipy_signal.stft(sig, sr, nperseg=1024)
        assert len(noise.temporal_envelope) == len(t)

    def test_white_noise_gives_relatively_flat_spectral_envelope(self):
        """White noise should produce a spectral envelope with low relative variance."""
        from morphogen.stdlib import audio_analysis
        rng = np.random.default_rng(42)
        white = rng.standard_normal(SR).astype(np.float64)
        noise = audio_analysis.model_noise(white, SR, num_bands=16)
        envelope = noise.spectral_envelope
        nonzero = envelope[envelope > 0]
        # Coefficient of variation < 1.5 for white noise (roughly flat)
        assert np.std(nonzero) / np.mean(nonzero) < 1.5

    def test_spectral_envelope_has_requested_band_count(self, decaying_tone):
        from morphogen.stdlib import audio_analysis
        sig, sr = decaying_tone
        for num_bands in [8, 16, 32]:
            noise = audio_analysis.model_noise(sig, sr, num_bands=num_bands)
            assert len(noise.spectral_envelope) == num_bands


# ---------------------------------------------------------------------------
# synthesize_note physics
# ---------------------------------------------------------------------------

class TestSynthesizeNotePhysics:
    """synthesize_note: velocity linearity, decay, silence."""

    def test_velocity_half_gives_peak_045_exactly(self, analyzed_model):
        """Peak amplitude at velocity=0.5 must be exactly 0.9 * 0.5 = 0.45.

        Normalization formula: normalize to 0.9 peak → multiply by velocity.
        """
        from morphogen.stdlib import instrument_model
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=0.5, duration=0.5)
        peak = np.max(np.abs(audio))
        assert abs(peak - 0.45) < 1e-9

    def test_velocity_one_gives_peak_09_exactly(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=1.0, duration=0.5)
        peak = np.max(np.abs(audio))
        assert abs(peak - 0.9) < 1e-9

    def test_zero_velocity_gives_silence(self, analyzed_model):
        """velocity=0 must produce all zeros."""
        from morphogen.stdlib import instrument_model
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=0.0, duration=0.5)
        assert np.all(audio == 0.0)

    def test_audio_decays_over_time(self, analyzed_model):
        """RMS of the second half must be less than RMS of the first half (decay)."""
        from morphogen.stdlib import instrument_model
        audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=1.0, duration=1.0)
        mid = len(audio) // 2
        rms_first = np.sqrt(np.mean(audio[:mid] ** 2))
        rms_second = np.sqrt(np.mean(audio[mid:] ** 2))
        assert rms_second < rms_first

    def test_exact_sample_count_for_given_duration(self, analyzed_model):
        from morphogen.stdlib import instrument_model
        for duration in [0.25, 0.5, 1.0]:
            audio = instrument_model.synthesize_note(analyzed_model, pitch=440.0, duration=duration)
            assert len(audio) == int(duration * analyzed_model.sample_rate)

    def test_velocity_linearity(self, analyzed_model):
        """Peak scales linearly: peak(v=0.3) / peak(v=0.6) == 0.5."""
        from morphogen.stdlib import instrument_model
        a_low = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=0.3, duration=0.5)
        a_high = instrument_model.synthesize_note(analyzed_model, pitch=440.0, velocity=0.6, duration=0.5)
        ratio = np.max(np.abs(a_low)) / np.max(np.abs(a_high))
        assert abs(ratio - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# morph_instruments physics
# ---------------------------------------------------------------------------

class TestMorphInstrumentsPhysics:
    """morph_instruments: strict linear interpolation invariants."""

    def test_self_morph_preserves_fundamental(self, analyzed_model):
        """morph(A, A, blend) must return A.fundamental regardless of blend."""
        from morphogen.stdlib import instrument_model
        for blend in [0.0, 0.25, 0.5, 0.75, 1.0]:
            morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model, blend=blend)
            assert abs(morphed.fundamental - analyzed_model.fundamental) < 1e-9, f"blend={blend}"

    def test_midpoint_fundamental_is_exact_average(self, analyzed_model, analyzed_model_880):
        """At blend=0.5, fundamental must be exactly (a + b) / 2."""
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.5)
        expected = (analyzed_model.fundamental + analyzed_model_880.fundamental) / 2.0
        assert abs(morphed.fundamental - expected) < 1e-9

    def test_linear_interpolation_at_quarter_blend(self, analyzed_model, analyzed_model_880):
        """At blend=0.25, fundamental == 0.75*a + 0.25*b exactly."""
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.25)
        expected = 0.75 * analyzed_model.fundamental + 0.25 * analyzed_model_880.fundamental
        assert abs(morphed.fundamental - expected) < 1e-9

    def test_blend_0_gives_model_a_fundamental_exactly(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=0.0)
        assert abs(morphed.fundamental - analyzed_model.fundamental) < 1e-9

    def test_blend_1_gives_model_b_fundamental_exactly(self, analyzed_model, analyzed_model_880):
        from morphogen.stdlib import instrument_model
        morphed = instrument_model.morph_instruments(analyzed_model, analyzed_model_880, blend=1.0)
        assert abs(morphed.fundamental - analyzed_model_880.fundamental) < 1e-9
