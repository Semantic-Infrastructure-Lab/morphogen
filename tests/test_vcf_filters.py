"""Assert-based tests for VCF highpass and bandpass filters.

Replaces the coverage lost when the old return-based test_vcf_filters.py was
removed (its `return False` failures silently passed under pytest). These are
proper pytest tests with frequency-domain assertions:

- vcf_highpass attenuates content below the cutoff and passes content above it.
- vcf_bandpass passes content near the center frequency and attenuates content
  far from it on both sides.
- Q factor and filter-state continuity behave sanely.

test_vcf_modulation.py covers vcf_lowpass; this file covers the other two VCF
operators so all three are exercised.
"""

import numpy as np
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer


SAMPLE_RATE = 48000
DURATION = 0.5


def _tone_energy(data: np.ndarray, freq: float, sample_rate: int = SAMPLE_RATE) -> float:
    """Magnitude of the FFT bin nearest `freq` — a proxy for that tone's energy."""
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), d=1.0 / sample_rate)
    bin_idx = int(np.argmin(np.abs(freqs - freq)))
    return float(spectrum[bin_idx])


def _const_cutoff(n: int, value: float) -> AudioBuffer:
    return AudioBuffer(data=np.full(n, value), sample_rate=SAMPLE_RATE)


def _two_tone(low_freq: float, high_freq: float) -> AudioBuffer:
    low = audio.sine(freq=low_freq, duration=DURATION, sample_rate=SAMPLE_RATE)
    high = audio.sine(freq=high_freq, duration=DURATION, sample_rate=SAMPLE_RATE)
    return AudioBuffer(data=low.data + high.data, sample_rate=SAMPLE_RATE)


# ============================================================================
# vcf_highpass
# ============================================================================

def test_vcf_highpass_output_shape_and_finite():
    """Highpass preserves length/sample-rate and never produces NaN/inf."""
    sig = audio.saw(freq=220.0, duration=DURATION, sample_rate=SAMPLE_RATE)
    cutoff = _const_cutoff(len(sig.data), 1000.0)

    out = audio.vcf_highpass(sig, cutoff, q=0.707)

    assert len(out.data) == len(sig.data)
    assert out.sample_rate == SAMPLE_RATE
    assert np.all(np.isfinite(out.data))


def test_vcf_highpass_attenuates_below_cutoff():
    """A 100 Hz tone is attenuated far more than a 6 kHz tone by a 2 kHz highpass."""
    sig = _two_tone(low_freq=100.0, high_freq=6000.0)
    cutoff = _const_cutoff(len(sig.data), 2000.0)

    low_before = _tone_energy(sig.data, 100.0)
    high_before = _tone_energy(sig.data, 6000.0)

    out = audio.vcf_highpass(sig, cutoff, q=0.707)

    low_after = _tone_energy(out.data, 100.0)
    high_after = _tone_energy(out.data, 6000.0)

    # Low tone is strongly attenuated; high tone largely survives.
    assert low_after < low_before * 0.5
    assert high_after > high_before * 0.7
    # And relative to each other the highpass clearly favors the high tone.
    assert (high_after / high_before) > (low_after / low_before)


# ============================================================================
# vcf_bandpass
# ============================================================================

def test_vcf_bandpass_output_shape_and_finite():
    """Bandpass preserves length/sample-rate and never produces NaN/inf."""
    sig = audio.saw(freq=220.0, duration=DURATION, sample_rate=SAMPLE_RATE)
    center = _const_cutoff(len(sig.data), 1000.0)

    out = audio.vcf_bandpass(sig, center, q=2.0)

    assert len(out.data) == len(sig.data)
    assert out.sample_rate == SAMPLE_RATE
    assert np.all(np.isfinite(out.data))


def test_vcf_bandpass_passes_center_attenuates_sidebands():
    """A tone at the center frequency survives better than tones far below/above."""
    center_freq = 1000.0
    low = audio.sine(freq=100.0, duration=DURATION, sample_rate=SAMPLE_RATE)
    mid = audio.sine(freq=center_freq, duration=DURATION, sample_rate=SAMPLE_RATE)
    high = audio.sine(freq=8000.0, duration=DURATION, sample_rate=SAMPLE_RATE)
    sig = AudioBuffer(data=low.data + mid.data + high.data, sample_rate=SAMPLE_RATE)

    center = _const_cutoff(len(sig.data), center_freq)
    out = audio.vcf_bandpass(sig, center, q=3.0)

    # Ratio kept for each tone (out energy / in energy).
    keep_low = _tone_energy(out.data, 100.0) / _tone_energy(sig.data, 100.0)
    keep_mid = _tone_energy(out.data, center_freq) / _tone_energy(sig.data, center_freq)
    keep_high = _tone_energy(out.data, 8000.0) / _tone_energy(sig.data, 8000.0)

    # The center tone is retained more than either sideband.
    assert keep_mid > keep_low
    assert keep_mid > keep_high


def test_vcf_bandpass_higher_q_is_more_selective():
    """A narrower (higher-Q) bandpass rejects an off-center tone more aggressively."""
    center_freq = 1000.0
    sig = _two_tone(low_freq=center_freq, high_freq=3000.0)
    center = _const_cutoff(len(sig.data), center_freq)

    off_before = _tone_energy(sig.data, 3000.0)

    wide = audio.vcf_bandpass(sig, center, q=1.0)
    narrow = audio.vcf_bandpass(sig, center, q=6.0)

    off_wide = _tone_energy(wide.data, 3000.0)
    off_narrow = _tone_energy(narrow.data, 3000.0)

    # Narrower band => the 3 kHz off-center tone is attenuated at least as much.
    assert off_narrow <= off_wide + 1e-6
    assert off_narrow < off_before


# ============================================================================
# Q factor / state continuity edge cases
# ============================================================================

def test_vcf_highpass_high_q_stable():
    """High resonance must not blow up into non-finite or runaway output."""
    sig = audio.saw(freq=220.0, duration=0.1, sample_rate=SAMPLE_RATE)
    cutoff = _const_cutoff(len(sig.data), 1000.0)

    out = audio.vcf_highpass(sig, cutoff, q=10.0)

    assert np.all(np.isfinite(out.data))
    assert np.max(np.abs(out.data)) < 10.0


def test_vcf_bandpass_filter_state_updates():
    """Passing a filter_state buffer carries state out for hop continuity."""
    sig = audio.saw(freq=220.0, duration=0.1, sample_rate=SAMPLE_RATE)
    center = _const_cutoff(len(sig.data), 1000.0)
    state = AudioBuffer(data=np.zeros(2), sample_rate=SAMPLE_RATE)

    out = audio.vcf_bandpass(sig, center, q=2.0, filter_state=state)

    assert np.all(np.isfinite(out.data))
    # The final biquad state was written back into the provided buffer.
    assert np.any(state.data != 0.0)
