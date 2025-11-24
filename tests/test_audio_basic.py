"""Unit tests for basic audio operations (oscillators, utilities)."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_buffer_creation(self):
        """Test basic buffer creation."""
        data = np.zeros(1000)
        buf = AudioBuffer(data=data, sample_rate=44100)
        assert buf.num_samples == 1000
        assert buf.sample_rate == 44100
        assert not buf.is_stereo

    def test_buffer_duration(self):
        """Test duration calculation."""
        buf = AudioBuffer(data=np.zeros(44100), sample_rate=44100)
        assert abs(buf.duration - 1.0) < 0.001

    def test_buffer_stereo(self):
        """Test stereo buffer detection."""
        mono = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        stereo = AudioBuffer(data=np.zeros((1000, 2)), sample_rate=44100)
        assert not mono.is_stereo
        assert stereo.is_stereo

    def test_buffer_copy(self):
        """Test buffer copy creates independent instance."""
        buf1 = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        buf2 = buf1.copy()
        buf2.data[0] = 1.0
        assert buf1.data[0] == 0.0


class TestOscillators:
    """Tests for oscillator operations."""

    def test_sine_basic(self):
        """Test basic sine wave generation."""
        buf = audio.sine(freq=440.0, duration=0.1)
        assert buf.num_samples == 4410  # 0.1s at 44.1kHz
        assert buf.sample_rate == 44100
        # Check amplitude is in range [-1, 1]
        assert np.max(np.abs(buf.data)) <= 1.0

    def test_sine_frequency(self):
        """Test sine wave has correct frequency."""
        # Generate 1 second at 1Hz
        buf = audio.sine(freq=1.0, duration=1.0, sample_rate=100)
        # Should complete exactly 1 cycle
        # Zero crossings should be at 0.25s and 0.75s
        assert buf.data[0] < 0.01  # Starts at 0
        assert buf.data[25] > 0.99  # Peak at 0.25s

    def test_sine_phase(self):
        """Test sine wave phase offset."""
        buf1 = audio.sine(freq=440.0, phase=0.0, duration=0.1)
        buf2 = audio.sine(freq=440.0, phase=np.pi, duration=0.1)
        # 180 degree phase shift should invert signal
        assert np.allclose(buf1.data, -buf2.data, atol=0.01)

    def test_saw_basic(self):
        """Test sawtooth wave generation."""
        buf = audio.saw(freq=440.0, duration=0.1)
        assert buf.num_samples == 4410
        # Sawtooth should ramp from -1 to 1 (PolyBLEP may overshoot slightly)
        assert -2.5 < np.min(buf.data) < -0.5
        assert 0.5 < np.max(buf.data) < 2.5

    def test_saw_blep_vs_naive(self):
        """Test that BLEP sawtooth differs from naive."""
        buf_blep = audio.saw(freq=440.0, duration=0.1, blep=True)
        buf_naive = audio.saw(freq=440.0, duration=0.1, blep=False)
        # Should be different due to band-limiting
        assert not np.allclose(buf_blep.data, buf_naive.data)

    def test_square_basic(self):
        """Test square wave generation."""
        buf = audio.square(freq=440.0, duration=0.1)
        assert buf.num_samples == 4410
        # Square wave should be mostly -1 or +1
        abs_data = np.abs(buf.data)
        assert np.mean(abs_data > 0.9) > 0.8  # Most samples near ±1

    def test_square_pwm(self):
        """Test pulse width modulation."""
        buf_50 = audio.square(freq=100.0, pwm=0.5, duration=0.1, sample_rate=10000)
        buf_25 = audio.square(freq=100.0, pwm=0.25, duration=0.1, sample_rate=10000)
        # 25% PWM should have more negative samples
        assert np.sum(buf_25.data > 0) < np.sum(buf_50.data > 0)

    def test_triangle_basic(self):
        """Test triangle wave generation."""
        buf = audio.triangle(freq=440.0, duration=0.1)
        assert buf.num_samples == 4410
        # Triangle should ramp between -1 and 1
        assert -1.1 < np.min(buf.data) < -0.9
        assert 0.9 < np.max(buf.data) < 1.1

    def test_noise_white(self):
        """Test white noise generation."""
        buf = audio.noise(noise_type="white", seed=42, duration=1.0)
        assert buf.num_samples == 44100
        # White noise should have wide distribution
        assert np.std(buf.data) > 0.2

    def test_noise_deterministic(self):
        """Test noise is deterministic with same seed."""
        buf1 = audio.noise(noise_type="white", seed=42, duration=0.1)
        buf2 = audio.noise(noise_type="white", seed=42, duration=0.1)
        assert np.allclose(buf1.data, buf2.data)

    def test_noise_different_seeds(self):
        """Test different seeds produce different noise."""
        buf1 = audio.noise(noise_type="white", seed=1, duration=0.1)
        buf2 = audio.noise(noise_type="white", seed=2, duration=0.1)
        assert not np.allclose(buf1.data, buf2.data)

    def test_noise_pink(self):
        """Test pink noise generation."""
        buf = audio.noise(noise_type="pink", seed=42, duration=1.0)
        assert buf.num_samples == 44100
        # Pink noise should be normalized
        assert np.max(np.abs(buf.data)) <= 1.0

    def test_noise_brown(self):
        """Test brown noise generation."""
        buf = audio.noise(noise_type="brown", seed=42, duration=1.0)
        assert buf.num_samples == 44100
        # Brown noise should be normalized
        assert np.max(np.abs(buf.data)) <= 1.0

    def test_impulse_basic(self):
        """Test impulse train generation."""
        buf = audio.impulse(rate=10.0, duration=1.0, sample_rate=1000)
        # Should have 10 impulses in 1 second
        num_impulses = np.sum(buf.data > 0.5)
        assert 9 <= num_impulses <= 11  # Allow some tolerance

    def test_constant_basic(self):
        """Test constant value AudioBuffer generation."""
        buf = audio.constant(value=5.0, duration=0.1, sample_rate=48000)
        assert buf.num_samples == 4800  # 0.1s at 48kHz
        assert buf.sample_rate == 48000
        # All samples should be exactly 5.0
        assert np.all(buf.data == 5.0)

    def test_constant_zero(self):
        """Test constant with zero value."""
        buf = audio.constant(value=0.0, duration=0.5)
        assert buf.num_samples == 22050  # 0.5s at 44.1kHz
        assert np.all(buf.data == 0.0)

    def test_constant_negative(self):
        """Test constant with negative value."""
        buf = audio.constant(value=-2.5, duration=0.1)
        assert np.all(buf.data == -2.5)

    def test_constant_for_vcf_cutoff(self):
        """Test constant operator for VCF filter cutoff use case."""
        # This is the primary use case: fixed cutoff for vcf_lowpass
        cutoff = audio.constant(value=2000.0, duration=1.0, sample_rate=48000)
        assert len(cutoff.data) == 48000
        assert np.all(cutoff.data == 2000.0)
        # Should be compatible with vcf_lowpass input
        assert isinstance(cutoff, AudioBuffer)


class TestUtilities:
    """Tests for utility operations."""

    def test_mix_two_signals(self):
        """Test mixing two signals."""
        sig1 = audio.sine(freq=440.0, duration=0.1)
        sig2 = audio.sine(freq=880.0, duration=0.1)
        mixed = audio.mix(sig1, sig2)
        assert mixed.num_samples == sig1.num_samples
        # Mixed signal should be gain-compensated
        assert np.max(np.abs(mixed.data)) <= 1.5

    def test_mix_multiple_signals(self):
        """Test mixing three signals."""
        sig1 = audio.sine(freq=200.0, duration=0.1)
        sig2 = audio.sine(freq=400.0, duration=0.1)
        sig3 = audio.sine(freq=600.0, duration=0.1)
        mixed = audio.mix(sig1, sig2, sig3)
        assert mixed.num_samples == sig1.num_samples

    def test_mix_different_lengths(self):
        """Test mixing signals of different lengths."""
        sig1 = audio.sine(freq=440.0, duration=0.1)
        sig2 = audio.sine(freq=880.0, duration=0.2)
        mixed = audio.mix(sig1, sig2)
        # Should use longest length
        assert mixed.num_samples == sig2.num_samples

    def test_gain_positive(self):
        """Test positive gain in dB."""
        sig = audio.sine(freq=440.0, duration=0.1)
        gained = audio.gain(sig, amount_db=6.0)
        # +6dB should approximately double amplitude
        assert np.max(np.abs(gained.data)) > np.max(np.abs(sig.data))

    def test_gain_negative(self):
        """Test negative gain in dB."""
        sig = audio.sine(freq=440.0, duration=0.1)
        gained = audio.gain(sig, amount_db=-6.0)
        # -6dB should approximately halve amplitude
        assert np.max(np.abs(gained.data)) < np.max(np.abs(sig.data))

    def test_db2lin_conversion(self):
        """Test dB to linear conversion."""
        assert abs(audio.db2lin(0.0) - 1.0) < 0.001
        assert abs(audio.db2lin(6.0) - 2.0) < 0.1
        assert abs(audio.db2lin(-6.0) - 0.5) < 0.1

    def test_lin2db_conversion(self):
        """Test linear to dB conversion."""
        assert abs(audio.lin2db(1.0) - 0.0) < 0.001
        assert abs(audio.lin2db(2.0) - 6.0) < 0.1
        assert abs(audio.lin2db(0.5) - (-6.0)) < 0.1

    def test_pan_center(self):
        """Test panning at center."""
        sig = audio.sine(freq=440.0, duration=0.1)
        stereo = audio.pan(sig, position=0.0)
        assert stereo.is_stereo
        # Center pan should have equal left/right
        assert np.allclose(stereo.data[:, 0], stereo.data[:, 1], atol=0.01)

    def test_pan_left(self):
        """Test panning to left."""
        sig = audio.sine(freq=440.0, duration=0.1)
        stereo = audio.pan(sig, position=-1.0)
        # Left should be louder than right
        left_energy = np.sum(stereo.data[:, 0] ** 2)
        right_energy = np.sum(stereo.data[:, 1] ** 2)
        assert left_energy > right_energy

    def test_pan_right(self):
        """Test panning to right."""
        sig = audio.sine(freq=440.0, duration=0.1)
        stereo = audio.pan(sig, position=1.0)
        # Right should be louder than left
        left_energy = np.sum(stereo.data[:, 0] ** 2)
        right_energy = np.sum(stereo.data[:, 1] ** 2)
        assert right_energy > left_energy

    def test_clip_basic(self):
        """Test hard clipping."""
        sig = audio.sine(freq=440.0, duration=0.1)
        sig.data *= 2.0  # Amplify to cause clipping
        clipped = audio.clip(sig, limit=0.5)
        assert np.max(np.abs(clipped.data)) <= 0.5

    def test_normalize_basic(self):
        """Test normalization."""
        sig = audio.sine(freq=440.0, duration=0.1)
        sig.data *= 0.5  # Reduce amplitude
        normalized = audio.normalize(sig, target=0.98)
        assert abs(np.max(np.abs(normalized.data)) - 0.98) < 0.01

    def test_normalize_preserves_silence(self):
        """Test that normalizing silence doesn't crash."""
        sig = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        normalized = audio.normalize(sig)
        assert np.all(normalized.data == 0.0)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_sine_deterministic(self):
        """Test sine wave is deterministic."""
        buf1 = audio.sine(freq=440.0, duration=0.1)
        buf2 = audio.sine(freq=440.0, duration=0.1)
        assert np.array_equal(buf1.data, buf2.data)

    def test_noise_deterministic_with_seed(self):
        """Test noise is deterministic with same seed."""
        buf1 = audio.noise(noise_type="white", seed=42, duration=0.1)
        buf2 = audio.noise(noise_type="white", seed=42, duration=0.1)
        assert np.array_equal(buf1.data, buf2.data)

    def test_composition_deterministic(self):
        """Test composed operations are deterministic."""
        sig1 = audio.sine(freq=440.0, duration=0.1)
        gained1 = audio.gain(sig1, amount_db=6.0)

        sig2 = audio.sine(freq=440.0, duration=0.1)
        gained2 = audio.gain(sig2, amount_db=6.0)

        assert np.allclose(gained1.data, gained2.data)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_duration(self):
        """Test handling of zero duration."""
        buf = audio.sine(freq=440.0, duration=0.0)
        assert buf.num_samples == 0

    def test_very_short_duration(self):
        """Test very short duration."""
        buf = audio.sine(freq=440.0, duration=0.001)
        assert buf.num_samples > 0

    def test_high_frequency(self):
        """Test high frequency oscillators."""
        buf = audio.sine(freq=20000.0, duration=0.1)
        # Should not crash, but may alias
        assert buf.num_samples == 4410

    def test_low_frequency(self):
        """Test low frequency oscillators."""
        buf = audio.sine(freq=1.0, duration=1.0)
        assert buf.num_samples == 44100

    def test_invalid_noise_type(self):
        """Test invalid noise type raises error."""
        with pytest.raises(ValueError, match="Unknown noise type"):
            audio.noise(noise_type="invalid", duration=0.1)

    def test_mix_empty_raises_error(self):
        """Test that mixing with no signals raises error."""
        with pytest.raises(ValueError, match="At least one signal required"):
            audio.mix()

    def test_negative_frequency(self):
        """Test negative frequency (should work, wraps around)."""
        buf = audio.sine(freq=-440.0, duration=0.1)
        # Negative frequency should produce same result as positive
        assert buf.num_samples == 4410
