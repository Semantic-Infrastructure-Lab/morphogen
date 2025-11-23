#!/usr/bin/env python3
"""Test multiply operator for ring modulation and AM synthesis."""

import numpy as np
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer


def test_basic_multiply():
    """Test basic multiplication of two signals."""
    print("\n" + "="*60)
    print("Test 1: Basic Signal Multiplication")
    print("="*60)

    # Create two simple sine waves
    sig1 = audio.sine(freq=440.0, duration=0.1, sample_rate=48000)
    sig2 = audio.sine(freq=220.0, duration=0.1, sample_rate=48000)

    # Multiply them
    result = audio.multiply(sig1, sig2)

    print(f"Signal 1: {sig1.num_samples} samples @ {sig1.sample_rate}Hz")
    print(f"Signal 2: {sig2.num_samples} samples @ {sig2.sample_rate}Hz")
    print(f"Result: {result.num_samples} samples @ {result.sample_rate}Hz")
    print(f"Result RMS: {np.sqrt(np.mean(result.data ** 2)):.4f}")
    print(f"Result peak: {np.max(np.abs(result.data)):.4f}")

    # Validate
    assert result.num_samples == sig1.num_samples
    assert result.sample_rate == sig1.sample_rate
    assert np.max(np.abs(result.data)) <= 1.0  # Should be within bounds

    print("✅ PASS: Basic multiplication works correctly")
    return True


def test_ring_modulation():
    """Test ring modulation (classic effect)."""
    print("\n" + "="*60)
    print("Test 2: Ring Modulation")
    print("="*60)

    # Ring mod: multiply two oscillators
    # Classic ring mod produces sum and difference frequencies
    carrier_freq = 440.0  # A4
    modulator_freq = 100.0  # Low frequency modulator

    carrier = audio.sine(freq=carrier_freq, duration=0.5, sample_rate=48000)
    modulator = audio.sine(freq=modulator_freq, duration=0.5, sample_rate=48000)

    # Ring modulation
    ring_mod = audio.multiply(carrier, modulator, gain=1.0)

    print(f"Carrier: {carrier_freq}Hz")
    print(f"Modulator: {modulator_freq}Hz")
    print(f"Expected sidebands: {carrier_freq + modulator_freq}Hz, {carrier_freq - modulator_freq}Hz")
    print(f"Ring mod RMS: {np.sqrt(np.mean(ring_mod.data ** 2)):.4f}")
    print(f"Ring mod peak: {np.max(np.abs(ring_mod.data)):.4f}")

    # Ring modulation should produce sidebands
    # RMS should be roughly 0.5 (product of two unit sine waves)
    rms = np.sqrt(np.mean(ring_mod.data ** 2))
    assert 0.3 < rms < 0.6, f"RMS {rms} outside expected range for ring mod"

    print("✅ PASS: Ring modulation produces expected output")
    return True


def test_amplitude_modulation():
    """Test amplitude modulation (envelope shaping)."""
    print("\n" + "="*60)
    print("Test 3: Amplitude Modulation (Envelope)")
    print("="*60)

    # Create a sawtooth wave
    saw = audio.saw(freq=220.0, duration=1.0, sample_rate=48000)

    # Create ADSR envelope
    envelope = audio.adsr(
        attack=0.1,
        decay=0.2,
        sustain=0.5,
        release=0.3,
        duration=1.0,
        sample_rate=48000
    )

    # Apply envelope via multiplication
    shaped = audio.multiply(saw, envelope, gain=1.0)

    print(f"Sawtooth: {saw.num_samples} samples")
    print(f"Envelope: {envelope.num_samples} samples")
    print(f"Shaped: {shaped.num_samples} samples")

    # Check that envelope shaping works
    # Attack region should have increasing amplitude
    attack_samples = int(0.1 * 48000)
    attack_region = shaped.data[:attack_samples]
    attack_envelope = envelope.data[:attack_samples]

    print(f"Attack region RMS: {np.sqrt(np.mean(attack_region ** 2)):.4f}")
    print(f"Sustain region RMS: {np.sqrt(np.mean(shaped.data[attack_samples:attack_samples*5] ** 2)):.4f}")

    # Validate that multiplication actually shaped the signal
    assert shaped.num_samples == saw.num_samples

    # The shaped signal should have variation due to envelope
    # Check that the last 10% (release phase) has lower amplitude than middle
    release_start = int(0.7 * 48000)
    release_region = shaped.data[release_start:]
    middle_region = shaped.data[attack_samples:int(0.5 * 48000)]

    release_rms = np.sqrt(np.mean(release_region ** 2))
    middle_rms = np.sqrt(np.mean(middle_region ** 2))

    print(f"Middle region RMS: {middle_rms:.4f}")
    print(f"Release region RMS: {release_rms:.4f}")

    assert release_rms < middle_rms, "Release phase should have lower amplitude than middle"

    print("✅ PASS: Amplitude modulation (envelope shaping) works correctly")
    return True


def test_different_lengths():
    """Test multiply with different signal lengths."""
    print("\n" + "="*60)
    print("Test 4: Different Signal Lengths")
    print("="*60)

    # Create signals of different lengths
    sig1 = audio.sine(freq=440.0, duration=0.5, sample_rate=48000)
    sig2 = audio.sine(freq=220.0, duration=0.3, sample_rate=48000)

    result = audio.multiply(sig1, sig2)

    print(f"Signal 1: {sig1.num_samples} samples")
    print(f"Signal 2: {sig2.num_samples} samples")
    print(f"Result: {result.num_samples} samples")

    # Result should be length of longer signal (sig1)
    assert result.num_samples == sig1.num_samples
    print("✅ PASS: Different length signals handled correctly (padded)")
    return True


def test_gain_parameter():
    """Test gain parameter."""
    print("\n" + "="*60)
    print("Test 5: Gain Parameter")
    print("="*60)

    sig1 = audio.sine(freq=440.0, duration=0.1, sample_rate=48000)
    sig2 = audio.sine(freq=220.0, duration=0.1, sample_rate=48000)

    # Test with different gains
    result_unity = audio.multiply(sig1, sig2, gain=1.0)
    result_half = audio.multiply(sig1, sig2, gain=0.5)
    result_double = audio.multiply(sig1, sig2, gain=2.0)

    rms_unity = np.sqrt(np.mean(result_unity.data ** 2))
    rms_half = np.sqrt(np.mean(result_half.data ** 2))
    rms_double = np.sqrt(np.mean(result_double.data ** 2))

    print(f"Unity gain (1.0) RMS: {rms_unity:.4f}")
    print(f"Half gain (0.5) RMS: {rms_half:.4f}")
    print(f"Double gain (2.0) RMS: {rms_double:.4f}")

    # Check gain relationships
    assert abs(rms_half - rms_unity * 0.5) < 0.01, "Half gain not working correctly"
    assert abs(rms_double - rms_unity * 2.0) < 0.01, "Double gain not working correctly"

    print("✅ PASS: Gain parameter works correctly")
    return True


def main():
    """Run all multiply operator tests."""
    print("\n" + "="*70)
    print("MULTIPLY OPERATOR TEST SUITE")
    print("="*70)

    tests = [
        ("Basic Multiplication", test_basic_multiply),
        ("Ring Modulation", test_ring_modulation),
        ("Amplitude Modulation", test_amplitude_modulation),
        ("Different Lengths", test_different_lengths),
        ("Gain Parameter", test_gain_parameter),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ FAIL: {name} - {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name.ljust(30)}: {status}")
        all_passed = all_passed and passed

    print("="*70)
    if all_passed:
        print("✅ All multiply operator tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
