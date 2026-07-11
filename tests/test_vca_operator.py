#!/usr/bin/env python3
"""
Comprehensive test suite for VCA (Voltage Controlled Amplifier) operator.

Tests:
1. Basic VCA with ADSR envelope
2. Linear vs Exponential curves
3. Different signal lengths
4. Tremolo effect (LFO modulation)
5. CV normalization (bipolar CV handling)
"""

import numpy as np
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer


def test_basic_vca_envelope():
    """Test VCA with classic ADSR envelope shaping."""
    print("\n" + "="*60)
    print("Test 1: Basic VCA with ADSR Envelope")
    print("="*60)

    # Generate audio signal (sawtooth)
    signal = audio.saw(freq=220.0, duration=2.0, sample_rate=48000)

    # Generate ADSR envelope
    envelope = audio.adsr(
        attack=0.1,
        decay=0.3,
        sustain=0.6,
        release=0.5,
        duration=2.0,
        sample_rate=48000
    )

    # Apply VCA
    shaped = audio.vca(signal, envelope, curve="linear")

    # Verify output
    assert shaped.num_samples == signal.num_samples
    assert shaped.sample_rate == signal.sample_rate

    # Envelope should reduce amplitude
    signal_rms = np.sqrt(np.mean(signal.data ** 2))
    shaped_rms = np.sqrt(np.mean(shaped.data ** 2))

    print(f"  Original signal RMS: {signal_rms:.4f}")
    print(f"  VCA-shaped signal RMS: {shaped_rms:.4f}")
    print(f"  Amplitude reduction: {(1.0 - shaped_rms/signal_rms)*100:.1f}%")

    # Shaped signal should have lower RMS than original
    assert shaped_rms < signal_rms

    # Check that release phase is quieter than attack/sustain
    attack_region = shaped.data[int(0.1*48000):int(0.2*48000)]
    release_region = shaped.data[int(1.5*48000):int(1.6*48000)]

    attack_rms = np.sqrt(np.mean(attack_region ** 2))
    release_rms = np.sqrt(np.mean(release_region ** 2))

    print(f"  Attack region RMS: {attack_rms:.4f}")
    print(f"  Release region RMS: {release_rms:.4f}")

    assert release_rms < attack_rms

    print("✅ PASS: VCA envelope shaping works correctly")
    return True


def test_linear_vs_exponential():
    """Test linear vs exponential VCA curves."""
    print("\n" + "="*60)
    print("Test 2: Linear vs Exponential Curves")
    print("="*60)

    # Generate signal
    signal = audio.sine(freq=440.0, duration=1.0, sample_rate=48000)

    # Generate simple envelope (linear ramp down)
    envelope_data = np.linspace(1.0, 0.0, 48000)
    envelope = AudioBuffer(data=envelope_data, sample_rate=48000)

    # Apply VCA with both curves
    linear_result = audio.vca(signal, envelope, curve="linear")
    exp_result = audio.vca(signal, envelope, curve="exponential")

    # Calculate RMS for each half
    first_half_linear = np.sqrt(np.mean(linear_result.data[:24000] ** 2))
    first_half_exp = np.sqrt(np.mean(exp_result.data[:24000] ** 2))

    second_half_linear = np.sqrt(np.mean(linear_result.data[24000:] ** 2))
    second_half_exp = np.sqrt(np.mean(exp_result.data[24000:] ** 2))

    print(f"  Linear curve:")
    print(f"    First half RMS: {first_half_linear:.4f}")
    print(f"    Second half RMS: {second_half_linear:.4f}")
    print(f"  Exponential curve:")
    print(f"    First half RMS: {first_half_exp:.4f}")
    print(f"    Second half RMS: {second_half_exp:.4f}")

    # Exponential (sqrt) curve resists amplitude changes
    # making fade-outs sound more natural and "musical"
    # Both halves should be louder than linear due to the sqrt compression
    assert first_half_exp > first_half_linear
    print(f"  ✓ Exponential curve louder in first half ({first_half_exp:.4f} > {first_half_linear:.4f})")

    assert second_half_exp > second_half_linear
    print(f"  ✓ Exponential curve louder in second half ({second_half_exp:.4f} > {second_half_linear:.4f})")

    # Overall, exponential should have higher RMS (resists fade)
    linear_total_rms = np.sqrt(np.mean(linear_result.data ** 2))
    exp_total_rms = np.sqrt(np.mean(exp_result.data ** 2))
    assert exp_total_rms > linear_total_rms
    print(f"  ✓ Exponential curve has higher overall RMS ({exp_total_rms:.4f} > {linear_total_rms:.4f})")

    print("✅ PASS: Curve types behave correctly")
    return True


def test_different_lengths():
    """Test VCA with different signal and CV lengths."""
    print("\n" + "="*60)
    print("Test 3: Different Signal Lengths")
    print("="*60)

    # Generate signals of different lengths
    signal_long = audio.sine(freq=440.0, duration=1.0, sample_rate=48000)  # 48000 samples
    signal_short = audio.sine(freq=440.0, duration=0.5, sample_rate=48000)  # 24000 samples

    cv = audio.adsr(attack=0.1, decay=0.2, sustain=0.7, release=0.3,
                   duration=0.7, sample_rate=48000)  # 33600 samples

    # Test: longer signal, shorter CV
    result1 = audio.vca(signal_long, signal_short)
    print(f"  Long signal (48000) + Short CV (24000) = {result1.num_samples} samples")
    assert result1.num_samples == 48000  # Should match longer signal

    # Test: shorter signal, longer CV
    result2 = audio.vca(signal_short, signal_long)
    print(f"  Short signal (24000) + Long CV (48000) = {result2.num_samples} samples")
    assert result2.num_samples == 48000  # Should match longer signal

    # Test: both different from CV
    result3 = audio.vca(signal_long, cv)
    print(f"  Signal (48000) + CV (33600) = {result3.num_samples} samples")
    assert result3.num_samples == 48000  # Should match longer

    print("✅ PASS: Length handling works correctly")
    return True


def test_tremolo_effect():
    """Test VCA for tremolo effect (amplitude modulation with LFO)."""
    print("\n" + "="*60)
    print("Test 4: Tremolo Effect (LFO)")
    print("="*60)

    # Generate sustained tone
    signal = audio.sine(freq=440.0, duration=2.0, sample_rate=48000)

    # Generate LFO (6 Hz tremolo rate)
    lfo = audio.sine(freq=6.0, duration=2.0, sample_rate=48000)

    # Apply VCA for tremolo
    tremolo = audio.vca(signal, lfo, curve="linear")

    # Verify modulation is happening
    # Split into segments and check for amplitude variation
    segment_size = 4000  # ~83ms segments
    num_segments = len(tremolo.data) // segment_size

    rms_values = []
    for i in range(num_segments):
        segment = tremolo.data[i*segment_size:(i+1)*segment_size]
        rms = np.sqrt(np.mean(segment ** 2))
        rms_values.append(rms)

    # Calculate variation in RMS across segments
    rms_std = np.std(rms_values)
    rms_mean = np.mean(rms_values)
    variation = rms_std / rms_mean

    print(f"  Tremolo rate: 6 Hz")
    print(f"  Mean RMS across segments: {rms_mean:.4f}")
    print(f"  RMS variation (std/mean): {variation:.4f}")

    # Should have significant variation (tremolo is working)
    assert variation > 0.1  # At least 10% variation

    print(f"✅ PASS: Tremolo effect produces amplitude modulation (variation: {variation:.2%})")
    return True


def test_cv_normalization():
    """Test VCA CV normalization (handles bipolar CVs)."""
    print("\n" + "="*60)
    print("Test 5: CV Normalization (Bipolar CV)")
    print("="*60)

    # Generate signal
    signal = audio.sine(freq=440.0, duration=1.0, sample_rate=48000)

    # Test with unipolar CV (0 to 1)
    cv_unipolar = AudioBuffer(
        data=np.linspace(0.0, 1.0, 48000),
        sample_rate=48000
    )

    # Test with bipolar CV (-1 to 1) - should be normalized to 0 to 1
    cv_bipolar = AudioBuffer(
        data=np.linspace(-1.0, 1.0, 48000),
        sample_rate=48000
    )

    # Apply VCA
    result_unipolar = audio.vca(signal, cv_unipolar)
    result_bipolar = audio.vca(signal, cv_bipolar)

    # Both should produce similar results (CV gets normalized)
    unipolar_rms = np.sqrt(np.mean(result_unipolar.data ** 2))
    bipolar_rms = np.sqrt(np.mean(result_bipolar.data ** 2))

    print(f"  Unipolar CV (0→1) result RMS: {unipolar_rms:.4f}")
    print(f"  Bipolar CV (-1→1) result RMS: {bipolar_rms:.4f}")
    print(f"  Difference: {abs(unipolar_rms - bipolar_rms):.6f}")

    # Should be very close (normalization working)
    assert abs(unipolar_rms - bipolar_rms) < 0.01

    # Test with arbitrary range CV
    cv_arbitrary = AudioBuffer(
        data=np.linspace(0.3, 0.7, 48000),
        sample_rate=48000
    )

    result_arbitrary = audio.vca(signal, cv_arbitrary)
    arbitrary_rms = np.sqrt(np.mean(result_arbitrary.data ** 2))

    print(f"  Arbitrary CV (0.3→0.7) result RMS: {arbitrary_rms:.4f}")

    # Should still normalize to full range
    assert abs(arbitrary_rms - unipolar_rms) < 0.01

    print("✅ PASS: CV normalization handles all ranges correctly")
    return True


def main():
    """Run all VCA operator tests."""
    print("\n" + "="*60)
    print("VCA OPERATOR TEST SUITE")
    print("="*60)

    tests = [
        ("Basic VCA with ADSR Envelope", test_basic_vca_envelope),
        ("Linear vs Exponential Curves", test_linear_vs_exponential),
        ("Different Signal Lengths", test_different_lengths),
        ("Tremolo Effect (LFO)", test_tremolo_effect),
        ("CV Normalization (Bipolar)", test_cv_normalization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ FAIL: {test_name}")
            print(f"   Error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:35} : {status}")

    print("="*60)
    if passed == total:
        print(f"✅ All VCA operator tests passed! ({passed}/{total})")
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
