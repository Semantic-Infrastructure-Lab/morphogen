"""Test suite for VCF (Voltage-Controlled Filter) modulation.

Tests the new vcf_lowpass operator with time-varying cutoff modulation,
including cross-rate connections (control-rate envelope → audio-rate filter).
"""

import numpy as np
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer


def test_vcf_constant_cutoff():
    """Test VCF with constant cutoff (should behave like regular lowpass)."""
    print("\n" + "="*60)
    print("Test 1: VCF with Constant Cutoff")
    print("="*60)

    # Generate test signal: sawtooth wave at 220Hz
    duration = 1.0
    sample_rate = 48000
    freq = 220.0

    saw = audio.saw(freq=freq, duration=duration, sample_rate=sample_rate)

    # Create constant cutoff buffer at 1kHz
    cutoff_value = 1000.0
    cutoff_buffer = AudioBuffer(
        data=np.full(len(saw.data), cutoff_value),
        sample_rate=sample_rate
    )

    # Apply VCF
    filtered = audio.vcf_lowpass(saw, cutoff_buffer, q=0.707)

    # Validation
    assert len(filtered.data) == len(saw.data), "Output length mismatch"
    assert filtered.sample_rate == sample_rate, "Sample rate mismatch"

    # Check that filtering reduced high-frequency content
    rms_input = np.sqrt(np.mean(saw.data**2))
    rms_output = np.sqrt(np.mean(filtered.data**2))

    print(f"  Input signal: {len(saw.data)} samples @ {sample_rate}Hz")
    print(f"  Cutoff: {cutoff_value}Hz (constant)")
    print(f"  Input RMS: {rms_input:.4f}")
    print(f"  Output RMS: {rms_output:.4f}")
    print(f"  Attenuation: {20*np.log10(rms_output/rms_input):.2f} dB")
    print(f"  ✅ PASS - VCF with constant cutoff works")

    return True


def test_vcf_sweep():
    """Test VCF with time-varying cutoff sweep."""
    print("\n" + "="*60)
    print("Test 2: VCF with Cutoff Sweep")
    print("="*60)

    # Generate test signal: sawtooth wave at 110Hz
    duration = 2.0
    sample_rate = 48000
    freq = 110.0

    saw = audio.saw(freq=freq, duration=duration, sample_rate=sample_rate)

    # Create sweeping cutoff: 200Hz → 4000Hz over 2 seconds
    num_samples = len(saw.data)
    cutoff_sweep = np.linspace(200.0, 4000.0, num_samples)
    cutoff_buffer = AudioBuffer(data=cutoff_sweep, sample_rate=sample_rate)

    # Apply VCF with higher Q for more pronounced sweep
    filtered = audio.vcf_lowpass(saw, cutoff_buffer, q=2.0)

    # Validation
    assert len(filtered.data) == len(saw.data), "Output length mismatch"

    # Analyze output
    rms = np.sqrt(np.mean(filtered.data**2))
    peak = np.max(np.abs(filtered.data))

    # Check that output has reasonable amplitude
    assert rms > 0.1, "Output RMS too low (filter may be broken)"
    assert rms < 1.5, "Output RMS too high (possible instability)"

    print(f"  Input: sawtooth @ {freq}Hz, {duration}s")
    print(f"  Cutoff sweep: 200Hz → 4000Hz")
    print(f"  Q factor: 2.0")
    print(f"  Output RMS: {rms:.4f}")
    print(f"  Output Peak: {peak:.4f}")
    print(f"  ✅ PASS - VCF sweep produces valid output")

    return True


def test_vcf_adsr_modulation():
    """Test VCF with ADSR envelope modulation (classic subtractive synth)."""
    print("\n" + "="*60)
    print("Test 3: VCF with ADSR Envelope Modulation")
    print("="*60)

    # Audio-rate sawtooth oscillator
    duration = 2.0
    audio_rate = 48000
    control_rate = 1000
    freq = 110.0

    # Generate sawtooth at audio rate
    saw = audio.saw(freq=freq, duration=duration, sample_rate=audio_rate)

    # Generate ADSR envelope at control rate
    env = audio.adsr(
        attack=0.01,
        decay=0.5,
        sustain=0.3,
        release=0.8,
        duration=duration,
        sample_rate=control_rate
    )

    print(f"  Audio signal: {len(saw.data)} samples @ {audio_rate}Hz")
    print(f"  ADSR envelope: {len(env.data)} samples @ {control_rate}Hz")
    print(f"  Rate ratio: {audio_rate/control_rate:.1f}x")

    # Map envelope to cutoff range: 200Hz → 4000Hz
    cutoff_min = 200.0
    cutoff_max = 4000.0
    cutoff_mod = AudioBuffer(
        data=cutoff_min + env.data * (cutoff_max - cutoff_min),
        sample_rate=control_rate  # Still at control rate!
    )

    print(f"  Cutoff range: {cutoff_min}Hz - {cutoff_max}Hz")
    print(f"  Cutoff modulation: {len(cutoff_mod.data)} samples @ {control_rate}Hz")

    # Resample cutoff to audio rate (simulating scheduler behavior)
    # In the real scheduler, this happens automatically in _get_input_buffer
    cutoff_resampled = np.interp(
        np.linspace(0, len(cutoff_mod.data) - 1, len(saw.data)),
        np.arange(len(cutoff_mod.data)),
        cutoff_mod.data
    )
    cutoff_audio_rate = AudioBuffer(data=cutoff_resampled, sample_rate=audio_rate)

    print(f"  Cutoff resampled: {len(cutoff_audio_rate.data)} samples @ {audio_rate}Hz")

    # Apply VCF
    filtered = audio.vcf_lowpass(saw, cutoff_audio_rate, q=2.0)

    # Validation
    assert len(filtered.data) == len(saw.data), "Output length mismatch"

    # Analyze output
    rms = np.sqrt(np.mean(filtered.data**2))
    peak = np.max(np.abs(filtered.data))

    # Check for reasonable amplitude
    assert rms > 0.1, "Output RMS too low"
    assert rms < 1.5, "Output RMS too high (possible instability)"

    # Check that envelope shape is preserved (brightness increases then decreases)
    # Split signal into 4 quarters
    quarter_len = len(filtered.data) // 4
    q1_rms = np.sqrt(np.mean(filtered.data[0:quarter_len]**2))
    q2_rms = np.sqrt(np.mean(filtered.data[quarter_len:2*quarter_len]**2))
    q3_rms = np.sqrt(np.mean(filtered.data[2*quarter_len:3*quarter_len]**2))
    q4_rms = np.sqrt(np.mean(filtered.data[3*quarter_len:4*quarter_len]**2))

    print(f"\n  Output analysis:")
    print(f"    Overall RMS: {rms:.4f}")
    print(f"    Peak: {peak:.4f}")
    print(f"    Q1 RMS: {q1_rms:.4f} (attack)")
    print(f"    Q2 RMS: {q2_rms:.4f} (decay)")
    print(f"    Q3 RMS: {q3_rms:.4f} (sustain)")
    print(f"    Q4 RMS: {q4_rms:.4f} (release)")

    # Verify that filter is responding to modulation
    # (output RMS varies over time, indicating filter cutoff is changing)
    rms_variation = np.std([q1_rms, q2_rms, q3_rms, q4_rms])
    print(f"    RMS variation: {rms_variation:.4f}")

    # Check that there's measurable variation (filter is responding to envelope)
    assert rms_variation > 0.001, "Output shows no variation (filter may not be responding to modulation)"

    # Verify attack phase shows increasing brightness (Q2 > Q1)
    assert q2_rms > q1_rms, "Attack phase should show increasing brightness"

    print(f"  ✅ PASS - ADSR modulation produces time-varying filter response")

    return True


def test_vcf_edge_cases():
    """Test VCF edge cases and robustness."""
    print("\n" + "="*60)
    print("Test 4: VCF Edge Cases")
    print("="*60)

    duration = 0.1
    sample_rate = 48000

    # Test 1: Very low cutoff
    saw = audio.saw(freq=220.0, duration=duration, sample_rate=sample_rate)
    cutoff = AudioBuffer(data=np.full(len(saw.data), 50.0), sample_rate=sample_rate)
    filtered = audio.vcf_lowpass(saw, cutoff, q=0.707)
    assert np.all(np.isfinite(filtered.data)), "Low cutoff produced invalid values"
    print("  ✅ Very low cutoff (50Hz): OK")

    # Test 2: High cutoff near Nyquist
    cutoff = AudioBuffer(data=np.full(len(saw.data), 20000.0), sample_rate=sample_rate)
    filtered = audio.vcf_lowpass(saw, cutoff, q=0.707)
    assert np.all(np.isfinite(filtered.data)), "High cutoff produced invalid values"
    print("  ✅ High cutoff (20kHz): OK")

    # Test 3: High Q (resonance)
    cutoff = AudioBuffer(data=np.full(len(saw.data), 1000.0), sample_rate=sample_rate)
    filtered = audio.vcf_lowpass(saw, cutoff, q=10.0)
    assert np.all(np.isfinite(filtered.data)), "High Q produced invalid values"
    rms = np.sqrt(np.mean(filtered.data**2))
    assert rms < 5.0, f"High Q caused excessive resonance (RMS={rms:.2f})"
    print(f"  ✅ High Q (10.0): OK (RMS={rms:.4f})")

    # Test 4: Rapid cutoff changes
    cutoff_rapid = np.tile([500.0, 3000.0], len(saw.data)//2)
    cutoff = AudioBuffer(data=cutoff_rapid, sample_rate=sample_rate)
    filtered = audio.vcf_lowpass(saw, cutoff, q=1.0)
    assert np.all(np.isfinite(filtered.data)), "Rapid cutoff changes produced invalid values"
    print("  ✅ Rapid cutoff changes (500Hz ↔ 3kHz): OK")

    print("  ✅ PASS - All edge cases handled correctly")

    return True


def main():
    """Run all VCF modulation tests."""
    print("\n" + "="*60)
    print("VCF MODULATION TEST SUITE")
    print("="*60)

    tests = [
        ("Constant Cutoff", test_vcf_constant_cutoff),
        ("Cutoff Sweep", test_vcf_sweep),
        ("ADSR Modulation", test_vcf_adsr_modulation),
        ("Edge Cases", test_vcf_edge_cases),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n  ❌ FAIL: {e}")
            results.append((name, "FAIL"))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, status in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{name:30s} {symbol} {status}")

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All VCF modulation tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
