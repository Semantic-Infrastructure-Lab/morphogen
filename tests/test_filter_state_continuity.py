"""Test suite for filter state continuity across buffer hops.

This module validates that VCF filters (lowpass, highpass, bandpass) maintain
perfect state continuity across buffer boundaries using the filter_state parameter.

Tests verify that:
1. Filters with state produce continuous output across hops
2. Filters without state show discontinuities (baseline)
3. State preservation reduces discontinuities to negligible levels
4. All three filter types behave correctly

Pattern follows phase continuity testing from spectral-pegasus-1123.
"""

import numpy as np
import pytest
from morphogen.stdlib.audio import AudioBuffer, AudioOperations as audio


class TestFilterStateContinuity:
    """Test filter state continuity for seamless long-running synthesis."""

    def test_vcf_lowpass_without_state_shows_discontinuity(self):
        """Baseline: Filters without state show discontinuities at hop boundaries."""
        # Generate test signal: sawtooth wave
        signal_full = audio.saw(freq=220.0, duration=0.2, sample_rate=48000)

        # Generate constant cutoff as AudioBuffer
        cutoff_full = AudioBuffer(
            data=np.ones(len(signal_full.data)) * 1000.0,
            sample_rate=48000
        )

        # Filter in one continuous pass (reference)
        filtered_continuous = audio.vcf_lowpass(signal_full, cutoff_full, q=2.0)

        # Filter in two hops WITHOUT state preservation
        mid = len(signal_full.data) // 2

        # Hop 1: First half
        signal_hop1 = AudioBuffer(
            data=signal_full.data[:mid],
            sample_rate=48000
        )
        cutoff_hop1 = AudioBuffer(
            data=cutoff_full.data[:mid],
            sample_rate=48000
        )
        filtered_hop1 = audio.vcf_lowpass(signal_hop1, cutoff_hop1, q=2.0)

        # Hop 2: Second half (starts from zero state - discontinuity!)
        signal_hop2 = AudioBuffer(
            data=signal_full.data[mid:],
            sample_rate=48000
        )
        cutoff_hop2 = AudioBuffer(
            data=cutoff_full.data[mid:],
            sample_rate=48000
        )
        filtered_hop2 = audio.vcf_lowpass(signal_hop2, cutoff_hop2, q=2.0)

        # Concatenate hops
        filtered_hopped = np.concatenate([filtered_hop1.data, filtered_hop2.data])

        # Measure discontinuity at hop boundary
        boundary_discontinuity = abs(filtered_hop1.data[-1] - filtered_hop2.data[0])

        # Expect SIGNIFICANT discontinuity without state
        # (This establishes the baseline that we're fixing)
        assert boundary_discontinuity > 0.01, \
            f"Expected discontinuity > 0.01, got {boundary_discontinuity:.6f}"

        print(f"✅ Baseline discontinuity without state: {boundary_discontinuity:.6f}")

    def test_vcf_lowpass_with_state_provides_continuity(self):
        """Filter state eliminates discontinuities for seamless continuation."""
        # Generate test signal: sawtooth wave
        signal_full = audio.saw(freq=220.0, duration=0.2, sample_rate=48000)

        # Generate constant cutoff as AudioBuffer
        cutoff_full = AudioBuffer(
            data=np.ones(len(signal_full.data)) * 1000.0,
            sample_rate=48000
        )

        # Filter in one continuous pass (reference)
        filtered_continuous = audio.vcf_lowpass(signal_full, cutoff_full, q=2.0)

        # Filter in two hops WITH state preservation
        mid = len(signal_full.data) // 2

        # Create filter state buffer (2 elements for biquad state)
        filter_state = AudioBuffer(
            data=np.zeros(2),
            sample_rate=48000
        )

        # Hop 1: First half
        signal_hop1 = AudioBuffer(
            data=signal_full.data[:mid],
            sample_rate=48000
        )
        cutoff_hop1 = AudioBuffer(
            data=cutoff_full.data[:mid],
            sample_rate=48000
        )
        filtered_hop1 = audio.vcf_lowpass(
            signal_hop1, cutoff_hop1, q=2.0, filter_state=filter_state
        )
        # filter_state now contains final state from hop 1

        # Hop 2: Second half (continues from hop 1 state)
        signal_hop2 = AudioBuffer(
            data=signal_full.data[mid:],
            sample_rate=48000
        )
        cutoff_hop2 = AudioBuffer(
            data=cutoff_full.data[mid:],
            sample_rate=48000
        )
        filtered_hop2 = audio.vcf_lowpass(
            signal_hop2, cutoff_hop2, q=2.0, filter_state=filter_state
        )

        # Concatenate hops
        filtered_hopped = np.concatenate([filtered_hop1.data, filtered_hop2.data])

        # Measure discontinuity at hop boundary
        boundary_discontinuity = abs(filtered_hop1.data[-1] - filtered_hop2.data[0])

        # Verify overall signal matches continuous processing (key metric)
        max_error = np.max(np.abs(filtered_hopped - filtered_continuous.data))
        rms_error = np.sqrt(np.mean((filtered_hopped - filtered_continuous.data)**2))

        # With state, overall error should be minimal (numerical precision only)
        assert max_error < 1e-4, f"Max error too large: {max_error:.10f}"
        assert rms_error < 1e-5, f"RMS error too large: {rms_error:.10f}"

        # Boundary discontinuity should be greatly reduced vs without state
        # (May not be zero due to filter transients, but should be small)
        assert boundary_discontinuity < 0.1, \
            f"Expected boundary discontinuity < 0.1, got {boundary_discontinuity:.10f}"

        print(f"✅ With state - Boundary discontinuity: {boundary_discontinuity:.10f}")
        print(f"✅ Max error vs continuous: {max_error:.10f}")
        print(f"✅ RMS error vs continuous: {rms_error:.10f}")

    def test_vcf_highpass_state_continuity(self):
        """VCF highpass maintains state continuity across hops."""
        # Generate test signal
        signal_full = audio.saw(freq=220.0, duration=0.2, sample_rate=48000)

        # Constant cutoff
        cutoff_full = AudioBuffer(
            data=np.ones(len(signal_full.data)) * 800.0,
            sample_rate=48000
        )

        # Continuous reference
        filtered_continuous = audio.vcf_highpass(signal_full, cutoff_full, q=2.0)

        # Process in two hops with state
        mid = len(signal_full.data) // 2
        filter_state = AudioBuffer(data=np.zeros(2), sample_rate=48000)

        # Hop 1
        signal_hop1 = AudioBuffer(data=signal_full.data[:mid], sample_rate=48000)
        cutoff_hop1 = AudioBuffer(data=cutoff_full.data[:mid], sample_rate=48000)
        filtered_hop1 = audio.vcf_highpass(
            signal_hop1, cutoff_hop1, q=2.0, filter_state=filter_state
        )

        # Hop 2
        signal_hop2 = AudioBuffer(data=signal_full.data[mid:], sample_rate=48000)
        cutoff_hop2 = AudioBuffer(data=cutoff_full.data[mid:], sample_rate=48000)
        filtered_hop2 = audio.vcf_highpass(
            signal_hop2, cutoff_hop2, q=2.0, filter_state=filter_state
        )

        # Verify continuity
        filtered_hopped = np.concatenate([filtered_hop1.data, filtered_hop2.data])

        # Verify match to continuous (key metric)
        max_error = np.max(np.abs(filtered_hopped - filtered_continuous.data))
        rms_error = np.sqrt(np.mean((filtered_hopped - filtered_continuous.data)**2))

        assert max_error < 1e-4, f"Highpass max error: {max_error:.10f}"
        assert rms_error < 1e-5, f"Highpass RMS error: {rms_error:.10f}"

        boundary_discontinuity = abs(filtered_hop1.data[-1] - filtered_hop2.data[0])
        print(f"✅ VCF Highpass - Boundary discontinuity: {boundary_discontinuity:.10f}")
        print(f"✅ VCF Highpass - Max error: {max_error:.10f}")
        print(f"✅ VCF Highpass - RMS error: {rms_error:.10f}")

    def test_vcf_bandpass_state_continuity(self):
        """VCF bandpass maintains state continuity across hops."""
        # Generate test signal
        signal_full = audio.saw(freq=220.0, duration=0.2, sample_rate=48000)

        # Constant center frequency
        center_full = AudioBuffer(
            data=np.ones(len(signal_full.data)) * 1200.0,
            sample_rate=48000
        )

        # Continuous reference
        filtered_continuous = audio.vcf_bandpass(signal_full, center_full, q=5.0)

        # Process in two hops with state
        mid = len(signal_full.data) // 2
        filter_state = AudioBuffer(data=np.zeros(2), sample_rate=48000)

        # Hop 1
        signal_hop1 = AudioBuffer(data=signal_full.data[:mid], sample_rate=48000)
        center_hop1 = AudioBuffer(data=center_full.data[:mid], sample_rate=48000)
        filtered_hop1 = audio.vcf_bandpass(
            signal_hop1, center_hop1, q=5.0, filter_state=filter_state
        )

        # Hop 2
        signal_hop2 = AudioBuffer(data=signal_full.data[mid:], sample_rate=48000)
        center_hop2 = AudioBuffer(data=center_full.data[mid:], sample_rate=48000)
        filtered_hop2 = audio.vcf_bandpass(
            signal_hop2, center_hop2, q=5.0, filter_state=filter_state
        )

        # Verify continuity
        filtered_hopped = np.concatenate([filtered_hop1.data, filtered_hop2.data])

        # Verify match to continuous (key metric)
        max_error = np.max(np.abs(filtered_hopped - filtered_continuous.data))
        rms_error = np.sqrt(np.mean((filtered_hopped - filtered_continuous.data)**2))

        assert max_error < 1e-4, f"Bandpass max error: {max_error:.10f}"
        assert rms_error < 1e-5, f"Bandpass RMS error: {rms_error:.10f}"

        boundary_discontinuity = abs(filtered_hop1.data[-1] - filtered_hop2.data[0])
        print(f"✅ VCF Bandpass - Boundary discontinuity: {boundary_discontinuity:.10f}")
        print(f"✅ VCF Bandpass - Max error: {max_error:.10f}")
        print(f"✅ VCF Bandpass - RMS error: {rms_error:.10f}")

    def test_multiple_hops_accumulate_no_error(self):
        """State continuity works across many hops without error accumulation."""
        # Generate longer signal
        signal_full = audio.saw(freq=220.0, duration=1.0, sample_rate=48000)
        cutoff_full = AudioBuffer(
            data=np.ones(len(signal_full.data)) * 1000.0,
            sample_rate=48000
        )

        # Continuous reference
        filtered_continuous = audio.vcf_lowpass(signal_full, cutoff_full, q=2.0)

        # Process in 10 hops with state
        hop_size = len(signal_full.data) // 10
        filter_state = AudioBuffer(data=np.zeros(2), sample_rate=48000)

        filtered_hops = []
        for i in range(10):
            start = i * hop_size
            end = start + hop_size if i < 9 else len(signal_full.data)

            signal_hop = AudioBuffer(
                data=signal_full.data[start:end],
                sample_rate=48000
            )
            cutoff_hop = AudioBuffer(
                data=cutoff_full.data[start:end],
                sample_rate=48000
            )

            filtered_hop = audio.vcf_lowpass(
                signal_hop, cutoff_hop, q=2.0, filter_state=filter_state
            )
            filtered_hops.append(filtered_hop.data)

        # Concatenate all hops
        filtered_multi = np.concatenate(filtered_hops)

        # Verify no accumulated error
        max_error = np.max(np.abs(filtered_multi - filtered_continuous.data))
        rms_error = np.sqrt(np.mean((filtered_multi - filtered_continuous.data)**2))

        assert max_error < 1e-4, f"10-hop max error: {max_error:.10f}"
        assert rms_error < 1e-5, f"10-hop RMS error: {rms_error:.10f}"

        print(f"✅ 10 hops - Max error: {max_error:.10f}")
        print(f"✅ 10 hops - RMS error: {rms_error:.10f}")

    def test_time_varying_cutoff_with_state(self):
        """State continuity works with time-varying cutoff modulation."""
        # Generate test signal
        signal_full = audio.saw(freq=220.0, duration=0.4, sample_rate=48000)

        # Time-varying cutoff (sweep from 500Hz to 2000Hz)
        num_samples = len(signal_full.data)
        cutoff_sweep = np.linspace(500.0, 2000.0, num_samples)
        cutoff_full = AudioBuffer(data=cutoff_sweep, sample_rate=48000)

        # Continuous reference with sweep
        filtered_continuous = audio.vcf_lowpass(signal_full, cutoff_full, q=2.0)

        # Process in two hops with state
        mid = num_samples // 2
        filter_state = AudioBuffer(data=np.zeros(2), sample_rate=48000)

        # Hop 1
        signal_hop1 = AudioBuffer(data=signal_full.data[:mid], sample_rate=48000)
        cutoff_hop1 = AudioBuffer(data=cutoff_full.data[:mid], sample_rate=48000)
        filtered_hop1 = audio.vcf_lowpass(
            signal_hop1, cutoff_hop1, q=2.0, filter_state=filter_state
        )

        # Hop 2
        signal_hop2 = AudioBuffer(data=signal_full.data[mid:], sample_rate=48000)
        cutoff_hop2 = AudioBuffer(data=cutoff_full.data[mid:], sample_rate=48000)
        filtered_hop2 = audio.vcf_lowpass(
            signal_hop2, cutoff_hop2, q=2.0, filter_state=filter_state
        )

        # Verify continuity with sweep
        filtered_hopped = np.concatenate([filtered_hop1.data, filtered_hop2.data])

        # Verify match to continuous (key metric)
        max_error = np.max(np.abs(filtered_hopped - filtered_continuous.data))
        rms_error = np.sqrt(np.mean((filtered_hopped - filtered_continuous.data)**2))

        assert max_error < 1e-3, f"Sweep max error: {max_error:.10f}"
        assert rms_error < 1e-4, f"Sweep RMS error: {rms_error:.10f}"

        boundary_discontinuity = abs(filtered_hop1.data[-1] - filtered_hop2.data[0])
        print(f"✅ Time-varying cutoff - Boundary discontinuity: {boundary_discontinuity:.10f}")
        print(f"✅ Time-varying cutoff - Max error: {max_error:.10f}")
        print(f"✅ Time-varying cutoff - RMS error: {rms_error:.10f}")


def run_tests():
    """Run all filter state continuity tests."""
    print("\n" + "="*70)
    print("Filter State Continuity Test Suite")
    print("="*70 + "\n")

    test_suite = TestFilterStateContinuity()

    tests = [
        ("Baseline: Without state shows discontinuity",
         test_suite.test_vcf_lowpass_without_state_shows_discontinuity),
        ("VCF Lowpass with state provides continuity",
         test_suite.test_vcf_lowpass_with_state_provides_continuity),
        ("VCF Highpass state continuity",
         test_suite.test_vcf_highpass_state_continuity),
        ("VCF Bandpass state continuity",
         test_suite.test_vcf_bandpass_state_continuity),
        ("Multiple hops without error accumulation",
         test_suite.test_multiple_hops_accumulate_no_error),
        ("Time-varying cutoff with state",
         test_suite.test_time_varying_cutoff_with_state),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n{'─'*70}")
            print(f"Test: {name}")
            print('─'*70)
            test_func()
            passed += 1
            print(f"✅ PASSED\n")
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {e}\n")
        except Exception as e:
            failed += 1
            print(f"💥 ERROR: {e}\n")

    print("\n" + "="*70)
    print(f"Summary: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"❌ {failed} tests failed")
    else:
        print("✅ All tests passed!")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
