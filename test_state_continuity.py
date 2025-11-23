#!/usr/bin/env python3
"""Test suite for operator state continuity.

Tests that stateful operators (oscillators, filters) maintain continuous
state across scheduler hops for seamless long-running synthesis.
"""

import numpy as np
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer


def test_phase_discontinuity_problem():
    """
    Demonstrate the phase discontinuity problem with current implementation.

    When oscillators are called multiple times with duration parameter,
    each call starts from phase=0, creating discontinuities.
    """
    print("\n" + "="*60)
    print("Test 1: Phase Discontinuity Problem (Current Behavior)")
    print("="*60)

    freq = 440.0
    sample_rate = 48000
    hop_size = 1024  # Small hops to emphasize discontinuity

    # Generate audio in 3 hops
    hop1 = audio.sine(freq=freq, duration=hop_size/sample_rate, sample_rate=sample_rate)
    hop2 = audio.sine(freq=freq, duration=hop_size/sample_rate, sample_rate=sample_rate)
    hop3 = audio.sine(freq=freq, duration=hop_size/sample_rate, sample_rate=sample_rate)

    # Check phase at hop boundaries
    # Phase should be continuous, but it isn't with current implementation

    # Last sample of hop1 should match first sample of hop2 (if continuous)
    print(f"\n  Hop 1 last sample: {hop1.data[-1]:.6f}")
    print(f"  Hop 2 first sample: {hop2.data[0]:.6f}")
    print(f"  Discontinuity: {abs(hop2.data[0] - hop1.data[-1]):.6f}")

    # Expected phase at end of hop1
    phase_end_hop1 = (2 * np.pi * freq * hop_size / sample_rate) % (2 * np.pi)
    expected_hop2_start = np.sin(phase_end_hop1)

    print(f"\n  Expected hop2 start (continuous phase): {expected_hop2_start:.6f}")
    print(f"  Actual hop2 start: {hop2.data[0]:.6f}")
    print(f"  Phase reset error: {abs(hop2.data[0] - expected_hop2_start):.6f}")

    # Concatenate hops
    combined = np.concatenate([hop1.data, hop2.data, hop3.data])

    # Check for discontinuities (large sample-to-sample jumps)
    diffs = np.abs(np.diff(combined))
    max_diff = np.max(diffs)
    discontinuity_threshold = 0.1  # Reasonable for 440Hz @ 48kHz

    discontinuities = np.where(diffs > discontinuity_threshold)[0]

    print(f"\n  Total samples: {len(combined)}")
    print(f"  Max sample-to-sample diff: {max_diff:.6f}")
    print(f"  Discontinuities found: {len(discontinuities)}")

    if len(discontinuities) > 0:
        print(f"  Discontinuity locations: {discontinuities[:5]}...")  # Show first 5
        print(f"\n  ⚠️  PROBLEM IDENTIFIED: Phase resets at hop boundaries")
        return False
    else:
        print(f"\n  ✅ No discontinuities (unexpected!)")
        return True


def test_filter_state_discontinuity():
    """
    Demonstrate filter state discontinuity problem.

    IIR filters have internal state that should be preserved across hops.
    """
    print("\n" + "="*60)
    print("Test 2: Filter State Discontinuity Problem")
    print("="*60)

    # Generate continuous saw wave
    saw = audio.saw(freq=220.0, duration=0.1, sample_rate=48000)

    # Split into 3 chunks
    chunk_size = len(saw.data) // 3
    chunk1 = AudioBuffer(data=saw.data[0:chunk_size], sample_rate=48000)
    chunk2 = AudioBuffer(data=saw.data[chunk_size:2*chunk_size], sample_rate=48000)
    chunk3 = AudioBuffer(data=saw.data[2*chunk_size:3*chunk_size], sample_rate=48000)

    # Filter each chunk independently (simulating hop-by-hop execution)
    cutoff = AudioBuffer(data=np.full(chunk_size, 1000.0), sample_rate=48000)

    filtered1 = audio.lowpass(chunk1, cutoff=1000.0, q=2.0)
    filtered2 = audio.lowpass(chunk2, cutoff=1000.0, q=2.0)
    filtered3 = audio.lowpass(chunk3, cutoff=1000.0, q=2.0)

    # Combine filtered chunks
    combined_chunked = np.concatenate([filtered1.data, filtered2.data, filtered3.data])

    # Filter entire signal at once (ideal behavior with state continuity)
    filtered_continuous = audio.lowpass(saw, cutoff=1000.0, q=2.0)

    # Compare
    diff = np.abs(combined_chunked - filtered_continuous.data[:len(combined_chunked)])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n  Chunk size: {chunk_size} samples")
    print(f"  Chunked filtering (3 independent calls):")
    print(f"    Max diff from continuous: {max_diff:.6f}")
    print(f"    Mean diff: {mean_diff:.6f}")

    # Check for discontinuities at chunk boundaries
    discontinuities = []
    for boundary in [chunk_size, 2*chunk_size]:
        if boundary < len(combined_chunked) - 1:
            diff_at_boundary = abs(combined_chunked[boundary] - combined_chunked[boundary-1])
            discontinuities.append((boundary, diff_at_boundary))

    print(f"\n  Discontinuities at chunk boundaries:")
    for loc, diff_val in discontinuities:
        print(f"    Sample {loc}: diff = {diff_val:.6f}")

    if max_diff > 0.01:
        print(f"\n  ⚠️  PROBLEM: Filter state not preserved across chunks")
        print(f"     Chunked output differs from continuous filtering")
        return False
    else:
        print(f"\n  ✅ Filter state preserved (unexpected!)")
        return True


def test_desired_stateful_behavior():
    """
    Describe the desired behavior for stateful operators.

    This test documents the API we want to implement.
    """
    print("\n" + "="*60)
    print("Test 3: Desired Stateful Behavior (Not Yet Implemented)")
    print("="*60)

    print("""
  Desired API for stateful oscillators:

  # Hop 1
  output1, state1 = audio.sine_stateful(
      freq=440.0,
      num_samples=1024,
      phase=0.0,  # Initial phase
      sample_rate=48000
  )

  # Hop 2 (continues from hop 1)
  output2, state2 = audio.sine_stateful(
      freq=440.0,
      num_samples=1024,
      phase=state1['phase'],  # Use final phase from hop 1
      sample_rate=48000
  )

  Result: output1 and output2 are phase-continuous

  Similarly for filters:

  # Hop 1
  output1, state1 = audio.lowpass_stateful(
      signal=chunk1,
      cutoff=1000.0,
      q=2.0,
      state=None  # Initial state (zeros)
  )

  # Hop 2
  output2, state2 = audio.lowpass_stateful(
      signal=chunk2,
      cutoff=1000.0,
      q=2.0,
      state=state1  # Continue with state from hop 1
  )

  Result: output1 and output2 are state-continuous (no IIR discontinuity)
    """)

    print("  📋 This test documents requirements, not yet implemented")
    return None


def main():
    """Run state continuity tests."""
    print("\n" + "="*70)
    print("OPERATOR STATE CONTINUITY TEST SUITE")
    print("="*70)
    print("\nPurpose: Identify phase/state discontinuities in current implementation")
    print("and define requirements for state management system.")

    tests = [
        ("Phase Discontinuity", test_phase_discontinuity_problem),
        ("Filter State Discontinuity", test_filter_state_discontinuity),
        ("Desired Behavior", test_desired_stateful_behavior),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            if result is None:
                results.append((name, "SPEC"))
            elif result:
                results.append((name, "PASS"))
            else:
                results.append((name, "FAIL"))
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            results.append((name, "ERROR"))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for name, status in results:
        symbol = {
            "PASS": "✅",
            "FAIL": "⚠️ ",
            "SPEC": "📋",
            "ERROR": "❌"
        }[status]
        print(f"{name:35s} {symbol} {status}")

    problems = sum(1 for _, status in results if status == "FAIL")

    if problems > 0:
        print(f"\n⚠️  {problems} problem(s) identified - state management needed")
    else:
        print(f"\n✅ All tests passed")

    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Implement stateful operator variants (sine_stateful, etc.)")
    print("  2. Update OperatorExecutor to manage state per node_id")
    print("  3. Test long-running synthesis with state continuity")
    print("="*70)


if __name__ == "__main__":
    main()
