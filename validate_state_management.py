#!/usr/bin/env python3
"""
Validation script for convention-based state management in OperatorExecutor.

This script validates that the refactored state management works correctly
for both phase continuity and filter state.
"""

import numpy as np
import sys
from morphogen.scheduler.operator_executor import (
    OperatorRegistry,
    OperatorExecutor,
    create_audio_registry,
    EXECUTOR_MANAGED_STATE,
    OPERATOR_MANAGED_STATE,
)
from morphogen.stdlib.audio import AudioBuffer, AudioOperations as audio


def test_stateful_params_config():
    """Test 1: Validate state management configuration."""
    print("\n=== Test 1: State Management Configuration ===")

    assert "phase" in EXECUTOR_MANAGED_STATE, "phase not in EXECUTOR_MANAGED_STATE"
    assert "filter_state" in OPERATOR_MANAGED_STATE, "filter_state not in OPERATOR_MANAGED_STATE"
    assert EXECUTOR_MANAGED_STATE["phase"] == float, f"phase type wrong: {EXECUTOR_MANAGED_STATE['phase']}"
    assert OPERATOR_MANAGED_STATE["filter_state"] == 2, f"filter_state size wrong: {OPERATOR_MANAGED_STATE['filter_state']}"

    print("✅ State management correctly configured")
    print(f"   - EXECUTOR-MANAGED: phase (type: {EXECUTOR_MANAGED_STATE['phase'].__name__})")
    print(f"   - OPERATOR-MANAGED: filter_state ({OPERATOR_MANAGED_STATE['filter_state']} samples)")
    return True


def test_phase_state_auto_creation():
    """Test 2: Phase state auto-creation for oscillators."""
    print("\n=== Test 2: Phase State Auto-Creation ===")

    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    # Execute sine oscillator
    result = executor.execute(
        operator_name="sine",
        node_id="osc1",
        params={"freq": 440.0},
        inputs={},
        num_samples=4800,
        rate_hz=48000.0,
    )

    assert "out" in result, "No output generated"
    assert len(result["out"]) == 4800, f"Wrong output length: {len(result['out'])}"

    # Check state was created
    assert "osc1" in executor._operator_state, "No state created for osc1"
    assert "phase" in executor._operator_state["osc1"], "No phase state created"

    phase_state = executor._operator_state["osc1"]["phase"]
    assert isinstance(phase_state, float), f"Phase state not float: {type(phase_state)}"

    print("✅ Phase state automatically created")
    print(f"   - State type: {type(phase_state).__name__}")
    print(f"   - Phase value: {phase_state:.6f} radians")
    return True


def test_phase_continuity():
    """Test 3: Phase continuity across multiple executions."""
    print("\n=== Test 3: Phase Continuity Across Executions ===")

    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    node_id = "osc_continuous"
    freq = 440.0
    num_samples = 4800  # 0.1 seconds
    num_chunks = 3

    # Execute oscillator multiple times
    outputs = []
    for i in range(num_chunks):
        result = executor.execute(
            operator_name="sine",
            node_id=node_id,
            params={"freq": freq},
            inputs={},
            num_samples=num_samples,
            rate_hz=48000.0,
        )
        outputs.append(result["out"])
        print(f"   Chunk {i+1}: phase = {executor._operator_state[node_id]['phase']:.6f}")

    # Concatenate outputs
    full_signal = np.concatenate(outputs)

    # Generate reference: continuous sine
    t = np.arange(len(full_signal)) / 48000.0
    reference = np.sin(2.0 * np.pi * freq * t)

    # Measure continuity error
    max_error = np.max(np.abs(full_signal - reference))

    assert max_error < 1e-5, f"Phase discontinuity: max error = {max_error}"

    print(f"✅ Phase continuity validated")
    print(f"   - Chunks processed: {num_chunks}")
    print(f"   - Max error vs continuous: {max_error:.2e}")
    print(f"   - Continuity quality: {'EXCELLENT' if max_error < 1e-6 else 'GOOD'}")
    return True


def test_filter_state_auto_creation():
    """Test 4: Filter state auto-creation for VCF filters."""
    print("\n=== Test 4: Filter State Auto-Creation ===")

    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    # Create test signal
    signal = np.random.randn(4800) * 0.1
    cutoff = np.ones(4800) * 1000.0

    # Execute vcf_lowpass
    result = executor.execute(
        operator_name="vcf_lowpass",
        node_id="filter1",
        params={"q": 0.707},
        inputs={"signal": signal, "cutoff": cutoff},
        num_samples=4800,
        rate_hz=48000.0,
    )

    assert "out" in result, "No output generated"
    assert len(result["out"]) == 4800, f"Wrong output length: {len(result['out'])}"

    # Check filter_state was created
    assert "filter1" in executor._operator_state, "No state created for filter1"
    assert "filter_state" in executor._operator_state["filter1"], "No filter_state created"

    filter_state = executor._operator_state["filter1"]["filter_state"]
    assert isinstance(filter_state, AudioBuffer), "Filter state not AudioBuffer"
    assert len(filter_state.data) == 2, f"Filter state wrong size: {len(filter_state.data)}"

    print("✅ Filter state automatically created")
    print(f"   - State type: {type(filter_state).__name__}")
    print(f"   - State size: {len(filter_state.data)} samples (biquad [z1, z2])")
    print(f"   - State values: [{filter_state.data[0]:.6f}, {filter_state.data[1]:.6f}]")
    return True


def test_filter_state_continuity():
    """Test 5: Filter state continuity across buffer hops."""
    print("\n=== Test 5: Filter State Continuity Across Hops ===")

    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    node_id = "filter_continuous"
    freq = 440.0
    cutoff_hz = 2000.0
    chunk_samples = 4800
    num_chunks = 5

    # Generate test signal (sawtooth)
    t_full = np.arange(num_chunks * chunk_samples) / 48000.0
    full_signal = 2.0 * (t_full * freq - np.floor(t_full * freq + 0.5))

    # Process in chunks with state management
    outputs_with_state = []
    for i in range(num_chunks):
        chunk = full_signal[i * chunk_samples : (i + 1) * chunk_samples]
        cutoff = np.ones(chunk_samples) * cutoff_hz

        result = executor.execute(
            operator_name="vcf_lowpass",
            node_id=node_id,
            params={"q": 2.0},
            inputs={"signal": chunk, "cutoff": cutoff},
            num_samples=chunk_samples,
            rate_hz=48000.0,
        )
        outputs_with_state.append(result["out"])

        state = executor._operator_state[node_id]["filter_state"].data
        print(f"   Chunk {i+1}: filter_state = [{state[0]:.6f}, {state[1]:.6f}]")

    # Concatenate hopped output
    hopped_signal = np.concatenate(outputs_with_state)

    # Generate reference: continuous filtering
    signal_buf = AudioBuffer(data=full_signal, sample_rate=48000)
    cutoff_buf = AudioBuffer(data=np.ones(len(full_signal)) * cutoff_hz, sample_rate=48000)
    reference = audio.vcf_lowpass(signal_buf, cutoff_buf, q=2.0)

    # Measure continuity
    max_error = np.max(np.abs(hopped_signal - reference.data))

    assert max_error < 1e-5, f"Filter discontinuity: max error = {max_error}"

    print(f"✅ Filter state continuity validated")
    print(f"   - Chunks processed: {num_chunks}")
    print(f"   - Max error vs continuous: {max_error:.2e}")
    print(f"   - Continuity quality: {'EXCELLENT' if max_error < 1e-6 else 'GOOD'}")
    return True


def test_state_isolation():
    """Test 6: State isolation per node_id."""
    print("\n=== Test 6: State Isolation Per Node ID ===")

    registry = create_audio_registry()
    executor = OperatorExecutor(registry, sample_rate=48000)

    # Execute two oscillators with different frequencies
    executor.execute(
        operator_name="sine",
        node_id="osc_A",
        params={"freq": 440.0},
        inputs={},
        num_samples=4800,
        rate_hz=48000.0,
    )

    executor.execute(
        operator_name="sine",
        node_id="osc_B",
        params={"freq": 880.0},
        inputs={},
        num_samples=4800,
        rate_hz=48000.0,
    )

    # Check state isolation
    assert "osc_A" in executor._operator_state
    assert "osc_B" in executor._operator_state

    phase_A = executor._operator_state["osc_A"]["phase"]
    phase_B = executor._operator_state["osc_B"]["phase"]

    assert isinstance(phase_A, float), "Phase A should be float"
    assert isinstance(phase_B, float), "Phase B should be float"
    assert phase_A != phase_B, f"Phase values should differ: {phase_A} vs {phase_B}"

    print("✅ State isolation validated")
    print(f"   - osc_A phase: {phase_A:.6f} radians")
    print(f"   - osc_B phase: {phase_B:.6f} radians")
    print(f"   - States are independent: True")
    return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("CONVENTION-BASED STATE MANAGEMENT VALIDATION")
    print("=" * 70)

    tests = [
        test_stateful_params_config,
        test_phase_state_auto_creation,
        test_phase_continuity,
        test_filter_state_auto_creation,
        test_filter_state_continuity,
        test_state_isolation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Convention-based state management working perfectly!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
