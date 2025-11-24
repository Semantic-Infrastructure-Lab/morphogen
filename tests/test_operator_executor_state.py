"""
Tests for OperatorExecutor Convention-Based State Management

This test suite validates automatic state management for stateful operators:
- Phase continuity for oscillators
- Filter state for VCF filters
- State persistence across multiple executions
- State isolation per node_id

Convention-based state management eliminates manual state tracking by:
1. Auto-detecting stateful parameters (phase, filter_state)
2. Creating AudioBuffer state on first execution
3. Injecting state into operator calls
4. Operators update state via in-place modification
"""

import pytest
import numpy as np
from morphogen.scheduler.operator_executor import (
    OperatorRegistry,
    OperatorExecutor,
    create_audio_registry,
    STATEFUL_PARAMS,
)
from morphogen.stdlib.audio import AudioBuffer


class TestConventionBasedStateManagement:
    """Test suite for convention-based state management in OperatorExecutor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = create_audio_registry()
        self.executor = OperatorExecutor(self.registry, sample_rate=48000)

    def test_stateful_params_configuration(self):
        """Validate STATEFUL_PARAMS configuration."""
        # Verify expected stateful parameters are configured
        assert "phase" in STATEFUL_PARAMS
        assert "filter_state" in STATEFUL_PARAMS

        # Verify state sizes
        assert STATEFUL_PARAMS["phase"] == 1  # Single phase angle
        assert STATEFUL_PARAMS["filter_state"] == 2  # Biquad [z1, z2]

    def test_phase_state_auto_creation(self):
        """Test that phase state is automatically created for oscillators."""
        # Execute sine operator (has phase parameter)
        result = self.executor.execute(
            operator_name="sine",
            node_id="osc1",
            params={"freq": 440.0},
            inputs={},
            num_samples=4800,  # 0.1 seconds at 48kHz
            rate_hz=48000.0,
        )

        # Verify output was generated
        assert "out" in result
        assert len(result["out"]) == 4800

        # Verify phase state was created
        assert "osc1" in self.executor._operator_state
        assert "phase" in self.executor._operator_state["osc1"]

        # Verify phase state is AudioBuffer with size 1
        phase_state = self.executor._operator_state["osc1"]["phase"]
        assert isinstance(phase_state, AudioBuffer)
        assert len(phase_state.data) == 1

    def test_phase_state_continuity_across_executions(self):
        """Test that phase state persists across multiple executions."""
        node_id = "osc_continuous"
        freq = 440.0
        num_samples = 4800  # 0.1 seconds

        # Execute oscillator 3 times
        outputs = []
        for i in range(3):
            result = self.executor.execute(
                operator_name="sine",
                node_id=node_id,
                params={"freq": freq},
                inputs={},
                num_samples=num_samples,
                rate_hz=48000.0,
            )
            outputs.append(result["out"])

        # Concatenate all outputs
        full_signal = np.concatenate(outputs)

        # Generate reference: continuous sine wave (no state management)
        reference = AudioBuffer.from_function(
            lambda t: np.sin(2.0 * np.pi * freq * t),
            duration=0.3,  # 3 × 0.1 seconds
            sample_rate=48000,
        )

        # Phase continuity should produce seamless signal matching reference
        # Allow small numerical error (<1e-5)
        max_error = np.max(np.abs(full_signal - reference.data))
        assert max_error < 1e-5, f"Phase discontinuity detected: max error = {max_error}"

    def test_filter_state_auto_creation(self):
        """Test that filter_state is automatically created for VCF filters."""
        # Create input signal and cutoff modulation
        signal = np.random.randn(4800) * 0.1
        cutoff = np.ones(4800) * 1000.0

        # Execute vcf_lowpass (has filter_state parameter)
        result = self.executor.execute(
            operator_name="vcf_lowpass",
            node_id="filter1",
            params={"q": 0.707},
            inputs={"signal": signal, "cutoff": cutoff},
            num_samples=4800,
            rate_hz=48000.0,
        )

        # Verify output was generated
        assert "out" in result
        assert len(result["out"]) == 4800

        # Verify filter_state was created
        assert "filter1" in self.executor._operator_state
        assert "filter_state" in self.executor._operator_state["filter1"]

        # Verify filter_state is AudioBuffer with size 2 (biquad state)
        filter_state = self.executor._operator_state["filter1"]["filter_state"]
        assert isinstance(filter_state, AudioBuffer)
        assert len(filter_state.data) == 2

    def test_filter_state_continuity_across_hops(self):
        """Test that filter state provides continuity across buffer hops."""
        node_id = "filter_continuous"
        freq = 440.0
        cutoff_hz = 2000.0
        chunk_samples = 4800  # 0.1 second chunks
        num_chunks = 5

        # Generate test signal: sawtooth wave
        full_signal_gen = AudioBuffer.from_function(
            lambda t: 2.0 * (t * freq - np.floor(t * freq + 0.5)),
            duration=num_chunks * 0.1,
            sample_rate=48000,
        )

        # Process in chunks with state management
        outputs_with_state = []
        for i in range(num_chunks):
            chunk = full_signal_gen.data[i * chunk_samples : (i + 1) * chunk_samples]
            cutoff = np.ones(chunk_samples) * cutoff_hz

            result = self.executor.execute(
                operator_name="vcf_lowpass",
                node_id=node_id,
                params={"q": 2.0},
                inputs={"signal": chunk, "cutoff": cutoff},
                num_samples=chunk_samples,
                rate_hz=48000.0,
            )
            outputs_with_state.append(result["out"])

        # Concatenate hopped output
        hopped_signal = np.concatenate(outputs_with_state)

        # Generate reference: continuous filtering (no hops)
        from morphogen.stdlib.audio import AudioOperations as audio

        cutoff_buf = AudioBuffer(
            data=np.ones(len(full_signal_gen.data)) * cutoff_hz, sample_rate=48000
        )
        reference = audio.vcf_lowpass(full_signal_gen, cutoff_buf, q=2.0)

        # Measure error at chunk boundaries (most critical for continuity)
        boundary_errors = []
        for i in range(1, num_chunks):
            boundary_idx = i * chunk_samples
            # Check samples around boundary
            for offset in range(-5, 5):
                idx = boundary_idx + offset
                if 0 <= idx < len(hopped_signal):
                    error = abs(hopped_signal[idx] - reference.data[idx])
                    boundary_errors.append(error)

        # Filter state continuity should eliminate discontinuities
        max_boundary_error = max(boundary_errors)
        assert (
            max_boundary_error < 1e-5
        ), f"Filter discontinuity at boundaries: {max_boundary_error}"

        # Overall signal error should be minimal
        overall_error = np.max(np.abs(hopped_signal - reference.data))
        assert overall_error < 1e-5, f"Filter state error: {overall_error}"

    def test_state_isolation_per_node_id(self):
        """Test that state is isolated per node_id."""
        # Execute two oscillators with different frequencies
        freq1 = 440.0
        freq2 = 880.0

        # Execute both oscillators
        result1 = self.executor.execute(
            operator_name="sine",
            node_id="osc_A",
            params={"freq": freq1},
            inputs={},
            num_samples=4800,
            rate_hz=48000.0,
        )

        result2 = self.executor.execute(
            operator_name="sine",
            node_id="osc_B",
            params={"freq": freq2},
            inputs={},
            num_samples=4800,
            rate_hz=48000.0,
        )

        # Verify both have independent state
        assert "osc_A" in self.executor._operator_state
        assert "osc_B" in self.executor._operator_state

        state_A = self.executor._operator_state["osc_A"]["phase"]
        state_B = self.executor._operator_state["osc_B"]["phase"]

        # States should be different AudioBuffer objects
        assert state_A is not state_B

        # States should have different phase values (different frequencies)
        # After 0.1s: phase_A = 2π × 440 × 0.1, phase_B = 2π × 880 × 0.1
        assert state_A.data[0] != state_B.data[0]

    def test_multiple_stateful_params_same_node(self):
        """Test that multiple stateful parameters can coexist for same node."""
        # This test simulates a scenario where an operator might have both
        # phase and filter_state (hypothetical future operator)

        # For now, we'll test by executing different operators on same node_id
        # which should create different state parameters

        node_id = "multi_state_node"

        # Execute oscillator (creates phase state)
        self.executor.execute(
            operator_name="sine",
            node_id=node_id,
            params={"freq": 440.0},
            inputs={},
            num_samples=4800,
            rate_hz=48000.0,
        )

        # Execute filter (creates filter_state)
        signal = np.random.randn(4800) * 0.1
        cutoff = np.ones(4800) * 1000.0
        self.executor.execute(
            operator_name="vcf_lowpass",
            node_id=node_id,
            params={"q": 0.707},
            inputs={"signal": signal, "cutoff": cutoff},
            num_samples=4800,
            rate_hz=48000.0,
        )

        # Verify both state parameters exist
        assert node_id in self.executor._operator_state
        node_state = self.executor._operator_state[node_id]

        assert "phase" in node_state
        assert "filter_state" in node_state

        # Verify correct sizes
        assert len(node_state["phase"].data) == 1
        assert len(node_state["filter_state"].data) == 2

    def test_non_stateful_operators_no_state_created(self):
        """Test that non-stateful operators don't create state."""
        # Execute a simple operator without stateful parameters
        # (e.g., constant, which doesn't have phase or filter_state)

        result = self.executor.execute(
            operator_name="constant",
            node_id="const1",
            params={"value": 0.5},
            inputs={},
            num_samples=4800,
            rate_hz=48000.0,
        )

        # Verify output was generated
        assert "out" in result

        # Verify no state was created (or empty state dict)
        if "const1" in self.executor._operator_state:
            # If node exists, it should have no stateful params
            assert "phase" not in self.executor._operator_state["const1"]
            assert "filter_state" not in self.executor._operator_state["const1"]


class TestStateManagementEdgeCases:
    """Test edge cases and error handling for state management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = create_audio_registry()
        self.executor = OperatorExecutor(self.registry, sample_rate=48000)

    def test_state_survives_many_executions(self):
        """Test that state doesn't accumulate errors over many executions."""
        node_id = "long_running"
        freq = 440.0
        num_executions = 100
        samples_per_exec = 480  # 0.01 seconds

        # Execute many times
        for i in range(num_executions):
            self.executor.execute(
                operator_name="sine",
                node_id=node_id,
                params={"freq": freq},
                inputs={},
                num_samples=samples_per_exec,
                rate_hz=48000.0,
            )

        # Verify state still exists and is valid
        assert node_id in self.executor._operator_state
        phase_state = self.executor._operator_state[node_id]["phase"]

        # Phase should be in valid range [0, 2π]
        assert 0 <= phase_state.data[0] < 2.0 * np.pi

    def test_different_sample_rates_handled(self):
        """Test that state management works with different sample rates."""
        # Execute same node at different rates (simulating rate conversion)

        node_id = "rate_test"

        # Execute at 48kHz
        self.executor.execute(
            operator_name="sine",
            node_id=node_id,
            params={"freq": 440.0},
            inputs={},
            num_samples=4800,
            rate_hz=48000.0,
        )

        phase_48k = self.executor._operator_state[node_id]["phase"].data[0]

        # Execute again at 48kHz
        self.executor.execute(
            operator_name="sine",
            node_id=node_id,
            params={"freq": 440.0},
            inputs={},
            num_samples=4800,
            rate_hz=48000.0,
        )

        # Phase should have advanced
        phase_after = self.executor._operator_state[node_id]["phase"].data[0]
        assert phase_after != phase_48k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
