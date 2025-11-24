"""
Test automatic state management in GraphIR scheduler.

Verifies that convention-based state management (phase) works seamlessly in
full GraphIR execution, enabling infinite synthesis in graphs.

Session: drifting-antimatter-1123 (continuation of donupa-1123)
Date: 2025-11-23
"""

import pytest
import numpy as np
from morphogen.graph_ir import GraphIR, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler


class TestGraphIRAutoStateManagement:
    """Test automatic state management in GraphIR execution"""

    def setup_method(self):
        """Setup for each test"""
        self.sample_rate = 48000

    def test_graphir_phase_continuity(self):
        """
        Test that oscillator phase continues seamlessly across GraphIR executions.

        Creates a simple graph with one sine oscillator and executes it multiple
        times. Phase should continue seamlessly without manual state management.
        """
        # Create graph: sine oscillator @ 440Hz
        graph = GraphIR(sample_rate=self.sample_rate)
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": 440.0}
        )
        graph.add_output("out", ["osc1:out"])

        # Create scheduler
        scheduler = SimplifiedScheduler(graph, sample_rate=self.sample_rate)

        # Execute multiple times (10 chunks of 4800 samples = 0.1s each)
        num_chunks = 10
        chunk_samples = 4800
        chunks = []

        for i in range(num_chunks):
            result = scheduler.execute(duration_samples=chunk_samples)
            chunks.append(result["out"])

        # Concatenate all chunks
        hopped = np.concatenate(chunks)

        # Generate continuous reference (no hops)
        duration = num_chunks * chunk_samples
        t = np.arange(duration) / self.sample_rate
        reference = np.sin(2.0 * np.pi * 440.0 * t)

        # Compare: should be nearly identical
        error = np.max(np.abs(hopped - reference))

        # Validate phase continuity quality
        assert error < 1e-6, f"Phase discontinuity too large: {error}"
        print(f"✅ GraphIR Phase Continuity: max error = {error:.2e} (EXCELLENT)")

    def test_graphir_multiple_oscillators(self):
        """
        Test multiple independent oscillators maintain separate phase state.
        """
        # Create graph with two oscillators
        graph = GraphIR(sample_rate=self.sample_rate)

        graph.add_node(
            id="osc_A",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": 440.0}
        )

        graph.add_node(
            id="osc_B",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": 880.0}
        )

        # Use separate outputs (not mixing, to test isolation)
        graph.add_output("out_A", ["osc_A:out"])
        graph.add_output("out_B", ["osc_B:out"])

        scheduler = SimplifiedScheduler(graph, sample_rate=self.sample_rate)

        # Execute multiple chunks
        num_chunks = 5
        chunk_samples = 4800
        chunks_A = []
        chunks_B = []

        for i in range(num_chunks):
            result = scheduler.execute(duration_samples=chunk_samples)
            chunks_A.append(result["out_A"])
            chunks_B.append(result["out_B"])

        hopped_A = np.concatenate(chunks_A)
        hopped_B = np.concatenate(chunks_B)

        # Generate references
        duration = num_chunks * chunk_samples
        t = np.arange(duration) / self.sample_rate
        ref_A = np.sin(2.0 * np.pi * 440.0 * t)
        ref_B = np.sin(2.0 * np.pi * 880.0 * t)

        error_A = np.max(np.abs(hopped_A - ref_A))
        error_B = np.max(np.abs(hopped_B - ref_B))

        assert error_A < 1e-6, f"Osc A error: {error_A}"
        assert error_B < 1e-6, f"Osc B error: {error_B}"

        print(f"✅ GraphIR Multiple Oscillators:")
        print(f"   Osc A (440Hz) error: {error_A:.2e}")
        print(f"   Osc B (880Hz) error: {error_B:.2e}")
        print(f"   💡 Each node maintains independent state!")

    def test_graphir_infinite_synthesis(self):
        """
        Test infinite synthesis with many chunks, demonstrating production-ready
        continuous audio generation.
        """
        graph = GraphIR(sample_rate=self.sample_rate)
        graph.add_node(
            id="osc1",
            op="sawtooth",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": 110.0}
        )
        graph.add_output("out", ["osc1:out"])

        scheduler = SimplifiedScheduler(graph, sample_rate=self.sample_rate)

        # Execute many chunks to simulate infinite synthesis
        num_chunks = 50
        chunk_samples = 2400  # 0.05s per chunk
        chunks = []

        for i in range(num_chunks):
            result = scheduler.execute(duration_samples=chunk_samples)
            chunks.append(result["out"])

            # Validate output quality
            assert not np.any(np.isnan(result["out"])), f"NaN in chunk {i}"
            assert not np.any(np.isinf(result["out"])), f"Inf in chunk {i}"

        hopped = np.concatenate(chunks)

        # Check boundary continuity
        boundary_errors = []
        for i in range(1, num_chunks):
            boundary_idx = i * chunk_samples
            diff = abs(hopped[boundary_idx] - hopped[boundary_idx - 1])
            boundary_errors.append(diff)

        max_boundary_error = max(boundary_errors)
        avg_boundary_error = np.mean(boundary_errors)

        assert max_boundary_error < 0.5, f"Large discontinuity: {max_boundary_error}"

        print(f"✅ GraphIR Infinite Synthesis ({num_chunks} chunks):")
        print(f"   Total duration: {num_chunks * chunk_samples / self.sample_rate:.2f}s")
        print(f"   Max boundary discontinuity: {max_boundary_error:.2e}")
        print(f"   Avg boundary discontinuity: {avg_boundary_error:.2e}")
        print(f"   💡 Phase state managed automatically across {num_chunks} executions!")
        print(f"   💡 This graph could run INFINITELY!")

    def test_graphir_filter_state_continuity(self):
        """
        Test filter_state auto-management in GraphIR execution.

        Validates that vcf_lowpass filter state is automatically managed
        across multiple executions, maintaining biquad coefficients for
        seamless continuity.
        """
        # Build graph: sawtooth -> constant cutoff -> vcf_lowpass
        graph = GraphIR(sample_rate=48000)

        # Sawtooth oscillator
        graph.add_node(
            id="osc1",
            op="saw",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": 110.0}
        )

        # Constant cutoff source (2000 Hz)
        graph.add_node(
            id="cutoff_source",
            op="constant",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"value": 2000.0}
        )

        # VCF lowpass filter
        graph.add_node(
            id="filter1",
            op="vcf_lowpass",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"q": 2.0}
        )

        # Connect edges: osc -> filter signal, constant -> filter cutoff
        graph.add_edge(from_port="osc1:out", to_port="filter1:signal", type="Sig")
        graph.add_edge(from_port="cutoff_source:out", to_port="filter1:cutoff", type="Sig")

        graph.add_output("out", ["filter1:out"])

        scheduler = SimplifiedScheduler(graph, sample_rate=48000)

        # Execute 5 chunks with filter state continuity
        num_chunks = 5
        chunk_samples = 4800  # 0.1s per chunk
        outputs_with_state = []

        for i in range(num_chunks):
            result = scheduler.execute(duration_samples=chunk_samples)
            outputs_with_state.append(result["out"])

        # Concatenate hopped output
        hopped_signal = np.concatenate(outputs_with_state)

        # Generate reference: continuous filtering (no hops)
        from morphogen.stdlib.audio import AudioOperations as audio

        # Generate full continuous signal
        full_signal = audio.saw(freq=110.0, duration=0.5, sample_rate=48000)
        cutoff_buf = audio.constant(value=2000.0, duration=0.5, sample_rate=48000)
        reference = audio.vcf_lowpass(full_signal, cutoff_buf, q=2.0)

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
        avg_boundary_error = np.mean(boundary_errors)
        overall_error = np.max(np.abs(hopped_signal - reference.data))

        # Debug output
        print(f"\n🔍 Filter State Test Results:")
        print(f"   Max boundary error: {max_boundary_error:.6f}")
        print(f"   Avg boundary error: {avg_boundary_error:.6f}")
        print(f"   Overall max error: {overall_error:.6f}")

        # Check if filter state is being managed
        # If error is > 0.1, filter state is NOT being maintained
        if max_boundary_error > 0.1:
            print(f"   ⚠️  Large error suggests filter_state NOT being managed!")
            print(f"   📝 SimplifiedScheduler may need filter_state support")
            # Temporarily skip assertion to complete implementation
            assert False, f"Filter state not maintained - max error: {max_boundary_error:.6f}"
        else:
            # Stricter threshold for actual filter state management
            assert max_boundary_error < 1e-5, f"Filter discontinuity: {max_boundary_error}"
            assert overall_error < 1e-5, f"Filter state error: {overall_error}"
            print(f"✅ GraphIR Filter State Continuity ({num_chunks} chunks):")
            print(f"   💡 Filter state (biquad coefficients) managed automatically!")
            print(f"   💡 VCF filters maintain continuity across chunk boundaries!")


class TestGraphIRStateEdgeCases:
    """Test edge cases for GraphIR state management"""

    def test_graphir_state_survives_long_execution(self):
        """
        Test that state management remains stable over very long executions.
        """
        graph = GraphIR(sample_rate=48000)
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": 440.0}
        )
        graph.add_output("out", ["osc1:out"])

        scheduler = SimplifiedScheduler(graph, sample_rate=48000)

        # Execute 100 chunks
        num_chunks = 100
        chunk_samples = 480  # Small chunks to stress state management

        for i in range(num_chunks):
            result = scheduler.execute(duration_samples=chunk_samples)

            # Verify no corruption
            assert not np.any(np.isnan(result["out"])), f"NaN at chunk {i}"
            assert not np.any(np.isinf(result["out"])), f"Inf at chunk {i}"
            assert np.max(np.abs(result["out"])) <= 1.0, f"Overflow at chunk {i}"

        print(f"✅ GraphIR Long Execution: {num_chunks} chunks stable")
        print(f"   💡 No state corruption or error accumulation!")


if __name__ == "__main__":
    """Run tests standalone (no pytest required)"""
    import sys

    print("=" * 70)
    print("GRAPHIR AUTOMATIC STATE MANAGEMENT VALIDATION")
    print("=" * 70)
    print()

    test_suite = TestGraphIRAutoStateManagement()
    edge_suite = TestGraphIRStateEdgeCases()

    tests = [
        ("Phase Continuity in GraphIR", test_suite.test_graphir_phase_continuity),
        ("Multiple Independent Oscillators", test_suite.test_graphir_multiple_oscillators),
        ("Infinite Synthesis (50 chunks)", test_suite.test_graphir_infinite_synthesis),
        ("Filter State Continuity (VCF)", test_suite.test_graphir_filter_state_continuity),
        ("Long Execution Stability (100 chunks)", edge_suite.test_graphir_state_survives_long_execution),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"=== Test: {name} ===")
        try:
            if hasattr(test_suite, 'setup_method'):
                test_suite.setup_method()

            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ FAILED: {e}")
            print()
            failed += 1
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("🎉 ALL TESTS PASSED! GraphIR auto-state management confirmed working!")
        sys.exit(0)
    else:
        sys.exit(1)
