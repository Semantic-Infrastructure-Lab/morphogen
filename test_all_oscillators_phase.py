#!/usr/bin/env python3
"""Test phase continuity for all oscillator types."""

import numpy as np
from morphogen.graph_ir import GraphIR, GraphIRNode, GraphIREdge, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler


def test_oscillator_phase_continuity(osc_type: str):
    """Test that an oscillator maintains phase continuity across hops.

    Args:
        osc_type: Oscillator type (sine, saw, square, triangle)

    Returns:
        True if phase continuity is maintained, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Test: {osc_type.upper()} Phase Continuity")
    print('='*60)

    # Create simple graph with oscillator
    graph = GraphIR(version="1.0")

    # Add oscillator node
    osc_node = GraphIRNode(
        id="osc1",
        op=osc_type,
        rate="audio",
        params={"freq": "440Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )
    graph.nodes.append(osc_node)

    # Set output
    graph.outputs = {"audio_out": ["osc1:out"]}

    # Validate graph
    errors = graph.validate()
    if errors:
        print(f"❌ Graph validation errors: {errors}")
        return False

    print("✅ Graph validated")

    # Create scheduler
    scheduler = SimplifiedScheduler(
        graph=graph,
        sample_rate=48000,
        hop_size=1024  # Small hops to test continuity
    )

    print(f"✅ Scheduler initialized (hop size: 1024 samples)")

    # Execute 3 hops
    num_hops = 3
    all_outputs = []

    for hop in range(num_hops):
        print(f"\nExecuting hop {hop + 1}/{num_hops}...")
        outputs = scheduler.execute(duration_samples=1024)
        audio_out = outputs["audio_out"]
        all_outputs.append(audio_out)
        print(f"  Generated {len(audio_out)} samples")
        print(f"  Last 3 samples: [{audio_out[-3]:.6f}, {audio_out[-2]:.6f}, {audio_out[-1]:.6f}]")

    # Concatenate all hops
    combined = np.concatenate(all_outputs)
    print(f"\n Total samples generated: {len(combined)}")

    # Check for discontinuities at hop boundaries
    print("\nChecking for discontinuities at hop boundaries...")

    discontinuity_threshold = 0.15  # Reasonable for 440Hz @ 48kHz (slightly higher for square)
    hop_boundaries = [1024, 2048]

    max_discontinuity = 0.0
    for boundary in hop_boundaries:
        if boundary < len(combined):
            diff = abs(combined[boundary] - combined[boundary-1])
            max_discontinuity = max(max_discontinuity, diff)
            status = "✅" if diff < discontinuity_threshold else "❌"
            print(f"  Hop boundary at sample {boundary}: diff = {diff:.6f} {status}")

    # Overall assessment
    print(f"\n Max discontinuity: {max_discontinuity:.6f}")

    if max_discontinuity < discontinuity_threshold:
        print(f"✅ PASS: {osc_type.upper()} phase continuity maintained!")
        return True
    else:
        print(f"❌ FAIL: {osc_type.upper()} phase discontinuities detected")
        return False


def main():
    """Run phase continuity tests for all oscillators."""
    print("\n" + "="*70)
    print("ALL OSCILLATORS PHASE CONTINUITY TEST")
    print("="*70)

    oscillators = ["sine", "saw", "square", "triangle"]
    results = {}

    for osc in oscillators:
        results[osc] = test_oscillator_phase_continuity(osc)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for osc, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{osc.ljust(10)}: {status}")
        all_passed = all_passed and passed

    print("="*70)
    if all_passed:
        print("✅ All oscillators maintain phase continuity!")
    else:
        print("❌ Some oscillators have phase discontinuities")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
