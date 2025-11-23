#!/usr/bin/env python3
"""Test phase continuity with state management."""

import numpy as np
from morphogen.graph_ir import GraphIR, GraphIRNode, GraphIREdge, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler


def test_sine_phase_continuity():
    """Test that sine oscillator maintains phase continuity across hops."""
    print("\n" + "="*60)
    print("Test: Sine Phase Continuity with State Management")
    print("="*60)

    # Create simple graph with sine oscillator
    graph = GraphIR(version="1.0")

    # Add sine node
    sine_node = GraphIRNode(
        id="osc1",
        op="sine",
        rate="audio",
        params={"freq": "440Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )
    graph.nodes.append(sine_node)

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

    discontinuity_threshold = 0.1  # Reasonable for 440Hz @ 48kHz
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
        print("✅ PASS: Phase continuity maintained across hops!")
        return True
    else:
        print("❌ FAIL: Phase discontinuities detected")
        return False


def main():
    """Run phase continuity test."""
    print("\n" + "="*70)
    print("PHASE CONTINUITY TEST")
    print("="*70)

    result = test_sine_phase_continuity()

    print("\n" + "="*70)
    if result:
        print("✅ Phase continuity working!")
    else:
        print("❌ Phase continuity broken")
    print("="*70)

    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
