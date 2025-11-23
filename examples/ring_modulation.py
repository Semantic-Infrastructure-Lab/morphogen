#!/usr/bin/env python3
"""Ring modulation synthesis example using multiply operator.

Demonstrates:
- multiply operator for ring modulation
- GraphIR graph construction
- SimplifiedScheduler execution
"""

import numpy as np
from morphogen.graph_ir import GraphIR, GraphIRNode, GraphIREdge, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler


def create_ring_mod_graph():
    """Create a ring modulation synthesis graph.

    Graph structure:
        carrier (sine 440Hz) ──┐
                                ├─> multiply ──> output
        modulator (sine 100Hz) ─┘

    Ring modulation produces sum and difference frequencies:
    440Hz + 100Hz = 540Hz
    440Hz - 100Hz = 340Hz
    """
    graph = GraphIR(version="1.0")

    # Carrier oscillator (440Hz)
    carrier_node = GraphIRNode(
        id="carrier",
        op="sine",
        rate="audio",
        params={"freq": "440Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )

    # Modulator oscillator (100Hz)
    modulator_node = GraphIRNode(
        id="modulator",
        op="sine",
        rate="audio",
        params={"freq": "100Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )

    # Multiply operator (ring modulation)
    multiply_node = GraphIRNode(
        id="ring_mod",
        op="multiply",
        rate="audio",
        params={"gain": "0.5"},  # Reduce gain to prevent clipping
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )

    # Add nodes
    graph.nodes.extend([carrier_node, modulator_node, multiply_node])

    # Connect carrier and modulator to multiply
    graph.edges.extend([
        GraphIREdge(from_port="carrier:out", to_port="ring_mod:signal1", type="Sig"),
        GraphIREdge(from_port="modulator:out", to_port="ring_mod:signal2", type="Sig"),
    ])

    # Set output
    graph.outputs = {"audio_out": ["ring_mod:out"]}

    return graph


def main():
    """Run ring modulation example."""
    print("\n" + "="*70)
    print("RING MODULATION SYNTHESIS EXAMPLE")
    print("="*70)

    # Create graph
    print("\n1. Creating ring modulation graph...")
    graph = create_ring_mod_graph()

    # Validate
    errors = graph.validate()
    if errors:
        print(f"❌ Graph validation errors: {errors}")
        return 1

    print("   ✅ Graph validated")
    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Edges: {len(graph.edges)}")

    # Create scheduler
    print("\n2. Creating scheduler...")
    scheduler = SimplifiedScheduler(
        graph=graph,
        sample_rate=48000,
        hop_size=4800  # 0.1 second hops
    )
    print("   ✅ Scheduler initialized")

    # Execute synthesis
    print("\n3. Executing ring modulation synthesis...")
    duration_samples = 48000  # 1 second
    outputs = scheduler.execute(duration_samples=duration_samples)

    audio_out = outputs["audio_out"]
    print(f"   Generated {len(audio_out)} samples ({len(audio_out)/48000:.2f}s)")

    # Analyze output
    print("\n4. Analyzing output...")
    rms = np.sqrt(np.mean(audio_out ** 2))
    peak = np.max(np.abs(audio_out))

    print(f"   RMS level: {rms:.4f}")
    print(f"   Peak level: {peak:.4f}")
    print(f"   Dynamic range: {20 * np.log10(peak/rms) if rms > 0 else 0:.2f} dB")

    # FFT analysis to show frequency content
    fft = np.fft.rfft(audio_out)
    freqs = np.fft.rfftfreq(len(audio_out), 1/48000)
    magnitudes = np.abs(fft)

    # Find peaks
    threshold = np.max(magnitudes) * 0.1
    peaks = []
    for i in range(1, len(magnitudes)-1):
        if magnitudes[i] > threshold and magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
            peaks.append((freqs[i], magnitudes[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    print("\n5. Frequency analysis (top peaks):")
    print(f"   Expected: 340Hz (440-100), 540Hz (440+100)")
    for freq, mag in peaks[:5]:
        print(f"   {freq:6.1f}Hz: magnitude {mag:8.0f}")

    print("\n" + "="*70)
    print("✅ Ring modulation synthesis complete!")
    print("="*70)
    print("\nRing modulation creates inharmonic sidebands by multiplying two")
    print("oscillators. The carrier (440Hz) and modulator (100Hz) produce:")
    print("  - Sum frequency: 440 + 100 = 540Hz")
    print("  - Difference frequency: 440 - 100 = 340Hz")
    print("\nThis classic effect is used in synthesizers and audio processing.")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
