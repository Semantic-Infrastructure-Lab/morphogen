#!/usr/bin/env python3
"""
Simple VCA (Voltage Controlled Amplifier) Example

Demonstrates classic synthesizer VCA usage with ADSR envelope
using Morphogen's GraphIR + Scheduler infrastructure.
"""

import numpy as np
from morphogen.graph_ir import GraphIR, GraphIRNode, GraphIREdge, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler
from morphogen.stdlib.audio import AudioBuffer


def create_vca_graph():
    """Create a VCA synthesis graph with ADSR envelope.

    Graph structure:
        oscillator (saw 220Hz) ──┐
                                  ├─> VCA ──> output
        ADSR envelope ───────────┘
    """
    graph = GraphIR(version="1.0")

    # Oscillator: Sawtooth at 220Hz
    osc_node = GraphIRNode(
        id="osc",
        op="saw",
        rate="audio",
        params={"freq": "220Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )

    # ADSR Envelope
    env_node = GraphIRNode(
        id="env",
        op="adsr",
        rate="control",  # 1kHz control rate for efficiency
        params={
            "attack": "0.1s",
            "decay": "0.3s",
            "sustain": "0.6",
            "release": "0.5s"
        },
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )

    # VCA: Apply envelope to oscillator
    vca_node = GraphIRNode(
        id="vca",
        op="vca",
        rate="audio",
        params={"curve": "linear"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )

    # Add nodes to graph
    graph.nodes.extend([osc_node, env_node, vca_node])

    # Connect nodes
    graph.edges.extend([
        GraphIREdge(from_port="osc:out", to_port="vca:signal", type="Sig"),
        GraphIREdge(from_port="env:out", to_port="vca:cv", type="Sig"),
    ])

    # Set output
    graph.outputs = {"audio_out": ["vca:out"]}

    return graph


def main():
    """Run VCA example."""
    print("\n" + "="*70)
    print("MORPHOGEN VCA EXAMPLE")
    print("Voltage-Controlled Amplifier with ADSR Envelope")
    print("="*70)

    # Create graph
    graph = create_vca_graph()
    print(f"\nGraph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Execute
    print("\nExecuting graph...")
    scheduler = SimplifiedScheduler(
        graph=graph,
        sample_rate=48000,
        hop_size=1024  # ~21ms hops
    )
    duration_samples = 2 * 48000  # 2 seconds
    results = scheduler.execute(duration_samples=duration_samples)

    # Analyze results
    audio_out = results["audio_out"]
    vca_output = AudioBuffer(data=audio_out, sample_rate=48000)

    print(f"\n✅ Synthesis complete!")
    print(f"Generated {vca_output.num_samples} samples ({vca_output.duration:.2f}s)")
    print(f"Sample rate: {vca_output.sample_rate} Hz")

    # Analyze envelope sections
    sr = vca_output.sample_rate
    attack_region = vca_output.data[int(0.1*sr):int(0.2*sr)]
    sustain_region = vca_output.data[int(0.5*sr):int(1.0*sr)]
    release_region = vca_output.data[int(1.5*sr):int(1.7*sr)]

    attack_rms = np.sqrt(np.mean(attack_region ** 2))
    sustain_rms = np.sqrt(np.mean(sustain_region ** 2))
    release_rms = np.sqrt(np.mean(release_region ** 2))

    print(f"\nAmplitude by envelope phase:")
    print(f"  Attack phase RMS:  {attack_rms:.4f}")
    print(f"  Sustain phase RMS: {sustain_rms:.4f}")
    print(f"  Release phase RMS: {release_rms:.4f}")

    print("\n" + "="*70)
    print("Key Features:")
    print("  • VCA provides classic envelope-based amplitude control")
    print("  • ADSR envelope runs at 1kHz (control rate)")
    print("  • Oscillator runs at 48kHz (audio rate)")
    print("  • Scheduler automatically handles rate conversion")
    print("  • Clean envelope shaping with smooth attack/decay/release")
    print("="*70)


if __name__ == "__main__":
    main()
