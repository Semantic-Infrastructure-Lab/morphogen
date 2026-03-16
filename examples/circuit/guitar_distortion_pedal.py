"""Guitar Distortion Pedal - Circuit→Audio Integration Demo

This demonstrates the killer feature: processing audio through circuits!

Two circuits are demonstrated:
1. Op-amp overdrive with tone control capacitor (tests RC memory in process_audio)
2. Tube Screamer-style soft clipping: op-amp gain + antiparallel diodes in feedback
   The diodes clip the output softly above ~0.6V — the classic overdrive sound.

This is modeled after classic overdrive pedals like the Ibanez Tube Screamer.
"""

import numpy as np
from morphogen.stdlib.circuit import CircuitOperations as circuit
from morphogen.stdlib.audio import AudioOperations as audio


def create_distortion_circuit(drive_gain: float = 10.0) -> 'Circuit':
    """Create an op-amp overdrive with tone control (RC lowpass on output).

    Args:
        drive_gain: Distortion amount (1-100, higher = more distortion)

    Returns:
        Circuit configured as distortion pedal
    """
    # Node 0: Ground
    # Node 1: Input (from audio)
    # Node 2: Op-amp inverting input
    # Node 3: Op-amp output (distorted signal)

    c = circuit.create(num_nodes=4, dt=1.0/48000)  # 48kHz sample rate

    # Input voltage source (will be modulated by audio)
    c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=0.0, name="Vin")

    # Input resistor (10kΩ)
    c = circuit.add_resistor(c, node1=1, node2=2, resistance=10000.0, name="Rin")

    # Op-amp inverting amplifier (in+ grounded, in- to node 2, out to node 3)
    c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")

    # Feedback resistor determines gain: Gain = -Rfb/Rin
    r_feedback = drive_gain * 10000.0
    c = circuit.add_resistor(c, node1=2, node2=3, resistance=r_feedback, name="Rfb")

    # Tone control: lowpass RC filter on output (Rtone + Ctone = -3dB at ~1.6 kHz)
    c = circuit.add_resistor(c, node1=3, node2=0, resistance=10000.0, name="Rtone")
    c = circuit.add_capacitor(c, node1=3, node2=0, capacitance=10e-9, name="Ctone")

    return c


def create_tube_screamer_circuit(drive_gain: float = 10.0) -> 'Circuit':
    """Create a Tube Screamer-style soft clipping circuit.

    Topology: inverting op-amp with antiparallel diodes in PARALLEL with Rfb.
    With virtual ground at the inverting input (node 2 ≈ 0V), the diodes clamp
    the output swing to ≈ ±0.6V when the op-amp tries to exceed that, producing
    smooth soft clipping — the classic overdrive sound.

    Node 0: Ground
    Node 1: Input
    Node 2: Op-amp inverting input (virtual ground)
    Node 3: Op-amp output

    Rfb, D1, D2 are all connected between node 3 and node 2.

    Args:
        drive_gain: Base linear gain before diodes clamp (Rfb/Rin = drive_gain)

    Returns:
        Circuit configured as Tube Screamer soft clipper
    """
    c = circuit.create(num_nodes=4, dt=1.0/48000)

    # Input voltage source
    c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=0.0, name="Vin")

    # Input resistor
    c = circuit.add_resistor(c, node1=1, node2=2, resistance=10000.0, name="Rin")

    # Op-amp: in+ grounded, in- = node 2, out = node 3
    c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")

    # Feedback resistor in parallel with diodes (sets linear gain = Rfb/Rin)
    r_feedback = drive_gain * 10000.0
    c = circuit.add_resistor(c, node1=3, node2=2, resistance=r_feedback, name="Rfb")

    # Antiparallel clipping diodes in parallel with Rfb (node 3 ↔ node 2)
    # D1: anode=3, cathode=2 — conducts when V_out > +0.6V (clips positive)
    # D2: anode=2, cathode=3 — conducts when V_out < -0.6V (clips negative)
    c = circuit.add_diode(c, node_anode=3, node_cathode=2, name="D1")
    c = circuit.add_diode(c, node_anode=2, node_cathode=3, name="D2")

    return c


def main():
    """Demonstrate guitar distortion pedal: op-amp overdrive + Tube Screamer soft clipping."""
    print("=" * 60)
    print("GUITAR DISTORTION PEDAL - Circuit→Audio Integration")
    print("=" * 60)
    print()

    # Create test signal (simulated guitar pluck at A2 = 110 Hz)
    print("Creating guitar signal...")
    sample_rate = 48000
    duration = 0.1  # 100ms (faster demo)
    t = np.linspace(0, duration, int(sample_rate * duration))
    freq = 110.0

    attack = np.exp(-t * 10)
    sustain = np.exp(-t * 2)
    string_signal = (
        attack * np.sin(2 * np.pi * freq * t) * sustain
        + 0.3 * attack * np.sin(2 * np.pi * freq * 2 * t) * sustain
        + 0.1 * attack * np.sin(2 * np.pi * freq * 3 * t) * sustain
    )
    string_signal *= 0.5  # ±0.5V peak

    from morphogen.stdlib.audio import AudioBuffer
    guitar_in = AudioBuffer(string_signal, sample_rate)

    print(f"  Sample rate: {sample_rate} Hz | Duration: {duration}s | "
          f"Peak: {np.max(np.abs(string_signal)):.3f}V")
    print()

    # --- Part 1: Op-amp overdrive with tone control ---
    print("Part 1: Op-amp overdrive (tone control RC filter)")
    print("-" * 50)
    results = {}
    for drive_name, drive_gain in [("Clean", 2.0), ("Medium", 5.0), ("Heavy", 10.0)]:
        pedal = create_distortion_circuit(drive_gain=drive_gain)
        distorted = circuit.process_audio(pedal, guitar_in, input_node=1,
                                          output_node=3, input_component="Vin")
        results[drive_name] = distorted
        peak = np.max(np.abs(distorted.data))
        rms = np.sqrt(np.mean(distorted.data ** 2))
        print(f"  {drive_name:6s} (gain={drive_gain:4.1f}x)  peak={peak:.3f}V  rms={rms:.4f}V")

    print()

    # --- Part 2: Tube Screamer soft clipping ---
    print("Part 2: Tube Screamer soft clipping (antiparallel diodes in feedback)")
    print("-" * 50)
    print("  Diodes clamp feedback above ~0.6V → smooth harmonic distortion")
    print()
    ts_results = {}
    for drive_name, drive_gain in [("Low", 2.0), ("Mid", 5.0), ("High", 10.0)]:
        ts = create_tube_screamer_circuit(drive_gain=drive_gain)
        clipped = circuit.process_audio(ts, guitar_in, input_node=1,
                                        output_node=3, input_component="Vin")
        ts_results[drive_name] = clipped
        peak = np.max(np.abs(clipped.data))
        rms = np.sqrt(np.mean(clipped.data ** 2))
        linear_peak = drive_gain * np.max(np.abs(string_signal))
        diode_knee = 0.65  # ~0.6V diode forward voltage
        print(f"  {drive_name:4s} (gain={drive_gain:4.1f}x)  "
              f"peak={peak:.3f}V  linear_would_be={linear_peak:.3f}V  "
              f"clamped={'YES' if peak < linear_peak * 0.7 else 'partial'}")

    print()
    print("  Note: clamped=YES means diodes limited the output below linear prediction.")
    print()

    # Save outputs (if audio save is available)
    try:
        print("Saving audio files...")
        for name, buf in {**results, **{f"ts_{k}": v for k, v in ts_results.items()}}.items():
            filename = f"distortion_{name.lower()}.wav"
            audio.save(buf, filename)
            print(f"  Saved: {filename}")
        print()
    except (AttributeError, ImportError):
        print("  (Audio save not available in this version)")
        print()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Demonstrated:")
    print("  Op-amp overdrive with tone control (RC lowpass with memory)")
    print("  Tube Screamer: antiparallel diodes in feedback path")
    print("  Nonlinear Newton-Raphson solver activated by diodes")
    print("  Transient stepping: capacitor state maintained across samples")
    print()


if __name__ == "__main__":
    main()
