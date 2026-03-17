"""Canonical Example 02: Circuit → Audio

Guitar signal routed through two analog circuit models → distorted WAVs.

Demonstrates cross-domain composition: an audio signal drives a
circuit simulation; the circuit output is an audio signal.

1. Op-amp overdrive  — gain amplification + RC tone control
2. Tube Screamer     — Antiparallel diodes in feedback → smooth soft clipping

Output: output/02_circuit_clean.wav, 02_circuit_overdrive.wav, 02_circuit_ts.wav

Run: python examples/canonical/02_circuit_to_audio.py
"""

from pathlib import Path
import numpy as np

from morphogen.stdlib.circuit import CircuitOperations as circuit
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 48000
DURATION = 0.5  # seconds


def make_guitar_signal(sample_rate: int = SAMPLE_RATE, duration: float = DURATION) -> AudioBuffer:
    """Simulate a guitar pluck: decaying multi-harmonic waveform at A2 (110 Hz)."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 110.0
    attack = np.exp(-t * 8)
    sustain = np.exp(-t * 1.5)
    wave = (
        np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(2 * np.pi * f0 * 2 * t)
        + 0.25 * np.sin(2 * np.pi * f0 * 3 * t)
        + 0.1 * np.sin(2 * np.pi * f0 * 4 * t)
    ) * attack * sustain * 0.5  # ±0.5V peak
    return AudioBuffer(wave, sample_rate)


def make_overdrive_circuit(drive_gain: float = 8.0):
    """Op-amp inverting amplifier with tone-control RC filter on the output."""
    c = circuit.create(num_nodes=4, dt=1.0 / SAMPLE_RATE)
    c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=0.0, name="Vin")
    c = circuit.add_resistor(c, node1=1, node2=2, resistance=10000.0, name="Rin")
    c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")
    c = circuit.add_resistor(c, node1=2, node2=3, resistance=drive_gain * 10000.0, name="Rfb")
    c = circuit.add_resistor(c, node1=3, node2=0, resistance=10000.0, name="Rtone")
    c = circuit.add_capacitor(c, node1=3, node2=0, capacitance=10e-9, name="Ctone")
    return c


def make_tube_screamer_circuit(drive_gain: float = 8.0):
    """Tube Screamer: op-amp with antiparallel clipping diodes in feedback."""
    c = circuit.create(num_nodes=4, dt=1.0 / SAMPLE_RATE)
    c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=0.0, name="Vin")
    c = circuit.add_resistor(c, node1=1, node2=2, resistance=10000.0, name="Rin")
    c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")
    c = circuit.add_resistor(c, node1=3, node2=2, resistance=drive_gain * 10000.0, name="Rfb")
    # Antiparallel diodes clamp output to ~±0.6V when linear gain would exceed that
    c = circuit.add_diode(c, node_anode=3, node_cathode=2, name="D1")
    c = circuit.add_diode(c, node_anode=2, node_cathode=3, name="D2")
    return c


def main():
    print("=" * 60)
    print("CANONICAL EXAMPLE 02: Circuit → Audio")
    print("=" * 60)

    guitar_in = make_guitar_signal()
    print(f"Guitar signal: {SAMPLE_RATE}Hz, {DURATION}s, "
          f"peak={np.max(np.abs(guitar_in.data)):.3f}V")

    # --- Clean passthrough reference ---
    audio.save(guitar_in, str(OUTPUT_DIR / "02_circuit_clean.wav"))
    print(f"\n✓ Clean guitar: output/02_circuit_clean.wav")

    # --- Op-amp overdrive ---
    print("\nPart 1: Op-amp overdrive (gain=8x + RC tone roll-off)")
    overdrive_circ = make_overdrive_circuit(drive_gain=8.0)
    overdrive_out = circuit.process_audio(overdrive_circ, guitar_in,
                                          input_node=1, output_node=3,
                                          input_component="Vin")
    audio.save(overdrive_out, str(OUTPUT_DIR / "02_circuit_overdrive.wav"))
    linear_peak = 8.0 * np.max(np.abs(guitar_in.data))
    actual_peak = np.max(np.abs(overdrive_out.data))
    print(f"  Linear prediction: {linear_peak:.3f}V  Actual: {actual_peak:.3f}V")
    print(f"  ✓ Saved: output/02_circuit_overdrive.wav")

    # --- Tube Screamer soft clipper ---
    print("\nPart 2: Tube Screamer — soft clipping via antiparallel diodes")
    ts_circ = make_tube_screamer_circuit(drive_gain=8.0)
    ts_out = circuit.process_audio(ts_circ, guitar_in,
                                   input_node=1, output_node=3,
                                   input_component="Vin")
    audio.save(ts_out, str(OUTPUT_DIR / "02_circuit_ts.wav"))
    ts_peak = np.max(np.abs(ts_out.data))
    clamped = ts_peak < linear_peak * 0.7
    print(f"  Linear prediction: {linear_peak:.3f}V  Actual: {ts_peak:.3f}V  "
          f"Diodes clamped: {'YES — soft clipping active' if clamped else 'partial'}")
    print(f"  ✓ Saved: output/02_circuit_ts.wav")

    print("\nWhat happened:")
    print("  AudioBuffer drove a circuit simulation via process_audio()")
    print("  Op-amp: Kirchhoff equations solved at 48kHz, sample-by-sample")
    print("  Diodes: Newton-Raphson nonlinear solver activated at clipping threshold")
    print("  Each output is a real circuit response, not a DSP approximation")


if __name__ == "__main__":
    main()
