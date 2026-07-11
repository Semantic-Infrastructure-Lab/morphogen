"""Task 1 — Circuit soft-clipper: guitar signal → diode clipping distortion.

The cross-domain operation is "audio drives a nonlinear analog circuit, circuit
output IS audio." This is Morphogen's STRONGEST case: it owns an integrated
cross-domain op (`circuit.process_audio`) backed by a real MNA + Newton-Raphson
diode solver.

Two implementations of the same task:
  - morphogen_version(): antiparallel-diode clipper via circuit.process_audio
  - glue_version():      the pragmatic raw-numpy approach a normal dev writes
                         (RC one-pole + tanh soft-clip approximation)

Returns (signal, label) so the runner can compare and score.
"""
import numpy as np

SR = 48000
DUR = 0.5


def _guitar(sr=SR, dur=DUR):
    t = np.linspace(0, dur, int(sr * dur))
    f0 = 110.0
    env = np.exp(-t * 8) * np.exp(-t * 1.5)
    wave = (np.sin(2 * np.pi * f0 * t)
            + 0.5 * np.sin(2 * np.pi * f0 * 2 * t)
            + 0.25 * np.sin(2 * np.pi * f0 * 3 * t)) * env * 0.5
    return wave


# ---- MORPHOGEN VERSION ------------------------------------------------------
def morphogen_version():
    from morphogen.stdlib.circuit import CircuitOperations as circuit
    from morphogen.stdlib.audio import AudioBuffer

    guitar = AudioBuffer(_guitar(), SR)
    c = circuit.create(num_nodes=4, dt=1.0 / SR)
    c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=0.0, name="Vin")
    c = circuit.add_resistor(c, node1=1, node2=2, resistance=10000.0, name="Rin")
    c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")
    c = circuit.add_resistor(c, node1=3, node2=2, resistance=80000.0, name="Rfb")
    c = circuit.add_diode(c, node_anode=3, node_cathode=2, name="D1")
    c = circuit.add_diode(c, node_anode=2, node_cathode=3, name="D2")
    out = circuit.process_audio(c, guitar, input_node=1, output_node=3, input_component="Vin")
    return np.asarray(out.data), "morphogen (real Shockley-diode Newton-Raphson clipper)"


# ---- RAW-GLUE VERSION -------------------------------------------------------
def glue_version():
    from scipy.signal import butter, lfilter

    guitar = _guitar()
    drive = 8.0
    clipped = np.tanh(drive * guitar)          # soft-clip APPROXIMATION (not a real diode)
    b, a = butter(1, 6000.0 / (SR / 2), btype="low")
    out = lfilter(b, a, clipped)               # RC tone roll-off approximation
    return out, "glue (tanh soft-clip + butterworth — a DSP approximation)"


if __name__ == "__main__":
    for fn in (morphogen_version, glue_version):
        sig, label = fn()
        print(f"{label}: peak={np.max(np.abs(sig)):.3f} rms={np.sqrt(np.mean(sig**2)):.4f} n={len(sig)}")
