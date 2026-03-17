---
title: "Circuit Domain — Narrative Guide"
type: guide
beth_topics:
  - morphogen
  - circuit
  - mna
  - audio-processing
  - analog-simulation
  - spice
---

# Circuit Domain

The `circuit` domain is a SPICE-style analog circuit simulator: resistors, capacitors,
inductors, voltage/current sources, op-amps, and diodes. Analysis modes include DC
operating point, AC frequency sweep, transient time-domain, and real-time audio
processing.

## Quick start

```python
import numpy as np
from morphogen.stdlib.circuit import (
    create, add_resistor, add_capacitor, add_inductor,
    add_voltage_source, add_opamp, add_diode,
    dc_analysis, ac_analysis, transient_analysis,
    process_audio, get_node_voltage, get_branch_current,
)
```

Node numbering: `0` is always ground. Nodes `1, 2, ..., N−1` are internal nodes.
Create a circuit with `create(num_nodes=N)`.

---

## Recipe 1 — RC low-pass filter

```python
# RC filter: Vin → R → node 2 → C → GND
# Cutoff frequency: f_c = 1/(2π·R·C) ≈ 159 Hz
R = 1000.0    # 1 kΩ
C = 1e-6      # 1 μF

c = create(num_nodes=3)
c = add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="Vin")
c = add_resistor(c, 1, 2, R, "R1")
c = add_capacitor(c, 2, 0, C, "C1")

# DC operating point
c = dc_analysis(c)
v_out = get_node_voltage(c, 2)
print(f"DC V_out: {v_out:.2f} V")    # 5.0 V (cap blocks DC, full voltage across)

# AC frequency sweep: observe −3dB at f_c
freqs = np.logspace(1, 5, 200)     # 10 Hz – 100 kHz
ac = ac_analysis(c, freqs)
# ac["node_voltages"][i] is a complex array over frequencies, shape (N-1, len(freqs))
node_mags = np.abs(ac["node_voltages"])    # shape (N-1, 200) — magnitude per node
v2_mag = node_mags[1, :]                  # node 2 magnitudes (0-indexed from node 1)
f_3db_idx = np.argmin(np.abs(v2_mag - v2_mag[0] / 2**0.5))
print(f"−3dB frequency: {freqs[f_3db_idx]:.1f} Hz (expected ~{1/(2*np.pi*R*C):.1f} Hz)")
```

---

## Recipe 2 — Op-amp inverting amplifier

```python
# Classic inverting amplifier: gain = −R_f / R_in = −10
c = create(num_nodes=5)
c = add_voltage_source(c, 1, 0, 1.0, "Vin")   # 1V input
c = add_resistor(c, 1, 2, 10e3, "R_in")        # 10 kΩ input
c = add_resistor(c, 2, 3, 100e3, "R_f")        # 100 kΩ feedback
c = add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, gain=1e5, name="U1")

c = dc_analysis(c)
v_in  = get_node_voltage(c, 1)
v_out = get_node_voltage(c, 3)
print(f"Gain: {v_out/v_in:.2f}×  (expected −10)")
```

---

## Recipe 3 — Guitar distortion pedal (diode clipping)

A Tube Screamer-style clipping stage: input signal hard-clipped by a diode pair
via `process_audio`, which drives the circuit sample-by-sample.

```python
from morphogen.stdlib.audio import AudioBuffer, save_wav

SR = 44100
t = np.linspace(0, 1.0, SR)
# 220 Hz guitar-like tone
clean = (
    np.sin(2 * np.pi * 220 * t) * 0.5
    + np.sin(2 * np.pi * 440 * t) * 0.25
).astype(np.float32)
audio_in = AudioBuffer(clean, sample_rate=SR)

# Build clipping stage: Vin → R → diodes (anti-parallel) → GND
c = create(num_nodes=3, dt=1/SR)
c = add_voltage_source(c, 1, 0, 0.0, "Vin")     # driven by audio input
c = add_resistor(c, 1, 2, 4.7e3, "R_drive")
c = add_diode(c, 2, 0, Is=1e-14, n_factor=1.0, name="D1")  # forward
c = add_diode(c, 0, 2, Is=1e-14, n_factor=1.0, name="D2")  # reverse (anti-parallel)

audio_out = process_audio(c, audio_in, input_node=1, output_node=2)
save_wav(audio_out, "/tmp/distorted.wav")

peak_in  = float(np.abs(clean).max())
peak_out = float(np.abs(audio_out.samples).max())
print(f"input peak:  {peak_in:.3f} V")
print(f"output peak: {peak_out:.3f} V  (diodes clamping at ~0.6–0.7 V)")
```

See [`examples/canonical/02_circuit_to_audio.py`](/examples/canonical/02_circuit_to_audio.py)
for a fuller Tube Screamer model including the tone stack.

---

## Recipe 4 — Transient analysis

Watch a capacitor charge through a resistor in the time domain.

```python
# RC circuit: Vin=5V, R=1kΩ, C=10μF → τ = 10ms
c = create(num_nodes=3, dt=1e-5)
c = add_voltage_source(c, 1, 0, 5.0, "Vin")
c = add_resistor(c, 1, 2, 1e3, "R1")
c = add_capacitor(c, 2, 0, 10e-6, "C1")

# Simulate 50ms
times, voltages = transient_analysis(c, duration=50e-3)
# times: np.ndarray, shape (N,)
# voltages: np.ndarray, shape (N, num_nodes-1) — one column per non-ground node

tau = 1e3 * 10e-6   # 10 ms
v_tau_idx = np.argmin(np.abs(times - tau))
v_at_tau = voltages[v_tau_idx, 1]   # node 2 (0-indexed from node 1)
print(f"V at t=τ: {v_at_tau:.3f} V  (expected {5*(1-np.e**-1):.3f} V = 5·(1−1/e))")
```

---

## Recipe 5 — Measure frequency response (impedance)

```python
from morphogen.stdlib.circuit import get_impedance

c = create(num_nodes=3)
c = add_resistor(c, 1, 2, 1000.0, "R1")
c = add_capacitor(c, 2, 0, 1e-6, "C1")
c = add_voltage_source(c, 1, 0, 1.0, "Vin")

# |Z| of R+C at crossover
freq = 1 / (2 * np.pi * 1000.0 * 1e-6)   # f_c ≈ 159 Hz
Z = get_impedance(c, 1, 0, freq)
print(f"Impedance at f_c: |Z|={abs(Z):.1f} Ω  (expect {1000*2**0.5:.1f} = R·√2)")
```

---

## Full operator reference

| Operator | Returns | Notes |
|----------|---------|-------|
| `create(num_nodes, dt)` | `Circuit` | `num_nodes` includes ground (node 0) |
| `add_resistor(circuit, n1, n2, R, name)` | `Circuit` | Ω |
| `add_capacitor(circuit, n1, n2, C, name)` | `Circuit` | F |
| `add_inductor(circuit, n1, n2, L, name)` | `Circuit` | H |
| `add_voltage_source(circuit, n+, n-, V, name)` | `Circuit` | DC volts |
| `add_current_source(circuit, n+, n-, I, name)` | `Circuit` | DC amps |
| `add_opamp(circuit, n_in+, n_in-, n_out, gain, name)` | `Circuit` | Default gain=100 000 |
| `add_diode(circuit, anode, cathode, Is, n_factor, name)` | `Circuit` | Shockley model |
| `dc_analysis(circuit)` | `Circuit` | MNA DC operating point; stores node voltages |
| `ac_analysis(circuit, frequencies)` | `Dict[str, np.ndarray]` | Keys: `"frequencies"`, `"node_voltages"`, `"impedances"` |
| `transient_analysis(circuit, duration, method)` | `(times, voltages)` | Methods: `"backward_euler"` |
| `process_audio(circuit, audio_in, input_node, output_node, input_component)` | `AudioBuffer` | Sample-accurate audio |
| `get_node_voltage(circuit, node)` | `float` | After `dc_analysis` |
| `get_branch_current(circuit, component_name)` | `float` | After `dc_analysis` |
| `get_power(circuit, component_name)` | `float` | Watts |
| `get_impedance(circuit, n1, n2, frequency)` | `complex` | Ω at given Hz |

`Circuit` is immutable — every `add_*` and `*_analysis` call returns a new `Circuit`.

---

## See also

- [`examples/canonical/02_circuit_to_audio.py`](/examples/canonical/02_circuit_to_audio.py) — Tube Screamer overdrive → WAV
- [`examples/circuit/guitar_distortion_pedal.py`](/examples/circuit/guitar_distortion_pedal.py) — extended guitar pedal
- [`docs/specifications/circuit.md`](/docs/specifications/circuit.md) — MNA solver and diode model spec
- [`morphogen/stdlib/circuit.py`](/morphogen/stdlib/circuit.py) — source with full docstrings
