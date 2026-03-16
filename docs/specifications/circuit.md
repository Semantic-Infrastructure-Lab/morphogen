# ⚡ Morphogen.Circuit Specification v1.0

**A declarative circuit simulation and analog modeling dialect built on the Morphogen kernel.**

**Inspired by SPICE, nodal analysis, and modern circuit simulation techniques.**

---

## 0. Overview

Morphogen.Circuit is a typed, declarative circuit simulation dialect layered on the Morphogen kernel.
It provides deterministic semantics, electrical component primitives, and composable circuit constructs.
It is an intermediate layer that sits between:

- **User applications** — Circuit design, PCB layout, analog audio modeling, power electronics, RF design
- **Morphogen Core** — the deterministic MLIR-based execution kernel
- **Backend engines** — ngspice, custom ODE solvers, GPU-accelerated linear algebra

**Unique capabilities:**
- Unified circuit + PCB geometry + audio modeling
- Multi-physics coupling (electromagnetic, thermal, mechanical)
- Cross-domain flows (circuit → audio, pattern → circuit, ML → circuit optimization)

---

## 1. Language Philosophy

| Principle | Meaning |
|-----------|---------|
| **Reference-based composition** | Components connected via typed ports, not manual node numbering. |
| **Deterministic simulation** | Same netlist + same inputs = same results. |
| **Typed components** | Every component has explicit electrical type (resistor, capacitor, op-amp, etc.). |
| **Multi-rate scheduling** | DC, AC, transient, harmonic balance all supported. |
| **Physics-accurate** | Modified nodal analysis (MNA), Newton-Raphson for nonlinear components. |
| **Cross-domain integration** | Circuit integrates with Geometry (PCB), Audio (DSP), Physics (thermal), ML (optimization). |
| **Unit safety** | Ohms, Farads, Henrys, Volts, Amperes tracked at compile time. |

**Key insight:** Circuits are **typed operator graphs** with **electrical constraints** (KCL, KVL).

---

## 2. Core Types

All circuit types are defined in the kernel's type system with explicit electrical semantics.

| Type | Description | Units | Examples |
|------|-------------|-------|----------|
| `Component` | Electrical component (R, L, C, etc.) | Varies | Resistor, Capacitor, OpAmp |
| `Net` | Electrical connection (wire) | V (voltage), A (current) | Power rail, ground, signal |
| `Node` | Circuit node (KCL junction) | V (voltage) | Junction of 3+ components |
| `Port` | Typed component terminal | V, A | `resistor.port["p"]`, `opamp.port["out"]` |
| `Signal<T, domain>` | Time-varying electrical quantity | V(t), A(t) | Voltage waveform, current waveform |
| `CircuitRef` | Unified reference to nodes/components/nets | Meta | See §4 |
| `StateVector` | Circuit operating point | V, A | DC solution, initial conditions |
| `FrequencyResponse` | AC analysis result | dB, phase | Bode plot, transfer function |
| `TimeSeries<T>` | Transient analysis result | V(t), A(t) | Oscilloscope trace |

**Electrical Units:**
- **Resistance:** `Ω` (Ohm), `kΩ`, `MΩ`
- **Capacitance:** `F` (Farad), `µF`, `nF`, `pF`
- **Inductance:** `H` (Henry), `mH`, `µH`, `nH`
- **Voltage:** `V` (Volt), `mV`, `kV`
- **Current:** `A` (Ampere), `mA`, `µA`
- **Frequency:** `Hz`, `kHz`, `MHz`, `GHz`
- **Time:** `s`, `ms`, `µs`, `ns`
- **Impedance:** `Ω` (complex)

**Type safety:** Prevents mixing incompatible units (can't add resistance + capacitance).

---

## 3. Structural Constructs

### 3.1 circuit

Defines a self-contained circuit block.

```morphogen
circuit RC_LowPass {
    params:
        R: f32<Ω> = 1000
        C: f32<F> = 100e-9

    components:
        r1 = resistor(R)
        c1 = capacitor(C)

    nets:
        input: net(name="vin")
        output: net(name="vout")
        ground: net(name="gnd", type="ground")

    connections:
        input.connect(r1.port["p"])
        r1.port["n"].connect(output)
        output.connect(c1.port["p"])
        c1.port["n"].connect(ground)

    analysis:
        dc_op: dc_operating_point()
        ac: ac_sweep(freq_start=10Hz, freq_end=100kHz, points=100)
        tran: transient(duration=10ms, timestep=1µs)
}
```

**Properties:**
- Circuits are **pure functions** (parameters → netlist)
- Circuits can be **instantiated multiple times**
- Circuits support **hierarchical composition**

---

### 3.2 subcircuit

Encapsulates reusable circuit blocks (like SPICE .subckt).

```morphogen
subcircuit OpAmpInverting {
    params:
        gain: f32 = -10
        Rin: f32<Ω> = 10000

    ports:
        input: port(type="signal_in")
        output: port(type="signal_out")
        vcc: port(type="power")
        vee: port(type="power")
        gnd: port(type="ground")

    components:
        opamp = op_amp(model="tl072")
        r_in = resistor(Rin)
        r_fb = resistor(Rin * abs(gain))

    connections:
        input.connect(r_in.port["p"])
        r_in.port["n"].connect(opamp.port["in-"])
        opamp.port["in-"].connect(r_fb.port["p"])
        r_fb.port["n"].connect(opamp.port["out"])
        opamp.port["in+"].connect(gnd)
        opamp.port["vcc"].connect(vcc)
        opamp.port["vee"].connect(vee)
        opamp.port["out"].connect(output)
}
```

**Usage:**
```morphogen
circuit AudioPreamp {
    components:
        stage1 = OpAmpInverting(gain=-10)
        stage2 = OpAmpInverting(gain=-5)

    connections:
        input.connect(stage1.input)
        stage1.output.connect(stage2.input)
        stage2.output.connect(output)
}
```

---

### 3.3 netlist import / export

Interoperability with SPICE and other simulators.

```morphogen
import spice("amplifier.cir")           # Import SPICE netlist
import touchstone("filter.s2p")         # Import S-parameters (2-port)

export spice("output.cir", circuit=AudioPreamp)
export yaml("circuit.morphogen.yaml", circuit=AudioPreamp)
```

**Supported formats:**
- **SPICE:** `.cir`, `.sp`, `.lib` (component models)
- **Touchstone:** `.s2p`, `.s4p` (RF S-parameters)
- **Verilog-A:** `.va` (device models)
- **IBIS:** `.ibs` (digital I/O buffer models)

---

## 4. Reference System: CircuitRef

Following Morphogen's unified reference pattern, the Circuit domain uses **`CircuitRef`** as its primary reference type.

**CircuitRef targets:**
```
CircuitRef
├── ComponentRef       (reference to R, L, C, opamp, etc.)
├── NetRef             (reference to electrical net)
├── NodeRef            (reference to circuit node)
└── PortRef            (reference to component terminal)
```

### 4.1 Auto-Generated Anchors

Every circuit component auto-generates typed anchors:

**Linear Components (2-terminal):**

| Component | Anchors | Example |
|-----------|---------|---------|
| `Resistor` | `.port["p"]`, `.port["n"]`, `.param["R"]`, `.voltage`, `.current`, `.power` | `r1.voltage` |
| `Capacitor` | `.port["p"]`, `.port["n"]`, `.param["C"]`, `.charge`, `.energy` | `c1.charge` |
| `Inductor` | `.port["p"]`, `.port["n"]`, `.param["L"]`, `.flux`, `.energy` | `l1.flux` |

**Sources:**

| Component | Anchors | Example |
|-----------|---------|---------|
| `VoltageSource` | `.port["p"]`, `.port["n"]`, `.param["V"]`, `.current` | `v1.current` |
| `CurrentSource` | `.port["p"]`, `.port["n"]`, `.param["I"]`, `.voltage` | `i1.voltage` |

**Semiconductors:**

| Component | Anchors | Example |
|-----------|---------|---------|
| `Diode` | `.port["anode"]`, `.port["cathode"]`, `.param["Is"]`, `.param["n"]`, `.Vf`, `.If` | `d1.Vf` |
| `BJT (NPN)` | `.port["base"]`, `.port["collector"]`, `.port["emitter"]`, `.beta`, `.Vbe`, `.Vce`, `.Ic` | `q1.beta` |
| `MOSFET (N)` | `.port["gate"]`, `.port["drain"]`, `.port["source"]`, `.port["bulk"]`, `.Vgs`, `.Vds`, `.Id` | `m1.Id` |

**Op-Amps:**

| Component | Anchors | Example |
|-----------|---------|---------|
| `OpAmp` | `.port["in+"]`, `.port["in-"]`, `.port["out"]`, `.port["vcc"]`, `.port["vee"]`, `.gain`, `.bandwidth` | `u1.gain` |

**Nets:**

| Net | Anchors | Example |
|-----|---------|---------|
| `Net` | `.voltage`, `.current`, `.nodes[]`, `.impedance`, `.power` | `net.voltage.at_time(1ms)` |

**PCB Traces (geometry integration):**

| Trace | Anchors | Example |
|-------|---------|---------|
| `PCBTrace` | `.start`, `.end`, `.path[]`, `.width`, `.length`, `.inductance`, `.capacitance`, `.resistance` | `trace.inductance` |

---

### 4.2 Example: Auto-Anchor Usage

```morphogen
# Define components
r1 = resistor(R=1kΩ)
c1 = capacitor(C=10µF)
opamp = op_amp(model="tl072")

# Connect using auto-anchors (no manual node numbering!)
input_net = net(name="input")
output_net = net(name="output")
ground = net(name="gnd", type="ground")

input_net.connect(r1.port["p"])
r1.port["n"].connect(opamp.port["in+"])
opamp.port["in-"].connect(ground)
opamp.port["out"].connect(output_net)

# Access derived properties
gain = opamp.gain                    # Auto-computed from feedback network
power = r1.power                     # P = I² R
energy = c1.energy                   # E = ½ C V²
cutoff_freq = 1 / (2π * r1.R * c1.C) # Computed property
```

**Benefits:**
- No manual node numbering (SPICE netlist style)
- Type-safe connections (can't connect incompatible ports)
- Auto-computed derived properties
- Seamless PCB geometry integration

---

## 5. Circuit Operators

Circuit operators are organized in **4 layers** (atomic → composite → constructs → presets).

### 5.1 Layer 1: Atomic Components

**Linear Passive Components:**
```morphogen
resistor(R: f32<Ω>) -> Component
capacitor(C: f32<F>) -> Component
inductor(L: f32<H>) -> Component
transformer(L1: f32<H>, L2: f32<H>, k: f32) -> Component  # k = coupling coefficient
mutual_inductor(L1: f32<H>, L2: f32<H>, M: f32<H>) -> Component
```

**Independent Sources:**
```morphogen
voltage_source(V: f32<V>, type: Enum["dc", "ac", "pulse", "sine"]) -> Component
current_source(I: f32<A>, type: Enum["dc", "ac", "pulse", "sine"]) -> Component

# Time-varying sources
voltage_sine(V_amplitude: f32<V>, freq: f32<Hz>, phase: f32<deg>) -> Component
voltage_pulse(V_low: f32<V>, V_high: f32<V>, period: f32<s>, duty_cycle: f32) -> Component
```

**Dependent Sources (controlled sources):**
```morphogen
vcvs(gain: f32) -> Component  # Voltage-controlled voltage source (E)
vccs(gm: f32<A/V>) -> Component  # Voltage-controlled current source (G) - transconductance
ccvs(rm: f32<V/A>) -> Component  # Current-controlled voltage source (H) - transresistance
cccs(gain: f32) -> Component  # Current-controlled current source (F)
```

**Nonlinear Semiconductors:**
```morphogen
diode(Is: f32<A>, n: f32, Vt: f32<V> = 26mV) -> Component  # Shockley equation
zener_diode(Vz: f32<V>, Iz: f32<A>) -> Component

bjt_npn(beta: f32, Vbe: f32<V>, Vce_sat: f32<V>) -> Component
bjt_pnp(beta: f32, Vbe: f32<V>, Vce_sat: f32<V>) -> Component

mosfet_n(Vth: f32<V>, kn: f32, lambda: f32) -> Component  # N-channel
mosfet_p(Vth: f32<V>, kp: f32, lambda: f32) -> Component  # P-channel
```

**Integrated Components:**
```morphogen
op_amp(model: str, gain: f32 = 1e6, bandwidth: f32<Hz> = 1MHz) -> Component
comparator(model: str, vref: f32<V>) -> Component
voltage_regulator(Vin: f32<V>, Vout: f32<V>, Imax: f32<A>) -> Component
```

**Examples:**
```morphogen
r1 = resistor(R=1kΩ)
c1 = capacitor(C=100nF)
l1 = inductor(L=10mH)
v1 = voltage_source(V=5V, type="dc")
d1 = diode(Is=1e-12A, n=1.0)
q1 = bjt_npn(beta=100, Vbe=0.7V, Vce_sat=0.2V)
u1 = op_amp(model="tl072")
```

---

### 5.2 Layer 2: Composite Circuit Blocks

**Passive Networks:**
```morphogen
voltage_divider(R1: f32<Ω>, R2: f32<Ω>) -> Component
current_divider(R1: f32<Ω>, R2: f32<Ω>) -> Component
rc_filter(R: f32<Ω>, C: f32<F>, type: Enum["lpf", "hpf"]) -> Component
rl_filter(R: f32<Ω>, L: f32<H>, type: Enum["lpf", "hpf"]) -> Component
rlc_resonator(R: f32<Ω>, L: f32<H>, C: f32<F>, type: Enum["series", "parallel"]) -> Component
```

**Biasing Networks:**
```morphogen
dc_bias_network(Vcc: f32<V>, Vbias: f32<V>, Ibias: f32<A>) -> Component
voltage_reference(Vref: f32<V>, tolerance: f32) -> Component
```

**Matching Networks (RF):**
```morphogen
pi_matching(Zin: Complex<Ω>, Zout: Complex<Ω>, freq: f32<Hz>) -> Component
t_matching(Zin: Complex<Ω>, Zout: Complex<Ω>, freq: f32<Hz>) -> Component
```

**Examples:**
```morphogen
divider = voltage_divider(R1=10kΩ, R2=10kΩ)  # 50% divider
lpf = rc_filter(R=1kΩ, C=100nF, type="lpf")    # Cutoff ≈ 1.6kHz
resonator = rlc_resonator(R=10Ω, L=10mH, C=100nF, type="series")
```

---

### 5.3 Layer 3: Circuit Constructs

**Analog Audio Circuits:**
```morphogen
opamp_inverting_amp(gain: f32, Rin: f32<Ω>) -> Circuit
opamp_non_inverting_amp(gain: f32, Rin: f32<Ω>) -> Circuit
opamp_summing_amp(num_inputs: i32, gain: f32) -> Circuit
opamp_integrator(R: f32<Ω>, C: f32<F>) -> Circuit
opamp_differentiator(R: f32<Ω>, C: f32<F>) -> Circuit

sallen_key_filter(fc: f32<Hz>, Q: f32, type: Enum["lpf", "hpf", "bpf"]) -> Circuit
twin_t_notch(fc: f32<Hz>, depth: f32) -> Circuit
state_variable_filter(fc: f32<Hz>, Q: f32) -> Circuit  # Simultaneous LP/BP/HP outputs
```

**Power Electronics:**
```morphogen
buck_converter(Vin: f32<V>, Vout: f32<V>, Iout: f32<A>, fsw: f32<Hz>) -> Circuit
boost_converter(Vin: f32<V>, Vout: f32<V>, Iout: f32<A>, fsw: f32<Hz>) -> Circuit
buck_boost_converter(Vin: f32<V>, Vout: f32<V>, Iout: f32<A>, fsw: f32<Hz>) -> Circuit
ldo_regulator(Vin: f32<V>, Vout: f32<V>, Imax: f32<A>, dropout: f32<V>) -> Circuit
```

**Oscillators:**
```morphogen
rc_oscillator(freq: f32<Hz>, amplitude: f32<V>) -> Circuit
wien_bridge_oscillator(freq: f32<Hz>) -> Circuit
colpitts_oscillator(freq: f32<Hz>, L: f32<H>) -> Circuit
crystal_oscillator(freq: f32<Hz>, load_cap: f32<F>) -> Circuit
relaxation_oscillator(freq: f32<Hz>, duty_cycle: f32) -> Circuit
```

**RF Circuits:**
```morphogen
bandpass_filter_butterworth(fc: f32<Hz>, bw: f32<Hz>, order: i32) -> Circuit
lowpass_filter_chebyshev(fc: f32<Hz>, ripple_db: f32, order: i32) -> Circuit
rf_mixer(lo_freq: f32<Hz>, if_freq: f32<Hz>) -> Circuit
lna(gain_db: f32, noise_figure_db: f32, freq: f32<Hz>) -> Circuit  # Low-noise amplifier
```

**Tube Amplifiers (Nonlinear Modeling):**
```morphogen
triode_stage(tube_model: str, bias_voltage: f32<V>, plate_load: f32<Ω>) -> Circuit
pentode_output_stage(tube_model: str, class: Enum["A", "AB", "B"]) -> Circuit
cathode_follower(tube_model: str, bias_voltage: f32<V>) -> Circuit
```

**Examples:**
```morphogen
# Op-amp non-inverting amplifier (gain = 11)
amp = opamp_non_inverting_amp(gain=11, Rin=10kΩ)

# Sallen-Key 2nd-order low-pass filter
lpf = sallen_key_filter(fc=1kHz, Q=0.707, type="lpf")  # Butterworth

# Buck converter (12V → 5V @ 2A, 100kHz switching)
buck = buck_converter(Vin=12V, Vout=5V, Iout=2A, fsw=100kHz)

# Triode preamp stage (12AX7 tube)
preamp = triode_stage(tube_model="12ax7", bias_voltage=-1.5V, plate_load=100kΩ)
```

---

### 5.4 Layer 4: Circuit Presets

**Guitar Pedals:**
```morphogen
guitar_pedal_overdrive(drive: f32, tone: f32, level: f32) -> Circuit
guitar_pedal_distortion(distortion: f32, tone: f32, level: f32) -> Circuit
guitar_pedal_fuzz(fuzz: f32, tone: f32, level: f32) -> Circuit
guitar_pedal_chorus(rate: f32<Hz>, depth: f32, mix: f32) -> Circuit
guitar_pedal_delay(delay_time: f32<s>, feedback: f32, mix: f32) -> Circuit
```

**Tube Amplifiers:**
```morphogen
tube_amp_preamp(channels: i32, gain_stages: i32, eq_bands: i32) -> Circuit
tube_amp_power_stage(power_watts: f32, class: Enum["A", "AB", "B"], tubes: str) -> Circuit
tube_amp_reverb_tank(springs: i32, decay_time: f32<s>) -> Circuit
```

**Synthesizer Modules:**
```morphogen
synth_vcf_moog(cutoff: f32<Hz>, resonance: f32) -> Circuit  # Moog ladder filter
synth_vcf_roland(cutoff: f32<Hz>, resonance: f32) -> Circuit  # Roland IR3109
synth_vco(freq: f32<Hz>, waveform: Enum["saw", "square", "triangle", "sine"]) -> Circuit
synth_adsr(attack: f32<s>, decay: f32<s>, sustain: f32, release: f32<s>) -> Circuit
```

**Audio Mixer:**
```morphogen
mixer_channel_strip(eq_bands: i32, dynamics: bool, fx_sends: i32) -> Circuit
mixer_master_bus(num_channels: i32, dynamics: bool, metering: bool) -> Circuit
```

**Examples:**
```morphogen
# Tube Screamer-style overdrive pedal
pedal = guitar_pedal_overdrive(drive=0.7, tone=0.5, level=0.8)

# Fender-style tube amp preamp (2 channels, 3 gain stages each)
preamp = tube_amp_preamp(channels=2, gain_stages=3, eq_bands=3)

# Moog-style VCF (cutoff=1kHz, resonance=0.8)
vcf = synth_vcf_moog(cutoff=1kHz, resonance=0.8)
```

---

## 6. Analysis Methods

Circuit simulation requires multiple analysis types (DC, AC, transient, etc.).

### 6.1 DC Operating Point

Solve for steady-state (DC) voltages and currents.

```morphogen
dc_operating_point(circuit: Circuit) -> StateVector

# Example:
op = dc_operating_point(amplifier)
print(op.node["vout"].voltage)  # DC voltage at output node
print(op.component["q1"].Ic)     # DC collector current
```

**Algorithm:** Modified nodal analysis (MNA) + Newton-Raphson for nonlinear components.

---

### 6.2 AC Sweep (Small-Signal Analysis)

Compute frequency response (Bode plot).

```morphogen
ac_sweep(
    circuit: Circuit,
    freq_start: f32<Hz>,
    freq_end: f32<Hz>,
    points: i32,
    sweep_type: Enum["linear", "log", "decade"] = "log"
) -> FrequencyResponse

# Example:
response = ac_sweep(filter, freq_start=10Hz, freq_end=100kHz, points=100)
plot(response.frequency, response.magnitude_db)  # Bode magnitude
plot(response.frequency, response.phase_deg)     # Bode phase
```

**Output:**
- `magnitude_db`: Gain in dB
- `phase_deg`: Phase shift in degrees
- `group_delay`: Group delay (dτ/dω)

---

### 6.3 Transient Analysis

Simulate time-domain response.

```morphogen
transient(
    circuit: Circuit,
    duration: f32<s>,
    timestep: f32<s> = auto,
    method: Enum["euler", "trapezoidal", "rk4", "backward_euler"] = "trapezoidal",
    initial_conditions: StateVector = zero
) -> TimeSeries

# Example:
output = transient(
    circuit=amplifier,
    duration=10ms,
    timestep=1µs,
    method="rk4"
)

plot(output.time, output.net["vout"].voltage)  # Oscilloscope trace
```

**Integrators:**
- **euler:** Forward Euler (explicit, 1st order, unstable for stiff systems)
- **trapezoidal:** Trapezoidal rule (implicit, 2nd order, stable)
- **rk4:** Runge-Kutta 4 (explicit, 4th order, accurate)
- **backward_euler:** Backward Euler (implicit, 1st order, very stable)

**Adaptive timestep:** Automatically adjusts timestep based on error estimate.

---

### 6.4 Harmonic Balance (Steady-State AC)

For RF circuits and oscillators (periodic steady-state).

```morphogen
harmonic_balance(
    circuit: Circuit,
    fundamental_freq: f32<Hz>,
    num_harmonics: i32
) -> PeriodicSolution

# Example (oscillator):
solution = harmonic_balance(colpitts_osc, fundamental_freq=10MHz, num_harmonics=10)
```

---

### 6.5 Noise Analysis

Compute noise spectral density.

```morphogen
noise_analysis(
    circuit: Circuit,
    freq_start: f32<Hz>,
    freq_end: f32<Hz>,
    source: ComponentRef
) -> NoiseSpectrum

# Example:
noise = noise_analysis(lna, freq_start=100kHz, freq_end=1GHz, source=input_source)
print(noise.total_rms_voltage)  # Integrated noise (V_rms)
```

---

### 6.6 Sensitivity Analysis

Compute parameter sensitivity (∂V/∂R, etc.).

```morphogen
sensitivity_analysis(
    circuit: Circuit,
    output: NodeRef,
    parameters: [str]
) -> SensitivityMatrix

# Example:
sens = sensitivity_analysis(filter, output=vout, parameters=["R1", "C1"])
print(sens["R1"])  # ∂V_out / ∂R1
```

---

## 7. Solver Architecture

Circuit simulation requires sophisticated numerical solvers.

### 7.1 Solver Hierarchy

```
Direct Solvers (dense matrices, < 100 nodes):
├── LU decomposition
├── QR decomposition
└── Cholesky (symmetric positive definite)

Iterative Solvers (sparse matrices, > 1000 nodes):
├── Conjugate gradient
├── GMRES
└── BiCGSTAB

Nonlinear Solvers (diodes, transistors, op-amps):
├── Newton-Raphson (quadratic convergence)
├── Modified Newton (reuse Jacobian)
└── Damped Newton (robustness)

Time Integrators (transient analysis):
├── Euler (explicit, 1st order)
├── Backward Euler (implicit, stable)
├── Trapezoidal (implicit, 2nd order)
└── Runge-Kutta 4 (explicit, 4th order)
```

### 7.2 Auto-Solver Selection

Morphogen automatically selects the best solver based on circuit characteristics:

```python
def select_solver(circuit: Circuit):
    num_nodes = circuit.num_nodes()
    is_linear = circuit.is_linear()
    is_stiff = circuit.is_stiff()

    if num_nodes < 100:
        linear_solver = "LU"  # Dense direct solver
    else:
        linear_solver = "GMRES"  # Sparse iterative solver

    if is_linear:
        nonlinear_solver = None
    else:
        nonlinear_solver = "newton_raphson"

    if is_stiff:
        integrator = "backward_euler"  # Implicit, stable
    else:
        integrator = "rk4"  # Explicit, accurate

    return (linear_solver, nonlinear_solver, integrator)
```

### 7.3 GPU Acceleration

Large circuits benefit from GPU-accelerated linear algebra:

```morphogen
transient(
    circuit=large_circuit,
    duration=1ms,
    backend="gpu"  # Offload to GPU
)
```

**Backends:**
- `cpu`: Standard CPU solver (LAPACK, Eigen)
- `gpu`: CUDA/HIP GPU solver (cuSPARSE, cuSOLVER)
- `auto`: Automatically select based on circuit size

---

## 8. Multi-Domain Integration

Circuit domain integrates with other Morphogen domains.

### 8.1 Circuit ↔ Geometry (PCB Layout)

**PCB trace modeling:**

```morphogen
# Geometry: PCB layout
pcb = GeometryDomain.rectangle(width=100mm, height=60mm)

# Circuit: Amplifier
amp = CircuitDomain.opamp_amplifier(gain=10)

# Route trace (manual or auto-router)
trace_input = GeometryDomain.pcb_trace(
    start=pcb.anchor["edge_left"].at(y=30mm),
    end=amp.component["u1"].location,
    width=0.254mm,  # 10 mil
    layer="top_copper"
)

# Compute parasitic inductance from geometry
L_parasitic = CircuitDomain.compute_trace_inductance(trace_input, method="fasthenry")

# Add parasitic inductance to circuit
amp.net["input"].add_parasitic(inductor(L=L_parasitic))

# Simulate with parasitics
output = transient(amp, duration=1ms)
```

**Parasitic extraction methods:**
- `fasthenry`: Fast inductance approximation (MIT FastHenry)
- `fastcap`: Fast capacitance approximation (MIT FastCap)
- `full_em`: Full-wave EM solver (FDTD or FEM)

---

### 8.2 Circuit ↔ Audio (Analog Modeling)

**Guitar pedal simulation:**

```morphogen
# Circuit: Tube Screamer-style overdrive
pedal = CircuitDomain.guitar_pedal_overdrive(drive=0.7, tone=0.5, level=0.8)

# Audio: Load guitar riff
guitar = AudioDomain.load_sample("riff.wav", sample_rate=96kHz)

# Simulate circuit with audio input (4x oversampling for harmonics)
output = CircuitDomain.transient(
    circuit=pedal,
    input=guitar,
    sample_rate=384kHz,  # 4x oversampling
    method="backward_euler"
)

# Audio: Downsample and add reverb
output_48k = AudioDomain.resample(output, target_rate=48kHz)
reverb = AudioDomain.reverb(mix=0.2, room_size=0.8)
final = output_48k >> reverb

# Export
AudioDomain.export(final, "pedal_output.wav")
```

---

### 8.3 Circuit ↔ Physics (Thermal Coupling)

**Power amplifier thermal simulation:**

```morphogen
# Circuit: Class AB power amplifier
amp = CircuitDomain.class_ab_amplifier(power=100W)

# Simulate to get power dissipation
op_point = dc_operating_point(amp)
power_loss = amp.component["q_output"].power_dissipation

# Physics: Thermal model
heatsink = PhysicsDomain.thermal_model(
    material="aluminum",
    geometry=heatsink_geometry,
    ambient_temp=25°C
)

# Couple power dissipation to heat source
PhysicsDomain.add_heat_source(
    model=heatsink,
    power=power_loss,
    location=amp.component["q_output"].location
)

# Solve thermal distribution
temp_distribution = PhysicsDomain.steady_state_heat(heatsink)

# Feed temperature back to circuit (temperature-dependent beta, Vbe)
amp.component["q_output"].temperature = temp_distribution.at(
    amp.component["q_output"].location
)

# Re-solve circuit with updated temperature
op_point_hot = dc_operating_point(amp)
```

---

### 8.4 Circuit ↔ Pattern (Modulation)

**PWM-driven buck converter:**

```morphogen
# Pattern: PWM generator (Strudel-like pattern domain)
pwm = PatternDomain.pwm(
    frequency=100kHz,
    duty_cycle=PatternDomain.sine(freq=1kHz, amplitude=0.5, offset=0.5)
)

# Circuit: Buck converter
buck = CircuitDomain.buck_converter(
    Vin=12V,
    Vout=5V,
    Iout=2A,
    fsw=100kHz
)

# Drive switch with PWM pattern
buck.component["switch"].control = pwm

# Simulate
output_voltage = transient(buck, duration=10ms)
plot(output_voltage.time, output_voltage.net["vout"].voltage)
```

---

### 8.5 Circuit ↔ ML (Optimization)

**Gradient-based filter optimization:**

```morphogen
# Circuit: RC filter
filter = CircuitDomain.rc_filter(R=1kΩ, C=100nF, type="lpf")

# Define target frequency response
target = FrequencyResponse(
    freq=[100Hz, 1kHz, 10kHz],
    gain_db=[0, -3, -20]
)

# ML: Optimize component values
optimized_filter = MLDomain.optimize(
    circuit=filter,
    parameters=["R", "C"],
    objective=lambda c: loss(c.ac_sweep(), target),
    optimizer="adam",
    learning_rate=0.01,
    iterations=1000
)

print(optimized_filter.component["r1"].R)  # Optimized resistance
print(optimized_filter.component["c1"].C)  # Optimized capacitance
```

---

## 9. Circuit Domain Passes

Domain-specific optimization and validation passes.

### 9.1 Validation Passes

**Kirchhoff's Laws Enforcement:**
```python
class KirchhoffLawsPass(CircuitPass):
    """Validate KCL (current) and KVL (voltage)."""

    def validate_kcl(self, node):
        # Sum of currents into node = 0
        assert sum(node.currents) < tolerance

    def validate_kvl(self, loop):
        # Sum of voltages around loop = 0
        assert sum(loop.voltages) < tolerance
```

**Component Value Range Check:**
```python
class ComponentValueRangePass(CircuitPass):
    """Ensure component values are physically reasonable."""

    def validate(self, component):
        # R > 0, C > 0, L > 0
        assert component.R > 0 if isinstance(component, Resistor)
        assert component.C > 0 if isinstance(component, Capacitor)
```

---

### 9.2 Optimization Passes

**Series/Parallel Reduction:**
```python
class SeriesParallelReductionPass(CircuitPass):
    """Simplify series/parallel R, L, C combinations."""

    def visit_resistor_chain(self, resistors):
        if all_in_series(resistors):
            return Resistor(R=sum(r.R for r in resistors))
        elif all_in_parallel(resistors):
            return Resistor(R=1 / sum(1/r.R for r in resistors))
```

**Thevenin/Norton Equivalent:**
```python
class TheveninEquivalentPass(CircuitPass):
    """Replace complex network with Thevenin equivalent."""

    def simplify_two_port(self, network):
        Vth = compute_open_circuit_voltage(network)
        Rth = compute_output_impedance(network)
        return VoltageSource(Vth) + Resistor(Rth)
```

---

### 9.3 Lowering Passes

**Circuit → ODE System:**
```python
class CircuitToODEPass(LoweringPass):
    """Convert circuit netlist to state-space ODE."""

    def lower(self, circuit):
        # Modified nodal analysis (MNA)
        # dx/dt = Ax + Bu
        # y = Cx + Du
        A, B, C, D = compute_mna_matrices(circuit)
        return ODESystem(A, B, C, D, x0=initial_conditions)
```

**ODE → MLIR:**
```python
class ODEToMLIRPass(LoweringPass):
    """Lower ODE to MLIR scf/linalg dialects."""

    def lower(self, ode):
        # State-space to loops + linear algebra
        # Generate MLIR code
        return mlir_module
```

---

## 10. Determinism Profile

Circuit simulation must be deterministic for reproducibility.

| Feature | Determinism | Notes |
|---------|-------------|-------|
| **DC operating point** | Strict | Same netlist → same DC solution |
| **AC sweep** | Strict | Frequency response deterministic |
| **Transient (fixed timestep)** | Strict | Same timestep → same result |
| **Transient (adaptive timestep)** | Approximate | Timestep adapts to error tolerance |
| **Newton-Raphson convergence** | Approximate | Initial guess affects convergence path |
| **Noise analysis** | Strict | Deterministic noise spectral density |
| **Monte Carlo** | Stochastic | Component tolerances require RNG |

**Determinism guarantees:**
- Same netlist + same analysis → same results (bit-exact)
- Cross-platform reproducibility (CPU, GPU)
- Snapshot-based caching (memoization)

---

## 11. Examples

### 11.1 RC Low-Pass Filter

```morphogen
circuit RC_LPF {
    params:
        R: f32<Ω> = 1000
        C: f32<F> = 100e-9

    components:
        r1 = resistor(R)
        c1 = capacitor(C)

    nets:
        input: net(name="vin")
        output: net(name="vout")
        ground: net(name="gnd", type="ground")

    connections:
        input.connect(r1.port["p"])
        r1.port["n"].connect(output)
        output.connect(c1.port["p"])
        c1.port["n"].connect(ground)

    analysis:
        ac: ac_sweep(freq_start=10Hz, freq_end=100kHz, points=100)

    assertions:
        # Cutoff frequency ≈ 1.59kHz
        assert abs(ac.magnitude_db.at(1.59kHz) - (-3dB)) < 0.1dB
}
```

---

### 11.2 Op-Amp Inverting Amplifier

```morphogen
circuit InvertingAmp {
    params:
        gain: f32 = -10
        Rin: f32<Ω> = 10000

    components:
        opamp = op_amp(model="tl072")
        r_in = resistor(Rin)
        r_fb = resistor(Rin * abs(gain))

    nets:
        input, output, ground

    connections:
        input.connect(r_in.port["p"])
        r_in.port["n"].connect(opamp.port["in-"])
        opamp.port["in-"].connect(r_fb.port["p"])
        r_fb.port["n"].connect(opamp.port["out"])
        opamp.port["in+"].connect(ground)
        opamp.port["out"].connect(output)

    analysis:
        dc: dc_operating_point()
        ac: ac_sweep(freq_start=10Hz, freq_end=1MHz, points=200)

    assertions:
        assert abs(ac.magnitude_db.at(1kHz) - 20*log10(abs(gain))) < 0.5dB
}
```

---

### 11.3 Guitar Pedal (Tube Screamer)

```morphogen
circuit TubeScreamer {
    params:
        drive: f32 = 0.7      # 0.0 to 1.0
        tone: f32 = 0.5       # 0.0 to 1.0
        level: f32 = 0.8      # 0.0 to 1.0

    components:
        # Input buffer
        input_buffer = opamp_non_inverting_amp(gain=1)

        # Clipping stage (asymmetric diodes)
        r_clip = resistor(51kΩ)
        d1 = diode(Is=1e-12, n=1.0)  # Silicon
        d2 = diode(Is=1e-12, n=1.0)

        # Tone control
        tone_control = rc_filter(R=(1-tone)*220kΩ, C=0.22µF, type="lpf")

        # Output buffer
        output_buffer = opamp_non_inverting_amp(gain=level*10)

    connections:
        # ... (netlist)

    analysis:
        # Audio input
        input_signal = AudioDomain.load("guitar.wav", sample_rate=96kHz)

        # Transient simulation (4x oversampling)
        output = transient(
            circuit=self,
            input=input_signal,
            sample_rate=384kHz,
            method="backward_euler"
        )

        # Export
        AudioDomain.export(output, "tube_screamer_output.wav")
}
```

---

## 12. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| **Linear components (R, L, C)** | ✅ Done | `add_resistor`, `add_inductor`, `add_capacitor` |
| **DC operating point** | ✅ Done | `dc_analysis` — MNA direct solve |
| **AC sweep** | ✅ Done | `ac_analysis` — complex impedance at each frequency |
| **Transient (backward Euler)** | ✅ Done | `transient_analysis` — companion models for C, L; op-amps supported |
| **Diode (Shockley model)** | ✅ Done | `add_diode` — Is, n_factor; forward/reverse validated against Brent |
| **Newton-Raphson solver** | ✅ Done | `_solve_newton_raphson` — 100 iter, 1e-8V tol, clamped companion |
| **Op-amp models** | ✅ Done | `add_opamp` — finite gain, MNA stamp; DC, AC, transient |
| **Audio integration** | ✅ Done | `process_audio` — transient stepping per sample; RC/LC/diode memory |
| **Nonlinear BJT, MOSFET** | ❌ Planned | Phase 2 |
| **Tube models** | ❌ Planned | Phase 3 |
| **PCB geometry integration** | ❌ Planned | Phase 4 |
| **Parasitic extraction** | ❌ Planned | Phase 4 |
| **Multi-physics (thermal)** | ❌ Planned | Phase 5 |
| **ML optimization** | ❌ Planned | Phase 5 |

---

## 13. References

- **ADR-003:** Circuit Modeling Domain Design
- **operator-registry.md:** Operator metadata schema
- **ngspice:** Open-source SPICE simulator (reference implementation)
- **Kirchhoff's Circuit Laws:** Foundation of circuit analysis
- **Modified Nodal Analysis (MNA):** Circuit equation formulation
- **FastHenry/FastCap:** MIT tools for parasitic extraction

---

## Appendix: Comparison with SPICE

| Feature | SPICE | Morphogen.Circuit |
|---------|-------|---------------|
| **Netlist format** | Text-based, manual node numbering | YAML/declarative, auto-generated nodes |
| **Component library** | Extensive (decades of models) | Growing (import SPICE models) |
| **Analysis types** | DC, AC, tran, noise, distortion | DC, AC, tran, noise, harmonic balance |
| **Nonlinear solvers** | Newton-Raphson | Newton-Raphson + variants |
| **PCB integration** | None | Seamless (geometry domain) |
| **Audio modeling** | Limited | Native (audio domain) |
| **ML optimization** | None | Native (ML domain) |
| **Type safety** | None (text-based) | Full (units, types) |
| **Cross-platform** | Yes | Yes (MLIR-based) |
| **GPU acceleration** | Limited | Planned |
| **Extensibility** | `.subckt`, models | Operator registry, plugins |

**Key advantage:** Morphogen unifies circuit + PCB + audio + physics in one framework.
