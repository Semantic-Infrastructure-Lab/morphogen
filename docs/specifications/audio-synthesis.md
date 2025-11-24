# Audio Synthesis Domain Specification

**Version:** 0.11.0
**Status:** Active Development
**Last Updated:** 2025-11-23
**Domain:** `audio`

---

## Overview

The Audio Synthesis domain provides comprehensive tools for digital audio synthesis, processing, and analysis. This specification documents all audio operators with focus on recent additions including voltage-controlled amplifiers (VCA), filters with modulation, and phase continuity for oscillators.

**Key Features:**
- **Oscillators** with phase continuity (sine, saw, square, triangle)
- **Filters** with AudioBuffer modulation (vcf_lowpass, vcf_highpass, vcf_bandpass)
- **Envelopes** (ADSR, AR)
- **Modulation** (VCA, multiply/ring-mod)
- **Effects** (reverb, delay, chorus, flanger)
- **Analysis** (FFT, STFT, spectral operations)
- **Cross-rate modulation** support (e.g., 1kHz envelopes → 48kHz audio)

---

## Recent Developments

### November 2025 Updates

**Phase Continuity System (2025-11-23)**
- All oscillators now support phase parameter for seamless looping
- Automatic phase state management via convention-based detection
- Zero-click audio across buffer boundaries
- Sessions: spectral-pegasus-1123, dodamuku-1123

**VCA Operator (2025-11-23)**
- Voltage-controlled amplifier for envelope shaping
- Linear and exponential response curves
- Automatic CV normalization (handles bipolar envelopes)
- Cross-rate modulation support
- Session: tropical-cloud-1123

**Multiply Operator (2025-11-23)**
- Ring modulation and amplitude modulation
- Cross-rate capability for tremolo/AM effects
- Session: dodamuku-1123

**VCF Modulation (2025-11-22)**
- vcf_lowpass now accepts AudioBuffer cutoff parameter
- Dynamic filter sweeps and modulation
- Session: spectral-pegasus-1123

**Operator Count:** 57 operators registered (as of tropical-cloud-1123)

---

## Oscillators

### Overview

All oscillators generate AudioBuffer output at specified frequency and sample rate. Recent updates added phase continuity for seamless looping.

### Core Oscillators

#### `sine(frequency, sample_rate, duration, phase=None) -> AudioBuffer`

**Category:** OpCategory.CONSTRUCT
**Signature:** `(frequency: float, sample_rate: int, duration: float, phase: Optional[AudioBuffer]) -> AudioBuffer`
**Deterministic:** Yes

Generates a sine wave oscillator with optional phase continuity.

**Parameters:**
- `frequency` (float): Oscillation frequency in Hz
- `sample_rate` (int): Sample rate in Hz (e.g., 48000)
- `duration` (float): Duration in seconds
- `phase` (Optional[AudioBuffer]): Phase state for continuity (auto-detected)

**Features:**
- Pure sine wave generation
- Phase continuity support for seamless looping
- High-quality output suitable for musical applications

**Example:**
```python
from morphogen.stdlib.audio import AudioOperations as audio

# Simple sine wave
sine_wave = audio.sine(frequency=440.0, sample_rate=48000, duration=1.0)

# With phase continuity
phase_state = AudioBuffer.zeros(sample_rate=48000)
sine1 = audio.sine(440.0, 48000, 1.0, phase=phase_state)
sine2 = audio.sine(440.0, 48000, 1.0, phase=phase_state)  # Seamless continuation
```

**Phase Continuity:**
The `phase` parameter enables click-free looping by maintaining phase state between calls. If provided, the oscillator:
1. Reads initial phase from `phase.samples[0]`
2. Generates waveform continuing from that phase
3. Writes final phase back to `phase.samples[0]`

This allows infinite seamless loops without audible clicks.

---

#### `saw(frequency, sample_rate, duration, phase=None) -> AudioBuffer`

**Category:** OpCategory.CONSTRUCT
**Signature:** `(frequency: float, sample_rate: int, duration: float, phase: Optional[AudioBuffer]) -> AudioBuffer`
**Deterministic:** Yes

Generates a sawtooth wave oscillator with band-limited synthesis and phase continuity.

**Parameters:**
- `frequency` (float): Oscillation frequency in Hz
- `sample_rate` (int): Sample rate in Hz
- `duration` (float): Duration in seconds
- `phase` (Optional[AudioBuffer]): Phase state for continuity

**Features:**
- Band-limited synthesis using PolyBLEP anti-aliasing
- Reduces aliasing artifacts in sawtooth waves
- Phase continuity support
- Rich harmonic content for subtractive synthesis

**Waveform:** Ramps linearly from -1 to +1, sharp drop at period boundary (band-limited).

---

#### `square(frequency, sample_rate, duration, phase=None, duty_cycle=0.5) -> AudioBuffer`

**Category:** OpCategory.CONSTRUCT
**Signature:** `(frequency: float, sample_rate: int, duration: float, phase: Optional[AudioBuffer], duty_cycle: float) -> AudioBuffer`
**Deterministic:** Yes

Generates a square wave oscillator with configurable duty cycle, band-limited synthesis, and phase continuity.

**Parameters:**
- `frequency` (float): Oscillation frequency in Hz
- `sample_rate` (int): Sample rate in Hz
- `duration` (float): Duration in seconds
- `phase` (Optional[AudioBuffer]): Phase state for continuity
- `duty_cycle` (float): Pulse width (0.0-1.0, default 0.5)

**Features:**
- Band-limited synthesis using PolyBLEP
- Adjustable duty cycle (0.5 = symmetric square wave)
- Phase continuity support
- Odd harmonics only (when duty_cycle=0.5)

**Waveform:** Alternates between -1 and +1 based on duty cycle.

---

#### `triangle(frequency, sample_rate, duration, phase=None) -> AudioBuffer`

**Category:** OpCategory.CONSTRUCT
**Signature:** `(frequency: float, sample_rate: int, duration: float, phase: Optional[AudioBuffer]) -> AudioBuffer`
**Deterministic:** Yes

Generates a triangle wave oscillator with band-limited synthesis and phase continuity.

**Parameters:**
- `frequency` (float): Oscillation frequency in Hz
- `sample_rate` (int): Sample rate in Hz
- `duration` (float): Duration in seconds
- `phase` (Optional[AudioBuffer]): Phase state for continuity

**Features:**
- Band-limited synthesis
- Phase continuity support
- Softer harmonic content than square/saw
- Useful for mellow tones

**Waveform:** Linear rise from -1 to +1, then linear fall back to -1.

---

### Phase Continuity Implementation

**Convention-Based Detection:**
All oscillators use automatic phase state detection. If a parameter named `phase` is provided as an AudioBuffer:
1. Initial phase read from `phase.samples[0]` (in radians, 0 to 2π)
2. Waveform generated continuing from initial phase
3. Final phase written to `phase.samples[0]`

**Benefits:**
- Seamless looping without clicks
- Infinite duration synthesis via repeated calls
- No special API required - just pass phase state

**Example Pattern:**
```python
# Initialize phase state
phase = AudioBuffer(samples=np.array([0.0]), sample_rate=48000)

# Generate loop segments
for i in range(10):
    segment = audio.sine(440.0, 48000, 0.1, phase=phase)
    # Each segment continues seamlessly from previous
```

---

## Filters

### Overview

Morphogen provides both static and voltage-controlled filters. Recent updates added AudioBuffer modulation for dynamic filter sweeps.

### Voltage-Controlled Filters (VCF)

#### `vcf_lowpass(signal, cutoff, q=0.707, filter_state=None) -> AudioBuffer`

**Category:** OpCategory.TRANSFORM
**Signature:** `(signal: AudioBuffer, cutoff: AudioBuffer, q: float, filter_state: Optional[AudioBuffer]) -> AudioBuffer`
**Deterministic:** Yes
**Status:** ✅ Complete (Priority 4 - Filter State Management)

Voltage-controlled lowpass filter with dynamic cutoff modulation and optional state continuity.

**Parameters:**
- `signal` (AudioBuffer): Input audio signal
- `cutoff` (AudioBuffer): Cutoff frequency modulation (Hz)
- `q` (float): Quality factor/resonance (0.707 = critical damping, default)
- `filter_state` (Optional[AudioBuffer]): Filter state for continuity across hops

**Features:**
- **Dynamic cutoff modulation** - cutoff can be an AudioBuffer for filter sweeps
- **Filter state management** - seamless continuity across buffer hops
- Cross-rate modulation support (e.g., 1kHz envelope → 48kHz audio)
- Automatic rate conversion via scheduler
- Time-varying biquad filter topology
- Resonance control for emphasis

**Filter Response:**
- 2-pole (12 dB/octave) rolloff
- Biquad filter design (Direct Form II)
- Q parameter controls resonance peak at cutoff frequency

**State Continuity (Priority 4):**
The optional `filter_state` parameter enables seamless continuation across buffer hops:
- **Without state:** Each hop starts from zero state → audible clicks/pops
- **With state:** State preserved across hops → smooth, click-free synthesis
- **Usage:** Create once, pass to each call: `AudioBuffer(data=np.zeros(2), sample_rate=sr)`
- **Improvement:** 575,000× error reduction, 99.6% discontinuity reduction

**Use Cases:**
- Filter envelope sweeps (classic analog synth sound)
- LFO-modulated filter wobble
- Dynamic timbre changes
- Subtractive synthesis
- **Infinite synthesis** - process audio incrementally without clicks

**Example:**
```python
# Basic usage
signal = audio.saw(110.0, 48000, 2.0)
cutoff = audio.constant(1000.0, 48000, 2.0)
filtered = audio.vcf_lowpass(signal, cutoff, q=2.0)

# With filter state (for long-running/infinite synthesis)
filter_state = AudioBuffer(data=np.zeros(2), sample_rate=48000)
for chunk in audio_chunks:
    filtered = audio.vcf_lowpass(chunk, cutoff, q=2.0, filter_state=filter_state)
    # filter_state automatically updated - seamless continuity!
```

---

#### `vcf_highpass(signal, cutoff, q=0.707, filter_state=None) -> AudioBuffer`

**Category:** OpCategory.TRANSFORM
**Signature:** `(signal: AudioBuffer, cutoff: AudioBuffer, q: float, filter_state: Optional[AudioBuffer]) -> AudioBuffer`
**Deterministic:** Yes
**Status:** ✅ Complete (Priority 4 - Filter State Management)

Voltage-controlled highpass filter with dynamic cutoff modulation and optional state continuity.

**Parameters:**
- `signal` (AudioBuffer): Input audio signal
- `cutoff` (AudioBuffer): Cutoff frequency modulation (Hz)
- `q` (float): Quality factor/resonance (0.707 = critical damping, default)
- `filter_state` (Optional[AudioBuffer]): Filter state for continuity across hops

**Features:**
- Attenuates frequencies below cutoff, passes above
- **Filter state management** - seamless continuity across buffer hops
- Dynamic cutoff modulation
- Time-varying biquad filter topology
- Complementary to vcf_lowpass

---

#### `vcf_bandpass(signal, center_freq, q=1.0, filter_state=None) -> AudioBuffer`

**Category:** OpCategory.TRANSFORM
**Signature:** `(signal: AudioBuffer, center_freq: AudioBuffer, q: float, filter_state: Optional[AudioBuffer]) -> AudioBuffer`
**Deterministic:** Yes
**Status:** ✅ Complete (Priority 4 - Filter State Management)

Voltage-controlled bandpass filter with dynamic center frequency modulation and optional state continuity.

**Parameters:**
- `signal` (AudioBuffer): Input audio signal
- `center_freq` (AudioBuffer): Center frequency modulation (Hz)
- `q` (float): Filter Q (bandwidth control, higher Q = narrower band)
- `filter_state` (Optional[AudioBuffer]): Filter state for continuity across hops

**Features:**
- Passes frequencies near center, attenuates above and below
- **Filter state management** - seamless continuity across buffer hops
- Dynamic center frequency modulation
- Q parameter controls bandwidth
- Useful for formant synthesis, vowel sounds, wah effects

---

### Static Filters

#### `lowpass(signal, cutoff, sample_rate) -> AudioBuffer`

Simple first-order lowpass filter with fixed cutoff.

#### `highpass(signal, cutoff, sample_rate) -> AudioBuffer`

Simple first-order highpass filter with fixed cutoff.

#### `bandpass(signal, low_cutoff, high_cutoff, sample_rate) -> AudioBuffer`

Simple bandpass filter (lowpass + highpass cascade).

#### `notch(signal, frequency, q, sample_rate) -> AudioBuffer`

Notch filter for removing specific frequencies.

---

## Envelopes

### ADSR Envelope

#### `envelope_adsr(attack, decay, sustain, release, sample_rate, gate_duration) -> AudioBuffer`

**Category:** OpCategory.CONSTRUCT
**Signature:** `(attack: float, decay: float, sustain: float, release: float, sample_rate: int, gate_duration: float) -> AudioBuffer`
**Deterministic:** Yes

Generates classic ADSR (Attack-Decay-Sustain-Release) envelope.

**Parameters:**
- `attack` (float): Attack time in seconds
- `decay` (float): Decay time in seconds
- `sustain` (float): Sustain level (0.0-1.0)
- `release` (float): Release time in seconds
- `sample_rate` (int): Envelope sample rate (can differ from audio rate)
- `gate_duration` (float): How long gate is held (attack + decay + sustain time)

**Envelope Stages:**
1. **Attack:** Linear rise from 0 to 1 over `attack` seconds
2. **Decay:** Exponential decay from 1 to `sustain` level over `decay` seconds
3. **Sustain:** Held at `sustain` level for remaining gate duration
4. **Release:** Exponential decay from sustain to 0 over `release` seconds

**Output:** AudioBuffer with envelope shape (values 0.0-1.0)

**Use Cases:**
- Amplitude envelopes for VCA
- Filter cutoff envelopes for VCF
- Any parameter modulation over time

**Cross-Rate Usage:**
ADSR envelopes are often generated at lower rates (1kHz) for efficiency, then used to modulate audio-rate signals (48kHz). The scheduler automatically handles rate conversion.

**Example:**
```python
# Generate 1kHz envelope
env = audio.envelope_adsr(
    attack=0.1, decay=0.2, sustain=0.7, release=0.5,
    sample_rate=1000, gate_duration=2.0
)

# Use to control 48kHz audio via VCA
signal = audio.saw(220.0, 48000, 2.5)
shaped = audio.vca(signal, env, curve="linear")
# Scheduler converts 1kHz → 48kHz automatically
```

---

### AR Envelope

#### `envelope_ar(attack, release, sample_rate, duration) -> AudioBuffer`

Simple Attack-Release envelope (no sustain stage).

---

## Modulation

### VCA (Voltage-Controlled Amplifier)

#### `vca(signal, cv, curve="linear") -> AudioBuffer`

**Category:** OpCategory.TRANSFORM
**Signature:** `(signal: AudioBuffer, cv: AudioBuffer, curve: str) -> AudioBuffer`
**Deterministic:** Yes
**Added:** 2025-11-23 (tropical-cloud-1123)
**Operators:** 56 → 57

Voltage-controlled amplifier for amplitude control and envelope shaping.

**Parameters:**
- `signal` (AudioBuffer): Input audio signal to be shaped
- `cv` (AudioBuffer): Control voltage (0-1 range recommended)
- `curve` (str): Response curve - "linear" or "exponential"

**Features:**
- **Linear curve:** Direct amplitude multiplication (y = x)
- **Exponential curve:** sqrt() for natural dynamics (y = √x)
- **Automatic CV normalization:**
  - Unipolar (0→1): Used as-is
  - Bipolar (-1→1): Normalized to 0→1
  - Arbitrary range: Normalized to full 0→1 range
- **Cross-rate modulation:** Works with different sample rates
- **Length handling:** Zero-pads shorter signal to match longer

**Response Curves:**

**Linear (curve="linear"):**
- Direct CV → amplitude mapping
- Uniform response across range
- Standard VCA behavior

**Exponential (curve="exponential"):**
- sqrt() curve: y = √x
- Resists amplitude changes at low values
- More musical fade-outs
- ~22% higher overall RMS vs linear (for ramp-down envelope)
- Natural-sounding dynamics

**Use Cases:**
- **Envelope shaping:** Apply ADSR to oscillators
- **Tremolo effects:** LFO-controlled amplitude modulation
- **Gating:** Rhythmic amplitude control
- **Dynamics:** General amplitude automation

**Example:**
```python
# Classic synth patch with envelope
signal = audio.saw(220.0, 48000, 2.5)
envelope = audio.envelope_adsr(0.1, 0.2, 0.7, 0.5, 1000, 2.0)
shaped = audio.vca(signal, envelope, curve="exponential")

# Tremolo effect with LFO
lfo = audio.sine(6.0, 48000, 2.0)  # 6 Hz tremolo
tremolo = audio.vca(signal, lfo, curve="linear")
```

**Implementation Details:**
- CV normalization uses 1e-10 epsilon for division safety
- Zero-padding matches longer signal duration
- Curve selection via string parameter
- State-free (no history needed)

**Testing:**
Comprehensive test suite validates:
- ADSR envelope shaping (41.5% amplitude reduction)
- Linear vs exponential curves (22% difference)
- Different signal lengths (zero-padding)
- Tremolo effects (6 Hz LFO, 55% variation)
- CV normalization (unipolar, bipolar, arbitrary ranges)

**Reference:** `test_vca_operator.py` (242 lines, 5/5 tests passing)

---

### Multiply (Ring Modulation / AM)

#### `multiply(signal1, signal2) -> AudioBuffer`

**Category:** OpCategory.TRANSFORM
**Signature:** `(signal1: AudioBuffer, signal2: AudioBuffer) -> AudioBuffer`
**Deterministic:** Yes
**Added:** 2025-11-23 (dodamuku-1123)

Simple multiplication of two audio signals for ring modulation and amplitude modulation.

**Parameters:**
- `signal1` (AudioBuffer): First signal (often carrier)
- `signal2` (AudioBuffer): Second signal (often modulator)

**Features:**
- Sample-by-sample multiplication
- Zero-padding for different lengths
- Cross-rate modulation support

**Use Cases:**
- **Ring modulation:** Inharmonic sidebands (f1 ± f2)
- **Amplitude modulation:** Tremolo, AM synthesis
- **Frequency shifting:** Complex modulation effects

**Example:**
```python
# Ring modulation
carrier = audio.sine(440.0, 48000, 2.0)
modulator = audio.sine(200.0, 48000, 2.0)
ring_mod = audio.multiply(carrier, modulator)
# Output contains 240 Hz and 640 Hz (440-200, 440+200)

# Tremolo (similar to VCA but direct multiplication)
lfo = audio.sine(6.0, 48000, 2.0)
tremolo = audio.multiply(signal, lfo)
```

---

## Effects

### Reverb

#### `reverb(signal, room_size, damping, wet) -> AudioBuffer`

Algorithmic reverb for spatial effects.

### Delay

#### `delay(signal, delay_time, feedback, mix) -> AudioBuffer`

Delay line with feedback and mix control.

### Chorus

#### `chorus(signal, rate, depth, mix) -> AudioBuffer`

Chorus effect using modulated delay lines.

### Flanger

#### `flanger(signal, rate, depth, feedback, mix) -> AudioBuffer`

Flanger effect with short modulated delays.

---

## Analysis

### FFT Operations

#### `fft(signal) -> AudioBuffer`

Fast Fourier Transform for frequency analysis.

#### `ifft(spectrum) -> AudioBuffer`

Inverse FFT for synthesis.

### STFT Operations

#### `stft(signal, window_size, hop_size) -> AudioBuffer`

Short-Time Fourier Transform for time-frequency analysis.

#### `istft(stft_data, hop_size) -> AudioBuffer`

Inverse STFT for resynthesis.

---

## Cross-Rate Modulation

### Overview

Morphogen's scheduler enables **cross-rate modulation** where control signals at one sample rate modulate audio signals at another rate. This is crucial for efficiency and matches analog synthesizer behavior.

### Common Patterns

**1kHz Envelopes → 48kHz Audio:**
```python
# Generate envelope at 1kHz (efficient)
env = audio.envelope_adsr(0.1, 0.2, 0.7, 0.5, sample_rate=1000, gate_duration=2.0)

# Generate audio at 48kHz (high quality)
osc = audio.saw(220.0, sample_rate=48000, duration=2.5)

# VCA automatically handles rate conversion
output = audio.vca(osc, env)
# Scheduler converts env from 1kHz → 48kHz
```

**LFO Modulation:**
```python
# Low-frequency oscillator at 1kHz
lfo = audio.sine(6.0, sample_rate=1000, duration=2.0)

# Audio signal at 48kHz
signal = audio.sine(440.0, sample_rate=48000, duration=2.0)

# Filter modulation
cutoff = audio.multiply(lfo, audio.constant(1000.0, 1000, 2.0))
cutoff = audio.add(cutoff, audio.constant(500.0, 1000, 2.0))  # Offset
filtered = audio.vcf_lowpass(signal, cutoff)
# Scheduler handles 1kHz → 48kHz conversion
```

### Scheduler Behavior

The scheduler (Week 5 infrastructure, GraphIR + Scheduler):
1. Detects sample rate mismatches
2. Inserts automatic rate conversion nodes
3. Uses linear interpolation for upsampling
4. Maintains synchronization across rates
5. Zero overhead when rates match

### Benefits

- **Efficiency:** Lower rate for slow-moving controls
- **Flexibility:** Mix control rates as needed
- **Analog-accurate:** Matches hardware synthesizer behavior
- **Automatic:** No manual resampling required

---

## AudioBuffer Data Structure

### Overview

All audio operators work with the `AudioBuffer` dataclass, which encapsulates samples and metadata.

### Structure

```python
@dataclass
class AudioBuffer:
    samples: np.ndarray      # Audio samples (1D array of floats)
    sample_rate: int         # Sample rate in Hz (e.g., 48000)
    channels: int = 1        # Number of channels (default mono)
```

### Methods

**`AudioBuffer.zeros(sample_rate: int, duration: float) -> AudioBuffer`**
Create silent buffer.

**`AudioBuffer.from_samples(samples: np.ndarray, sample_rate: int) -> AudioBuffer`**
Create from existing sample array.

**`buffer.duration -> float`**
Computed duration in seconds.

**`buffer.num_samples -> int`**
Number of samples.

---

## Operator Registry

### Current Count

**Total Operators:** 57 (as of tropical-cloud-1123)

### Recent Additions

| Operator | Added | Session | Purpose |
|----------|-------|---------|---------|
| vca | 2025-11-23 | tropical-cloud-1123 | Voltage-controlled amplifier |
| multiply | 2025-11-23 | dodamuku-1123 | Ring modulation / AM |
| vcf_lowpass | 2025-11-22 | spectral-pegasus-1123 | AudioBuffer modulation |
| Phase params | 2025-11-23 | dodamuku-1123 | Oscillator continuity |

### Evolution

```
Week 4: 54 operators
Week 5: 55 operators (+ multiply)
Week 5: 56 operators (+ vcf modulation)
Week 5: 57 operators (+ vca)
```

---

## Implementation Patterns

### Convention-Based State Management

Morphogen uses convention-based detection for stateful operators:

**Phase State:**
- Parameter named `phase` with type `AudioBuffer`
- Read from `phase.samples[0]`, write back final phase
- No special API required

**Filter State (Planned - Priority 4):**
- Parameter named `filter_state` with type `AudioBuffer`
- Will preserve IIR filter coefficients across hops
- Similar pattern to phase continuity

### Zero-Padding for Length Mismatches

All binary operators (VCA, multiply, filters) handle different signal lengths:
```python
# If signal lengths differ
max_len = max(len(signal1), len(signal2))
# Shorter signal zero-padded to max_len
# Output has max_len samples
```

### Curve Selection

Operators with response curves use string parameters:
```python
# Linear response
vca(signal, cv, curve="linear")

# Exponential response
vca(signal, cv, curve="exponential")
```

Future: May extend to callable curves for custom responses.

---

## GraphIR Integration

### Overview

All audio operators integrate with GraphIR (Weeks 1-4 infrastructure) for graph-based synthesis.

### Pattern

```python
from morphogen.graphir import GraphIR, GraphIRNode

# Build graph
graph = GraphIR()
osc_node = graph.add_node(GraphIRNode(
    op="audio.saw",
    inputs={},
    params={"frequency": 220.0, "sample_rate": 48000, "duration": 2.0}
))

env_node = graph.add_node(GraphIRNode(
    op="audio.envelope_adsr",
    inputs={},
    params={"attack": 0.1, "decay": 0.2, "sustain": 0.7, "release": 0.5,
            "sample_rate": 1000, "gate_duration": 2.0}
))

vca_node = graph.add_node(GraphIRNode(
    op="audio.vca",
    inputs={"signal": osc_node.id, "cv": env_node.id},
    params={"curve": "linear"}
))

# Execute with scheduler
from morphogen.scheduler import Scheduler
scheduler = Scheduler()
result = scheduler.execute(graph, output_node=vca_node.id)
```

### Scheduler (Week 5)

The scheduler:
1. Topologically sorts graph nodes
2. Executes operators in dependency order
3. Handles cross-rate modulation automatically
4. Caches intermediate results
5. Returns final output

**Reference:** `examples/vca_simple.py` for complete GraphIR + Scheduler example

---

## Testing Standards

### Test Coverage Requirements

Each audio operator should have:
1. **Basic functionality test** - Verify operator produces expected output
2. **Parameter validation** - Test edge cases and invalid inputs
3. **Length handling** - Different input lengths
4. **Cross-rate test** - If applicable
5. **Integration test** - GraphIR + Scheduler execution

### Example: VCA Test Suite

From `test_vca_operator.py` (242 lines):

**Test 1:** Basic VCA with ADSR Envelope
- Verifies envelope shaping reduces amplitude
- Checks attack louder than release

**Test 2:** Linear vs Exponential Curves
- Compares response curves
- Validates exponential resists fade-out

**Test 3:** Different Signal Lengths
- Zero-padding working correctly
- Long + short = long output

**Test 4:** Tremolo Effect (LFO)
- 6 Hz LFO modulation
- Measures amplitude variation

**Test 5:** CV Normalization
- Unipolar, bipolar, arbitrary ranges
- All produce identical normalized results

**Result:** 5/5 tests passing (100%)

### Validation Metrics

Tests should measure:
- **RMS amplitude** - Average signal level
- **Peak amplitude** - Maximum value
- **Frequency content** - FFT analysis (where applicable)
- **Phase accuracy** - For oscillators
- **Numerical accuracy** - Comparison to theoretical values

---

## Automatic State Management in Scheduler

### Overview

The Morphogen scheduler (`OperatorExecutor`) provides **automatic state management** for stateful operators via convention-based parameter detection. No manual state tracking required - the scheduler handles everything!

**Session:** donupa-1123 (2025-11-23)
**Feature:** Convention-Based State Management Refactor

### State Management Patterns

The scheduler supports two state management patterns:

#### 1. Executor-Managed State

**Pattern:** Executor calculates and updates state after operator execution
**Use case:** When operators don't self-update state

**Supported Parameters:**
- `phase` (float): Oscillator phase continuity

**How it works:**
1. Scheduler detects `phase` parameter in operator signature
2. Creates/retrieves float state for node_id
3. Injects current phase into operator call
4. **Calculates final phase** after execution: `(initial_phase + 2π × freq × duration) % 2π`
5. Stores final phase for next execution

**Result:** Seamless phase continuity across multiple scheduler executions

#### 2. Operator-Managed State

**Pattern:** Operators update state via in-place AudioBuffer modification
**Use case:** When operators handle their own state updates

**Supported Parameters:**
- `filter_state` (AudioBuffer with 2 samples): Biquad filter state [z1, z2]

**How it works:**
1. Scheduler detects `filter_state` parameter in operator signature
2. Creates/retrieves AudioBuffer state (2 samples) for node_id
3. Injects AudioBuffer into operator call
4. **Operator updates state** via `filter_state.data[:2] = final_state`
5. State automatically persists for next execution

**Result:** Seamless filter continuity across buffer hops (575,000× error reduction!)

### Usage Example

```python
from morphogen.scheduler.operator_executor import OperatorExecutor, create_audio_registry

# Create executor
registry = create_audio_registry()
executor = OperatorExecutor(registry, sample_rate=48000)

# Execute oscillator multiple times - phase continuity automatic!
for i in range(10):
    result = executor.execute(
        operator_name="sine",
        node_id="osc1",  # Same node_id = state persists
        params={"freq": 440.0},
        inputs={},
        num_samples=4800,
        rate_hz=48000.0,
    )
    # Each execution continues seamlessly from previous phase
    # No clicks, no discontinuities!

# Execute filter multiple times - filter state automatic!
signal = np.random.randn(4800)
cutoff = np.ones(4800) * 1000.0

for i in range(5):
    result = executor.execute(
        operator_name="vcf_lowpass",
        node_id="filter1",  # Same node_id = state persists
        params={"q": 2.0},
        inputs={"signal": signal, "cutoff": cutoff},
        num_samples=4800,
        rate_hz=48000.0,
    )
    # Each hop continues filter from previous state
    # Perfect continuity!
```

### Implementation Details

**State Storage:**
- Per-node state: `executor._operator_state[node_id][param_name]`
- **Executor-managed:** Stored as native Python types (float)
- **Operator-managed:** Stored as AudioBuffer objects
- State persists across all executions with same node_id

**State Isolation:**
- Each `node_id` gets independent state
- Different operators can coexist on same node
- No state leakage between nodes

**Performance:**
- Zero overhead for non-stateful operators
- State lookup: O(1) dictionary access
- Memory: ~8 bytes per phase state, ~16 bytes per filter state

### Benefits

✅ **Zero-config** - No manual state management
✅ **Convention-based** - Works via parameter names
✅ **Extensible** - Easy to add new stateful parameters
✅ **Robust** - State automatically created/persisted
✅ **Efficient** - Minimal overhead

### Validation

Comprehensive test suite validates:
- ✅ Phase continuity: <3e-8 max error (EXCELLENT)
- ✅ Filter state continuity: <1e-6 max error (EXCELLENT)
- ✅ State isolation per node_id
- ✅ Multiple executions without error accumulation

**Test Results:** 6/6 passing
**Location:** `validate_state_management.py`

---

## GraphIR Integration with Auto-State Management

### Overview

The GraphIR scheduler (`SimplifiedScheduler`) **automatically inherits** all state management capabilities from `OperatorExecutor`, providing seamless infinite synthesis in full graph execution.

**Session:** drifting-antimatter-1123 (2025-11-23)
**Feature:** GraphIR Auto-State Validation & Documentation

### How It Works

The `SimplifiedScheduler` creates a single `OperatorExecutor` instance and reuses it for all node executions. Since state is stored per `node_id` in the executor, each GraphIR node automatically maintains its own state across multiple `execute()` calls.

```python
# SimplifiedScheduler initialization (morphogen/scheduler/simplified.py:127)
self.operator_executor = OperatorExecutor(self.operator_registry, self.sample_rate)

# Node execution (morphogen/scheduler/simplified.py:317-324)
outputs = self.operator_executor.execute(
    operator_name=node.op,
    node_id=node_id,           # Each node has unique ID
    params=node.params,
    inputs=inputs,
    num_samples=num_samples,
    rate_hz=rate_hz,
)
```

**Key Insight:** Because the scheduler passes each node's unique `node_id` to the executor, state is automatically isolated per node and persisted across graph executions!

### Usage Example

```python
from morphogen.graph_ir import GraphIR, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler

# Create graph with stateful oscillator
graph = GraphIR(sample_rate=48000)
graph.add_node(
    id="osc1",
    op="sine",
    outputs=[GraphIROutputPort(name="out", type="Sig")],
    rate="audio",
    params={"freq": 440.0}
)
graph.add_output("out", ["osc1:out"])

# Create scheduler (creates OperatorExecutor internally)
scheduler = SimplifiedScheduler(graph, sample_rate=48000)

# Execute multiple times - phase state persists automatically!
for chunk in range(50):
    result = scheduler.execute(duration_samples=2400)  # 0.05s chunks
    # Perfect continuity - no manual state management!
    # Could run infinitely!
```

### Validation

**Test Suite:** `tests/test_graphir_state_management.py`

Comprehensive tests validate:
- ✅ **Phase Continuity:** Single oscillator across 10 executions (<3e-8 error)
- ✅ **Multiple Nodes:** Independent state isolation (2 oscillators at different frequencies)
- ✅ **Infinite Synthesis:** 50 consecutive executions with perfect boundary continuity
- ✅ **Long Execution:** 100 chunks with no state corruption or error accumulation

**Test Results:** 4/4 passing (100%)

### Performance

**Benchmark:** `benchmarks/state_management_benchmark.py`

Key metrics:
- **State lookup:** 228ns per operation (GOOD)
- **Dictionary access:** O(1) with no performance concerns
- **Real-time audio:** No bottlenecks, production-ready

### Benefits for GraphIR Users

✅ **Zero Configuration** - Stateful graphs just work
✅ **Infinite Synthesis** - Run graphs indefinitely without clicks
✅ **Multiple Nodes** - Each node maintains independent state
✅ **Robust** - State automatically created and persisted
✅ **Fast** - Minimal overhead (<1% typical)

### Production Use Cases

**Streaming Audio:**
```python
# Process audio in real-time chunks
while streaming:
    chunk = scheduler.execute(duration_samples=buffer_size)
    audio_output.write(chunk["out"])
    # Phase and filter states persist automatically!
```

**Long-Running Synthesis:**
```python
# Generate hours of continuous audio
for hour in range(24):
    for minute in range(60):
        chunk = scheduler.execute(duration_seconds=60.0)
        save_audio(f"hour{hour}_min{minute}.wav", chunk["out"])
        # No discontinuities across file boundaries!
```

**Interactive Patches:**
```python
# User can start/stop execution without state loss
while user_playing:
    chunk = scheduler.execute(duration_samples=480)  # 10ms chunks
    # State preserved even with frequent starts/stops
```

---

## Roadmap

### Completed (2025-11-23)

✅ Phase continuity for all oscillators (sine, saw, square, triangle)
✅ VCF with AudioBuffer modulation (vcf_lowpass, vcf_highpass, vcf_bandpass)
✅ VCA operator with linear/exponential curves
✅ Multiply operator for ring mod / AM
✅ Cross-rate modulation support
✅ Filter state management (Priority 4) - IIR continuity across buffer hops
✅ Convention-based `filter_state` parameter for seamless long-running synthesis
✅ Comprehensive test suites (6/6 tests passing, 575,000× error reduction)

### Priority 4 Complete ✅ (fated-crown-1123)

**Filter State Management:**
- ✅ Convention-based `filter_state` parameter for vcf_lowpass, vcf_highpass, vcf_bandpass
- ✅ Biquad state preservation across buffer hops (2-element state vector)
- ✅ Eliminates audible clicks and pops at buffer boundaries
- ✅ 99.6% discontinuity reduction, <1e-6 max error
- ✅ Comprehensive tests validating state continuity
- ✅ Working examples demonstrating infinite synthesis
- ✅ Pattern mirrors phase continuity implementation

### OperatorExecutor Auto-State Management ✅ (donupa-1123)

**Automatic State Management in Scheduler:**
- ✅ Convention-based state detection (`phase`, `filter_state`)
- ✅ Dual pattern support (executor-managed + operator-managed)
- ✅ Zero-config state persistence across executions
- ✅ Per-node state isolation
- ✅ Phase continuity: <3e-8 error (EXCELLENT quality)
- ✅ Filter state continuity: <1e-6 error (EXCELLENT quality)

### GraphIR Auto-State Integration ✅ (drifting-antimatter-1123)

**GraphIR Automatic State Management:**
- ✅ SimplifiedScheduler inherits all OperatorExecutor state capabilities
- ✅ Per-node state isolation in graph execution
- ✅ Infinite synthesis in full graphs (tested 50+ consecutive executions)
- ✅ Multiple independent stateful nodes working simultaneously
- ✅ Comprehensive test suite: 4/4 passing (100%)
- ✅ Performance benchmarks: 228ns state lookup (production-ready)
- ✅ Documentation with production use cases
- ✅ Comprehensive validation suite (6/6 tests passing)
- ✅ Complete documentation with examples

### Future Enhancements

- [ ] Additional modulation operators (FM, waveshaping)
- [ ] Additional envelope shapes (exponential, logarithmic)
- [ ] Mixer improvements (if needed)
- [ ] SIMD vectorization for DSP operators
- [ ] Block processing optimization
- [ ] Memory allocation reduction

---

## Performance Considerations

### Optimization Status

**Current Implementation:**
- Pure NumPy operations (vectorized)
- No special SIMD optimizations yet
- Typical performance: real-time capable for moderate complexity

**Future Optimizations:**
- SIMD intrinsics for critical paths
- MLIR lowering for compiled kernels
- GPU acceleration for effects (reverb, FFT)
- Block processing for reduced overhead

### Benchmarking

Operators should be benchmarked for:
- Single-call latency
- Throughput (samples/second)
- Memory allocation overhead
- Cache efficiency

**Benchmark Suite:** `.benchmarks/` directory (ongoing)

---

## Related Domains

### Cross-Domain Integration

**Circuit Domain:**
- `circuit.process_audio()` enables analog circuit modeling
- Guitar pedals, analog filters, tube emulation
- Seamless integration via AudioBuffer

**Future Integrations:**
- Geometry domain: Physical modeling (string/membrane vibration)
- Optimization domain: Parameter tuning via ML
- Field domain: Acoustic simulation

---

## References

### Documentation

- **Main Specification:** `SPECIFICATION.md` - High-level overview
- **Audio Examples:** `examples/audio/` - Working examples
- **Test Suite:** Various test files in project root
- **GraphIR Documentation:** `docs/specifications/graphir.md`
- **Scheduler Documentation:** `docs/specifications/scheduler.md`

### Session History

- **spectral-pegasus-1123:** VCF modulation + state management planning
- **dodamuku-1123:** Phase parameters + multiply operator
- **tropical-cloud-1123:** VCA operator implementation
- **turbulent-breeze-1123:** GraphIR + Scheduler (Weeks 1-4)
- **fierce-sleet-1123:** Operator integration (Week 5)
- **pattering-squall-1123:** Planning session

### Related Specifications

- `docs/specifications/graphir.md` - Graph IR specification
- `docs/specifications/scheduler.md` - Scheduler specification
- `docs/adr/` - Architecture decision records
- `docs/CIRCUIT_DOMAIN_IMPLEMENTATION.md` - Circuit domain docs

---

## Contributing

### Adding New Audio Operators

1. **Implement operator** in `morphogen/stdlib/audio.py`
2. **Add @operator decorator** with proper metadata
3. **Export in AudioOperations** class
4. **Write comprehensive tests** (5+ test cases)
5. **Create working example** demonstrating GraphIR integration
6. **Update this specification** with operator documentation
7. **Increment operator count** in registry

### Design Guidelines

- **Deterministic:** Audio operators should be deterministic (same input → same output)
- **AudioBuffer I/O:** Use AudioBuffer for all audio signals
- **Convention-based state:** Follow phase continuity pattern for stateful operators
- **Zero-padding:** Handle different input lengths gracefully
- **Cross-rate support:** Work correctly with scheduler rate conversion
- **Test thoroughly:** Cover edge cases, parameter validation, integration

---

## Summary

The Audio Synthesis domain provides a comprehensive toolkit for digital audio synthesis with:

- **57 operators** registered and growing
- **Phase continuity** system for seamless looping
- **Cross-rate modulation** for efficient control signals
- **Dynamic modulation** via AudioBuffer parameters
- **GraphIR integration** for complex synthesis graphs
- **Production-ready** testing and examples

Recent additions (VCA, multiply, vcf modulation, phase continuity) enable classic subtractive synthesis patches and modern modulation techniques. Future work focuses on additional filter variants and filter state management.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-23
**Maintainer:** TIA (The Intelligent Agent)
**Session:** misty-rain-1123
