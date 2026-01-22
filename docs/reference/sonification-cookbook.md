# Sonification Cookbook: Cross-Domain Audio Synthesis

**Version:** 1.0
**Status:** Concept Exploration
**Last Updated:** 2025-11-20

**Related Documentation:**
- [Visualization Cookbook](./visualization-cookbook.md) - Visual representation catalog (complementary to this sonification guide)
- [Ambient Music Specification](../specifications/ambient-music.md) - Audio domain specification
- [Mathematical Music Frameworks](./mathematical-music-frameworks.md) - Theoretical foundations for musical structure

---

## Overview

This document explores novel approaches to **data sonification** and **audio-visual coupling** within Morphogen's multi-domain ecosystem. Unlike traditional audio synthesis, these techniques treat **computation itself as a musical instrument** — where particles, fields, graphs, and optimization landscapes become audible phenomena.

**Core Philosophy:**
- **Hearing computation** — Make invisible processes audible
- **Multi-sensory feedback** — Audio + visual for richer understanding
- **Musical data science** — Turn analysis into performance
- **Cross-domain composition** — Couple disparate simulations through sound

---

## Table of Contents

1. [Physics & Dynamics Sonification](#physics--dynamics-sonification)
2. [Field & PDE Audio](#field--pde-audio)
3. [Cellular Automata Music](#cellular-automata-music)
4. [Graph & Network Audio](#graph--network-audio)
5. [Optimization Soundscapes](#optimization-soundscapes)
6. [Agent & Swarm Compositions](#agent--swarm-compositions)
7. [Terrain & Procedural Audio](#terrain--procedural-audio)
8. [Cross-Domain Audio Pipelines](#cross-domain-audio-pipelines)
9. [Interactive Audio-Visual Instruments](#interactive-audio-visual-instruments)
10. [Implementation Patterns](#implementation-patterns)

---

## Physics & Dynamics Sonification

### 💡 Hearing the N-Body Problem

**Concept:** Gravitational simulations as generative music

**Mappings:**
```python
# Mass → Pitch/Timbre
frequency = base_freq * (mass / reference_mass)
timbre = "sine" if mass < 1.0 else "saw"

# Velocity → Volume/Envelope
amplitude = np.clip(velocity_magnitude / max_velocity, 0, 1)
attack_time = 1.0 / (velocity_magnitude + 1.0)

# Distance → Harmony/Dissonance
for body_i, body_j in pairs:
    distance = norm(body_i.pos - body_j.pos)
    # Close bodies → consonant intervals
    interval_ratio = quantize_to_harmony(distance, scale="just_intonation")

# Gravitational potential → Filter cutoff
potential_energy = -G * m1 * m2 / distance
filter_cutoff = map_range(potential_energy, (min_E, max_E), (200, 8000))
```

**Audio Implementation:**
```python
def nbody_sonification(bodies, dt, duration):
    """Convert N-body simulation to audio."""
    audio_buffer = []

    # Each body gets a voice
    voices = [
        audio.oscillator(
            freq=body_to_frequency(body),
            shape=body_to_waveform(body)
        ) for body in bodies
    ]

    for t in range(int(duration * sample_rate)):
        # Physics step
        bodies = physics.nbody_step(bodies, dt)

        # Update voice parameters
        sample = 0.0
        for i, body in enumerate(bodies):
            freq = body_to_frequency(body)
            amp = body_to_amplitude(body)

            # Apply gravitational interactions as modulation
            for j, other in enumerate(bodies):
                if i != j:
                    distance = np.linalg.norm(body.pos - other.pos)
                    # Gravitational coupling → FM synthesis
                    mod_depth = gravitational_coupling(body, other, distance)
                    freq += mod_depth * body_to_frequency(other)

            # Render voice
            voices[i].set_frequency(freq)
            voices[i].set_amplitude(amp)
            sample += voices[i].next_sample()

        audio_buffer.append(sample / len(bodies))

    return np.array(audio_buffer)
```

**Visual Coupling:**
```python
# Simultaneous audio-visual rendering
def nbody_audiovisual():
    bodies = init_nbody_system(n=5)

    while True:
        # Physics
        bodies = physics.nbody_step(bodies, dt=0.016)

        # Visual
        vis = visual.agents(
            bodies,
            color_property='mass',
            size_property='velocity',
            trail=True
        )

        # Audio (real-time synthesis)
        audio_frame = nbody_to_audio_frame(bodies)

        yield vis, audio_frame
```

**Status:** 💡 Concept - needs Physics → Audio interface
**Domains:** RigidBody, Physics, Audio, Visual
**Use Cases:** Educational physics, generative music, data exploration

---

### 💡 Collision Orchestra

**Concept:** Physical collisions as percussion instruments

**Mappings:**
```python
# Impulse magnitude → Strike velocity
velocity = np.clip(impulse / max_impulse, 0, 1)

# Material properties → Timbre
density_ratio = body.density / reference_density
if density_ratio < 0.5:
    instrument = "wood_block"
elif density_ratio < 2.0:
    instrument = "snare"
else:
    instrument = "metal_bell"

# Impact position → Stereo pan
pan = map_range(collision.position.x, (world_min, world_max), (-1, 1))

# Relative velocity → Pitch
closing_speed = np.dot(collision.normal, relative_velocity)
pitch_shift = cents_to_ratio(closing_speed * 100)
```

**Audio Synthesis:**
```python
def collision_to_percussion(collision, instrument_bank):
    """Generate percussion sound from collision event."""

    # Select instrument based on material
    instrument = select_instrument(
        collision.body1.material,
        collision.body2.material,
        instrument_bank
    )

    # Compute excitation
    impulse = collision.impulse_magnitude
    velocity = np.clip(impulse / 100.0, 0, 1)

    # Physical model synthesis
    if instrument.type == "membrane":
        # 2D wave equation (drum)
        audio = synthesize_drum(
            size=instrument.size,
            tension=instrument.tension,
            strike_velocity=velocity,
            strike_position=collision.local_position
        )
    elif instrument.type == "bar":
        # Modal synthesis (xylophone, bell)
        audio = synthesize_modal(
            modes=instrument.modal_frequencies,
            dampings=instrument.modal_dampings,
            excitation=velocity
        )

    # Spatial positioning
    audio_stereo = audio.pan(collision.position.x / world_width)

    return audio_stereo
```

**Integration Example:**
```python
from morphogen.cross_domain import PhysicsToAudioInterface

# Create collision event stream
bodies = rigidbody.create_system([...])
instrument_bank = load_instrument_bank("percussion.yaml")

audio_stream = []
for frame in range(num_frames):
    bodies, collisions = rigidbody.step_with_collisions(bodies, dt)

    # Sonify each collision
    for collision in collisions:
        audio_event = collision_to_percussion(collision, instrument_bank)
        audio_stream.append(audio_event)
```

**Status:** 🚧 Partially implemented (physics exists, needs audio synthesis)
**Domains:** RigidBody, Physics, Audio
**Use Cases:** Game audio, physics education, procedural sound design

---

### 💡 Spring Network Harmonics

**Concept:** Mass-spring systems as polyphonic synthesizers

**Physical Model:**
```python
# Each spring is an oscillator
for spring in springs:
    # Natural frequency from Hooke's law
    omega = np.sqrt(spring.stiffness / spring.mass)
    frequency = omega / (2 * np.pi)

    # Tension → amplitude
    extension = spring.current_length - spring.rest_length
    amplitude = np.abs(extension) / spring.rest_length

    # Damping → envelope
    decay_time = 1.0 / spring.damping

    # Synthesize oscillator
    osc = audio.oscillator(freq=frequency, shape="sine")
    env = audio.adsr(a=1ms, d=decay_time, s=0, r=0)
    spring_audio = osc * env * amplitude
```

**Coupled Oscillators:**
```python
def spring_network_synthesis(mass_spring_system):
    """Synthesize audio from coupled spring-mass system."""

    # Physical simulation
    masses, springs = mass_spring_system
    positions = [m.position for m in masses]
    velocities = [m.velocity for m in masses]

    audio_buffer = []

    for t in range(num_steps):
        # Physics step
        forces = compute_spring_forces(masses, springs)
        positions, velocities = integrate(forces, dt)

        # Audio: sum all spring oscillations
        sample = 0.0
        for spring in springs:
            # Spring velocity → audio sample
            extension = spring.current_length - spring.rest_length
            extension_velocity = spring.extension_rate

            # Velocity is proportional to pressure wave
            sample += extension_velocity * spring.audio_coupling

        audio_buffer.append(sample)

    return np.array(audio_buffer)
```

**Visual + Audio:**
```python
def spring_network_audiovisual():
    system = create_spring_network(grid_size=(10, 10))

    while True:
        # Physics
        system = physics.spring_step(system, dt=1/48000)

        # Visual: springs colored by tension
        vis = visual.spring_network(
            system,
            color_property='tension',
            palette='coolwarm'
        )

        # Audio: direct spring velocity sampling
        audio_sample = spring_to_audio(system)

        yield vis, audio_sample
```

**Status:** 💡 Concept - needs spring physics and audio coupling
**Domains:** Physics, Audio, Visual, Geometry
**Use Cases:** String instrument modeling, soft body audio, physical modeling synthesis

---

## Field & PDE Audio

### 💡 Temperature Field Sonification

**Concept:** Hear diffusion, convection, and thermal gradients

**Mappings:**
```python
# Mean temperature → Base frequency
mean_temp = field.mean(temperature)
base_freq = map_range(mean_temp, (0, 100), (110, 880))  # A2 to A5

# Gradient magnitude → Modulation depth
gradient = field.gradient(temperature)
grad_magnitude = field.magnitude(gradient)
mod_depth = np.mean(grad_magnitude) * 1000

# Variance → Noise amount
variance = field.variance(temperature)
noise_mix = np.clip(variance / 10.0, 0, 1)

# Laplacian → Filter resonance
laplacian = field.laplacian(temperature)
resonance = np.clip(np.abs(np.mean(laplacian)), 0.1, 10.0)
```

**Audio Synthesis:**
```python
def temperature_field_audio(temp_field, sample_rate=48000):
    """Sonify temperature field evolution."""

    # Base oscillator
    base_freq = field_to_frequency(temp_field)
    carrier = audio.oscillator(freq=base_freq, shape="saw")

    # Gradient → FM modulation
    gradient = field.gradient(temp_field)
    mod_freq = np.mean(field.magnitude(gradient)) * 100
    modulator = audio.oscillator(freq=mod_freq, shape="sine")

    # FM synthesis
    signal = carrier.fm(modulator, depth=mod_freq * 2)

    # Laplacian → filter
    laplacian = field.laplacian(temp_field)
    cutoff = map_range(np.mean(np.abs(laplacian)), (0, 1), (200, 8000))
    filtered = audio.lpf(signal, cutoff=cutoff, q=2.0)

    # Variance → noise
    noise = audio.noise(type="white")
    noise_amount = np.clip(field.variance(temp_field) / 10, 0, 0.3)

    return filtered * (1 - noise_amount) + noise * noise_amount
```

**Realtime Diffusion Audio:**
```python
def heat_diffusion_audio():
    temp = field.random((128, 128), low=0, high=1)

    while True:
        # Physics
        temp = field.diffuse(temp, rate=0.2, dt=0.1)

        # Visual
        vis = visual.colorize(temp, palette="fire")

        # Audio (44100 samples per frame at 30fps = 1470 samples)
        audio_frame = temperature_field_audio(temp)

        yield vis, audio_frame
```

**Status:** 🚧 Has field operations, needs Field → Audio interface
**Domains:** Field, Audio, Visual
**Use Cases:** Scientific sonification, ambient music, data exploration

---

### 💡 Vorticity Sonification

**Concept:** Hear fluid rotation and turbulence

**Mappings:**
```python
# Vorticity magnitude → Frequency
vorticity = field.curl(velocity_field)
vort_magnitude = np.abs(vorticity)
frequency = map_range(np.mean(vort_magnitude), (0, 10), (55, 440))

# Vorticity sign → Stereo position
vorticity_signed = np.mean(vorticity)
pan = np.tanh(vorticity_signed)  # -1 to +1

# Enstrophy (vorticity²) → Distortion
enstrophy = np.mean(vorticity ** 2)
distortion = np.clip(enstrophy / 100, 0, 0.8)

# Turbulent kinetic energy → Volume
tke = 0.5 * np.mean(velocity_field ** 2)
amplitude = np.clip(tke / 10, 0, 1)
```

**Audio Synthesis:**
```python
def vorticity_audio(velocity_field):
    """Sonify vorticity field."""

    # Compute vorticity
    vorticity = field.curl(velocity_field)

    # Oscillator bank (left/right rotation)
    vort_positive = np.clip(vorticity, 0, None)
    vort_negative = np.clip(-vorticity, 0, None)

    # Right-hand rotation → right channel
    freq_right = 220 + np.mean(vort_positive) * 100
    osc_right = audio.oscillator(freq=freq_right, shape="saw")

    # Left-hand rotation → left channel
    freq_left = 220 + np.mean(vort_negative) * 100
    osc_left = audio.oscillator(freq=freq_left, shape="saw")

    # Turbulence → noise modulation
    enstrophy = np.mean(vorticity ** 2)
    noise = audio.noise(type="pink") * enstrophy * 0.1

    left = osc_left + noise
    right = osc_right + noise

    return audio.stereo(left, right)
```

**Status:** 💡 Concept - needs fluid domain and audio coupling
**Domains:** Field, Fluid, Audio
**Use Cases:** CFD visualization, weather sonification, turbulence analysis

---

### 💡 Wave Equation Audio (Direct Physical Synthesis)

**Concept:** PDE solutions as audio signals

**1D String:**
```python
def vibrating_string_audio():
    """Physically accurate string synthesis via wave equation."""

    # 1D wave equation: u_tt = c² u_xx
    string_length = 1.0  # meters
    wave_speed = 343.0   # m/s (fundamental ~171.5 Hz)

    # Discretization
    nx = 256
    dx = string_length / nx
    dt = dx / (wave_speed * 2)  # Courant condition

    # Initial condition: pluck at 1/4 length
    u = field.zeros((nx,))
    u[nx//4] = 1.0
    u_prev = u.copy()

    audio_buffer = []
    pickup_position = int(3 * nx / 4)  # Pickup at 3/4 length

    for t in range(int(sample_rate * duration)):
        # Wave equation step
        u_new = (
            2 * u - u_prev +
            (wave_speed * dt / dx)**2 * (
                np.roll(u, 1) - 2*u + np.roll(u, -1)
            )
        )

        # Boundary conditions (fixed ends)
        u_new[0] = 0
        u_new[-1] = 0

        # Sample at pickup
        audio_sample = u_new[pickup_position]
        audio_buffer.append(audio_sample)

        # Update
        u_prev = u
        u = u_new

    return np.array(audio_buffer)
```

**2D Membrane (Drum):**
```python
def drum_synthesis():
    """2D wave equation for drum/membrane synthesis."""

    # 2D wave equation: u_tt = c² (u_xx + u_yy)
    size = 128
    wave_speed = 100.0
    dx = 1.0 / size
    dt = dx / (wave_speed * np.sqrt(2))

    u = field.zeros((size, size))
    u_prev = u.copy()

    # Strike the drum
    strike_pos = (size//3, size//2)
    u[strike_pos] = 1.0

    audio_buffer = []

    for t in range(int(sample_rate * duration)):
        # 2D Laplacian
        laplacian = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
            4 * u
        ) / (dx ** 2)

        u_new = 2*u - u_prev + (wave_speed * dt)**2 * laplacian

        # Fixed boundary
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0

        # Sample at center
        audio_sample = u_new[size//2, size//2]
        audio_buffer.append(audio_sample)

        u_prev = u
        u = u_new

    return np.array(audio_buffer)
```

**With Visualization:**
```python
def drum_audiovisual():
    """Real-time drum visualization with audio output."""

    membrane = init_membrane((128, 128))

    while True:
        # Physics (audio rate!)
        for _ in range(sample_rate // fps):
            membrane = wave_equation_step(membrane)
            audio_sample = membrane[64, 64]
            yield None, audio_sample

        # Visual (frame rate)
        vis = visual.colorize(membrane, palette="coolwarm")
        yield vis, None
```

**Status:** 📋 Needs 1D/2D wave equation operators
**Domains:** Field, Physics, Audio, Visual
**Use Cases:** Physical modeling synthesis, instrument design, acoustics education

---

## Cellular Automata Music

### 💡 Conway's Game of Life Sequencer

**Concept:** CA patterns as generative music patterns

**Mappings:**
```python
# Cell birth → Note onset
for cell in new_births:
    note_on(pitch=cell_to_pitch(cell.x, cell.y))

# Cell death → Note release
for cell in new_deaths:
    note_off(pitch=cell_to_pitch(cell.x, cell.y))

# Living cell density → Chord complexity
num_alive = ca.count_alive()
chord_size = map_range(num_alive, (0, total_cells), (1, 7))

# Pattern detection → Musical motifs
if detect_glider(ca):
    play_arpeggio(direction="up")
elif detect_oscillator(ca):
    play_rhythm(pattern="syncopated")
```

**Sequencer Implementation:**
```python
def gameoflife_sequencer(ca_state, scale="pentatonic"):
    """Convert CA to MIDI-like event stream."""

    prev_state = ca_state.copy()

    while True:
        # CA step
        ca_state = emergence.step(ca_state)

        # Detect changes
        births = (ca_state == 1) & (prev_state == 0)
        deaths = (ca_state == 0) & (prev_state == 1)

        # Convert to notes
        events = []

        # Births → note-ons
        birth_coords = np.argwhere(births)
        for y, x in birth_coords:
            pitch = spatial_to_pitch(x, y, scale)
            velocity = local_density(ca_state, x, y)
            events.append(NoteOn(pitch, velocity, time=current_time))

        # Deaths → note-offs
        death_coords = np.argwhere(deaths)
        for y, x in death_coords:
            pitch = spatial_to_pitch(x, y, scale)
            events.append(NoteOff(pitch, time=current_time))

        # Synthesize
        audio_frame = events_to_audio(events)

        # Visualize
        vis = visual.colorize(ca_state, palette="grayscale")

        yield vis, audio_frame

        prev_state = ca_state.copy()
        current_time += dt
```

**Pattern-Based Composition:**
```python
def ca_pattern_music():
    """Different CA patterns → different musical elements."""

    ca = emergence.cellular_automaton(rules="conway")

    # Detect patterns
    gliders = emergence.detect_gliders(ca)
    oscillators = emergence.detect_oscillators(ca)
    still_lifes = emergence.detect_still_lifes(ca)

    # Musical assignment
    bass_line = []
    melody = []
    percussion = []

    for glider in gliders:
        # Gliders → melody (moving pitch)
        melody.append(glider_to_note(glider, octave=5))

    for osc in oscillators:
        # Oscillators → rhythm (periodic triggers)
        if osc.period == 2:  # Blinker
            percussion.append(Kick(time=current_time))
        elif osc.period == 3:  # Pulsar
            percussion.append(Snare(time=current_time))

    for still in still_lifes:
        # Still lifes → bass drone
        bass_line.append(Drone(pitch=pattern_to_pitch(still)))

    return mix(bass_line, melody, percussion)
```

**Status:** 🚧 Has CA system, needs event → audio synthesis
**Domains:** Emergence, Audio, Visual
**Use Cases:** Generative music, algorithmic composition, live coding

---

### 💡 Reaction-Diffusion Audio Textures

**Concept:** RD pattern formation as evolving timbres

**Mappings:**
```python
# Pattern wavelength → Frequency
dominant_wavelength = fft_peak_wavelength(rd_pattern)
frequency = wave_speed / dominant_wavelength

# Pattern complexity → Harmonic content
complexity = measure_pattern_complexity(rd_pattern)
num_harmonics = int(complexity * 10)

# Growth rate → Envelope
growth_rate = measure_growth_rate(rd_pattern)
attack_time = 1.0 / (growth_rate + 0.1)

# Spatial correlation → Stereo width
correlation = spatial_autocorrelation(rd_pattern)
stereo_width = 1.0 - correlation
```

**Audio Synthesis:**
```python
def rd_audio_texture(u, v):
    """Sonify reaction-diffusion patterns."""

    # Analyze pattern
    fft_u = np.fft.fft2(u)
    dominant_freq = find_dominant_frequency(fft_u)

    # Additive synthesis based on spatial frequencies
    signal = 0.0

    for kx in range(num_harmonics_x):
        for ky in range(num_harmonics_y):
            # Spatial frequency → audio frequency
            spatial_freq = np.sqrt(kx**2 + ky**2)
            audio_freq = 110 * (1 + spatial_freq / 10)

            # FFT magnitude → amplitude
            magnitude = np.abs(fft_u[ky, kx])
            amplitude = magnitude / np.sum(np.abs(fft_u))

            # Synthesize partial
            signal += amplitude * audio.oscillator(
                freq=audio_freq,
                shape="sine"
            )

    # Pattern dynamics → modulation
    growth = np.mean(np.abs(u - u_prev))
    vibrato = audio.oscillator(freq=5, shape="sine") * growth * 10

    return signal * (1 + vibrato)
```

**Status:** 🚧 Needs RD operators and audio synthesis
**Domains:** Field, Emergence, Audio
**Use Cases:** Texture synthesis, ambient music, generative soundscapes

---

## Graph & Network Audio

### 💡 Network Topology Composition

**Concept:** Graph structure as musical form

**Mappings:**
```python
# Node degree → Note duration
duration = node.degree * 0.25  # seconds

# Betweenness centrality → Volume
amplitude = betweenness[node] / max_betweenness

# Community membership → Instrument
instrument = community_to_instrument[node.community]

# Edge weight → Harmony
if graph.has_edge(node_i, node_j):
    weight = graph[node_i][node_j]['weight']
    interval = weight_to_interval(weight)
    play_harmony(node_i_pitch, node_i_pitch * interval)
```

**Graph Traversal Sequencer:**
```python
def graph_walk_music(G, start_node, num_steps):
    """Musical random walk on graph."""

    current = start_node
    sequence = []

    for step in range(num_steps):
        # Node properties → note
        degree = G.degree(current)
        betweenness = nx.betweenness_centrality(G)[current]

        pitch = degree_to_pitch(degree, scale="minor")
        velocity = betweenness
        duration = 1.0 / (degree + 1)

        sequence.append(Note(pitch, velocity, duration))

        # Random walk
        neighbors = list(G.neighbors(current))
        if neighbors:
            # Edge weight → transition probability
            weights = [G[current][n]['weight'] for n in neighbors]
            current = random.choices(neighbors, weights=weights)[0]

    return sequence_to_audio(sequence)
```

**Community Detection Harmony:**
```python
def community_harmony(G):
    """Each community plays a chord."""

    communities = graph.community_detection(G, method="louvain")

    chord_progression = []

    for community_id, nodes in enumerate(communities):
        # Root note from community size
        root_pitch = 60 + community_id * 7  # MIDI note

        # Chord quality from internal connectivity
        internal_edges = sum(1 for u, v in G.edges()
                           if u in nodes and v in nodes)
        density = internal_edges / (len(nodes) * (len(nodes)-1) / 2)

        if density > 0.5:
            chord_type = "major"
        elif density > 0.3:
            chord_type = "minor"
        else:
            chord_type = "diminished"

        chord = build_chord(root_pitch, chord_type)
        chord_progression.append(chord)

    return chord_progression
```

**Status:** 🚧 Has graph domain, needs Graph → Audio interface
**Domains:** Graph, Audio, Visual
**Use Cases:** Network sonification, data music, social network analysis

---

## Optimization Soundscapes

### 💡 Gradient Descent Sonification

**Concept:** Hear the optimization landscape

**Mappings:**
```python
# Cost function value → Pitch
pitch = map_range(cost, (min_cost, max_cost), (110, 880))
# Lower cost = lower pitch (descending = pleasant)

# Gradient magnitude → Tempo/urgency
gradient_mag = np.linalg.norm(gradient)
tempo = 60 + gradient_mag * 100  # BPM

# Step size → Volume
amplitude = np.clip(step_size / max_step, 0, 1)

# Convergence → Harmonic resolution
if converged:
    play_chord("major")  # Consonant
else:
    play_chord("diminished")  # Tense
```

**Audio Implementation:**
```python
def gradient_descent_audio(cost_function, start_pos, lr=0.01):
    """Sonify optimization process."""

    pos = start_pos
    audio_events = []

    for step in range(max_steps):
        # Optimization step
        cost = cost_function(pos)
        gradient = numerical_gradient(cost_function, pos)
        pos = pos - lr * gradient

        # Cost → frequency
        freq = cost_to_frequency(cost)

        # Gradient → modulation
        grad_mag = np.linalg.norm(gradient)
        mod_depth = grad_mag * 100

        # Synthesize
        osc = audio.oscillator(freq=freq, shape="sine")
        mod = audio.oscillator(freq=freq/4, shape="sine")
        signal = osc.fm(mod, depth=mod_depth)

        # Step size → envelope
        env = audio.ar(attack=lr*100, release=lr*500)

        audio_events.append(signal * env)

    return concatenate(audio_events)
```

**Multi-Agent Optimization:**
```python
def pso_audio(cost_function, num_particles=30):
    """Sonify particle swarm optimization."""

    particles = pso_init(num_particles)

    while not converged:
        particles = pso_step(particles, cost_function)

        # Each particle is a voice
        audio_frame = 0.0

        for p in particles:
            # Personal best → pitch
            freq = cost_to_frequency(p.best_cost)

            # Velocity → vibrato
            vel_mag = np.linalg.norm(p.velocity)
            vibrato = audio.oscillator(freq=5, shape="sine") * vel_mag

            # Synthesize
            osc = audio.oscillator(freq=freq * (1 + vibrato))
            audio_frame += osc.next_sample() / num_particles

        yield audio_frame
```

**Status:** 🚧 Has optimization domain, needs audio coupling
**Domains:** Optimization, Audio, Visual
**Use Cases:** Algorithm visualization, educational demos, generative music

---

## Agent & Swarm Compositions

### 💡 Boid Chorus

**Concept:** Flocking behavior as polyphonic music

**Mappings:**
```python
# Velocity → Pitch (Doppler effect)
pitch = base_pitch * (1 + velocity.x / max_velocity * 0.1)

# Separation force → Dissonance
if separation_force > threshold:
    add_dissonant_interval()

# Alignment → Unison/harmony
alignment_score = measure_alignment(boid, neighbors)
if alignment_score > 0.9:
    play_unison()

# Cohesion → Chord density
num_neighbors = count_neighbors(boid, radius)
chord_notes = min(num_neighbors, 7)
```

**Audio Implementation:**
```python
def boid_audio(boids, num_voices=8):
    """Polyphonic synthesis from flocking."""

    # Select representative boids
    selected = sample_boids(boids, num_voices)

    audio_frame = 0.0

    for boid in selected:
        # Velocity → frequency (Doppler)
        speed = np.linalg.norm(boid.velocity)
        doppler_shift = 1 + boid.velocity.x / max_velocity * 0.05
        freq = 220 * doppler_shift

        # Neighbor count → timbre
        neighbors = query_neighbors(boid, radius=5.0)
        if len(neighbors) > 5:
            waveform = "saw"  # Dense flock
        elif len(neighbors) > 2:
            waveform = "square"
        else:
            waveform = "sine"  # Isolated

        # Alignment → amplitude
        alignment = measure_alignment(boid, neighbors)
        amplitude = alignment * 0.3

        # Synthesize
        osc = audio.oscillator(freq=freq, shape=waveform)
        audio_frame += osc * amplitude

    return audio_frame / num_voices
```

**Status:** 💡 Concept - needs boid implementation and audio coupling
**Domains:** Agents, Audio, Visual
**Use Cases:** Swarm sonification, generative music, educational demos

---

## Terrain & Procedural Audio

### 💡 Landscape Soundscapes

**Concept:** Elevation and terrain features as ambient sound

**Mappings:**
```python
# Elevation → Pitch
elevation = terrain.sample(x, y)
pitch = map_range(elevation, (min_elev, max_elev), (55, 440))

# Slope → Volume
slope = terrain.calculate_slope(x, y)
amplitude = slope / max_slope

# Biome → Timbre/instrument
biome = terrain.get_biome(x, y)
instrument = biome_to_instrument[biome]
# forest → woodwinds, desert → brass, ocean → pads

# Roughness → Noise content
roughness = terrain.roughness(x, y)
noise_mix = roughness
```

**Audio Synthesis:**
```python
def terrain_soundscape(terrain, path):
    """Generate soundscape by traversing terrain."""

    audio_buffer = []

    for pos in path:
        x, y = pos

        # Sample terrain
        elevation = terrain[int(y), int(x)]
        slope = terrain_slope(terrain, x, y)
        biome = terrain_biome(terrain, x, y)

        # Elevation → base frequency
        freq = 110 + elevation * 2

        # Biome → synthesis method
        if biome == "forest":
            signal = audio.oscillator(freq=freq, shape="sine")
            signal = audio.lpf(signal, cutoff=2000)
        elif biome == "mountain":
            signal = audio.oscillator(freq=freq, shape="saw")
            signal = audio.hpf(signal, cutoff=500)
        elif biome == "water":
            signal = audio.noise(type="pink")
            signal = audio.bpf(signal, center=freq, q=0.5)

        # Slope → amplitude
        amplitude = slope * 0.5

        audio_buffer.append(signal * amplitude)

    return np.array(audio_buffer)
```

**Status:** 🚧 Has terrain domain, needs audio coupling
**Domains:** Terrain, Audio, Procedural
**Use Cases:** Procedural music, game audio, ambient soundscapes

---

## Cross-Domain Audio Pipelines

### 💡 Terrain → Field → Audio

**Complete multi-hop pipeline:**

```python
from morphogen.cross_domain import TransformComposer

# Terrain elevation → scalar field
terrain = terrain.generate((256, 256), algorithm="perlin")

# Field → field operations (diffusion)
field = terrain_to_field(terrain)
field = field.diffuse(field, rate=0.1)

# Field statistics → audio parameters
composer = TransformComposer()
pipeline = composer.compose_path("field", "audio")

audio_params = pipeline(field)
# Returns: {frequency, amplitude, modulation, ...}

# Synthesize
signal = audio.oscillator(
    freq=audio_params['frequency'],
    shape="saw"
)
signal = audio.lpf(signal, cutoff=audio_params['modulation'])
```

**Status:** ✅ Supported by Phase 2 cross-domain system
**Domains:** Terrain, Field, Audio
**Use Cases:** Procedural soundtracks, generative music

---

### 💡 Physics → Acoustics → Audio

**Physical modeling chain:**

```python
# Rigid body simulation
bodies = rigidbody.create_system([...])

# Physics → collision events
for frame in range(num_frames):
    bodies, collisions = rigidbody.step_with_collisions(bodies, dt)

    # Collision → acoustic excitation
    for collision in collisions:
        # Map collision to waveguide excitation
        pipe = acoustics_geometry[collision.body_id]
        waveguide = acoustics.waveguide_from_geometry(pipe)

        # Impulse → excitation
        impulse_magnitude = collision.impulse
        excitation = impulse_to_acoustic_excitation(impulse_magnitude)

        # Propagate through waveguide
        p_fwd, p_bwd = acoustics.waveguide_step(
            p_fwd, p_bwd, waveguide,
            excitation=excitation
        )

        # Waveguide pressure → audio sample
        audio_sample = acoustics.total_pressure(p_fwd, p_bwd)
```

**Status:** 🚧 Physics and acoustics exist, needs coupling
**Domains:** Physics, Acoustics, Audio
**Use Cases:** Physical instrument modeling, impact sounds, acoustic simulation

---

## Interactive Audio-Visual Instruments

### 💡 Field Synthesizer

**Concept:** Draw/paint audio with field operations

**Interaction:**
```python
# User draws on canvas → field
field = field.zeros((512, 512))

on_mouse_drag(x, y):
    # Add Gaussian bump at cursor
    impulse = field.gaussian_bump((512, 512), center=(x, y), sigma=10)
    field = field + impulse

# Field → audio in real-time
while running:
    # Field operations
    field = field.diffuse(field, rate=0.1)
    field = field * 0.99  # Decay

    # Field → audio (per frame)
    audio_frame = field_to_audio_frame(field)

    # Visual feedback
    vis = visual.colorize(field, palette="viridis")

    render(vis)
    play_audio(audio_frame)
```

**Status:** 💡 Concept - needs interactive field input
**Domains:** Field, Audio, Visual, Interactive
**Use Cases:** Live performance, VJ tools, experimental instruments

---

### 💡 Agent Orchestra Controller

**Concept:** Control swarm = control ensemble

**Interaction:**
```python
# Agent properties → instrument parameters
agents = agents.create(n=100)

# User influences agents
on_mouse_position(x, y):
    # Agents attracted to cursor
    attractor = field.gaussian_bump((512, 512), (x, y), sigma=50)
    agents = agents.apply_field_force(attractor)

# Agents → polyphonic audio
while running:
    agents = agents.step(dt=0.016)

    # Each agent is a voice
    audio_frame = 0.0
    for agent in agents[:16]:  # Limit polyphony
        freq = agent_to_frequency(agent)
        amp = agent_to_amplitude(agent)
        audio_frame += synthesize_voice(freq, amp)

    vis = visual.agents(agents, color_property='velocity')

    render(vis)
    play_audio(audio_frame)
```

**Status:** 💡 Concept - needs interactive agent control
**Domains:** Agents, Audio, Visual, Interactive
**Use Cases:** Live coding, experimental music, performance tools

---

## Implementation Patterns

### Pattern 1: Statistical Field Sonification

```python
def field_to_audio_params(field):
    """Extract audio-relevant statistics from field."""

    return {
        'frequency': map_range(field.mean(), (0, 1), (110, 880)),
        'amplitude': np.clip(field.std(), 0, 1),
        'modulation': field.gradient().magnitude().mean() * 1000,
        'filter_cutoff': map_range(field.max(), (0, 1), (200, 8000)),
        'resonance': np.clip(np.abs(field.laplacian().mean()), 0.1, 10)
    }
```

### Pattern 2: Event-Based Synthesis

```python
def simulation_to_events(simulation_state, prev_state):
    """Convert state changes to discrete audio events."""

    events = []

    # Detect significant changes
    threshold_crossings = detect_crossings(simulation_state, prev_state, threshold=0.5)

    for crossing in threshold_crossings:
        events.append({
            'type': 'note_on',
            'pitch': position_to_pitch(crossing.position),
            'velocity': crossing.magnitude,
            'time': crossing.time
        })

    return events
```

### Pattern 3: Continuous Parameter Mapping

```python
def continuous_sonification(simulation):
    """Map continuous simulation values to audio parameters."""

    while True:
        state = simulation.step()

        # Real-time parameter extraction
        params = extract_audio_params(state)

        # Update oscillators
        for osc in oscillator_bank:
            osc.set_frequency(params['frequency'])
            osc.set_amplitude(params['amplitude'])

        # Render audio frame
        audio_frame = sum(osc.render() for osc in oscillator_bank)

        yield audio_frame
```

### Pattern 4: Spatial Audio

```python
def spatial_sonification(agents, listener_pos):
    """3D spatial audio from agent positions."""

    audio_frame_left = 0.0
    audio_frame_right = 0.0

    for agent in agents:
        # Distance attenuation
        distance = np.linalg.norm(agent.pos - listener_pos)
        attenuation = 1.0 / (1.0 + distance)

        # Stereo panning
        angle = np.arctan2(agent.pos.y - listener_pos.y,
                          agent.pos.x - listener_pos.x)
        pan = np.sin(angle)  # -1 (left) to +1 (right)

        # Synthesize
        signal = synthesize_agent(agent) * attenuation

        audio_frame_left += signal * (1 - pan) / 2
        audio_frame_right += signal * (1 + pan) / 2

    return audio_frame_left, audio_frame_right
```

---

## Utility Functions

### Frequency Mapping Helpers

```python
def linear_to_frequency(value, min_val=0, max_val=1, min_freq=110, max_freq=880):
    """Map linear value to frequency (Hz)."""
    normalized = (value - min_val) / (max_val - min_val)
    return min_freq + normalized * (max_freq - min_freq)

def exponential_to_frequency(value, min_val=0, max_val=1, min_freq=110, max_freq=880):
    """Map value to frequency with exponential scaling."""
    normalized = (value - min_val) / (max_val - min_val)
    return min_freq * (max_freq / min_freq) ** normalized

def quantize_to_scale(frequency, scale="chromatic"):
    """Quantize frequency to musical scale."""
    scales = {
        "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "pentatonic": [0, 2, 4, 7, 9]
    }

    # Convert frequency to MIDI
    midi = 69 + 12 * np.log2(frequency / 440)

    # Quantize to scale
    octave = int(midi // 12)
    note = int(midi % 12)

    # Find nearest scale degree
    scale_degrees = scales[scale]
    nearest = min(scale_degrees, key=lambda x: abs(x - note))

    # Convert back to frequency
    quantized_midi = octave * 12 + nearest
    return 440 * 2 ** ((quantized_midi - 69) / 12)
```

### Audio Buffer Management

```python
class AudioBuffer:
    """Ring buffer for audio sample accumulation."""

    def __init__(self, sample_rate=48000, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer = []

    def add_frame(self, samples):
        """Add audio frame to buffer."""
        self.buffer.extend(samples)

    def get_frame(self, num_samples):
        """Extract frame from buffer."""
        if len(self.buffer) >= num_samples:
            frame = self.buffer[:num_samples]
            self.buffer = self.buffer[num_samples:]
            return np.array(frame)
        else:
            # Pad with zeros if insufficient samples
            frame = self.buffer + [0] * (num_samples - len(self.buffer))
            self.buffer = []
            return np.array(frame)
```

---

## Performance Considerations

### Real-Time Audio Constraints

```python
# Audio callback rate
sample_rate = 48000
block_size = 512  # ~10ms latency

# Simulation timestep must align
simulation_dt = block_size / sample_rate  # 0.0106 seconds

# Visual frame rate
fps = 30
samples_per_frame = sample_rate // fps  # 1600 samples
```

### Optimization Strategies

1. **Limit polyphony**: Max 16-32 voices for real-time
2. **Spatial culling**: Only sonify visible/nearby objects
3. **Event throttling**: Limit event rate to prevent audio glitching
4. **Buffer pre-computation**: Compute audio ahead of visualization
5. **Downsampling**: Run simulation at lower rate, upsample audio

---

## Future Directions

### Phase 2 Ideas

- **ML → Audio**: Neural network training as music (loss → pitch, gradient → rhythm)
- **Circuit → Audio**: Circuit simulation audio output
- **Chemistry → Audio**: Molecular dynamics sonification
- **Weather → Audio**: Meteorological data as ambient soundscapes

### Advanced Techniques

- **Spatial audio**: 3D positioning with HRTF
- **Granular synthesis**: Field/agent data as grain clouds
- **Spectral processing**: FFT-based transformations
- **Physical modeling**: Direct PDE → audio (no synthesis layer)

---

## Examples & Demos

See `examples/audio_visualization/`:

- `01_nbody_sonification.py` - N-body gravitational music
- `02_field_audio.py` - Temperature field sonification
- `03_ca_sequencer.py` - Game of Life sequencer
- `04_graph_music.py` - Network topology composition
- `05_terrain_soundscape.py` - Landscape audio generation
- `06_collision_percussion.py` - Physics-driven drums
- `07_wave_equation_synthesis.py` - Direct physical audio synthesis

---

## References

### Sonification Research

- **"The Sonification Handbook"** — Hermann, Hunt, Neuhoff (2011)
- **"Auditory Display"** — Kramer (1994)
- **"Data Sonification: A Design Pattern Approach"** — Barrass (1997)

### Physical Modeling

- **"Physical Audio Signal Processing"** — Julius O. Smith III
- **"Digital Waveguide Modeling"** — Välimäki et al. (2006)
- **"Real Sound Synthesis for Interactive Applications"** — Cook (2002)

### Cross-Domain Inspiration

- **Cymatics** — Sound → visual patterns (Chladni plates)
- **Theremin** — Spatial gesture → audio
- **Reactable** — Tangible interface for audio-visual performance

---

**Document Status:** Concept Exploration
**Next Steps:** Implement Phase 1 cross-domain audio transforms
**Maintainer:** Morphogen Development Team
**Last Updated:** 2025-11-20
