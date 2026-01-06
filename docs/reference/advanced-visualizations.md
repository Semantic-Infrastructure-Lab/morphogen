# Advanced Visualizations Guide (v0.11.0)

Comprehensive guide to Morphogen's advanced visualization capabilities for analyzing multi-domain simulations.

---

## Overview

Morphogen v0.11.0 introduces powerful new visualization tools for analyzing complex systems:

- **Spectrogram** - Audio frequency-time analysis
- **Graph Networks** - Network topology visualization with multiple layouts
- **Phase Space** - Dynamical systems analysis (position-velocity diagrams)
- **Metrics Dashboard** - Real-time statistics overlay

These tools integrate seamlessly with existing field and agent visualizations, enabling comprehensive multi-domain analysis.

---

## Spectrogram Visualization

### Basic Usage

Visualize audio signals in the frequency-time domain:

```python
from morphogen.stdlib import audio, visual

# Load or generate audio
audio_buffer = audio.AudioBuffer(signal_data, sample_rate=44100)

# Create spectrogram
spec_vis = visual.spectrogram(
    audio_buffer,
    window_size=2048,
    hop_size=512,
    palette="fire",
    log_scale=True
)

visual.output(spec_vis, "spectrogram.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | AudioBuffer or ndarray | required | Audio signal to analyze |
| `sample_rate` | int | 44100 | Sample rate in Hz (auto-detected from AudioBuffer) |
| `window_size` | int | 2048 | FFT window size (larger = better frequency resolution) |
| `hop_size` | int | 512 | Hop between windows (smaller = better time resolution) |
| `palette` | str | "viridis" | Color palette ("grayscale", "fire", "viridis", "coolwarm") |
| `log_scale` | bool | True | Use logarithmic (dB) scale for magnitude |
| `freq_range` | tuple or None | None | (min_freq, max_freq) in Hz to display |

### Use Cases

**Harmonic Analysis**
```python
# Use larger window for better frequency resolution
spec_vis = visual.spectrogram(
    audio,
    window_size=4096,
    hop_size=1024,
    palette="viridis",
    freq_range=(0, 3000)  # Focus on harmonics
)
```

**Transient Analysis**
```python
# Use smaller window for better time resolution
spec_vis = visual.spectrogram(
    percussion_audio,
    window_size=1024,
    hop_size=256,
    palette="fire",
    log_scale=True
)
```

**Speech Analysis**
```python
# Focus on vocal frequency range
spec_vis = visual.spectrogram(
    voice_audio,
    window_size=2048,
    hop_size=512,
    freq_range=(80, 8000),  # Human voice range
    palette="coolwarm"
)
```

---

## Graph Network Visualization

### Basic Usage

Visualize network topologies with multiple layout algorithms:

```python
from morphogen.stdlib import graph, visual

# Create network
g = graph.create(20)
g = graph.add_edge(g, 0, 1, 1.0)
g = graph.add_edge(g, 1, 2, 1.0)
# ... add more edges

# Visualize with force-directed layout
vis = visual.graph(
    g,
    width=800,
    height=800,
    layout="force",
    color_by_centrality=True,
    palette="viridis"
)

visual.output(vis, "network.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | Graph | required | Graph instance to visualize |
| `width` | int | 800 | Output image width |
| `height` | int | 800 | Output image height |
| `node_size` | float | 8.0 | Node radius in pixels |
| `node_color` | tuple | (0.3, 0.6, 1.0) | Default node color (R, G, B) |
| `edge_color` | tuple | (0.5, 0.5, 0.5) | Edge color (R, G, B) |
| `edge_width` | float | 1.0 | Edge line width in pixels |
| `layout` | str | "force" | Layout algorithm (see below) |
| `iterations` | int | 50 | Force-directed iterations |
| `color_by_centrality` | bool | False | Color nodes by degree centrality |
| `palette` | str | "viridis" | Palette for centrality coloring |
| `show_labels` | bool | False | Show node labels (future) |
| `background` | tuple | (0, 0, 0) | Background color (R, G, B) |

### Layout Algorithms

**Force-Directed (Fruchterman-Reingold)**
```python
vis = visual.graph(
    g,
    layout="force",
    iterations=100,  # More iterations = better layout
    color_by_centrality=True,
    palette="fire"
)
```

Best for: General-purpose networks, revealing community structure

**Circular**
```python
vis = visual.graph(
    g,
    layout="circular",
    color_by_centrality=True
)
```

Best for: Ring networks, showing equal node spacing, small networks

**Grid**
```python
vis = visual.graph(
    g,
    layout="grid"
)
```

Best for: Lattice networks, spatial grids, structured topologies

### Centrality Coloring

Visualize node importance by degree centrality:

```python
vis = visual.graph(
    g,
    color_by_centrality=True,
    palette="fire",  # Hot colors = high centrality
    node_size=10.0
)
```

Nodes are colored based on their degree (number of connections):
- Low degree → Cool colors (dark on "fire" palette)
- High degree → Hot colors (bright on "fire" palette)

### Network Types

**Small-World Networks**
```python
# High clustering, short path lengths
vis = visual.graph(
    small_world_network,
    layout="force",
    iterations=150,
    color_by_centrality=True,
    palette="viridis"
)
```

**Scale-Free Networks**
```python
# Power-law degree distribution
vis = visual.graph(
    scale_free_network,
    layout="force",
    color_by_centrality=True,  # Shows hubs clearly
    palette="fire",
    node_size=15.0
)
```

**Grid/Lattice Networks**
```python
# Regular structure
vis = visual.graph(
    lattice_network,
    layout="grid",  # Preserves spatial structure
    node_color=(0.3, 0.8, 0.3)
)
```

---

## Phase Space Visualization

### Basic Usage

Analyze dynamical systems by plotting position vs velocity:

```python
from morphogen.stdlib import agents, visual

# Create particles with positions and velocities
particles = agents.create(1000, pos=positions)
particles = agents.set(particles, 'vel', velocities)

# Visualize phase space
vis = visual.phase_space(
    particles,
    width=700,
    height=700,
    color_property='energy',
    palette='fire'
)

visual.output(vis, "phase_space.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | Agents | required | Agents instance to visualize |
| `position_property` | str | 'pos' | Name of position property |
| `velocity_property` | str | 'vel' | Name of velocity property |
| `width` | int | 512 | Output image width |
| `height` | int | 512 | Output image height |
| `color_property` | str or None | None | Property to color points by |
| `palette` | str | "viridis" | Color palette |
| `point_size` | float | 2.0 | Point radius in pixels |
| `alpha` | float | 0.6 | Point transparency [0, 1] |
| `show_trajectories` | bool | False | Connect points in agent order |
| `background` | tuple | (0, 0, 0) | Background color (R, G, B) |

### Use Cases

**Harmonic Oscillators**
```python
# Visualize phase space portrait
vis = visual.phase_space(
    oscillators,
    color_property='energy',
    palette='fire',
    alpha=0.7
)
```

Shows elliptical trajectories for simple harmonic motion.

**Chaotic Systems**
```python
# Show sensitive dependence on initial conditions
vis = visual.phase_space(
    chaotic_system,
    color_property='divergence',
    palette='coolwarm',
    point_size=2.5,
    alpha=0.6
)
```

Reveals strange attractors and chaotic dynamics.

**Orbital Dynamics**
```python
# Analyze orbital mechanics
vis = visual.phase_space(
    planetary_system,
    color_property='orbital_radius',
    palette='viridis',
    show_trajectories=False
)
```

Shows conservation of angular momentum.

**Damped Systems**
```python
# Visualize energy dissipation
vis = visual.phase_space(
    damped_particles,
    color_property='time',
    palette='grayscale',
    show_trajectories=True  # Shows spiral trajectories
)
```

Reveals exponential decay toward fixed point.

### Multidimensional Data

For 2D or 3D positions/velocities, the visualization uses vector magnitude:

```python
# 2D positions and velocities
positions_2d = np.random.randn(1000, 2)
velocities_2d = np.random.randn(1000, 2)

particles = agents.create(1000, pos=positions_2d)
particles = agents.set(particles, 'vel', velocities_2d)

# Automatically uses |pos| vs |vel|
vis = visual.phase_space(particles, palette='viridis')
```

---

## Metrics Dashboard

### Basic Usage

Overlay real-time statistics on any visualization:

```python
from morphogen.stdlib import visual

# Create base visualization
vis = visual.colorize(field, palette="fire")

# Add metrics overlay
metrics = {
    "Frame": 42,
    "Temperature": 273.15,
    "Particles": 1000,
    "FPS": 59.8
}

vis_with_metrics = visual.add_metrics(
    vis,
    metrics,
    position="top-left"
)

visual.output(vis_with_metrics, "output.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `visual` | Visual | required | Visual to add metrics to |
| `metrics` | dict | required | Dictionary of name → value pairs |
| `position` | str | "top-left" | Position (see below) |
| `font_size` | int | 14 | Font size in pixels |
| `text_color` | tuple | (1, 1, 1) | Text color (R, G, B) |
| `bg_color` | tuple | (0, 0, 0) | Background color (R, G, B) |
| `bg_alpha` | float | 0.7 | Background transparency [0, 1] |

### Positions

- `"top-left"` - Upper left corner
- `"top-right"` - Upper right corner
- `"bottom-left"` - Lower left corner
- `"bottom-right"` - Lower right corner

### Value Formatting

Metrics are automatically formatted based on type:

```python
metrics = {
    "Integer": 42,           # → "Integer: 42"
    "Float": 3.14159,       # → "Float: 3.14"
    "String": "Running",    # → "String: Running"
    "Mixed": "T=273.15K"    # → "Mixed: T=273.15K"
}
```

### Integration Examples

**Field Simulation**
```python
metrics = {
    "Step": step_count,
    "Time": t,
    "Max Temp": np.max(field.data),
    "Min Temp": np.min(field.data),
    "Avg Temp": np.mean(field.data)
}
vis = visual.add_metrics(field_vis, metrics, position="top-left")
```

**Agent Simulation**
```python
metrics = {
    "Agents": len(particles.get('pos')),
    "Avg Speed": np.mean(np.linalg.norm(particles.get('vel'), axis=1)),
    "Kinetic E": total_kinetic_energy,
    "Potential E": total_potential_energy
}
vis = visual.add_metrics(agent_vis, metrics, position="top-right")
```

**Audio Analysis**
```python
metrics = {
    "Duration": "2.0 s",
    "Sample Rate": "44.1 kHz",
    "Window": 2048,
    "Hop": 512,
    "Scale": "dB"
}
spec_vis = visual.add_metrics(spec_vis, metrics, position="bottom-left")
```

**Network Analysis**
```python
n_edges = np.sum(graph.adj > 0) // 2
avg_degree = np.mean(np.sum(graph.adj > 0, axis=1))

metrics = {
    "Nodes": graph.n_nodes,
    "Edges": n_edges,
    "Avg Degree": f"{avg_degree:.2f}",
    "Type": "Small-world"
}
vis = visual.add_metrics(graph_vis, metrics, position="top-left")
```

---

## Integration Patterns

### Multi-Domain Analysis

Combine visualizations from different domains:

```python
# Audio + Spectrogram
audio_signal = generate_sound()
spec_vis = visual.spectrogram(audio_signal, palette="fire")

metrics = {
    "Type": "Chirp",
    "f0": "200 Hz",
    "f1": "2000 Hz",
    "Duration": "2.0 s"
}
spec_vis = visual.add_metrics(spec_vis, metrics)
visual.output(spec_vis, "audio_analysis.png")
```

```python
# Field + Agents + Metrics
field_vis = visual.colorize(temperature, palette="fire")
agent_vis = visual.agents(particles, alpha_property='alpha', blend_mode='additive')
combined = visual.composite(field_vis, agent_vis, mode='add')

metrics = {
    "Temperature": np.mean(temperature.data),
    "Particles": len(particles.get('pos')),
    "Frame": frame_number
}
final_vis = visual.add_metrics(combined, metrics)
```

### Time Series Analysis

Create spectrograms over time:

```python
for i, audio_chunk in enumerate(audio_chunks):
    spec_vis = visual.spectrogram(
        audio_chunk,
        window_size=2048,
        palette="viridis"
    )

    metrics = {"Chunk": i+1, "Time": f"{i*chunk_duration:.1f} s"}
    spec_vis = visual.add_metrics(spec_vis, metrics)

    visual.output(spec_vis, f"spec_chunk_{i:04d}.png")
```

### Network Evolution

Visualize network growth:

```python
for step, g in enumerate(network_sequence):
    vis = visual.graph(
        g,
        layout="force",
        iterations=100,
        color_by_centrality=True,
        palette="fire"
    )

    n_edges = np.sum(g.adj > 0) // 2
    metrics = {
        "Step": step,
        "Nodes": g.n_nodes,
        "Edges": n_edges,
        "Density": f"{2*n_edges/(g.n_nodes*(g.n_nodes-1)):.3f}"
    }
    vis = visual.add_metrics(vis, metrics)

    visual.output(vis, f"network_step_{step:04d}.png")
```

### Dynamical Systems Analysis

Track phase space evolution:

```python
trajectory_points = []

for step in range(n_steps):
    # Simulate system
    particles = simulate_step(particles, dt)

    # Collect phase space points
    trajectory_points.append({
        'pos': particles.get('pos').copy(),
        'vel': particles.get('vel').copy()
    })

# Visualize full trajectory
all_pos = np.vstack([p['pos'] for p in trajectory_points])
all_vel = np.vstack([p['vel'] for p in trajectory_points])

particles_all = agents.create(len(all_pos), pos=all_pos)
particles_all = agents.set(particles_all, 'vel', all_vel)

vis = visual.phase_space(
    particles_all,
    show_trajectories=True,
    color_property='time',
    palette='viridis'
)
```

---

## Performance Considerations

### Spectrogram

- **Window size**: Larger = better frequency resolution, slower computation
- **Hop size**: Smaller = better time resolution, more computation
- **Frequency range**: Filtering reduces output size

Optimal settings:
- Music: `window_size=2048, hop_size=512`
- Speech: `window_size=1024, hop_size=256`
- Transients: `window_size=512, hop_size=128`

### Graph Networks

- **Force-directed layout**: O(iterations × nodes²) complexity
  - Use 50-100 iterations for <100 nodes
  - Use 30-50 iterations for 100-500 nodes
  - Consider circular/grid for >500 nodes

- **Circular/Grid layouts**: O(nodes) complexity
  - Fast for any network size
  - Less aesthetically pleasing

### Phase Space

- **Point count**: Linear with number of agents
  - <1000 points: Fast rendering
  - 1000-10000 points: Moderate speed
  - >10000 points: Consider sampling

- **Trajectories**: Enable only for <500 points

### Metrics Dashboard

- Negligible overhead (<1ms for typical metrics)
- Text rendering is simplified raster-based

---

## Examples

Complete example programs are available in `examples/`:

1. **`spectrogram_visualization_demo.py`**
   - Chirp signals, harmonic series, percussive hits
   - Different window sizes and palettes
   - Frequency range filtering

2. **`graph_visualization_demo.py`**
   - Small-world, scale-free, star, and grid networks
   - All three layout algorithms
   - Centrality coloring

3. **`phase_space_visualization_demo.py`**
   - Harmonic oscillators, double pendulums
   - Orbital dynamics, Brownian motion
   - Energy and chaos coloring

4. **`advanced_visualization_showcase.py`**
   - Multi-domain integration
   - Combined visualizations
   - Real-world analysis workflows

---

## Color Palettes

Available palettes for all visualizations:

| Palette | Description | Best For |
|---------|-------------|----------|
| `grayscale` | Black to white | Simple contrast, publications |
| `fire` | Black → Red → Yellow → White | Heat maps, energy |
| `viridis` | Purple → Blue → Green → Yellow | Perceptually uniform, accessible |
| `coolwarm` | Blue → White → Red | Diverging data, ±values |

See [Palette Domain](./palette-domain.md) for custom palette creation.

---

## Best Practices

### Spectrogram
1. Use logarithmic scale (dB) for most audio
2. Match window size to analysis goal (frequency vs time resolution)
3. Filter to relevant frequency range
4. Add metrics showing analysis parameters

### Graph Networks
1. Use force-directed for unknown structure
2. Use circular for ring/symmetric networks
3. Use grid for spatial/lattice networks
4. Enable centrality coloring to reveal hubs
5. Add metrics showing network statistics

### Phase Space
1. Simulate long enough to reach steady state
2. Use energy/property coloring to reveal structure
3. Enable trajectories for <500 points
4. Match point size to density (smaller for dense plots)

### Metrics Dashboard
1. Keep to 4-8 metrics for readability
2. Use consistent formatting (units, precision)
3. Position to avoid obscuring visualization
4. Update metrics every frame for animations

---

## Troubleshooting

### Spectrogram appears blank
- Check signal amplitude (should be normalized to [-1, 1])
- Verify sample_rate matches audio data
- Try linear scale instead of log scale
- Check frequency range includes signal content

### Graph layout looks messy
- Increase iterations (50 → 150)
- Try different layout algorithm
- Reduce node size for dense networks
- Check graph is connected

### Phase space shows single point
- Verify positions and velocities have variance
- Check property names match ('pos', 'vel')
- Simulate system for multiple timesteps
- Ensure initial conditions are diverse

### Metrics not visible
- Check position doesn't overlap visualization features
- Increase font_size (14 → 18)
- Adjust bg_alpha for better contrast
- Try different text_color

---

## Future Enhancements (Roadmap)

### v0.12.0
- [ ] Wavelet transform visualization
- [ ] 3D network layouts (spring embedding)
- [ ] Poincaré sections for phase space
- [ ] Interactive metric overlays (click to toggle)

### v0.13.0
- [ ] Custom palettes from images
- [ ] Node labels in graph visualization
- [ ] Phase space with vector field overlay
- [ ] Real-time metric streaming

### v0.14.0
- [ ] GPU-accelerated force-directed layout
- [ ] Community detection coloring for graphs
- [ ] Multi-dimensional phase space (3D scatter)
- [ ] Animated metrics (smooth transitions)

---

## External Analysis Tools

### Modal Decomposition with PyDMD

While Morphogen provides built-in spectral analysis (`spectrogram`, `fft`, `stft`), **external tools like PyDMD** can reveal deeper mathematical structure through **Dynamic Mode Decomposition (DMD)**.

**What DMD Offers:**
- Extract coherent spatial-temporal modes from simulations
- Reveal dominant patterns driving complex dynamics
- Create explanatory animations (mode galleries, progressive reconstruction)
- Compare modes across domains (e.g., do fluid modes match audio spectrum?)
- Compress simulations (reconstruct with 5-10 modes instead of 10,000 timesteps)

**When to Use DMD:**
- ✅ Understanding *what patterns drive* behavior (not just visualizing output)
- ✅ Creating showcase animations for papers/presentations
- ✅ Detecting bifurcations, attractors, and regime changes
- ✅ Cross-domain mode correlation studies

**Workflow:**
1. Export Morphogen simulation data (NumPy arrays)
2. Run PyDMD analysis externally
3. Visualize modes and create animations
4. Optionally: Feed insights back into Morphogen design

**Learn More:**
📘 **[Analysis and Visualization Guide](../guides/analysis-visualization.md)** - Complete PyDMD tutorial with Morphogen examples

**Example Use Cases:**
- `examples/cross_domain/fluid_acoustics_audio.py` - 3-domain pipeline perfect for DMD mode correlation
- `examples/reaction_diffusion.py` - Spatial pattern modes
- `examples/smoke_simulation.py` - Turbulent flow vortex modes

**Resources:**
- PyDMD: [https://github.com/PyDMD/PyDMD](https://github.com/PyDMD/PyDMD)
- PyDMD Docs: [https://pydmd.github.io/PyDMD/](https://pydmd.github.io/PyDMD/)

---

## See Also

### Related Documentation
- **[Analysis and Visualization Guide](../guides/analysis-visualization.md)** ⭐ NEW - Complete guide to analyzing Morphogen simulations
- [Mathematical Transformation Metaphors](./math-transformation-metaphors.md) - Intuitive frameworks for understanding transforms
- [Visualization Ideas by Domain](./visualization-ideas-by-domain.md) - Comprehensive visualization catalog
- [Transform Specification](../specifications/transform.md) - Technical transform details

### Domain Documentation
- [Visual Domain](./visual-scene-domain.md) - Core visualization architecture
- [Field Operations](./field-operations.md) - Field-based visualizations
- [Agents Domain](./agents-domain.md) - Agent visualization
- [Audio Domain](./audio-domain.md) - Audio synthesis and analysis
- [Graph Domain](./graph-domain.md) - Network algorithms

### Examples
- [Examples](../../examples/) - Complete demonstration programs
