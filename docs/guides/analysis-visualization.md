# Analysis and Visualization Guide

**Purpose**: Guide to analyzing and visualizing Morphogen simulation results using external tools
**Audience**: Users wanting deeper insight into simulation dynamics
**Last Updated**: 2026-01-05

---

## Overview

Morphogen generates rich spatiotemporal dynamics across multiple domains. While Morphogen provides built-in visualization (`colorize()`, `render()`, etc.), external analysis tools can reveal deeper mathematical structure and create compelling animations.

This guide covers:
1. Built-in analysis capabilities (what's already available)
2. Modal decomposition with PyDMD (revealing coherent structures)
3. Spectral analysis workflows (understanding frequency content)
4. Creating showcase animations (for presentations, papers, marketing)

---

## Built-In Analysis Capabilities

**Before reaching for external tools**, leverage Morphogen's native analysis operators:

### Audio Analysis Domain

```morphogen
use audio_analysis

# Extract modal structure from audio recordings
modes = analyze_modes(signal, sample_rate, num_modes=20)
# Returns: ModalModel with frequencies, amplitudes, decay_rates, phases

# Track fundamental frequency over time
f0 = track_fundamental(signal, sample_rate, method="autocorrelation")

# Measure inharmonicity (important for instrument modeling)
inharmonicity = measure_inharmonicity(signal, sample_rate, f0=440.0)

# Extract spectral envelope (timbre signature)
envelope = spectral_envelope(stft_result, smoothing_factor=0.1)
```

**When to use**: Instrument modeling, timbre analysis, physical modeling synthesis

**See**: `morphogen/stdlib/audio_analysis.py` for full operator list

### Signal Domain

```morphogen
use signal

# FFT analysis
spectrum = fft(signal)
spectrogram = stft(signal, frame_size=2048, hop_size=512)

# Filtering and transforms
filtered = bandpass(signal, low=100.0, high=2000.0, sample_rate)
coeffs = dct(signal)  # Discrete cosine transform
```

**When to use**: Frequency analysis, filter design, transform-domain processing

### Graph Domain

```morphogen
use graph

# Network analysis
centrality = betweenness_centrality(graph)
communities = detect_communities(graph)
paths = all_shortest_paths(graph, source, target)
```

**When to use**: Network dynamics, connectivity analysis, agent interaction patterns

---

## External Analysis: Dynamic Mode Decomposition (DMD)

### What is DMD?

**Dynamic Mode Decomposition** extracts coherent spatial-temporal structures from time-series data. Think of it as:
- **FFT for spatiotemporal patterns** (not just time or space alone)
- **Principal component analysis with dynamics** (modes have frequencies and growth rates)
- **A "spectral microscope"** revealing hidden structure in complex simulations

### When DMD is Valuable

**Use DMD when you want to:**
- ✅ Understand **what patterns drive** complex behavior (not just visualize output)
- ✅ Compare **modes across domains** (do fluid modes match audio spectrum?)
- ✅ Create **explanatory animations** (show decomposition, not just final result)
- ✅ **Compress simulations** (reconstruct with 5-10 modes instead of 10,000 timesteps)
- ✅ **Detect regime changes** (bifurcations, instabilities, attractors)

**Don't use DMD when:**
- ❌ Built-in spectral analysis (`fft`, `stft`) already solves your problem
- ❌ You just need pretty output (Morphogen's `colorize()` is simpler)
- ❌ Simulation is too chaotic (DMD works best on structured dynamics)

### Installation

```bash
pip install pydmd
```

**PyDMD**: [https://github.com/PyDMD/PyDMD](https://github.com/PyDMD/PyDMD)
**Documentation**: [https://pydmd.github.io/PyDMD/](https://pydmd.github.io/PyDMD/)

---

## Tutorial 1: Basic DMD Workflow

### Step 1: Export Morphogen Data

Run your Morphogen simulation and save snapshots:

```python
# morphogen_simulation.py
import numpy as np
from morphogen.stdlib import field, visual

# Run simulation
grid_size = 128
timesteps = 1000
snapshots = []

# Your Morphogen code here (simplified example)
temp = field.random_normal(seed=42, shape=(grid_size, grid_size))

for t in range(timesteps):
    temp = field.diffuse(temp, rate=0.1, dt=0.01, iterations=20)
    snapshots.append(temp)  # Store each timestep

# Convert to NumPy array: (spatial_points, timesteps)
X = np.array(snapshots).T  # Shape: (128*128, 1000)
X = X.reshape(grid_size * grid_size, timesteps)

# Save for DMD analysis
np.save('morphogen_data.npy', X)
print(f"Saved data: {X.shape} (spatial points × timesteps)")
```

### Step 2: Perform DMD Analysis

```python
# dmd_analysis.py
import numpy as np
from pydmd import DMD
from pydmd.plotter import plot_summary
import matplotlib.pyplot as plt

# Load Morphogen data
X = np.load('morphogen_data.npy')
print(f"Loaded data: {X.shape}")

# Initialize DMD
dmd = DMD(svd_rank=10)  # Extract top 10 modes
dmd.fit(X)

# Summary plot (modes, eigenvalues, reconstruction)
plot_summary(dmd, x=np.arange(X.shape[1]))
plt.savefig('dmd_summary.png', dpi=300)
plt.show()

# Access individual modes
modes = dmd.modes  # Shape: (spatial_points, num_modes)
eigenvalues = dmd.eigs  # Complex eigenvalues (frequency + growth)
amplitudes = dmd.amplitudes  # Mode strengths

print(f"Extracted {len(eigenvalues)} modes")
print(f"Mode frequencies (Hz): {np.angle(eigenvalues) / (2 * np.pi)}")
print(f"Mode growth rates: {np.log(np.abs(eigenvalues))}")
```

### Step 3: Visualize Individual Modes

```python
# Reshape modes back to 2D spatial grid
grid_size = 128
num_modes = dmd.modes.shape[1]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(min(10, num_modes)):
    ax = axes[i // 5, i % 5]
    mode_2d = dmd.modes[:, i].real.reshape(grid_size, grid_size)
    im = ax.imshow(mode_2d, cmap='RdBu_r', vmin=-np.abs(mode_2d).max(),
                   vmax=np.abs(mode_2d).max())
    ax.set_title(f"Mode {i+1}\nω={np.angle(dmd.eigs[i])/(2*np.pi):.2f} Hz")
    ax.axis('off')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('dmd_modes_gallery.png', dpi=300)
plt.show()
```

---

## Tutorial 2: Cross-Domain Mode Correlation

**Showcase Example**: Fluid → Acoustics → Audio pipeline with mode comparison

### Morphogen Simulation

```python
# cross_domain_dmd.py
import numpy as np
from morphogen.stdlib import field, acoustics, audio

# Simulate fluid dynamics
grid_size = 128
timesteps = 500
fluid_snapshots = []
audio_samples = []

pressure = field.zeros((grid_size, grid_size))

for t in range(timesteps):
    # Fluid simulation (simplified)
    pressure = field.diffuse(pressure, rate=0.1, dt=0.01)
    fluid_snapshots.append(pressure)

    # Extract audio from pressure field at "microphone" position
    mic_pos = (grid_size // 2, grid_size // 2)
    audio_sample = pressure[mic_pos[0], mic_pos[1]]
    audio_samples.append(audio_sample)

# Save both domains
fluid_data = np.array(fluid_snapshots).T.reshape(grid_size * grid_size, timesteps)
audio_data = np.array(audio_samples)

np.save('fluid_pressure.npy', fluid_data)
np.save('audio_signal.npy', audio_data)
```

### DMD Analysis: Fluid Domain

```python
from pydmd import DMD

# Fluid DMD
fluid_X = np.load('fluid_pressure.npy')
dmd_fluid = DMD(svd_rank=10)
dmd_fluid.fit(fluid_X)

# Extract frequencies
fluid_freqs = np.angle(dmd_fluid.eigs) / (2 * np.pi)
print(f"Fluid mode frequencies: {fluid_freqs}")
```

### Spectral Analysis: Audio Domain

```python
from scipy.fft import fft, fftfreq

# Audio FFT
audio_signal = np.load('audio_signal.npy')
sample_rate = 1000  # Adjust based on your dt
audio_fft = fft(audio_signal)
audio_freqs = fftfreq(len(audio_signal), 1/sample_rate)

# Find dominant frequencies
dominant_idx = np.argsort(np.abs(audio_fft))[-10:]
dominant_freqs = audio_freqs[dominant_idx]
print(f"Audio dominant frequencies: {dominant_freqs}")
```

### Cross-Domain Comparison

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Fluid modes (spatial patterns)
mode_2d = dmd_fluid.modes[:, 0].real.reshape(128, 128)
ax1.imshow(mode_2d, cmap='RdBu_r')
ax1.set_title(f'Fluid Mode 1: {fluid_freqs[0]:.2f} Hz')

# Audio spectrum
ax2.plot(audio_freqs[:len(audio_freqs)//2],
         np.abs(audio_fft)[:len(audio_fft)//2])
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Audio Spectrum')
ax2.axvline(fluid_freqs[0], color='red', linestyle='--',
            label=f'Fluid Mode 1: {fluid_freqs[0]:.2f} Hz')
ax2.legend()

plt.tight_layout()
plt.savefig('cross_domain_comparison.png', dpi=300)
plt.show()
```

**What to look for**: Do fluid mode frequencies match peaks in audio spectrum? This proves cross-domain coupling is working correctly!

---

## Creating Showcase Animations

### Animation 1: Mode Decomposition Gallery

**Purpose**: Show how a complex simulation is built from simple modes
**Duration**: 10-15 seconds
**Format**: 3×3 grid of spatial modes

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

def update(frame):
    for i in range(9):
        axes[i].clear()
        mode_2d = dmd.modes[:, i].real.reshape(128, 128)
        # Animate phase
        mode_with_time = mode_2d * np.cos(2 * np.pi * frame / 100)
        axes[i].imshow(mode_with_time, cmap='RdBu_r',
                      vmin=-np.abs(mode_2d).max(), vmax=np.abs(mode_2d).max())
        axes[i].set_title(f'Mode {i+1}')
        axes[i].axis('off')
    return axes

anim = FuncAnimation(fig, update, frames=100, interval=50)
anim.save('mode_gallery_animated.mp4', fps=20, dpi=150)
```

### Animation 2: Progressive Reconstruction

**Purpose**: Show simulation = sum of modes
**Duration**: 20 seconds
**Format**: Side-by-side (original vs reconstruction with N modes)

```python
# Reconstruct with increasing number of modes
original = X[:, 0].reshape(128, 128)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

def update(num_modes):
    # Reconstruct with 1...N modes
    X_reconstructed = dmd.reconstructed_data[:, 0]
    # (Simplified - actual reconstruction needs proper DMD method)

    ax1.clear()
    ax1.imshow(original, cmap='viridis')
    ax1.set_title('Original Simulation')
    ax1.axis('off')

    ax2.clear()
    reconstructed_2d = X_reconstructed.real.reshape(128, 128)
    ax2.imshow(reconstructed_2d, cmap='viridis')
    ax2.set_title(f'Reconstruction ({num_modes} modes)')
    ax2.axis('off')

    ax3.clear()
    error = np.abs(original - reconstructed_2d)
    ax3.imshow(error, cmap='hot')
    ax3.set_title(f'Error: {np.mean(error):.2e}')
    ax3.axis('off')

anim = FuncAnimation(fig, update, frames=range(1, 11), interval=1000)
anim.save('progressive_reconstruction.mp4', fps=1, dpi=150)
```

### Animation 3: Temporal Evolution

**Purpose**: Show how modes oscillate/decay over time
**Duration**: 30 seconds loop
**Format**: 2×2 grid of top 4 modes

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

def update(frame):
    t = frame / 30.0  # Time in seconds
    for i in range(4):
        axes[i].clear()
        mode_2d = dmd.modes[:, i].real.reshape(128, 128)

        # Apply temporal evolution: e^(λt)
        eigenvalue = dmd.eigs[i]
        temporal_factor = np.exp(eigenvalue * t)
        mode_with_time = mode_2d * temporal_factor.real

        axes[i].imshow(mode_with_time, cmap='RdBu_r',
                      vmin=-2*np.abs(mode_2d).max(),
                      vmax=2*np.abs(mode_2d).max())
        axes[i].set_title(f'Mode {i+1} at t={t:.2f}s\n'
                         f'ω={np.angle(eigenvalue)/(2*np.pi):.2f} Hz')
        axes[i].axis('off')
    return axes

anim = FuncAnimation(fig, update, frames=900, interval=33)  # 30fps
anim.save('mode_evolution.mp4', fps=30, dpi=150)
```

---

## Advanced Techniques

### Multi-Resolution DMD (mrDMD)

For simulations with multiple timescales (e.g., fast audio + slow thermal):

```python
from pydmd import MrDMD

# Multi-resolution DMD
mrdmd = MrDMD(max_level=3, max_cycles=2)
mrdmd.fit(X)

# Extract modes at different timescales
for level in range(mrdmd.max_level):
    print(f"Level {level} modes: {mrdmd.modes[level].shape}")
```

### Hankel DMD (for Attractors)

For detecting limit cycles and attractors:

```python
from pydmd import HankelDMD

# Hankel DMD (delay embedding)
hdmd = HankelDMD(svd_rank=10, d=50)  # d = delay dimension
hdmd.fit(X)
```

### Sparsity-Promoting DMD

For extracting only the most important modes:

```python
from pydmd import SpDMD

# Sparse DMD (L1 regularization)
spdmd = SpDMD(svd_rank=50, rho=1.0)  # rho = sparsity weight
spdmd.fit(X)
print(f"Selected {np.sum(np.abs(spdmd.amplitudes) > 1e-6)} of {len(spdmd.amplitudes)} modes")
```

---

## Integration Roadmap

### Current Status (v0.12.0)

**Tier 1: External Workflow** ✅ (This guide)
- Users export Morphogen data manually
- Run PyDMD in separate Python scripts
- No code changes to Morphogen

### Future Possibilities

**Tier 2: Analysis Domain** (Post-v1.0)
```morphogen
use field, analysis

flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)

    # Capture snapshots for DMD
    analysis.record_snapshot(temp)
}

# After simulation
modes = analysis.dmd(num_modes=10)
output modes.visualize(palette="viridis")
```

**Tier 3: Real-Time Visualization** (Future)
```morphogen
flow(dt=0.01, dual_viz=true) {
    temp = diffuse(temp, rate=0.1, dt)

    output_left colorize(temp)           # Direct output
    output_right dmd_modes(temp, top_k=5) # Live mode decomposition
}
```

**Status**: Tier 2/3 will be evaluated based on community demand post-v1.0 release.

---

## Best Practices

### Data Export

**✅ Do:**
- Save data in NumPy `.npy` format (fast, lossless)
- Include metadata (grid size, timesteps, sample rate)
- Normalize spatial dimensions to avoid numerical issues
- Use consistent timestep intervals (DMD assumes uniform sampling)

**❌ Avoid:**
- CSV export (slow, precision loss)
- Irregular timesteps (DMD needs uniform intervals)
- Mixed data types (keep spatial data separate from scalars)

### Mode Selection

**How many modes?**
- Start with `svd_rank=-1` (automatic selection based on energy)
- Manually select based on energy spectrum: 90-99% cumulative energy
- Typical range: 5-20 modes for most Morphogen simulations
- More modes ≠ better (can include noise)

### Visualization

**Make it interpretable:**
- Always label modes with frequency and growth rate
- Use diverging colormaps (`RdBu_r`) for modes (have positive/negative regions)
- Show energy spectrum (which modes matter most?)
- Compare reconstruction to original (validate quality)

---

## Performance Tips

### Memory Management

```python
# For large grids, subsample spatially
X_downsampled = X[::2, :]  # Keep every 2nd spatial point

# Or temporally
X_downsampled = X[:, ::2]  # Keep every 2nd timestep
```

### Faster DMD Variants

```python
# Standard DMD: O(n³) for n spatial points
dmd = DMD(svd_rank=10)

# Forward-Backward DMD (more stable for noisy data)
from pydmd import FbDMD
fbdmd = FbDMD(svd_rank=10)

# Optimized DMD (faster convergence)
from pydmd import OptDMD
optdmd = OptDMD(svd_rank=10)
```

---

## Example Gallery

**See these working examples:**
- `examples/cross_domain/fluid_acoustics_audio.py` - 3-domain pipeline perfect for DMD
- `examples/reaction_diffusion.py` - Spatial pattern formation (beautiful modes!)
- `examples/smoke_simulation.py` - Turbulent flow (shows vortex modes)
- `examples/phase_space_visualization_demo.py` - Attractor detection with Hankel DMD

---

## References

**PyDMD Resources:**
- GitHub: [https://github.com/PyDMD/PyDMD](https://github.com/PyDMD/PyDMD)
- Documentation: [https://pydmd.github.io/PyDMD/](https://pydmd.github.io/PyDMD/)
- Paper: Tezzele et al. (2024) "PyDMD: A Python package for robust dynamic mode decomposition"

**DMD Theory:**
- Kutz et al. (2016) "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems"
- Brunton & Kutz (2019) "Data-Driven Science and Engineering" (Chapter 7)

**Morphogen Documentation:**
- `docs/reference/advanced-visualizations.md` - Visualization techniques
- `docs/guides/output-generation.md` - Exporting simulation data
- `morphogen/stdlib/audio_analysis.py` - Built-in modal analysis operators

---

## Getting Help

**Questions or showcase examples to share?**
- GitHub Discussions: [https://github.com/scottsen/morphogen/discussions](https://github.com/scottsen/morphogen/discussions)
- Issues: [https://github.com/scottsen/morphogen/issues](https://github.com/scottsen/morphogen/issues)
- Tag your DMD visualizations with `#MorphogenDMD` on social media!

---

**Last Updated**: 2026-01-05
**Maintainer**: Morphogen Documentation Team
**Next Review**: Post-v1.0 (evaluate Tier 2 integration based on usage)
