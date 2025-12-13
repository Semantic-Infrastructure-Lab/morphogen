"""Real-Time Audio Visualizer - Cross-Domain Showcase

This example demonstrates real-time audio visualization patterns using
multiple Morphogen domains working together:

- Audio: Synthesis and signal processing
- Field: Diffusion and field-based effects
- Cellular: Audio-reactive cellular automata
- Palette: Color mapping for visualization
- Visual: Rendering and composition
- Signal: FFT spectral analysis

The demo shows how temporal audio data can drive spatial visual effects,
creating stunning synchronized visualizations.

Cross-Domain Integration Patterns:
1. Audio → FFT → Field (spectral energy driving diffusion)
2. Audio → Cellular (amplitude controlling cell birth/death)
3. Audio → Palette (frequency-based color selection)
4. Field + Cellular → Visual (spatial composition)

Run: python examples/audio_visualizer/real_time_demo.py
"""

import numpy as np
from pathlib import Path
from morphogen.stdlib import (
    audio, field, cellular, palette, visual,
    signal, noise, image, color
)
from morphogen.stdlib.field import Field2D
from morphogen.stdlib.audio import AudioBuffer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_field(f: Field2D, vmin: float = 0.0, vmax: float = 1.0) -> Field2D:
    """Normalize field to target range."""
    data = f.data.copy()
    data_min, data_max = data.min(), data.max()

    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
        data = data * (vmax - vmin) + vmin

    return Field2D(data)


def extract_amplitude_envelope(audio_buffer: AudioBuffer, window_size: int = 2048) -> np.ndarray:
    """Extract amplitude envelope from audio buffer."""
    data = audio_buffer.data
    num_windows = len(data) // window_size

    envelope = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        envelope[i] = np.sqrt(np.mean(data[start:end] ** 2))  # RMS

    return envelope


def extract_spectral_centroid(audio_buffer: AudioBuffer, n_bands: int = 32) -> np.ndarray:
    """Extract spectral centroid over time (frequency center of mass)."""
    # Simple FFT-based spectral analysis
    window_size = 2048
    hop_size = window_size // 2
    data = audio_buffer.data

    num_frames = (len(data) - window_size) // hop_size + 1
    centroids = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        # Windowed FFT
        window = data[start:end] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(window))

        # Compute centroid
        freqs = np.arange(len(spectrum))
        if spectrum.sum() > 0:
            centroids[i] = np.sum(freqs * spectrum) / np.sum(spectrum)

    # Normalize to 0-1
    if centroids.max() > 0:
        centroids = centroids / centroids.max()

    return centroids


# ============================================================================
# CROSS-DOMAIN VISUALIZATION MODES
# ============================================================================

def mode1_spectral_field_diffusion(audio_buffer: AudioBuffer,
                                    width: int = 512,
                                    height: int = 512) -> visual.Visual:
    """Mode 1: Audio Spectrum → Field Diffusion → Visual

    Audio spectral energy creates heat sources that diffuse through a field.
    Demonstrates: Audio → Signal → Field → Palette → Visual
    """
    print("  Mode 1: Spectral Field Diffusion")

    # Extract spectral features
    window_size = 2048
    hop_size = 512
    data = audio_buffer.data

    # Initialize field
    heat_field = field.alloc((height, width), dtype=np.float32, fill_value=0.0)

    # Process audio in windows
    num_windows = min(100, (len(data) - window_size) // hop_size + 1)

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        # FFT analysis
        window = data[start:end] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(window))
        spectrum = spectrum[:64]  # Use lower frequencies

        # Normalize
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        # Add heat sources based on spectrum
        for freq_idx, magnitude in enumerate(spectrum):
            if magnitude > 0.3:
                # Position based on frequency
                x = int((freq_idx / len(spectrum)) * width)
                y = height // 2

                # Add heat spot
                radius = 10
                y_start = max(0, y - radius)
                y_end = min(height, y + radius)
                x_start = max(0, x - radius)
                x_end = min(width, x + radius)

                heat_field.data[y_start:y_end, x_start:x_end] += magnitude * 2.0

        # Diffuse
        heat_field = field.diffuse(heat_field, diffusion_coeff=0.2, dt=0.05)

        # Decay
        heat_field.data *= 0.95

    # Normalize
    heat_field = normalize_field(heat_field)

    # Colorize with hot palette
    pal = palette.fire(256)
    img = palette.map(pal, heat_field.data)

    return visual.Visual(img)


def mode2_audio_reactive_cellular(audio_buffer: AudioBuffer,
                                   size: int = 512,
                                   n_steps: int = 200) -> visual.Visual:
    """Mode 2: Audio Amplitude → Cellular Automata → Visual

    Audio amplitude controls cellular automata evolution.
    Demonstrates: Audio → Cellular → Palette → Visual
    """
    print("  Mode 2: Audio-Reactive Cellular Automata")

    # Extract amplitude envelope
    envelope = extract_amplitude_envelope(audio_buffer, window_size=4096)

    # Initialize CA with Game of Life rules
    ca_field, rule = cellular.game_of_life((size, size), density=0.2, seed=42)

    # Evolve CA with audio reactivity
    steps_per_env = max(1, n_steps // len(envelope))

    for env_idx, amplitude in enumerate(envelope[:n_steps // steps_per_env]):
        for _ in range(steps_per_env):
            # Evolve
            ca_field = cellular.step(ca_field, rule)

            # Add cells based on audio amplitude
            if amplitude > 0.15:
                num_cells = int(amplitude * 200)
                rng = np.random.RandomState(env_idx)

                for _ in range(num_cells):
                    x = rng.randint(0, size)
                    y = rng.randint(0, size)
                    ca_field.data[y, x] = 1

    # Colorize
    pal = palette.magma(256)
    img = palette.map(pal, ca_field.data.astype(np.float32))

    return visual.Visual(img)


def mode3_frequency_field_visualization(audio_buffer: AudioBuffer,
                                        width: int = 512,
                                        height: int = 512) -> visual.Visual:
    """Mode 3: Audio Frequencies → Field Patterns → Visual

    Different frequency bands create different field patterns.
    Demonstrates: Audio → Field → Noise → Palette → Visual
    """
    print("  Mode 3: Frequency Field Visualization")

    # Create three fields for low, mid, high frequencies
    field_low = field.alloc((height, width), dtype=np.float32, fill_value=0.0)
    field_mid = field.alloc((height, width), dtype=np.float32, fill_value=0.0)
    field_high = field.alloc((height, width), dtype=np.float32, fill_value=0.0)

    # Process audio
    window_size = 4096
    hop_size = 2048
    data = audio_buffer.data
    num_windows = min(50, (len(data) - window_size) // hop_size + 1)

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        # FFT
        window = data[start:end] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(window))

        # Split into bands
        n_bins = len(spectrum)
        low_band = spectrum[:n_bins // 3]
        mid_band = spectrum[n_bins // 3:2 * n_bins // 3]
        high_band = spectrum[2 * n_bins // 3:]

        # Energy per band
        energy_low = np.mean(low_band)
        energy_mid = np.mean(mid_band)
        energy_high = np.mean(high_band)

        # Add to fields at different positions
        if energy_low > 0.1:
            # Low frequencies at center
            cx, cy = width // 2, height // 2
            radius = int(energy_low * 100)
            y_grid, x_grid = np.ogrid[:height, :width]
            mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
            field_low.data[mask] += energy_low * 5.0

        if energy_mid > 0.1:
            # Mid frequencies in ring pattern
            cx, cy = width // 2, height // 2
            radius = int(150 + energy_mid * 50)
            y_grid, x_grid = np.ogrid[:height, :width]
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            mask = np.abs(dist - radius) < 20
            field_mid.data[mask] += energy_mid * 3.0

        if energy_high > 0.1:
            # High frequencies scattered
            for _ in range(int(energy_high * 10)):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                field_high.data[max(0, y-5):min(height, y+5),
                               max(0, x-5):min(width, x+5)] += energy_high

        # Diffuse all fields
        field_low = field.diffuse(field_low, diffusion_coeff=0.15, dt=0.1)
        field_mid = field.diffuse(field_mid, diffusion_coeff=0.10, dt=0.1)
        field_high = field.diffuse(field_high, diffusion_coeff=0.05, dt=0.1)

        # Decay
        field_low.data *= 0.92
        field_mid.data *= 0.90
        field_high.data *= 0.88

    # Normalize fields
    field_low = normalize_field(field_low)
    field_mid = normalize_field(field_mid)
    field_high = normalize_field(field_high)

    # Create RGB composite (R=low, G=mid, B=high)
    rgb_img = np.stack([
        field_low.data,
        field_mid.data,
        field_high.data
    ], axis=-1)

    # Scale to 0-255
    rgb_img = (rgb_img * 255).astype(np.uint8)

    return visual.Visual(rgb_img)


def mode4_beat_synchronized_patterns(audio_buffer: AudioBuffer,
                                     size: int = 512) -> visual.Visual:
    """Mode 4: Beat Detection → Pattern Generation → Visual

    Detects beats and creates synchronized visual patterns.
    Demonstrates: Audio → Field → Palette → Visual
    """
    print("  Mode 4: Beat-Synchronized Patterns")

    # Simple beat detection (energy threshold)
    window_size = 2048
    hop_size = 512
    data = audio_buffer.data
    num_windows = (len(data) - window_size) // hop_size + 1

    # Compute energy envelope
    energy = []
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        window_energy = np.sum(data[start:end] ** 2)
        energy.append(window_energy)

    energy = np.array(energy)

    # Normalize
    if energy.max() > 0:
        energy = energy / energy.max()

    # Detect beats (simple threshold)
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    beats = energy > threshold

    # Create pattern field
    pattern_field = field.alloc((size, size), dtype=np.float32, fill_value=0.0)

    # Add patterns at beat times
    cx, cy = size // 2, size // 2

    for i, is_beat in enumerate(beats):
        if is_beat:
            # Add radial pattern
            radius = int(20 + energy[i] * 80)

            # Create radial gradient
            y_grid, x_grid = np.ogrid[:size, :size]
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

            # Ring pattern
            ring = np.exp(-((dist - radius) ** 2) / (2 * 10 ** 2))
            pattern_field.data += ring * energy[i] * 2.0

    # Add some noise texture
    noise_field = noise.perlin((size, size), scale=8.0, seed=42)
    pattern_field.data += noise_field.data * 0.3

    # Normalize
    pattern_field = normalize_field(pattern_field)

    # Colorize with viridis
    pal = palette.viridis(256)
    img = palette.map(pal, pattern_field.data)

    return visual.Visual(img)


# ============================================================================
# MAIN DEMO
# ============================================================================

def create_test_audio(duration: float = 5.0,
                      sample_rate: int = 44100) -> AudioBuffer:
    """Create test audio with multiple frequency components and rhythm."""
    print("  Generating test audio...")

    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Musical sequence with rhythm
    audio_data = np.zeros_like(t)

    # Chord progression (C, F, G, C)
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [349.23, 440.00, 523.25],  # F major
        [392.00, 493.88, 587.33],  # G major
        [261.63, 329.63, 392.00],  # C major
    ]

    chord_duration = duration / len(chords)

    for chord_idx, chord_freqs in enumerate(chords):
        start_time = chord_idx * chord_duration
        end_time = (chord_idx + 1) * chord_duration
        mask = (t >= start_time) & (t < end_time)

        # Add chord tones
        for freq in chord_freqs:
            note_t = t[mask] - start_time
            envelope = np.exp(-2 * note_t)
            audio_data[mask] += np.sin(2 * np.pi * freq * note_t) * envelope * 0.3

    # Add rhythmic kick drum
    for beat_time in np.arange(0, duration, 0.5):
        beat_idx = int(beat_time * sample_rate)
        decay_samples = int(0.2 * sample_rate)

        if beat_idx + decay_samples < len(audio_data):
            decay = np.exp(-10 * np.arange(decay_samples) / sample_rate)
            kick = np.sin(2 * np.pi * 60 * np.arange(decay_samples) / sample_rate) * decay
            audio_data[beat_idx:beat_idx + decay_samples] += kick * 0.4

    # Add hi-hat pattern
    for beat_time in np.arange(0, duration, 0.25):
        beat_idx = int(beat_time * sample_rate)
        hihat_samples = int(0.05 * sample_rate)

        if beat_idx + hihat_samples < len(audio_data):
            hihat = np.random.randn(hihat_samples) * np.exp(-50 * np.arange(hihat_samples) / sample_rate)
            audio_data[beat_idx:beat_idx + hihat_samples] += hihat * 0.15

    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8

    print(f"  ✓ Audio: {duration}s, {sample_rate}Hz")

    return AudioBuffer(data=audio_data, sample_rate=sample_rate)


def main():
    """Run audio visualizer demonstration."""
    print("=" * 70)
    print("REAL-TIME AUDIO VISUALIZER - CROSS-DOMAIN SHOWCASE")
    print("=" * 70)
    print()
    print("Demonstrating cross-domain integration:")
    print("  • Audio synthesis and analysis")
    print("  • Field-based diffusion effects")
    print("  • Audio-reactive cellular automata")
    print("  • Frequency-based color mapping")
    print("  • Beat-synchronized patterns")
    print()

    # Create output directory
    output_dir = Path("examples/audio_visualizer/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test audio
    print("Step 1: Audio Generation")
    print("-" * 70)
    audio_buffer = create_test_audio(duration=5.0)

    # Save audio
    audio_file = output_dir / "test_audio.wav"
    audio.save(audio_buffer, str(audio_file))
    print(f"  ✓ Saved: {audio_file}")
    print()

    # Mode 1: Spectral field diffusion
    print("Step 2: Visualization Mode 1")
    print("-" * 70)
    vis1 = mode1_spectral_field_diffusion(audio_buffer, width=512, height=512)
    output1 = output_dir / "mode1_spectral_diffusion.png"
    image.save(vis1.data, str(output1))
    print(f"  ✓ Saved: {output1}")
    print()

    # Mode 2: Audio-reactive cellular
    print("Step 3: Visualization Mode 2")
    print("-" * 70)
    vis2 = mode2_audio_reactive_cellular(audio_buffer, size=512, n_steps=200)
    output2 = output_dir / "mode2_cellular_automata.png"
    image.save(vis2.data, str(output2))
    print(f"  ✓ Saved: {output2}")
    print()

    # Mode 3: Frequency field visualization
    print("Step 4: Visualization Mode 3")
    print("-" * 70)
    vis3 = mode3_frequency_field_visualization(audio_buffer, width=512, height=512)
    output3 = output_dir / "mode3_frequency_fields.png"
    image.save(vis3.data, str(output3))
    print(f"  ✓ Saved: {output3}")
    print()

    # Mode 4: Beat-synchronized patterns
    print("Step 5: Visualization Mode 4")
    print("-" * 70)
    vis4 = mode4_beat_synchronized_patterns(audio_buffer, size=512)
    output4 = output_dir / "mode4_beat_patterns.png"
    image.save(vis4.data, str(output4))
    print(f"  ✓ Saved: {output4}")
    print()

    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Cross-Domain Integration Demonstrated:")
    print("  ✓ Audio → Field: Spectral energy driving diffusion")
    print("  ✓ Audio → Cellular: Amplitude controlling CA evolution")
    print("  ✓ Audio → Field: Frequency bands creating patterns")
    print("  ✓ Audio → Visual: Beat detection synchronizing graphics")
    print()
    print(f"All outputs saved to: {output_dir}/")
    print()
    print("Key Insights:")
    print("  • Temporal audio data naturally drives spatial visual effects")
    print("  • Different frequency ranges can control different visual layers")
    print("  • Beat detection enables synchronized pattern generation")
    print("  • Field diffusion creates smooth, organic visual flow")
    print("  • Cellular automata add emergent complexity to visualizations")


if __name__ == "__main__":
    main()
