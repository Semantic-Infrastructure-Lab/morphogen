"""Audio Visualizer - Advanced Cross-Domain Showcase

This example demonstrates the power of combining multiple Morphogen domains:
- Audio synthesis and processing
- FFT spectral analysis
- Field operations for audio-reactive effects
- Cellular automata driven by audio
- Palette and color for stunning visuals
- Image composition and export
- Video output with embedded audio

Creates beautiful visualizations that react to audio:
- Spectrum analyzers with smooth animations
- Audio-reactive cellular automata
- Waveform visualizations
- Beat-synchronized patterns
- Multi-domain integration showcase
- Video + audio synchronization (the killer feature!)
"""

import numpy as np
import subprocess
from pathlib import Path
from morphogen.stdlib import audio, field, cellular, palette, color, image, noise, visual, Visual
from morphogen.stdlib.field import Field2D


def get_palette(colormap_name: str, resolution: int = 256):
    """Get a palette by name.

    Args:
        colormap_name: Name of the colormap (plasma, magma, viridis, inferno, etc.)
        resolution: Number of colors in palette

    Returns:
        Palette object
    """
    colormap_name = colormap_name.lower()

    # Map common colormap names to palette methods
    if colormap_name == 'plasma':
        return palette.plasma(resolution)
    elif colormap_name == 'magma':
        return palette.magma(resolution)
    elif colormap_name == 'viridis':
        return palette.viridis(resolution)
    elif colormap_name == 'inferno':
        return palette.inferno(resolution)
    elif colormap_name in ['hot', 'fire']:
        return palette.fire(resolution)
    elif colormap_name in ['cool', 'ice']:
        return palette.ice(resolution)
    elif colormap_name == 'rainbow':
        return palette.rainbow(resolution)
    elif colormap_name == 'greyscale':
        return palette.greyscale(resolution)
    else:
        # Default to viridis for unknown colormaps
        print(f"Warning: Unknown colormap '{colormap_name}', using viridis")
        return palette.viridis(resolution)


def create_video_with_audio(frames, audio_buffer, output_path, fps=30, cleanup=True):
    """Create video file with embedded audio using ffmpeg.

    Args:
        frames: List of Visual objects or RGB numpy arrays
        audio_buffer: AudioBuffer to embed
        output_path: Output path for final video (e.g., 'output.mp4')
        fps: Frames per second
        cleanup: Remove temporary files after creation

    Returns:
        Path to created video file
    """
    output_path = Path(output_path)
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Temporary files
    temp_video = temp_dir / f"{output_path.stem}_novideo.mp4"
    temp_audio = temp_dir / f"{output_path.stem}_audio.wav"

    try:
        # 1. Export video (without audio)
        print(f"  Creating video frames...")
        visual.video(frames, str(temp_video), fps=fps)

        # 2. Export audio
        print(f"  Exporting audio...")
        audio.save(audio_buffer, str(temp_audio))

        # 3. Combine with ffmpeg
        print(f"  Combining video + audio with ffmpeg...")
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', str(temp_video),  # Input video
            '-i', str(temp_audio),  # Input audio
            '-c:v', 'copy',  # Copy video codec
            '-c:a', 'aac',  # AAC audio codec
            '-strict', 'experimental',
            '-shortest',  # Match shortest stream
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠ FFmpeg warning: {result.stderr}")
            # Fallback: just save video without audio
            import shutil
            shutil.copy(temp_video, output_path)
            print(f"  ⚠ Saved video without audio (ffmpeg failed)")
        else:
            print(f"  ✓ Video with audio created: {output_path.name}")

    finally:
        # Cleanup temporary files
        if cleanup:
            if temp_video.exists():
                temp_video.unlink()
            if temp_audio.exists():
                temp_audio.unlink()
            if temp_dir.exists() and not list(temp_dir.iterdir()):
                temp_dir.rmdir()

    return output_path


def compute_fft_spectrum(audio_buffer, window_size=2048, hop_size=512):
    """Compute FFT spectrum from audio buffer.

    Args:
        audio_buffer: Audio buffer to analyze
        window_size: FFT window size
        hop_size: Hop size for STFT

    Returns:
        2D array of spectral magnitudes (time x frequency)
    """
    data = audio_buffer.data

    # Handle edge case: audio too short for FFT
    if len(data) < window_size:
        return np.zeros((1, window_size // 2))

    num_frames = (len(data) - window_size) // hop_size + 1
    num_frames = max(1, num_frames)  # Ensure at least 1 frame

    spectrum = np.zeros((num_frames, window_size // 2))

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        # Extract window
        window = data[start:end]

        # Apply Hann window
        window = window * np.hanning(window_size)

        # Compute FFT
        fft = np.fft.rfft(window)
        magnitude = np.abs(fft)[:window_size // 2]

        # Convert to dB scale
        magnitude = 20 * np.log10(magnitude + 1e-10)

        spectrum[i, :] = magnitude

    return spectrum


def create_spectrum_analyzer(audio_buffer, width=800, height=400,
                             colormap='plasma'):
    """Create a classic spectrum analyzer visualization.

    Args:
        audio_buffer: Audio to visualize
        width: Output width
        height: Output height
        colormap: Color palette

    Returns:
        RGB image of spectrum analyzer
    """
    # Compute spectrum
    spectrum = compute_fft_spectrum(audio_buffer)

    # Resize to target dimensions
    # Interpolate spectrum to match output size
    time_frames, freq_bins = spectrum.shape

    # Create visualization field
    if time_frames < width:
        # Upsample in time
        indices = np.linspace(0, time_frames - 1, width).astype(int)
        spectrum_resized = spectrum[indices, :]
    else:
        # Downsample in time
        indices = np.linspace(0, time_frames - 1, width).astype(int)
        spectrum_resized = spectrum[indices, :]

    if freq_bins < height:
        # Upsample in frequency
        spectrum_final = np.zeros((width, height))
        for i in range(width):
            spectrum_final[i, :] = np.interp(
                np.linspace(0, freq_bins - 1, height),
                np.arange(freq_bins),
                spectrum_resized[i, :]
            )
    else:
        # Downsample in frequency
        spectrum_final = spectrum_resized[:, :height]

    # Transpose to get (frequency, time) orientation
    spectrum_final = spectrum_final.T

    # Normalize
    spectrum_final = spectrum_final - spectrum_final.min()
    if spectrum_final.max() > 0:
        spectrum_final = spectrum_final / spectrum_final.max()

    # Apply colormap
    pal = get_palette(colormap, 256)
    img = palette.map(pal,spectrum_final)

    return img


def create_waveform_visualization(audio_buffer, width=800, height=200,
                                  colormap='viridis'):
    """Create waveform visualization.

    Args:
        audio_buffer: Audio to visualize
        width: Output width
        height: Output height
        colormap: Color palette

    Returns:
        RGB image of waveform
    """
    data = audio_buffer.data
    samples_per_pixel = len(data) // width

    # Create field
    waveform_field = np.zeros((height, width), dtype=np.float32)

    for i in range(width):
        start = i * samples_per_pixel
        end = start + samples_per_pixel

        if end > len(data):
            break

        # Get min/max for this window
        window = data[start:end]
        min_val = np.min(window)
        max_val = np.max(window)

        # Map to pixel coordinates
        min_y = int((min_val + 1.0) * 0.5 * height)
        max_y = int((max_val + 1.0) * 0.5 * height)

        # Clamp
        min_y = np.clip(min_y, 0, height - 1)
        max_y = np.clip(max_y, 0, height - 1)

        # Draw vertical line
        waveform_field[min_y:max_y+1, i] = 1.0

    # Apply colormap
    pal = get_palette(colormap, 256)
    img = palette.map(pal,waveform_field)

    return img


def audio_reactive_cellular_automata(audio_buffer, ca_size=200, duration=5.0):
    """Create cellular automaton that reacts to audio.

    Audio amplitude controls birth rate / density.

    Args:
        audio_buffer: Audio signal
        ca_size: CA grid size
        duration: Duration in seconds

    Returns:
        List of CA frames
    """
    sample_rate = audio_buffer.sample_rate
    data = audio_buffer.data

    # Number of CA steps
    fps = 30
    num_frames = int(duration * fps)
    samples_per_frame = len(data) // num_frames

    # Initialize CA
    field_ca, rule = cellular.game_of_life((ca_size, ca_size),
                                           density=0.3, seed=42)

    frames = []

    for frame_idx in range(num_frames):
        # Get audio segment for this frame
        start = frame_idx * samples_per_frame
        end = start + samples_per_frame

        if end > len(data):
            break

        audio_segment = data[start:end]
        amplitude = np.mean(np.abs(audio_segment))

        # Evolve CA
        field_ca = cellular.step(field_ca, rule)

        # Add random cells based on audio amplitude
        if amplitude > 0.1:
            num_cells = int(amplitude * 100)
            rng = np.random.RandomState(frame_idx)
            for _ in range(num_cells):
                x = rng.randint(0, ca_size)
                y = rng.randint(0, ca_size)
                field_ca.data[y, x] = 1

        frames.append(field_ca.copy())

    return frames


def create_beat_synchronized_patterns(audio_buffer, pattern_size=300):
    """Create patterns synchronized to audio beats.

    Uses energy in signal to trigger pattern changes.

    Args:
        audio_buffer: Audio signal
        pattern_size: Size of pattern grid

    Returns:
        RGB image of final pattern
    """
    data = audio_buffer.data

    # Compute energy envelope
    window_size = 2048
    hop_size = 512
    num_windows = (len(data) - window_size) // hop_size

    energy = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = data[start:end]
        energy[i] = np.sum(window ** 2)

    # Normalize energy
    energy = energy / (energy.max() + 1e-10)

    # Create field that accumulates based on energy
    pattern = Field2D(np.zeros((pattern_size, pattern_size), dtype=np.float32))

    for i, e in enumerate(energy):
        if e > 0.5:  # Beat detected
            # Add radial pattern
            cx, cy = pattern_size // 2, pattern_size // 2
            radius = int(e * 50)

            y, x = np.ogrid[:pattern_size, :pattern_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

            pattern.data[mask] += e * 0.5

    # Normalize
    pattern.data = np.clip(pattern.data, 0, 1)

    # Apply colormap
    pal = get_palette('hot', 256)
    img = palette.map(pal, pattern.data)

    return img


def create_audio_reactive_field(audio_buffer, width=400, height=400):
    """Create diffusion field driven by audio spectrum.

    Audio frequencies create heat sources that diffuse.

    Args:
        audio_buffer: Audio signal
        width: Field width
        height: Field height

    Returns:
        RGB image of final field state
    """
    # Compute spectrum
    spectrum = compute_fft_spectrum(audio_buffer, window_size=2048)

    # Create field
    heat_field = field.alloc((height, width), dtype=np.float32, fill_value=0.0)

    # Add heat based on spectrum
    num_freq_bins = spectrum.shape[1]

    for time_idx in range(min(spectrum.shape[0], 100)):  # Limit iterations
        # Get spectrum at this time
        spectrum_slice = spectrum[time_idx, :]

        # Normalize
        spectrum_slice = spectrum_slice - spectrum_slice.min()
        if spectrum_slice.max() > 0:
            spectrum_slice = spectrum_slice / spectrum_slice.max()

        # Add heat at positions corresponding to frequencies
        for freq_idx, magnitude in enumerate(spectrum_slice[:50]):  # Use lower freqs
            if magnitude > 0.3:
                # Position based on frequency
                x = int((freq_idx / 50) * width)
                y = height // 2

                # Add heat
                heat_field.data[y-5:y+5, x-5:x+5] += magnitude * 0.5

        # Diffuse
        heat_field = field.diffuse(heat_field, diffusion_coeff=0.1, dt=0.1)

        # Decay
        heat_field.data *= 0.98

    # Normalize
    heat_field.data = np.clip(heat_field.data, 0, 1)

    # Apply colormap
    pal = get_palette('inferno', 256)
    img = palette.map(pal, heat_field.data)

    return img


def demo_spectrum_analyzer():
    """Demo: Classic spectrum analyzer."""
    print("Creating spectrum analyzer...")

    # Generate test audio: multiple sine waves
    duration = 3.0
    sample_rate = 44100

    # Create chord (C major)
    c_note = audio.sine(freq=261.63, duration=duration, sample_rate=sample_rate)
    e_note = audio.sine(freq=329.63, duration=duration, sample_rate=sample_rate)
    g_note = audio.sine(freq=392.00, duration=duration, sample_rate=sample_rate)

    # Mix
    chord = audio.AudioBuffer(
        data=(c_note.data + e_note.data + g_note.data) / 3.0,
        sample_rate=sample_rate
    )

    # Add some noise for texture
    noise_data = np.random.randn(len(chord.data)) * 0.05
    chord.data += noise_data

    # Create visualization
    img = create_spectrum_analyzer(chord, width=800, height=400,
                                   colormap='plasma')

    # Save
    image.save(img, "output_audio_spectrum_analyzer.png")
    print("   ✓ Saved: output_audio_spectrum_analyzer.png")


def demo_waveform():
    """Demo: Waveform visualization."""
    print("Creating waveform visualization...")

    # Generate test audio: amplitude modulated sine
    duration = 2.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Carrier frequency
    carrier = np.sin(2 * np.pi * 440 * t)

    # Modulation
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)

    # AM synthesis
    am_signal = carrier * modulator

    buf = audio.AudioBuffer(data=am_signal, sample_rate=sample_rate)

    # Visualize
    img = create_waveform_visualization(buf, width=800, height=200,
                                       colormap='cool')

    image.save(img, "output_audio_waveform.png")
    print("   ✓ Saved: output_audio_waveform.png")


def demo_audio_reactive_ca():
    """Demo: Audio-reactive cellular automaton."""
    print("Creating audio-reactive cellular automaton...")

    # Generate rhythmic audio
    duration = 5.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Create rhythm: kick drum pattern
    kick_pattern = np.zeros_like(t)
    beat_interval = 0.5  # 120 BPM
    for beat_time in np.arange(0, duration, beat_interval):
        beat_idx = int(beat_time * sample_rate)
        # Exponential decay envelope
        decay_samples = int(0.2 * sample_rate)
        envelope = np.exp(-10 * np.arange(decay_samples) / decay_samples)
        # Sine wave at low frequency
        kick = np.sin(2 * np.pi * 60 * np.arange(decay_samples) / sample_rate) * envelope

        end_idx = min(beat_idx + decay_samples, len(kick_pattern))
        kick_pattern[beat_idx:end_idx] += kick[:end_idx - beat_idx]

    buf = audio.AudioBuffer(data=kick_pattern, sample_rate=sample_rate)

    # Create CA frames
    ca_frames = audio_reactive_cellular_automata(buf, ca_size=200, duration=duration)

    # Visualize a few key frames
    frame_indices = [0, len(ca_frames)//4, len(ca_frames)//2, 3*len(ca_frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx < len(ca_frames):
            ca_field = ca_frames[frame_idx]

            # Visualize
            pal = get_palette('magma', 256)
            img = palette.map(pal, ca_field.data.astype(np.float32))

            output_path = f"output_audio_reactive_ca_frame{idx:02d}.png"
            image.save(img, output_path)
            print(f"   ✓ Saved: {output_path}")


def demo_beat_patterns():
    """Demo: Beat-synchronized patterns."""
    print("Creating beat-synchronized patterns...")

    # Generate beat pattern
    duration = 4.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Create drum pattern
    pattern = np.zeros_like(t)

    # Kick on 1 and 3
    for beat in [0.0, 1.0, 2.0, 3.0]:
        idx = int(beat * sample_rate)
        decay = np.exp(-20 * np.arange(2000) / sample_rate)
        kick = np.sin(2 * np.pi * 80 * np.arange(2000) / sample_rate) * decay
        pattern[idx:idx+2000] += kick

    # Snare on 2 and 4
    for beat in [0.5, 1.5, 2.5, 3.5]:
        idx = int(beat * sample_rate)
        snare = np.random.randn(4000) * np.exp(-15 * np.arange(4000) / sample_rate)
        pattern[idx:idx+4000] += snare * 0.5

    buf = audio.AudioBuffer(data=pattern, sample_rate=sample_rate)

    # Create visualization
    img = create_beat_synchronized_patterns(buf, pattern_size=400)

    image.save(img, "output_audio_beat_patterns.png")
    print("   ✓ Saved: output_audio_beat_patterns.png")


def demo_audio_field_diffusion():
    """Demo: Audio-driven field diffusion."""
    print("Creating audio-driven field diffusion...")

    # Generate sweeping tone
    duration = 3.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Frequency sweep
    freq_start = 100
    freq_end = 2000
    freq = freq_start + (freq_end - freq_start) * (t / duration)

    # Instantaneous phase
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate

    sweep = np.sin(phase)

    buf = audio.AudioBuffer(data=sweep, sample_rate=sample_rate)

    # Create field visualization
    img = create_audio_reactive_field(buf, width=400, height=400)

    image.save(img, "output_audio_field_diffusion.png")
    print("   ✓ Saved: output_audio_field_diffusion.png")


def create_animated_spectrum_video(audio_buffer, output_path="output_spectrum_video.mp4",
                                    width=800, height=400, fps=30):
    """Create animated spectrum analyzer video with audio.

    Args:
        audio_buffer: Audio to visualize
        output_path: Output video path
        width: Video width
        height: Video height
        fps: Frames per second

    Returns:
        Path to created video
    """
    print(f"Creating animated spectrum video ({fps} fps)...")

    # Calculate number of frames needed
    duration = audio_buffer.duration
    n_frames = int(duration * fps)

    # Window size and hop for FFT
    window_size = 2048
    hop_size = audio_buffer.sample_rate // fps  # Hop to match frame rate

    frames = []

    for frame_idx in range(n_frames):
        # Get audio segment for this frame
        start_sample = frame_idx * hop_size
        end_sample = start_sample + window_size * 4  # Use 4 windows worth of context

        if end_sample > len(audio_buffer.data):
            end_sample = len(audio_buffer.data)

        # Extract segment
        segment = audio.AudioBuffer(
            data=audio_buffer.data[start_sample:end_sample],
            sample_rate=audio_buffer.sample_rate
        )

        # Compute spectrum for this segment
        spectrum = compute_fft_spectrum(segment, window_size=window_size, hop_size=hop_size//2)

        if spectrum.shape[0] == 0:
            # Not enough data, use previous frame or black
            if frames:
                frames.append(frames[-1])
            continue

        # Resize spectrum to fit display
        time_frames, freq_bins = spectrum.shape
        freq_bins_display = min(freq_bins, height)

        # Take recent time frames (scrolling effect)
        time_frames_display = min(time_frames, width)
        spectrum_slice = spectrum[-time_frames_display:, :freq_bins_display]

        # Pad if needed
        if spectrum_slice.shape[0] < width:
            padding = np.zeros((width - spectrum_slice.shape[0], freq_bins_display))
            spectrum_slice = np.vstack([padding, spectrum_slice])

        if spectrum_slice.shape[1] < height:
            spectrum_resized = np.zeros((width, height))
            for i in range(width):
                spectrum_resized[i, :] = np.interp(
                    np.linspace(0, spectrum_slice.shape[1] - 1, height),
                    np.arange(spectrum_slice.shape[1]),
                    spectrum_slice[i, :]
                )
        else:
            spectrum_resized = spectrum_slice[:, :height]

        # Transpose for proper orientation
        spectrum_resized = spectrum_resized.T

        # Normalize
        spectrum_resized = spectrum_resized - spectrum_resized.min()
        if spectrum_resized.max() > 0:
            spectrum_resized = spectrum_resized / spectrum_resized.max()

        # Apply colormap
        pal = get_palette('plasma', 256)
        img = palette.map(pal, spectrum_resized)

        # Convert to Visual
        vis = Visual(img)
        frames.append(vis)

        if frame_idx % (fps * 2) == 0:
            print(f"  Frame {frame_idx}/{n_frames} ({frame_idx/fps:.1f}s)")

    # Create video with audio
    print(f"  Generated {len(frames)} frames")
    create_video_with_audio(frames, audio_buffer, output_path, fps=fps)

    return output_path


def create_waveform_animation(audio_buffer, output_path="output_waveform_video.mp4",
                               width=800, height=400, fps=30):
    """Create animated waveform video with audio.

    Shows a sliding window of the waveform as the audio plays.

    Args:
        audio_buffer: Audio to visualize
        output_path: Output video path
        width: Video width
        height: Video height
        fps: Frames per second
    """
    print(f"Creating animated waveform video ({fps} fps)...")

    duration = audio_buffer.duration
    n_frames = int(duration * fps)
    samples_per_frame = len(audio_buffer.data) // n_frames
    window_samples = samples_per_frame * 30  # Show 30 frames worth of audio

    frames = []

    for frame_idx in range(n_frames):
        # Center position in audio
        center_sample = frame_idx * samples_per_frame

        # Window around center
        start = max(0, center_sample - window_samples // 2)
        end = min(len(audio_buffer.data), center_sample + window_samples // 2)

        # Extract window
        window_data = audio_buffer.data[start:end]

        # Create waveform field
        waveform_field = np.zeros((height, width), dtype=np.float32)

        samples_per_pixel = len(window_data) // width
        if samples_per_pixel == 0:
            samples_per_pixel = 1

        for i in range(min(width, len(window_data) // samples_per_pixel)):
            win_start = i * samples_per_pixel
            win_end = win_start + samples_per_pixel

            if win_end > len(window_data):
                break

            pixel_window = window_data[win_start:win_end]
            min_val = np.min(pixel_window)
            max_val = np.max(pixel_window)

            # Map to pixel coordinates
            min_y = int((min_val + 1.0) * 0.5 * height)
            max_y = int((max_val + 1.0) * 0.5 * height)

            min_y = np.clip(min_y, 0, height - 1)
            max_y = np.clip(max_y, 0, height - 1)

            # Draw vertical line
            waveform_field[min_y:max_y+1, i] = 1.0

        # Add current position marker
        marker_x = width // 2
        waveform_field[:, marker_x] = 0.5

        # Apply colormap
        pal = get_palette('cool', 256)
        img = palette.map(pal, waveform_field)

        # Convert to Visual
        vis = Visual(img)
        frames.append(vis)

        if frame_idx % (fps * 2) == 0:
            print(f"  Frame {frame_idx}/{n_frames}")

    print(f"  Generated {len(frames)} frames")
    create_video_with_audio(frames, audio_buffer, output_path, fps=fps)

    return output_path


def demo_video_exports():
    """Demo: Create video exports with embedded audio."""
    print("\n" + "=" * 60)
    print("VIDEO EXPORT DEMOS (WITH AUDIO)")
    print("=" * 60)
    print()

    # Generate test audio for videos
    print("Generating test audio...")
    duration = 5.0
    sample_rate = 44100

    # Create a musical sequence
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Arpeggio pattern (C major scale)
    notes = [261.63, 329.63, 392.00, 523.25]  # C, E, G, C
    audio_data = np.zeros_like(t)

    for i, freq in enumerate(notes):
        start = i * duration / 4
        end = (i + 1) * duration / 4
        mask = (t >= start) & (t < end)

        # Sine wave with envelope
        note_t = t[mask] - start
        envelope = np.exp(-3 * note_t)
        audio_data[mask] = np.sin(2 * np.pi * freq * note_t) * envelope * 0.5

    test_audio = audio.AudioBuffer(data=audio_data, sample_rate=sample_rate)

    # Video 1: Animated spectrum
    print("\nVideo 1: Animated Spectrum Analyzer")
    print("-" * 60)
    create_animated_spectrum_video(
        test_audio,
        output_path="output_spectrum_animated.mp4",
        width=800,
        height=400,
        fps=30
    )
    print("   ✓ Saved: output_spectrum_animated.mp4")

    # Video 2: Animated waveform
    print("\nVideo 2: Animated Waveform")
    print("-" * 60)
    create_waveform_animation(
        test_audio,
        output_path="output_waveform_animated.mp4",
        width=800,
        height=400,
        fps=30
    )
    print("   ✓ Saved: output_waveform_animated.mp4")

    # Also create GIFs (without audio, for web)
    print("\nCreating GIFs for web...")
    print("-" * 60)

    # Generate shorter clips for GIFs
    short_duration = 2.0
    short_audio = audio.AudioBuffer(
        data=audio_data[:int(short_duration * sample_rate)],
        sample_rate=sample_rate
    )

    # Spectrum GIF
    print("  Creating spectrum GIF...")
    spectrum_frames = []
    n_frames = int(short_duration * 15)  # 15 fps for GIF
    hop_size = len(short_audio.data) // n_frames

    for i in range(n_frames):
        start = i * hop_size
        end = min(start + 2048 * 4, len(short_audio.data))
        segment = audio.AudioBuffer(data=short_audio.data[start:end], sample_rate=sample_rate)

        spectrum = compute_fft_spectrum(segment, window_size=2048, hop_size=512)
        if spectrum.shape[0] > 0:
            # Resize to 400x200 for smaller GIF
            spectrum_slice = spectrum[-200:, :100] if spectrum.shape[0] >= 200 else spectrum[:, :100]

            # Pad if needed
            if spectrum_slice.shape[0] < 200:
                padding = np.zeros((200 - spectrum_slice.shape[0], spectrum_slice.shape[1]))
                spectrum_slice = np.vstack([padding, spectrum_slice])

            spectrum_resized = spectrum_slice.T

            # Normalize
            spectrum_resized = spectrum_resized - spectrum_resized.min()
            if spectrum_resized.max() > 0:
                spectrum_resized = spectrum_resized / spectrum_resized.max()

            # Apply colormap
            pal = get_palette('plasma', 256)
            img = palette.map(pal, spectrum_resized)
            spectrum_frames.append(Visual(img))

    if spectrum_frames:
        visual.video(spectrum_frames, "output_spectrum_loop.gif", fps=15)
        print("   ✓ Saved: output_spectrum_loop.gif")

    print()


def generate_audio_visualizer(
    output_generator,
    seed: int = 42,
    duration_seconds: float = None
):
    """
    Generate audio visualizer showcase output for OutputGenerator framework.

    Creates a composite visualization showing multiple audio-reactive modes:
    - Spectrum analyzer
    - Audio-reactive cellular automata
    - Waveform display

    Args:
        output_generator: OutputGenerator instance with preset configuration
        seed: Random seed for deterministic output
        duration_seconds: Duration in seconds (uses preset if None)

    Returns:
        Tuple of (frames, audio_data, metadata)
        - frames: List of Visual objects
        - audio_data: Audio samples (numpy array)
        - metadata: Dict with generation parameters
    """
    print("Generating audio visualizer showcase...")

    if duration_seconds is None:
        duration_seconds = min(10, output_generator.preset['max_duration'])

    # Get configuration from preset
    width, height = output_generator.preset['resolution']
    fps = output_generator.preset['fps']
    sample_rate = output_generator.preset['audio_sr']

    # Set random seed for deterministic audio generation
    np.random.seed(seed)

    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration_seconds}s @ {fps} fps")
    print(f"  Audio: {sample_rate}Hz")
    print()

    # Generate test audio: Musical arpeggio with rhythm
    print("  Generating audio...")
    t = np.arange(int(duration_seconds * sample_rate)) / sample_rate

    # Create musical sequence with rhythm
    audio_data = np.zeros_like(t)

    # Arpeggio pattern (C major chord with rhythm)
    notes = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63]  # C, E, G, C, G, E
    note_duration = duration_seconds / len(notes)

    for i, freq in enumerate(notes):
        start_time = i * note_duration
        end_time = (i + 1) * note_duration
        mask = (t >= start_time) & (t < end_time)

        # Sine wave with envelope
        note_t = t[mask] - start_time
        envelope = np.exp(-3 * note_t)
        audio_data[mask] = np.sin(2 * np.pi * freq * note_t) * envelope * 0.5

    # Add subtle background rhythm (kick pattern)
    for beat_time in np.arange(0, duration_seconds, 0.5):
        beat_idx = int(beat_time * sample_rate)
        decay_samples = int(0.15 * sample_rate)
        if beat_idx + decay_samples < len(audio_data):
            envelope = np.exp(-15 * np.arange(decay_samples) / sample_rate)
            kick = np.sin(2 * np.pi * 80 * np.arange(decay_samples) / sample_rate) * envelope
            audio_data[beat_idx:beat_idx+decay_samples] += kick * 0.3

    audio_buffer = audio.AudioBuffer(data=audio_data, sample_rate=sample_rate)

    print("  ✓ Audio generated")
    print()

    # Generate frames with composite visualization
    print(f"  Generating {int(duration_seconds * fps)} frames...")

    n_frames = int(duration_seconds * fps)
    frames = []

    # Calculate layout for composite view (3 panels stacked vertically)
    panel_height = height // 3
    spectrum_height = panel_height
    ca_height = panel_height
    waveform_height = height - (spectrum_height + ca_height)  # Remaining space

    # Pre-compute cellular automata frames
    print("  Computing audio-reactive cellular automata...")
    ca_frames = audio_reactive_cellular_automata(audio_buffer, ca_size=min(width, 512), duration=duration_seconds)

    # Generate composite frames
    for frame_idx in range(n_frames):
        if frame_idx % (fps * 2) == 0:
            print(f"    Frame {frame_idx}/{n_frames} ({frame_idx/fps:.1f}s)")

        # Current time
        current_time = frame_idx / fps
        start_sample = int(current_time * sample_rate)

        # Panel 1: Spectrum analyzer
        # Use a sliding window of audio
        window_samples = int(sample_rate * 0.5)  # 0.5 second window
        end_sample = min(start_sample + window_samples, len(audio_data))
        audio_segment = audio.AudioBuffer(
            data=audio_data[start_sample:end_sample],
            sample_rate=sample_rate
        )

        if len(audio_segment.data) > 0:
            spectrum_img = create_spectrum_analyzer(
                audio_segment,
                width=width,
                height=spectrum_height,
                colormap='plasma'
            )
        else:
            spectrum_img = np.zeros((spectrum_height, width, 3), dtype=np.uint8)

        # Panel 2: Audio-reactive CA
        ca_frame_idx = min(frame_idx, len(ca_frames) - 1)
        ca_field = ca_frames[ca_frame_idx]

        # Resize CA to panel size
        from scipy.ndimage import zoom
        scale_y = ca_height / ca_field.data.shape[0]
        scale_x = width / ca_field.data.shape[1]
        ca_resized = zoom(ca_field.data.astype(np.float32), (scale_y, scale_x), order=1)

        # Colorize
        pal = get_palette('magma', 256)
        ca_img = palette.map(pal, ca_resized)

        # Panel 3: Waveform
        # Show waveform around current time
        window_duration = 1.0  # Show 1 second of waveform
        waveform_start = max(0, start_sample - int(window_duration * sample_rate // 2))
        waveform_end = min(len(audio_data), waveform_start + int(window_duration * sample_rate))

        waveform_segment = audio.AudioBuffer(
            data=audio_data[waveform_start:waveform_end],
            sample_rate=sample_rate
        )

        waveform_img = create_waveform_visualization(
            waveform_segment,
            width=width,
            height=waveform_height,
            colormap='cool'
        )

        # Composite: Stack three panels vertically
        composite = np.vstack([spectrum_img, ca_img, waveform_img])

        # Convert to Visual
        vis = Visual(composite)
        frames.append(vis)

    print(f"  ✓ Generated {len(frames)} frames")
    print()

    # Prepare metadata
    metadata = {
        'example': 'audio_visualizer',
        'description': 'Cross-domain audio visualization with spectrum, cellular automata, and waveform',
        'domains': ['Audio', 'Field', 'Cellular', 'Palette', 'Visual'],
        'cross_domain_operations': [
            'Audio → Spectrum (FFT)',
            'Audio → Cellular Automata (amplitude → cell birth)',
            'Audio → Waveform (temporal visualization)'
        ],
        'frames': len(frames),
        'fps': fps,
        'duration_seconds': duration_seconds,
        'resolution': [width, height],
        'audio_sample_rate': sample_rate,
        'audio_duration_seconds': len(audio_data) / sample_rate,
        'seed': seed,
        'visualization_modes': [
            'Spectrum analyzer (plasma colormap)',
            'Audio-reactive cellular automata (magma colormap)',
            'Waveform display (cool colormap)'
        ]
    }

    return frames, audio_data, metadata


def main():
    """Run all audio visualizer demonstrations."""
    print("=" * 60)
    print("AUDIO VISUALIZER - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Domains: Audio + Field + Cellular + Palette + Image + Video")
    print()

    # Demo 1: Spectrum analyzer
    print("Demo 1: Spectrum Analyzer")
    print("-" * 60)
    demo_spectrum_analyzer()
    print()

    # Demo 2: Waveform
    print("Demo 2: Waveform Visualization")
    print("-" * 60)
    demo_waveform()
    print()

    # Demo 3: Audio-reactive CA
    print("Demo 3: Audio-Reactive Cellular Automaton")
    print("-" * 60)
    demo_audio_reactive_ca()
    print()

    # Demo 4: Beat patterns
    print("Demo 4: Beat-Synchronized Patterns")
    print("-" * 60)
    demo_beat_patterns()
    print()

    # Demo 5: Field diffusion
    print("Demo 5: Audio-Driven Field Diffusion")
    print("-" * 60)
    demo_audio_field_diffusion()
    print()

    # Demo 6: Video exports
    print("Demo 6: Video Exports with Audio")
    print("-" * 60)
    demo_video_exports()
    print()

    print("=" * 60)
    print("ALL AUDIO VISUALIZER DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("This showcase demonstrates:")
    print("  • Audio synthesis and analysis (FFT, spectrum)")
    print("  • Field operations (diffusion, heat propagation)")
    print("  • Cellular automata (audio-reactive patterns)")
    print("  • Color mapping and palettes")
    print("  • Video output with embedded audio ⭐")
    print("  • Cross-domain integration")
    print()
    print("Key insight: Temporal domains (audio) can drive")
    print("spatial domains (field, cellular) to create")
    print("stunning audio-reactive visualizations!")
    print()
    print("🎬 Video files created with synchronized audio!")
    print("   - output_spectrum_animated.mp4")
    print("   - output_waveform_animated.mp4")
    print("   - output_spectrum_loop.gif (web-friendly)")


if __name__ == "__main__":
    main()
