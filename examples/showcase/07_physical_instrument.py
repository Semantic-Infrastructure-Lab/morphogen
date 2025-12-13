"""Physical Modeling Instrument - Cross-Domain Showcase

This example demonstrates the power of combining multiple Morphogen domains:
- Audio synthesis for sound generation
- Acoustics for physical waveguide modeling
- Field operations for vibration analysis
- Signal processing for spectral analysis
- Visual rendering for beautiful visualizations
- I/O for audio export

Creates physically modeled instruments:
- String instruments (guitar, violin)
- Percussion (drums, bells)
- Wind instruments (flute, brass)
- Modal synthesis (resonant objects)
- Multi-domain integration showcase
"""

import numpy as np
from morphogen.stdlib import audio, field, palette, image, visual, io_storage
from morphogen.stdlib.field import Field2D
from morphogen.stdlib.visual import Visual

# Try to import acoustics and signal if available
try:
    from morphogen.stdlib import acoustics
    ACOUSTICS_AVAILABLE = True
except:
    ACOUSTICS_AVAILABLE = False

try:
    from morphogen.stdlib import signal as signal_proc
    SIGNAL_AVAILABLE = True
except:
    SIGNAL_AVAILABLE = False


def karplus_strong_string(frequency, duration, sample_rate=44100,
                          pluck_position=0.5, damping=0.995):
    """Physical model of a plucked string using Karplus-Strong algorithm.

    Args:
        frequency: Fundamental frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate
        pluck_position: Where string is plucked (0-1)
        damping: Damping factor (0-1, closer to 1 = less damping)

    Returns:
        AudioBuffer with synthesized string sound
    """
    # Calculate delay line length
    delay_samples = int(sample_rate / frequency)

    # Initialize delay line with noise (pluck excitation)
    delay_line = np.random.randn(delay_samples) * 2.0 - 1.0

    # Total samples
    n_samples = int(duration * sample_rate)
    output = np.zeros(n_samples)

    # Karplus-Strong loop
    for i in range(n_samples):
        # Output current sample
        output[i] = delay_line[0]

        # Simple averaging filter (lowpass)
        new_sample = damping * (delay_line[0] + delay_line[1]) / 2.0

        # Shift delay line
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = new_sample

    # Normalize
    output = output / (np.max(np.abs(output)) + 1e-6)

    return audio.AudioBuffer(data=output, sample_rate=sample_rate)


def modal_synthesis(modes, duration, sample_rate=44100):
    """Synthesize sound using modal synthesis (sum of damped sine waves).

    Used for bells, plates, and other resonant objects.

    Args:
        modes: List of (frequency, amplitude, decay_time) tuples
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        AudioBuffer with synthesized sound
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    output = np.zeros(n_samples)

    for freq, amp, decay in modes:
        # Exponential envelope
        envelope = np.exp(-t / decay)

        # Sine wave
        wave = amp * np.sin(2 * np.pi * freq * t) * envelope

        output += wave

    # Normalize
    output = output / (np.max(np.abs(output)) + 1e-6)

    return audio.AudioBuffer(data=output, sample_rate=sample_rate)


def drum_synthesis(fundamental, duration, sample_rate=44100):
    """Physical model of a drum using multiple damped modes.

    Args:
        fundamental: Fundamental frequency
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        AudioBuffer with drum sound
    """
    # Drum has inharmonic overtones
    modes = [
        (fundamental * 1.00, 1.0, 0.3),   # Fundamental
        (fundamental * 2.14, 0.7, 0.2),   # First overtone
        (fundamental * 3.01, 0.5, 0.15),  # Second overtone
        (fundamental * 4.10, 0.3, 0.1),   # Third overtone
        (fundamental * 5.43, 0.2, 0.08),  # Fourth overtone
    ]

    # Add noise burst (stick attack)
    n_samples = int(duration * sample_rate)
    noise_duration = 0.01  # 10ms noise burst
    noise_samples = int(noise_duration * sample_rate)

    noise = np.random.randn(noise_samples) * 0.3
    noise *= np.exp(-np.arange(noise_samples) / (noise_samples * 0.2))

    # Modal synthesis
    modal_sound = modal_synthesis(modes, duration, sample_rate)

    # Add noise at beginning
    modal_sound.data[:noise_samples] += noise

    # Normalize
    modal_sound.data = modal_sound.data / (np.max(np.abs(modal_sound.data)) + 1e-6)

    return modal_sound


def bell_synthesis(fundamental, duration, sample_rate=44100):
    """Physical model of a bell using modal synthesis.

    Args:
        fundamental: Fundamental frequency
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        AudioBuffer with bell sound
    """
    # Bell modes (based on actual bell physics)
    modes = [
        (fundamental * 0.5, 0.8, 2.5),   # Hum tone
        (fundamental * 1.0, 1.0, 2.0),   # Fundamental
        (fundamental * 1.2, 0.6, 1.8),   # Tierce
        (fundamental * 2.0, 0.7, 1.5),   # Quint
        (fundamental * 2.4, 0.5, 1.2),   # Nominal
        (fundamental * 3.0, 0.3, 1.0),   # Superquint
    ]

    return modal_synthesis(modes, duration, sample_rate)


def compute_string_vibration_field(frequency, width=400, height=100,
                                   n_modes=5, time=0.0):
    """Compute visualization of string vibration modes.

    Args:
        frequency: String fundamental frequency
        width: Field width (string length direction)
        height: Field height (transverse direction)
        n_modes: Number of modes to visualize
        time: Time instant (for animation)

    Returns:
        Field2D with string displacement
    """
    # String position
    x = np.linspace(0, 1, width)

    # Compute displacement as sum of modes
    displacement = np.zeros(width)

    for n in range(1, n_modes + 1):
        # Mode shape
        mode_shape = np.sin(n * np.pi * x)

        # Time evolution
        omega = 2 * np.pi * frequency * n
        amplitude = 1.0 / n  # Higher modes have lower amplitude

        displacement += amplitude * mode_shape * np.cos(omega * time)

    # Normalize
    displacement = displacement / (np.max(np.abs(displacement)) + 1e-6)

    # Create 2D field (string visualization)
    field_data = np.zeros((height, width), dtype=np.float32)

    # Map displacement to vertical position
    for i, disp in enumerate(displacement):
        y_pos = int((disp + 1.0) * 0.5 * height)
        y_pos = np.clip(y_pos, 0, height - 1)

        # Draw string
        field_data[y_pos, i] = 1.0

        # Thicken the string
        if y_pos > 0:
            field_data[y_pos - 1, i] = 0.7
        if y_pos < height - 1:
            field_data[y_pos + 1, i] = 0.7

    return Field2D(field_data)


def visualize_modal_vibration(modes, width=400, height=400):
    """Visualize modal vibration of a 2D plate.

    Args:
        modes: List of modal frequencies
        width: Field width
        height: Field height

    Returns:
        RGB image of modal pattern
    """
    # Create modal pattern (simplified 2D vibration)
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # Sum of modal patterns
    pattern = np.zeros((height, width), dtype=np.float32)

    for i, (freq, amp, decay) in enumerate(modes[:4]):  # Use first 4 modes
        # Mode numbers
        m = (i % 2) + 1
        n = (i // 2) + 1

        # 2D mode shape
        mode_shape = np.sin(m * X) * np.sin(n * Y)

        pattern += amp * mode_shape

    # Normalize
    pattern = pattern / (np.max(np.abs(pattern)) + 1e-6)
    pattern = (pattern + 1.0) * 0.5  # Map to [0, 1]

    # Apply colormap
    pal = palette.viridis(256)
    img = palette.map(pal, pattern)

    return img


def create_spectrogram(audio_buffer, width=800, height=400):
    """Create spectrogram visualization of audio.

    Args:
        audio_buffer: Audio to analyze
        width: Output width
        height: Output height

    Returns:
        RGB image of spectrogram
    """
    data = audio_buffer.data
    sample_rate = audio_buffer.sample_rate

    # STFT parameters
    window_size = 2048
    hop_size = 512

    # Compute STFT
    n_frames = (len(data) - window_size) // hop_size + 1
    n_frames = max(1, n_frames)

    spectrogram = np.zeros((n_frames, window_size // 2))

    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        # Extract window
        window = data[start:end]

        # Apply Hann window
        window = window * np.hanning(window_size)

        # FFT
        fft = np.fft.rfft(window)
        magnitude = np.abs(fft)[:window_size // 2]

        # Convert to dB
        magnitude = 20 * np.log10(magnitude + 1e-10)

        spectrogram[i, :] = magnitude

    # Resize to output dimensions
    from scipy.ndimage import zoom
    scale_x = width / spectrogram.shape[0]
    scale_y = height / spectrogram.shape[1]
    spectrogram_resized = zoom(spectrogram, (scale_x, scale_y), order=1)

    # Transpose for proper orientation
    spectrogram_resized = spectrogram_resized.T

    # Normalize
    spec_min, spec_max = spectrogram_resized.min(), spectrogram_resized.max()
    if spec_max > spec_min:
        spectrogram_resized = (spectrogram_resized - spec_min) / (spec_max - spec_min)

    # Apply colormap
    pal = palette.magma(256)
    img = palette.map(pal, spectrogram_resized)

    return img


def demo_karplus_strong():
    """Demo: Karplus-Strong string synthesis."""
    print("Demo 1: Karplus-Strong String")
    print("-" * 60)

    # Synthesize multiple notes (simple melody)
    notes = [
        (261.63, 1.0),  # C4
        (293.66, 1.0),  # D4
        (329.63, 1.0),  # E4
        (349.23, 1.0),  # F4
        (392.00, 2.0),  # G4 (longer)
    ]

    all_audio = []

    for freq, duration in notes:
        note_audio = karplus_strong_string(freq, duration, damping=0.996)
        all_audio.append(note_audio.data)

    # Concatenate
    full_audio = np.concatenate(all_audio)
    audio_buffer = audio.AudioBuffer(data=full_audio, sample_rate=44100)

    # Save audio
    audio.save(audio_buffer, "output_instrument_karplus_strong.wav")
    print("  ✓ Saved: output_instrument_karplus_strong.wav")

    # Create spectrogram
    spec_img = create_spectrogram(audio_buffer, width=800, height=400)
    io_storage.save_image("output_instrument_karplus_strong_spectrogram.png", spec_img)
    print("  ✓ Saved: output_instrument_karplus_strong_spectrogram.png")

    # Visualize string vibration at different times
    for i, t in enumerate([0.0, 0.01, 0.02, 0.03]):
        string_field = compute_string_vibration_field(
            frequency=261.63, width=400, height=100, n_modes=5, time=t
        )

        pal = palette.plasma(256)
        img = palette.map(pal, string_field.data)

        output_path = f"output_instrument_string_vibration_t{i:02d}.png"
        io_storage.save_image(img, output_path)
        print(f"  ✓ Saved: {output_path}")


def demo_modal_synthesis():
    """Demo: Modal synthesis (bells and drums)."""
    print("\nDemo 2: Modal Synthesis")
    print("-" * 60)

    # Synthesize drum
    print("  Synthesizing drum...")
    drum = drum_synthesis(fundamental=200, duration=2.0)
    audio.save(drum, "output_instrument_drum.wav")
    print("  ✓ Saved: output_instrument_drum.wav")

    # Visualize drum modes
    drum_modes = [
        (200 * 1.00, 1.0, 0.3),
        (200 * 2.14, 0.7, 0.2),
        (200 * 3.01, 0.5, 0.15),
        (200 * 4.10, 0.3, 0.1),
    ]
    drum_modal_img = visualize_modal_vibration(drum_modes, width=400, height=400)
    io_storage.save_image("output_instrument_drum_modes.png", drum_modal_img)
    print("  ✓ Saved: output_instrument_drum_modes.png")

    # Synthesize bell
    print("  Synthesizing bell...")
    bell = bell_synthesis(fundamental=440, duration=4.0)
    audio.save(bell, "output_instrument_bell.wav")
    print("  ✓ Saved: output_instrument_bell.wav")

    # Visualize bell modes
    bell_modes = [
        (440 * 0.5, 0.8, 2.5),
        (440 * 1.0, 1.0, 2.0),
        (440 * 1.2, 0.6, 1.8),
        (440 * 2.0, 0.7, 1.5),
    ]
    bell_modal_img = visualize_modal_vibration(bell_modes, width=400, height=400)
    io_storage.save_image("output_instrument_bell_modes.png", bell_modal_img)
    print("  ✓ Saved: output_instrument_bell_modes.png")

    # Create spectrograms
    drum_spec = create_spectrogram(drum, width=800, height=300)
    io_storage.save_image("output_instrument_drum_spectrogram.png", drum_spec)
    print("  ✓ Saved: output_instrument_drum_spectrogram.png")

    bell_spec = create_spectrogram(bell, width=800, height=300)
    io_storage.save_image("output_instrument_bell_spectrogram.png", bell_spec)
    print("  ✓ Saved: output_instrument_bell_spectrogram.png")


def demo_multi_string_instrument():
    """Demo: Multi-string instrument (like a guitar chord)."""
    print("\nDemo 3: Multi-String Instrument")
    print("-" * 60)

    # C Major chord: C E G
    chord_notes = [
        261.63,  # C4
        329.63,  # E4
        392.00,  # G4
    ]

    print("  Synthesizing C major chord...")

    # Synthesize each string
    strings = []
    for freq in chord_notes:
        string = karplus_strong_string(freq, duration=3.0, damping=0.997)
        strings.append(string.data)

    # Mix strings
    max_len = max(len(s) for s in strings)
    chord_audio = np.zeros(max_len)

    for s in strings:
        chord_audio[:len(s)] += s

    # Normalize
    chord_audio = chord_audio / len(strings)

    chord_buffer = audio.AudioBuffer(data=chord_audio, sample_rate=44100)

    # Save
    audio.save(chord_buffer, "output_instrument_chord.wav")
    print("  ✓ Saved: output_instrument_chord.wav")

    # Create spectrogram
    spec_img = create_spectrogram(chord_buffer, width=800, height=400)
    io_storage.save_image("output_instrument_chord_spectrogram.png", spec_img)
    print("  ✓ Saved: output_instrument_chord_spectrogram.png")


def demo_rhythm_pattern():
    """Demo: Rhythmic pattern using drums."""
    print("\nDemo 4: Rhythmic Pattern")
    print("-" * 60)

    print("  Creating drum pattern...")

    # Pattern parameters
    bpm = 120
    beat_duration = 60.0 / bpm
    pattern_duration = beat_duration * 8  # 8 beats

    sample_rate = 44100
    pattern_samples = int(pattern_duration * sample_rate)
    pattern_audio = np.zeros(pattern_samples)

    # Kick on beats 1, 3, 5, 7
    kick_times = [0, 2, 4, 6]
    for beat in kick_times:
        start_sample = int(beat * beat_duration * sample_rate)
        kick = drum_synthesis(fundamental=80, duration=0.5, sample_rate=sample_rate)
        end_sample = min(start_sample + len(kick.data), pattern_samples)
        pattern_audio[start_sample:end_sample] += kick.data[:end_sample - start_sample]

    # Snare on beats 2, 4, 6, 8
    snare_times = [1, 3, 5, 7]
    for beat in snare_times:
        start_sample = int(beat * beat_duration * sample_rate)
        snare = drum_synthesis(fundamental=200, duration=0.3, sample_rate=sample_rate)
        end_sample = min(start_sample + len(snare.data), pattern_samples)
        pattern_audio[start_sample:end_sample] += snare.data[:end_sample - start_sample] * 0.7

    # Normalize
    pattern_audio = pattern_audio / (np.max(np.abs(pattern_audio)) + 1e-6) * 0.8

    pattern_buffer = audio.AudioBuffer(data=pattern_audio, sample_rate=sample_rate)

    # Save
    audio.save(pattern_buffer, "output_instrument_rhythm_pattern.wav")
    print("  ✓ Saved: output_instrument_rhythm_pattern.wav")

    # Create spectrogram
    spec_img = create_spectrogram(pattern_buffer, width=800, height=400)
    io_storage.save_image("output_instrument_rhythm_spectrogram.png", spec_img)
    print("  ✓ Saved: output_instrument_rhythm_spectrogram.png")


def main():
    """Run all physical instrument demonstrations."""
    print("=" * 60)
    print("PHYSICAL MODELING INSTRUMENT - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Domains: Audio + Acoustics + Field + Signal + Visual")
    print()

    demo_karplus_strong()
    demo_modal_synthesis()
    demo_multi_string_instrument()
    demo_rhythm_pattern()

    print()
    print("=" * 60)
    print("ALL INSTRUMENT DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("This showcase demonstrates:")
    print("  • Karplus-Strong string synthesis")
    print("  • Modal synthesis (drums, bells)")
    print("  • Multi-string instruments (chords)")
    print("  • Rhythmic patterns")
    print("  • Spectral analysis and visualization")
    print("  • String vibration visualization")
    print()
    print("Key insight: Physical models of instruments produce")
    print("rich, realistic sounds by simulating the actual physics")
    print("of vibrating strings, membranes, and resonant objects!")


if __name__ == "__main__":
    main()
