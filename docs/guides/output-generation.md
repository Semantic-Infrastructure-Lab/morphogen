# Morphogen Output Generation Guide 🎨

**How to create stunning, compelling outputs that showcase Morphogen's unique value**

This guide covers the philosophy, techniques, and best practices for generating high-quality outputs from Morphogen examples that demonstrate the platform's power and uniqueness.

---

## Table of Contents

1. [Philosophy: What Makes Morphogen Outputs Special](#philosophy-what-makes-morphogen-outputs-special)
2. [Output Types & Formats](#output-types--formats)
3. [The Output Generation Pipeline](#the-output-generation-pipeline)
4. [Cross-Domain Output Recipes](#cross-domain-output-recipes)
5. [Visual Composition Techniques](#visual-composition-techniques)
6. [Audio-Visual Synchronization](#audio-visual-synchronization)
7. [Quality Checklist](#quality-checklist)
8. [Social Media Optimization](#social-media-optimization)
9. [Examples & Code Patterns](#examples--code-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Philosophy: What Makes Morphogen Outputs Special

### The Morphogen Difference

Unlike other creative coding platforms (Processing, p5.js, TouchDesigner), Morphogen's outputs are unique because:

**1. True Multi-Domain Integration** 🔗
```
Physics simulation → Audio synthesis
Field operations → Particle systems
Circuit design → Acoustics
All in ONE deterministic system
```

**2. Deterministic by Design** 🎯
```python
# Same seed = identical output, always
np.random.seed(42)
# Bitwise-identical across platforms, runs, and GPUs
```

**3. Professional Quality** 📊
- Gamma-corrected sRGB output
- Sample-accurate audio (44.1kHz)
- Publication-ready visualizations
- Multiple synchronized time rates

**4. Impossible Elsewhere** ⚡
You literally cannot do "rigid body physics that generates spatial audio" or "fluid simulation controlling synthesizer parameters" in other platforms without extensive glue code.

### Output Generation Goals

When creating Morphogen outputs, aim for:

1. **Visual Impact** - Immediate "wow, what is that?" reaction
2. **Technical Demonstration** - Shows something impossible elsewhere
3. **Shareability** - Right format/duration for target platform
4. **Reproducibility** - Deterministic and documented
5. **Educational Value** - Teaches Morphogen's capabilities

---

## Output Types & Formats

### Visual Outputs

| Format | Use Case | Morphogen API | Quality Settings |
|--------|----------|-----------|------------------|
| **PNG** | High-res stills, thumbnails | `visual.output()` | Lossless, gamma-corrected |
| **JPEG** | Web images, social media | `visual.output()` | 8-bit, adjustable quality |
| **MP4** | Video demos, animations | `visual.video()` | H.264, 30-60 fps, YUV420 |
| **GIF** | Looping demos, READMEs | `visual.video()` | 256 colors, optimized |

### Audio Outputs

| Format | Use Case | Morphogen API | Quality Settings |
|--------|----------|-----------|------------------|
| **WAV** | Lossless audio | `audio.save()` | 44.1kHz, 16/24-bit |
| **FLAC** | Compressed lossless | `audio.save()` | 44.1kHz, lossless compression |
| **MP3** | Web audio (future) | TBD | Lossy compression |

### Interactive Outputs

| Type | Use Case | Morphogen API |
|------|----------|-----------|
| **Live Display** | Development, demos | `visual.display()` |
| **Real-time Audio** | Sound design | `audio.play()` |

### Combined Outputs

**Audio-Visual Synchronization**
```python
# Generate both visual and audio from same simulation
frames = []
audio_samples = []

for t in range(simulation_steps):
    state = simulate_step(t)
    frames.append(render_visual(state))
    audio_samples.append(generate_audio(state))

# Export both
visual.video(frames, "output.mp4", fps=30)
audio.save(np.concatenate(audio_samples), "output.wav")

# Combine with ffmpeg (external)
# ffmpeg -i output.mp4 -i output.wav -c:v copy -c:a aac combined.mp4
```

---

## The Output Generation Pipeline

### Standard Workflow

```
┌─────────────┐
│ 1. PROTOTYPE│  Interactive development
│   display() │  Fast iteration, parameter tuning
└──────┬──────┘
       ↓
┌─────────────┐
│ 2. REFINE   │  Deterministic seeding
│   Params    │  Quality settings, resolution
└──────┬──────┘
       ↓
┌─────────────┐
│ 3. EXPORT   │  Multi-format generation
│   Files     │  PNG/MP4/WAV/FLAC
└──────┬──────┘
       ↓
┌─────────────┐
│4. POST-PROC │  Optional enhancement
│   (optional)│  Color grading, ffmpeg muxing
└─────────────┘
```

### Phase 1: Prototype

**Goal**: Fast iteration with immediate visual feedback

```python
import numpy as np
from morphogen.stdlib import visual, field

# Quick parameter exploration
def prototype():
    # Use display() for real-time feedback
    temp = field.alloc((256, 256), fill_value=0.0)

    for step in range(1000):
        temp = field.diffuse(temp, rate=0.1, dt=0.01)

        # Live visualization
        vis = visual.colorize(temp, palette="fire")
        visual.display(vis)  # Interactive window

        # Controls: SPACE=pause, →=step, ↑↓=speed, Q=quit
```

**Tips**:
- Use smaller resolutions (128×128) for speed
- Shorter time spans for quick iteration
- Try different palettes interactively
- Test composition ideas

### Phase 2: Refine

**Goal**: Finalize parameters and ensure reproducibility

```python
def refine():
    # Deterministic seed for reproducibility
    np.random.seed(42)

    # Production resolution
    width, height = 1920, 1080  # Full HD

    # High-quality settings
    temp = field.alloc((height, width), fill_value=0.0)

    # Fine-tuned parameters
    diffusion_rate = 0.15  # Tested value
    dt = 0.01
    n_steps = 2000  # Longer evolution

    # ... simulation ...
```

**Best Practices**:
- Document all parameter choices
- Use consistent seeding
- Test full resolution performance
- Verify determinism (run twice, compare output)

### Phase 3: Export

**Goal**: Generate publication-quality outputs

```python
def export_outputs():
    """Generate multiple output formats from one simulation."""
    np.random.seed(42)  # Reproducible

    frames = []

    # Run simulation
    state = initialize_simulation()
    for step in range(n_frames):
        state = simulate_step(state)
        vis = render_frame(state)
        frames.append(vis)

    # Export multiple formats

    # 1. High-res thumbnail (first frame)
    visual.output(frames[0], "thumbnail.png")

    # 2. Key frame sequence
    for i in [0, len(frames)//4, len(frames)//2, len(frames)-1]:
        visual.output(frames[i], f"frame_{i:04d}.png")

    # 3. Full animation (MP4)
    visual.video(frames, "animation.mp4", fps=30)

    # 4. Looping GIF for README
    visual.video(frames[:100], "demo_loop.gif", fps=15)

    print("✓ All outputs generated successfully")
```

### Phase 4: Post-Process (Optional)

**External Tools**:

```bash
# Combine video + audio with ffmpeg
ffmpeg -i visual.mp4 -i audio.wav -c:v copy -c:a aac output.mp4

# Create high-quality GIF with palette
ffmpeg -i input.mp4 -vf "palettegen" palette.png
ffmpeg -i input.mp4 -i palette.png -lavfi "paletteuse" output.gif

# Extract thumbnail
ffmpeg -i video.mp4 -ss 00:00:05 -vframes 1 thumbnail.jpg

# Color grading (optional)
ffmpeg -i input.mp4 -vf "eq=contrast=1.1:brightness=0.05" graded.mp4
```

---

## Cross-Domain Output Recipes

### Recipe 1: "Singing Physics" 🎵

**Concept**: Rigid body collisions generate musical notes

**Domains**: Physics + Audio + Visual

```python
from morphogen.stdlib import rigidbody, audio, visual

def singing_physics():
    """Physics simulation that generates music."""

    # Setup physics
    world = rigidbody.create_world(gravity=(0, -9.8))
    balls = [rigidbody.create_sphere(radius=1.0) for _ in range(10)]

    frames = []
    audio_samples = []

    for step in range(3000):  # 60 sec at 50 Hz physics
        # Physics step
        collisions = world.step(dt=0.02)

        # Generate audio from collisions
        for collision in collisions:
            impact_force = collision.impulse_magnitude
            velocity = collision.relative_velocity

            # Impact force → volume
            volume = min(1.0, impact_force / 100.0)

            # Velocity → pitch (MIDI note)
            note = 36 + int(abs(velocity) * 2)  # C2 to ~C6
            frequency = 440.0 * (2 ** ((note - 69) / 12.0))

            # Generate tone (50ms)
            duration = 0.05
            n_samples = int(44100 * duration)
            t = np.linspace(0, duration, n_samples)

            # ADSR envelope
            attack = 0.01
            decay = 0.02
            sustain = 0.7
            release = 0.02

            envelope = audio.adsr(attack, decay, sustain, release, duration)
            tone = np.sin(2 * np.pi * frequency * t) * envelope * volume

            audio_samples.append(tone)

        # Render visualization every 2 physics steps (25 fps)
        if step % 2 == 0:
            vis = visual.agents(balls, width=1920, height=1080,
                              color_property='velocity',
                              size=20.0,
                              palette='fire')
            frames.append(vis)

    # Export
    visual.video(frames, "singing_physics.mp4", fps=25)
    audio.save(np.concatenate(audio_samples), "singing_physics.wav")

    print("✓ Generated physics → audio demonstration")
    print("  Combine with: ffmpeg -i singing_physics.mp4 -i singing_physics.wav ...")
```

**Output**: MP4 video + WAV audio showing physics creating music

**Why it's compelling**: Impossible in other platforms without extensive glue code

---

### Recipe 2: "Turbulent Tones" 🌊

**Concept**: Fluid vorticity drives audio synthesis

**Domains**: Fluids + Audio + Visual + Agents

```python
from morphogen.stdlib import field, audio, visual, agents

def turbulent_tones():
    """Fluid simulation controls synthesizer parameters."""

    # Initialize velocity field
    vx = field.alloc((256, 256), fill_value=0.0)
    vy = field.alloc((256, 256), fill_value=0.0)

    # Particles that emit sound
    particles = agents.create_swarm(n=100)

    frames = []
    audio_buffer = []

    for step in range(1800):  # 60 seconds at 30 fps
        # Fluid simulation
        vx, vy = field.advect_velocity(vx, vy, dt=0.1)
        vx = field.diffuse(vx, rate=0.01, dt=0.1)
        vy = field.diffuse(vy, rate=0.01, dt=0.1)

        # Add vortex forces
        vx, vy = field.add_vortex(vx, vy, center=(128, 128), strength=50.0)

        # Compute vorticity (curl of velocity)
        vorticity = field.curl(vx, vy)

        # Sample vorticity at particle positions
        for particle in particles:
            local_vort = field.sample(vorticity, particle.position)

            # Vorticity magnitude → frequency
            freq = 200.0 + abs(local_vort) * 500.0

            # Vorticity sign → pan (left/right)
            pan = 0.5 + local_vort / 10.0  # -1 to 1 → 0 to 1

            # Generate audio chunk (1/30 second = 1470 samples)
            samples_per_frame = 44100 // 30
            t = np.linspace(0, 1/30, samples_per_frame)

            tone = np.sin(2 * np.pi * freq * t) * 0.01  # Low volume per particle

            # Apply stereo panning
            left = tone * (1 - pan)
            right = tone * pan

            audio_buffer.append(np.stack([left, right], axis=1))

        # Visualize: field + particles
        field_vis = visual.colorize(vorticity, palette="coolwarm",
                                    vmin=-5.0, vmax=5.0)

        particle_vis = visual.agents(particles, width=256, height=256,
                                     color_property='frequency',
                                     size=4.0,
                                     blend_mode='additive')

        composed = visual.composite([field_vis, particle_vis],
                                   modes=['over', 'add'],
                                   opacity=[1.0, 0.8])
        frames.append(composed)

    # Export
    visual.video(frames, "turbulent_tones.mp4", fps=30)

    # Mix audio from all particles
    mixed_audio = np.sum(audio_buffer, axis=0)
    audio.save(mixed_audio, "turbulent_tones.wav")

    print("✓ Generated fluid → audio synthesis")
```

**Output**: Synchronized video + stereo audio

**Why it's compelling**: Shows cross-domain coupling in real-time

---

### Recipe 3: "Living Patterns" 🦠

**Concept**: Reaction-diffusion patterns control harmonic content

**Domains**: Fields + Audio + Visual

```python
from morphogen.stdlib import field, audio, visual

def living_patterns():
    """Gray-Scott reaction-diffusion generates evolving soundscape."""

    # Reaction-diffusion parameters (for spots/stripes)
    Du, Dv = 0.16, 0.08
    F, K = 0.060, 0.062

    u = field.alloc((256, 256), fill_value=1.0)
    v = field.alloc((256, 256), fill_value=0.0)

    # Initialize with random perturbation
    np.random.seed(42)
    u.data[100:150, 100:150] = 0.5
    v.data[100:150, 100:150] = 0.25 + np.random.rand(50, 50) * 0.1

    frames = []
    audio_chunks = []

    for step in range(600):  # 20 seconds at 30 fps
        # Gray-Scott step
        uvv = u.data * v.data * v.data
        du_dt = Du * field.laplacian(u).data - uvv + F * (1.0 - u.data)
        dv_dt = Dv * field.laplacian(v).data + uvv - (F + K) * v.data

        u.data += du_dt
        v.data += dv_dt

        # Analyze pattern for audio generation

        # 1. Pattern density → bass frequency
        density = np.mean(v.data)
        bass_freq = 80.0 + density * 120.0  # 80-200 Hz

        # 2. Pattern complexity → harmonic richness
        gradient_mag = field.magnitude(field.gradient(v))
        complexity = np.mean(gradient_mag.data)
        n_harmonics = int(1 + complexity * 8)

        # 3. Spatial moments → stereo field
        center_x = np.sum(v.data * np.arange(256)[None, :]) / np.sum(v.data)
        pan = center_x / 256.0  # 0 to 1

        # Generate audio frame
        samples_per_frame = 44100 // 30
        t = np.linspace(0, 1/30, samples_per_frame)

        # Additive synthesis with harmonics
        signal = np.zeros(samples_per_frame)
        for h in range(1, n_harmonics + 1):
            amplitude = 1.0 / h  # Harmonic falloff
            signal += amplitude * np.sin(2 * np.pi * bass_freq * h * t)

        signal *= 0.1  # Volume

        # Stereo panning
        left = signal * (1 - pan)
        right = signal * pan
        stereo = np.stack([left, right], axis=1)

        audio_chunks.append(stereo)

        # Visualize
        if step % 1 == 0:  # Every frame
            vis = visual.colorize(v, palette="viridis", vmin=0.0, vmax=1.0)
            frames.append(vis)

    # Export
    visual.video(frames, "living_patterns.mp4", fps=30)
    audio.save(np.concatenate(audio_chunks), "living_patterns.wav")

    print("✓ Generated reaction-diffusion → audio")
```

**Output**: Mesmerizing patterns with evolving soundscape

**Why it's compelling**: Pattern formation literally sounds like what it looks like

---

## Visual Composition Techniques

### Layer-Based Composition

**The Power of Blend Modes**

```python
from morphogen.stdlib import visual, field

def advanced_composition():
    """Multi-layer visual composition with blend modes."""

    # Layer 1: Background field (heat, fluid, etc.)
    background_field = generate_field_simulation()
    layer1 = visual.colorize(background_field, palette="viridis")

    # Layer 2: Overlay texture (noise, details)
    noise_field = generate_noise_texture()
    layer2 = visual.colorize(noise_field, palette="grayscale")

    # Layer 3: Particles (additive for glow effect)
    particles = generate_particle_system()
    layer3 = visual.agents(particles, width=1920, height=1080,
                          color_property='energy',
                          size=8.0,
                          blend_mode='additive')

    # Layer 4: Trail history (subtle overlay)
    trails = generate_particle_trails()
    layer4 = visual.agents(trails, width=1920, height=1080,
                          alpha=0.3,
                          size=4.0)

    # Compose from bottom to top
    result = visual.composite(
        [layer1, layer2, layer3, layer4],
        modes=['over', 'overlay', 'add', 'over'],
        opacity=[1.0, 0.4, 0.9, 0.6]
    )

    return result
```

### Blend Mode Guide

| Mode | Effect | Best For |
|------|--------|----------|
| `over` | Alpha blending | Standard layering |
| `add` | Additive | Glows, light effects, particles |
| `multiply` | Darkens | Shadows, depth |
| `screen` | Brightens | Highlights, atmospheric effects |
| `overlay` | Contrast boost | Texture overlays |

### Color Palette Strategies

**Scientific/Data Visualization**
```python
# Diverging: for data with meaningful zero
visual.colorize(field, palette="coolwarm", vmin=-1.0, vmax=1.0)

# Sequential: for positive quantities
visual.colorize(field, palette="viridis", vmin=0.0, vmax=max_val)

# Perceptually uniform
visual.colorize(field, palette="viridis")  # or "plasma", "inferno"
```

**Artistic/Aesthetic**
```python
# Fire/heat effects
visual.colorize(field, palette="fire")

# Dramatic contrast
visual.colorize(field, palette="grayscale")

# Custom gradients (future feature)
custom_palette = create_gradient(["#1a1a2e", "#16213e", "#0f3460", "#533483"])
visual.colorize(field, palette=custom_palette)
```

---

## Audio-Visual Synchronization

### Synchronized Timeline

**Challenge**: Different time rates for audio (44.1kHz) and video (30fps)

**Solution**: Sample-accurate scheduling

```python
def synchronized_av():
    """Generate perfectly synchronized audio and video."""

    # Configuration
    video_fps = 30
    audio_sr = 44100
    duration_seconds = 10

    # Calculate frame/sample counts
    n_video_frames = duration_seconds * video_fps  # 300 frames
    n_audio_samples = duration_seconds * audio_sr  # 441000 samples
    samples_per_frame = audio_sr // video_fps      # 1470 samples

    video_frames = []
    audio_buffer = np.zeros(n_audio_samples)

    # Shared simulation state
    state = initialize_state()

    for frame_idx in range(n_video_frames):
        # Current time
        t = frame_idx / video_fps

        # Update simulation
        state = update_state(state, dt=1/video_fps)

        # Generate video frame
        vis = render_visual(state)
        video_frames.append(vis)

        # Generate audio chunk (synchronized)
        audio_start = frame_idx * samples_per_frame
        audio_end = audio_start + samples_per_frame

        audio_chunk = generate_audio(state, samples=samples_per_frame)
        audio_buffer[audio_start:audio_end] = audio_chunk

    # Export (perfectly synchronized)
    visual.video(video_frames, "output.mp4", fps=video_fps)
    audio.save(audio_buffer, "output.wav", sample_rate=audio_sr)

    # Combine (maintains sync)
    # ffmpeg -i output.mp4 -i output.wav -c:v copy -c:a aac final.mp4
```

### Audio-Reactive Visuals

**Variant**: Drive visuals from audio analysis

```python
from morphogen.stdlib import audio

def audio_reactive_visuals():
    """Create visuals that respond to audio."""

    # Load or generate audio
    audio_signal = generate_audio_signal()

    # Analyze audio in frames
    video_fps = 30
    audio_sr = 44100
    samples_per_frame = audio_sr // video_fps

    frames = []

    for frame_idx in range(len(audio_signal) // samples_per_frame):
        # Extract audio chunk
        chunk_start = frame_idx * samples_per_frame
        chunk_end = chunk_start + samples_per_frame
        chunk = audio_signal[chunk_start:chunk_end]

        # Audio analysis
        # 1. Energy (RMS)
        energy = np.sqrt(np.mean(chunk ** 2))

        # 2. Spectral centroid
        fft = np.fft.rfft(chunk)
        centroid = audio.spectral_centroid(fft)

        # 3. Beat detection (simplified)
        beat_detected = energy > threshold

        # Drive visualization parameters
        particle_size = 2.0 + energy * 20.0
        particle_color_shift = centroid / audio_sr  # 0 to 0.5

        if beat_detected:
            emit_new_particles()

        # Render frame
        vis = render_with_params(particle_size, particle_color_shift)
        frames.append(vis)

    visual.video(frames, "audio_reactive.mp4", fps=video_fps)
```

---

## Quality Checklist

Before releasing an output, verify:

### Technical Quality

- [ ] **Resolution**: Appropriate for use case
  - Social media: 1080p minimum
  - Print/publication: 4K (3840×2160)
  - Thumbnails: 1280×720 or higher

- [ ] **Frame rate**: Smooth motion
  - Standard: 30 fps
  - Smooth: 60 fps
  - Artistic: 24 fps (cinematic)

- [ ] **Audio quality**: Clear and distortion-free
  - Sample rate: 44.1kHz minimum
  - Bit depth: 16-bit minimum, 24-bit for production
  - No clipping (check peak levels < 0 dB)

- [ ] **Compression**: Appropriate for delivery
  - MP4: H.264, CRF 18-23 (lower = higher quality)
  - Audio: AAC for web, FLAC for lossless

- [ ] **Color**: Properly gamma-corrected
  - Morphogen handles this automatically via `visual.output()`
  - Verify no color banding

### Content Quality

- [ ] **Visual clarity**: Immediately understandable
  - Clear subject/focus
  - Good contrast
  - Not too cluttered

- [ ] **Duration**: Right length for platform
  - Twitter/X: 15-30 seconds
  - Instagram: 30-60 seconds
  - YouTube: 1-3 minutes
  - Portfolio: 30-90 seconds

- [ ] **Pacing**: Engaging throughout
  - Hook in first 3 seconds
  - Build complexity
  - Satisfying conclusion

- [ ] **Audio mix**: Balanced and pleasant
  - No harsh frequencies
  - Good dynamic range
  - Appropriate volume levels

### Reproducibility

- [ ] **Deterministic**: Same code produces identical output
  ```python
  np.random.seed(42)  # Always set seed
  ```

- [ ] **Documented**: Parameters and process recorded
  ```python
  # Include config dict or docstring
  config = {
      'resolution': (1920, 1080),
      'fps': 30,
      'duration': 20,  # seconds
      'seed': 42,
      'diffusion_rate': 0.15,
      # ... all parameters ...
  }
  ```

- [ ] **Version tracked**: Morphogen version noted
  ```python
  # Tested with Morphogen v0.8.2
  ```

---

## Social Media Optimization

### Platform-Specific Guidelines

| Platform | Aspect Ratio | Max Duration | Max Size | Format |
|----------|-------------|--------------|----------|--------|
| **Twitter/X** | 16:9, 1:1 | 2:20 | 512MB | MP4 |
| **Instagram Feed** | 1:1, 4:5 | 60s | 100MB | MP4 |
| **Instagram Stories** | 9:16 | 60s | 100MB | MP4 |
| **YouTube** | 16:9 | ∞ | 128GB | MP4 |
| **GitHub README** | Any | Short | 10MB | GIF |
| **LinkedIn** | 16:9, 1:1 | 10min | 200MB | MP4 |

### Aspect Ratio Templates

```python
# 16:9 - Standard widescreen (YouTube, Twitter)
width, height = 1920, 1080

# 1:1 - Square (Instagram feed)
width, height = 1080, 1080

# 4:5 - Portrait (Instagram feed)
width, height = 1080, 1350

# 9:16 - Vertical (Instagram/TikTok stories)
width, height = 1080, 1920
```

### Thumbnail Best Practices

**For Video Posts**:
```python
# Extract compelling middle frame
frames = generate_frames()
thumbnail_frame = frames[len(frames) // 2]  # Or best frame
visual.output(thumbnail_frame, "thumbnail.jpg")
```

**For Static Posts**:
- High contrast
- Clear subject
- Text overlay (optional, done externally)
- Bright, saturated colors

### File Size Optimization

**GIF Optimization**:
```python
# Reduce colors and resolution for smaller files
visual.video(frames[::2],  # Skip every other frame
            "demo.gif",
            fps=15,  # Lower fps
            # resolution reduced in frame generation
            )
```

**MP4 Compression**:
```bash
# External: Use ffmpeg for size reduction
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output_compressed.mp4

# CRF values:
# 18 = visually lossless
# 23 = high quality (default)
# 28 = good quality, smaller size
```

---

## Examples & Code Patterns

### Pattern 1: Multi-Format Export

```python
def generate_all_formats(simulation_name):
    """Generate all output formats from one simulation run."""

    np.random.seed(42)

    # Run simulation once, collect all frames
    frames = []
    audio = []

    for step in simulation_generator():
        frames.append(render_visual(step))
        audio.append(generate_audio(step))

    output_dir = Path(f"outputs/{simulation_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thumbnail (high-res PNG)
    visual.output(frames[len(frames)//2],
                 output_dir / "thumbnail.png")

    # 2. Key frames
    for i, idx in enumerate([0, len(frames)//4, len(frames)//2, -1]):
        visual.output(frames[idx],
                     output_dir / f"keyframe_{i}.png")

    # 3. Full animation (MP4, 1080p)
    visual.video(frames,
                output_dir / "full_1080p.mp4",
                fps=30)

    # 4. Twitter-optimized (MP4, 720p, 30s max)
    twitter_frames = frames[:30*30]  # 30 seconds
    # Downscale frames here if needed
    visual.video(twitter_frames,
                output_dir / "twitter.mp4",
                fps=30)

    # 5. README loop (GIF, 10s, small)
    readme_frames = frames[::2][:150]  # 10s at 15fps
    visual.video(readme_frames,
                output_dir / "readme_loop.gif",
                fps=15)

    # 6. Audio track
    audio.save(np.concatenate(audio),
              output_dir / "soundtrack.wav")

    # 7. Metadata
    metadata = {
        'simulation': simulation_name,
        'resolution': frames[0].shape[:2],
        'frames': len(frames),
        'fps': 30,
        'duration_seconds': len(frames) / 30,
        'seed': 42,
        'morphogen_version': '0.8.2',
        'generated': datetime.now().isoformat(),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Generated all formats in {output_dir}")
    return output_dir
```

### Pattern 2: Parameter Sweep

```python
def parameter_sweep():
    """Generate outputs across parameter ranges."""

    diffusion_rates = [0.05, 0.10, 0.15, 0.20]

    for rate in diffusion_rates:
        print(f"Generating with diffusion_rate={rate}")

        frames = run_simulation(diffusion_rate=rate)

        visual.video(frames,
                    f"outputs/diffusion_rate_{rate:.2f}.mp4",
                    fps=30)

    print("✓ Parameter sweep complete")
    print("  Review outputs/ directory to choose best parameters")
```

### Pattern 3: Comparison Grid

```python
def comparison_grid():
    """Generate side-by-side comparison of techniques."""

    techniques = ['technique_a', 'technique_b', 'technique_c', 'technique_d']

    for frame_idx in range(n_frames):
        # Generate frame for each technique
        subframes = []
        for tech in techniques:
            subframe = generate_frame(tech, frame_idx)
            subframes.append(subframe)

        # Arrange in 2×2 grid
        top_row = np.hstack([subframes[0], subframes[1]])
        bottom_row = np.hstack([subframes[2], subframes[3]])
        grid = np.vstack([top_row, bottom_row])

        frames.append(grid)

    visual.video(frames, "comparison_grid.mp4", fps=30)
```

---

## Troubleshooting

### Common Issues

**1. "Output looks washed out / too dark"**

*Cause*: Incorrect normalization or gamma

*Solution*:
```python
# Ensure proper value range for colorization
field_normalized = field.normalize(field, vmin=0.0, vmax=1.0)
vis = visual.colorize(field_normalized, palette="viridis")

# visual.output() handles gamma correction automatically
```

**2. "GIF is huge / too large to upload"**

*Cause*: Too many frames or high resolution

*Solution*:
```python
# Reduce frames (skip frames, lower fps)
frames_reduced = frames[::2]  # Every other frame

# Reduce resolution (resize frames before export)
# Or generate at lower resolution

# Shorter duration
frames_short = frames[:150]  # 10 seconds at 15fps

visual.video(frames_short, "output.gif", fps=15)
```

**3. "Audio is clipping / distorted"**

*Cause*: Audio peaks exceed ±1.0

*Solution*:
```python
# Check and normalize audio
peak = np.max(np.abs(audio_signal))
if peak > 1.0:
    print(f"Warning: audio peak = {peak}, normalizing...")
    audio_signal = audio_signal / peak * 0.95  # Leave headroom

audio.save(audio_signal, "output.wav")
```

**4. "Video and audio out of sync"**

*Cause*: Mismatched frame/sample counts

*Solution*:
```python
# Calculate exact sample counts
video_fps = 30
audio_sr = 44100
samples_per_frame = audio_sr // video_fps  # Must be integer!

n_frames = len(video_frames)
expected_audio_samples = n_frames * samples_per_frame

# Ensure audio is exact length
audio_signal = audio_signal[:expected_audio_samples]

# Pad if too short
if len(audio_signal) < expected_audio_samples:
    padding = np.zeros(expected_audio_samples - len(audio_signal))
    audio_signal = np.concatenate([audio_signal, padding])
```

**5. "Out of memory during export"**

*Cause*: Storing all frames in memory

*Solution*:
```python
# Use generator pattern (memory-efficient)
def frame_generator():
    for step in range(n_steps):
        yield generate_frame(step)

# visual.video() supports generators
visual.video(frame_generator(), "output.mp4", fps=30)
```

**6. "Colors look different on different devices"**

*Cause*: Color space / gamma issues

*Solution*:
- Morphogen uses linear RGB internally and converts to sRGB for output
- Ensure you're using `visual.output()` or `visual.video()` (handles gamma)
- Don't manually manipulate pixel values after colorization

---

## Conclusion

Creating compelling Morphogen outputs is about:

1. **Showcasing uniqueness** - Do what other platforms can't
2. **Cross-domain magic** - Combine domains in surprising ways
3. **Professional quality** - Publication-ready from the start
4. **Reproducibility** - Deterministic and documented
5. **Shareability** - Right format for the platform

**Remember**: The most compelling outputs are those that make viewers ask:

> *"Wait, how did you do that? I didn't know that was possible!"*

That's when you know you've captured Morphogen's unique value.

---

## Additional Resources

- **[Showcase Examples](../../examples/showcase/README.md)** - Advanced cross-domain demos
- **[Visual API Reference](../../morphogen/stdlib/visual.py)** - Complete visual documentation
- **[Audio API Reference](../../morphogen/stdlib/audio.py)** - Complete audio documentation
- **[Cross-Domain Patterns](../adr/002-cross-domain-architectural-patterns.md)** - Integration patterns
- **[Domain Mesh Catalog](../reference/domain-mesh-catalog.md)** - Domain connectivity reference

---

**Happy Creating!** 🚀

*Last Updated: 2025-11-16*
*Morphogen Version: v0.8.0+*
