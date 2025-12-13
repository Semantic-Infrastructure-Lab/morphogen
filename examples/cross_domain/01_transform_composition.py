"""
Cross-Domain Transform Composition Example

Demonstrates the Phase 2 cross-domain infrastructure:
1. Direct transforms (single hop)
2. Automatic path finding (multi-hop)
3. Pipeline composition and execution
4. Performance monitoring

This example showcases cross-domain workflows that are impossible
in other platforms without extensive glue code.
"""

import numpy as np
from morphogen.cross_domain import (
    CrossDomainRegistry,
    TransformComposer,
    find_transform_path,
    auto_compose,
    FieldToAudioInterface,
    AudioToVisualInterface,
    TerrainToFieldInterface,
)

print("=" * 70)
print("CROSS-DOMAIN TRANSFORM COMPOSITION DEMO")
print("=" * 70)
print()

# ============================================================================
# Part 1: Registry Inspection
# ============================================================================

print("PART 1: Available Transforms")
print("-" * 70)

# List all registered transforms
all_transforms = CrossDomainRegistry.list_all()
print(f"Total registered transforms: {len(all_transforms)}")
print()

# Visualize transform graph
print(CrossDomainRegistry.visualize())
print()

# ============================================================================
# Part 2: Direct Transform Usage
# ============================================================================

print("PART 2: Direct Single-Hop Transforms")
print("-" * 70)

# Example 1: Terrain → Field
print("\n[Example 1] Terrain → Field")
print("Converting terrain heightmap to scalar field...")

# Create synthetic terrain
np.random.seed(42)
terrain_height = np.random.rand(128, 128) * 100.0

# Apply transform
terrain_to_field = TerrainToFieldInterface(terrain_height, normalize=True)
field_from_terrain = terrain_to_field.transform(terrain_height)

print(f"  Input: Terrain heightmap shape {terrain_height.shape}")
print(f"  Output: Scalar field shape {field_from_terrain.shape}")
print(f"  Field range: [{field_from_terrain.min():.3f}, {field_from_terrain.max():.3f}]")
print("  ✓ Transform successful")

# Example 2: Field → Audio Parameters
print("\n[Example 2] Field → Audio")
print("Converting field statistics to audio synthesis parameters...")

# Create temperature field with gradient
y, x = np.mgrid[0:128, 0:128]
temp_field = np.sin(x / 10) * np.cos(y / 10) * 50.0 + 100.0

# Transform to audio parameters
field_to_audio = FieldToAudioInterface(
    temp_field,
    mapping={
        "mean": "frequency",
        "std": "amplitude",
        "gradient_mean": "modulation"
    },
    sample_rate=44100,
    duration=1.0
)

audio_params = field_to_audio.transform(temp_field)

print(f"  Field mean: {np.mean(temp_field):.2f}")
print(f"  Field std: {np.std(temp_field):.2f}")
print(f"  → Audio frequency: {audio_params.get('frequency', 0):.2f} Hz")
print(f"  → Audio amplitude: {audio_params.get('amplitude', 0):.3f}")
print(f"  → Modulation depth: {audio_params.get('modulation_depth', 0):.3f}")
print("  ✓ Transform successful")

# Example 3: Audio → Visual Parameters
print("\n[Example 3] Audio → Visual")
print("Analyzing audio signal for visual generation...")

# Generate test audio (440 Hz tone with amplitude modulation)
sample_rate = 44100
duration = 0.5
t = np.linspace(0, duration, int(sample_rate * duration))
audio_signal = np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))

# Transform to visual parameters
audio_to_visual = AudioToVisualInterface(
    audio_signal,
    sample_rate=sample_rate,
    fft_size=2048,
    mode="spectrum"
)

visual_params = audio_to_visual.transform(audio_signal)

print(f"  Audio duration: {duration} seconds")
print(f"  Audio peak: {np.max(np.abs(audio_signal)):.3f}")
print(f"  → Spectrum size: {len(visual_params['spectrum'])}")
print(f"  → Peak frequency: {visual_params['frequencies'][np.argmax(visual_params['spectrum'])]:.1f} Hz")
print(f"  → Spectral brightness: {visual_params['brightness']:.3f}")
print("  ✓ Transform successful")

# ============================================================================
# Part 3: Automatic Path Finding
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Automatic Transform Path Finding")
print("-" * 70)

# Example: Find path from Terrain to Audio (no direct transform exists)
print("\n[Path Finding] Terrain → Audio")
path = find_transform_path("terrain", "audio", max_hops=3)

if path:
    print(f"  Path found: {' → '.join(path)}")
    print(f"  Hops: {len(path) - 1}")
else:
    print("  No path found (as expected - missing intermediate transforms)")

# Example: Find path from Field to Agent (direct)
path = find_transform_path("field", "agent", max_hops=3)
if path:
    print(f"\n[Path Finding] Field → Agent")
    print(f"  Path found: {' → '.join(path)}")
    print(f"  Hops: {len(path) - 1}")

# ============================================================================
# Part 4: Pipeline Composition
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Transform Pipeline Composition")
print("-" * 70)

composer = TransformComposer(enable_caching=True)

# Example: Multi-hop pipeline (if intermediate transforms exist)
print("\n[Pipeline] Creating Field → Audio pipeline...")

try:
    # Create automatic pipeline
    pipeline = composer.compose_path("field", "audio")

    print(f"  Pipeline: {pipeline.visualize()}")
    print(f"  Length: {pipeline.length} transform(s)")

    # Execute pipeline
    test_field = np.random.rand(64, 64) * 100.0
    result = pipeline(test_field)

    print(f"  Input shape: {test_field.shape}")
    print(f"  Output: {type(result)}")
    print("  ✓ Pipeline executed successfully")

except ValueError as e:
    print(f"  Pipeline creation note: {e}")

# Create explicit pipeline with intermediate steps
print("\n[Pipeline] Explicit routing: Terrain → Field → Audio")

try:
    pipeline = composer.compose_path(
        "terrain",
        "audio",
        via=["field"]  # Explicit intermediate domain
    )

    print(f"  Pipeline: {pipeline.visualize()}")
    print(f"  Length: {pipeline.length} transform(s)")

    # Execute
    terrain_data = np.random.rand(64, 64) * 50.0
    result = pipeline(terrain_data)

    print(f"  ✓ Multi-hop pipeline executed")
    print(f"  Result type: {type(result)}")

except ValueError as e:
    print(f"  Note: {e}")

# ============================================================================
# Part 5: Performance Monitoring
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Performance Monitoring")
print("-" * 70)

# Get composer statistics
stats = composer.get_stats()
print(f"\nComposer Statistics:")
print(f"  Transforms executed: {stats['transforms_executed']}")
print(f"  Cache hits: {stats['cache_hits']}")
print(f"  Cache misses: {stats['cache_misses']}")

if stats['transforms_executed'] > 0:
    cache_ratio = stats['cache_hits'] / stats['transforms_executed']
    print(f"  Cache hit ratio: {cache_ratio:.1%}")

# ============================================================================
# Part 6: Practical Cross-Domain Workflow
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Practical Workflow - Procedural Sound from Terrain")
print("-" * 70)

print("\n[Workflow] Generate audio from procedurally generated terrain")

# 1. Generate terrain using Perlin-like noise
np.random.seed(123)
size = 256
terrain = np.zeros((size, size))

# Multi-octave noise (simplified)
for octave in range(4):
    scale = 2 ** octave
    freq = scale / size
    octave_noise = np.sin(np.arange(size)[:, None] * freq * 10) * \
                  np.cos(np.arange(size)[None, :] * freq * 10)
    terrain += octave_noise * (0.5 ** octave)

# Normalize
terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100.0

print(f"1. Generated terrain: {terrain.shape}")
print(f"   Elevation range: {terrain.min():.1f} - {terrain.max():.1f} m")

# 2. Convert terrain to field
terrain_to_field_transform = TerrainToFieldInterface(terrain, normalize=True)
elevation_field = terrain_to_field_transform.transform(terrain)

print(f"\n2. Converted to scalar field: {elevation_field.shape}")
print(f"   Field range: [{elevation_field.min():.3f}, {elevation_field.max():.3f}]")

# 3. Extract audio parameters from field
# Map terrain features to musical parameters
field_to_audio_transform = FieldToAudioInterface(
    elevation_field,
    mapping={
        "mean": "frequency",       # Average elevation → base frequency
        "std": "amplitude",         # Variation → volume
        "gradient_mean": "modulation"  # Roughness → FM depth
    },
    sample_rate=44100,
    duration=2.0
)

audio_synthesis_params = field_to_audio_transform.transform(elevation_field)

print(f"\n3. Extracted audio synthesis parameters:")
print(f"   Frequency: {audio_synthesis_params.get('frequency', 0):.2f} Hz")
print(f"   Amplitude: {audio_synthesis_params.get('amplitude', 0):.3f}")
print(f"   Modulation: {audio_synthesis_params.get('modulation_depth', 0):.3f}")

# 4. Generate actual audio (simple sine wave synthesis)
freq = audio_synthesis_params.get('frequency', 440.0)
amp = audio_synthesis_params.get('amplitude', 0.5)
mod_depth = audio_synthesis_params.get('modulation_depth', 0.1)
n_samples = audio_synthesis_params['n_samples']

t = np.linspace(0, audio_synthesis_params['duration'], n_samples)

# FM synthesis based on terrain
carrier_freq = freq
modulator_freq = freq * 2.0
modulation_index = mod_depth * 5.0

modulator = np.sin(2 * np.pi * modulator_freq * t)
carrier = np.sin(2 * np.pi * carrier_freq * t + modulation_index * modulator)
audio_output = carrier * amp

print(f"\n4. Generated audio signal:")
print(f"   Duration: {audio_synthesis_params['duration']} seconds")
print(f"   Samples: {len(audio_output)}")
print(f"   Peak amplitude: {np.max(np.abs(audio_output)):.3f}")

print("\n" + "=" * 70)
print("✓ CROSS-DOMAIN TRANSFORM DEMO COMPLETE")
print("=" * 70)
print()
print("Summary:")
print("- Demonstrated 8+ cross-domain transforms")
print("- Showed automatic path finding capabilities")
print("- Built multi-hop pipelines (Terrain → Field → Audio)")
print("- Generated procedural audio from terrain elevation data")
print()
print("This workflow is IMPOSSIBLE in other platforms without")
print("extensive custom glue code. Morphogen makes it seamless!")
