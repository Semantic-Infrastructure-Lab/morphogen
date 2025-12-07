# Easy Domain Transforms to Add

Quick implementation guide for high-impact, low-effort transforms.

---

## 1. audio → time (TRIVIAL - 15 lines) 🔥🔥🔥

**Impact:** Connects the entire time-frequency island to main network!

**Why it's easy:** Audio data IS already time-domain - just wrap it with metadata.

```python
from morphogen.cross_domain.interface import DomainInterface

class AudioToTimeInterface(DomainInterface):
    """Convert audio signal to time-domain representation."""

    def transform(self, audio_data, **kwargs):
        """
        Args:
            audio_data: 1D numpy array (audio signal)
            **kwargs: sample_rate (default: 44100)

        Returns:
            Time-domain representation (dict or array)
        """
        sample_rate = kwargs.get('sample_rate', 44100)

        return {
            'signal': audio_data,
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate,
            'num_samples': len(audio_data)
        }
```

**Test it:**
```python
# Register the transform
from morphogen.cross_domain.registry import CrossDomainRegistry
CrossDomainRegistry.register("audio", "time", AudioToTimeInterface)

# Use it
import numpy as np
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # 440 Hz tone

transform = AudioToTimeInterface()
time_data = transform.transform(audio, sample_rate=44100)

print(f"Duration: {time_data['duration']} seconds")
# Output: Duration: 1.0 seconds
```

**What this unlocks:**
```python
# Now you can do:
terrain → field → audio → time → wavelet
                              ↓
                         cepstral

# Full spectral analysis pipeline!
```

---

## 2. visual → field (VERY EASY - 20 lines) 🔥🔥🔥

**Impact:** Creates infinite feedback loops for generative art!

**Why it's easy:** Both are 2D/3D arrays - just convert pixels to field values.

```python
import numpy as np
from morphogen.cross_domain.interface import DomainInterface

class VisualToFieldInterface(DomainInterface):
    """Convert visual image to field data."""

    def transform(self, visual_data, **kwargs):
        """
        Args:
            visual_data: 2D (H, W) or 3D (H, W, C) numpy array
            **kwargs: channel (default: 'grayscale')

        Returns:
            2D field array normalized to [0, 1]
        """
        # Handle RGB/RGBA by converting to grayscale
        if len(visual_data.shape) == 3:
            # Grayscale: 0.299*R + 0.587*G + 0.114*B
            field = (0.299 * visual_data[:, :, 0] +
                    0.587 * visual_data[:, :, 1] +
                    0.114 * visual_data[:, :, 2])
        else:
            field = visual_data.copy()

        # Normalize to [0, 1]
        field_min, field_max = field.min(), field.max()
        if field_max > field_min:
            field = (field - field_min) / (field_max - field_min)

        return field
```

**Test it:**
```python
# Register
CrossDomainRegistry.register("visual", "field", VisualToFieldInterface)

# Use it
visual = np.random.rand(256, 256, 3)  # RGB image
transform = VisualToFieldInterface()
field = transform.transform(visual)

print(f"Field shape: {field.shape}")  # (256, 256)
print(f"Field range: [{field.min()}, {field.max()}]")  # [0.0, 1.0]
```

**What this unlocks:**
```python
# Infinite generative loops!
vision → field → audio → visual → field → audio → ...

# Visual feedback
visual → field → terrain → field → audio → visual

# Breaks visual sink node!
```

---

## 3. time → audio (TRIVIAL - 10 lines) 🔥🔥

**Impact:** Completes the audio ↔ time bidirectional pair!

**Why it's easy:** Just unwrap the signal from time-domain metadata.

```python
from morphogen.cross_domain.interface import DomainInterface

class TimeToAudioInterface(DomainInterface):
    """Convert time-domain representation back to audio signal."""

    def transform(self, time_data, **kwargs):
        """
        Args:
            time_data: Time-domain dict or array

        Returns:
            1D audio signal array
        """
        # If it's a dict (from AudioToTime), extract signal
        if isinstance(time_data, dict):
            return time_data['signal']

        # Otherwise, assume it's already the signal
        return time_data
```

**Test it:**
```python
# Register both transforms
CrossDomainRegistry.register("audio", "time", AudioToTimeInterface)
CrossDomainRegistry.register("time", "audio", TimeToAudioInterface)

# Round-trip test
import numpy as np
original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

# audio → time
time_transform = AudioToTimeInterface()
time_data = time_transform.transform(original_audio)

# time → audio
audio_transform = TimeToAudioInterface()
recovered_audio = audio_transform.transform(time_data)

# Check they're identical
assert np.allclose(original_audio, recovered_audio)
print("✓ Round-trip successful!")
```

**What this unlocks:**
```python
# Bidirectional audio ↔ time processing
audio → time → [process] → audio

# Time-domain visualization
time → audio → visual

# Complete integration of time-frequency island
```

---

## 4. field → visual (VERY EASY - 15 lines) 🔥

**Impact:** Direct field visualization shortcut!

**Why it's easy:** Field is 2D array, just colorize it.

```python
import numpy as np
from morphogen.cross_domain.interface import DomainInterface

class FieldToVisualInterface(DomainInterface):
    """Convert field data to visual image."""

    def transform(self, field_data, **kwargs):
        """
        Args:
            field_data: 2D numpy array
            **kwargs: colormap (default: 'grayscale')

        Returns:
            3D RGB image array (H, W, 3)
        """
        # Normalize to [0, 255]
        field_min, field_max = field_data.min(), field_data.max()
        normalized = (field_data - field_min) / (field_max - field_min)
        visual = (normalized * 255).astype(np.uint8)

        # Convert grayscale to RGB
        return np.stack([visual, visual, visual], axis=-1)
```

**Test it:**
```python
field = np.random.rand(128, 128)
transform = FieldToVisualInterface()
visual = transform.transform(field)

print(f"Visual shape: {visual.shape}")  # (128, 128, 3)
print(f"Visual range: [{visual.min()}, {visual.max()}]")  # [0, 255]
```

**What this unlocks:**
```python
# Direct visualization (skip audio step)
terrain → field → visual  (2 hops instead of 3!)

# Faster field debugging
field → visual  (instant visualization)
```

---

## 5. geometry → cartesian (EASY - 15 lines) 🔥🔥

**Impact:** Connects coordinate system island!

**Why it's easy:** Geometry has vertices - just extract coordinates.

```python
import numpy as np
from morphogen.cross_domain.interface import DomainInterface

class GeometryToCartesianInterface(DomainInterface):
    """Convert geometry vertices to cartesian coordinates."""

    def transform(self, geometry_data, **kwargs):
        """
        Args:
            geometry_data: Dict with 'vertices' or raw vertex array

        Returns:
            Dict with cartesian points (N, 2) or (N, 3)
        """
        # Extract vertices
        if isinstance(geometry_data, dict):
            vertices = geometry_data.get('vertices', geometry_data.get('points'))
        else:
            vertices = geometry_data

        return {
            'points': vertices,
            'dimensions': vertices.shape[1]  # 2D or 3D
        }
```

**Test it:**
```python
# Create a simple triangle
geometry = {
    'vertices': np.array([[0, 0], [1, 0], [0.5, 0.866]]),
    'faces': [[0, 1, 2]]
}

transform = GeometryToCartesianInterface()
cartesian = transform.transform(geometry)

print(f"Points: {cartesian['points']}")
print(f"Dimensions: {cartesian['dimensions']}")  # 2
```

**What this unlocks:**
```python
# Full coordinate system integration!
geometry → cartesian → polar

# Geometric data in spatial workflows
geometry → cartesian → [polar → field once implemented]
```

---

## 6. audio → vision (EASY - 20 lines) 🔥🔥

**Impact:** True synesthesia - sound becomes image!

**Why it's easy:** Spectrogram is already an image.

```python
import numpy as np
from scipy import signal
from morphogen.cross_domain.interface import DomainInterface

class AudioToVisionInterface(DomainInterface):
    """Convert audio to visual spectrogram."""

    def transform(self, audio_data, **kwargs):
        """
        Args:
            audio_data: 1D audio signal
            **kwargs: sample_rate (default: 44100)

        Returns:
            2D spectrogram array (frequency × time)
        """
        sample_rate = kwargs.get('sample_rate', 44100)

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate)

        # Log scale for better visualization
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Normalize to [0, 1]
        spec_min, spec_max = Sxx_db.min(), Sxx_db.max()
        spec_normalized = (Sxx_db - spec_min) / (spec_max - spec_min)

        return spec_normalized
```

**Test it:**
```python
# Create a chirp signal (frequency sweep)
t = np.linspace(0, 1, 44100)
audio = np.sin(2 * np.pi * (100 + 300*t) * t)

transform = AudioToVisionInterface()
vision = transform.transform(audio, sample_rate=44100)

print(f"Spectrogram shape: {vision.shape}")
# Output: Spectrogram shape: (129, 342) - frequency × time
```

**What this unlocks:**
```python
# Sound → image (synesthesia!)
audio → vision

# Reverse path
vision → field → audio
audio → vision  (different path back!)

# Audio-reactive visuals
physics → audio → vision
```

---

## Implementation Checklist

For each transform:

1. **Create the Interface class**
   ```python
   class SourceToTargetInterface(DomainInterface):
       def transform(self, data, **kwargs):
           # Implementation
           return result
   ```

2. **Register it**
   ```python
   # In morphogen/cross_domain/registry.py
   # Inside register_builtin_transforms():

   from .interface import SourceToTargetInterface
   CrossDomainRegistry.register("source", "target", SourceToTargetInterface)
   ```

3. **Test it**
   ```python
   # Create test data
   # Apply transform
   # Verify output
   ```

4. **Update documentation**
   - Add to `CROSS_DOMAIN_MESH_CATALOG.md`
   - Update transform count
   - Add example to `MESH_USER_GUIDE.md`
   - Regenerate visualization: `python tools/visualize_mesh.py`

---

## Quick Impact Summary

| Transform | Lines | Impact | What it does |
|-----------|-------|--------|--------------|
| audio → time | 15 | 🔥🔥🔥 | Connects time-frequency island |
| visual → field | 20 | 🔥🔥🔥 | Enables feedback loops |
| time → audio | 10 | 🔥🔥 | Completes bidirectional pair |
| field → visual | 15 | 🔥 | Direct field viz shortcut |
| geometry → cartesian | 15 | 🔥🔥 | Connects coordinate island |
| audio → vision | 20 | 🔥🔥 | Synesthesia! |

**Total effort:** ~95 lines of code
**Total impact:** Connects 2 isolated islands + feedback loops + shortcuts

---

## Next Steps

**Start with the top 3:**
1. `audio → time` (connects island)
2. `visual → field` (feedback loops)
3. `time → audio` (completes pair)

These three transforms add ~45 lines of code but unlock:
- Entire time-frequency domain
- Infinite generative loops
- Round-trip audio processing
- Full spectral analysis workflows

**After that:**
4. `field → visual` (nice shortcut)
5. `geometry → cartesian` (coordinate integration)
6. `audio → vision` (synesthesia)

---

**Ready to implement?** Each transform follows the same pattern - copy, modify, test, register!
