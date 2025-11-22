# Troubleshooting Guide

Common issues and solutions for Creative Computation DSL v0.2.2 MVP.

## Installation Issues

### Error: `ModuleNotFoundError: No module named 'numpy'`

**Problem**: Required dependencies not installed.

**Solution**:
```bash
pip install numpy pillow
# Or reinstall the package
pip install -e .
```

### Error: `ModuleNotFoundError: No module named 'creative_computation'`

**Problem**: Package not installed or not in Python path.

**Solution**:
```bash
# Install in development mode
cd /path/to/morphogen
pip install -e .
```

## Runtime Errors

### Error: `TypeError: Expected Field2D, got <type>`

**Problem**: Passing wrong type to field operation.

**Example Error**:
```python
temp = field.diffuse([1, 2, 3], rate=0.1, dt=0.01)  # Wrong!
```

**Solution**: Ensure you're passing Field2D objects:
```python
temp = field.random((64, 64), seed=0)  # Create Field2D
temp = field.diffuse(temp, rate=0.1, dt=0.01)  # ✓ Correct
```

### Error: `ValueError: Field shapes must match`

**Problem**: Combining fields of different sizes.

**Example Error**:
```python
a = field.alloc((64, 64))
b = field.alloc((128, 128))
c = field.combine(a, b, operation="add")  # Error!
```

**Solution**: Ensure fields have the same dimensions:
```python
a = field.alloc((64, 64))
b = field.alloc((64, 64))  # Match the size
c = field.combine(a, b, operation="add")  # ✓ Correct
```

### Error: `ValueError: Projection requires 2-channel velocity field`

**Problem**: Trying to project a scalar field instead of a velocity field.

**Example Error**:
```python
temp = field.random((64, 64), seed=0)
temp = field.project(temp)  # Error! temp is scalar
```

**Solution**: Only use `field.project()` on velocity fields (2-channel):
```python
import numpy as np

# Create velocity field with 2 channels (vx, vy)
vx = field.random((64, 64), seed=1, low=-1, high=1)
vy = field.random((64, 64), seed=2, low=-1, high=1)
velocity_data = np.stack([vx.data, vy.data], axis=-1)
velocity = field.Field2D(velocity_data)

# Now projection works
velocity = field.project(velocity)  # ✓ Correct
```

### Error: `ImportError: PIL/Pillow is required for visual output`

**Problem**: Pillow not installed.

**Solution**:
```bash
pip install pillow
```

## Visualization Issues

### No Output Image Generated

**Problem**: `visual.output()` not called or path issue.

**Checklist**:
1. Did you call `visual.output()`?
   ```python
   vis = visual.colorize(field, palette="fire")
   visual.output(vis, path="output.png")  # Don't forget this!
   ```

2. Is the path writable?
   ```python
   # Use absolute path if unsure
   visual.output(vis, path="/tmp/output.png")
   ```

3. Check for errors:
   ```python
   try:
       visual.output(vis, path="output.png")
   except Exception as e:
       print(f"Error: {e}")
   ```

### Image is All Black/White

**Problem**: Field values out of expected range or constant.

**Debug**:
```python
import numpy as np

print(f"Field min: {np.min(field.data)}")
print(f"Field max: {np.max(field.data)}")
print(f"Field mean: {np.mean(field.data)}")
```

**Solutions**:
1. **Constant field**: Initialize with varying values
   ```python
   field = field.random((64, 64), seed=0)  # Not all zeros
   ```

2. **Extreme values**: Specify vmin/vmax
   ```python
   vis = visual.colorize(field, palette="fire", vmin=0.0, vmax=1.0)
   ```

### Colors Don't Match Expected Palette

**Problem**: Linear RGB vs sRGB, or value range issues.

**Solution**: The visual system uses linear RGB internally and converts to sRGB for output. This is correct. If colors look wrong:

1. Check your value range:
   ```python
   # Normalize to [0, 1]
   data = (field.data - field.data.min()) / (field.data.max() - field.data.min())
   normalized = field.Field2D(data)
   vis = visual.colorize(normalized, palette="fire")
   ```

2. Try different palettes:
   ```python
   for palette in ["grayscale", "fire", "viridis", "coolwarm"]:
       vis = visual.colorize(field, palette=palette)
       visual.output(vis, path=f"output_{palette}.png")
   ```

## Field Operation Issues

### Diffusion Has No Effect

**Problem**: Rate or iterations too low.

**Example**:
```python
# Too subtle
smooth = field.diffuse(rough, rate=0.001, dt=0.01, iterations=1)
```

**Solution**: Increase rate or iterations:
```python
# More noticeable
smooth = field.diffuse(rough, rate=0.3, dt=0.1, iterations=20)
```

### Advection Produces NaN or Inf

**Problem**: Velocity field has extreme values or dt is too large.

**Debug**:
```python
print(f"Max velocity: {np.max(np.abs(velocity.data))}")
print(f"dt: {dt}")
print(f"CFL: {np.max(np.abs(velocity.data)) * dt}")
```

**Solution**: Use smaller timestep (dt) or clamp velocities:
```python
# Reduce dt for stability
dt = 0.01  # Instead of 0.1

# Or clamp velocities
velocity.data = np.clip(velocity.data, -10, 10)
```

### Projection Doesn't Make Velocity Divergence-Free

**Problem**: Not enough iterations or boundaries not handled.

**Solution**:
```python
# Use more iterations
velocity = field.project(velocity, iterations=40)  # Instead of 10

# Apply boundaries after projection
velocity = field.boundary(velocity, spec="periodic")
```

## Performance Issues

### Simulation is Too Slow

**Common Causes**:

1. **Field too large**:
   ```python
   # Slow
   field = field.alloc((1024, 1024))

   # Faster for testing
   field = field.alloc((128, 128))
   ```

2. **Too many iterations**:
   ```python
   # Slow
   field = field.diffuse(field, rate=0.1, dt=0.01, iterations=100)

   # Faster
   field = field.diffuse(field, rate=0.1, dt=0.01, iterations=20)
   ```

3. **Nested loops**:
   ```python
   # Slow - nested field operations
   for i in range(100):
       for j in range(10):
           field = field.diffuse(field, rate=0.1, dt=0.01)

   # Faster - combine iterations
   for i in range(100):
       field = field.diffuse(field, rate=0.1, dt=0.01, iterations=10)
   ```

### Memory Issues

**Problem**: Field too large for available RAM.

**Solution**:
1. Reduce field size
2. Use float16 for less precision:
   ```python
   import numpy as np
   field = field.alloc((512, 512))
   field.data = field.data.astype(np.float16)
   ```

## Determinism Issues

### Results Not Reproducible

**Problem**: Random seed not set or non-deterministic operations.

**Solutions**:

1. **Always use seeds**:
   ```python
   # Non-deterministic
   field = field.random((100, 100))  # Different each time

   # Deterministic
   field = field.random((100, 100), seed=42)  # Same each time
   ```

2. **Check for external randomness**:
   ```python
   # If using NumPy directly, set seed
   import numpy as np
   np.random.seed(42)
   ```

3. **Avoid time-based operations** (for MVP, not applicable yet)

## Parser Errors (DSL Files)

### Error: `Expected RPAREN, got COMMA`

**Problem**: Parser doesn't support tuple syntax yet (MVP limitation).

**Example Error**:
```
temp = field.random((128, 128), seed=42)  # Parser error!
```

**Workaround**: Use Python API directly instead of DSL files:
```python
# In Python (.py file)
from creative_computation.stdlib.field import field
temp = field.random((128, 128), seed=42)  # ✓ Works
```

**Note**: Full DSL parser support is planned for post-MVP.

### Error: `Undefined variable: <name>`

**Problem**: Variable used before definition.

**Solution**: Ensure variables are defined before use:
```python
# Wrong order
result = field.diffuse(temp, rate=0.1, dt=0.01)
temp = field.random((64, 64), seed=0)

# Correct order
temp = field.random((64, 64), seed=0)
result = field.diffuse(temp, rate=0.1, dt=0.01)
```

## Type Errors

### Error: `TypeError: 'FieldOperations' object is not callable`

**Problem**: Trying to call the field namespace directly.

**Example Error**:
```python
result = field((64, 64))  # Wrong!
```

**Solution**: Use the method name:
```python
result = field.alloc((64, 64))  # ✓ Correct
```

### Error: `AttributeError: 'FieldOperations' object has no attribute '<name>'`

**Problem**: Method name typo or non-existent method.

**Check**: Available field methods for MVP:
- `field.alloc`
- `field.random`
- `field.advect`
- `field.diffuse`
- `field.project`
- `field.combine`
- `field.map`
- `field.boundary`

## Getting More Help

### Enable Debug Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Field State

```python
import numpy as np

def debug_field(f, name="field"):
    print(f"\n{name}:")
    print(f"  Shape: {f.shape}")
    print(f"  Dtype: {f.data.dtype}")
    print(f"  Min: {np.min(f.data):.6f}")
    print(f"  Max: {np.max(f.data):.6f}")
    print(f"  Mean: {np.mean(f.data):.6f}")
    print(f"  NaN count: {np.sum(np.isnan(f.data))}")
    print(f"  Inf count: {np.sum(np.isinf(f.data))}")

# Use it
debug_field(temperature, "temperature")
```

### Minimal Reproduction

If you find a bug, create a minimal example:

```python
from creative_computation.stdlib.field import field

# Minimal test case
f = field.random((10, 10), seed=0)
result = field.diffuse(f, rate=0.1, dt=0.01, iterations=1)

# What did you expect?
# What did you get?
print(result)
```

## Known MVP Limitations

These are expected limitations of the MVP release:

1. ✗ No agent-based operations
2. ✗ No signal/audio processing
3. ✗ Limited DSL parser (use Python API)
4. ✗ No MLIR compilation (NumPy only)
5. ✗ Only Jacobi solver (no CG, multigrid)
6. ✗ No GPU acceleration
7. ✗ No real-time visualization
8. ✗ Limited solver profiles

See `MVP_ROADMAP.md` for the full feature plan.

## Reporting Issues

When reporting issues, include:

1. **Creative Computation DSL version**: `ccdsl version`
2. **Python version**: `python --version`
3. **Operating system**: (Linux, macOS, Windows)
4. **Minimal reproduction code**
5. **Expected vs actual behavior**
6. **Error message** (full traceback)

Example:
```
Creative Computation DSL v0.2.2
Python 3.10.0
Ubuntu 22.04

Code:
  field = field.random((64, 64), seed=0)
  result = field.diffuse(field, rate=-0.1, dt=0.01)  # Negative rate

Expected: Error or warning about negative rate
Actual: Produces incorrect results silently

Error: None (silent failure)
```

---

**Still stuck?** Check the examples in `examples/` or read the Getting Started guide at `docs/GETTING_STARTED.md`.
