# Testing Strategy for Creative Computation DSL

**Last Updated:** 2025-11-05
**Status:** Recommendations for improved test coverage

---

## Current State Analysis

### ✅ What's Working Well

**Strong Foundation:**
- 66 total tests with 100% pass rate
- Good test organization (separate files per module)
- Comprehensive field and visual operation coverage
- Determinism testing baked in
- Edge case testing present

**Good Practices:**
- Class-based test organization
- Descriptive test names
- Edge case coverage
- Determinism verification
- Integration tests for full pipeline

### ⚠️ Gaps and Improvement Areas

1. **No pytest installed in environment** - Using unittest runner
2. **No code coverage tracking** - Don't know actual coverage %
3. **Missing integration tests** - Need end-to-end DSL programs
4. **No performance benchmarks** - Can't detect regressions
5. **Missing fuzzing/property tests** - Could catch edge cases
6. **No regression test suite** - Bug fixes not captured as tests
7. **Limited runtime/parser integration tests** - Components tested in isolation
8. **No cross-platform CI** - Manual testing only

---

## Recommended Testing Improvements

### 1. Testing Infrastructure ⚡ **HIGH PRIORITY**

#### Install Full Test Suite
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Should include:
# - pytest>=7.0.0
# - pytest-cov>=4.0.0 (coverage)
# - pytest-xdist>=3.0.0 (parallel testing)
# - pytest-benchmark>=4.0.0 (performance testing)
# - hypothesis>=6.0.0 (property-based testing)
```

#### Add pytest Configuration
**File:** `pyproject.toml`
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--verbose",
    "--strict-markers",
    "--cov=creative_computation",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as benchmarks",
    "determinism: marks tests that verify deterministic behavior",
]

[tool.coverage.run]
source = ["creative_computation"]
omit = [
    "*/tests/*",
    "*/examples/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

#### GitHub Actions CI/CD
**File:** `.github/workflows/test.yml`
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

### 2. Integration Tests 🔗 **HIGH PRIORITY**

#### End-to-End DSL Execution Tests
**File:** `tests/test_integration.py`

```python
"""Integration tests for full DSL programs."""

import pytest
import tempfile
from pathlib import Path
from creative_computation.cli import main
from creative_computation.runtime.runtime import Runtime, ExecutionContext


class TestFullProgramExecution:
    """Test complete DSL program execution."""

    def test_simple_diffusion_program(self):
        """Test executing a simple diffusion program."""
        program = """
        # Simple heat diffusion
        temp = field.random((64, 64), seed=42)
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=10)
        vis = visual.colorize(temp, palette="fire")
        """

        # Parse and execute
        ctx = ExecutionContext(global_seed=42)
        runtime = Runtime(ctx)
        runtime.execute(program)

        # Verify execution completed
        assert 'temp' in runtime.context.variables
        assert 'vis' in runtime.context.variables

    def test_multi_step_simulation(self):
        """Test program with multiple simulation steps."""
        program = """
        temp = field.alloc((32, 32), fill_value=0.0)

        # Step 1
        temp = field.random((32, 32), seed=1)
        temp = field.diffuse(temp, rate=0.1, dt=0.01, iterations=5)

        # Step 2
        temp = field.diffuse(temp, rate=0.2, dt=0.01, iterations=5)

        vis = visual.colorize(temp, palette="viridis")
        """

        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)
        runtime.execute(program)

        assert runtime.context.variables['temp'].shape == (32, 32)

    def test_cli_run_command(self, tmp_path):
        """Test CLI execution of DSL file."""
        # Create temporary DSL file
        dsl_file = tmp_path / "test.ccdsl"
        dsl_file.write_text("""
        temp = field.random((16, 16), seed=42)
        temp = field.diffuse(temp, rate=0.1, dt=0.01, iterations=5)
        vis = visual.colorize(temp, palette="fire")
        visual.output(vis, path="output.png")
        """)

        # Execute via CLI
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ['ccdsl', 'run', str(dsl_file)]
            # Should not raise
            # main()  # Would need to capture output properly
        finally:
            sys.argv = old_argv


class TestExamplePrograms:
    """Test that example programs execute correctly."""

    @pytest.mark.parametrize("example", [
        "examples/mvp_test_simple.ccdsl",
        "examples/test_heat_diffusion.ccdsl",
    ])
    def test_example_executes(self, example):
        """Test that example program executes without error."""
        example_path = Path(__file__).parent.parent / example

        if not example_path.exists():
            pytest.skip(f"Example {example} not found")

        # Parse and validate
        with open(example_path) as f:
            content = f.read()

        # Should parse without error
        from creative_computation.lexer.lexer import Lexer
        from creative_computation.parser.parser import Parser

        lexer = Lexer(content)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast is not None


class TestDeterminismAcrossRuns:
    """Test that programs produce identical results."""

    def test_same_seed_same_output(self):
        """Test that same seed produces identical results."""
        program = """
        temp = field.random((64, 64), seed=12345)
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
        """

        # Run 1
        ctx1 = ExecutionContext(global_seed=12345)
        runtime1 = Runtime(ctx1)
        runtime1.execute(program)
        result1 = runtime1.context.variables['temp'].data.copy()

        # Run 2
        ctx2 = ExecutionContext(global_seed=12345)
        runtime2 = Runtime(ctx2)
        runtime2.execute(program)
        result2 = runtime2.context.variables['temp'].data.copy()

        # Should be bit-identical
        import numpy as np
        assert np.array_equal(result1, result2)

    def test_different_seed_different_output(self):
        """Test that different seeds produce different results."""
        program = """
        temp = field.random((64, 64), seed={seed})
        """

        ctx1 = ExecutionContext(global_seed=1)
        runtime1 = Runtime(ctx1)
        runtime1.execute(program.format(seed=1))
        result1 = runtime1.context.variables['temp'].data.copy()

        ctx2 = ExecutionContext(global_seed=2)
        runtime2 = Runtime(ctx2)
        runtime2.execute(program.format(seed=2))
        result2 = runtime2.context.variables['temp'].data.copy()

        import numpy as np
        assert not np.array_equal(result1, result2)
```

---

### 3. Property-Based Testing 🎲 **MEDIUM PRIORITY**

Use Hypothesis for generative testing:

**File:** `tests/test_properties.py`

```python
"""Property-based tests using Hypothesis."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from creative_computation.stdlib.field import field


# Strategies for test generation
field_shapes = st.tuples(
    st.integers(min_value=8, max_value=128),
    st.integers(min_value=8, max_value=128),
)

field_values = st.floats(
    min_value=-1000.0, max_value=1000.0,
    allow_nan=False, allow_infinity=False
)


class TestFieldProperties:
    """Property-based tests for field operations."""

    @given(shape=field_shapes, fill_value=field_values)
    @settings(max_examples=50)
    def test_alloc_creates_constant_field(self, shape, fill_value):
        """Allocated fields should be constant."""
        f = field.alloc(shape, fill_value=fill_value)
        assert f.shape == shape
        assert np.all(f.data == fill_value)

    @given(shape=field_shapes, seed=st.integers(0, 10000))
    @settings(max_examples=50)
    def test_random_is_deterministic(self, shape, seed):
        """Random fields should be deterministic given seed."""
        f1 = field.random(shape, seed=seed)
        f2 = field.random(shape, seed=seed)
        assert np.array_equal(f1.data, f2.data)

    @given(
        shape=field_shapes,
        rate=st.floats(min_value=0.0, max_value=1.0),
        dt=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=30, deadline=5000)
    def test_diffusion_preserves_shape(self, shape, rate, dt):
        """Diffusion should never change field shape."""
        f = field.random(shape, seed=42)
        f_diffused = field.diffuse(f, rate=rate, dt=dt, iterations=5)
        assert f_diffused.shape == shape

    @given(
        shape=field_shapes,
        v1=field_values,
        v2=field_values,
    )
    @settings(max_examples=50)
    def test_combine_operations_commutative(self, shape, v1, v2):
        """Add and mul should be commutative."""
        f1 = field.alloc(shape, fill_value=v1)
        f2 = field.alloc(shape, fill_value=v2)

        # Test add commutativity
        r1 = field.combine(f1, f2, operation="add")
        r2 = field.combine(f2, f1, operation="add")
        assert np.allclose(r1.data, r2.data)

        # Test mul commutativity
        r1 = field.combine(f1, f2, operation="mul")
        r2 = field.combine(f2, f1, operation="mul")
        assert np.allclose(r1.data, r2.data)


class TestVisualProperties:
    """Property-based tests for visual operations."""

    @given(
        shape=field_shapes,
        palette=st.sampled_from(["grayscale", "fire", "viridis", "coolwarm"])
    )
    @settings(max_examples=30)
    def test_colorize_produces_valid_rgb(self, shape, palette):
        """Colorized fields should have valid RGB values."""
        f = field.random(shape, seed=42)
        vis = visual.colorize(f, palette=palette)

        # RGB values in [0, 1]
        assert np.all(vis.data >= 0.0)
        assert np.all(vis.data <= 1.0)

        # Has 3 channels
        assert vis.data.shape == (*shape, 3)
```

---

### 4. Performance/Benchmark Tests ⚡ **MEDIUM PRIORITY**

**File:** `tests/test_benchmarks.py`

```python
"""Performance benchmarks and regression tests."""

import pytest
import numpy as np
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual


@pytest.mark.benchmark
class TestFieldPerformance:
    """Benchmark field operations."""

    def test_diffusion_performance_128(self, benchmark):
        """Benchmark diffusion on 128x128 grid."""
        f = field.random((128, 128), seed=42)

        result = benchmark(
            field.diffuse,
            f, rate=0.5, dt=0.1, iterations=20
        )

        assert result.shape == (128, 128)

    def test_diffusion_performance_256(self, benchmark):
        """Benchmark diffusion on 256x256 grid."""
        f = field.random((256, 256), seed=42)

        result = benchmark(
            field.diffuse,
            f, rate=0.5, dt=0.1, iterations=20
        )

        assert result.shape == (256, 256)

    def test_advection_performance(self, benchmark):
        """Benchmark advection."""
        scalar = field.random((128, 128), seed=1)
        vx = field.alloc((128, 128), fill_value=0.5)
        vy = field.alloc((128, 128), fill_value=0.5)
        velocity = field.Field2D(np.stack([vx.data, vy.data], axis=-1))

        result = benchmark(
            field.advect,
            scalar, velocity, dt=0.01
        )

        assert result.shape == (128, 128)

    def test_full_simulation_step(self, benchmark):
        """Benchmark full simulation step (advect + diffuse + project)."""
        def simulation_step():
            # Create fields
            vx = field.random((64, 64), seed=1, low=-1, high=1)
            vy = field.random((64, 64), seed=2, low=-1, high=1)
            velocity = field.Field2D(np.stack([vx.data, vy.data], axis=-1))

            # Advect
            velocity = field.advect(velocity, velocity, dt=0.01)

            # Project
            velocity = field.project(velocity, iterations=20)

            return velocity

        result = benchmark(simulation_step)
        assert result.shape == (64, 64, 2)


@pytest.mark.benchmark
class TestScalingBehavior:
    """Test performance scaling with problem size."""

    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_diffusion_scaling(self, size, benchmark):
        """Test how diffusion scales with grid size."""
        f = field.random((size, size), seed=42)

        result = benchmark(
            field.diffuse,
            f, rate=0.5, dt=0.1, iterations=10
        )

        assert result.shape == (size, size)

        # Time should scale roughly as O(n^2) for 2D grid
        # Can analyze results with: pytest-benchmark compare
```

---

### 5. Regression Tests 🐛 **HIGH PRIORITY**

**File:** `tests/test_regressions.py`

```python
"""Regression tests for previously discovered bugs."""

import pytest
import numpy as np
from creative_computation.stdlib.field import field


class TestKnownBugs:
    """Tests for bugs that have been fixed."""

    def test_bug_boundary_corner_handling(self):
        """
        Regression test for bug #XXX:
        Boundary conditions didn't handle corners correctly.
        """
        f = field.alloc((10, 10), fill_value=1.0)
        f.data[5, 5] = 5.0

        f_bound = field.boundary(f, spec="reflect")

        # Corner should be handled correctly
        assert not np.isnan(f_bound.data[0, 0])
        assert not np.isnan(f_bound.data[-1, -1])

    def test_bug_diffusion_zero_iterations(self):
        """
        Regression test for bug #XXX:
        Zero iterations caused crash.
        """
        f = field.random((32, 32), seed=42)

        # Should handle gracefully
        f_diff = field.diffuse(f, rate=0.1, dt=0.01, iterations=0)
        assert np.allclose(f.data, f_diff.data)

    def test_bug_colorize_constant_field(self):
        """
        Regression test for bug #XXX:
        Constant fields caused division by zero.
        """
        f = field.alloc((32, 32), fill_value=0.5)

        # Should not crash
        vis = visual.colorize(f, palette="fire")
        assert vis.shape == (32, 32)
```

---

### 6. Error Handling Tests ⚠️ **MEDIUM PRIORITY**

**File:** `tests/test_error_handling.py`

```python
"""Tests for error handling and validation."""

import pytest
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual


class TestFieldValidation:
    """Test field operation input validation."""

    def test_invalid_shape(self):
        """Test that invalid shapes are rejected."""
        with pytest.raises((ValueError, TypeError)):
            field.alloc((-1, -1))

    def test_mismatched_shapes(self):
        """Test that mismatched shapes in combine are caught."""
        f1 = field.alloc((10, 10))
        f2 = field.alloc((20, 20))

        with pytest.raises(ValueError, match="shape"):
            field.combine(f1, f2, operation="add")

    def test_invalid_operation(self):
        """Test that invalid combine operations are caught."""
        f1 = field.alloc((10, 10))
        f2 = field.alloc((10, 10))

        with pytest.raises(ValueError, match="operation"):
            field.combine(f1, f2, operation="invalid_op")

    def test_negative_iterations(self):
        """Test that negative iterations are handled."""
        f = field.random((32, 32), seed=42)

        with pytest.raises(ValueError):
            field.diffuse(f, rate=0.1, dt=0.01, iterations=-5)


class TestVisualValidation:
    """Test visual operation input validation."""

    def test_invalid_palette(self):
        """Test that invalid palettes are rejected."""
        f = field.random((32, 32), seed=42)

        with pytest.raises(ValueError, match="palette"):
            visual.colorize(f, palette="nonexistent")

    def test_invalid_vmin_vmax(self):
        """Test that vmin > vmax is caught."""
        f = field.random((32, 32), seed=42)

        with pytest.raises(ValueError):
            visual.colorize(f, palette="fire", vmin=1.0, vmax=0.0)


class TestHelpfulErrorMessages:
    """Test that error messages are helpful."""

    def test_shape_mismatch_message(self):
        """Test that shape mismatch provides helpful info."""
        f1 = field.alloc((10, 20))
        f2 = field.alloc((15, 25))

        with pytest.raises(ValueError) as exc_info:
            field.combine(f1, f2, operation="add")

        # Error should mention both shapes
        error_msg = str(exc_info.value)
        assert "10" in error_msg or "20" in error_msg
        assert "15" in error_msg or "25" in error_msg
```

---

### 7. Parser/Type Checker Tests 📝 **MEDIUM PRIORITY**

**File:** `tests/test_parser_integration.py`

```python
"""Tests for parser and type checker integration."""

import pytest
from creative_computation.lexer.lexer import Lexer
from creative_computation.parser.parser import Parser
from creative_computation.ast.visitors import TypeChecker


class TestParseRealPrograms:
    """Test parsing real DSL programs."""

    def test_parse_diffusion_program(self):
        """Test parsing a diffusion program."""
        program = """
        temp = field.random((64, 64), seed=42, low=0.0, high=1.0)
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
        vis = visual.colorize(temp, palette="fire")
        """

        lexer = Lexer(program)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast is not None
        assert len(ast.statements) == 3

    def test_parse_with_comments(self):
        """Test that comments are handled correctly."""
        program = """
        # Create initial field
        temp = field.random((64, 64), seed=42)

        # Smooth it out
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
        """

        lexer = Lexer(program)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast is not None

    def test_type_checker_catches_errors(self):
        """Test that type checker catches type errors."""
        program = """
        temp = field.random((64, 64), seed=42)
        # This should fail: can't add field to number
        result = temp + 123
        """

        lexer = Lexer(program)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        checker = TypeChecker()
        # Should detect type error
        # (Depends on type checker implementation)
```

---

## Testing Best Practices

### Test Organization

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_lexer.py              # Unit: Lexer
├── test_parser.py             # Unit: Parser
├── test_field_operations.py   # Unit: Field ops
├── test_visual_operations.py  # Unit: Visual ops
├── test_runtime.py            # Unit: Runtime engine
├── test_integration.py        # Integration: Full programs
├── test_properties.py         # Property-based tests
├── test_benchmarks.py         # Performance tests
├── test_regressions.py        # Bug regression tests
├── test_error_handling.py     # Error cases
└── test_examples.py           # Example program tests
```

### Shared Fixtures

**File:** `tests/conftest.py`

```python
"""Shared pytest fixtures."""

import pytest
import tempfile
from pathlib import Path
from creative_computation.stdlib.field import field
from creative_computation.runtime.runtime import ExecutionContext


@pytest.fixture
def temp_dir():
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def execution_context():
    """Provide fresh execution context."""
    return ExecutionContext(global_seed=42)


@pytest.fixture
def simple_field():
    """Provide simple test field."""
    return field.random((32, 32), seed=42)


@pytest.fixture
def velocity_field():
    """Provide velocity field for testing."""
    import numpy as np
    from creative_computation.stdlib.field import Field2D

    vx = field.alloc((32, 32), fill_value=0.5)
    vy = field.alloc((32, 32), fill_value=0.5)
    return Field2D(np.stack([vx.data, vy.data], axis=-1))
```

---

## Quick Wins (Implement First)

### Priority 1: Set Up Infrastructure
1. ✅ Install pytest and coverage tools
2. ✅ Add pytest.ini configuration
3. ✅ Set up GitHub Actions CI
4. ✅ Add codecov.io integration

### Priority 2: Add Missing Tests
1. ✅ Runtime engine tests
2. ✅ End-to-end integration tests
3. ✅ CLI tests
4. ✅ Error handling tests

### Priority 3: Improve Quality
1. ✅ Property-based tests
2. ✅ Performance benchmarks
3. ✅ Regression test framework

---

## Commands Cheat Sheet

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_field_operations.py

# Run specific test
pytest tests/test_field_operations.py::TestFieldDiffusion::test_diffuse_smooths_field

# Run benchmarks
pytest --benchmark-only

# Run in parallel (4 workers)
pytest -n 4

# Show slowest 10 tests
pytest --durations=10

# Watch for changes and re-run
pytest-watch
```

---

## Metrics and Goals

### Coverage Goals
- **MVP (Current):** 80% coverage minimum
- **v0.3.0:** 85% coverage
- **v1.0.0:** 90% coverage

### Test Count Goals
- **Current:** 66 tests
- **Short term:** 150+ tests
- **v1.0.0:** 300+ tests

### Performance Goals
- All tests run in < 30 seconds
- Benchmarks tracked over time
- No performance regressions

---

## Next Steps

1. **Today:** Install pytest and run with coverage
2. **This Week:** Add integration tests
3. **Next Week:** Set up CI/CD
4. **This Month:** Property-based tests and benchmarks

---

**Questions?** See existing tests for examples or ask for clarification.
