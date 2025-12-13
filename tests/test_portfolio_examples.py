"""Tests for portfolio example programs (v0.3.1).

These tests verify that the portfolio examples run correctly and produce
expected visual outputs. They test both parsing and execution.
"""

import pytest
import tempfile
import os
from pathlib import Path
from morphogen.parser.parser import parse
from morphogen.runtime.runtime import Runtime, ExecutionContext
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual


@pytest.mark.slow
class TestPortfolioExamples:
    """Test portfolio examples parse and execute correctly.

    Note: These tests are marked as slow because they run full simulations.
    Run with: pytest -m slow tests/test_portfolio_examples.py
    """

    @pytest.fixture
    def examples_dir(self):
        """Get examples directory path."""
        return Path(__file__).parent.parent / "examples"

    def test_01_hello_heat_parses(self, examples_dir):
        """Test that 01_hello_heat.morphogen parses correctly."""
        example_path = examples_dir / "01_hello_heat.morph"
        source = example_path.read_text()
        program = parse(source)
        assert program is not None

    def test_01_hello_heat_executes(self, examples_dir):
        """Test that 01_hello_heat.morph executes without errors."""
        example_path = examples_dir / "01_hello_heat.morph"
        source = example_path.read_text()
        program = parse(source)

        runtime = Runtime(ExecutionContext(global_seed=42))
        # Should not raise
        runtime.execute_program(program)

        # Verify temp field exists
        assert runtime.context.has_variable('temp')
        temp = runtime.context.get_variable('temp')
        assert isinstance(temp, Field2D)

    def test_02_pulsing_circle_parses(self, examples_dir):
        """Test that 02_pulsing_circle.morphogen parses correctly."""
        example_path = examples_dir / "02_pulsing_circle.morph"
        source = example_path.read_text()
        program = parse(source)
        assert program is not None

    def test_02_pulsing_circle_executes(self, examples_dir):
        """Test that 02_pulsing_circle.morph executes without errors."""
        example_path = examples_dir / "02_pulsing_circle.morph"
        source = example_path.read_text()
        program = parse(source)

        runtime = Runtime(ExecutionContext(global_seed=42))
        runtime.execute_program(program)

        # Verify time variable exists and has been updated
        assert runtime.context.has_variable('time')
        time_value = runtime.context.get_variable('time')
        assert time_value > 0.0  # Should have advanced

    def test_03_wave_ripples_parses(self, examples_dir):
        """Test that 03_wave_ripples.morphogen parses correctly."""
        example_path = examples_dir / "03_wave_ripples.morph"
        source = example_path.read_text()
        program = parse(source)
        assert program is not None

    def test_03_wave_ripples_executes(self, examples_dir):
        """Test that 03_wave_ripples.morph executes without errors."""
        example_path = examples_dir / "03_wave_ripples.morph"
        source = example_path.read_text()
        program = parse(source)

        runtime = Runtime(ExecutionContext(global_seed=42))
        runtime.execute_program(program)

        # Verify wave fields exist
        assert runtime.context.has_variable('u')
        assert runtime.context.has_variable('v')
        u = runtime.context.get_variable('u')
        v = runtime.context.get_variable('v')
        assert isinstance(u, Field2D)
        assert isinstance(v, Field2D)

    def test_10_heat_equation_parses(self, examples_dir):
        """Test that 10_heat_equation.morphogen parses correctly."""
        example_path = examples_dir / "10_heat_equation.morph"
        source = example_path.read_text()
        program = parse(source)
        assert program is not None

    def test_10_heat_equation_executes(self, examples_dir):
        """Test that 10_heat_equation.morph executes without errors."""
        example_path = examples_dir / "10_heat_equation.morph"
        source = example_path.read_text()
        program = parse(source)

        runtime = Runtime(ExecutionContext(global_seed=42))
        runtime.execute_program(program)

        # Verify temp field exists
        assert runtime.context.has_variable('temp')
        temp = runtime.context.get_variable('temp')
        assert isinstance(temp, Field2D)

    def test_11_gray_scott_parses(self, examples_dir):
        """Test that 11_gray_scott.morphogen parses correctly."""
        example_path = examples_dir / "11_gray_scott.morph"
        source = example_path.read_text()
        program = parse(source)
        assert program is not None

    def test_11_gray_scott_executes(self, examples_dir):
        """Test that 11_gray_scott.morph executes without errors."""
        example_path = examples_dir / "11_gray_scott.morph"
        source = example_path.read_text()
        program = parse(source)

        runtime = Runtime(ExecutionContext(global_seed=42))
        runtime.execute_program(program)

        # Verify reaction-diffusion fields exist
        assert runtime.context.has_variable('u')
        assert runtime.context.has_variable('v')
        u = runtime.context.get_variable('u')
        v = runtime.context.get_variable('v')
        assert isinstance(u, Field2D)
        assert isinstance(v, Field2D)


@pytest.mark.slow
class TestPortfolioVisualOutput:
    """Test that portfolio examples generate visual output."""

    @pytest.fixture
    def examples_dir(self):
        """Get examples directory path."""
        return Path(__file__).parent.parent / "examples"

    @pytest.fixture
    def output_dir(self):
        """Create temp directory for outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_hello_heat_visual_output(self, examples_dir, output_dir):
        """Test visual output generation for hello heat example."""
        # Create a simple heat field and visualize it
        temp = field.alloc((128, 128), fill_value=0.0)

        # Set center hot spot
        cx, cy = 64, 64
        for y in range(128):
            for x in range(128):
                dx = x - cx
                dy = y - cy
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < 10:
                    temp.data[y, x] = 100.0

        # Diffuse a few times
        for _ in range(10):
            temp = field.diffuse(temp, rate=0.1, dt=0.1, iterations=10)

        # Visualize
        vis = visual.colorize(temp, palette="fire", vmin=0.0, vmax=100.0)
        output_path = output_dir / "01_hello_heat_test.png"
        visual.output(vis, path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 1000

    def test_wave_ripples_visual_output(self, examples_dir, output_dir):
        """Test visual output generation for wave ripples example."""
        # Create wave fields
        u = field.alloc((128, 128), fill_value=0.0)
        v = field.alloc((128, 128), fill_value=0.0)

        # Initialize with Gaussian bump
        cx, cy = 64, 64
        sigma = 5.0
        for y in range(128):
            for x in range(128):
                dx = x - cx
                dy = y - cy
                dist_sq = dx*dx + dy*dy
                import math
                u.data[y, x] = 1.0 * math.exp(-dist_sq / (2.0 * sigma * sigma))

        # Simulate a few steps
        c_squared = 0.5 * 0.5
        for _ in range(20):
            lap = field.laplacian(u)
            v.data = v.data + lap.data * c_squared * 0.1
            u.data = u.data + v.data * 0.1
            v.data = v.data * 0.995

        # Visualize
        vis = visual.colorize(u, palette="coolwarm", vmin=-1.0, vmax=1.0)
        output_path = output_dir / "03_wave_ripples_test.png"
        visual.output(vis, path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 1000

    def test_gray_scott_visual_output(self, examples_dir, output_dir):
        """Test visual output generation for Gray-Scott example."""
        # Create reaction-diffusion fields
        u = field.alloc((128, 128), fill_value=1.0)
        v = field.alloc((128, 128), fill_value=0.0)

        # Initialize with perturbation in center
        cx, cy = 64, 64
        radius = 10.0
        for y in range(128):
            for x in range(128):
                dx = x - cx
                dy = y - cy
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < radius:
                    u.data[y, x] = 0.5
                    v.data[y, x] = 0.25

        # Simulate Gray-Scott for a few steps
        Du, Dv = 0.16, 0.08
        F, K = 0.060, 0.062
        for _ in range(100):
            uvv = u.data * v.data * v.data
            du_dt = Du * field.laplacian(u).data - uvv + F * (1.0 - u.data)
            dv_dt = Dv * field.laplacian(v).data + uvv - (F + K) * v.data

            u.data = u.data + du_dt * 1.0
            v.data = v.data + dv_dt * 1.0

        # Visualize
        vis = visual.colorize(v, palette="viridis", vmin=0.0, vmax=1.0)
        output_path = output_dir / "11_gray_scott_test.png"
        visual.output(vis, path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 1000


@pytest.mark.slow
class TestPortfolioDeterminism:
    """Test that portfolio examples are deterministic."""

    @pytest.fixture
    def examples_dir(self):
        """Get examples directory path."""
        return Path(__file__).parent.parent / "examples"

    def test_hello_heat_deterministic(self, examples_dir):
        """Test that hello heat example is deterministic."""
        example_path = examples_dir / "01_hello_heat.morph"
        source = example_path.read_text()

        # Run twice with same seed
        program1 = parse(source)
        runtime1 = Runtime(ExecutionContext(global_seed=42))
        runtime1.execute_program(program1)
        temp1 = runtime1.context.get_variable('temp')

        program2 = parse(source)
        runtime2 = Runtime(ExecutionContext(global_seed=42))
        runtime2.execute_program(program2)
        temp2 = runtime2.context.get_variable('temp')

        # Results should be identical
        import numpy as np
        assert np.array_equal(temp1.data, temp2.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
