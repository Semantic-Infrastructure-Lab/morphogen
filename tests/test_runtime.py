"""Unit tests for runtime execution engine."""

import pytest
import numpy as np
from morphogen.parser.parser import parse
from morphogen.runtime.runtime import Runtime, ExecutionContext
from morphogen.stdlib.field import field, Field2D


class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_context_creation(self):
        """Test creating execution context."""
        ctx = ExecutionContext(global_seed=42)
        assert ctx.global_seed == 42
        assert ctx.timestep == 0
        assert len(ctx.symbols) == 0

    def test_context_variable_storage(self):
        """Test storing and retrieving variables."""
        ctx = ExecutionContext(global_seed=1)

        # Store variable
        f = field.alloc((10, 10), fill_value=5.0)
        ctx.set_variable('test_field', f)

        # Retrieve variable
        retrieved = ctx.get_variable('test_field')
        assert retrieved is f
        assert np.all(retrieved.data == 5.0)

    def test_context_step_increment(self):
        """Test incrementing step counter."""
        ctx = ExecutionContext(global_seed=1)
        assert ctx.timestep == 0

        ctx.timestep += 1
        assert ctx.timestep == 1

    def test_context_seed_determinism(self):
        """Test that same seed produces same context."""
        ctx1 = ExecutionContext(global_seed=12345)
        ctx2 = ExecutionContext(global_seed=12345)

        assert ctx1.global_seed == ctx2.global_seed


class TestRuntimeExecution:
    """Tests for Runtime class and program execution."""

    def test_runtime_creation(self):
        """Test creating runtime."""
        ctx = ExecutionContext(global_seed=42)
        runtime = Runtime(ctx)
        assert runtime.context is ctx

    def test_simple_field_allocation(self):
        """Test executing simple field allocation."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Simulate: f = field.alloc((32, 32), fill_value=1.0)
        f = field.alloc((32, 32), fill_value=1.0)
        runtime.context.set_variable('f', f)

        assert runtime.context.has_variable('f')
        assert runtime.context.get_variable('f').shape == (32, 32)
        assert np.all(runtime.context.get_variable('f').data == 1.0)

    def test_field_operation_chain(self):
        """Test chaining multiple field operations."""
        ctx = ExecutionContext(global_seed=42)
        runtime = Runtime(ctx)

        # Create field
        f = field.random((64, 64), seed=42)
        runtime.context.set_variable('f', f)

        # Diffuse
        f = field.diffuse(f, rate=0.5, dt=0.1, iterations=10)
        runtime.context.set_variable('f', f)

        # Boundary
        f = field.boundary(f, spec="reflect")
        runtime.context.set_variable('f', f)

        # Should still be correct shape
        assert runtime.context.get_variable('f').shape == (64, 64)

    def test_multiple_variables(self):
        """Test managing multiple variables."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Create multiple fields
        f1 = field.alloc((10, 10), fill_value=1.0)
        f2 = field.alloc((10, 10), fill_value=2.0)
        f3 = field.combine(f1, f2, operation="add")

        runtime.context.set_variable('f1', f1)
        runtime.context.set_variable('f2', f2)
        runtime.context.set_variable('f3', f3)

        # Runtime pre-populates symbols, so check our variables exist
        assert runtime.context.has_variable('f1')
        assert runtime.context.has_variable('f2')
        assert runtime.context.has_variable('f3')
        assert np.allclose(runtime.context.get_variable('f3').data, 3.0)


class TestRuntimeDeterminism:
    """Tests for deterministic runtime behavior."""

    def test_same_seed_same_execution(self):
        """Test that same seed produces identical execution."""
        # Run 1
        ctx1 = ExecutionContext(global_seed=12345)
        runtime1 = Runtime(ctx1)
        f1 = field.random((64, 64), seed=12345)
        f1 = field.diffuse(f1, rate=0.5, dt=0.1, iterations=20)
        runtime1.context.set_variable('result', f1)

        # Run 2
        ctx2 = ExecutionContext(global_seed=12345)
        runtime2 = Runtime(ctx2)
        f2 = field.random((64, 64), seed=12345)
        f2 = field.diffuse(f2, rate=0.5, dt=0.1, iterations=20)
        runtime2.context.set_variable('result', f2)

        # Should be identical
        result1 = runtime1.context.get_variable('result').data
        result2 = runtime2.context.get_variable('result').data
        assert np.array_equal(result1, result2)

    def test_different_seed_different_execution(self):
        """Test that different seeds produce different results."""
        # Run 1
        ctx1 = ExecutionContext(global_seed=1)
        runtime1 = Runtime(ctx1)
        f1 = field.random((64, 64), seed=1)
        runtime1.context.set_variable('result', f1)

        # Run 2
        ctx2 = ExecutionContext(global_seed=2)
        runtime2 = Runtime(ctx2)
        f2 = field.random((64, 64), seed=2)
        runtime2.context.set_variable('result', f2)

        # Should be different
        result1 = runtime1.context.get_variable('result').data
        result2 = runtime2.context.get_variable('result').data
        assert not np.array_equal(result1, result2)


class TestRuntimeErrorHandling:
    """Tests for runtime error handling."""

    def test_undefined_variable_access(self):
        """Test accessing undefined variable."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Should raise KeyError or similar
        with pytest.raises(KeyError):
            _ = runtime.context.get_variable('nonexistent')

    def test_type_mismatch_operations(self):
        """Test operations with mismatched types."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Create fields with different shapes
        f1 = field.alloc((10, 10))
        f2 = field.alloc((20, 20))

        runtime.context.set_variable('f1', f1)
        runtime.context.set_variable('f2', f2)

        # Combining should fail
        with pytest.raises(ValueError):
            field.combine(f1, f2, operation="add")


class TestDoubleBuffering:
    """Tests for double-buffer semantics."""

    def test_buffer_swapping_concept(self):
        """Test double-buffer pattern."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Initial state
        temp_current = field.random((32, 32), seed=1)
        temp_next = field.random((32, 32), seed=1)

        runtime.context.set_variable('temp_current', temp_current)
        runtime.context.set_variable('temp_next', temp_next)

        # Process: temp_next = diffuse(temp_current)
        temp_next = field.diffuse(temp_current, rate=0.1, dt=0.01, iterations=5)
        runtime.context.set_variable('temp_next', temp_next)

        # Swap (would be done by runtime)
        temp_current = temp_next
        runtime.context.set_variable('temp_current', temp_current)

        # Verify buffers are swapped
        assert runtime.context.get_variable('temp_current') is temp_next


class TestRuntimeMemory:
    """Tests for runtime memory management."""

    def test_variable_overwrite(self):
        """Test that variables can be overwritten."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Create initial field
        f = field.alloc((10, 10), fill_value=1.0)
        runtime.context.set_variable('f', f)

        # Overwrite with new field
        f_new = field.alloc((10, 10), fill_value=2.0)
        runtime.context.set_variable('f', f_new)

        # Should have new value
        assert np.all(runtime.context.get_variable('f').data == 2.0)

    def test_multiple_references(self):
        """Test that multiple variables can reference same data."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Create field
        f = field.alloc((10, 10), fill_value=5.0)

        # Store as two variables
        runtime.context.set_variable('f1', f)
        runtime.context.set_variable('f2', f)

        # Both should reference same object
        assert runtime.context.get_variable('f1') is runtime.context.get_variable('f2')


class TestRuntimeState:
    """Tests for runtime state management."""

    def test_step_tracking(self):
        """Test tracking simulation steps."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        # Simulate multiple steps
        for step in range(10):
            ctx.timestep = step
            f = field.random((32, 32), seed=step)
            runtime.context.set_variable(f'step_{step}', f)

        assert ctx.timestep == 9
        # Check all step variables were created
        for step in range(10):
            assert runtime.context.has_variable(f'step_{step}')

    def test_state_persistence(self):
        """Test that state persists across operations."""
        ctx = ExecutionContext(global_seed=42)
        runtime = Runtime(ctx)

        # Create and modify field
        f = field.random((32, 32), seed=42)
        runtime.context.set_variable('temp', f)

        # Multiple operations
        for _ in range(5):
            f = field.diffuse(f, rate=0.1, dt=0.01, iterations=5)
            runtime.context.set_variable('temp', f)

        # State should be preserved
        assert runtime.context.has_variable('temp')
        assert runtime.context.get_variable('temp').shape == (32, 32)


class TestRuntimeControlBlocks:
    """Tests for step and substep execution."""

    def test_execute_step_block_uses_body(self):
        """Step blocks should execute their body and advance one timestep."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)

        program = parse("step { x = 1 }")
        runtime.execute_statement(program.statements[0])

        assert runtime.context.get_variable('x') == 1
        assert runtime.context.timestep == 1

    def test_execute_substep_block_uses_body(self):
        """Substep blocks should execute their body N times and advance each substep."""
        ctx = ExecutionContext(global_seed=1)
        runtime = Runtime(ctx)
        runtime.context.set_variable('x', 0)

        program = parse("substep(2) { x = x + 1 }")
        runtime.execute_statement(program.statements[0])

        assert runtime.context.get_variable('x') == 2
        assert runtime.context.timestep == 2
