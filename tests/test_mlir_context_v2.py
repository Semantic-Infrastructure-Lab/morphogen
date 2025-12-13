"""Unit tests for morphogen.mlir.context (v0.7.0)

Tests for the real MLIR context management using Python bindings.
These tests will be skipped if MLIR is not installed.
"""

import pytest
from morphogen.mlir.context import (
    MorphogenMLIRContext,
    get_mlir_context,
    is_mlir_available,
    MLIR_AVAILABLE,
)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestMorphogenMLIRContext:
    """Tests for MorphogenMLIRContext when MLIR is available."""

    def test_context_creation(self):
        """Test that we can create an MLIR context."""
        ctx = MorphogenMLIRContext()
        assert ctx is not None
        assert ctx.ctx is not None

    def test_context_manager(self):
        """Test context manager protocol."""
        with MorphogenMLIRContext() as ctx:
            assert ctx is not None
            assert ctx.ctx is not None

    def test_module_creation(self):
        """Test creating an MLIR module."""
        with MorphogenMLIRContext() as ctx:
            module = ctx.create_module()
            assert module is not None

    def test_named_module_creation(self):
        """Test creating a named MLIR module."""
        with MorphogenMLIRContext() as ctx:
            module = ctx.create_module("test_module")
            assert module is not None
            # Check that the module has the expected name
            assert "test_module" in str(module)

    def test_global_context(self):
        """Test global context singleton."""
        ctx1 = get_mlir_context()
        ctx2 = get_mlir_context()
        # Should return the same instance
        assert ctx1 is ctx2

    def test_dialect_registration(self):
        """Test that standard dialects are registered."""
        ctx = MorphogenMLIRContext()
        # Context should allow unregistered dialects during development
        assert ctx.ctx.allow_unregistered_dialects is True


class TestMLIRAvailability:
    """Tests that work regardless of MLIR availability."""

    def test_is_mlir_available(self):
        """Test MLIR availability check."""
        result = is_mlir_available()
        assert isinstance(result, bool)
        assert result == MLIR_AVAILABLE

    @pytest.mark.skipif(MLIR_AVAILABLE, reason="Test for when MLIR is NOT installed")
    def test_context_creation_fails_without_mlir(self):
        """Test that creating context fails gracefully without MLIR."""
        with pytest.raises(RuntimeError, match="MLIR Python bindings are required"):
            MorphogenMLIRContext()


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestMLIRIntegration:
    """Integration tests for MLIR context and module operations."""

    def test_simple_module_with_function(self):
        """Test creating a module with a simple function."""
        from mlir import ir
        from mlir.dialects import func, arith

        with MorphogenMLIRContext() as ctx:
            module = ctx.create_module("add_test")

            with ctx.ctx, ir.Location.unknown():
                # Create a simple function: () -> f32
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [f32])

                with ir.InsertionPoint(module.body):
                    func_op = func.FuncOp(name="test_add", type=func_type)
                    func_op.add_entry_block()

                    with ir.InsertionPoint(func_op.entry_block):
                        # Create constants
                        c1 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 1.0))
                        c2 = arith.ConstantOp(f32, ir.FloatAttr.get(f32, 2.0))

                        # Add them
                        result = arith.AddFOp(c1.result, c2.result)

                        # Return
                        func.ReturnOp([result.result])

            # Verify module contains expected IR
            module_str = str(module)
            assert "func.func" in module_str
            assert "test_add" in module_str
            assert "arith.constant" in module_str
            assert "arith.addf" in module_str

    def test_multiple_modules(self):
        """Test creating multiple independent modules."""
        with MorphogenMLIRContext() as ctx:
            module1 = ctx.create_module("module1")
            module2 = ctx.create_module("module2")

            assert module1 is not None
            assert module2 is not None
            assert module1 is not module2
