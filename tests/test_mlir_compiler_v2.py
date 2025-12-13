"""Unit tests for morphogen.mlir.compiler_v2 (v0.7.0)

Tests for the real MLIR compiler using Python bindings.
These tests will be skipped if MLIR is not installed.
"""

import pytest
from morphogen.mlir.context import MorphogenMLIRContext, MLIR_AVAILABLE
from morphogen.mlir.compiler_v2 import MLIRCompilerV2, is_legacy_compiler
from morphogen.ast.nodes import Literal


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestMLIRCompilerV2:
    """Tests for MLIRCompilerV2 when MLIR is available."""

    def test_compiler_creation(self):
        """Test that we can create a compiler instance."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)
        assert compiler is not None
        assert compiler.context is ctx
        assert compiler.symbols == {}

    def test_compile_float_literal(self):
        """Test compiling a float literal."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")
            with ir.InsertionPoint(module.body):
                # Create a simple literal
                lit = Literal(3.14)
                # We need an insertion point for compile_literal
                # Create a dummy function to get an insertion point
                from mlir.dialects import func
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [f32])
                func_op = func.FuncOp(name="test", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    result = compiler.compile_literal(lit, None)
                    assert result is not None
                    # Verify it's an f32 constant
                    assert "f32" in str(result.type)

    def test_compile_int_literal(self):
        """Test compiling an integer literal."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")
            with ir.InsertionPoint(module.body):
                from mlir.dialects import func
                i32 = ir.I32Type.get()
                func_type = ir.FunctionType.get([], [i32])
                func_op = func.FuncOp(name="test", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    lit = Literal(42)
                    result = compiler.compile_literal(lit, None)
                    assert result is not None
                    # Verify it's an i32 constant
                    assert "i32" in str(result.type)

    def test_compile_bool_literal(self):
        """Test compiling a boolean literal."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")
            with ir.InsertionPoint(module.body):
                from mlir.dialects import func
                i1 = ir.IntegerType.get_signless(1)
                func_type = ir.FunctionType.get([], [i1])
                func_op = func.FuncOp(name="test", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    lit_true = Literal(True)
                    result_true = compiler.compile_literal(lit_true, None)
                    assert result_true is not None

                    lit_false = Literal(False)
                    result_false = compiler.compile_literal(lit_false, None)
                    assert result_false is not None

    def test_compile_binary_op_not_implemented(self):
        """Test that compile_binary_op raises NotImplementedError."""
        from morphogen.ast.nodes import BinaryOp, Literal

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        binop = BinaryOp(op="+", left=Literal(3.0), right=Literal(4.0))

        with pytest.raises(NotImplementedError, match="Phase 1 implementation in progress"):
            compiler.compile_binary_op(binop, None)

    def test_compile_program_not_implemented(self):
        """Test that compile_program raises NotImplementedError."""
        from morphogen.ast.nodes import Program

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        program = Program(statements=[])

        with pytest.raises(NotImplementedError, match="Phase 1 implementation in progress"):
            compiler.compile_program(program)


class TestCompilerAvailability:
    """Tests that work regardless of MLIR availability."""

    def test_is_legacy_compiler(self):
        """Test legacy compiler detection."""
        result = is_legacy_compiler()
        assert isinstance(result, bool)
        # If MLIR is available, we're NOT using legacy
        assert result == (not MLIR_AVAILABLE)

    @pytest.mark.skipif(MLIR_AVAILABLE, reason="Test for when MLIR is NOT installed")
    def test_compiler_creation_fails_without_mlir(self):
        """Test that creating compiler fails gracefully without MLIR."""
        # We can't create MorphogenMLIRContext without MLIR, so we can't test
        # the compiler initialization. Just verify the check works.
        assert is_legacy_compiler() is True


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestMLIRCompilerIntegration:
    """Integration tests for compiler operations."""

    def test_compile_simple_arithmetic_module(self):
        """Test compiling a simple arithmetic expression to a complete module."""
        from mlir import ir
        from mlir.dialects import func, arith

        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # Build a simple function: () -> f32 that returns 3.0 + 4.0
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("add_example")

            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [f32])
                func_op = func.FuncOp(name="add", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    # Use compiler to create literals
                    lit3 = Literal(3.0)
                    lit4 = Literal(4.0)

                    val3 = compiler.compile_literal(lit3, None)
                    val4 = compiler.compile_literal(lit4, None)

                    # Manually add them (until compile_binary_op is implemented)
                    result = arith.AddFOp(val3, val4)

                    func.ReturnOp([result.result])

        # Verify the module
        module_str = str(module)
        assert "func.func @add" in module_str
        assert "arith.constant 3.0" in module_str or "3.000000e+00" in module_str
        assert "arith.constant 4.0" in module_str or "4.000000e+00" in module_str
        assert "arith.addf" in module_str
        assert "func.return" in module_str
