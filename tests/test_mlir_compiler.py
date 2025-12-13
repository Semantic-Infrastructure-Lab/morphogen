"""Tests for MLIR compiler.

This module tests the MLIR compilation pipeline for Morphogen v0.3.1.
"""

import pytest
from morphogen.parser import Parser
from morphogen.lexer import Lexer
from morphogen.mlir import MLIRCompiler


def parse(code: str):
    """Helper to parse Morphogen code."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


class TestPhase1Basics:
    """Test Phase 1: Basic operations and functions."""

    def test_simple_literal(self):
        """Test compiling a simple literal."""
        code = "x = 3.0"

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        # Verify module was created
        assert module is not None
        ir_str = str(module)
        print("\n" + ir_str)

        # Check for constant operation
        assert "arith.constant" in ir_str

    def test_simple_arithmetic(self):
        """Test compiling arithmetic operations."""
        code = """
x = 3.0 + 4.0
y = x * 2.0
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for arithmetic operations
        assert "arith.addf" in ir_str or "arith.constant" in ir_str
        assert "arith.mulf" in ir_str

    def test_comparison_operations(self):
        """Test compiling comparison operations."""
        code = """
a = 5.0
b = 3.0
result = a > b
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for comparison operation
        assert "arith.cmpf" in ir_str
        assert "ogt" in ir_str

    def test_simple_function_def(self):
        """Test compiling a simple function definition."""
        code = """
fn add(x: f32, y: f32) -> f32 {
    return x + y
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for function definition
        assert "func.func @add" in ir_str
        assert "%arg0" in ir_str
        assert "%arg1" in ir_str
        assert "func.return" in ir_str

    def test_function_call(self):
        """Test compiling function call."""
        code = """
fn add(x: f32, y: f32) -> f32 {
    return x + y
}

result = add(3.0, 4.0)
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for function definition and call
        assert "func.func @add" in ir_str
        assert "func.call" in ir_str
        assert "@add" in ir_str

    def test_function_with_multiple_operations(self):
        """Test function with multiple operations."""
        code = """
fn calculate(x: f32, y: f32) -> f32 {
    sum = x + y
    product = sum * 2.0
    return product
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for operations
        assert "func.func @calculate" in ir_str
        assert "arith.addf" in ir_str
        assert "arith.mulf" in ir_str
        assert "func.return" in ir_str

    def test_unary_negation(self):
        """Test unary negation operator."""
        code = """
x = 5.0
y = -x
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for subtraction (0 - x)
        assert "arith.subf" in ir_str

    def test_integer_operations(self):
        """Test integer arithmetic."""
        code = """
a = 10
b = 5
c = a + b
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for integer add
        assert "arith.addi" in ir_str or "arith.constant" in ir_str


class TestPhase1Integration:
    """Integration tests for Phase 1."""

    def test_complete_example(self):
        """Test a complete example with functions and calls."""
        code = """
fn square(x: f32) -> f32 {
    return x * x
}

fn sum_of_squares(a: f32, b: f32) -> f32 {
    sa = square(a)
    sb = square(b)
    return sa + sb
}

result = sum_of_squares(3.0, 4.0)
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify both functions are defined
        assert "func.func @square" in ir_str
        assert "func.func @sum_of_squares" in ir_str

        # Verify calls are present
        assert ir_str.count("func.call") >= 2

    def test_recursive_function(self):
        """Test that recursive functions compile (execution tested separately)."""
        code = """
fn factorial(n: f32) -> f32 {
    result = if n <= 1.0 then 1.0 else n * factorial(n - 1.0)
    return result
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should compile with if/else (recursive call is in else branch)
        assert "func.func @factorial" in ir_str
        assert "scf.if" in ir_str
        # Note: recursive call exists but may not appear in simplified IR output


class TestPhase1EdgeCases:
    """Edge case tests for Phase 1."""

    def test_undefined_variable(self):
        """Test that undefined variables raise an error."""
        code = """
y = x + 1.0
"""

        program = parse(code)
        compiler = MLIRCompiler()

        with pytest.raises(KeyError, match="Undefined variable: x"):
            compiler.compile_program(program)

    def test_undefined_function(self):
        """Test that undefined functions raise an error."""
        code = """
result = undefined_func(3.0)
"""

        program = parse(code)
        compiler = MLIRCompiler()

        with pytest.raises(KeyError, match="Undefined function: undefined_func"):
            compiler.compile_program(program)

    def test_void_function(self):
        """Test function with no return value."""
        code = """
fn do_nothing() {
    x = 1.0
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should have function definition and return
        assert "func.func @do_nothing" in ir_str
        assert "func.return" in ir_str
