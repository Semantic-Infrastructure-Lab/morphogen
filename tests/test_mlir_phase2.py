"""Tests for MLIR compiler Phase 2: Control flow and structs.

This module tests if/else expressions and struct support.
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


class TestPhase2IfElse:
    """Test Phase 2.1: If/else expressions."""

    def test_simple_if_else(self):
        """Test simple if/else expression."""
        code = """
fn classify(x: f32) -> f32 {
    result = if x > 0.0 then 1.0 else 0.0
    return result
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for if operation
        assert "scf.if" in ir_str
        assert "arith.cmpf" in ir_str

    def test_nested_if_else(self):
        """Test nested if/else expressions."""
        code = """
fn classify(x: f32) -> f32 {
    result = if x > 10.0 then 3.0 else if x > 5.0 then 2.0 else 1.0
    return result
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should have at least one if operation (nested structure is implicit)
        assert "scf.if" in ir_str


class TestPhase2Structs:
    """Test Phase 2.2-2.4: Struct support."""

    def test_struct_definition(self):
        """Test struct type definition."""
        code = """
struct Point {
    x: f32
    y: f32
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for struct definition
        assert "struct Point" in ir_str or "struct<f32, f32>" in ir_str

    def test_struct_literal(self):
        """Test struct literal instantiation."""
        code = """
struct Point {
    x: f32
    y: f32
}

p = Point { x: 3.0, y: 4.0 }
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for struct construction
        assert "struct.construct" in ir_str

    def test_field_access(self):
        """Test field access."""
        code = """
struct Point {
    x: f32
    y: f32
}

p = Point { x: 3.0, y: 4.0 }
x_val = p.x
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for field extraction
        assert "struct.extract" in ir_str

    def test_struct_in_function(self):
        """Test struct operations in functions."""
        code = """
struct Point {
    x: f32
    y: f32
}

fn create_point(x: f32, y: f32) -> Point {
    return Point { x: x, y: y }
}

fn get_x(p: Point) -> f32 {
    return p.x
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for struct operations
        assert "struct.construct" in ir_str
        assert "struct.extract" in ir_str


class TestPhase2Integration:
    """Integration tests for Phase 2."""

    def test_if_else_with_structs(self):
        """Test if/else with struct operations."""
        code = """
struct Point {
    x: f32
    y: f32
}

fn classify_point(p: Point) -> f32 {
    x_positive = if p.x > 0.0 then 1.0 else 0.0
    return x_positive
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Check for both struct and if operations
        assert "struct.extract" in ir_str
        assert "scf.if" in ir_str

    def test_nested_structs(self):
        """Test nested struct types."""
        code = """
struct Vector2D {
    x: f32
    y: f32
}

struct Particle {
    position: Vector2D
    velocity: Vector2D
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Both structs should be defined
        assert "Vector2D" in ir_str
        assert "Particle" in ir_str
