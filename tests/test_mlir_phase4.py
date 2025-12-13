"""
MLIR Phase 4 Tests: Lambda Expressions

Tests lambda expression compilation to MLIR, including:
- Simple lambdas with parameters
- Lambdas with closures (captured variables)
- Lambdas in assignments
- Lambda calls
- Higher-order functions with lambdas
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


class TestPhase4BasicLambdas:
    """Test basic lambda expression compilation."""

    def test_simple_lambda_no_capture(self):
        """Test lambda with no captured variables."""
        source = """
fn main() {
    double = |x| x * 2.0
    result = double(5.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        # Should generate a lambda function
        assert "__lambda" in mlir_str
        # Should call the lambda
        assert "func.call" in mlir_str
        # Should have the computation
        assert "arith.mulf" in mlir_str

    def test_lambda_with_arithmetic(self):
        """Test lambda with arithmetic operations."""
        source = """
fn main() {
    add_then_double = |x, y| (x + y) * 2.0
    result = add_then_double(3.0, 4.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        # Should have both add and multiply
        assert "arith.addf" in mlir_str
        assert "arith.mulf" in mlir_str
        assert "func.call" in mlir_str
        assert "__lambda" in mlir_str

    def test_lambda_single_param(self):
        """Test lambda with single parameter."""
        source = """
fn main() {
    square = |x| x * x
    result = square(7.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        assert "__lambda" in mlir_str
        assert "arith.mulf" in mlir_str
        assert "func.call" in mlir_str

    def test_lambda_no_params(self):
        """Test lambda with no parameters (constant function)."""
        source = """
fn main() {
    get_pi = || 3.14159
    result = get_pi()
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        assert "__lambda" in mlir_str
        assert "3.14159" in mlir_str
        assert "func.call" in mlir_str


class TestPhase4LambdaWithCapture:
    """Test lambda expressions that capture variables."""

    def test_lambda_captures_scalar(self):
        """Test lambda capturing a scalar variable."""
        source = """
fn main() {
    multiplier = 3.0
    scale = |x| x * multiplier
    result = scale(4.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        # Should generate lambda function
        assert "__lambda" in mlir_str
        # Should have multiply operation
        assert "arith.mulf" in mlir_str
        # Should call lambda
        assert "func.call" in mlir_str

    def test_lambda_captures_multiple_vars(self):
        """Test lambda capturing multiple variables."""
        source = """
fn main() {
    a = 2.0
    b = 3.0
    compute = |x| a * x + b
    result = compute(5.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        assert "__lambda" in mlir_str
        assert "arith.mulf" in mlir_str
        assert "arith.addf" in mlir_str

    def test_lambda_in_flow_block(self):
        """Test lambda with closure in flow block."""
        source = """
@state position = 0.0
@state velocity = 1.0

flow(dt=0.1, steps=5) {
    update_pos = |p| p + velocity * dt
    position = update_pos(position)
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        # Should have flow loop
        assert "scf.for" in mlir_str
        # Should have lambda
        assert "__lambda" in mlir_str
        # Should have operations
        assert "arith.mulf" in mlir_str
        assert "arith.addf" in mlir_str


class TestPhase4LambdaConditionals:
    """Test lambdas with conditional logic."""

    def test_lambda_with_if_else(self):
        """Test lambda containing if/else expression."""
        source = """
fn main() {
    abs_diff = |a, b| if a > b then a - b else b - a
    result = abs_diff(3.0, 7.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        assert "__lambda" in mlir_str
        assert "scf.if" in mlir_str
        # The lambda compiles and can handle if/else
        assert "arith.cmpf" in mlir_str

    def test_lambda_conditional_with_capture(self):
        """Test lambda with if/else that captures a variable."""
        source = """
fn main() {
    threshold = 5.0
    clamp = |x| if x > threshold then threshold else x
    result = clamp(7.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        assert "__lambda" in mlir_str
        assert "scf.if" in mlir_str
        assert "arith.cmpf" in mlir_str


class TestPhase4HigherOrderFunctions:
    """Test higher-order functions with lambdas - simplified for Phase 4.1."""

    def test_lambda_in_variable(self):
        """Test assigning lambda to variable and calling it."""
        source = """
fn main() {
    double = |x| x * 2.0
    result = double(5.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        # Should have lambda function
        assert "__lambda" in mlir_str
        # Should have function call
        assert "func.call" in mlir_str
        assert "arith.mulf" in mlir_str


class TestPhase4EdgeCases:
    """Test edge cases and error conditions."""

    def test_nested_lambda_calls(self):
        """Test nested lambda invocations."""
        source = """
fn main() {
    add_one = |x| x + 1.0
    add_two = |x| add_one(add_one(x))
    result = add_two(5.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        assert "__lambda" in mlir_str
        assert "arith.addf" in mlir_str
        assert "func.call" in mlir_str

    def test_lambda_returns_lambda_result(self):
        """Test lambda that calls another lambda."""
        source = """
fn main() {
    f = |x| x * 2.0
    g = |x| f(x) + 1.0
    result = g(3.0)
    return result
}
"""
        program = parse(source)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)
        mlir_str = str(module)

        # Should have two lambdas
        lambda_count = mlir_str.count("__lambda")
        assert lambda_count >= 2
        assert "arith.mulf" in mlir_str
        assert "arith.addf" in mlir_str
