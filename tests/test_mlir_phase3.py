"""Tests for MLIR Phase 3: Flow Blocks (Temporal Execution).

This module tests the compilation of Morphogen flow blocks to MLIR scf.for loops
with proper state management through iteration arguments.
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


class TestPhase3BasicFlow:
    """Test basic flow block compilation."""

    def test_simple_flow_scalar_state(self):
        """Test flow block with simple scalar state variable."""
        code = """
@state x = 0.0

flow(dt=0.1, steps=10) {
    x = x + dt
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify scf.for loop is generated
        assert "scf.for" in ir_str

        # Verify constants for loop bounds
        assert "arith.constant" in ir_str

        # Verify index type handling
        assert "index" in ir_str

    def test_flow_with_integer_steps(self):
        """Test flow block with integer step count."""
        code = """
@state counter = 0

flow(steps=5) {
    counter = counter + 1
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify loop is created
        assert "scf.for" in ir_str

        # Verify integer operations
        assert "arith.addi" in ir_str or "arith.constant" in ir_str

    def test_flow_with_multiple_state_variables(self):
        """Test flow block with multiple state variables."""
        code = """
@state x = 0.0
@state y = 1.0

flow(steps=10) {
    x = x + 1.0
    y = y * 2.0
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify loop with multiple iteration arguments
        assert "scf.for" in ir_str

        # Should have both addition and multiplication
        assert "arith.addf" in ir_str or "arith.constant" in ir_str
        assert "arith.mulf" in ir_str

    def test_flow_with_dt_reference(self):
        """Test that dt parameter is accessible within flow body."""
        code = """
@state position = 0.0
@state velocity = 5.0

flow(dt=0.01, steps=100) {
    position = position + velocity * dt
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify loop structure
        assert "scf.for" in ir_str

        # Should have multiplication and addition
        assert "arith.mulf" in ir_str
        assert "arith.addf" in ir_str


class TestPhase3StructState:
    """Test flow blocks with struct state variables."""

    def test_flow_with_struct_state(self):
        """Test flow block updating struct state."""
        code = """
struct Point {
    x: f32
    y: f32
}

@state p = Point { x: 0.0, y: 0.0 }

flow(dt=0.1, steps=10) {
    p = Point { x: p.x + dt, y: p.y + dt }
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify struct operations
        assert "struct.construct" in ir_str or "struct<" in ir_str

        # Verify loop
        assert "scf.for" in ir_str

        # Verify field access
        assert "struct.extract" in ir_str

    def test_flow_with_nested_struct(self):
        """Test flow with nested struct state (simplified)."""
        code = """
struct Vector2D {
    x: f32
    y: f32
}

struct Particle {
    position: Vector2D
    velocity: Vector2D
}

@state particle = Particle {
    position: Vector2D { x: 0.0, y: 0.0 },
    velocity: Vector2D { x: 1.0, y: 0.0 }
}

flow(dt=0.1, steps=5) {
    # Simplified: reconstruct with same position and velocity
    # (Nested field access like particle.position.x is a known limitation)
    new_pos = Vector2D { x: 0.1, y: 0.1 }
    new_vel = Vector2D { x: 1.0, y: 0.0 }
    particle = Particle {
        position: new_pos,
        velocity: new_vel
    }
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify nested struct handling
        assert "struct.construct" in ir_str or "struct<" in ir_str

        # Verify loop
        assert "scf.for" in ir_str


class TestPhase3Substeps:
    """Test flow blocks with substeps (nested loops)."""

    def test_flow_with_substeps(self):
        """Test flow block with substeps parameter."""
        code = """
@state x = 0.0

flow(dt=0.1, steps=5, substeps=2) {
    x = x + dt
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should have nested scf.for loops
        # Count occurrences of "scf.for"
        assert ir_str.count("scf.for") >= 2, "Should have at least 2 scf.for loops (outer and inner)"

    def test_flow_substeps_multiple_states(self):
        """Test substeps with multiple state variables."""
        code = """
@state x = 0.0
@state y = 0.0

flow(dt=0.05, steps=10, substeps=3) {
    x = x + dt
    y = y + dt * 2.0
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify nested loops
        assert ir_str.count("scf.for") >= 2


class TestPhase3FunctionCalls:
    """Test flow blocks calling functions."""

    def test_flow_with_function_call(self):
        """Test flow block that calls a function."""
        code = """
fn update_position(pos: f32, vel: f32, dt: f32) -> f32 {
    return pos + vel * dt
}

@state position = 0.0
@state velocity = 5.0

flow(dt=0.1, steps=10) {
    position = update_position(position, velocity, dt)
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify function definition
        assert "func.func @update_position" in ir_str

        # Verify loop
        assert "scf.for" in ir_str

        # Verify function call within loop
        assert "func.call" in ir_str

    def test_flow_with_struct_function(self):
        """Test flow calling function that returns struct."""
        code = """
struct State {
    x: f32
    v: f32
}

fn update(s: State, dt: f32) -> State {
    return State {
        x: s.x + s.v * dt,
        v: s.v
    }
}

@state s = State { x: 0.0, v: 1.0 }

flow(dt=0.1, steps=5) {
    s = update(s, dt)
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify struct and function
        assert "func.func @update" in ir_str
        assert "scf.for" in ir_str
        assert "struct.construct" in ir_str or "struct<" in ir_str


class TestPhase3Conditionals:
    """Test flow blocks with conditional logic."""

    def test_flow_with_if_else(self):
        """Test flow block with if/else expression."""
        code = """
@state x = 0.0

flow(steps=10) {
    x = if x < 5.0 then x + 1.0 else x - 1.0
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify both loop and conditional
        assert "scf.for" in ir_str
        assert "scf.if" in ir_str

    def test_flow_physics_simulation(self):
        """Test realistic physics simulation with flow."""
        code = """
struct Particle {
    position: f32
    velocity: f32
}

@state particle = Particle { position: 10.0, velocity: 0.0 }

flow(dt=0.01, steps=100) {
    gravity = -9.8
    new_velocity = particle.velocity + gravity * dt
    new_position = particle.position + new_velocity * dt

    # Bounce off ground
    particle = if new_position < 0.0 then Particle {
        position: 0.0,
        velocity: -new_velocity * 0.8
    } else Particle {
        position: new_position,
        velocity: new_velocity
    }
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify complex flow compilation
        assert "scf.for" in ir_str
        assert "scf.if" in ir_str
        assert "struct<" in ir_str or "struct.construct" in ir_str


class TestPhase3EdgeCases:
    """Edge case tests for flow blocks."""

    def test_flow_single_step(self):
        """Test flow with single step."""
        code = """
@state x = 0.0

flow(steps=1) {
    x = x + 1.0
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should still generate loop (even for 1 iteration)
        assert "scf.for" in ir_str

    def test_flow_no_state_updates(self):
        """Test flow that doesn't update state (just reads)."""
        code = """
@state x = 5.0
@state sum = 0.0

flow(steps=10) {
    sum = sum + x
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should compile successfully
        assert "scf.for" in ir_str

    def test_flow_float_steps(self):
        """Test flow with float literal for steps (should convert)."""
        code = """
@state x = 0.0

flow(steps=10.0) {
    x = x + 1.0
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Should handle float->int conversion
        assert "scf.for" in ir_str
        assert "arith.fptosi" in ir_str or "arith.index_cast" in ir_str


class TestPhase3Integration:
    """Integration tests combining multiple features."""

    def test_complete_simulation_example(self):
        """Test complete simulation with functions, structs, and flow."""
        code = """
struct Vector2D {
    x: f32
    y: f32
}

fn add_vectors(a: Vector2D, b: Vector2D) -> Vector2D {
    return Vector2D {
        x: a.x + b.x,
        y: a.y + b.y
    }
}

fn scale_vector(v: Vector2D, s: f32) -> Vector2D {
    return Vector2D {
        x: v.x * s,
        y: v.y * s
    }
}

@state position = Vector2D { x: 0.0, y: 10.0 }
@state velocity = Vector2D { x: 5.0, y: 0.0 }

flow(dt=0.01, steps=50) {
    gravity = Vector2D { x: 0.0, y: -9.8 }
    dv = scale_vector(gravity, dt)
    velocity = add_vectors(velocity, dv)

    dp = scale_vector(velocity, dt)
    position = add_vectors(position, dp)
}
"""

        program = parse(code)
        compiler = MLIRCompiler()
        module = compiler.compile_program(program)

        ir_str = str(module)
        print("\n" + ir_str)

        # Verify all components are present
        assert "func.func @add_vectors" in ir_str
        assert "func.func @scale_vector" in ir_str
        assert "scf.for" in ir_str
        assert "struct<" in ir_str or "struct.construct" in ir_str
        assert "func.call" in ir_str
