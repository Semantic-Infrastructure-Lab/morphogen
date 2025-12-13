"""Core AST node definitions for Creative Computation DSL."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
from enum import Enum


class NodeType(Enum):
    """Types of AST nodes."""
    # Expressions
    LITERAL = "literal"
    IDENTIFIER = "identifier"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    CALL = "call"
    FIELD_ACCESS = "field_access"
    TUPLE = "tuple"
    LAMBDA = "lambda"
    IF_ELSE = "if_else"
    BLOCK = "block"

    # Statements
    ASSIGNMENT = "assignment"
    EXPRESSION_STATEMENT = "expression_statement"
    STEP = "step"
    SUBSTEP = "substep"
    FLOW = "flow"
    FUNCTION = "function"
    STRUCT = "struct"
    RETURN = "return"
    MODULE = "module"
    COMPOSE = "compose"
    LINK = "link"
    USE = "use"
    OUTPUT = "output"

    # Type annotations
    TYPE_ANNOTATION = "type_annotation"

    # Decorators
    DECORATOR = "decorator"


@dataclass
class SourceLocation:
    """Location in source code."""
    line: int
    column: int
    file: Optional[str] = None


class ASTNode:
    """Base class for all AST nodes."""

    def __init__(self, location: Optional[SourceLocation] = None):
        self.location = location

    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Accept a visitor for AST traversal."""
        raise NotImplementedError


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class Expression(ASTNode):
    """Base class for expressions."""
    pass


@dataclass
class Literal(Expression):
    """Literal value (number, string, bool)."""
    value: Union[int, float, str, bool]
    node_type: NodeType = field(default=NodeType.LITERAL)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_literal(self)


@dataclass
class Identifier(Expression):
    """Variable or function identifier."""
    name: str
    node_type: NodeType = field(default=NodeType.IDENTIFIER)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_identifier(self)


@dataclass
class BinaryOp(Expression):
    """Binary operation (a + b, a * b, etc.)."""
    left: Expression
    operator: str  # +, -, *, /, etc.
    right: Expression
    node_type: NodeType = field(default=NodeType.BINARY_OP)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOp(Expression):
    """Unary operation (-x, !x, etc.)."""
    operator: str  # -, !, etc.
    operand: Expression
    node_type: NodeType = field(default=NodeType.UNARY_OP)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_unary_op(self)


@dataclass
class Call(Expression):
    """Function or method call."""
    callee: Expression  # Can be Identifier or FieldAccess
    args: List[Expression]
    kwargs: dict[str, Expression]
    node_type: NodeType = field(default=NodeType.CALL)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_call(self)


@dataclass
class FieldAccess(Expression):
    """Field or method access (object.field)."""
    object: Expression
    field: str
    node_type: NodeType = field(default=NodeType.FIELD_ACCESS)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_field_access(self)


@dataclass
class Tuple(Expression):
    """Tuple expression (e.g., (1, 2, 3) or (128, 128))."""
    elements: List[Expression]
    node_type: NodeType = field(default=NodeType.TUPLE)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_tuple(self)


@dataclass
class Lambda(Expression):
    """Lambda expression (|args| expr or |args| { stmts })."""
    params: List[str]  # Parameter names
    body: Expression  # Single expression body (can be Block for multi-statement)
    node_type: NodeType = field(default=NodeType.LAMBDA)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_lambda(self)


@dataclass
class Block(Expression):
    """Block expression ({ stmt1; stmt2; ... }) - evaluates to last expression."""
    statements: List['Statement']  # Multiple statements
    node_type: NodeType = field(default=NodeType.BLOCK)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_block(self)


@dataclass
class IfElse(Expression):
    """If/else expression (if cond then expr1 else expr2)."""
    condition: Expression
    then_expr: Expression
    else_expr: Expression
    node_type: NodeType = field(default=NodeType.IF_ELSE)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_if_else(self)


@dataclass
class StructLiteral(Expression):
    """Struct literal instantiation (Point { x: 3.0, y: 4.0 })."""
    struct_name: str  # Name of the struct type
    field_values: dict[str, Expression]  # Field name -> expression mapping
    node_type: NodeType = field(default=NodeType.LITERAL)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_struct_literal(self)


# ============================================================================
# Statements
# ============================================================================

@dataclass
class Statement(ASTNode):
    """Base class for statements."""
    pass


@dataclass
class Assignment(Statement):
    """Variable assignment."""
    target: str
    value: Expression
    type_annotation: Optional['TypeAnnotation'] = None
    decorators: List['Decorator'] = field(default_factory=list)
    is_const: bool = False
    node_type: NodeType = field(default=NodeType.ASSIGNMENT)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_assignment(self)


@dataclass
class ExpressionStatement(Statement):
    """Expression as a statement (e.g., function call without assignment)."""
    expression: Expression
    node_type: NodeType = field(default=NodeType.EXPRESSION_STATEMENT)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_expression_statement(self)


@dataclass
class Step(Statement):
    """Step block (single timestep)."""
    body: List[Statement]
    node_type: NodeType = field(default=NodeType.STEP)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_step(self)


@dataclass
class Substep(Statement):
    """Substep block (subdivided timestep)."""
    count: Expression
    body: List[Statement]
    node_type: NodeType = field(default=NodeType.SUBSTEP)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_substep(self)


@dataclass
class Module(Statement):
    """Module definition (reusable subsystem)."""
    name: str
    params: List[tuple[str, 'TypeAnnotation']]
    body: List[Statement]
    node_type: NodeType = field(default=NodeType.MODULE)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_module(self)


@dataclass
class Compose(Statement):
    """Parallel composition of modules."""
    modules: List[Expression]
    node_type: NodeType = field(default=NodeType.COMPOSE)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_compose(self)


@dataclass
class Link(Statement):
    """Dependency link metadata (no runtime cost)."""
    target: Expression  # Module or variable to link to
    metadata: Optional[dict] = None  # Dependency metadata
    node_type: NodeType = field(default=NodeType.LINK)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_link(self)


@dataclass
class Flow(Statement):
    """Flow block (temporal scope with explicit dt and steps)."""
    dt: Optional[Expression]  # Timestep (required)
    steps: Optional[Expression]  # Number of iterations (optional)
    substeps: Optional[Expression]  # Inner iterations per step (optional)
    body: List[Statement]
    node_type: NodeType = field(default=NodeType.FLOW)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_flow(self)


@dataclass
class Function(Statement):
    """Function definition (fn name(params) -> return_type { body })."""
    name: str
    params: List[tuple[str, Optional['TypeAnnotation']]]  # (name, type) pairs
    return_type: Optional['TypeAnnotation']
    body: List[Statement]
    node_type: NodeType = field(default=NodeType.FUNCTION)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_function(self)


@dataclass
class Struct(Statement):
    """Struct definition (struct Name { fields })."""
    name: str
    fields: List[tuple[str, 'TypeAnnotation']]  # (name, type) pairs
    node_type: NodeType = field(default=NodeType.STRUCT)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_struct(self)


@dataclass
class Return(Statement):
    """Return statement (return expr)."""
    value: Optional[Expression]
    node_type: NodeType = field(default=NodeType.RETURN)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_return(self)


@dataclass
class Use(Statement):
    """Use statement to import domain operators.

    Syntax:
        use field                    # Import field domain
        use field, agent, visual     # Multiple imports
        use field as f               # Aliased import (future)
    """
    domains: List[str]  # Domain names to import
    aliases: dict[str, str] = field(default_factory=dict)  # Optional aliases
    node_type: NodeType = field(default=NodeType.USE)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_use(self)


@dataclass
class Output(Statement):
    """Output statement for visual rendering.

    Syntax:
        output colorize(temp, palette="fire")   # Output visual to frame
        output field                            # Output raw field data
    """
    value: Expression  # Expression to output
    node_type: NodeType = field(default=NodeType.OUTPUT)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_output(self)


# ============================================================================
# Type Annotations
# ============================================================================

@dataclass
class TypeAnnotation(ASTNode):
    """Type annotation for variables and parameters."""
    base_type: str  # Field2D, Agents, Signal, etc.
    type_params: List['TypeAnnotation']  # Generic type parameters
    unit: Optional[str] = None  # Physical unit (m, m/s, kg, etc.)
    node_type: NodeType = field(default=NodeType.TYPE_ANNOTATION)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_type_annotation(self)


# ============================================================================
# Decorators
# ============================================================================

@dataclass
class Decorator(ASTNode):
    """Decorator (@double_buffer, @benchmark, etc.)."""
    name: str
    args: List[Expression]
    kwargs: dict[str, Expression]
    node_type: NodeType = field(default=NodeType.DECORATOR)

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_decorator(self)


# ============================================================================
# Program
# ============================================================================

@dataclass
class Program:
    """Top-level program containing all statements."""
    statements: List[Statement]

    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_program(self)
