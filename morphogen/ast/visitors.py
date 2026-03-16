"""Visitor pattern for AST traversal."""

from typing import Any
from .nodes import *


class ASTVisitor:
    """Base visitor for AST traversal."""

    def visit(self, node: ASTNode) -> Any:
        """Visit a node by dispatching to the appropriate visit method."""
        return node.accept(self)

    def visit_program(self, node: Program) -> Any:
        """Visit a program."""
        results = []
        for stmt in node.statements:
            results.append(self.visit(stmt))
        return results

    def visit_literal(self, node: Literal) -> Any:
        """Visit a literal."""
        return node.value

    def visit_identifier(self, node: Identifier) -> Any:
        """Visit an identifier."""
        return node.name

    def visit_binary_op(self, node: BinaryOp) -> Any:
        """Visit a binary operation."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return (node.operator, left, right)

    def visit_unary_op(self, node: UnaryOp) -> Any:
        """Visit a unary operation."""
        operand = self.visit(node.operand)
        return (node.operator, operand)

    def visit_call(self, node: Call) -> Any:
        """Visit a function call."""
        callee = self.visit(node.callee)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {k: self.visit(v) for k, v in node.kwargs.items()}
        return (callee, args, kwargs)

    def visit_field_access(self, node: FieldAccess) -> Any:
        """Visit a field access."""
        obj = self.visit(node.object)
        return (obj, node.field)

    def visit_tuple(self, node: Tuple) -> Any:
        """Visit a tuple expression."""
        elements = [self.visit(elem) for elem in node.elements]
        return tuple(elements)

    def visit_assignment(self, node: Assignment) -> Any:
        """Visit an assignment."""
        value = self.visit(node.value)
        return (node.target, value)

    def visit_expression_statement(self, node: ExpressionStatement) -> Any:
        """Visit an expression statement."""
        return self.visit(node.expression)

    def visit_step(self, node: Step) -> Any:
        """Visit a step block."""
        return [self.visit(stmt) for stmt in node.body]

    def visit_substep(self, node: Substep) -> Any:
        """Visit a substep block."""
        count = self.visit(node.count)
        body = [self.visit(stmt) for stmt in node.body]
        return (count, body)

    def visit_module(self, node: Module) -> Any:
        """Visit a module definition."""
        body = [self.visit(stmt) for stmt in node.body]
        return (node.name, node.params, body)

    def visit_compose(self, node: Compose) -> Any:
        """Visit a compose statement."""
        return [self.visit(mod) for mod in node.modules]

    def visit_type_annotation(self, node: TypeAnnotation) -> Any:
        """Visit a type annotation."""
        return (node.base_type, node.type_params, node.unit)

    def visit_decorator(self, node: Decorator) -> Any:
        """Visit a decorator."""
        args = [self.visit(arg) for arg in node.args]
        kwargs = {k: self.visit(v) for k, v in node.kwargs.items()}
        return (node.name, args, kwargs)


class TypeChecker(ASTVisitor):
    """Type checker visitor with dimensional analysis."""

    def __init__(self):
        self.symbol_table: dict[str, 'Type'] = {}
        self.errors: List[str] = []

    def _validate_unit_expression(self, unit_str: Optional[str]) -> bool:
        """Validate that a unit expression is parseable.

        Args:
            unit_str: Unit expression string

        Returns:
            True if valid or None, False if invalid
        """
        if unit_str is None:
            return True

        try:
            from morphogen.types.units import parse_unit
            parse_unit(unit_str)
            return True
        except (ValueError, ImportError) as e:
            self.errors.append(f"Invalid unit expression '{unit_str}': {e}")
            return False

    def _infer_additive_unit(self, left_unit, right_unit, op, parse_unit) -> Optional[str]:
        """Unit inference for + and - operators (units must be compatible)."""
        if left_unit is None and right_unit is None:
            return None
        if left_unit is None:
            return right_unit
        if right_unit is None:
            return left_unit
        left_parsed = parse_unit(left_unit)
        right_parsed = parse_unit(right_unit)
        if not left_parsed.is_compatible_with(right_parsed):
            self.errors.append(
                f"Unit mismatch in {op} operation: [{left_unit}] and [{right_unit}] are not compatible"
            )
            return None
        return left_unit

    def _infer_multiplicative_unit(self, left_unit, right_unit, parse_unit) -> Optional[str]:
        """Unit inference for * operator (units multiply)."""
        if left_unit is None and right_unit is None:
            return None
        if left_unit is None:
            return right_unit
        if right_unit is None:
            return left_unit
        result = parse_unit(left_unit) * parse_unit(right_unit)
        return result.symbol

    def _infer_division_unit(self, left_unit, right_unit, parse_unit) -> Optional[str]:
        """Unit inference for / operator (units divide)."""
        if left_unit is None and right_unit is None:
            return None
        result = parse_unit(left_unit or "1") / parse_unit(right_unit or "1")
        return result.symbol if not result.is_dimensionless() else None

    def _infer_power_unit(self, left_unit, right_unit, parse_unit) -> Optional[str]:
        """Unit inference for ^ operator (exponent must be dimensionless)."""
        if right_unit is not None:
            if not parse_unit(right_unit).is_dimensionless():
                self.errors.append(f"Exponent must be dimensionless, got [{right_unit}]")
                return None
        return left_unit  # None or the base unit (integer exponent assumed)

    def _infer_arithmetic_unit(self, left_type: 'Type', right_type: 'Type', op: str) -> Optional[str]:
        """Infer the resulting unit from an arithmetic operation."""
        try:
            from morphogen.types.units import parse_unit
            left_unit = getattr(left_type, 'unit', None)
            right_unit = getattr(right_type, 'unit', None)
            if op in ('+', '-'):
                return self._infer_additive_unit(left_unit, right_unit, op, parse_unit)
            if op == '*':
                return self._infer_multiplicative_unit(left_unit, right_unit, parse_unit)
            if op == '/':
                return self._infer_division_unit(left_unit, right_unit, parse_unit)
            if op == '^':
                return self._infer_power_unit(left_unit, right_unit, parse_unit)
            return None
        except (ImportError, ValueError, AttributeError):
            return None

    def visit_assignment(self, node: Assignment) -> Any:
        """Type-check an assignment with unit validation."""
        # Visit the value to infer its type
        value_type = self.visit(node.value)

        # If there's a type annotation, check compatibility
        if node.type_annotation:
            # Validate the unit expression first
            self._validate_unit_expression(node.type_annotation.unit)

            declared_type = self._resolve_type_annotation(node.type_annotation)
            if declared_type and value_type:
                if not value_type.is_compatible_with(declared_type):
                    # Provide helpful error message with unit information
                    value_unit = getattr(value_type, 'unit', None)
                    declared_unit = getattr(declared_type, 'unit', None)
                    if value_unit or declared_unit:
                        self.errors.append(
                            f"Type mismatch in assignment to '{node.target}': "
                            f"cannot assign {value_type} with unit [{value_unit or 'dimensionless'}] "
                            f"to {declared_type} with unit [{declared_unit or 'dimensionless'}]"
                        )
                    else:
                        self.errors.append(
                            f"Type mismatch: cannot assign {value_type} to {declared_type}"
                        )
                    return None

        # Store the symbol in the symbol table
        self.symbol_table[node.target] = value_type
        return value_type

    def visit_identifier(self, node: Identifier) -> Any:
        """Look up an identifier's type."""
        if node.name not in self.symbol_table:
            self.errors.append(f"Undefined identifier: {node.name}")
            return None
        return self.symbol_table[node.name]

    def visit_call(self, node: Call) -> Any:
        """Type-check a function call."""
        # This would need to look up the function signature
        # and check argument types
        # For now, we'll implement a basic version
        callee_type = self.visit(node.callee)

        # Visit all arguments
        arg_types = [self.visit(arg) for arg in node.args]
        kwarg_types = {k: self.visit(v) for k, v in node.kwargs.items()}

        # Return the function's return type (to be determined from signature)
        # This is simplified - real implementation would look up function signatures
        return None

    def _resolve_type_annotation(self, annotation: TypeAnnotation) -> 'Type':
        """Resolve a type annotation to a Type object."""
        from .types import (
            ScalarType, VectorType, FieldType, SignalType,
            VisualType, AgentType, BaseType
        )

        base_type = annotation.base_type

        # Map string type names to Type objects
        if base_type in ["f32", "f64", "f16", "i32", "i64", "u32", "u64"]:
            return ScalarType(BaseType[base_type.upper()], annotation.unit)
        elif base_type == "Vec2":
            return VectorType(
                BaseType.VEC2,
                ScalarType(BaseType.F32, annotation.unit),
                annotation.unit
            )
        elif base_type == "Vec3":
            return VectorType(
                BaseType.VEC3,
                ScalarType(BaseType.F32, annotation.unit),
                annotation.unit
            )
        elif base_type in ["Field2D", "Field3D"]:
            if annotation.type_params:
                elem_type = self._resolve_type_annotation(annotation.type_params[0])
            else:
                elem_type = ScalarType(BaseType.F32)
            return FieldType(
                BaseType[base_type.upper()],
                elem_type,
                annotation.unit
            )
        elif base_type == "Signal":
            if annotation.type_params:
                elem_type = self._resolve_type_annotation(annotation.type_params[0])
            else:
                elem_type = ScalarType(BaseType.F32)
            return SignalType(elem_type, annotation.unit)
        elif base_type == "Visual":
            return VisualType()
        else:
            self.errors.append(f"Unknown type: {base_type}")
            return None


class ASTPrinter(ASTVisitor):
    """Pretty-print AST for debugging."""

    def __init__(self):
        self.indent_level = 0

    def _indent(self) -> str:
        return "  " * self.indent_level

    def visit_literal(self, node: Literal) -> str:
        """Format a literal value as a string."""
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)

    def visit_identifier(self, node: Identifier) -> str:
        """Format an identifier as a string."""
        return node.name

    def visit_program(self, node: Program) -> str:
        lines = ["Program:"]
        self.indent_level += 1
        for stmt in node.statements:
            lines.append(self._indent() + self.visit(stmt))
        self.indent_level -= 1
        return "\n".join(lines)

    def visit_assignment(self, node: Assignment) -> str:
        decorators = ""
        if node.decorators:
            decorators = " ".join(f"@{d.name}" for d in node.decorators) + " "

        type_ann = ""
        if node.type_annotation:
            type_ann = f": {node.type_annotation.base_type}"

        value = self.visit(node.value)
        return f"{decorators}{node.target}{type_ann} = {value}"

    def visit_expression_statement(self, node: ExpressionStatement) -> str:
        return self.visit(node.expression)

    def visit_call(self, node: Call) -> str:
        callee = self.visit(node.callee)
        args = ", ".join(self.visit(arg) for arg in node.args)
        kwargs = ", ".join(f"{k}={self.visit(v)}" for k, v in node.kwargs.items())
        all_args = ", ".join(filter(None, [args, kwargs]))
        return f"{callee}({all_args})"

    def visit_field_access(self, node: FieldAccess) -> str:
        obj = self.visit(node.object)
        return f"{obj}.{node.field}"

    def visit_tuple(self, node: Tuple) -> str:
        elements = ", ".join(self.visit(elem) for elem in node.elements)
        return f"({elements})"
