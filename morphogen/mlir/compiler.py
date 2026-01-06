"""MLIR Compiler for Morphogen v0.3.1

This module implements the core MLIR compilation pipeline, transforming
Morphogen AST nodes into MLIR IR for native code generation.
"""

from typing import Dict, List, Optional, Any, Union
from .ir_builder import (
    IRBuilder, IRValue, IRType, IRFunction, IRBlock, IROperation, IRModule
)

from ..ast.nodes import (
    Program, Statement, Expression,
    Function, Return, Assignment, Flow, Struct, ExpressionStatement,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess,
    IfElse, Lambda, StructLiteral, Tuple,
    TypeAnnotation
)


class MLIRCompiler:
    """Compiles Morphogen AST to MLIR IR.

    This compiler transforms Morphogen programs into MLIR's multi-level intermediate
    representation, enabling progressive lowering to native machine code.

    Compilation Strategy:
    - SSA (Single Static Assignment) for immutable values
    - memref for state variables in flow blocks
    - Progressive lowering: high-level dialects → LLVM dialect → machine code

    Supported Phases:
    - Phase 1: Basic operations, functions, arithmetic ✅
    - Phase 2: Control flow (if/else), structs
    - Phase 3: Temporal execution (flow blocks)
    - Phase 4: Advanced features (lambdas, recursion)
    """

    # Arithmetic operator mappings: operator → (float_opcode, int_opcode)
    _ARITH_OPS = {
        '+': ("arith.addf", "arith.addi"),
        '-': ("arith.subf", "arith.subi"),
        '*': ("arith.mulf", "arith.muli"),
        '/': ("arith.divf", "arith.divsi"),
        '%': ("arith.remf", "arith.remsi"),
    }

    # Comparison operator mappings: operator → (float_predicate, int_predicate)
    _CMP_PREDICATES = {
        '<': ("olt", "slt"),
        '>': ("ogt", "sgt"),
        '==': ("oeq", "eq"),
        '!=': ("one", "ne"),
        '<=': ("ole", "sle"),
        '>=': ("oge", "sge"),
    }

    def __init__(self):
        """Initialize MLIR compiler."""
        self.builder = IRBuilder()

        # Symbol tables
        self.symbols: Dict[str, IRValue] = {}  # Variable name → SSA value
        self.functions: Dict[str, IRFunction] = {}  # Function name → IR function
        self.struct_types: Dict[str, Dict[str, Any]] = {}  # Struct metadata

        # State tracking
        self.current_function: Optional[str] = None
        self.state_vars: Dict[str, Dict[str, Any]] = {}  # State variable metadata

        # Lambda counter for unique naming
        self.lambda_counter = 0

    def compile_program(self, program: Program) -> IRModule:
        """Compile a Morphogen program to MLIR module.

        Args:
            program: Morphogen Program AST node

        Returns:
            MLIR Module ready for lowering and execution

        Raises:
            ValueError: If compilation fails

        Note:
            Top-level code (assignments, expressions) is wrapped in a main() function.
            Function definitions are compiled as separate functions.
        """
        # Separate function/struct definitions from other statements
        function_defs = []
        struct_defs = []
        top_level_stmts = []

        for stmt in program.statements:
            if isinstance(stmt, Function):
                function_defs.append(stmt)
            elif isinstance(stmt, Struct):
                struct_defs.append(stmt)
            else:
                top_level_stmts.append(stmt)

        # Compile struct definitions first (needed for type system)
        for struct_def in struct_defs:
            self.compile_struct_def(struct_def)

        # Compile function definitions
        for func_def in function_defs:
            self.compile_function_def(func_def)

        # If there are top-level statements, wrap them in a main function
        if top_level_stmts:
            # Create implicit main function
            ir_func = self.builder.create_function(
                name="main",
                args=[],
                return_types=[]
            )
            self.functions["main"] = ir_func
            self.current_function = "main"

            # Create entry block
            self.builder.create_block(label="entry")

            # Compile top-level statements
            for stmt in top_level_stmts:
                self.compile_statement(stmt)

            # Add void return
            self.builder.add_operation(
                "func.return",
                operands=[],
                result_types=[]
            )

            self.current_function = None

        # Get and verify module
        module = self.builder.get_module()
        module.verify()

        return module

    def compile_statement(self, stmt: Statement) -> Optional[IRValue]:
        """Compile a statement.

        Args:
            stmt: Statement AST node

        Returns:
            MLIR operation (if any)
        """
        if isinstance(stmt, Function):
            return self.compile_function_def(stmt)
        elif isinstance(stmt, Return):
            return self.compile_return(stmt)
        elif isinstance(stmt, Assignment):
            return self.compile_assignment(stmt)
        elif isinstance(stmt, Flow):
            return self.compile_flow_block(stmt)
        elif isinstance(stmt, Struct):
            return self.compile_struct_def(stmt)
        elif isinstance(stmt, ExpressionStatement):
            return self.compile_expression(stmt.expression)
        else:
            raise NotImplementedError(f"Statement type not yet implemented: {type(stmt).__name__}")

    def compile_expression(self, expr: Expression) -> IRValue:
        """Compile an expression to an MLIR value.

        Args:
            expr: Expression AST node

        Returns:
            MLIR SSA value
        """
        if isinstance(expr, Literal):
            return self.compile_literal(expr)
        elif isinstance(expr, Identifier):
            return self.compile_identifier(expr)
        elif isinstance(expr, BinaryOp):
            return self.compile_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self.compile_unary_op(expr)
        elif isinstance(expr, Call):
            return self.compile_call(expr)
        elif isinstance(expr, FieldAccess):
            return self.compile_field_access(expr)
        elif isinstance(expr, IfElse):
            return self.compile_if_else(expr)
        elif isinstance(expr, Lambda):
            return self.compile_lambda(expr)
        elif isinstance(expr, StructLiteral):
            return self.compile_struct_literal(expr)
        elif isinstance(expr, Tuple):
            return self.compile_tuple(expr)
        else:
            raise NotImplementedError(f"Expression type not yet implemented: {type(expr).__name__}")

    # =========================================================================
    # Type System
    # =========================================================================

    def lower_type(self, morphogen_type: Optional[TypeAnnotation]) -> IRType:
        """Convert Morphogen type to MLIR type.

        Args:
            morphogen_type: Morphogen type annotation (may be None)

        Returns:
            MLIR type

        Examples:
            f32 → F32Type
            f32[m] → F32Type (units stripped)
            i32 → IntegerType(32)
            bool → IntegerType(1)
        """
        if morphogen_type is None:
            # Default to f32 for untyped expressions
            return IRType.F32

        # Extract base type (strip physical units)
        base_type = morphogen_type.base_type.lower()

        # Map Morphogen base types to MLIR types
        if base_type == 'f32':
            return IRType.F32
        elif base_type == 'f64':
            return IRType.F64
        elif base_type == 'i32':
            return IRType.I32
        elif base_type == 'i64':
            return IRType.I64
        elif base_type == 'bool':
            return IRType.I1
        elif base_type in self.struct_types:
            # Struct type
            return self.struct_types[base_type]['mlir_type']
        else:
            # Unknown type, default to f32
            return IRType.F32

    def infer_type(self, expr: Expression) -> IRType:
        """Infer MLIR type from expression.

        Args:
            expr: Expression to infer type from

        Returns:
            Inferred MLIR type
        """
        if isinstance(expr, Literal):
            if isinstance(expr.value, float):
                return IRType.F32
            elif isinstance(expr.value, int):
                return IRType.I32
            elif isinstance(expr.value, bool):
                return IRType.I1
        elif isinstance(expr, Identifier):
            # Look up in symbol table and get type
            if expr.name in self.symbols:
                return self.symbols[expr.name].type
        elif isinstance(expr, BinaryOp):
            # Binary ops preserve type of operands (simplified)
            return self.infer_type(expr.left)

        # Default to f32
        return IRType.F32

    # =========================================================================
    # Phase 1.3: Literals and Identifiers
    # =========================================================================

    def compile_literal(self, literal: Literal) -> IRValue:
        """Compile a literal constant.

        Args:
            literal: Literal AST node

        Returns:
            MLIR constant value

        Examples:
            3.0 → arith.constant 3.0 : f32
            42 → arith.constant 42 : i32
            true → arith.constant 1 : i1
        """
        if isinstance(literal.value, float):
            # Float literal
            results = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.F32],
                attributes={"value": float(literal.value)}
            )
            return results[0]

        elif isinstance(literal.value, int):
            # Integer literal
            results = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.I32],
                attributes={"value": int(literal.value)}
            )
            return results[0]

        elif isinstance(literal.value, bool):
            # Boolean literal
            results = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.I1],
                attributes={"value": 1 if literal.value else 0}
            )
            return results[0]

        else:
            raise ValueError(f"Unsupported literal type: {type(literal.value)}")

    def compile_identifier(self, identifier: Identifier) -> IRValue:
        """Compile an identifier lookup.

        Args:
            identifier: Identifier AST node

        Returns:
            MLIR value from symbol table

        Raises:
            KeyError: If identifier is undefined
        """
        if identifier.name not in self.symbols:
            raise KeyError(f"Undefined variable: {identifier.name}")

        return self.symbols[identifier.name]

    # =========================================================================
    # Phase 1.4: Binary and Unary Operations
    # =========================================================================

    def _compile_comparison(
        self, left: IRValue, right: IRValue, operator: str, is_float: bool
    ) -> IRValue:
        """Compile a comparison operation.

        Args:
            left: Left operand SSA value
            right: Right operand SSA value
            operator: Comparison operator (<, >, ==, !=, <=, >=)
            is_float: True if operands are floating point

        Returns:
            Boolean (i1) result SSA value
        """
        float_pred, int_pred = self._CMP_PREDICATES[operator]
        opcode = "arith.cmpf" if is_float else "arith.cmpi"
        predicate = float_pred if is_float else int_pred

        results = self.builder.add_operation(
            opcode,
            operands=[left, right],
            result_types=[IRType.I1],
            attributes={"predicate": predicate}
        )
        return results[0]

    def compile_binary_op(self, binop: BinaryOp) -> IRValue:
        """Compile binary operation.

        Args:
            binop: BinaryOp AST node

        Returns:
            Result SSA value

        Examples:
            a + b → arith.addf %a, %b : f32
            x * y → arith.mulf %x, %y : f32
            i < j → arith.cmpf olt, %i, %j : f32
        """
        left = self.compile_expression(binop.left)
        right = self.compile_expression(binop.right)
        is_float = left.type in [IRType.F32, IRType.F64]
        op = binop.operator

        # Handle comparison operators
        if op in self._CMP_PREDICATES:
            return self._compile_comparison(left, right, op, is_float)

        # Handle arithmetic operators
        if op in self._ARITH_OPS:
            float_op, int_op = self._ARITH_OPS[op]
            opcode = float_op if is_float else int_op
            results = self.builder.add_operation(
                opcode,
                operands=[left, right],
                result_types=[left.type]
            )
            return results[0]

        raise ValueError(f"Unsupported binary operator: {op}")

    def compile_unary_op(self, unop: UnaryOp) -> IRValue:
        """Compile unary operation.

        Args:
            unop: UnaryOp AST node

        Returns:
            Result SSA value

        Examples:
            -x → arith.negf %x : f32
            !x → arith.xori %x, %true : i1
        """
        # Compile operand
        operand = self.compile_expression(unop.operand)

        if unop.operator == '-':
            # Negation
            is_float = operand.type in [IRType.F32, IRType.F64]
            if is_float:
                # For floats: 0.0 - x
                zero = self.builder.add_operation(
                    "arith.constant",
                    operands=[],
                    result_types=[operand.type],
                    attributes={"value": 0.0}
                )[0]
                results = self.builder.add_operation(
                    "arith.subf",
                    operands=[zero, operand],
                    result_types=[operand.type]
                )
            else:
                # For ints: 0 - x
                zero = self.builder.add_operation(
                    "arith.constant",
                    operands=[],
                    result_types=[operand.type],
                    attributes={"value": 0}
                )[0]
                results = self.builder.add_operation(
                    "arith.subi",
                    operands=[zero, operand],
                    result_types=[operand.type]
                )
            return results[0]

        elif unop.operator == '!':
            # Logical NOT: xor with 1
            one = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.I1],
                attributes={"value": 1}
            )[0]
            results = self.builder.add_operation(
                "arith.xori",
                operands=[operand, one],
                result_types=[IRType.I1]
            )
            return results[0]

        else:
            raise ValueError(f"Unsupported unary operator: {unop.operator}")

    # =========================================================================
    # Phase 1.7: Assignments (SSA)
    # =========================================================================

    def compile_assignment(self, assign: Assignment) -> None:
        """Compile assignment statement (SSA style).

        Args:
            assign: Assignment AST node

        Note:
            In SSA, each variable gets a new value. The symbol table
            tracks the latest SSA value for each variable name.

        Example:
            x = 3.0 + 4.0
            → %0 = arith.constant 3.0 : f32
            → %1 = arith.constant 4.0 : f32
            → %x = arith.addf %0, %1 : f32
        """
        # Compile RHS expression
        value = self.compile_expression(assign.value)

        # Update symbol table (SSA: create new binding)
        self.symbols[assign.target] = value

    # =========================================================================
    # Phase 1.5: Function Definitions
    # =========================================================================

    def compile_function_def(self, func_node: Function) -> None:
        """Compile function definition.

        Args:
            func_node: Function AST node

        Example:
            fn add(x: f32, y: f32) -> f32 {
                return x + y
            }

            Becomes:
            func.func @add(%arg0: f32, %arg1: f32) -> f32 {
              %0 = arith.addf %arg0, %arg1 : f32
              func.return %0 : f32
            }
        """
        # Build function signature
        arg_types = []
        arg_values = []
        for i, (param_name, param_type) in enumerate(func_node.params):
            ir_type = self.lower_type(param_type)
            arg_types.append(ir_type)

            # Handle struct types - use string representation
            if param_type and param_type.base_type in self.struct_types:
                # For struct parameters, use the struct type string
                type_repr = self.struct_types[param_type.base_type]['mlir_type']
                arg_value = IRValue(name=f"%arg{i}", type=type_repr)
            else:
                arg_value = IRValue(name=f"%arg{i}", type=ir_type)
            arg_values.append(arg_value)

        # Return types
        return_types = []
        if func_node.return_type:
            ret_type = self.lower_type(func_node.return_type)
            # Handle struct return types
            if func_node.return_type.base_type in self.struct_types:
                return_types.append(self.struct_types[func_node.return_type.base_type]['mlir_type'])
            else:
                return_types.append(ret_type)

        # Create function
        ir_func = self.builder.create_function(
            name=func_node.name,
            args=arg_values,
            return_types=return_types
        )

        # Store function
        self.functions[func_node.name] = ir_func
        self.current_function = func_node.name

        # Create entry block
        self.builder.create_block(label="entry")

        # Save current symbol table
        saved_symbols = self.symbols.copy()

        # Map parameters to block arguments
        for (param_name, _), arg_value in zip(func_node.params, arg_values):
            self.symbols[param_name] = arg_value

        # Compile function body
        for stmt in func_node.body:
            if isinstance(stmt, Return):
                self.compile_return(stmt)
                break  # Return terminates the block
            else:
                self.compile_statement(stmt)

        # If no explicit return, add implicit void return
        if not func_node.body or not isinstance(func_node.body[-1], Return):
            self.builder.add_operation(
                "func.return",
                operands=[],
                result_types=[]
            )

        # Restore symbol table
        self.symbols = saved_symbols
        self.current_function = None

    def compile_return(self, return_node: Return) -> None:
        """Compile return statement.

        Args:
            return_node: Return AST node

        Example:
            return x + y
            → %0 = arith.addf %x, %y : f32
            → func.return %0 : f32
        """
        if return_node.value is not None:
            # Evaluate return value
            value = self.compile_expression(return_node.value)
            # Return with value
            self.builder.add_operation(
                "func.return",
                operands=[value],
                result_types=[]
            )
        else:
            # Void return
            self.builder.add_operation(
                "func.return",
                operands=[],
                result_types=[]
            )

    # =========================================================================
    # Phase 1.6: Function Calls
    # =========================================================================

    def compile_call(self, call: Call) -> IRValue:
        """Compile function call.

        Args:
            call: Call AST node

        Returns:
            Result value (or None for void functions)

        Example:
            result = add(3.0, 4.0)
            → %0 = arith.constant 3.0 : f32
            → %1 = arith.constant 4.0 : f32
            → %result = func.call @add(%0, %1) : (f32, f32) -> f32

        Phase 4: Also handles lambda calls:
            double = |x| x * 2.0
            result = double(5.0)
            → Calls the generated __lambda_0 function with captured vars
        """
        # Get function name or lambda
        if isinstance(call.callee, Identifier):
            callee_name = call.callee.name

            # Check if this is a lambda variable (Phase 4)
            if callee_name in self.symbols:
                callee_value = self.symbols[callee_name]
                if hasattr(callee_value, '_lambda_meta'):
                    # This is a lambda call
                    lambda_meta = callee_value._lambda_meta
                    lambda_func_name = lambda_meta['function']
                    captured_values = lambda_meta['captured_values']

                    # Compile explicit arguments
                    args = [self.compile_expression(arg) for arg in call.args]

                    # Add captured values as additional arguments
                    all_args = args + captured_values

                    # Get lambda function
                    ir_func = self.functions[lambda_func_name]

                    # Call the lambda function
                    if ir_func.return_types:
                        results = self.builder.add_operation(
                            "func.call",
                            operands=all_args,
                            result_types=ir_func.return_types,
                            attributes={"callee": f"@{lambda_func_name}"}
                        )
                        return results[0]
                    else:
                        self.builder.add_operation(
                            "func.call",
                            operands=all_args,
                            result_types=[],
                            attributes={"callee": f"@{lambda_func_name}"}
                        )
                        return None

            # Regular function call
            func_name = callee_name
        else:
            raise NotImplementedError("Only simple function calls supported in Phase 1")

        # Check if function exists
        if func_name not in self.functions:
            raise KeyError(f"Undefined function: {func_name}")

        # Compile arguments
        args = [self.compile_expression(arg) for arg in call.args]

        # Get function info
        ir_func = self.functions[func_name]

        # Create call operation
        if ir_func.return_types:
            # Function with return value
            results = self.builder.add_operation(
                "func.call",
                operands=args,
                result_types=ir_func.return_types,
                attributes={"callee": f"@{func_name}"}
            )
            return results[0]
        else:
            # Void function
            self.builder.add_operation(
                "func.call",
                operands=args,
                result_types=[],
                attributes={"callee": f"@{func_name}"}
            )
            # Return a dummy value (void functions don't have results)
            return None

    # =========================================================================
    # Phase 2.1: If/Else Expressions
    # =========================================================================

    def compile_if_else(self, if_else: IfElse) -> IRValue:
        """Compile if/else expression using scf.if.

        Args:
            if_else: IfElse AST node

        Returns:
            Result value from then or else branch

        Example:
            result = if x > 0.0 then x * 2.0 else x / 2.0

            Becomes:
            %cond = arith.cmpf ogt, %x, %c0 : f32
            %result = scf.if %cond -> (f32) {
              %then_val = arith.mulf %x, %c2 : f32
              scf.yield %then_val : f32
            } else {
              %else_val = arith.divf %x, %c2 : f32
              scf.yield %else_val : f32
            }
        """
        # Compile condition
        condition = self.compile_expression(if_else.condition)

        # Infer result type from then branch
        result_type = self.infer_type(if_else.then_expr)

        # Create scf.if operation
        # Since we don't have real MLIR SCF, simulate it with a pseudo-operation
        # In real MLIR, this would be: scf.if %cond -> (result_type) { ... } else { ... }

        # Compile both branches
        # Save current builder state
        saved_current_block = self.builder.current_block

        # Create then block (simulate)
        then_block = IRBlock(label="then")
        self.builder.current_block = then_block
        then_value = self.compile_expression(if_else.then_expr)

        # Create else block (simulate)
        else_block = IRBlock(label="else")
        self.builder.current_block = else_block
        else_value = self.compile_expression(if_else.else_expr)

        # Restore builder state
        self.builder.current_block = saved_current_block

        # Create scf.if operation with both branches
        # This is a simplified representation
        results = self.builder.add_operation(
            "scf.if",
            operands=[condition],
            result_types=[result_type],
            attributes={
                "then_value": then_value.name,
                "else_value": else_value.name
            }
        )

        return results[0]

    # =========================================================================
    # Phase 2.2: Struct Type Definitions
    # =========================================================================

    def compile_struct_def(self, struct: Struct) -> None:
        """Compile struct definition.

        Args:
            struct: Struct AST node

        Note:
            Structs are represented as LLVM struct types with metadata
            tracking field names and types.

        Example:
            struct Point {
                x: f32
                y: f32
            }

            Creates metadata:
            {
                'mlir_type': 'struct<f32, f32>',
                'fields': {'x': 0, 'y': 1},
                'field_types': [f32, f32]
            }
        """
        # Get field types
        field_types = []
        field_names = {}

        for i, (field_name, field_type) in enumerate(struct.fields):
            ir_type = self.lower_type(field_type)
            field_types.append(ir_type)
            field_names[field_name] = i

        # Create struct type representation
        # In simplified IR, represent as tuple type
        type_repr = f"struct<{', '.join(str(t.value) for t in field_types)}>"

        # Store struct metadata
        self.struct_types[struct.name] = {
            'mlir_type': type_repr,
            'fields': field_names,
            'field_types': field_types
        }

        # Add struct definition to module (for documentation)
        struct_def = f"// struct {struct.name} = {type_repr}"
        self.builder.module.structs.append(struct_def)

    # =========================================================================
    # Phase 2.3: Struct Literals
    # =========================================================================

    def compile_struct_literal(self, struct_lit: StructLiteral) -> IRValue:
        """Compile struct literal instantiation.

        Args:
            struct_lit: StructLiteral AST node

        Returns:
            Struct value (aggregate)

        Example:
            p = Point { x: 3.0, y: 4.0 }

            Becomes:
            %x_val = arith.constant 3.0 : f32
            %y_val = arith.constant 4.0 : f32
            %p = struct.construct { fields: [%x_val, %y_val] } : struct<f32, f32>
        """
        # Look up struct metadata
        if struct_lit.struct_name not in self.struct_types:
            raise KeyError(f"Undefined struct type: {struct_lit.struct_name}")

        struct_meta = self.struct_types[struct_lit.struct_name]

        # Compile field values in order
        field_values = []
        for field_name, field_index in sorted(struct_meta['fields'].items(),
                                              key=lambda x: x[1]):
            if field_name not in struct_lit.field_values:
                raise ValueError(f"Missing field '{field_name}' in struct literal")

            field_expr = struct_lit.field_values[field_name]
            field_value = self.compile_expression(field_expr)
            field_values.append(field_value)

        # Create struct construction operation
        results = self.builder.add_operation(
            "struct.construct",
            operands=field_values,
            result_types=[struct_meta['mlir_type']],
            attributes={"struct_name": struct_lit.struct_name}
        )

        return results[0]

    # =========================================================================
    # Phase 2.4: Field Access
    # =========================================================================

    def compile_field_access(self, field_access: FieldAccess) -> IRValue:
        """Compile struct field access.

        Args:
            field_access: FieldAccess AST node

        Returns:
            Field value

        Example:
            x_val = p.x

            Becomes:
            %x_val = struct.extract %p[0] : struct<f32, f32> -> f32
        """
        # Compile struct expression
        struct_value = self.compile_expression(field_access.object)

        # Determine struct type from value
        # In simplified IR, we need to track this via type annotations
        # For now, extract from the type representation
        if not isinstance(struct_value.type, str) or not struct_value.type.startswith('struct<'):
            raise TypeError(f"Cannot access field on non-struct type: {struct_value.type}")

        # Find the struct definition that matches this type
        struct_name = None
        field_index = None
        field_type = None

        for name, meta in self.struct_types.items():
            if meta['mlir_type'] == struct_value.type:
                struct_name = name
                if field_access.field in meta['fields']:
                    field_index = meta['fields'][field_access.field]
                    field_type = meta['field_types'][field_index]
                break

        if field_index is None:
            raise ValueError(f"Unknown field: {field_access.field}")

        # Extract field value
        results = self.builder.add_operation(
            "struct.extract",
            operands=[struct_value],
            result_types=[field_type],
            attributes={
                "field_index": field_index,
                "field_name": field_access.field
            }
        )

        return results[0]

    def _find_free_variables(self, expr: Expression, bound_vars: set) -> set:
        """Find free variables in an expression (variables not in bound_vars).

        Args:
            expr: Expression to analyze
            bound_vars: Set of variable names that are bound (in scope)

        Returns:
            Set of free variable names (captured variables)
        """
        free_vars = set()

        if isinstance(expr, Identifier):
            if expr.name not in bound_vars:
                free_vars.add(expr.name)
        elif isinstance(expr, BinaryOp):
            free_vars.update(self._find_free_variables(expr.left, bound_vars))
            free_vars.update(self._find_free_variables(expr.right, bound_vars))
        elif isinstance(expr, UnaryOp):
            free_vars.update(self._find_free_variables(expr.operand, bound_vars))
        elif isinstance(expr, Call):
            if isinstance(expr.callee, Identifier):
                # Don't count function names or lambda names as captured variables
                # They can be called directly through the symbol table
                callee_name = expr.callee.name
                is_function = callee_name in self.functions
                is_lambda = (callee_name in self.symbols and
                           hasattr(self.symbols.get(callee_name), '_lambda_meta'))

                if not is_function and not is_lambda and callee_name not in bound_vars:
                    free_vars.add(callee_name)
            for arg in expr.args:
                free_vars.update(self._find_free_variables(arg, bound_vars))
        elif isinstance(expr, IfElse):
            free_vars.update(self._find_free_variables(expr.condition, bound_vars))
            free_vars.update(self._find_free_variables(expr.then_expr, bound_vars))
            free_vars.update(self._find_free_variables(expr.else_expr, bound_vars))
        elif isinstance(expr, FieldAccess):
            free_vars.update(self._find_free_variables(expr.object, bound_vars))
        elif isinstance(expr, StructLiteral):
            for field_expr in expr.fields.values():
                free_vars.update(self._find_free_variables(field_expr, bound_vars))
        elif isinstance(expr, Lambda):
            # Nested lambda: its params are bound within its body
            nested_bound = bound_vars | set(expr.params)
            free_vars.update(self._find_free_variables(expr.body, nested_bound))

        return free_vars

    def _build_lambda_signature(
        self, params: List[str], captured_var_list: List[str]
    ) -> tuple:
        """Build argument types and values for a lambda function.

        Args:
            params: Lambda parameter names
            captured_var_list: Captured variable names from enclosing scope

        Returns:
            Tuple of (arg_types, arg_values) lists
        """
        arg_types = []
        arg_values = []

        # Add lambda parameters
        for i, param_name in enumerate(params):
            param_type = IRType.F32  # TODO: Better type inference
            arg_types.append(param_type)
            arg_values.append(IRValue(name=f"%arg{i}", type=param_type))

        # Add captured variables as additional parameters
        for i, captured_var in enumerate(captured_var_list):
            captured_val = self.symbols[captured_var]
            param_idx = len(params) + i
            arg_types.append(captured_val.type)
            arg_values.append(IRValue(name=f"%arg{param_idx}", type=captured_val.type))

        return arg_types, arg_values

    def _create_lambda_value(
        self, lambda_name: str, captured_var_list: List[str]
    ) -> IRValue:
        """Create IRValue representing a compiled lambda with metadata.

        Args:
            lambda_name: Generated function name for the lambda
            captured_var_list: Names of captured variables

        Returns:
            IRValue with _lambda_meta attached
        """
        lambda_value = IRValue(
            name=f"%{lambda_name}",
            type=IRType.F32  # Placeholder type
        )
        lambda_value._lambda_meta = {
            'function': lambda_name,
            'captured_vars': captured_var_list,
            'captured_values': [self.symbols[v] for v in captured_var_list]
        }
        return lambda_value

    def compile_lambda(self, lambda_expr: Lambda) -> IRValue:
        """Compile lambda expression (Phase 4.1).

        Lambdas are compiled to regular MLIR functions with closure capture.
        Each lambda gets a unique function name and captures free variables
        from the enclosing scope.

        Args:
            lambda_expr: Lambda AST node

        Returns:
            IRValue representing the lambda (stored as function reference)

        Example:
            multiplier = 3.0
            scale = |x| x * multiplier

            Generates:
            func.func @__lambda_0(%arg0: f32, %captured_multiplier: f32) -> f32 {
                %0 = arith.mulf %arg0, %captured_multiplier : f32
                func.return %0 : f32
            }
        """
        # Generate unique lambda name
        lambda_name = f"__lambda_{self.lambda_counter}"
        self.lambda_counter += 1

        # Find captured variables (free variables in lambda body)
        bound_vars = set(lambda_expr.params)
        captured_vars = self._find_free_variables(lambda_expr.body, bound_vars)
        captured_vars = {v for v in captured_vars if v in self.symbols}
        captured_var_list = sorted(list(captured_vars))

        # Build function signature
        arg_types, arg_values = self._build_lambda_signature(
            lambda_expr.params, captured_var_list
        )

        # Save current context
        saved_symbols = self.symbols.copy()
        saved_function = self.current_function

        # Create function with placeholder return type
        ir_func = self.builder.create_function(
            name=lambda_name, args=arg_values, return_types=[]
        )
        self.functions[lambda_name] = ir_func
        self.current_function = lambda_name
        self.builder.create_block(label="entry")

        # Set up symbol table for lambda body
        self.symbols = saved_symbols.copy()
        for param_name, arg_value in zip(lambda_expr.params, arg_values[:len(lambda_expr.params)]):
            self.symbols[param_name] = arg_value
        for captured_var, arg_value in zip(captured_var_list, arg_values[len(lambda_expr.params):]):
            self.symbols[captured_var] = arg_value

        # Compile lambda body
        result = self.compile_expression(lambda_expr.body)
        ir_func.return_types = [result.type]

        # Add return statement
        self.builder.add_operation("func.return", operands=[result], result_types=[])

        # Restore context
        self.symbols = saved_symbols
        self.current_function = saved_function

        return self._create_lambda_value(lambda_name, captured_var_list)

    def compile_tuple(self, tuple_expr: Tuple) -> IRValue:
        """Compile tuple expression."""
        raise NotImplementedError("Tuple expressions")

    # =========================================================================
    # Flow Block Helpers
    # =========================================================================

    def _create_loop_bounds(self, num_steps_expr: IRValue) -> tuple:
        """Create loop bounds (zero, end, one) for a for-loop.

        Args:
            num_steps_expr: Expression for number of iterations

        Returns:
            Tuple of (zero, end, one) IRValues of INDEX type
        """
        zero = self.builder.add_operation(
            "arith.constant",
            operands=[],
            result_types=[IRType.INDEX],
            attributes={"value": 0}
        )[0]

        one = self.builder.add_operation(
            "arith.constant",
            operands=[],
            result_types=[IRType.INDEX],
            attributes={"value": 1}
        )[0]

        # Convert num_steps to index type
        if num_steps_expr.type in [IRType.I32, IRType.I64]:
            end = self.builder.add_operation(
                "arith.index_cast",
                operands=[num_steps_expr],
                result_types=[IRType.INDEX]
            )[0]
        else:
            # Float: convert to int first
            num_steps_int = self.builder.add_operation(
                "arith.fptosi",
                operands=[num_steps_expr],
                result_types=[IRType.I32]
            )[0]
            end = self.builder.add_operation(
                "arith.index_cast",
                operands=[num_steps_int],
                result_types=[IRType.INDEX]
            )[0]

        return zero, end, one

    def _execute_flow_body(
        self, loop, state_var_names: List[str], body: List[Statement], dt_value
    ) -> None:
        """Execute flow block body within a loop context.

        Args:
            loop: Loop context with iter_arg_values and yield_values
            state_var_names: Names of state variables to thread through loop
            body: List of statements to compile
            dt_value: Optional dt value to add to symbols
        """
        saved_symbols = self.symbols.copy()

        # Map iteration arguments to variable names
        for i, var_name in enumerate(state_var_names):
            self.symbols[var_name] = loop.iter_arg_values[i]

        # Add dt to symbol table if present
        if dt_value is not None:
            self.symbols['dt'] = dt_value

        # Compile body statements
        for stmt in body:
            self.compile_statement(stmt)

        # Collect updated state variable values and yield
        new_state_values = [self.symbols[var_name] for var_name in state_var_names]
        loop.yield_values(new_state_values)

        # Restore non-state symbols
        for var_name in saved_symbols:
            if var_name not in state_var_names:
                self.symbols[var_name] = saved_symbols[var_name]

    def compile_flow_block(self, flow: Flow) -> None:
        """Compile flow block to scf.for loop (Phase 3).

        Args:
            flow: Flow statement AST node

        Flow blocks are temporal iteration constructs that update state variables
        over time. They compile to scf.for loops with iteration arguments threading
        state through loop iterations.

        Example:
            @state x = 0.0

            flow(dt=0.1, steps=10) {
                x = x + dt
            }

            Compiles to:
            %dt = arith.constant 0.1 : f32
            %steps = arith.constant 10 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %end = arith.index_cast %steps : i32 to index

            %final_x = scf.for %iv = %c0 to %end step %c1
                iter_args(%x = %initial_x) -> (f32) {
                %new_x = arith.addf %x, %dt : f32
                scf.yield %new_x : f32
            }
        """
        # Determine number of iterations
        if flow.steps is not None:
            num_steps_expr = self.compile_expression(flow.steps)
        else:
            raise ValueError("Flow block must specify 'steps' parameter")

        # Compile dt if provided (for time-based flow)
        dt_value = self.compile_expression(flow.dt) if flow.dt is not None else None

        # Create loop bounds
        zero, end, one = self._create_loop_bounds(num_steps_expr)

        # Identify and prepare state variables
        state_vars = self._identify_state_variables(flow.body)
        iter_args, iter_types, state_var_names = [], [], []

        for var_name in state_vars:
            if var_name in self.symbols:
                var_value = self.symbols[var_name]
                iter_args.append(var_value)
                iter_types.append(var_value.type)
                state_var_names.append(var_name)

        # Handle substeps (nested loops) or single loop
        if flow.substeps is not None:
            self._compile_flow_with_substeps(
                zero, end, one, flow.substeps,
                iter_args, iter_types, state_var_names,
                flow.body, dt_value
            )
        else:
            with self.builder.create_for_loop(zero, end, one, iter_args, iter_types) as loop:
                self._execute_flow_body(loop, state_var_names, flow.body, dt_value)

            # Update symbol table with final loop values
            for i, var_name in enumerate(state_var_names):
                self.symbols[var_name] = loop.result_values[i]

    def _identify_state_variables(self, body: List[Statement]) -> List[str]:
        """Identify variables that are modified in the flow body.

        Args:
            body: Flow block body statements

        Returns:
            List of variable names that are assigned in the body
        """
        state_vars = []
        for stmt in body:
            if isinstance(stmt, Assignment):
                if stmt.target not in state_vars:
                    state_vars.append(stmt.target)
            # Could also check for other statement types that modify variables
        return state_vars

    def _compile_flow_with_substeps(self, start: IRValue, end: IRValue, step: IRValue,
                                    substeps_expr: Expression,
                                    iter_args: List[IRValue], iter_types: List[Union[IRType, str]],
                                    state_var_names: List[str],
                                    body: List[Statement], dt_value: Optional[IRValue]):
        """Compile flow block with substeps (nested loops).

        Args:
            start, end, step: Outer loop bounds
            substeps_expr: Expression for number of substeps
            iter_args: Initial iteration argument values
            iter_types: Types of iteration arguments
            state_var_names: Names of state variables
            body: Flow body statements
            dt_value: Timestep value (if time-based)
        """
        # Create inner loop bounds from substeps expression
        num_substeps = self.compile_expression(substeps_expr)
        zero_inner, substeps_end, one_inner = self._create_loop_bounds(num_substeps)

        # Outer loop (main steps)
        with self.builder.create_for_loop(start, end, step, iter_args, iter_types) as outer_loop:
            saved_symbols = self.symbols.copy()

            # Map outer iteration arguments
            for i, var_name in enumerate(state_var_names):
                self.symbols[var_name] = outer_loop.iter_arg_values[i]
            if dt_value is not None:
                self.symbols['dt'] = dt_value

            # Inner loop (substeps)
            with self.builder.create_for_loop(
                zero_inner, substeps_end, one_inner,
                outer_loop.iter_arg_values, iter_types
            ) as inner_loop:
                # Map inner iteration arguments
                for i, var_name in enumerate(state_var_names):
                    self.symbols[var_name] = inner_loop.iter_arg_values[i]
                if dt_value is not None:
                    self.symbols['dt'] = dt_value

                # Compile body and yield
                for stmt in body:
                    self.compile_statement(stmt)
                inner_loop.yield_values([self.symbols[v] for v in state_var_names])

            # Yield inner results to outer loop
            outer_loop.yield_values(inner_loop.result_values)

            # Restore non-state variables
            for var_name in saved_symbols:
                if var_name not in state_var_names:
                    self.symbols[var_name] = saved_symbols[var_name]

        # Update symbol table with final values
        for i, var_name in enumerate(state_var_names):
            self.symbols[var_name] = outer_loop.result_values[i]
