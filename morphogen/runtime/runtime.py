"""Runtime execution engine for Creative Computation DSL.

This module provides the core runtime infrastructure for executing
Creative Computation DSL programs using NumPy as the backend.
"""

from typing import Any, Dict, Optional, List


class ReturnValue(Exception):
    """Exception used to implement early return from functions."""

    def __init__(self, value: Any):
        """Initialize with return value.

        Args:
            value: The value being returned
        """
        self.value = value
        super().__init__()


class UserDefinedFunction:
    """Represents a user-defined function."""

    def __init__(self, name: str, params: List[tuple], body: List, runtime: 'Runtime'):
        """Initialize user-defined function.

        Args:
            name: Function name
            params: List of (param_name, type_annotation) tuples
            body: List of statements in function body
            runtime: Runtime instance for execution
        """
        self.name = name
        self.params = params
        self.body = body
        self.runtime = runtime

    def __call__(self, *args, **kwargs):
        """Execute the function with given arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function return value (or None)
        """
        # Check argument count
        if len(args) != len(self.params):
            raise TypeError(
                f"{self.name}() takes {len(self.params)} positional arguments "
                f"but {len(args)} were given"
            )

        # Save current symbol table state
        saved_symbols = self.runtime.context.symbols.copy()

        try:
            # Bind parameters to arguments
            for (param_name, _), arg_value in zip(self.params, args):
                self.runtime.context.set_variable(param_name, arg_value)

            # Execute function body
            result = None
            try:
                for stmt in self.body:
                    self.runtime.execute_statement(stmt)
            except ReturnValue as ret:
                result = ret.value

            return result

        finally:
            # Preserve state variable modifications before restoring symbol table
            state_var_updates = {}
            for name in self.runtime.context.state_symbols:
                if name in self.runtime.context.symbols:
                    state_var_updates[name] = self.runtime.context.symbols[name]

            # Restore symbol table
            self.runtime.context.symbols = saved_symbols

            # Re-apply state variable modifications
            for name, value in state_var_updates.items():
                self.runtime.context.symbols[name] = value


class LambdaFunction:
    """Represents a lambda expression (closure)."""

    def __init__(self, params: List[str], body, runtime: 'Runtime', captured_vars: Dict[str, Any]):
        """Initialize lambda function.

        Args:
            params: List of parameter names
            body: Expression to evaluate
            runtime: Runtime instance for execution
            captured_vars: Variables captured from enclosing scope
        """
        self.params = params
        self.body = body
        self.runtime = runtime
        self.captured_vars = captured_vars

    def __call__(self, *args, **kwargs):
        """Execute lambda with given arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of body expression
        """
        # Check argument count
        if len(args) != len(self.params):
            raise TypeError(
                f"Lambda takes {len(self.params)} positional arguments "
                f"but {len(args)} were given"
            )

        # Save current symbol table state
        saved_symbols = self.runtime.context.symbols.copy()

        try:
            # Restore captured variables
            for name, value in self.captured_vars.items():
                self.runtime.context.set_variable(name, value)

            # Bind parameters to arguments
            for param_name, arg_value in zip(self.params, args):
                self.runtime.context.set_variable(param_name, arg_value)

            # Evaluate body expression
            result = self.runtime.execute_expression(self.body)
            return result

        except ReturnValue as rv:
            # Handle explicit return statements in lambda bodies
            return rv.value

        finally:
            # Restore symbol table
            self.runtime.context.symbols = saved_symbols


class StructType:
    """Represents a struct type definition."""

    def __init__(self, name: str, fields: List[tuple]):
        """Initialize struct type.

        Args:
            name: Struct name
            fields: List of (field_name, type_annotation) tuples
        """
        self.name = name
        self.fields = fields
        self.field_names = [name for name, _ in fields]

    def __call__(self, **field_values):
        """Create an instance of the struct.

        Args:
            **field_values: Field values as keyword arguments

        Returns:
            StructInstance
        """
        return StructInstance(self, field_values)


class StructInstance:
    """Represents an instance of a struct."""

    def __init__(self, struct_type: StructType, field_values: Dict[str, Any]):
        """Initialize struct instance.

        Args:
            struct_type: The struct type
            field_values: Dictionary of field values
        """
        self.struct_type = struct_type
        self.fields = field_values

    def __getattr__(self, name: str) -> Any:
        """Get field value.

        Args:
            name: Field name

        Returns:
            Field value
        """
        if name in ['struct_type', 'fields']:
            return object.__getattribute__(self, name)
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(f"Struct {self.struct_type.name} has no field '{name}'")

    def __repr__(self) -> str:
        """Return string representation of struct instance.

        Returns:
            String like "Point { x: 3.0, y: 4.0 }"
        """
        field_strs = [f"{k}: {v}" for k, v in self.fields.items()]
        return f"{self.struct_type.name} {{ {', '.join(field_strs)} }}"

    def __eq__(self, other) -> bool:
        """Check equality with another struct instance.

        Args:
            other: Another object to compare with

        Returns:
            True if both are instances of the same struct type with equal field values
        """
        if not isinstance(other, StructInstance):
            return False
        if self.struct_type.name != other.struct_type.name:
            return False
        return self.fields == other.fields


class ExecutionContext:
    """Manages execution state across timesteps.

    Handles:
    - Symbol table for variable storage
    - Double-buffered resources
    - Timestep management
    - Configuration settings
    """

    def __init__(self, global_seed: int = 42):
        """Initialize execution context.

        Args:
            global_seed: Global random seed for deterministic execution
        """
        self.symbols: Dict[str, Any] = {}
        self.const_symbols: set = set()  # Track which symbols are const
        self.state_symbols: set = set()  # Track which symbols are state variables
        self.double_buffers: Dict[str, tuple] = {}  # name -> (front, back)
        self.config: Dict[str, Any] = {}
        self.timestep: int = 0
        self.global_seed: int = global_seed
        self.dt: float = 0.01  # default timestep

    def set_variable(self, name: str, value: Any, is_const: bool = False,
                     is_state: bool = False) -> None:
        """Set a variable in the symbol table.

        Args:
            name: Variable name
            value: Variable value
            is_const: Whether this is a const variable
            is_state: Whether this is a state variable (@state decorator)

        Raises:
            RuntimeError: If trying to reassign a const variable
        """
        if name in self.const_symbols:
            raise RuntimeError(f"Cannot reassign const variable: {name}")
        self.symbols[name] = value
        if is_const:
            self.const_symbols.add(name)
        if is_state:
            self.state_symbols.add(name)

    def get_variable(self, name: str) -> Any:
        """Get a variable from the symbol table.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        if name not in self.symbols:
            raise KeyError(f"Undefined variable: {name}")
        return self.symbols[name]

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists.

        Args:
            name: Variable name

        Returns:
            True if variable exists
        """
        return name in self.symbols

    def register_double_buffer(self, name: str, front_buffer: Any, back_buffer: Any) -> None:
        """Register a double-buffered variable.

        Args:
            name: Variable name
            front_buffer: Front buffer (read from)
            back_buffer: Back buffer (write to)
        """
        self.double_buffers[name] = (front_buffer, back_buffer)
        self.symbols[name] = front_buffer

    def swap_buffers(self) -> None:
        """Swap double buffers at end of timestep."""
        for name, (front, back) in self.double_buffers.items():
            self.double_buffers[name] = (back, front)
            self.symbols[name] = back

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

        # Special handling for dt
        if key == "dt":
            self.dt = float(value)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def advance_timestep(self) -> None:
        """Advance to next timestep."""
        self.timestep += 1
        self.swap_buffers()


class Runtime:
    """Main runtime for executing Creative Computation DSL programs.

    Provides the interpreter that walks the AST and executes operations
    using NumPy-based implementations.
    """

    def __init__(self, context: Optional[ExecutionContext] = None):
        """Initialize runtime.

        Args:
            context: Execution context (creates new one if None)
        """
        self.context = context or ExecutionContext()
        self._setup_builtins()

    def _setup_builtins(self) -> None:
        """Set up built-in namespaces (field, visual, agents, audio, etc.)."""
        import math
        from ..stdlib.field import field
        from ..stdlib.visual import visual
        from ..stdlib.agents import agents
        from ..stdlib.audio import audio

        # Register built-in namespaces
        self.context.set_variable("field", field)
        self.context.set_variable("visual", visual)
        self.context.set_variable("agents", agents)
        self.context.set_variable("audio", audio)

        # Register math builtins
        self.context.set_variable("sqrt", math.sqrt)
        self.context.set_variable("sin", math.sin)
        self.context.set_variable("cos", math.cos)
        self.context.set_variable("tan", math.tan)
        self.context.set_variable("abs", abs)
        self.context.set_variable("min", min)
        self.context.set_variable("max", max)
        self.context.set_variable("floor", math.floor)
        self.context.set_variable("ceil", math.ceil)
        self.context.set_variable("exp", math.exp)
        self.context.set_variable("log", math.log)
        self.context.set_variable("pow", pow)
        self.context.set_variable("pi", math.pi)
        self.context.set_variable("e", math.e)

    def execute_program(self, program) -> None:
        """Execute a complete DSL program.

        Args:
            program: Program AST node
        """
        from ..ast.nodes import Program

        if not isinstance(program, Program):
            raise TypeError(f"Expected Program node, got {type(program)}")

        # Execute statements in order
        for stmt in program.statements:
            try:
                self.execute_statement(stmt)
            except ReturnValue:
                # Return outside function - raise error
                raise RuntimeError("Return statement outside function")

    # Dispatch table for statement types (type name -> handler method name)
    _STATEMENT_DISPATCH = {
        'Output': 'execute_output',
        'Use': 'execute_use',
        'Assignment': 'execute_assignment',
        'Function': 'execute_function',
        'Return': 'execute_return',
        'Flow': 'execute_flow',
        'Struct': 'execute_struct',
        'Step': 'execute_step',
        'Substep': 'execute_substep',
    }

    def execute_statement(self, stmt) -> Any:
        """Execute a single statement.

        Args:
            stmt: Statement AST node

        Returns:
            Result of statement execution (if any)
        """
        type_name = type(stmt).__name__

        # Fast path: dispatch table lookup
        if type_name in self._STATEMENT_DISPATCH:
            handler = getattr(self, self._STATEMENT_DISPATCH[type_name])
            return handler(stmt)

        # ExpressionStatement: unwrap and execute the inner expression
        if type_name == 'ExpressionStatement':
            return self.execute_expression(stmt.expression)

        # Call: check for special 'set' statement
        if type_name == 'Call':
            if type(stmt.func).__name__ == 'Identifier' and stmt.func.name == "set":
                return self.execute_set_statement(stmt)
            return self.execute_expression(stmt)

        # Post-MVP features
        if type_name in ('Module', 'Compose'):
            raise NotImplementedError(f"{type_name} execution not yet implemented (post-MVP)")

        # Fallback: try to execute as expression
        return self.execute_expression(stmt)

    def execute_assignment(self, assign) -> None:
        """Execute an assignment statement.

        Args:
            assign: Assignment AST node
        """
        # Evaluate right-hand side
        value = self.execute_expression(assign.value)

        # Check for @state decorator
        is_state = any(d.name == 'state' for d in getattr(assign, 'decorators', []))

        # Store in context (target is a string)
        self.context.set_variable(assign.target, value, is_const=assign.is_const,
                                  is_state=is_state)

    def execute_set_statement(self, call) -> None:
        """Execute a 'set' configuration statement.

        Args:
            call: Call node representing 'set variable = value'
        """
        if len(call.args) != 1:
            raise ValueError("set statement requires exactly one argument")

        # Parse as assignment
        arg = call.args[0]
        from ..ast.nodes import BinaryOp

        if isinstance(arg, BinaryOp) and arg.op == "=":
            key = arg.left.name if hasattr(arg.left, 'name') else str(arg.left)
            value = self.execute_expression(arg.right)
            self.context.set_config(key, value)
        else:
            raise ValueError("set statement requires assignment syntax: set key = value")

    def execute_step(self, step) -> None:
        """Execute a step block.

        Args:
            step: Step AST node
        """
        # Execute all statements in the step
        for stmt in step.statements:
            self.execute_statement(stmt)

        # Advance timestep (swap buffers)
        self.context.advance_timestep()

    def execute_substep(self, substep) -> None:
        """Execute a substep block.

        Args:
            substep: Substep AST node
        """
        # Get iteration count
        n = self.execute_expression(substep.count)

        # Save original dt and divide by n
        original_dt = self.context.dt
        self.context.dt = original_dt / n

        # Execute n times
        for _ in range(n):
            for stmt in substep.statements:
                self.execute_statement(stmt)
            self.context.advance_timestep()

        # Restore original dt
        self.context.dt = original_dt

    # Dispatch table for expression types (type name -> handler method name)
    _EXPRESSION_DISPATCH = {
        'BinaryOp': 'execute_binary_op',
        'UnaryOp': 'execute_unary_op',
        'Call': 'execute_call',
        'FieldAccess': 'execute_field_access',
        'Lambda': 'execute_lambda',
        'IfElse': 'execute_if_else',
        'Block': 'execute_block',
        'StructLiteral': 'execute_struct_literal',
    }

    def execute_expression(self, expr) -> Any:
        """Execute an expression and return its value.

        Args:
            expr: Expression AST node

        Returns:
            Evaluated expression value
        """
        type_name = type(expr).__name__

        # Fast path: dispatch table lookup
        if type_name in self._EXPRESSION_DISPATCH:
            handler = getattr(self, self._EXPRESSION_DISPATCH[type_name])
            return handler(expr)

        # Inline handlers for simple cases
        if type_name == 'Literal':
            return expr.value

        if type_name == 'Identifier':
            return self.context.get_variable(expr.name)

        if type_name == 'Tuple':
            return tuple(self.execute_expression(elem) for elem in expr.elements)

        raise TypeError(f"Unknown expression type: {type(expr)}")

    def execute_binary_op(self, binop) -> Any:
        """Execute a binary operation.

        Args:
            binop: BinaryOp AST node

        Returns:
            Result of operation
        """
        left = self.execute_expression(binop.left)
        right = self.execute_expression(binop.right)

        # Map operators to NumPy operations
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '%': lambda a, b: a % b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
        }

        if binop.operator not in ops:
            raise ValueError(f"Unknown binary operator: {binop.operator}")

        return ops[binop.operator](left, right)

    def execute_unary_op(self, unop) -> Any:
        """Execute a unary operation.

        Args:
            unop: UnaryOp AST node

        Returns:
            Result of operation
        """
        operand = self.execute_expression(unop.operand)

        if unop.operator == '-':
            return -operand
        elif unop.operator == '!':
            return not operand
        else:
            raise ValueError(f"Unknown unary operator: {unop.operator}")

    def execute_call(self, call) -> Any:
        """Execute a function call.

        Args:
            call: Call AST node

        Returns:
            Result of function call
        """
        from ..ast.nodes import FieldAccess

        # Evaluate arguments
        args = [self.execute_expression(arg) for arg in call.args]
        kwargs = {k: self.execute_expression(v) for k, v in call.kwargs.items()}

        # Handle method calls (e.g., field.alloc(...))
        if isinstance(call.callee, FieldAccess):
            obj = self.execute_expression(call.callee.object)
            method_name = call.callee.field

            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return method(*args, **kwargs)
                else:
                    raise TypeError(f"'{method_name}' is not callable")
            else:
                raise AttributeError(f"Object has no method '{method_name}'")

        # Handle regular function calls
        func = self.execute_expression(call.callee)

        if callable(func):
            return func(*args, **kwargs)
        else:
            raise TypeError(f"Cannot call non-function: {type(func)}")

    def execute_field_access(self, field_access) -> Any:
        """Execute field access (method call or attribute access).

        Args:
            field_access: FieldAccess AST node

        Returns:
            Field or method
        """
        # Get the object
        obj = self.execute_expression(field_access.object)

        # Access the field
        if hasattr(obj, field_access.field):
            return getattr(obj, field_access.field)
        else:
            # Try as dictionary access
            if isinstance(obj, dict) and field_access.field in obj:
                return obj[field_access.field]

            raise AttributeError(f"Object has no field '{field_access.field}'")

    def execute_function(self, func_node) -> None:
        """Execute a function definition.

        Args:
            func_node: Function AST node
        """
        # Create user-defined function and store in symbol table
        user_func = UserDefinedFunction(
            name=func_node.name,
            params=func_node.params,
            body=func_node.body,
            runtime=self
        )
        self.context.set_variable(func_node.name, user_func)

    def execute_return(self, return_node) -> None:
        """Execute a return statement.

        Args:
            return_node: Return AST node

        Raises:
            ReturnValue: Exception carrying the return value
            RuntimeError: If return used outside function
        """
        # Evaluate return value (if any)
        value = None
        if return_node.value is not None:
            value = self.execute_expression(return_node.value)

        # Raise exception to exit function
        raise ReturnValue(value)

    def execute_output(self, output_node) -> None:
        """Execute an output statement for visual rendering.

        Evaluates the expression (typically a colorize call) and stores it
        in the context's output buffer for visualization.

        Args:
            output_node: Output AST node
        """
        # Evaluate the output expression
        value = self.execute_expression(output_node.value)

        # Store in output buffer (create if needed)
        if not hasattr(self.context, 'outputs'):
            self.context.outputs = []
        self.context.outputs.append(value)

        return value

    # Domain module mapping (constant)
    _DOMAIN_MODULES = {
        'field': 'morphogen.stdlib.field',
        'visual': 'morphogen.stdlib.visual',
        'agents': 'morphogen.stdlib.agents',
        'audio': 'morphogen.stdlib.audio',
        'cellular': 'morphogen.stdlib.cellular',
        'geometry': 'morphogen.stdlib.geometry',
        'noise': 'morphogen.stdlib.noise',
        'palette': 'morphogen.stdlib.palette',
        'signal': 'morphogen.stdlib.signal',
        'temporal': 'morphogen.stdlib.temporal',
    }

    def _import_domain_exports(self, module_path: str) -> None:
        """Import all exported names from a domain module.

        Args:
            module_path: Full module path (e.g., 'morphogen.stdlib.field')
        """
        import importlib
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            return

        if not hasattr(module, '__all__'):
            return

        for name in module.__all__:
            if hasattr(module, name) and name not in self.context.symbols:
                self.context.set_variable(name, getattr(module, name))

    def execute_use(self, use_node) -> None:
        """Execute a use statement to import domain operators.

        Args:
            use_node: Use AST node

        Raises:
            ValueError: If a domain is not registered
        """
        from ..core.domain_registry import DomainRegistry
        DomainRegistry.initialize()

        for domain_name in use_node.domains:
            if not DomainRegistry.has_domain(domain_name):
                available = DomainRegistry.list_domains()
                raise ValueError(
                    f"Domain '{domain_name}' not found. "
                    f"Available domains: {', '.join(available)}"
                )

            if domain_name in self._DOMAIN_MODULES:
                self._import_domain_exports(self._DOMAIN_MODULES[domain_name])

    def execute_if_else(self, if_else_node) -> Any:
        """Execute an if/else expression.

        Args:
            if_else_node: IfElse AST node

        Returns:
            Result of then_expr or else_expr
        """
        # Evaluate condition
        condition = self.execute_expression(if_else_node.condition)

        # Choose branch based on condition
        if condition:
            return self.execute_expression(if_else_node.then_expr)
        else:
            return self.execute_expression(if_else_node.else_expr)

    def execute_block(self, block_node) -> Any:
        """Execute a block expression ({ stmt1; stmt2; ... }).

        Args:
            block_node: Block AST node

        Returns:
            Value of last expression in block, or None
        """
        result = None
        for stmt in block_node.statements:
            # Execute statement - if it's an expression statement, capture its value
            self.execute_statement(stmt)
            # If the statement was an expression, get its value
            from ..ast.nodes import ExpressionStatement, Return
            if isinstance(stmt, ExpressionStatement):
                result = self.execute_expression(stmt.expression)
            elif isinstance(stmt, Return):
                # Return statements will raise ReturnValue, so this won't be reached
                # but we include it for completeness
                pass
        return result

    def execute_lambda(self, lambda_node) -> LambdaFunction:
        """Execute a lambda expression (create closure).

        Args:
            lambda_node: Lambda AST node

        Returns:
            LambdaFunction instance
        """
        # Capture current variables (closure)
        # We capture all current symbols except built-ins and consts
        # Consts are already accessible in scope and can't change
        captured_vars = {}
        for name, value in self.context.symbols.items():
            # Don't capture built-in namespaces or const variables
            if name not in ['field', 'visual'] and name not in self.context.const_symbols:
                captured_vars[name] = value

        # Create lambda function
        return LambdaFunction(
            params=lambda_node.params,
            body=lambda_node.body,
            runtime=self,
            captured_vars=captured_vars
        )

    def execute_struct(self, struct_node) -> None:
        """Execute a struct definition.

        Args:
            struct_node: Struct AST node
        """
        # Create struct type and store in symbol table
        struct_type = StructType(
            name=struct_node.name,
            fields=struct_node.fields
        )
        self.context.set_variable(struct_node.name, struct_type)

    def execute_struct_literal(self, struct_literal) -> StructInstance:
        """Execute a struct literal instantiation.

        Args:
            struct_literal: StructLiteral AST node

        Returns:
            StructInstance with initialized fields

        Raises:
            RuntimeError: If struct type is undefined or fields are invalid

        Example:
            Point { x: 3.0, y: 4.0 } creates a Point instance
        """
        # Look up struct type
        struct_name = struct_literal.struct_name
        if not self.context.has_variable(struct_name):
            raise RuntimeError(f"Undefined struct type: '{struct_name}'")

        struct_type = self.context.get_variable(struct_name)

        if not isinstance(struct_type, StructType):
            raise RuntimeError(f"'{struct_name}' is not a struct type")

        # Evaluate field values
        field_values = {}
        for field_name, field_expr in struct_literal.field_values.items():
            # Validate field exists in struct definition
            if field_name not in struct_type.field_names:
                raise RuntimeError(
                    f"Struct '{struct_name}' has no field '{field_name}'. "
                    f"Available fields: {', '.join(struct_type.field_names)}"
                )

            # Evaluate field expression
            field_values[field_name] = self.execute_expression(field_expr)

        # Validate all required fields provided
        missing_fields = set(struct_type.field_names) - set(field_values.keys())
        if missing_fields:
            raise RuntimeError(
                f"Missing required fields in '{struct_name}' literal: {{{', '.join(sorted(missing_fields))}}}. "
                f"Provided: {{{', '.join(sorted(field_values.keys()))}}}. "
                f"Required: {{{', '.join(struct_type.field_names)}}}"
            )

        # Create struct instance
        return StructInstance(struct_type, field_values)

    def _init_flow_state_vars(self, body) -> Dict[str, Any]:
        """Initialize state variables from flow body.

        Args:
            body: List of statements in flow body

        Returns:
            Dict mapping state variable names to initial values
        """
        state_vars = {}
        for stmt in body:
            if type(stmt).__name__ != 'Assignment':
                continue
            if not any(d.name == 'state' for d in stmt.decorators):
                continue
            initial_value = self.execute_expression(stmt.value)
            state_vars[stmt.target] = initial_value
            self.context.set_variable(stmt.target, initial_value)
        return state_vars

    def _execute_flow_body(self, body, state_vars: Dict[str, Any]) -> None:
        """Execute flow body statements.

        Args:
            body: List of statements
            state_vars: Dict of state variable names
        """
        for stmt in body:
            is_assignment = type(stmt).__name__ == 'Assignment'
            is_state = is_assignment and any(d.name == 'state' for d in stmt.decorators)
            if is_state:
                if stmt.target in state_vars:
                    self.execute_statement(stmt)
                continue
            self.execute_statement(stmt)

    def _execute_flow_iteration(self, body, state_vars: Dict[str, Any], dt: float) -> None:
        """Execute a single flow iteration with dt context.

        Args:
            body: List of statements
            state_vars: Dict of state variable names
            dt: Delta time for this iteration
        """
        old_dt = self.context.dt
        self.context.dt = dt
        self.context.set_variable('dt', dt)
        self._execute_flow_body(body, state_vars)
        self.context.dt = old_dt

    def execute_flow(self, flow_node) -> None:
        """Execute a flow block with temporal semantics.

        Args:
            flow_node: Flow AST node
        """
        # Evaluate flow parameters
        dt = self.execute_expression(flow_node.dt) if flow_node.dt else 0.01
        steps = int(self.execute_expression(flow_node.steps) if flow_node.steps else 1)
        substeps = self.execute_expression(flow_node.substeps) if flow_node.substeps else None

        # Initialize state variables
        state_vars = self._init_flow_state_vars(flow_node.body)

        # Execute flow loop
        if substeps:
            sub_dt = dt / substeps
            for _ in range(steps):
                for _ in range(int(substeps)):
                    self._execute_flow_iteration(flow_node.body, state_vars, sub_dt)
                self.context.advance_timestep()
        else:
            for _ in range(steps):
                self._execute_flow_iteration(flow_node.body, state_vars, dt)
                self.context.advance_timestep()
