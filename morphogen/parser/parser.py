"""Recursive descent parser for Creative Computation DSL."""

from typing import List, Optional
from morphogen.lexer.lexer import Token, TokenType, Lexer
from morphogen.ast.nodes import *


class ParseError(Exception):
    """Exception raised during parsing."""
    pass


class Parser:
    """Recursive descent parser for the DSL."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        """Get the current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Peek at a token ahead."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        """Advance to the next token."""
        token = self.current_token()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type and advance."""
        token = self.current_token()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at {token.line}:{token.column}"
            )
        return self.advance()

    def skip_newlines(self):
        """Skip any newline tokens."""
        while self.current_token().type == TokenType.NEWLINE:
            self.advance()

    def parse(self) -> Program:
        """Parse the entire program."""
        statements = []
        self.skip_newlines()

        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        return Program(statements)

    def parse_statement(self) -> Optional[Statement]:
        """Parse a single statement."""
        self.skip_newlines()
        token = self.current_token()

        # Parse decorators
        decorators = []
        while token.type == TokenType.AT:
            decorators.append(self.parse_decorator())
            self.skip_newlines()
            token = self.current_token()

        # Step block (legacy)
        if token.type == TokenType.STEP:
            return self.parse_step()

        # Substep block (legacy)
        if token.type == TokenType.SUBSTEP:
            return self.parse_substep()

        # Flow block (v0.3.1)
        if token.type == TokenType.FLOW:
            return self.parse_flow()

        # Function definition (v0.3.1)
        if token.type == TokenType.FN:
            return self.parse_function()

        # Struct definition (v0.3.1)
        if token.type == TokenType.STRUCT:
            return self.parse_struct()

        # Return statement (v0.3.1)
        if token.type == TokenType.RETURN:
            return self.parse_return()

        # Module definition
        if token.type == TokenType.MODULE:
            return self.parse_module()

        # Compose statement
        if token.type == TokenType.COMPOSE:
            return self.parse_compose()

        # Link statement
        if token.type == TokenType.LINK:
            return self.parse_link()

        # Use statement
        if token.type == TokenType.USE:
            return self.parse_use()

        # Output statement (visual output)
        if token.type == TokenType.OUTPUT:
            return self.parse_output()

        # Const declaration
        if token.type == TokenType.CONST:
            return self.parse_const_declaration(decorators)

        # Assignment or expression statement
        if token.type == TokenType.IDENTIFIER:
            # Look ahead to determine if it's an assignment or expression statement
            # Check if there's '=' or ':' after the identifier (for type annotation)
            if (self.peek_token().type == TokenType.ASSIGN or
                self.peek_token().type == TokenType.COLON):
                return self.parse_assignment(decorators)
            else:
                # It's an expression statement (e.g., function call)
                expr = self.parse_expression()
                return ExpressionStatement(expr)

        # Type definition
        if token.type == TokenType.TYPE:
            return self.parse_type_definition()

        # Set statement (for configuration)
        if token.type == TokenType.SET:
            return self.parse_set_statement()

        # If expression as statement (e.g., inside lambda blocks)
        if token.type == TokenType.IF:
            expr = self.parse_if_else()
            return ExpressionStatement(expr)

        return None

    def parse_decorator(self) -> Decorator:
        """Parse a decorator (@name or @name(args))."""
        self.expect(TokenType.AT)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value

        args = []
        kwargs = {}

        # Check for arguments
        if self.current_token().type == TokenType.LPAREN:
            self.advance()
            self.skip_newlines()

            while self.current_token().type != TokenType.RPAREN:
                # Check if it's a keyword argument
                if (self.current_token().type == TokenType.IDENTIFIER and
                    self.peek_token().type == TokenType.ASSIGN):
                    key = self.current_token().value
                    self.advance()
                    self.advance()  # Skip '='
                    value = self.parse_expression()
                    kwargs[key] = value
                else:
                    args.append(self.parse_expression())

                if self.current_token().type == TokenType.COMMA:
                    self.advance()
                    self.skip_newlines()

            self.expect(TokenType.RPAREN)

        return Decorator(name, args, kwargs)

    def parse_step(self) -> Step:
        """Parse a step block."""
        self.expect(TokenType.STEP)
        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        body = []
        while self.current_token().type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        return Step(body)

    def parse_substep(self) -> Substep:
        """Parse a substep block."""
        self.expect(TokenType.SUBSTEP)
        self.expect(TokenType.LPAREN)
        count = self.parse_expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        body = []
        while self.current_token().type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        return Substep(count, body)

    def parse_flow(self) -> Flow:
        """Parse a flow block: flow(dt=0.01, steps=100) { body }"""
        self.expect(TokenType.FLOW)
        self.expect(TokenType.LPAREN)

        dt = None
        steps = None
        substeps = None

        # Parse keyword arguments
        while self.current_token().type != TokenType.RPAREN:
            if self.current_token().type == TokenType.IDENTIFIER:
                key = self.current_token().value
                self.advance()
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()

                if key == "dt":
                    dt = value
                elif key == "steps":
                    steps = value
                elif key == "substeps":
                    substeps = value
                else:
                    raise ParseError(f"Unknown flow parameter: {key}")

                if self.current_token().type == TokenType.COMMA:
                    self.advance()
            else:
                break

        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        # Parse body
        body = []
        while self.current_token().type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        return Flow(dt=dt, steps=steps, substeps=substeps, body=body)

    def parse_function(self) -> Function:
        """Parse a function definition: fn name(params) -> return_type { body }"""
        self.expect(TokenType.FN)
        name = self.expect(TokenType.IDENTIFIER).value

        # Parse parameters
        self.expect(TokenType.LPAREN)
        params = []
        while self.current_token().type != TokenType.RPAREN:
            param_name = self.expect(TokenType.IDENTIFIER).value
            param_type = None

            if self.current_token().type == TokenType.COLON:
                self.advance()
                param_type = self.parse_type_annotation()

            params.append((param_name, param_type))

            if self.current_token().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)

        # Parse return type
        return_type = None
        if self.current_token().type == TokenType.ARROW:
            self.advance()
            return_type = self.parse_type_annotation()

        # Parse body
        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        body = []
        while self.current_token().type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        return Function(name=name, params=params, return_type=return_type, body=body)

    def parse_struct(self) -> Struct:
        """Parse a struct definition: struct Name { fields }"""
        self.expect(TokenType.STRUCT)
        name = self.expect(TokenType.IDENTIFIER).value

        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        fields = []
        while self.current_token().type != TokenType.RBRACE:
            field_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            field_type = self.parse_type_annotation()
            fields.append((field_name, field_type))

            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        return Struct(name=name, fields=fields)

    def parse_return(self) -> Return:
        """Parse a return statement: return expr"""
        self.expect(TokenType.RETURN)

        # Check if there's an expression to return
        value = None
        if self.current_token().type not in [TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF]:
            value = self.parse_expression()

        return Return(value=value)

    def parse_output(self) -> 'Output':
        """Parse an output statement: output expr"""
        from morphogen.ast.nodes import Output
        self.expect(TokenType.OUTPUT)

        # Output requires an expression
        if self.current_token().type in [TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF]:
            raise ParseError("Output statement requires an expression", self.current_token())

        value = self.parse_expression()
        return Output(value=value)

    def parse_lambda(self) -> Lambda:
        """Parse a lambda expression: |args| expr or |args| { stmts }"""
        self.expect(TokenType.PIPE)

        # Parse parameters (may be empty for || expr)
        params = []
        # Check if we immediately have closing pipe (no params case)
        if self.current_token().type != TokenType.PIPE:
            while True:
                params.append(self.expect(TokenType.IDENTIFIER).value)
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
                elif self.current_token().type == TokenType.PIPE:
                    break
                else:
                    raise ParseError(
                        f"Expected COMMA or PIPE in lambda parameters, got {self.current_token().type.name} "
                        f"at {self.current_token().line}:{self.current_token().column}"
                    )

        self.expect(TokenType.PIPE)

        # Parse body - can be single expression or block
        if self.current_token().type == TokenType.LBRACE:
            # Block body: |args| { stmts }
            self.advance()  # consume {
            self.skip_newlines()

            statements = []
            while self.current_token().type != TokenType.RBRACE:
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
                self.skip_newlines()

            self.expect(TokenType.RBRACE)
            body = Block(statements=statements)
        else:
            # Single expression body: |args| expr
            body = self.parse_expression()

        return Lambda(params=params, body=body)

    def parse_if_else(self) -> IfElse:
        """Parse if/else expression: if cond then expr1 else expr2 or if cond { expr1 } else { expr2 }"""
        self.expect(TokenType.IF)

        # Parse condition
        condition = self.parse_expression()

        # Check for 'then' keyword (for inline syntax) or '{' (for block syntax)
        then_expr = None
        if self.current_token().type == TokenType.THEN:
            # Inline syntax: if cond then expr
            self.advance()
            then_expr = self.parse_expression()
        elif self.current_token().type == TokenType.LBRACE:
            # Block syntax: if cond { expr }
            self.advance()
            self.skip_newlines()
            then_expr = self.parse_expression()
            self.skip_newlines()
            self.expect(TokenType.RBRACE)
        else:
            raise ParseError(
                f"Expected 'then' or '{{' after if condition at {self.current_token().line}:{self.current_token().column}"
            )

        # Parse else branch
        self.expect(TokenType.ELSE)

        else_expr = None
        if self.current_token().type == TokenType.LBRACE:
            # Block syntax: else { expr }
            self.advance()
            self.skip_newlines()
            else_expr = self.parse_expression()
            self.skip_newlines()
            self.expect(TokenType.RBRACE)
        elif self.current_token().type == TokenType.IF:
            # Chained if: else if ...
            else_expr = self.parse_if_else()
        else:
            # Inline syntax: else expr
            else_expr = self.parse_expression()

        return IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr)

    def parse_module(self) -> Module:
        """Parse a module definition."""
        self.expect(TokenType.MODULE)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value

        # Parse parameters
        self.expect(TokenType.LPAREN)
        params = []
        while self.current_token().type != TokenType.RPAREN:
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            param_type = self.parse_type_annotation()
            params.append((param_name, param_type))

            if self.current_token().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        # Parse body
        body = []
        while self.current_token().type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        return Module(name, params, body)

    def parse_compose(self) -> Compose:
        """Parse a compose statement."""
        self.expect(TokenType.COMPOSE)
        self.expect(TokenType.LPAREN)

        modules = []
        while self.current_token().type != TokenType.RPAREN:
            modules.append(self.parse_expression())
            if self.current_token().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        return Compose(modules)

    def parse_link(self) -> 'Link':
        """
        Parse a link statement for dependency metadata.

        Syntax:
            link module_name
            link module_name { key: value, ... }
        """
        self.expect(TokenType.LINK)

        # Parse target identifier (not a full expression to avoid struct literal interpretation)
        target_token = self.expect(TokenType.IDENTIFIER)
        target = Identifier(target_token.value)

        # Parse optional metadata dict
        metadata = None
        if self.current_token().type == TokenType.LBRACE:
            self.advance()
            self.skip_newlines()

            metadata = {}
            while self.current_token().type != TokenType.RBRACE:
                self.skip_newlines()  # Skip newlines before each key

                # Check again for RBRACE after skipping newlines
                if self.current_token().type == TokenType.RBRACE:
                    break

                # Parse key
                key = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.COLON)

                # Parse value (can be any expression)
                value = self.parse_expression()

                metadata[key] = value

                if self.current_token().type == TokenType.COMMA:
                    self.advance()
                    self.skip_newlines()

            self.expect(TokenType.RBRACE)

        from morphogen.ast.nodes import Link
        return Link(target=target, metadata=metadata)

    def parse_use(self) -> 'Use':
        """
        Parse a use statement to import domain operators.

        Syntax:
            use field                    # Import single domain
            use field, agent, visual     # Multiple imports
            use field as f               # Aliased import (future)
        """
        self.expect(TokenType.USE)

        domains = []
        aliases = {}

        # Parse first domain
        domain = self.expect(TokenType.IDENTIFIER).value
        domains.append(domain)

        # Check for alias (future: use field as f)
        if (self.current_token().type == TokenType.IDENTIFIER and
            self.current_token().value == "as"):
            self.advance()  # Skip 'as'
            alias = self.expect(TokenType.IDENTIFIER).value
            aliases[domain] = alias

        # Parse additional domains (comma-separated)
        while self.current_token().type == TokenType.COMMA:
            self.advance()  # Skip comma
            self.skip_newlines()  # Allow newlines after comma

            domain = self.expect(TokenType.IDENTIFIER).value
            domains.append(domain)

            # Check for alias
            if (self.current_token().type == TokenType.IDENTIFIER and
                self.current_token().value == "as"):
                self.advance()  # Skip 'as'
                alias = self.expect(TokenType.IDENTIFIER).value
                aliases[domain] = alias

        from morphogen.ast.nodes import Use
        return Use(domains=domains, aliases=aliases)

    def parse_const_declaration(self, decorators: List[Decorator] = None) -> Assignment:
        """Parse a const declaration: const NAME : Type = value"""
        if decorators is None:
            decorators = []

        self.expect(TokenType.CONST)
        target = self.expect(TokenType.IDENTIFIER).value

        # Type annotation is required for const (based on examples)
        type_annotation = None
        if self.current_token().type == TokenType.COLON:
            self.advance()
            type_annotation = self.parse_type_annotation()

        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()

        return Assignment(target, value, type_annotation, decorators, is_const=True)

    def parse_assignment(self, decorators: List[Decorator] = None) -> Assignment:
        """Parse an assignment statement."""
        if decorators is None:
            decorators = []

        target = self.expect(TokenType.IDENTIFIER).value

        # Check for type annotation
        type_annotation = None
        if self.current_token().type == TokenType.COLON:
            self.advance()
            type_annotation = self.parse_type_annotation()

        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()

        return Assignment(target, value, type_annotation, decorators)

    def parse_unit_expression(self) -> str:
        """Parse physical unit expression like 'm/s' or 'kg*m/s^2'.

        Collects all tokens between [ and ] that form a unit expression.
        Supports operators: / (division), * (multiplication), ^ (exponentiation)
        and parentheses for grouping.
        """
        unit_tokens = []
        while self.current_token().type != TokenType.RBRACKET:
            token = self.current_token()
            # Collect identifiers (m, s, kg, etc.) and operators (/, *, ^)
            if token.type in [TokenType.IDENTIFIER, TokenType.SLASH,
                             TokenType.STAR, TokenType.NUMBER]:
                unit_tokens.append(str(token.value))
                self.advance()
            elif token.type == TokenType.IDENTIFIER and token.value == '^':
                # Handle caret as string if it comes as identifier
                unit_tokens.append('^')
                self.advance()
            elif token.type == TokenType.EOF:
                raise ParseError(f"Unexpected EOF while parsing unit expression at {token.line}:{token.column}")
            else:
                # Stop on unexpected token
                break

        # Join tokens into string: ["m", "/", "s"] → "m/s"
        return ''.join(unit_tokens)

    def parse_type_annotation(self) -> TypeAnnotation:
        """Parse a type annotation."""
        base_type = self.expect(TokenType.IDENTIFIER).value

        # Parse generic type parameters
        type_params = []
        if self.current_token().type == TokenType.LT:
            self.advance()
            while self.current_token().type != TokenType.GT:
                type_params.append(self.parse_type_annotation())
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
            self.expect(TokenType.GT)

        # Parse unit annotation
        unit = None
        if self.current_token().type == TokenType.LBRACKET:
            self.advance()
            unit = self.parse_unit_expression()
            self.expect(TokenType.RBRACKET)

        return TypeAnnotation(base_type, type_params, unit)

    def parse_type_definition(self) -> Statement:
        """Parse a type definition."""
        # This is a simplified version
        # Full implementation would handle record types
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        # Skip the type definition for now
        # Would need to parse record/struct definitions
        return None

    def parse_set_statement(self) -> Assignment:
        """Parse a set statement (configuration)."""
        self.expect(TokenType.SET)
        target = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        return Assignment(target, value)

    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_assignment_expression()

    def parse_assignment_expression(self) -> Expression:
        """Parse assignment or comparison expression."""
        return self.parse_comparison()

    def parse_comparison(self) -> Expression:
        """Parse comparison expression."""
        left = self.parse_additive()

        while self.current_token().type in [
            TokenType.EQ, TokenType.NE,
            TokenType.LT, TokenType.LE,
            TokenType.GT, TokenType.GE
        ]:
            op_token = self.advance()
            right = self.parse_additive()
            left = BinaryOp(left, op_token.value, right)

        return left

    def parse_additive(self) -> Expression:
        """Parse additive expression (+ -)."""
        left = self.parse_multiplicative()

        while self.current_token().type in [TokenType.PLUS, TokenType.MINUS]:
            op_token = self.advance()
            right = self.parse_multiplicative()
            left = BinaryOp(left, op_token.value, right)

        return left

    def parse_multiplicative(self) -> Expression:
        """Parse multiplicative expression (* / %)."""
        left = self.parse_unary()

        while self.current_token().type in [TokenType.STAR, TokenType.SLASH, TokenType.PERCENT]:
            op_token = self.advance()
            right = self.parse_unary()
            left = BinaryOp(left, op_token.value, right)

        return left

    def parse_unary(self) -> Expression:
        """Parse unary expression (- !)."""
        if self.current_token().type in [TokenType.MINUS]:
            op_token = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op_token.value, operand)

        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Parse postfix expression (function calls, field access, struct literals)."""
        expr = self.parse_primary()

        while True:
            token = self.current_token()

            # Struct literal: Identifier { field: value, ... }
            # Only parse as struct literal if expr is an Identifier
            # AND after { (possibly with newlines), we see field: pattern or }
            # This disambiguates from if/else block syntax: if cond { expr }
            if isinstance(expr, Identifier) and token.type == TokenType.LBRACE:
                # Look ahead past the { and any newlines to check the pattern
                # without modifying parser state
                lookahead_pos = self.pos + 1  # Position after '{'

                # Skip newlines in lookahead
                while lookahead_pos < len(self.tokens) and self.tokens[lookahead_pos].type == TokenType.NEWLINE:
                    lookahead_pos += 1

                # Check if it matches struct literal pattern
                if lookahead_pos < len(self.tokens):
                    next_token = self.tokens[lookahead_pos]
                    next_next_token = self.tokens[lookahead_pos + 1] if lookahead_pos + 1 < len(self.tokens) else self.tokens[-1]

                    is_struct_literal = (
                        next_token.type == TokenType.RBRACE or  # Empty struct
                        (next_token.type == TokenType.IDENTIFIER and
                         next_next_token.type == TokenType.COLON)
                    )
                else:
                    is_struct_literal = False

                if is_struct_literal:
                    struct_name = expr.name
                    self.advance()  # Skip '{'
                    self.skip_newlines()

                    field_values = {}
                    while self.current_token().type != TokenType.RBRACE:
                        # Parse field name
                        field_name = self.expect(TokenType.IDENTIFIER).value
                        self.expect(TokenType.COLON)

                        # Parse field value expression
                        field_value = self.parse_expression()
                        field_values[field_name] = field_value

                        # Handle comma or end of fields
                        if self.current_token().type == TokenType.COMMA:
                            self.advance()
                            self.skip_newlines()
                        elif self.current_token().type == TokenType.RBRACE:
                            break
                        else:
                            self.skip_newlines()
                            if self.current_token().type != TokenType.RBRACE:
                                raise ParseError(
                                    f"Expected COMMA or RBRACE in struct literal, got {self.current_token().type.name} "
                                    f"at {self.current_token().line}:{self.current_token().column}"
                                )

                    self.expect(TokenType.RBRACE)
                    expr = StructLiteral(struct_name=struct_name, field_values=field_values)
                else:
                    # Not a struct literal, { is not a valid postfix operator
                    break

            # Function call
            elif token.type == TokenType.LPAREN:
                self.advance()
                args = []
                kwargs = {}

                while self.current_token().type != TokenType.RPAREN:
                    # Check for keyword argument
                    if (self.current_token().type == TokenType.IDENTIFIER and
                        self.peek_token().type == TokenType.ASSIGN):
                        key = self.current_token().value
                        self.advance()
                        self.advance()  # Skip '='
                        value = self.parse_expression()
                        kwargs[key] = value
                    else:
                        args.append(self.parse_expression())

                    if self.current_token().type == TokenType.COMMA:
                        self.advance()

                self.expect(TokenType.RPAREN)
                expr = Call(expr, args, kwargs)

            # Field access
            elif token.type == TokenType.DOT:
                self.advance()
                field_name = self.expect(TokenType.IDENTIFIER).value
                expr = FieldAccess(expr, field_name)

            else:
                break

        return expr

    def parse_primary(self) -> Expression:
        """Parse primary expression (literals, identifiers, parenthesized)."""
        token = self.current_token()

        # Lambda expression: |args| expr
        if token.type == TokenType.PIPE:
            return self.parse_lambda()

        # If/else expression: if cond then expr1 else expr2 or if cond { expr1 } else { expr2 }
        if token.type == TokenType.IF:
            return self.parse_if_else()

        # Number literal
        if token.type == TokenType.NUMBER:
            self.advance()
            return Literal(token.value)

        # String literal
        if token.type == TokenType.STRING:
            self.advance()
            return Literal(token.value)

        # Boolean literal
        if token.type == TokenType.BOOL:
            self.advance()
            return Literal(token.value)

        # Identifier
        if token.type == TokenType.IDENTIFIER:
            self.advance()
            return Identifier(token.value)

        # Parenthesized expression or tuple
        if token.type == TokenType.LPAREN:
            self.advance()

            # Check for empty tuple
            if self.current_token().type == TokenType.RPAREN:
                self.advance()
                return Tuple([])

            # Parse first element
            first_expr = self.parse_expression()

            # Check if it's a tuple (has comma) or just parenthesized expression
            if self.current_token().type == TokenType.COMMA:
                # It's a tuple
                elements = [first_expr]
                while self.current_token().type == TokenType.COMMA:
                    self.advance()
                    # Allow trailing comma
                    if self.current_token().type == TokenType.RPAREN:
                        break
                    elements.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                return Tuple(elements)
            else:
                # Just a parenthesized expression
                self.expect(TokenType.RPAREN)
                return first_expr

        # List literal
        if token.type == TokenType.LBRACKET:
            self.advance()
            elements = []
            while self.current_token().type != TokenType.RBRACKET:
                elements.append(self.parse_expression())
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
            self.expect(TokenType.RBRACKET)
            # Return a list literal (would need to add ListLiteral to AST)
            return Literal(elements)

        raise ParseError(
            f"Unexpected token {token.type.name} at {token.line}:{token.column}"
        )


def parse(source: str) -> Program:
    """Parse source code into an AST."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
