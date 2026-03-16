"""Lexical analysis for Creative Computation DSL."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto


class TokenType(Enum):
    """Token types for the DSL."""
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOL = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()
    STEP = auto()
    SUBSTEP = auto()
    FLOW = auto()
    FN = auto()
    STRUCT = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    RETURN = auto()
    MODULE = auto()
    COMPOSE = auto()
    SET = auto()
    TYPE = auto()
    PROFILE = auto()
    SOLVER = auto()
    ITERATE = auto()
    LINK = auto()
    USE = auto()
    CONST = auto()
    OUTPUT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()
    ARROW = auto()
    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()

    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()

    # Special
    AT = auto()  # For decorators
    HASH = auto()  # For comments
    PIPE = auto()  # For lambdas |args| expr
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    """A lexical token."""
    type: TokenType
    value: any
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class Lexer:
    """Lexical analyzer for the DSL."""

    _TWO_CHAR_OPS = {
        ('=', '='): (TokenType.EQ, '=='),
        ('!', '='): (TokenType.NE, '!='),
        ('<', '='): (TokenType.LE, '<='),
        ('>', '='): (TokenType.GE, '>='),
        ('-', '>'): (TokenType.ARROW, '->'),
    }

    _SINGLE_CHAR_OPS = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.STAR,
        '/': TokenType.SLASH,
        '%': TokenType.PERCENT,
        '=': TokenType.ASSIGN,
        '<': TokenType.LT,
        '>': TokenType.GT,
        '.': TokenType.DOT,
        ',': TokenType.COMMA,
        ':': TokenType.COLON,
        ';': TokenType.SEMICOLON,
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE,
        '[': TokenType.LBRACKET,
        ']': TokenType.RBRACKET,
        '@': TokenType.AT,
        '|': TokenType.PIPE,
    }

    KEYWORDS = {
        "step": TokenType.STEP,
        "substep": TokenType.SUBSTEP,
        "flow": TokenType.FLOW,
        "fn": TokenType.FN,
        "struct": TokenType.STRUCT,
        "if": TokenType.IF,
        "then": TokenType.THEN,
        "else": TokenType.ELSE,
        "return": TokenType.RETURN,
        "module": TokenType.MODULE,
        "compose": TokenType.COMPOSE,
        "set": TokenType.SET,
        "type": TokenType.TYPE,
        "profile": TokenType.PROFILE,
        "solver": TokenType.SOLVER,
        "iterate": TokenType.ITERATE,
        "link": TokenType.LINK,
        "use": TokenType.USE,
        "const": TokenType.CONST,
        "output": TokenType.OUTPUT,
        "true": TokenType.BOOL,
        "false": TokenType.BOOL,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def current_char(self) -> Optional[str]:
        """Get the current character."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at a character ahead."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]

    def advance(self) -> Optional[str]:
        """Advance to the next character."""
        if self.pos >= len(self.source):
            return None
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def skip_whitespace(self):
        """Skip whitespace (except newlines)."""
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()

    def skip_comment(self):
        """Skip a comment line."""
        if self.current_char() == '#':
            while self.current_char() and self.current_char() != '\n':
                self.advance()

    def read_number(self) -> Token:
        """Read a numeric literal."""
        start_line = self.line
        start_column = self.column
        num_str = ""

        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            num_str += self.current_char()
            self.advance()

        # Check for scientific notation
        if self.current_char() and self.current_char() in 'eE':
            num_str += self.current_char()
            self.advance()
            if self.current_char() and self.current_char() in '+-':
                num_str += self.current_char()
                self.advance()
            while self.current_char() and self.current_char().isdigit():
                num_str += self.current_char()
                self.advance()

        # Parse the number
        if '.' in num_str or 'e' in num_str or 'E' in num_str:
            value = float(num_str)
        else:
            value = int(num_str)

        return Token(TokenType.NUMBER, value, start_line, start_column)

    def read_string(self) -> Token:
        """Read a string literal."""
        start_line = self.line
        start_column = self.column
        quote_char = self.current_char()
        self.advance()  # Skip opening quote

        string_value = ""
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                # Handle escape sequences
                escape_char = self.current_char()
                if escape_char == 'n':
                    string_value += '\n'
                elif escape_char == 't':
                    string_value += '\t'
                elif escape_char == '\\':
                    string_value += '\\'
                elif escape_char == quote_char:
                    string_value += quote_char
                else:
                    string_value += escape_char
                self.advance()
            else:
                string_value += self.current_char()
                self.advance()

        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, string_value, start_line, start_column)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_line = self.line
        start_column = self.column
        identifier = ""

        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            identifier += self.current_char()
            self.advance()

        # Check if it's a keyword
        token_type = self.KEYWORDS.get(identifier, TokenType.IDENTIFIER)

        # Special handling for boolean literals
        if token_type == TokenType.BOOL:
            value = identifier == "true"
        else:
            value = identifier

        return Token(token_type, value, start_line, start_column)

    def _scan_operator(self, char: str, line: int, column: int) -> Token:
        """Scan a one- or two-character operator/delimiter token."""
        two_char = self._TWO_CHAR_OPS.get((char, self.peek_char()))
        if two_char:
            tok_type, lexeme = two_char
            self.advance()
            self.advance()
            return Token(tok_type, lexeme, line, column)
        if char in self._SINGLE_CHAR_OPS:
            self.advance()
            return Token(self._SINGLE_CHAR_OPS[char], char, line, column)
        raise ValueError(f"Unexpected character '{char}' at {line}:{column}")

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source."""
        while self.current_char():
            self.skip_whitespace()
            if not self.current_char():
                break
            char = self.current_char()
            if char == '#':
                self.skip_comment()
            elif char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
            elif char.isdigit():
                self.tokens.append(self.read_number())
            elif char in '"\'':
                self.tokens.append(self.read_string())
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            else:
                self.tokens.append(self._scan_operator(char, self.line, self.column))

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens
