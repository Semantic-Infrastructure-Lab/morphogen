"""Simplified MLIR-like IR Builder for Morphogen

This module provides a simplified intermediate representation that mimics
MLIR's structure and semantics, allowing us to develop and test the compilation
pipeline without requiring the full LLVM/MLIR build infrastructure.

The generated IR is textual and follows MLIR conventions, making it easy to
later replace with real MLIR bindings.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class IRType(Enum):
    """IR type enumeration."""
    F32 = "f32"
    F64 = "f64"
    I32 = "i32"
    I64 = "i64"
    I1 = "i1"  # Boolean
    INDEX = "index"


@dataclass
class IRValue:
    """Represents an SSA value in IR.

    In SSA (Single Static Assignment), each value is assigned exactly once
    and represents the result of an operation.
    """
    name: str  # SSA value name (e.g., %0, %result)
    type: Union[IRType, str]  # Value type

    def __str__(self) -> str:
        type_str = self.type.value if isinstance(self.type, IRType) else self.type
        return f"{self.name} : {type_str}"


@dataclass
class IROperation:
    """Represents an operation in IR.

    Operations are the fundamental unit of IR, producing zero or more results
    and having zero or more operands.
    """
    opcode: str  # Operation name (e.g., "arith.addf", "func.call")
    operands: List[IRValue] = field(default_factory=list)  # Input values
    results: List[IRValue] = field(default_factory=list)  # Output values
    attributes: Dict[str, Any] = field(default_factory=dict)  # Attributes
    regions: List['IRRegion'] = field(default_factory=list)  # Nested regions

    def __str__(self) -> str:
        """Generate MLIR-like textual representation."""
        parts = []

        # Results (if any)
        if self.results:
            result_strs = [r.name for r in self.results]
            parts.append(f"{', '.join(result_strs)} = ")

        # Opcode
        parts.append(self.opcode)

        # Operands
        if self.operands:
            operand_strs = [o.name for o in self.operands]
            parts.append(f"({', '.join(operand_strs)})")

        # Attributes
        if self.attributes:
            attr_strs = []
            for key, value in self.attributes.items():
                if isinstance(value, str):
                    attr_strs.append(f'{key}="{value}"')
                else:
                    attr_strs.append(f'{key}={value}')
            parts.append(f" {{ {', '.join(attr_strs)} }}")

        # Type signature
        if self.results:
            type_strs = [r.type.value if isinstance(r.type, IRType) else r.type
                        for r in self.results]
            parts.append(f" : {', '.join(type_strs)}")

        # Regions (blocks)
        region_str = ""
        if self.regions:
            for region in self.regions:
                region_str += f"\n{region}"

        return ''.join(parts) + region_str


@dataclass
class IRBlock:
    """Represents a basic block (sequence of operations)."""
    label: str  # Block label
    args: List[IRValue] = field(default_factory=list)  # Block arguments
    operations: List[IROperation] = field(default_factory=list)  # Operations in block

    def __str__(self) -> str:
        lines = []

        # Block label with arguments
        if self.args:
            arg_strs = [str(a) for a in self.args]
            lines.append(f"{self.label}({', '.join(arg_strs)}):")
        else:
            lines.append(f"{self.label}:")

        # Operations (indented)
        for op in self.operations:
            lines.append(f"  {op}")

        return '\n'.join(lines)


@dataclass
class IRRegion:
    """Represents a region (collection of blocks)."""
    blocks: List[IRBlock] = field(default_factory=list)

    def __str__(self) -> str:
        return '\n'.join(str(block) for block in self.blocks)


@dataclass
class IRFunction:
    """Represents a function in IR."""
    name: str  # Function name (e.g., @add)
    args: List[IRValue]  # Function arguments
    return_types: List[Union[IRType, str]]  # Return types
    body: IRRegion  # Function body (regions and blocks)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Generate MLIR-like function representation."""
        # Function signature
        arg_strs = [str(arg) for arg in self.args]
        ret_type_strs = [t.value if isinstance(t, IRType) else t for t in self.return_types]

        sig = f"func.func @{self.name}({', '.join(arg_strs)})"
        if ret_type_strs:
            sig += f" -> {', '.join(ret_type_strs)}"

        # Attributes
        if self.attributes:
            attr_strs = [f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                        for k, v in self.attributes.items()]
            sig += f" attributes {{ {', '.join(attr_strs)} }}"

        # Body
        lines = [sig + " {"]

        # Indent body
        body_lines = str(self.body).split('\n')
        for line in body_lines:
            lines.append(f"  {line}")

        lines.append("}")

        return '\n'.join(lines)


@dataclass
class IRModule:
    """Represents a complete IR module."""
    functions: List[IRFunction] = field(default_factory=list)
    structs: List[str] = field(default_factory=list)  # Struct definitions

    def __str__(self) -> str:
        """Generate MLIR-like module representation."""
        lines = ["module {"]

        # Struct definitions (if any)
        for struct_def in self.structs:
            lines.append(f"  {struct_def}")

        # Functions
        for func in self.functions:
            func_lines = str(func).split('\n')
            for line in func_lines:
                lines.append(f"  {line}")
            lines.append("")  # Blank line between functions

        lines.append("}")

        return '\n'.join(lines)

    def verify(self) -> bool:
        """Verify IR module is well-formed.

        Returns:
            True if module is valid

        Raises:
            ValueError: If module has errors
        """
        # Basic verification (can be extended)
        # Allow modules with only struct definitions (no functions required)
        if not self.functions and not self.structs:
            raise ValueError("Module is empty (no functions or structs)")

        # Check each function has at least one block
        for func in self.functions:
            if not func.body.blocks:
                raise ValueError(f"Function @{func.name} has no blocks")

        return True


class IRForLoop:
    """Helper for constructing scf.for loops.

    This class manages the construction of for loops with iteration arguments,
    providing a simple interface for loop body construction and yielding.
    """

    def __init__(self, builder: 'IRBuilder', start: IRValue, end: IRValue,
                 step: IRValue, iter_args: List[IRValue], iter_types: List[Union[IRType, str]]):
        """Initialize for loop builder.

        Args:
            builder: Parent IR builder
            start: Loop start bound
            end: Loop end bound
            step: Loop step
            iter_args: Initial iteration argument values
            iter_types: Types of iteration arguments
        """
        self.builder = builder
        self.start = start
        self.end = end
        self.step = step
        self.iter_args = iter_args
        self.iter_types = iter_types
        self.result_values: List[IRValue] = []
        self.loop_var: Optional[IRValue] = None
        self.body_block: Optional[IRBlock] = None
        self.iter_arg_values: List[IRValue] = []  # Block arguments for iteration values

    def __enter__(self) -> 'IRForLoop':
        """Enter loop context - sets up loop header and body block."""
        # Create result values for the loop
        self.result_values = [self.builder.create_value(t) for t in self.iter_types]

        # Create loop variable (induction variable)
        self.loop_var = self.builder.create_value(IRType.INDEX)

        # Create body block with iteration arguments
        self.iter_arg_values = [self.builder.create_value(t) for t in self.iter_types]

        # Save current block
        self.saved_block = self.builder.current_block

        # Create loop body block (simulated - operations go to parent block)
        self.body_block = IRBlock(label="loop_body", args=[self.loop_var] + self.iter_arg_values)
        self.builder.current_block = self.body_block

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit loop context - generates the scf.for operation."""
        # Restore original block
        self.builder.current_block = self.saved_block

        # Generate scf.for operation with the body
        # Note: This is a simplified representation
        # In real MLIR, this would use regions and blocks properly
        if self.result_values:
            # Format: %results = scf.for %iv = %start to %end step %step iter_args(...) -> (...) { body }
            result_names = ', '.join(r.name for r in self.result_values)
            iter_arg_names = ', '.join(a.name for a in self.iter_args)
            type_strs = ', '.join(t.value if isinstance(t, IRType) else str(t) for t in self.iter_types)

            op_str = f"{result_names} = scf.for {self.loop_var.name} = {self.start.name} to {self.end.name} step {self.step.name}"
            if self.iter_args:
                op_str += f" iter_args({iter_arg_names}) -> ({type_strs})"

            # Create the operation with the body block embedded
            op = IROperation(
                opcode="scf.for",
                operands=[self.start, self.end, self.step] + self.iter_args,
                results=self.result_values,
                attributes={
                    "loop_var": self.loop_var.name,
                    "body_operations": self.body_block.operations
                }
            )

            if self.builder.current_block:
                self.builder.current_block.operations.append(op)

    def yield_values(self, values: List[IRValue]):
        """Yield values at end of loop iteration.

        Args:
            values: Values to yield (must match iter_types)
        """
        # Create scf.yield operation
        op = IROperation(
            opcode="scf.yield",
            operands=values,
            results=[],
            attributes={}
        )

        if self.body_block:
            self.body_block.operations.append(op)


class IRBuilder:
    """Builder for constructing IR operations and blocks.

    Provides a convenient API for building IR similar to MLIR's IRBuilder.
    """

    def __init__(self):
        """Initialize IR builder."""
        self.module = IRModule()
        self.current_function: Optional[IRFunction] = None
        self.current_block: Optional[IRBlock] = None
        self.value_counter = 0  # For generating unique SSA names

    def create_function(self, name: str, args: List[IRValue],
                       return_types: List[Union[IRType, str]]) -> IRFunction:
        """Create a new function.

        Args:
            name: Function name
            args: Function arguments
            return_types: Return types

        Returns:
            Created function
        """
        func = IRFunction(name=name, args=args, return_types=return_types,
                         body=IRRegion())
        self.module.functions.append(func)
        self.current_function = func
        return func

    def create_block(self, label: str = "entry",
                    args: Optional[List[IRValue]] = None) -> IRBlock:
        """Create a new basic block.

        Args:
            label: Block label
            args: Block arguments (for control flow)

        Returns:
            Created block
        """
        block = IRBlock(label=label, args=args or [])
        if self.current_function:
            self.current_function.body.blocks.append(block)
        self.current_block = block
        return block

    def create_value(self, type: Union[IRType, str],
                    name: Optional[str] = None) -> IRValue:
        """Create a new SSA value.

        Args:
            type: Value type
            name: Optional name (auto-generated if None)

        Returns:
            Created value
        """
        if name is None:
            name = f"%{self.value_counter}"
            self.value_counter += 1
        return IRValue(name=name, type=type)

    def add_operation(self, opcode: str, operands: Optional[List[IRValue]] = None,
                     result_types: Optional[List[Union[IRType, str]]] = None,
                     attributes: Optional[Dict[str, Any]] = None) -> List[IRValue]:
        """Add an operation to the current block.

        Args:
            opcode: Operation name
            operands: Input operands
            result_types: Types of results
            attributes: Operation attributes

        Returns:
            List of result values
        """
        operands = operands or []
        result_types = result_types or []
        attributes = attributes or {}

        # Create result values
        results = [self.create_value(t) for t in result_types]

        # Create operation
        op = IROperation(opcode=opcode, operands=operands, results=results,
                        attributes=attributes)

        # Add to current block
        if self.current_block:
            self.current_block.operations.append(op)

        return results

    def create_for_loop(self, start: IRValue, end: IRValue, step: IRValue,
                       iter_args: Optional[List[IRValue]] = None,
                       iter_types: Optional[List[Union[IRType, str]]] = None) -> 'IRForLoop':
        """Create an scf.for loop structure.

        Args:
            start: Loop start bound (index type)
            end: Loop end bound (index type)
            step: Loop step (index type)
            iter_args: Initial values for iteration arguments (optional)
            iter_types: Types of iteration arguments (optional)

        Returns:
            IRForLoop context manager for building loop body

        Example:
            loop = builder.create_for_loop(start, end, step, [init_val], [IRType.F32])
            # Add operations to loop.body_block
            # End with yield operation
        """
        return IRForLoop(self, start, end, step, iter_args or [], iter_types or [])

    def get_module(self) -> IRModule:
        """Get the constructed IR module.

        Returns:
            IR module
        """
        return self.module
