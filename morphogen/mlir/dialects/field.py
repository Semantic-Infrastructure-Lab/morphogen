"""Morphogen Field Dialect (v0.7.0 Phase 2)

This module defines the Morphogen Field dialect for MLIR, providing high-level
operations for spatial field computations that lower to SCF loops and memref operations.

Status: Phase 2 Implementation (Months 4-6)

Operations:
- morphogen.field.create: Allocate a new field
- morphogen.field.gradient: Compute spatial gradient (central difference)
- morphogen.field.laplacian: Compute 5-point stencil Laplacian
- morphogen.field.diffuse: Apply Jacobi diffusion solver

Type System:
- !morphogen.field<T>: Generic field type (opaque for Phase 2)
"""

from __future__ import annotations
from typing import Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class FieldType:
    """Wrapper for !morphogen.field<T> type.

    In Phase 2, we use OpaqueType to represent custom field types.
    In Phase 3+, this will be replaced with proper IRDL dialect definition.

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> field_type = FieldType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(field_type)  # !morphogen.field<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get field type for given element type.

        Args:
            element_type: MLIR element type (e.g., F32Type, F64Type)
            context: MLIR context

        Returns:
            Opaque field type !morphogen.field<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        # Use OpaqueType for custom types in Phase 2
        # Format: !morphogen.field<element_type>
        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"field<{element_str}>", context=context)


class FieldCreateOp:
    """Operation: morphogen.field.create

    Creates a new field with specified dimensions and initial value.

    Syntax:
        %field = morphogen.field.create %width, %height, %fill : !morphogen.field<f32>

    Attributes:
        - width: Field width (index type)
        - height: Field height (index type)
        - fill_value: Initial fill value (element type)

    Results:
        - Field of type !morphogen.field<element_type>

    Lowering:
        Lowers to memref.alloc + initialization loops
    """

    @staticmethod
    def create(
        width: Any,  # ir.Value with index type
        height: Any,  # ir.Value with index type
        fill_value: Any,  # ir.Value with element type
        element_type: Any,  # MLIR type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a field creation operation.

        Args:
            width: Width dimension value
            height: Height dimension value
            fill_value: Fill value for initialization
            element_type: Element type (f32, f64, etc.)
            loc: Source location
            ip: Insertion point

        Returns:
            Field value representing the created field

        Note:
            This creates a custom operation using GenericOp for Phase 2.
            In Phase 3+, this will use proper dialect definition.
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # For Phase 2, we create a placeholder operation using builtin.unrealized_conversion_cast
            # This will be replaced with proper custom op in Phase 3

            # Create the field type
            field_type = FieldType.get(element_type, loc.context)

            # Create attributes for the operation
            attrs = {
                "width": width,
                "height": height,
                "fill_value": fill_value,
                "op_name": ir.StringAttr.get("morphogen.field.create", context=loc.context)
            }

            # Use unrealized_conversion_cast as a placeholder
            # This gets lowered properly in the field_to_scf pass
            result = builtin.UnrealizedConversionCastOp(
                [field_type],
                [width, height, fill_value]
            )

            return result.results[0]


class FieldGradientOp:
    """Operation: morphogen.field.gradient

    Computes spatial gradient using central differences.

    Syntax:
        %grad = morphogen.field.gradient %field : !morphogen.field<f32> -> !morphogen.field<vector<2xf32>>

    Arguments:
        - field: Input field

    Results:
        - Gradient field (2-channel: dx, dy)

    Lowering:
        Lowers to nested scf.for loops with central difference stencil:

        grad[i,j] = [
            (field[i+1,j] - field[i-1,j]) / 2,
            (field[i,j+1] - field[i,j-1]) / 2
        ]
    """

    @staticmethod
    def create(
        field: Any,  # ir.Value with field type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a gradient operation.

        Args:
            field: Input field value
            loc: Source location
            ip: Insertion point

        Returns:
            Gradient field (2-channel vector field)
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is a vector field (2 channels for dx, dy)
            # For Phase 2, we use a placeholder with 2-channel marker
            f32 = ir.F32Type.get()
            grad_type = FieldType.get(f32, loc.context)

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [grad_type],
                [field]
            )

            # Add attribute to mark as gradient op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.field.gradient", context=loc.context
            )
            result.operation.attributes["channels"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), 2
            )

            return result.results[0]


class FieldLaplacianOp:
    """Operation: morphogen.field.laplacian

    Computes 5-point stencil Laplacian: ∇²f.

    Syntax:
        %lapl = morphogen.field.laplacian %field : !morphogen.field<f32>

    Arguments:
        - field: Input field

    Results:
        - Laplacian field (same type as input)

    Lowering:
        Lowers to nested scf.for loops with 5-point stencil:

        lapl[i,j] = (field[i+1,j] + field[i-1,j] +
                     field[i,j+1] + field[i,j-1] -
                     4*field[i,j])
    """

    @staticmethod
    def create(
        field: Any,  # ir.Value with field type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a Laplacian operation.

        Args:
            field: Input field value
            loc: Source location
            ip: Insertion point

        Returns:
            Laplacian field
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same type as input
            field_type = field.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [field_type],
                [field]
            )

            # Add attribute to mark as laplacian op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.field.laplacian", context=loc.context
            )

            return result.results[0]


class FieldDiffuseOp:
    """Operation: morphogen.field.diffuse

    Applies Jacobi diffusion solver for heat equation.

    Syntax:
        %diffused = morphogen.field.diffuse %field, %rate, %dt, %iters : !morphogen.field<f32>

    Arguments:
        - field: Input field
        - rate: Diffusion rate coefficient (f32)
        - dt: Time step (f32)
        - iterations: Number of Jacobi iterations (i32)

    Results:
        - Diffused field

    Lowering:
        Lowers to nested iteration loops with Jacobi stencil:

        for iter in range(iterations):
            for i, j in field:
                new[i,j] = field[i,j] + rate * dt * laplacian[i,j]
            swap(field, new)
    """

    @staticmethod
    def create(
        field: Any,  # ir.Value with field type
        rate: Any,  # ir.Value with f32 type
        dt: Any,  # ir.Value with f32 type
        iterations: Any,  # ir.Value with i32 type
        loc: Any,  # ir.Location
        ip: Any  # ir.InsertionPoint
    ) -> Any:  # ir.OpResult
        """Create a diffusion operation.

        Args:
            field: Input field value
            rate: Diffusion rate
            dt: Time step
            iterations: Number of iterations
            loc: Source location
            ip: Insertion point

        Returns:
            Diffused field
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Result is same type as input
            field_type = field.type

            # Create placeholder op
            result = builtin.UnrealizedConversionCastOp(
                [field_type],
                [field, rate, dt, iterations]
            )

            # Add attribute to mark as diffuse op
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.field.diffuse", context=loc.context
            )

            return result.results[0]


class FieldDialect:
    """Field operations dialect.

    This class serves as a namespace for field dialect operations
    and provides utility methods for working with field types.

    Operations:
        - create: Allocate new field
        - gradient: Compute gradient
        - laplacian: Compute Laplacian
        - diffuse: Apply diffusion

    Example:
        >>> from morphogen.mlir.dialects.field import FieldDialect
        >>>
        >>> # Create a field
        >>> width = arith.ConstantOp(ir.IndexType.get(), 256)
        >>> height = arith.ConstantOp(ir.IndexType.get(), 256)
        >>> fill = arith.ConstantOp(ir.F32Type.get(), 0.0)
        >>> field = FieldDialect.create(width, height, fill, f32, loc, ip)
        >>>
        >>> # Compute gradient
        >>> grad = FieldDialect.gradient(field, loc, ip)
    """

    create = FieldCreateOp.create
    gradient = FieldGradientOp.create
    laplacian = FieldLaplacianOp.create
    diffuse = FieldDiffuseOp.create

    @staticmethod
    def is_field_op(op: Any) -> bool:
        """Check if an operation is a field operation.

        Args:
            op: MLIR operation to check

        Returns:
            True if op is a field operation
        """
        if not MLIR_AVAILABLE:
            return False

        # Check if op has our marker attribute
        if not hasattr(op, "attributes"):
            return False

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return False

        op_name = str(op_name_attr)
        return "morphogen.field." in op_name

    @staticmethod
    def get_field_op_name(op: Any) -> Optional[str]:
        """Get the field operation name.

        Args:
            op: Field operation

        Returns:
            Operation name (e.g., "morphogen.field.gradient") or None
        """
        if not MLIR_AVAILABLE:
            return None

        if not hasattr(op, "attributes"):
            return None

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return None

        return str(op_name_attr).strip('"')


# Export public API
__all__ = [
    "FieldType",
    "FieldCreateOp",
    "FieldGradientOp",
    "FieldLaplacianOp",
    "FieldDiffuseOp",
    "FieldDialect",
    "MLIR_AVAILABLE",
]
