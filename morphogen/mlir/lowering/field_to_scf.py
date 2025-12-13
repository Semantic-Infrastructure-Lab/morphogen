"""Field-to-SCF Lowering Pass for Morphogen v0.7.0 Phase 2

This module implements the lowering pass that transforms Morphogen field operations
into Structured Control Flow (SCF) loops with memref operations.

Transformation:
    morphogen.field.* ops → scf.for loops + memref.load/store + arith ops

Example:
    Input (High-level):
        %grad = morphogen.field.gradient %field : !morphogen.field<f32>

    Output (Low-level):
        %mem = memref.alloc(%h, %w, %c2) : memref<?x?x2xf32>
        scf.for %i = %c1 to %h_minus_1 step %c1 {
          scf.for %j = %c1 to %w_minus_1 step %c1 {
            // Central difference stencil
            %dx = ...
            %dy = ...
            memref.store %dx, %mem[%i, %j, %c0]
            memref.store %dy, %mem[%i, %j, %c1]
          }
        }
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf, func
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class FieldToSCFPass:
    """Lowering pass: Field operations → SCF loops + memref.

    This pass traverses the MLIR module and replaces field operations
    with nested scf.for loops operating on memref storage.

    Operations Lowered:
        - morphogen.field.create → memref.alloc + initialization loop
        - morphogen.field.gradient → nested loops with central difference
        - morphogen.field.laplacian → nested loops with 5-point stencil
        - morphogen.field.diffuse → iteration loops with Jacobi solver

    Usage:
        >>> pass_obj = FieldToSCFPass(context)
        >>> pass_obj.run(module)
    """

    def __init__(self, context: MorphogenMLIRContext):
        """Initialize field-to-SCF pass.

        Args:
            context: Morphogen MLIR context
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        self.context = context

    def run(self, module: Any) -> None:
        """Run lowering pass on module.

        Args:
            module: MLIR module to transform (in-place)
        """
        with self.context.ctx:
            # Walk through all operations in the module
            for op in module.body.operations:
                self._process_operation(op)

    def _process_operation(self, op: Any) -> None:
        """Process a single operation recursively.

        Args:
            op: MLIR operation to process
        """
        # Import field dialect to check for field ops
        from ..dialects.field import FieldDialect

        # Check if this is a field operation
        if FieldDialect.is_field_op(op):
            op_name = FieldDialect.get_field_op_name(op)
            if op_name == "morphogen.field.create":
                self._lower_field_create(op)
            elif op_name == "morphogen.field.gradient":
                self._lower_field_gradient(op)
            elif op_name == "morphogen.field.laplacian":
                self._lower_field_laplacian(op)
            elif op_name == "morphogen.field.diffuse":
                self._lower_field_diffuse(op)

        # Recursively process nested regions
        if hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        self._process_operation(nested_op)

    def _lower_field_create(self, op: Any) -> None:
        """Lower morphogen.field.create to memref.alloc + initialization loop.

        Input:
            %field = morphogen.field.create %w, %h, %fill : !morphogen.field<f32>

        Output:
            %mem = memref.alloc(%h, %w) : memref<?x?xf32>
            scf.for %i = %c0 to %h step %c1 {
              scf.for %j = %c0 to %w step %c1 {
                memref.store %fill, %mem[%i, %j]
              }
            }
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands: width, height, fill_value
            operands = op.operands
            if len(operands) < 3:
                raise ValueError("field.create requires 3 operands")

            width = operands[0]
            height = operands[1]
            fill_value = operands[2]

            # Determine element type from fill_value
            element_type = fill_value.type

            # Create insertion point before the op
            with ir.InsertionPoint(op):
                # Allocate memref
                memref_type = ir.MemRefType.get([ir.ShapedType.get_dynamic_size(),
                                                   ir.ShapedType.get_dynamic_size()],
                                                  element_type)
                mem = memref.AllocOp(memref_type, [height, width], []).result

                # Create constants for loop bounds
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result

                # Create initialization loops
                # for i in range(height):
                for_i = scf.ForOp(c0, height, c1)
                with ir.InsertionPoint(for_i.body):
                    i = for_i.induction_variable

                    # for j in range(width):
                    for_j = scf.ForOp(c0, width, c1)
                    with ir.InsertionPoint(for_j.body):
                        j = for_j.induction_variable

                        # mem[i, j] = fill_value
                        memref.StoreOp(fill_value, mem, [i, j])

                        # Yield from inner loop
                        scf.YieldOp([])

                    # Yield from outer loop
                    scf.YieldOp([])

            # Replace uses of field op with memref
            op.results[0].replace_all_uses_with(mem)

            # Erase the original op
            op.operation.erase()

    def _lower_field_gradient(self, op: Any) -> None:
        """Lower morphogen.field.gradient to central difference stencil.

        Input:
            %grad = morphogen.field.gradient %field : !morphogen.field<f32>

        Output:
            %h = memref.dim %field, %c0
            %w = memref.dim %field, %c1
            %grad = memref.alloc(%h, %w, %c2) : memref<?x?x2xf32>

            scf.for %i = %c1 to %h_minus_1 step %c1 {
              scf.for %j = %c1 to %w_minus_1 step %c1 {
                // dx = (field[i+1,j] - field[i-1,j]) / 2
                // dy = (field[i,j+1] - field[i,j-1]) / 2
                %i_prev = arith.subi %i, %c1
                %i_next = arith.addi %i, %c1
                %val_prev = memref.load %field[%i_prev, %j]
                %val_next = memref.load %field[%i_next, %j]
                %dx = arith.subf %val_next, %val_prev
                %dx_scaled = arith.divf %dx, %c2_f32

                %j_prev = arith.subi %j, %c1
                %j_next = arith.addi %j, %c1
                %val_prev_y = memref.load %field[%i, %j_prev]
                %val_next_y = memref.load %field[%i, %j_next]
                %dy = arith.subf %val_next_y, %val_prev_y
                %dy_scaled = arith.divf %dy, %c2_f32

                memref.store %dx_scaled, %grad[%i, %j, %c0]
                memref.store %dy_scaled, %grad[%i, %j, %c1]
              }
            }
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operand: input field (memref)
            field = op.operands[0]

            # Infer element type
            if isinstance(field.type, ir.MemRefType):
                element_type = field.type.element_type
            else:
                element_type = ir.F32Type.get()

            with ir.InsertionPoint(op):
                # Get dimensions
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c2_idx = arith.ConstantOp(ir.IndexType.get(), 2).result

                h = memref.DimOp(field, c0).result
                w = memref.DimOp(field, c1).result

                # Allocate gradient field (height x width x 2)
                grad_memref_type = ir.MemRefType.get(
                    [ir.ShapedType.get_dynamic_size(),
                     ir.ShapedType.get_dynamic_size(),
                     2],
                    element_type
                )
                grad = memref.AllocOp(grad_memref_type, [h, w], []).result

                # Constants for stencil
                c2_f32 = arith.ConstantOp(element_type, 2.0).result

                # Compute loop bounds (exclude boundaries)
                # h_minus_1 = h - 1
                h_minus_1 = arith.SubIOp(h, c1).result

                # w_minus_1 = w - 1
                w_minus_1 = arith.SubIOp(w, c1).result

                # Nested loops: for i in [1, h-1):
                for_i = scf.ForOp(c1, h_minus_1, c1)
                with ir.InsertionPoint(for_i.body):
                    i = for_i.induction_variable

                    # for j in [1, w-1):
                    for_j = scf.ForOp(c1, w_minus_1, c1)
                    with ir.InsertionPoint(for_j.body):
                        j = for_j.induction_variable

                        # Compute gradient in x direction (dx)
                        i_prev = arith.SubIOp(i, c1).result
                        i_next = arith.AddIOp(i, c1).result
                        val_prev = memref.LoadOp(field, [i_prev, j]).result
                        val_next = memref.LoadOp(field, [i_next, j]).result
                        dx = arith.SubFOp(val_next, val_prev).result
                        dx_scaled = arith.DivFOp(dx, c2_f32).result

                        # Compute gradient in y direction (dy)
                        j_prev = arith.SubIOp(j, c1).result
                        j_next = arith.AddIOp(j, c1).result
                        val_prev_y = memref.LoadOp(field, [i, j_prev]).result
                        val_next_y = memref.LoadOp(field, [i, j_next]).result
                        dy = arith.SubFOp(val_next_y, val_prev_y).result
                        dy_scaled = arith.DivFOp(dy, c2_f32).result

                        # Store gradient components
                        memref.StoreOp(dx_scaled, grad, [i, j, c0])
                        memref.StoreOp(dy_scaled, grad, [i, j, c1])

                        # Yield from inner loop
                        scf.YieldOp([])

                    # Yield from outer loop
                    scf.YieldOp([])

            # Replace uses
            op.results[0].replace_all_uses_with(grad)
            op.operation.erase()

    def _lower_field_laplacian(self, op: Any) -> None:
        """Lower morphogen.field.laplacian to 5-point stencil.

        Input:
            %lapl = morphogen.field.laplacian %field : !morphogen.field<f32>

        Output:
            %h = memref.dim %field, %c0
            %w = memref.dim %field, %c1
            %lapl = memref.alloc(%h, %w) : memref<?x?xf32>

            scf.for %i = %c1 to %h_minus_1 step %c1 {
              scf.for %j = %c1 to %w_minus_1 step %c1 {
                // lapl[i,j] = field[i+1,j] + field[i-1,j] +
                //              field[i,j+1] + field[i,j-1] - 4*field[i,j]
                %center = memref.load %field[%i, %j]
                %up = memref.load %field[%i_prev, %j]
                %down = memref.load %field[%i_next, %j]
                %left = memref.load %field[%i, %j_prev]
                %right = memref.load %field[%i, %j_next]

                %sum = arith.addf %up, %down
                %sum2 = arith.addf %sum, %left
                %sum3 = arith.addf %sum2, %right
                %center_4 = arith.mulf %center, %c4_f32
                %lapl_val = arith.subf %sum3, %center_4

                memref.store %lapl_val, %lapl[%i, %j]
              }
            }
        """
        with self.context.ctx, ir.Location.unknown():
            field = op.operands[0]

            # Infer element type
            if isinstance(field.type, ir.MemRefType):
                element_type = field.type.element_type
            else:
                element_type = ir.F32Type.get()

            with ir.InsertionPoint(op):
                # Constants
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c4_f32 = arith.ConstantOp(element_type, 4.0).result

                # Get dimensions
                h = memref.DimOp(field, c0).result
                w = memref.DimOp(field, c1).result

                # Allocate result
                lapl_memref_type = ir.MemRefType.get(
                    [ir.ShapedType.get_dynamic_size(),
                     ir.ShapedType.get_dynamic_size()],
                    element_type
                )
                lapl = memref.AllocOp(lapl_memref_type, [h, w], []).result

                # Loop bounds
                h_minus_1 = arith.SubIOp(h, c1).result
                w_minus_1 = arith.SubIOp(w, c1).result

                # Nested loops
                for_i = scf.ForOp(c1, h_minus_1, c1)
                with ir.InsertionPoint(for_i.body):
                    i = for_i.induction_variable

                    for_j = scf.ForOp(c1, w_minus_1, c1)
                    with ir.InsertionPoint(for_j.body):
                        j = for_j.induction_variable

                        # Load 5-point stencil
                        i_prev = arith.SubIOp(i, c1).result
                        i_next = arith.AddIOp(i, c1).result
                        j_prev = arith.SubIOp(j, c1).result
                        j_next = arith.AddIOp(j, c1).result

                        center = memref.LoadOp(field, [i, j]).result
                        up = memref.LoadOp(field, [i_prev, j]).result
                        down = memref.LoadOp(field, [i_next, j]).result
                        left = memref.LoadOp(field, [i, j_prev]).result
                        right = memref.LoadOp(field, [i, j_next]).result

                        # Compute Laplacian: up + down + left + right - 4*center
                        sum1 = arith.AddFOp(up, down).result
                        sum2 = arith.AddFOp(sum1, left).result
                        sum3 = arith.AddFOp(sum2, right).result
                        center_4 = arith.MulFOp(center, c4_f32).result
                        lapl_val = arith.SubFOp(sum3, center_4).result

                        # Store result
                        memref.StoreOp(lapl_val, lapl, [i, j])
                        scf.YieldOp([])

                    scf.YieldOp([])

            # Replace uses
            op.results[0].replace_all_uses_with(lapl)
            op.operation.erase()

    def _lower_field_diffuse(self, op: Any) -> None:
        """Lower morphogen.field.diffuse to Jacobi iteration loops.

        Input:
            %diffused = morphogen.field.diffuse %field, %rate, %dt, %iters

        Output:
            // Allocate buffers
            %h = memref.dim %field, %c0
            %w = memref.dim %field, %c1
            %buf1 = memref.alloc(%h, %w) : memref<?x?xf32>
            %buf2 = memref.alloc(%h, %w) : memref<?x?xf32>

            // Copy input to buf1
            [copy loop]

            // Jacobi iterations
            scf.for %iter = %c0 to %iters step %c1 {
              scf.for %i = %c1 to %h_minus_1 step %c1 {
                scf.for %j = %c1 to %w_minus_1 step %c1 {
                  %lapl = [5-point stencil on buf1]
                  %val = memref.load %buf1[%i, %j]
                  %diff = arith.mulf %rate, %dt
                  %diff2 = arith.mulf %diff, %lapl
                  %new_val = arith.addf %val, %diff2
                  memref.store %new_val, %buf2[%i, %j]
                }
              }
              // Swap buffers (copy buf2 to buf1)
              [swap loop]
            }
        """
        with self.context.ctx, ir.Location.unknown():
            # Extract operands
            operands = op.operands
            if len(operands) < 4:
                raise ValueError("field.diffuse requires 4 operands")

            field = operands[0]
            rate = operands[1]
            dt = operands[2]
            iterations = operands[3]

            # Infer element type
            if isinstance(field.type, ir.MemRefType):
                element_type = field.type.element_type
            else:
                element_type = ir.F32Type.get()

            with ir.InsertionPoint(op):
                # Constants
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c4_f32 = arith.ConstantOp(element_type, 4.0).result

                # Get dimensions
                h = memref.DimOp(field, c0).result
                w = memref.DimOp(field, c1).result

                # Allocate double buffers
                memref_type = ir.MemRefType.get(
                    [ir.ShapedType.get_dynamic_size(),
                     ir.ShapedType.get_dynamic_size()],
                    element_type
                )
                buf1 = memref.AllocOp(memref_type, [h, w], []).result
                buf2 = memref.AllocOp(memref_type, [h, w], []).result

                # Copy input to buf1
                for_copy_i = scf.ForOp(c0, h, c1)
                with ir.InsertionPoint(for_copy_i.body):
                    i_copy = for_copy_i.induction_variable
                    for_copy_j = scf.ForOp(c0, w, c1)
                    with ir.InsertionPoint(for_copy_j.body):
                        j_copy = for_copy_j.induction_variable
                        val = memref.LoadOp(field, [i_copy, j_copy]).result
                        memref.StoreOp(val, buf1, [i_copy, j_copy])
                        scf.YieldOp([])
                    scf.YieldOp([])

                # Precompute rate * dt
                rate_dt = arith.MulFOp(rate, dt).result

                # Loop bounds
                h_minus_1 = arith.SubIOp(h, c1).result
                w_minus_1 = arith.SubIOp(w, c1).result

                # Jacobi iteration loop
                for_iter = scf.ForOp(c0, iterations, c1)
                with ir.InsertionPoint(for_iter.body):
                    # Spatial loops
                    for_i = scf.ForOp(c1, h_minus_1, c1)
                    with ir.InsertionPoint(for_i.body):
                        i = for_i.induction_variable

                        for_j = scf.ForOp(c1, w_minus_1, c1)
                        with ir.InsertionPoint(for_j.body):
                            j = for_j.induction_variable

                            # Compute Laplacian from buf1
                            i_prev = arith.SubIOp(i, c1).result
                            i_next = arith.AddIOp(i, c1).result
                            j_prev = arith.SubIOp(j, c1).result
                            j_next = arith.AddIOp(j, c1).result

                            center = memref.LoadOp(buf1, [i, j]).result
                            up = memref.LoadOp(buf1, [i_prev, j]).result
                            down = memref.LoadOp(buf1, [i_next, j]).result
                            left = memref.LoadOp(buf1, [i, j_prev]).result
                            right = memref.LoadOp(buf1, [i, j_next]).result

                            sum1 = arith.AddFOp(up, down).result
                            sum2 = arith.AddFOp(sum1, left).result
                            sum3 = arith.AddFOp(sum2, right).result
                            center_4 = arith.MulFOp(center, c4_f32).result
                            lapl = arith.SubFOp(sum3, center_4).result

                            # Update: new = center + rate * dt * lapl
                            diff = arith.MulFOp(rate_dt, lapl).result
                            new_val = arith.AddFOp(center, diff).result

                            # Store to buf2
                            memref.StoreOp(new_val, buf2, [i, j])
                            scf.YieldOp([])
                        scf.YieldOp([])

                    # Swap: copy buf2 to buf1
                    for_swap_i = scf.ForOp(c0, h, c1)
                    with ir.InsertionPoint(for_swap_i.body):
                        i_swap = for_swap_i.induction_variable
                        for_swap_j = scf.ForOp(c0, w, c1)
                        with ir.InsertionPoint(for_swap_j.body):
                            j_swap = for_swap_j.induction_variable
                            val_swap = memref.LoadOp(buf2, [i_swap, j_swap]).result
                            memref.StoreOp(val_swap, buf1, [i_swap, j_swap])
                            scf.YieldOp([])
                        scf.YieldOp([])

                    scf.YieldOp([])

            # Replace uses with buf1 (final result)
            op.results[0].replace_all_uses_with(buf1)
            op.operation.erase()


def create_field_to_scf_pass(context: MorphogenMLIRContext) -> FieldToSCFPass:
    """Factory function to create field-to-SCF lowering pass.

    Args:
        context: Morphogen MLIR context

    Returns:
        Configured FieldToSCFPass instance
    """
    return FieldToSCFPass(context)


# Export public API
__all__ = [
    "FieldToSCFPass",
    "create_field_to_scf_pass",
    "MLIR_AVAILABLE",
]
