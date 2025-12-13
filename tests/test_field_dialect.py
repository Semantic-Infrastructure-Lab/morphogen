"""Unit tests for Morphogen Field Dialect (Phase 2)

Tests the field dialect operations and type system.
These tests require MLIR Python bindings to be installed.
"""

import pytest
from morphogen.mlir.context import MorphogenMLIRContext, MLIR_AVAILABLE
from morphogen.mlir.dialects.field import (
    FieldType, FieldDialect,
    FieldCreateOp, FieldGradientOp, FieldLaplacianOp, FieldDiffuseOp
)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldType:
    """Tests for FieldType wrapper."""

    def test_field_type_creation_f32(self):
        """Test creating f32 field type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            field_type = FieldType.get(f32, ctx.ctx)

            assert field_type is not None
            type_str = str(field_type)
            assert "morphogen" in type_str
            assert "field" in type_str

    def test_field_type_creation_f64(self):
        """Test creating f64 field type."""
        from mlir import ir

        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f64 = ir.F64Type.get()
            field_type = FieldType.get(f64, ctx.ctx)

            assert field_type is not None
            type_str = str(field_type)
            assert "morphogen" in type_str
            assert "field" in type_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldCreateOp:
    """Tests for field creation operation."""

    def test_create_field_operation(self):
        """Test creating a field creation operation."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create constants
                width = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 256)
                ).result
                height = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 256)
                ).result
                fill = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result

                # Create field
                field = FieldCreateOp.create(
                    width, height, fill,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert field is not None

            # Verify module contains the operation
            module_str = str(module)
            assert "builtin.unrealized_conversion_cast" in module_str

    def test_create_field_with_different_fill_values(self):
        """Test creating fields with different fill values."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        fill_values = [0.0, 1.0, 42.5, -3.14]

        for fill_val in fill_values:
            with ctx.ctx, ir.Location.unknown():
                module = ctx.create_module("test")

                with ir.InsertionPoint(module.body):
                    width = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), 128)
                    ).result
                    height = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), 128)
                    ).result
                    fill = arith.ConstantOp(
                        ir.F32Type.get(),
                        ir.FloatAttr.get(ir.F32Type.get(), fill_val)
                    ).result

                    field = FieldCreateOp.create(
                        width, height, fill,
                        ir.F32Type.get(),
                        ir.Location.unknown(),
                        ir.InsertionPoint(module.body)
                    )

                    assert field is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldGradientOp:
    """Tests for gradient operation."""

    def test_gradient_operation(self):
        """Test creating a gradient operation."""
        from mlir import ir
        from mlir.dialects import arith, memref

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a dummy field (memref for lowering)
                memref_type = ir.MemRefType.get([256, 256], ir.F32Type.get())
                field = memref.AllocOp(memref_type, [], []).result

                # Compute gradient
                grad = FieldGradientOp.create(
                    field,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert grad is not None

            # Verify module
            module_str = str(module)
            assert "memref.alloc" in module_str
            assert "builtin.unrealized_conversion_cast" in module_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldLaplacianOp:
    """Tests for Laplacian operation."""

    def test_laplacian_operation(self):
        """Test creating a Laplacian operation."""
        from mlir import ir
        from mlir.dialects import arith, memref

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a dummy field
                memref_type = ir.MemRefType.get([256, 256], ir.F32Type.get())
                field = memref.AllocOp(memref_type, [], []).result

                # Compute Laplacian
                lapl = FieldLaplacianOp.create(
                    field,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert lapl is not None

            # Verify module
            module_str = str(module)
            assert "memref.alloc" in module_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldDiffuseOp:
    """Tests for diffusion operation."""

    def test_diffuse_operation(self):
        """Test creating a diffusion operation."""
        from mlir import ir
        from mlir.dialects import arith, memref

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a dummy field
                memref_type = ir.MemRefType.get([256, 256], ir.F32Type.get())
                field = memref.AllocOp(memref_type, [], []).result

                # Create parameters
                rate = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.1)
                ).result
                dt = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.01)
                ).result
                iterations = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 10)
                ).result

                # Apply diffusion
                diffused = FieldDiffuseOp.create(
                    field, rate, dt, iterations,
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                assert diffused is not None

            # Verify module
            module_str = str(module)
            assert "memref.alloc" in module_str
            assert "arith.constant" in module_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldDialect:
    """Tests for FieldDialect utilities."""

    def test_is_field_op(self):
        """Test checking if an operation is a field op."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                # Create a field operation
                width = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 256)
                ).result
                height = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 256)
                ).result
                fill = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 0.0)
                ).result

                field = FieldDialect.create(
                    width, height, fill,
                    ir.F32Type.get(),
                    ir.Location.unknown(),
                    ir.InsertionPoint(module.body)
                )

                # Check operations in module
                for op in module.body.operations:
                    if hasattr(op, 'attributes') and 'op_name' in op.attributes:
                        assert FieldDialect.is_field_op(op)
                        op_name = FieldDialect.get_field_op_name(op)
                        assert "morphogen.field." in op_name

    def test_dialect_namespace(self):
        """Test that FieldDialect provides all operations."""
        assert hasattr(FieldDialect, 'create')
        assert hasattr(FieldDialect, 'gradient')
        assert hasattr(FieldDialect, 'laplacian')
        assert hasattr(FieldDialect, 'diffuse')


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not installed")
class TestFieldDialectIntegration:
    """Integration tests for field dialect."""

    def test_chained_operations(self):
        """Test chaining multiple field operations."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                loc = ir.Location.unknown()
                ip = ir.InsertionPoint(module.body)

                # Create field
                width = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 128)
                ).result
                height = arith.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), 128)
                ).result
                fill = arith.ConstantOp(
                    ir.F32Type.get(),
                    ir.FloatAttr.get(ir.F32Type.get(), 1.0)
                ).result

                field = FieldDialect.create(width, height, fill, ir.F32Type.get(), loc, ip)

                # Compute gradient
                grad = FieldDialect.gradient(field, loc, ip)
                assert grad is not None

                # Compute Laplacian
                lapl = FieldDialect.laplacian(field, loc, ip)
                assert lapl is not None

            # Verify module structure
            module_str = str(module)
            print("\nChained operations module:")
            print(module_str)

    def test_multiple_fields(self):
        """Test creating multiple independent fields."""
        from mlir import ir
        from mlir.dialects import arith

        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test")

            with ir.InsertionPoint(module.body):
                loc = ir.Location.unknown()
                ip = ir.InsertionPoint(module.body)

                # Create multiple fields
                for i in range(3):
                    width = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), 64 + i * 64)
                    ).result
                    height = arith.ConstantOp(
                        ir.IndexType.get(),
                        ir.IntegerAttr.get(ir.IndexType.get(), 64 + i * 64)
                    ).result
                    fill = arith.ConstantOp(
                        ir.F32Type.get(),
                        ir.FloatAttr.get(ir.F32Type.get(), float(i))
                    ).result

                    field = FieldDialect.create(width, height, fill, ir.F32Type.get(), loc, ip)
                    assert field is not None

            # Verify all fields were created
            module_str = str(module)
            # Count constant ops (should have 9: 3 widths, 3 heights, 3 fills)
            assert module_str.count("arith.constant") >= 9


class TestFieldDialectAvailability:
    """Tests that work regardless of MLIR availability."""

    def test_mlir_available_flag(self):
        """Test that MLIR_AVAILABLE flag is set correctly."""
        from morphogen.mlir.dialects.field import MLIR_AVAILABLE as field_mlir_available
        assert isinstance(field_mlir_available, bool)

    @pytest.mark.skipif(MLIR_AVAILABLE, reason="Test for when MLIR is NOT installed")
    def test_operations_fail_without_mlir(self):
        """Test that operations raise RuntimeError without MLIR."""
        # FieldType.get should fail
        with pytest.raises(RuntimeError, match="MLIR not available"):
            FieldType.get(None, None)
