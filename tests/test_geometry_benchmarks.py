"""Performance benchmarks for geometry domain operations.

These tests use pytest-benchmark to measure and track performance of critical
geometry operations, especially computationally expensive algorithms like
Delaunay triangulation, Voronoi diagrams, and mesh operations.

Run with: pytest tests/test_geometry_benchmarks.py --benchmark-only

Requires the ``pytest-benchmark`` plugin (part of the ``dev`` extra:
``pip install -e ".[dev]"``). When it is absent — e.g. a bare
``pip install -e .`` — the whole module skips cleanly instead of erroring at
``benchmark`` fixture setup, so the default test run stays green out of the box.
"""

import pytest

pytest.importorskip(
    "pytest_benchmark",
    reason="pytest-benchmark not installed (install the 'dev' extra to run benchmarks)",
)

import numpy as np
from morphogen.stdlib.geometry import (
    # Primitives
    point2d, circle, polygon, regular_polygon,
    box3d, sphere, point3d,
    # Advanced algorithms
    convex_hull, delaunay_triangulation, voronoi_diagram,
    collision_mesh,
)


@pytest.mark.benchmark
class TestDelaunayTriangulationBenchmarks:
    """Benchmarks for Delaunay triangulation - critical performance tests."""

    def test_benchmark_delaunay_small(self, benchmark):
        """Benchmark Delaunay triangulation (10 points)."""
        np.random.seed(42)
        points = np.random.rand(10, 2) * 100.0
        result = benchmark(delaunay_triangulation, points)
        assert hasattr(result, 'vertices')
        assert hasattr(result, 'faces')

    def test_benchmark_delaunay_medium(self, benchmark):
        """Benchmark Delaunay triangulation (50 points)."""
        np.random.seed(42)
        points = np.random.rand(50, 2) * 100.0
        result = benchmark(delaunay_triangulation, points)
        assert len(result.vertices) == 50

    def test_benchmark_delaunay_large(self, benchmark):
        """Benchmark Delaunay triangulation (100 points)."""
        np.random.seed(42)
        points = np.random.rand(100, 2) * 100.0
        result = benchmark(delaunay_triangulation, points)
        assert len(result.vertices) == 100

    def test_benchmark_delaunay_very_large(self, benchmark):
        """Benchmark Delaunay triangulation (500 points)."""
        np.random.seed(42)
        points = np.random.rand(500, 2) * 100.0
        result = benchmark(delaunay_triangulation, points)
        assert len(result.vertices) == 500

    @pytest.mark.slow
    def test_benchmark_delaunay_huge(self, benchmark):
        """Benchmark Delaunay triangulation (1000 points) - slow test."""
        np.random.seed(42)
        points = np.random.rand(1000, 2) * 100.0
        result = benchmark(delaunay_triangulation, points)
        assert len(result.vertices) == 1000


@pytest.mark.benchmark
class TestVoronoiDiagramBenchmarks:
    """Benchmarks for Voronoi diagram generation - critical performance tests."""

    def test_benchmark_voronoi_small(self, benchmark):
        """Benchmark Voronoi diagram (10 points)."""
        np.random.seed(42)
        points = np.random.rand(10, 2) * 100.0
        result = benchmark(voronoi_diagram, points)
        assert 'vertices' in result
        assert 'regions' in result

    def test_benchmark_voronoi_medium(self, benchmark):
        """Benchmark Voronoi diagram (50 points)."""
        np.random.seed(42)
        points = np.random.rand(50, 2) * 100.0
        result = benchmark(voronoi_diagram, points)
        assert len(result['regions']) > 0

    def test_benchmark_voronoi_large(self, benchmark):
        """Benchmark Voronoi diagram (100 points)."""
        np.random.seed(42)
        points = np.random.rand(100, 2) * 100.0
        result = benchmark(voronoi_diagram, points)
        assert len(result['regions']) > 0

    def test_benchmark_voronoi_very_large(self, benchmark):
        """Benchmark Voronoi diagram (500 points)."""
        np.random.seed(42)
        points = np.random.rand(500, 2) * 100.0
        result = benchmark(voronoi_diagram, points)
        assert len(result['regions']) > 0

    @pytest.mark.slow
    def test_benchmark_voronoi_huge(self, benchmark):
        """Benchmark Voronoi diagram (1000 points) - slow test."""
        np.random.seed(42)
        points = np.random.rand(1000, 2) * 100.0
        result = benchmark(voronoi_diagram, points)
        assert len(result['regions']) > 0


@pytest.mark.benchmark
class TestConvexHullBenchmarks:
    """Benchmarks for convex hull computation."""

    def test_benchmark_convex_hull_2d_small(self, benchmark):
        """Benchmark 2D convex hull (20 points)."""
        np.random.seed(42)
        points = np.random.rand(20, 2) * 100.0
        result = benchmark(convex_hull, points, dim=2)
        assert hasattr(result, 'vertices')

    def test_benchmark_convex_hull_2d_medium(self, benchmark):
        """Benchmark 2D convex hull (100 points)."""
        np.random.seed(42)
        points = np.random.rand(100, 2) * 100.0
        result = benchmark(convex_hull, points, dim=2)
        assert len(result.vertices) > 0

    def test_benchmark_convex_hull_2d_large(self, benchmark):
        """Benchmark 2D convex hull (500 points)."""
        np.random.seed(42)
        points = np.random.rand(500, 2) * 100.0
        result = benchmark(convex_hull, points, dim=2)
        assert len(result.vertices) > 0

    def test_benchmark_convex_hull_3d_small(self, benchmark):
        """Benchmark 3D convex hull (20 points)."""
        np.random.seed(42)
        points = np.random.rand(20, 3) * 100.0
        result = benchmark(convex_hull, points, dim=3)
        assert hasattr(result, 'vertices')
        assert hasattr(result, 'faces')

    def test_benchmark_convex_hull_3d_medium(self, benchmark):
        """Benchmark 3D convex hull (100 points)."""
        np.random.seed(42)
        points = np.random.rand(100, 3) * 100.0
        result = benchmark(convex_hull, points, dim=3)
        assert len(result.vertices) > 0

    def test_benchmark_convex_hull_3d_large(self, benchmark):
        """Benchmark 3D convex hull (500 points)."""
        np.random.seed(42)
        points = np.random.rand(500, 3) * 100.0
        result = benchmark(convex_hull, points, dim=3)
        assert len(result.vertices) > 0


@pytest.mark.benchmark
class TestCollisionMeshBenchmarks:
    """Benchmarks for collision mesh simplification."""

    # Note: collision_mesh currently only works with Mesh objects
    # Box3D and Sphere would need to be converted to meshes first

    def test_benchmark_collision_mesh_complex(self, benchmark):
        """Benchmark collision mesh generation for complex hull."""
        # Create a complex point cloud
        np.random.seed(42)
        points = np.random.randn(100, 3) * 5.0
        hull = convex_hull(points, dim=3)

        # Benchmark simplification
        result = benchmark(collision_mesh, hull, target_faces=30)
        assert hasattr(result, 'vertices')
        assert hasattr(result, 'faces')

    def test_benchmark_collision_mesh_high_detail(self, benchmark):
        """Benchmark high-detail collision mesh generation."""
        np.random.seed(42)
        points = np.random.randn(200, 3) * 10.0
        hull = convex_hull(points, dim=3)

        # Request high detail
        result = benchmark(collision_mesh, hull, target_faces=100)
        assert len(result.faces) > 0


@pytest.mark.benchmark
class TestPolygonOperationBenchmarks:
    """Benchmarks for common polygon operations."""

    def test_benchmark_regular_polygon_construction(self, benchmark):
        """Benchmark regular polygon construction."""
        def construct_polygons():
            # Create 100 polygons
            polygons = []
            for i in range(100):
                poly = regular_polygon(
                    center=point2d(float(i), float(i)),
                    radius=5.0,
                    num_sides=6
                )
                polygons.append(poly)
            return polygons

        result = benchmark(construct_polygons)
        assert len(result) == 100

    def test_benchmark_large_polygon_area(self, benchmark):
        """Benchmark area computation for large polygon."""
        from morphogen.stdlib.geometry import area

        # Create a large polygon
        theta = np.linspace(0, 2 * np.pi, 1000)
        vertices = np.column_stack([np.cos(theta), np.sin(theta)]) * 100
        poly = polygon(vertices)

        result = benchmark(area, poly)
        assert result > 0
