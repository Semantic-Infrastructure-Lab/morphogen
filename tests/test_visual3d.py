"""Tests for visual3d module - 3D visualization capabilities."""

import pytest
import numpy as np


class TestVisual3DTypes:
    """Test core 3D visualization types."""

    def test_camera_creation(self):
        """Test Camera3D creation."""
        from morphogen.stdlib import visual3d

        cam = visual3d.camera(
            position=(100, 100, 50),
            focal_point=(0, 0, 0),
            up_vector=(0, 0, 1),
            fov=30.0
        )

        assert cam.position == (100, 100, 50)
        assert cam.focal_point == (0, 0, 0)
        assert cam.up_vector == (0, 0, 1)
        assert cam.fov == 30.0

    def test_mesh_creation(self):
        """Test Visual3D mesh creation from vertices/faces."""
        from morphogen.stdlib import visual3d

        # Simple triangle
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        mesh = visual3d.mesh(vertices, faces)

        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)

    def test_mesh_with_scalars(self):
        """Test Visual3D mesh with scalar coloring."""
        from morphogen.stdlib import visual3d

        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        scalars = np.array([0.0, 0.5, 1.0], dtype=np.float32)

        mesh = visual3d.mesh(vertices, faces, scalars=scalars, colormap="viridis")

        assert mesh.scalars is not None
        assert len(mesh.scalars) == 3
        assert mesh.colormap == "viridis"

    def test_mesh_with_color(self):
        """Test Visual3D mesh with solid color."""
        from morphogen.stdlib import visual3d

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)

        mesh = visual3d.mesh(vertices, color=(1.0, 0.0, 0.0))

        assert mesh.color == (1.0, 0.0, 0.0)


class TestMeshFromField:
    """Test heightmap to mesh conversion."""

    def test_mesh_from_field_basic(self):
        """Test basic mesh creation from heightmap."""
        from morphogen.stdlib import field, visual3d

        heightmap = field.random((32, 32), seed=42)
        mesh = visual3d.mesh_from_field(heightmap, scale_z=5.0)

        assert mesh.n_vertices == 32 * 32
        assert mesh.n_faces > 0
        assert mesh.scalars is not None  # Color by height enabled by default

    def test_mesh_from_field_scales(self):
        """Test mesh scaling factors."""
        from morphogen.stdlib import field, visual3d

        heightmap = field.random((16, 16), seed=42)
        mesh = visual3d.mesh_from_field(
            heightmap,
            scale_x=2.0,
            scale_y=3.0,
            scale_z=10.0
        )

        # Verify scales are applied
        x_range = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
        y_range = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()

        # Should be approximately (size-1) * scale
        assert abs(x_range - 15 * 2.0) < 0.01
        assert abs(y_range - 15 * 3.0) < 0.01


class TestOrbitCamera:
    """Test orbital camera path generation."""

    def test_orbit_camera_basic(self):
        """Test basic orbit camera generation."""
        from morphogen.stdlib import visual3d

        cameras = visual3d.orbit_camera(
            center=(0, 0, 0),
            radius=10,
            frames=12
        )

        assert len(cameras) == 12
        for cam in cameras:
            assert cam.focal_point == (0.0, 0.0, 0.0)

    def test_orbit_camera_radius(self):
        """Test orbit camera maintains radius."""
        from morphogen.stdlib import visual3d

        radius = 10.0
        cameras = visual3d.orbit_camera(
            center=(0, 0, 0),
            radius=radius,
            elevation=0,  # Horizontal orbit
            frames=8
        )

        for cam in cameras:
            pos = np.array(cam.position)
            distance = np.linalg.norm(pos)
            assert abs(distance - radius) < 0.01


class TestScene3D:
    """Test Scene3D container."""

    def test_scene_creation(self):
        """Test Scene3D creation."""
        from morphogen.stdlib import visual3d, Scene3D

        scene = Scene3D()
        assert len(scene.objects) == 0
        assert scene.camera is None

    def test_scene_add_object(self):
        """Test adding objects to scene."""
        from morphogen.stdlib import visual3d, Scene3D

        scene = Scene3D()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
        mesh = visual3d.mesh(vertices)

        scene.add(mesh)
        assert len(scene.objects) == 1


class TestPrimitives:
    """Test primitive shape generators (require PyVista)."""

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_sphere_creation(self):
        """Test sphere primitive creation."""
        from morphogen.stdlib import visual3d

        sphere = visual3d.sphere(center=(0, 0, 0), radius=1.0)

        assert sphere.n_vertices > 0
        assert sphere.n_faces > 0

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_box_creation(self):
        """Test box primitive creation."""
        from morphogen.stdlib import visual3d

        box = visual3d.box(bounds=(-1, 1, -1, 1, -1, 1))

        assert box.n_vertices > 0
        assert box.n_faces > 0


class TestIsosurface:
    """Test isosurface extraction (requires PyVista)."""

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_isosurface_sphere(self):
        """Test isosurface extraction from sphere SDF."""
        from morphogen.stdlib import visual3d

        # Create sphere SDF volume
        x, y, z = np.mgrid[-2:2:32j, -2:2:32j, -2:2:32j]
        volume = np.sqrt(x**2 + y**2 + z**2)

        # Extract isosurface at radius 1.5
        mesh = visual3d.isosurface(volume, isovalue=1.5)

        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_isosurface_with_spacing(self):
        """Test isosurface with custom grid spacing."""
        from morphogen.stdlib import visual3d

        x, y, z = np.mgrid[-2:2:32j, -2:2:32j, -2:2:32j]
        volume = np.sqrt(x**2 + y**2 + z**2)

        mesh = visual3d.isosurface(volume, isovalue=1.5, spacing=(0.5, 0.5, 0.5))

        assert mesh.n_vertices > 0

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_isosurface_invalid_value(self):
        """Test isosurface raises error for invalid isovalue."""
        from morphogen.stdlib import visual3d

        # Values from 0 to 1
        volume = np.random.rand(16, 16, 16).astype(np.float32)

        # Isovalue outside range should raise
        with pytest.raises(ValueError, match="No isosurface found"):
            visual3d.isosurface(volume, isovalue=5.0)


class TestParametricSurface:
    """Test parametric surface generation."""

    def test_parametric_sphere(self):
        """Test parametric sphere creation."""
        from morphogen.stdlib import visual3d

        def sphere(u, v):
            theta = u * 2 * np.pi
            phi = v * np.pi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            return x, y, z

        mesh = visual3d.parametric_surface(
            sphere,
            u_range=(0, 1),
            v_range=(0, 1),
            u_resolution=32,
            v_resolution=32
        )

        assert mesh.n_vertices == 32 * 32
        assert mesh.n_faces > 0

    def test_parametric_torus(self):
        """Test parametric torus creation."""
        from morphogen.stdlib import visual3d

        def torus(u, v, R=2, r=0.5):
            theta = u * 2 * np.pi
            phi = v * 2 * np.pi
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            return x, y, z

        mesh = visual3d.parametric_surface(torus, u_resolution=48, v_resolution=24)

        assert mesh.n_vertices == 48 * 24
        assert mesh.scalars is not None  # Default coloring by v

    def test_parametric_color_by(self):
        """Test parametric surface color_by options."""
        from morphogen.stdlib import visual3d

        def plane(u, v):
            return u, v, u * v

        # Color by u
        mesh_u = visual3d.parametric_surface(plane, color_by="u")
        assert mesh_u.scalars is not None

        # Color by z
        mesh_z = visual3d.parametric_surface(plane, color_by="z")
        assert mesh_z.scalars is not None

        # No coloring
        mesh_none = visual3d.parametric_surface(plane, color_by="none")
        assert mesh_none.scalars is None


class TestLighting:
    """Test light source configuration."""

    def test_light_creation(self):
        """Test Light3D creation via light()."""
        from morphogen.stdlib import visual3d

        # Point light
        point = visual3d.light(
            position=(10, 10, 10),
            color=(1.0, 0.9, 0.8),
            intensity=1.5,
            light_type="point"
        )

        assert point.position == (10, 10, 10)
        assert point.color == (1.0, 0.9, 0.8)
        assert point.intensity == 1.5
        assert point.light_type == "point"

    def test_light_types(self):
        """Test different light types."""
        from morphogen.stdlib import visual3d

        directional = visual3d.light(light_type="directional")
        assert directional.light_type == "directional"

        ambient = visual3d.light(light_type="ambient", intensity=0.3)
        assert ambient.light_type == "ambient"
        assert ambient.intensity == 0.3


class TestRendering:
    """Test 3D rendering (requires PyVista)."""

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_render_basic(self):
        """Test basic 3D rendering."""
        from morphogen.stdlib import field, visual3d

        heightmap = field.random((32, 32), seed=42)
        mesh = visual3d.mesh_from_field(heightmap, scale_z=5.0)
        camera = visual3d.camera((50, 50, 30), (16, 16, 0))

        image = visual3d.render_3d(
            mesh,
            camera=camera,
            width=400,
            height=300
        )

        assert image.shape == (300, 400)
        assert image.data.shape == (300, 400, 3)

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_render_multiple_objects(self):
        """Test rendering multiple objects."""
        from morphogen.stdlib import field, visual3d

        heightmap = field.random((32, 32), seed=42)
        terrain = visual3d.mesh_from_field(heightmap, scale_z=5.0)
        sphere = visual3d.sphere(center=(16, 16, 10), radius=3.0, color=(1, 0, 0))

        camera = visual3d.camera((50, 50, 30), (16, 16, 0))

        image = visual3d.render_3d(
            terrain,
            sphere,
            camera=camera,
            width=400,
            height=300
        )

        assert image.shape == (300, 400)

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_render_with_lights(self):
        """Test rendering with custom lighting."""
        from morphogen.stdlib import field, visual3d

        heightmap = field.random((32, 32), seed=42)
        mesh = visual3d.mesh_from_field(heightmap, scale_z=5.0)
        camera = visual3d.camera((50, 50, 30), (16, 16, 0))

        # Custom lights
        lights = [
            visual3d.light((50, 50, 50), color=(1.0, 0.9, 0.8), light_type="point"),
            visual3d.light(color=(0.2, 0.2, 0.3), intensity=0.5, light_type="ambient"),
        ]

        image = visual3d.render_3d(
            mesh,
            camera=camera,
            lights=lights,
            width=400,
            height=300
        )

        assert image.shape == (300, 400)

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_render_isosurface(self):
        """Test rendering an isosurface."""
        from morphogen.stdlib import visual3d

        # Create sphere SDF
        x, y, z = np.mgrid[-2:2:32j, -2:2:32j, -2:2:32j]
        volume = np.sqrt(x**2 + y**2 + z**2)

        mesh = visual3d.isosurface(volume, isovalue=1.5, color=(0.2, 0.6, 1.0))
        camera = visual3d.camera((5, 5, 3), (0, 0, 0))

        image = visual3d.render_3d(mesh, camera=camera, width=400, height=300)

        assert image.shape == (300, 400)

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_render_parametric(self):
        """Test rendering a parametric surface."""
        from morphogen.stdlib import visual3d

        def torus(u, v, R=2, r=0.5):
            theta = u * 2 * np.pi
            phi = v * 2 * np.pi
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            return x, y, z

        mesh = visual3d.parametric_surface(torus, u_resolution=32, v_resolution=16)
        camera = visual3d.camera((5, 5, 3), (0, 0, 0))

        image = visual3d.render_3d(mesh, camera=camera, width=400, height=300)

        assert image.shape == (300, 400)


class TestMolecularVisualization:
    """Test molecular 3D visualization."""

    def test_molecule_ball_and_stick(self):
        """Test ball-and-stick molecular visualization."""
        from morphogen.stdlib import visual3d
        from morphogen.stdlib.molecular import Atom, Bond, Molecule

        # Create water molecule
        atoms = [
            Atom.from_element('O'),
            Atom.from_element('H'),
            Atom.from_element('H'),
        ]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ])
        bonds = [Bond(0, 1, 1.0), Bond(0, 2, 1.0)]
        mol = Molecule(atoms=atoms, bonds=bonds, positions=positions)

        vis = visual3d.molecule(mol, style='ball_and_stick')

        assert vis.n_vertices > 0
        assert vis.n_faces > 0
        assert vis.point_colors is not None
        assert vis.point_colors.shape[1] == 3  # RGB colors

    def test_molecule_spacefill(self):
        """Test spacefill molecular visualization."""
        from morphogen.stdlib import visual3d
        from morphogen.stdlib.molecular import Atom, Bond, Molecule

        # Create methane molecule
        atoms = [
            Atom.from_element('C'),
            Atom.from_element('H'),
            Atom.from_element('H'),
            Atom.from_element('H'),
            Atom.from_element('H'),
        ]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ])
        bonds = [Bond(0, i, 1.0) for i in range(1, 5)]
        mol = Molecule(atoms=atoms, bonds=bonds, positions=positions)

        # Spacefill should have larger atoms (no bonds visible)
        vis = visual3d.molecule(mol, style='spacefill')

        assert vis.n_vertices > 0
        # Should be more vertices than ball_and_stick due to larger spheres

    def test_molecule_color_by_element(self):
        """Test coloring by element type."""
        from morphogen.stdlib import visual3d
        from morphogen.stdlib.molecular import Atom, Bond, Molecule

        atoms = [Atom.from_element('O'), Atom.from_element('H')]
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bonds = [Bond(0, 1, 1.0)]
        mol = Molecule(atoms=atoms, bonds=bonds, positions=positions)

        vis = visual3d.molecule(mol, color_by='element')

        assert vis.point_colors is not None
        # Verify colors are present (oxygen=red, hydrogen=white)

    def test_molecule_color_by_charge(self):
        """Test coloring by atomic charge."""
        from morphogen.stdlib import visual3d
        from morphogen.stdlib.molecular import Atom, Bond, Molecule

        atoms = [
            Atom.from_element('O', charge=-0.5),
            Atom.from_element('H', charge=0.25),
            Atom.from_element('H', charge=0.25),
        ]
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.3, 0.95, 0.0]])
        bonds = [Bond(0, 1, 1.0), Bond(0, 2, 1.0)]
        mol = Molecule(atoms=atoms, bonds=bonds, positions=positions)

        vis = visual3d.molecule(mol, color_by='charge')

        assert vis.point_colors is not None

    def test_molecule_empty(self):
        """Test empty molecule handling."""
        from morphogen.stdlib import visual3d
        from morphogen.stdlib.molecular import Molecule

        mol = Molecule(atoms=[], bonds=[], positions=np.zeros((0, 3)))

        vis = visual3d.molecule(mol)

        assert vis.n_vertices == 0


class TestOrbitalVisualization:
    """Test orbital isosurface visualization."""

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_orbital_basic(self):
        """Test basic orbital isosurface extraction."""
        from morphogen.stdlib import visual3d

        # Create simple orbital-like field (positive and negative regions)
        x, y, z = np.mgrid[-3:3:32j, -3:3:32j, -3:3:32j]
        # 2p-like orbital: positive on one side, negative on other
        orbital_field = x * np.exp(-np.sqrt(x**2 + y**2 + z**2))

        vis = visual3d.orbital(orbital_field, isovalues=(0.1, -0.1))

        assert vis.n_vertices > 0
        # Should have both positive and negative lobes

    @pytest.mark.skipif(
        not pytest.importorskip("pyvista", reason="PyVista not installed"),
        reason="PyVista not installed"
    )
    def test_orbital_custom_colors(self):
        """Test orbital with custom colors."""
        from morphogen.stdlib import visual3d

        x, y, z = np.mgrid[-3:3:24j, -3:3:24j, -3:3:24j]
        orbital_field = x * np.exp(-np.sqrt(x**2 + y**2 + z**2))

        vis = visual3d.orbital(
            orbital_field,
            isovalues=(0.08, -0.08),
            positive_color=(0.0, 0.5, 1.0),
            negative_color=(1.0, 0.5, 0.0),
            opacity=0.8
        )

        assert vis.opacity == 0.8


class TestFlyPath:
    """Test camera fly path generation."""

    def test_fly_path_linear(self):
        """Test linear camera path interpolation."""
        from morphogen.stdlib import visual3d

        waypoints = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
        ]

        cameras = visual3d.fly_path(waypoints, frames=10, interpolation="linear")

        assert len(cameras) == 10
        # First position should be at start
        assert np.allclose(cameras[0].position, waypoints[0], atol=0.01)
        # Last position should be at end
        assert np.allclose(cameras[-1].position, waypoints[-1], atol=0.01)

    def test_fly_path_spline(self):
        """Test spline camera path interpolation."""
        from morphogen.stdlib import visual3d

        waypoints = [
            (0.0, 0.0, 5.0),
            (10.0, 0.0, 5.0),
            (10.0, 10.0, 5.0),
            (0.0, 10.0, 5.0),
        ]

        cameras = visual3d.fly_path(waypoints, frames=30, interpolation="spline")

        assert len(cameras) == 30
        # Spline should start and end at waypoints
        assert np.allclose(cameras[0].position, waypoints[0], atol=0.01)
        assert np.allclose(cameras[-1].position, waypoints[-1], atol=0.01)

    def test_fly_path_look_at(self):
        """Test camera path with fixed look_at point."""
        from morphogen.stdlib import visual3d

        waypoints = [(10, 0, 5), (0, 10, 5), (-10, 0, 5)]
        look_at = (0, 0, 0)

        cameras = visual3d.fly_path(
            waypoints, frames=15, look_at=look_at, interpolation="linear"
        )

        assert len(cameras) == 15
        # All cameras should look at origin
        for cam in cameras:
            assert cam.focal_point == (0, 0, 0)

    def test_fly_path_look_ahead(self):
        """Test camera path with look-ahead."""
        from morphogen.stdlib import visual3d

        waypoints = [(0, 0, 5), (10, 0, 5), (10, 10, 5)]

        cameras = visual3d.fly_path(
            waypoints, frames=10, look_ahead=True, interpolation="linear"
        )

        assert len(cameras) == 10
        # Each camera (except last) should look toward next position

    def test_fly_path_too_few_waypoints(self):
        """Test error handling for too few waypoints."""
        from morphogen.stdlib import visual3d

        with pytest.raises(ValueError, match="Need at least 2 waypoints"):
            visual3d.fly_path([(0, 0, 0)], frames=10)
