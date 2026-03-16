"""Tests for fluid_jet domain operators."""

import pytest
import numpy as np
from morphogen.stdlib import fluid_jet


class TestJetCreation:
    """Test jet construction operators."""

    def test_jet_from_tube_basic(self):
        """Test basic jet creation from tube exit conditions.

        Expected velocity: v = m_dot / (rho * A)
        rho = 101325/(287*300) ≈ 1.177 kg/m³
        A = π*(0.025)² ≈ 1.963e-3 m²
        v ≈ 0.1 / (1.177 * 1.963e-3) ≈ 43.3 m/s
        """
        jet = fluid_jet.jet_from_tube(
            tube_diameter=0.05,  # 50mm
            tube_position=(0.0, 0.0, 0.0),
            tube_direction=(0.0, 0.0, 1.0),
            m_dot=0.1,  # 0.1 kg/s
            T_out=300.0,  # 300 K
        )

        assert jet is not None
        assert hasattr(jet, 'position')
        assert hasattr(jet, 'velocity')
        rho_expected = 101325.0 / (287.0 * 300.0)
        area_expected = np.pi * (0.025)**2
        v_expected = 0.1 / (rho_expected * area_expected)
        assert jet.velocity == pytest.approx(v_expected, rel=0.01), (
            f"velocity={jet.velocity:.3f} m/s, expected {v_expected:.3f} m/s")

    def test_jet_from_tube_with_density(self):
        """Test jet creation with explicit density."""
        jet = fluid_jet.jet_from_tube(
            tube_diameter=0.05,
            tube_position=(0.0, 0.0, 0.0),
            tube_direction=(0.0, 0.0, 1.0),
            m_dot=0.1,
            T_out=300.0,
            rho=1.2  # kg/m³
        )

        assert jet is not None
        assert jet.velocity > 0.0

    def test_create_jet_array_radial(self):
        """Test radial jet array creation."""
        jet_array = fluid_jet.create_jet_array_radial(
            n_jets=6,
            radius=0.1,  # 100mm
            jet_diameter=0.01,  # 10mm
            m_dot_per_jet=0.01,
            temperature=350.0,
            height=0.0
        )

        assert jet_array is not None
        assert hasattr(jet_array, 'jets')
        assert len(jet_array.jets) == 6

    def test_create_jet_array_angled(self):
        """Test radial jet array with inward angle."""
        jet_array = fluid_jet.create_jet_array_radial(
            n_jets=8,
            radius=0.15,
            jet_diameter=0.01,
            m_dot_per_jet=0.02,
            temperature=400.0,
            height=0.05,
            angle_inward=15.0  # degrees
        )

        assert jet_array is not None
        assert len(jet_array.jets) == 8


class TestJetQueries:
    """Test jet analysis and query operators."""

    def setup_method(self):
        """Create a test jet for query operations."""
        self.jet = fluid_jet.jet_from_tube(
            tube_diameter=0.05,
            tube_position=(0.0, 0.0, 0.0),
            tube_direction=(0.0, 0.0, 1.0),
            m_dot=0.1,
            T_out=300.0
        )

    def test_jet_reynolds(self):
        """Test Reynolds number calculation: Re = rho*v*D/mu.

        Jet: D=0.05m, m_dot=0.1 kg/s, T=300K
        rho = 101325/(287*300) ≈ 1.177 kg/m³
        A = pi*(0.025)^2 ≈ 1.963e-3 m²
        v = 0.1/(1.177*1.963e-3) ≈ 43.3 m/s
        Re = 1.177 * 43.3 * 0.05 / 1.8e-5 ≈ 141,400
        """
        re = fluid_jet.jet_reynolds(self.jet)

        assert re > 0.0
        rho = 101325.0 / (287.0 * self.jet.temperature)
        expected_re = rho * self.jet.velocity * self.jet.diameter / 1.8e-5
        assert re == pytest.approx(expected_re, rel=0.01), (
            f"Re={re:.0f}, expected {expected_re:.0f}")
        assert re > 2300, f"Expected turbulent flow (Re>2300), got Re={re:.0f}"

    def test_jet_reynolds_custom_viscosity(self):
        """Test Reynolds number with custom viscosity."""
        re_default = fluid_jet.jet_reynolds(self.jet)
        re_high_visc = fluid_jet.jet_reynolds(self.jet, mu=5e-5)

        # Higher viscosity should give lower Reynolds number
        assert re_high_visc < re_default

    def test_jet_centerline_velocity(self):
        """Test centerline velocity decay calculation."""
        v0 = fluid_jet.jet_centerline_velocity(self.jet, distance=0.0)
        v1 = fluid_jet.jet_centerline_velocity(self.jet, distance=0.5)
        v2 = fluid_jet.jet_centerline_velocity(self.jet, distance=1.0)

        # Velocity should decay with distance
        assert v0 > v1 > v2 > 0.0

    def test_jet_spreading_width(self):
        """Test jet spreading width calculation."""
        w0 = fluid_jet.jet_spreading_width(self.jet, distance=0.0)
        w1 = fluid_jet.jet_spreading_width(self.jet, distance=0.5)
        w2 = fluid_jet.jet_spreading_width(self.jet, distance=1.0)

        # Jet should spread with distance
        assert 0.0 < w0 < w1 < w2

    def test_jet_spreading_custom_rate(self):
        """Test jet spreading with custom spreading rate."""
        w_normal = fluid_jet.jet_spreading_width(self.jet, distance=1.0)
        w_fast = fluid_jet.jet_spreading_width(self.jet, distance=1.0, spreading_rate=0.2)

        # Faster spreading rate should give wider jet
        assert w_fast > w_normal

    def test_jet_entrainment(self):
        """Test entrainment rate calculation."""
        entrainment = fluid_jet.jet_entrainment(
            self.jet,
            plume_velocity=1.0,
            plume_density=1.2
        )

        assert entrainment >= 0.0

    def test_jet_entrainment_low_velocity(self):
        """Test entrainment with low plume velocity."""
        e_high = fluid_jet.jet_entrainment(
            self.jet,
            plume_velocity=5.0,
            plume_density=1.2
        )
        e_low = fluid_jet.jet_entrainment(
            self.jet,
            plume_velocity=0.5,
            plume_density=1.2
        )

        # Higher relative velocity should increase entrainment
        assert e_high != e_low


class TestJetFieldVisualization:
    """Test 2D field visualization of jets."""

    def test_jet_field_2d_single_jet(self):
        """Test 2D field generation for single jet."""
        jet_array = fluid_jet.create_jet_array_radial(
            n_jets=1,
            radius=0.0,
            jet_diameter=0.01,
            m_dot_per_jet=0.05,
            temperature=350.0
        )

        field = fluid_jet.jet_field_2d(
            jet_array,
            grid_size=(64, 64),
            grid_bounds=(-0.5, 0.5, -0.5, 0.5)
        )

        assert field.shape == (64, 64, 2)  # 2D vector field
        assert field.dtype == np.float64

    def test_jet_field_2d_multiple_jets(self):
        """Test 2D field with multiple jets."""
        jet_array = fluid_jet.create_jet_array_radial(
            n_jets=6,
            radius=0.2,
            jet_diameter=0.01,
            m_dot_per_jet=0.02,
            temperature=400.0
        )

        field = fluid_jet.jet_field_2d(
            jet_array,
            grid_size=(128, 128),
            grid_bounds=(-0.5, 0.5, -0.5, 0.5),
            decay=0.05
        )

        assert field.shape == (128, 128, 2)
        # Field should not be all zeros
        assert np.any(field != 0.0)

    def test_jet_field_2d_custom_decay(self):
        """Test field decay parameter effect."""
        jet_array = fluid_jet.create_jet_array_radial(
            n_jets=4,
            radius=0.15,
            jet_diameter=0.01,
            m_dot_per_jet=0.03,
            temperature=350.0
        )

        field_slow = fluid_jet.jet_field_2d(
            jet_array,
            grid_size=(64, 64),
            grid_bounds=(-0.5, 0.5, -0.5, 0.5),
            decay=0.01
        )

        field_fast = fluid_jet.jet_field_2d(
            jet_array,
            grid_size=(64, 64),
            grid_bounds=(-0.5, 0.5, -0.5, 0.5),
            decay=0.5
        )

        # Different decay rates should produce different fields
        assert not np.allclose(field_slow, field_fast)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_mass_flow(self):
        """Test behavior with zero mass flow rate."""
        jet = fluid_jet.jet_from_tube(
            tube_diameter=0.05,
            tube_position=(0.0, 0.0, 0.0),
            tube_direction=(0.0, 0.0, 1.0),
            m_dot=0.0,
            T_out=300.0
        )

        assert jet is not None
        # Zero mass flow should give zero velocity
        assert jet.velocity == 0.0

    def test_small_jet_array(self):
        """Test minimum jet array size."""
        jet_array = fluid_jet.create_jet_array_radial(
            n_jets=1,
            radius=0.0,
            jet_diameter=0.001,
            m_dot_per_jet=0.001,
            temperature=300.0
        )

        assert jet_array is not None
        assert len(jet_array.jets) == 1

    def test_large_distance_queries(self):
        """Test queries at large distances from jet."""
        jet = fluid_jet.jet_from_tube(
            tube_diameter=0.05,
            tube_position=(0.0, 0.0, 0.0),
            tube_direction=(0.0, 0.0, 1.0),
            m_dot=0.1,
            T_out=300.0
        )

        # Should handle large distances gracefully
        v_far = fluid_jet.jet_centerline_velocity(jet, distance=100.0)
        w_far = fluid_jet.jet_spreading_width(jet, distance=100.0)

        assert v_far >= 0.0
        assert w_far > 0.0
