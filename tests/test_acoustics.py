"""Unit tests for acoustics operations."""

import pytest
import numpy as np
from morphogen.stdlib.acoustics import (
    acoustics, PipeGeometry, WaveguideNetwork, ReflectionCoeff,
    FrequencyResponse, create_pipe, create_expansion_chamber,
    SPEED_OF_SOUND
)


class TestPipeGeometry:
    """Tests for PipeGeometry data structure."""

    def test_create_simple_pipe(self):
        """Test creating a simple uniform pipe."""
        pipe = PipeGeometry(diameter=0.025, length=1.0)
        assert pipe.diameter == 0.025
        assert pipe.length == 1.0
        assert pipe.segments is None

    def test_create_variable_pipe(self):
        """Test creating a pipe with variable diameter."""
        segments = [(0.0, 0.04), (0.5, 0.12), (1.0, 0.05)]
        pipe = PipeGeometry(diameter=0.04, length=1.0, segments=segments)
        assert pipe.segments is not None
        assert len(pipe.segments) == 3

    def test_create_pipe_helper(self):
        """Test create_pipe helper function."""
        pipe = create_pipe(diameter=0.05, length=2.0)
        assert pipe.diameter == 0.05
        assert pipe.length == 2.0

    def test_create_expansion_chamber(self):
        """Test create_expansion_chamber helper function."""
        chamber = create_expansion_chamber(
            inlet_diameter=0.04,
            belly_diameter=0.12,
            outlet_diameter=0.05,
            total_length=1.0
        )
        assert chamber.length == 1.0
        assert chamber.segments is not None
        assert len(chamber.segments) == 3


class TestWaveguideConstruction:
    """Tests for waveguide construction from geometry."""

    def test_waveguide_from_uniform_pipe(self):
        """Test creating waveguide from uniform pipe."""
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(
            pipe,
            discretization=0.01,
            sample_rate=44100
        )

        assert wg.num_segments == 100  # 1.0m / 0.01m
        assert wg.segment_length == 0.01
        assert wg.total_length == 1.0
        assert np.all(wg.diameters == 0.025)

    def test_waveguide_from_expansion_chamber(self):
        """Test creating waveguide from expansion chamber."""
        chamber = create_expansion_chamber(
            inlet_diameter=0.04,
            belly_diameter=0.12,
            outlet_diameter=0.05,
            total_length=1.0
        )
        wg = acoustics.waveguide_from_geometry(chamber, discretization=0.01)

        assert wg.num_segments == 100
        # Check interpolated diameters
        assert wg.diameters[0] == pytest.approx(0.04, abs=0.01)
        assert wg.diameters[50] == pytest.approx(0.12, abs=0.01)
        assert wg.diameters[-1] == pytest.approx(0.05, abs=0.01)

    def test_waveguide_delay_samples(self):
        """Test waveguide delay calculation."""
        pipe = create_pipe(diameter=0.05, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe, sample_rate=44100)

        # Round-trip delay = 2 * length / speed_of_sound * sample_rate
        expected_delay = int(2 * 1.0 / SPEED_OF_SOUND * 44100)
        assert wg.delay_samples == expected_delay


class TestReflectionCoefficients:
    """Tests for reflection coefficient calculation."""

    def test_reflection_open_end(self):
        """Test reflection coefficient for open pipe end."""
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe)
        reflections = acoustics.reflection_coefficients(wg, end_condition="open")

        # Should have one reflection at end
        assert len(reflections) >= 1
        end_refl = reflections[-1]
        assert end_refl.position == wg.num_segments - 1
        assert end_refl.coefficient == pytest.approx(-0.95, abs=0.1)

    def test_reflection_closed_end(self):
        """Test reflection coefficient for closed pipe end."""
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe)
        reflections = acoustics.reflection_coefficients(wg, end_condition="closed")

        # Should have one reflection at end
        end_refl = reflections[-1]
        assert end_refl.coefficient == pytest.approx(0.95, abs=0.1)

    def test_reflection_matched_end(self):
        """Test reflection coefficient for matched impedance."""
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe)
        reflections = acoustics.reflection_coefficients(wg, end_condition="matched")

        # Should have one reflection at end with R=0
        end_refl = reflections[-1]
        assert end_refl.coefficient == pytest.approx(0.0, abs=0.01)

    def test_reflection_at_area_change(self):
        """Test reflection at area discontinuity."""
        # Create pipe with significant area change
        segments = [(0.0, 0.04), (0.5, 0.12), (1.0, 0.04)]
        pipe = PipeGeometry(diameter=0.04, length=1.0, segments=segments)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.01)
        reflections = acoustics.reflection_coefficients(wg)

        # Should detect reflections at area changes (excluding end)
        # Note: may have multiple reflections depending on discretization
        assert len(reflections) >= 1


class TestWaveguidePropagation:
    """Tests for waveguide propagation."""

    def test_waveguide_step_initialization(self):
        """Test waveguide step with zero initial conditions."""
        pipe = create_pipe(diameter=0.025, length=0.5)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.01)
        reflections = acoustics.reflection_coefficients(wg)

        p_fwd = np.zeros(wg.num_segments)
        p_bwd = np.zeros(wg.num_segments)

        p_fwd_new, p_bwd_new = acoustics.waveguide_step(
            p_fwd, p_bwd, wg, reflections
        )

        # With zero initial conditions and no excitation, should remain zero
        assert np.allclose(p_fwd_new, 0.0)
        assert np.allclose(p_bwd_new, 0.0)

    def test_waveguide_step_with_excitation(self):
        """Test waveguide step with impulse excitation.

        After injecting 1.0 at position 0, the forward wave is at segment 0
        (injection appended after propagation shift). No backward wave yet —
        the impulse hasn't reached the far end to reflect.
        """
        pipe = create_pipe(diameter=0.025, length=0.5)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.01)
        reflections = acoustics.reflection_coefficients(wg)

        p_fwd = np.zeros(wg.num_segments)
        p_bwd = np.zeros(wg.num_segments)

        excitation = np.array([1.0])
        p_fwd_new, p_bwd_new = acoustics.waveguide_step(
            p_fwd, p_bwd, wg, reflections,
            excitation=excitation, excitation_pos=0
        )

        # Impulse should be at entry segment (injected after propagation shift)
        assert p_fwd_new[0] == pytest.approx(1.0, abs=0.01), (
            f"Impulse expected at segment 0, got p_fwd[0]={p_fwd_new[0]:.4f}")
        # No backward wave yet — reflection hasn't been reached
        assert np.allclose(p_bwd_new, 0.0), (
            f"No backward wave expected on first step, got max={np.max(np.abs(p_bwd_new)):.4f}")

    def test_waveguide_propagation_multiple_steps(self):
        """Test multiple steps of waveguide propagation.

        After injecting at position 0 then propagating N more steps, the impulse
        peak in the forward wave should be at segment N (one shift per step).
        The open end is at segment 49 — far enough that no reflection arrives
        in 10 steps.
        """
        pipe = create_pipe(diameter=0.025, length=0.5)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.01)
        reflections = acoustics.reflection_coefficients(wg, end_condition="open")

        p_fwd = np.zeros(wg.num_segments)
        p_bwd = np.zeros(wg.num_segments)

        # Step 1: inject impulse (ends up at segment 0)
        excitation = np.array([1.0])
        p_fwd, p_bwd = acoustics.waveguide_step(
            p_fwd, p_bwd, wg, reflections,
            excitation=excitation, excitation_pos=0
        )

        # Propagate 10 more steps — impulse should be at segment 10
        for _ in range(10):
            p_fwd, p_bwd = acoustics.waveguide_step(
                p_fwd, p_bwd, wg, reflections
            )

        # Impulse should be at exactly segment 10 in the forward wave (no losses in uniform pipe)
        assert p_fwd[10] > 0.9, (
            f"Impulse expected at segment 10, p_fwd[10]={p_fwd[10]:.4f}")
        assert np.argmax(p_fwd) == 10, (
            f"Peak expected at segment 10, got segment {np.argmax(p_fwd)}")

    def test_total_pressure(self):
        """Test total pressure calculation."""
        p_fwd = np.array([1.0, 2.0, 3.0])
        p_bwd = np.array([0.5, 1.0, 1.5])

        p_total = acoustics.total_pressure(p_fwd, p_bwd)

        expected = np.array([1.5, 3.0, 4.5])
        assert np.allclose(p_total, expected)


class TestHelmholtzResonator:
    """Tests for Helmholtz resonator calculations."""

    def test_helmholtz_frequency_calculation(self):
        """Test Helmholtz resonant frequency formula."""
        # Example: 500 cm³ volume, 50mm neck, 20cm² area
        f_res = acoustics.helmholtz_frequency(
            volume=500e-6,  # 500 cm³ = 500e-6 m³
            neck_length=0.05,  # 50 mm
            neck_area=20e-4  # 20 cm² = 20e-4 m²
        )

        # Expected: f ≈ (343 / 2π) * sqrt(20e-4 / (500e-6 * 0.05))
        # f ≈ 54.6 * sqrt(0.002 / 0.000025) ≈ 54.6 * sqrt(80) ≈ 488 Hz
        assert f_res > 0
        assert 450 < f_res < 550  # Reasonable range around 488 Hz

    def test_helmholtz_frequency_small_volume(self):
        """Test that smaller volume gives higher frequency."""
        f1 = acoustics.helmholtz_frequency(
            volume=1000e-6, neck_length=0.05, neck_area=20e-4
        )
        f2 = acoustics.helmholtz_frequency(
            volume=500e-6, neck_length=0.05, neck_area=20e-4
        )

        # Smaller volume -> higher frequency
        assert f2 > f1

    def test_helmholtz_impedance(self):
        """Test Helmholtz impedance calculation."""
        Z = acoustics.helmholtz_impedance(
            frequency=100.0,
            volume=500e-6,
            neck_length=0.05,
            neck_area=20e-4,
            damping=0.1
        )

        # Should be complex impedance
        assert isinstance(Z, complex)
        # Should have non-zero real part (resistance)
        assert Z.real != 0
        # Should have imaginary part (reactance)
        assert Z.imag != 0


class TestRadiationImpedance:
    """Tests for radiation impedance calculations."""

    def test_radiation_impedance_low_frequency(self):
        """Test radiation impedance at low frequency."""
        Z = acoustics.radiation_impedance_unflanged(
            diameter=0.05,
            frequency=100.0
        )

        # Should be complex
        assert isinstance(Z, complex)
        # Real part (radiation resistance) should be positive
        assert Z.real > 0
        # Imaginary part (reactance) should be positive at low freq
        assert Z.imag > 0

    def test_radiation_impedance_high_frequency(self):
        """Test radiation impedance at high frequency."""
        Z = acoustics.radiation_impedance_unflanged(
            diameter=0.05,
            frequency=5000.0
        )

        # At high frequency, should approach characteristic impedance
        assert isinstance(Z, complex)
        assert Z.real > 0

    def test_radiation_impedance_increases_with_frequency(self):
        """Test that radiation resistance increases with frequency."""
        Z1 = acoustics.radiation_impedance_unflanged(
            diameter=0.05, frequency=100.0
        )
        Z2 = acoustics.radiation_impedance_unflanged(
            diameter=0.05, frequency=1000.0
        )

        # Higher frequency -> higher radiation resistance
        assert Z2.real > Z1.real


class TestTransferFunction:
    """Tests for transfer function analysis."""

    def test_transfer_function_basic(self):
        """Test basic transfer function calculation."""
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.02)
        reflections = acoustics.reflection_coefficients(wg, end_condition="open")

        # Compute transfer function (low resolution for fast test)
        response = acoustics.transfer_function(
            wg, reflections,
            freq_range=(100.0, 1000.0),
            resolution=100.0
        )

        assert isinstance(response, FrequencyResponse)
        assert len(response.frequencies) > 0
        assert len(response.magnitude) == len(response.frequencies)
        assert len(response.phase) == len(response.frequencies)

    def test_transfer_function_pipe_resonances(self):
        """Test that transfer function shows pipe resonances."""
        # Open-open pipe: resonances at f = n * c / (2L)
        # For L = 1m, c = 343 m/s: f = 171.5, 343, 514.5 Hz, ...
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.02)
        reflections = acoustics.reflection_coefficients(wg, end_condition="open")

        response = acoustics.transfer_function(
            wg, reflections,
            freq_range=(50.0, 600.0),
            resolution=20.0
        )

        # Find peaks
        resonances = acoustics.resonant_frequencies(response, threshold_db=-10.0)

        # Should find at least one resonance
        assert len(resonances) > 0

        # First resonance should be near 171.5 Hz (open-open: f = c/2L)
        expected_f1 = SPEED_OF_SOUND / (2 * 1.0)
        assert any(abs(f - expected_f1) / expected_f1 < 0.2 for f in resonances), (
            f"Expected resonance near {expected_f1:.1f}Hz (±20%), got: {resonances}")


class TestResonantFrequencies:
    """Tests for resonant frequency detection."""

    def test_resonant_frequencies_peak_detection(self):
        """Test peak detection in frequency response."""
        # Create synthetic frequency response with known peaks
        freqs = np.linspace(100, 1000, 100)
        mag = -20 * np.ones_like(freqs)  # -20 dB baseline

        # Add peaks at 300 Hz and 700 Hz
        mag[np.abs(freqs - 300) < 20] = 0  # 0 dB peak
        mag[np.abs(freqs - 700) < 20] = 0  # 0 dB peak

        response = FrequencyResponse(
            frequencies=freqs,
            magnitude=mag,
            phase=np.zeros_like(freqs)
        )

        resonances = acoustics.resonant_frequencies(response, threshold_db=-10.0)

        # Should detect both peaks
        assert len(resonances) >= 2
        # Check that detected peaks are near expected frequencies
        assert any(abs(f - 300) < 50 for f in resonances)
        assert any(abs(f - 700) < 50 for f in resonances)


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_simple_pipe_simulation(self):
        """Test complete simulation of simple pipe."""
        # Create 1m open pipe
        pipe = create_pipe(diameter=0.025, length=1.0)
        wg = acoustics.waveguide_from_geometry(pipe, discretization=0.02)
        reflections = acoustics.reflection_coefficients(wg, end_condition="open")

        # Initialize
        p_fwd = np.zeros(wg.num_segments)
        p_bwd = np.zeros(wg.num_segments)

        # Create impulse excitation
        impulse = np.zeros(100)
        impulse[0] = 1.0

        # Simulate
        output = []
        for t in range(len(impulse)):
            exc = np.array([impulse[t]])
            p_fwd, p_bwd = acoustics.waveguide_step(
                p_fwd, p_bwd, wg, reflections,
                excitation=exc, excitation_pos=0
            )
            p_total = acoustics.total_pressure(p_fwd, p_bwd)
            output.append(p_total[-1])  # Record output at end

        output = np.array(output)

        # For a 1m pipe with 50 segments (0.02m each), the impulse travels
        # 49 segments to the end. The output at the last segment should be
        # non-zero well before step 100, and zero in the first few steps.
        assert np.max(np.abs(output[:5])) < 0.1, (
            "No output expected in first 5 steps — impulse hasn't reached end yet")
        assert np.max(np.abs(output)) > 0.5, (
            f"Expected strong response when impulse arrives at output, max={np.max(np.abs(output)):.4f}")

    def test_expansion_chamber_simulation(self):
        """Test simulation of expansion chamber (muffler)."""
        # Create expansion chamber
        chamber = create_expansion_chamber(
            inlet_diameter=0.04,
            belly_diameter=0.12,
            outlet_diameter=0.05,
            total_length=1.0
        )
        wg = acoustics.waveguide_from_geometry(chamber, discretization=0.02)
        reflections = acoustics.reflection_coefficients(wg, end_condition="open")

        # Should have reflections at area changes
        assert len(reflections) > 1

        # Simulate impulse
        p_fwd = np.zeros(wg.num_segments)
        p_bwd = np.zeros(wg.num_segments)

        for _ in range(50):
            exc = np.array([1.0]) if _ == 0 else None
            p_fwd, p_bwd = acoustics.waveguide_step(
                p_fwd, p_bwd, wg, reflections,
                excitation=exc
            )

        # Should propagate energy
        p_total = acoustics.total_pressure(p_fwd, p_bwd)
        assert np.sum(np.abs(p_total)) > 0
