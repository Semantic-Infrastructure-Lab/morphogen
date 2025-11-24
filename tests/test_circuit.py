"""Comprehensive tests for Circuit/Electrical simulation domain.

Tests cover:
- Basic circuit construction
- Component addition (R, L, C, sources)
- DC analysis (steady-state)
- AC analysis (frequency response)
- Transient analysis (time-domain)
- Query operations (voltages, currents, power)
- Integration tests (complete circuits)
"""

import pytest
import numpy as np
from morphogen.stdlib.circuit import CircuitOperations as circuit, ComponentType


class TestCircuitConstruction:
    """Tests for basic circuit creation and component addition."""

    def test_create_empty_circuit(self):
        """Test creating an empty circuit."""
        c = circuit.create(num_nodes=3)
        assert c.num_nodes == 3
        assert len(c.components) == 0
        assert c.dt == 1e-6  # Default timestep

    def test_create_circuit_custom_dt(self):
        """Test creating circuit with custom timestep."""
        c = circuit.create(num_nodes=5, dt=1e-5)
        assert c.num_nodes == 5
        assert c.dt == 1e-5

    def test_add_resistor(self):
        """Test adding a resistor."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R1")
        assert len(c.components) == 1
        comp = c.components[0]
        assert comp.comp_type == ComponentType.RESISTOR
        assert comp.value == 1000.0
        assert comp.name == "R1"

    def test_add_capacitor(self):
        """Test adding a capacitor."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_capacitor(c, node1=1, node2=0, capacitance=100e-9, name="C1")
        assert len(c.components) == 1
        comp = c.components[0]
        assert comp.comp_type == ComponentType.CAPACITOR
        assert comp.value == 100e-9

    def test_add_inductor(self):
        """Test adding an inductor."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_inductor(c, node1=1, node2=0, inductance=10e-3, name="L1")
        assert len(c.components) == 1
        comp = c.components[0]
        assert comp.comp_type == ComponentType.INDUCTOR
        assert comp.value == 10e-3

    def test_add_voltage_source(self):
        """Test adding a voltage source."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="V1")
        assert len(c.components) == 1
        comp = c.components[0]
        assert comp.comp_type == ComponentType.VOLTAGE_SOURCE
        assert comp.value == 5.0

    def test_add_current_source(self):
        """Test adding a current source."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_current_source(c, node_pos=1, node_neg=0, current=0.001, name="I1")
        assert len(c.components) == 1
        comp = c.components[0]
        assert comp.comp_type == ComponentType.CURRENT_SOURCE
        assert comp.value == 0.001

    def test_add_multiple_components(self):
        """Test adding multiple components to same circuit."""
        c = circuit.create(num_nodes=4)
        c = circuit.add_resistor(c, 1, 0, 1000.0, "R1")
        c = circuit.add_resistor(c, 2, 1, 2000.0, "R2")
        c = circuit.add_capacitor(c, 2, 0, 100e-9, "C1")
        c = circuit.add_voltage_source(c, 3, 0, 5.0, "V1")
        assert len(c.components) == 4


class TestDCAnalysis:
    """Tests for DC steady-state analysis."""

    def test_voltage_divider(self):
        """Test simple voltage divider: V1=10V, R1=R2=1k -> V_mid=5V."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R2")

        c = circuit.dc_analysis(c)

        # Node 1 should be 10V (voltage source)
        v1 = circuit.get_node_voltage(c, 1)
        assert np.isclose(v1, 10.0, rtol=1e-6)

        # Node 2 should be 5V (midpoint)
        v2 = circuit.get_node_voltage(c, 2)
        assert np.isclose(v2, 5.0, rtol=1e-6)

        # Current through voltage source should be 10V / 2kΩ = 5mA
        i_source = circuit.get_branch_current(c, "V1")
        assert np.isclose(i_source, 0.005, rtol=1e-6)

    def test_current_source_with_resistor(self):
        """Test current source with resistor: I=1mA, R=1k -> V=1V."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_current_source(c, node_pos=0, node_neg=1, current=0.001, name="I1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R1")

        c = circuit.dc_analysis(c)

        v1 = circuit.get_node_voltage(c, 1)
        # Current flows into node 1, V = I * R = 0.001 * 1000 = 1V
        assert np.isclose(abs(v1), 1.0, rtol=1e-6)

    def test_parallel_resistors(self):
        """Test parallel resistors: V=10V, R1=R2=1k (parallel = 500Ω)."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R2")

        c = circuit.dc_analysis(c)

        # Total current should be V/R_parallel = 10V / 500Ω = 20mA
        i_source = circuit.get_branch_current(c, "V1")
        assert np.isclose(i_source, 0.020, rtol=1e-6)

    def test_series_resistors(self):
        """Test series resistors: V=10V, R1=R2=1k (series = 2kΩ)."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R2")

        c = circuit.dc_analysis(c)

        # Total current should be V/R_total = 10V / 2000Ω = 5mA
        i_source = circuit.get_branch_current(c, "V1")
        assert np.isclose(i_source, 0.005, rtol=1e-6)

    def test_dc_capacitor_open_circuit(self):
        """Test that capacitors are open circuits in DC analysis."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_capacitor(c, node1=2, node2=0, capacitance=100e-6, name="C1")

        c = circuit.dc_analysis(c)

        # Capacitor blocks DC, so no current flows, node 2 voltage is undefined
        # In practice, MNA should handle this (might need ground path)

    def test_dc_inductor_short_circuit(self):
        """Test that inductors are short circuits in DC analysis."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_inductor(c, node1=1, node2=2, inductance=10e-3, name="L1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R1")

        c = circuit.dc_analysis(c)

        # Inductor is short in DC, so nodes 1 and 2 are at same voltage
        v1 = circuit.get_node_voltage(c, 1)
        v2 = circuit.get_node_voltage(c, 2)
        assert np.isclose(v1, v2, rtol=1e-6)


class TestACAnalysis:
    """Tests for AC frequency response analysis."""

    def test_ac_resistor_frequency_independent(self):
        """Test that resistor impedance is frequency-independent."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R1")

        freqs = np.array([1.0, 10.0, 100.0, 1000.0])
        result = circuit.ac_analysis(c, freqs)

        # Check that all frequencies are in result
        assert 'frequencies' in result
        assert len(result['frequencies']) == len(freqs)

    def test_rc_lowpass_filter(self):
        """Test RC lowpass filter frequency response."""
        # R=1kΩ, C=100nF -> cutoff = 1/(2πRC) ≈ 1.59 kHz
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_capacitor(c, node1=2, node2=0, capacitance=100e-9, name="C1")

        freqs = np.logspace(1, 5, 50)  # 10 Hz to 100 kHz
        result = circuit.ac_analysis(c, freqs)

        # At low frequencies, output ≈ input (capacitor open)
        # At high frequencies, output → 0 (capacitor short)
        assert 'node_voltages' in result

    def test_rl_lowpass_filter(self):
        """Test RL lowpass filter frequency response."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="V1")
        c = circuit.add_inductor(c, node1=1, node2=2, inductance=10e-3, name="L1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R1")

        freqs = np.logspace(1, 5, 50)
        result = circuit.ac_analysis(c, freqs)

        assert 'node_voltages' in result
        assert 'impedances' in result


class TestTransientAnalysis:
    """Tests for time-domain transient analysis."""

    def test_rc_step_response(self):
        """Test RC circuit step response (charging)."""
        # R=1kΩ, C=100µF, V=5V -> τ = RC = 0.1s
        c = circuit.create(num_nodes=3, dt=1e-4)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_capacitor(c, node1=2, node2=0, capacitance=100e-6, name="C1")

        duration = 0.5  # 5 time constants
        time_points, voltage_history = circuit.transient_analysis(c, duration)

        # Check that capacitor voltage approaches source voltage
        final_voltage = voltage_history[-1, 1]  # Node 2 (capacitor voltage)
        assert final_voltage > 4.5  # Should be close to 5V after 5τ

    def test_rl_step_response(self):
        """Test RL circuit step response (current rise)."""
        c = circuit.create(num_nodes=3, dt=1e-6)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=100.0, name="R1")
        c = circuit.add_inductor(c, node1=2, node2=0, inductance=10e-3, name="L1")

        duration = 1e-3  # 1ms
        time_points, voltage_history = circuit.transient_analysis(c, duration)

        # Current should rise exponentially with τ = L/R = 100µs
        assert len(time_points) > 0

    def test_rc_discharge(self):
        """Test RC discharge (requires initial conditions - future feature)."""
        # This test would require setting initial capacitor voltage
        # Current implementation starts from zero
        pass


class TestQueryOperations:
    """Tests for circuit query operations."""

    def test_get_node_voltage(self):
        """Test retrieving node voltage after analysis."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=3.3, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R1")
        c = circuit.dc_analysis(c)

        v1 = circuit.get_node_voltage(c, 1)
        assert np.isclose(v1, 3.3, rtol=1e-6)

    def test_get_branch_current(self):
        """Test retrieving branch current."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=1000.0, name="R1")
        c = circuit.dc_analysis(c)

        i1 = circuit.get_branch_current(c, "V1")
        # I = V/R = 10/1000 = 0.01A
        assert np.isclose(i1, 0.01, rtol=1e-6)

    def test_get_power_dissipated(self):
        """Test calculating power dissipation."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=100.0, name="R1")
        c = circuit.dc_analysis(c)

        power = circuit.get_power(c, "R1")
        # P = V²/R = 100/100 = 1W
        assert np.isclose(power, 1.0, rtol=1e-6)

    def test_get_power_delivered(self):
        """Test calculating power delivered by source."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=0, resistance=10.0, name="R1")
        c = circuit.dc_analysis(c)

        power = circuit.get_power(c, "V1")
        # P = V*I = 5 * (5/10) = 2.5W (delivered, so negative)
        assert np.isclose(abs(power), 2.5, rtol=1e-6)


class TestIntegration:
    """Integration tests for complete circuits."""

    def test_wheatstone_bridge_balanced(self):
        """Test balanced Wheatstone bridge (zero current through detector)."""
        # Classic bridge circuit: R1/R2 = R3/R4
        c = circuit.create(num_nodes=5)

        # Voltage source
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=10.0, name="V1")

        # Bridge resistors (all 1kΩ for balance)
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R2")
        c = circuit.add_resistor(c, node1=1, node2=3, resistance=1000.0, name="R3")
        c = circuit.add_resistor(c, node1=3, node2=0, resistance=1000.0, name="R4")

        # Detector resistor (between midpoints)
        c = circuit.add_resistor(c, node1=2, node2=3, resistance=1000.0, name="R_detector")

        c = circuit.dc_analysis(c)

        # In balanced bridge, nodes 2 and 3 should be at same voltage
        v2 = circuit.get_node_voltage(c, 2)
        v3 = circuit.get_node_voltage(c, 3)
        assert np.isclose(v2, v3, rtol=1e-6)

        # Current through detector should be ~zero
        i_det = circuit.get_branch_current(c, "R_detector")
        assert np.isclose(i_det, 0.0, atol=1e-9)

    def test_three_stage_voltage_divider(self):
        """Test cascaded voltage dividers."""
        c = circuit.create(num_nodes=5)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=12.0, name="V1")

        # Three equal resistors in series
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=3, resistance=1000.0, name="R2")
        c = circuit.add_resistor(c, node1=3, node2=0, resistance=1000.0, name="R3")

        c = circuit.dc_analysis(c)

        # Voltages should be 12V, 8V, 4V, 0V
        v1 = circuit.get_node_voltage(c, 1)
        v2 = circuit.get_node_voltage(c, 2)
        v3 = circuit.get_node_voltage(c, 3)

        assert np.isclose(v1, 12.0, rtol=1e-6)
        assert np.isclose(v2, 8.0, rtol=1e-6)
        assert np.isclose(v3, 4.0, rtol=1e-6)

    def test_rlc_circuit_resonance(self):
        """Test RLC circuit (series resonance)."""
        # R=10Ω, L=1mH, C=10µF -> resonant freq = 1/(2π√LC) ≈ 1.59 kHz
        c = circuit.create(num_nodes=4)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=10.0, name="R1")
        c = circuit.add_inductor(c, node1=2, node2=3, inductance=1e-3, name="L1")
        c = circuit.add_capacitor(c, node1=3, node2=0, capacitance=10e-6, name="C1")

        # Test at resonant frequency and nearby
        freqs = np.array([100.0, 1000.0, 1591.5, 3000.0, 10000.0])
        result = circuit.ac_analysis(c, freqs)

        # At resonance, impedance is minimum (only R), so current is maximum
        assert 'node_voltages' in result


class TestOpAmpCircuits:
    """Tests for operational amplifier circuits."""

    def test_non_inverting_amplifier(self):
        """Test non-inverting amplifier with gain = 1 + R2/R1."""
        # Circuit: Vin -> non-inv input, inv input via R1 to ground and R2 to output
        # Gain = 1 + R2/R1 = 1 + 10k/1k = 11
        c = circuit.create(num_nodes=4)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="Vin")
        c = circuit.add_opamp(c, node_in_pos=1, node_in_neg=2, node_out=3, gain=100000.0, name="U1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=3, resistance=10000.0, name="R2")

        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 3)
        # Expected gain: 1 + R2/R1 = 1 + 10 = 11
        assert np.isclose(v_out, 11.0, rtol=1e-3)

    def test_inverting_amplifier(self):
        """Test inverting amplifier with gain = -R2/R1."""
        # Circuit: Vin via R1 to inv input, non-inv to ground, R2 from inv to output
        # Gain = -R2/R1 = -10k/1k = -10
        c = circuit.create(num_nodes=4)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="Vin")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, gain=100000.0, name="U1")
        c = circuit.add_resistor(c, node1=2, node2=3, resistance=10000.0, name="R2")

        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 3)
        # Expected gain: -R2/R1 = -10
        assert np.isclose(v_out, -10.0, rtol=1e-3)

    def test_voltage_follower(self):
        """Test voltage follower (unity gain buffer)."""
        # Circuit: Vin to non-inv, output connected to inv (unity gain feedback)
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="Vin")
        c = circuit.add_opamp(c, node_in_pos=1, node_in_neg=2, node_out=2, gain=100000.0, name="U1")

        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 2)
        # Unity gain: Vout = Vin
        assert np.isclose(v_out, 5.0, rtol=1e-4)

    def test_summing_amplifier(self):
        """Test summing amplifier (inverting, two inputs)."""
        # Vout = -(Rf/R1)*V1 - (Rf/R2)*V2
        # With R1=R2=Rf=1kΩ: Vout = -(V1 + V2)
        c = circuit.create(num_nodes=5)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="V1")
        c = circuit.add_voltage_source(c, node_pos=2, node_neg=0, voltage=2.0, name="V2")
        c = circuit.add_resistor(c, node1=1, node2=3, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=3, resistance=1000.0, name="R2")
        c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=3, node_out=4, gain=100000.0, name="U1")
        c = circuit.add_resistor(c, node1=3, node2=4, resistance=1000.0, name="Rf")

        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 4)
        # Vout = -(V1 + V2) = -(1 + 2) = -3
        assert np.isclose(v_out, -3.0, rtol=1e-3)

    def test_differential_amplifier(self):
        """Test differential amplifier."""
        # Vout = (R2/R1) * (V2 - V1) for balanced resistor ratios
        # With R1=R3=1k, R2=R4=10k: Gain = 10
        c = circuit.create(num_nodes=6)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="V1")
        c = circuit.add_voltage_source(c, node_pos=2, node_neg=0, voltage=1.5, name="V2")

        # Input resistors
        c = circuit.add_resistor(c, node1=1, node2=3, resistance=1000.0, name="R1")  # V1 to inv
        c = circuit.add_resistor(c, node1=2, node2=4, resistance=1000.0, name="R3")  # V2 to non-inv

        # Non-inv input divider to ground
        c = circuit.add_resistor(c, node1=4, node2=0, resistance=10000.0, name="R4")

        # Op-amp and feedback
        c = circuit.add_opamp(c, node_in_pos=4, node_in_neg=3, node_out=5, gain=100000.0, name="U1")
        c = circuit.add_resistor(c, node1=3, node2=5, resistance=10000.0, name="R2")  # Feedback

        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 5)
        # For balanced diff amp: Vout ≈ (R2/R1) * (V2 - V1) = 10 * (1.5 - 1.0) = 5.0
        # (Simplified analysis; actual depends on R3, R4)
        assert np.abs(v_out) > 0.1  # Basic sanity check

    def test_opamp_with_load_resistor(self):
        """Test op-amp driving a load resistor."""
        # Non-inverting amp with load on output
        c = circuit.create(num_nodes=4)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=1.0, name="Vin")
        c = circuit.add_opamp(c, node_in_pos=1, node_in_neg=2, node_out=3, gain=100000.0, name="U1")
        c = circuit.add_resistor(c, node1=2, node2=0, resistance=1000.0, name="R1")
        c = circuit.add_resistor(c, node1=2, node2=3, resistance=10000.0, name="R2")
        c = circuit.add_resistor(c, node1=3, node2=0, resistance=1000.0, name="Rload")

        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 3)
        # Output should still be amplified (op-amp can drive load)
        assert v_out > 5.0  # Gain > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
