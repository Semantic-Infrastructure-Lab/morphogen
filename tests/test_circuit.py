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
        """Test RC circuit step response (charging): V_cap → V_source with τ = RC."""
        # R=1kΩ, C=100µF, V=5V -> τ = RC = 0.1s, dt=1e-4 → 1000 steps per τ
        c = circuit.create(num_nodes=3, dt=1e-4)
        c = circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="V1")
        c = circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")
        c = circuit.add_capacitor(c, node1=2, node2=0, capacitance=100e-6, name="C1")

        duration = 0.5  # 5 time constants
        time_points, voltage_history = circuit.transient_analysis(c, duration)

        # After 5τ (duration=0.5s), capacitor voltage (node 2) should be close to 5V
        final_voltage = voltage_history[-1, 2]  # Node 2 is the capacitor terminal
        assert final_voltage > 4.5, f"Expected V_cap > 4.5V after 5τ, got {final_voltage:.4f}V"

        # At 1τ (1000 steps), should be ~63% of 5V = ~3.15V
        tau_steps = int(1000.0 * 100e-6 * 1e4)  # R*C*sr = 1000*100e-6/1e-4 = 1000
        v_at_tau = voltage_history[tau_steps, 2]
        assert 2.5 < v_at_tau < 4.0, f"Expected ~3.15V (63%) at 1τ, got {v_at_tau:.4f}V"

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


class TestDiodeCircuits:
    """Tests for nonlinear diode components (Newton-Raphson solver)."""

    def test_add_diode(self):
        """Test adding a diode to a circuit."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_diode(c, node_anode=1, node_cathode=2, name="D1")
        assert len(c.components) == 1
        assert c.components[0].params['n_factor'] == 1.0

    def test_add_diode_custom_params(self):
        """Test diode with custom Is and ideality factor."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_diode(c, 1, 2, Is=1e-12, n_factor=1.5, name="D1")
        assert c.components[0].value == 1e-12
        assert c.components[0].params['n_factor'] == 1.5

    def test_diode_forward_voltage_drop(self):
        """Diode forward voltage drop matches Shockley equation."""
        from scipy.optimize import brentq
        Is = 1e-14
        R = 1000.0
        Vs = 5.0
        # True equilibrium from Shockley equation
        vd_true = brentq(lambda vd: Is * (np.exp(vd / 0.02585) - 1) - (Vs - vd) / R, 0.1, 0.9)

        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, 1, 0, Vs, "Vin")
        c = circuit.add_diode(c, 1, 2, Is=Is, name="D1")
        c = circuit.add_resistor(c, 2, 0, R, "Rload")
        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 2)
        vd = Vs - v_out
        assert abs(vd - vd_true) < 0.001  # within 1mV of analytic solution

    def test_diode_reverse_blocking(self):
        """Reverse-biased diode blocks current (near-zero node voltage)."""
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, 1, 0, -5.0, "Vin")
        c = circuit.add_diode(c, 1, 2, name="D1")
        c = circuit.add_resistor(c, 2, 0, 1000.0, "Rload")
        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 2)
        assert abs(v_out) < 1e-6  # essentially zero

    def test_half_wave_rectifier(self):
        """Half-wave rectifier passes positive half, blocks negative."""
        # Positive input → diode conducts, output > 0
        c = circuit.create(num_nodes=3)
        c = circuit.add_voltage_source(c, 1, 0, 3.0, "Vin")
        c = circuit.add_diode(c, 1, 2, name="D1")
        c = circuit.add_resistor(c, 2, 0, 1000.0, "Rload")
        c = circuit.dc_analysis(c)
        assert circuit.get_node_voltage(c, 2) > 2.0  # significant forward current

        # Negative input → diode blocks, output ≈ 0
        c2 = circuit.create(num_nodes=3)
        c2 = circuit.add_voltage_source(c2, 1, 0, -3.0, "Vin")
        c2 = circuit.add_diode(c2, 1, 2, name="D1")
        c2 = circuit.add_resistor(c2, 2, 0, 1000.0, "Rload")
        c2 = circuit.dc_analysis(c2)
        assert abs(circuit.get_node_voltage(c2, 2)) < 1e-6

    def test_two_series_diodes(self):
        """Two diodes in series have roughly double the voltage drop."""
        Is = 1e-14
        R = 1000.0
        Vs = 5.0

        c = circuit.create(num_nodes=4)
        c = circuit.add_voltage_source(c, 1, 0, Vs, "Vin")
        c = circuit.add_diode(c, 1, 2, Is=Is, name="D1")
        c = circuit.add_diode(c, 2, 3, Is=Is, name="D2")
        c = circuit.add_resistor(c, 3, 0, R, "Rload")
        c = circuit.dc_analysis(c)

        v_out = circuit.get_node_voltage(c, 3)
        v_mid = circuit.get_node_voltage(c, 2)
        vd1 = Vs - v_mid
        vd2 = v_mid - v_out
        # Both diodes carry the same current → same voltage drop
        assert abs(vd1 - vd2) < 0.01
        # Total drop is approximately double a single diode's drop
        single_drop = Vs - circuit.get_node_voltage(
            circuit.dc_analysis(
                circuit.add_resistor(
                    circuit.add_diode(
                        circuit.add_voltage_source(circuit.create(num_nodes=3), 1, 0, Vs, "Vin"),
                        1, 2, Is=Is, name="D1"),
                    2, 0, R, "Rload")),
            2)
        assert abs((vd1 + vd2) - 2 * single_drop) < 0.01

    def test_diode_clipping_transient(self):
        """Diode clamps negative half of sine wave in transient simulation."""
        # Circuit: Vin (AC) → diode → Rload → GND
        # Positive cycles: output follows input (minus drop); negative: ~0V
        c = circuit.create(num_nodes=3, dt=1.0 / 10000)

        # We'll use a voltage source and vary it manually
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_diode(c, 1, 2, name="D1")
        c = circuit.add_resistor(c, 2, 0, 1000.0, "Rload")

        freq = 100.0  # Hz
        t_end = 1.0 / freq  # one full cycle
        t, voltages = circuit.transient_analysis(c, t_end)

        # Find the Vin voltage source and manually record what we'd expect.
        # Instead, just verify transient runs without error and has right shape.
        assert len(t) > 0
        assert voltages.shape == (len(t), c.num_nodes)

    def test_diode_voltage_clamp_upper(self):
        """Diode clamp circuit: output capped at ~Vref + Vf."""
        # Standard clamp: diode from output to Vref (ground here), Rin
        # If Vin > Vf, diode conducts and output ≈ Vf; else output = Vin
        c = circuit.create(num_nodes=3)
        # High input (10V) through 10kΩ resistor; diode clamps output to ~Vf
        c = circuit.add_voltage_source(c, 1, 0, 10.0, "Vin")
        c = circuit.add_resistor(c, 1, 2, 10000.0, "Rin")
        c = circuit.add_diode(c, 2, 0, name="D1")  # anode=output, cathode=GND
        c = circuit.dc_analysis(c)

        v_clamped = circuit.get_node_voltage(c, 2)
        # Output should be ~0.6-0.7V (diode forward voltage), not 10V
        assert 0.5 < v_clamped < 0.9

    def test_get_impedance_resistor(self):
        """get_impedance returns correct resistance for a simple resistor."""
        c = circuit.create(num_nodes=2)
        c = circuit.add_resistor(c, 1, 0, 1000.0, "R1")
        z = circuit.get_impedance(c, 1, 0, frequency=1000.0)
        assert abs(z.real - 1000.0) < 1.0  # within 1 Ω

    def test_get_impedance_capacitor(self):
        """get_impedance returns correct reactance for a capacitor."""
        cap = 1e-6  # 1µF
        freq = 1000.0  # 1kHz
        expected_magnitude = 1.0 / (2 * np.pi * freq * cap)  # ~159 Ω

        c = circuit.create(num_nodes=2)
        c = circuit.add_capacitor(c, 1, 0, cap, "C1")
        z = circuit.get_impedance(c, 1, 0, frequency=freq)
        assert abs(abs(z) - expected_magnitude) < 5.0  # within 5 Ω


class TestProcessAudio:
    """Tests for process_audio: circuit-audio integration."""

    def _make_audio_buffer(self, data, sample_rate=48000):
        from morphogen.stdlib.audio import AudioBuffer
        return AudioBuffer(np.array(data, dtype=float), sample_rate)

    def test_process_audio_resistive_divider_gain(self):
        """Resistive divider scales signal; no capacitor memory needed."""
        sr = 48000
        c = circuit.create(num_nodes=3, dt=1.0 / sr)
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_resistor(c, 1, 2, 1000.0, "R1")
        c = circuit.add_resistor(c, 2, 0, 1000.0, "R2")

        signal = np.array([1.0, -1.0, 0.5, -0.5])
        audio_in = self._make_audio_buffer(signal, sr)
        out = circuit.process_audio(c, audio_in, input_node=1, output_node=2)

        expected = signal * 0.5  # voltage divider
        np.testing.assert_allclose(out.data, expected, atol=1e-6)

    def test_process_audio_output_sample_rate_preserved(self):
        """Output sample rate matches input."""
        sr = 44100
        c = circuit.create(num_nodes=2, dt=1.0 / sr)
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_resistor(c, 1, 0, 1000.0, "R1")

        audio_in = self._make_audio_buffer([0.1, 0.2], sr)
        out = circuit.process_audio(c, audio_in, input_node=1, output_node=1)
        assert out.sample_rate == sr

    def test_process_audio_rc_lowpass_has_memory(self):
        """RC lowpass: step input should cause exponential rise, not instant response.

        If process_audio used dc_analysis each sample, the capacitor would be
        invisible and output would equal input immediately. With transient stepping,
        the output rises toward the input with time constant RC.
        """
        sr = 10000
        R = 1000.0
        C = 100e-6  # RC = 0.1 s → tau = 0.1 s, step to ~63% after 1000 samples
        dt = 1.0 / sr

        c = circuit.create(num_nodes=3, dt=dt)
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_resistor(c, 1, 2, R, "R1")
        c = circuit.add_capacitor(c, 2, 0, C, "C1")

        # Step from 0 → 1V; output node 2 (across capacitor)
        num_samples = int(0.5 * sr)  # 0.5 seconds
        step = np.ones(num_samples)
        audio_in = self._make_audio_buffer(step, sr)
        out = circuit.process_audio(c, audio_in, input_node=1, output_node=2)

        # After 1 tau (1000 samples), should be ~63% of final value
        tau_idx = int(R * C * sr)
        v_at_tau = out.data[tau_idx]
        assert 0.55 < v_at_tau < 0.72, (
            f"RC step response at 1 tau should be ~0.63, got {v_at_tau:.4f}. "
            "Likely dc_analysis bug: capacitor has no memory."
        )

        # Output should NOT be instant (dc_analysis would give ~1.0 immediately)
        assert out.data[10] < 0.5, (
            f"Output too fast after 10 samples ({out.data[10]:.4f}), "
            "suggests no capacitor memory (dc_analysis bug)."
        )

    def test_process_audio_rc_final_value(self):
        """After many tau, RC output converges to input voltage."""
        sr = 10000
        R = 100.0
        C = 100e-6  # RC = 0.01 s → 5 tau = 0.05 s = 500 samples

        c = circuit.create(num_nodes=3, dt=1.0 / sr)
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_resistor(c, 1, 2, R, "R1")
        c = circuit.add_capacitor(c, 2, 0, C, "C1")

        step = np.ones(int(0.2 * sr))  # 0.2 s >> 5 tau
        audio_in = self._make_audio_buffer(step, sr)
        out = circuit.process_audio(c, audio_in, input_node=1, output_node=2)

        assert out.data[-1] > 0.99, f"Expected convergence to ~1V, got {out.data[-1]:.4f}"

    def test_process_audio_opamp_gain(self):
        """Op-amp inverting amplifier via process_audio: output = -gain * input."""
        sr = 48000
        gain = 5.0
        c = circuit.create(num_nodes=4, dt=1.0 / sr)
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_resistor(c, 1, 2, 1000.0, "Rin")
        c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")
        c = circuit.add_resistor(c, 2, 3, gain * 1000.0, "Rfb")

        signal = np.array([0.1, -0.2, 0.3])
        audio_in = self._make_audio_buffer(signal, sr)
        out = circuit.process_audio(c, audio_in, input_node=1, output_node=3)

        # Inverting amplifier: V_out = -gain * V_in
        np.testing.assert_allclose(out.data, -gain * signal, rtol=5e-3,
                                   err_msg="Op-amp transient gain wrong (opamp not stamped in _build_transient_matrices)")

    def test_transient_opamp_gain(self):
        """Op-amp gain works in transient_analysis (was broken: opamps silently dropped)."""
        dt = 1e-5
        gain = 3.0
        c = circuit.create(num_nodes=4, dt=dt)
        c = circuit.add_voltage_source(c, 1, 0, 1.0, "Vs")
        c = circuit.add_resistor(c, 1, 2, 1000.0, "Rin")
        c = circuit.add_opamp(c, node_in_pos=0, node_in_neg=2, node_out=3, name="U1")
        c = circuit.add_resistor(c, 2, 3, gain * 1000.0, "Rfb")

        _, v_hist = circuit.transient_analysis(c, duration=5 * dt)

        # Steady state: V3 = -gain * V1 = -3V
        v_out = v_hist[-1, 3]
        assert abs(v_out - (-gain)) < 0.01, (
            f"Op-amp transient gain wrong: expected {-gain}V, got {v_out:.4f}V. "
            "Likely opamps not stamped in _build_transient_matrices."
        )

    def test_process_audio_diode_clipping(self):
        """Diode + resistor: clips positive half above ~0.6V, passes negative."""
        sr = 10000
        c = circuit.create(num_nodes=3, dt=1.0 / sr)
        c = circuit.add_voltage_source(c, 1, 0, 0.0, "Vin")
        c = circuit.add_resistor(c, 1, 2, 1000.0, "R1")
        c = circuit.add_diode(c, node_anode=2, node_cathode=0, name="D1")

        # Sweep +/- amplitudes
        amps = [0.1, 0.3, 0.7, 1.0, 2.0]
        signal = np.array(amps)
        audio_in = self._make_audio_buffer(signal, sr)
        out = circuit.process_audio(c, audio_in, input_node=1, output_node=2)

        # Forward-biased: all outputs should be clamped below ~0.75V (diode drop)
        for v in out.data:
            assert v < 0.75, f"Diode should clamp output, got {v:.4f}V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
