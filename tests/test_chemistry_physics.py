"""Physics-level tests for chemistry domains: kinetics, electrochem, catalysis, transport.

Each test validates a specific formula or physical invariant rather than metadata.
"""

import math
import numpy as np
import pytest

from morphogen.stdlib.kinetics import (
    arrhenius, modified_arrhenius, vant_hoff,
    reaction_rates, batch_reactor, cstr, mass_transfer_limited,
    Reaction, RateLaw, RateLawType,
)
from morphogen.stdlib.electrochem import (
    butler_volmer, tafel_equation, nernst, limiting_current,
)
from morphogen.stdlib.catalysis import (
    langmuir_hinshelwood, langmuir_adsorption, competitive_adsorption,
)
from morphogen.stdlib.transport import (
    conduction, radiation, nusselt_correlation,
)
from morphogen.stdlib import electrochem as _ec
from morphogen.stdlib import transport as _tr

R_GAS = 8.314                    # J/(mol·K)
FARADAY = _ec.FARADAY            # 96485.3329 C/mol
STEFAN_BOLTZMANN = _tr.STEFAN_BOLTZMANN  # 5.670374419e-8 W/(m²·K⁴)


# ============================================================================
# Kinetics
# ============================================================================

class TestArrhenius:
    def test_formula_exact(self):
        """k = A * exp(-Ea / (R*T)) — exact numeric check."""
        A, Ea, T = 1e13, 50000.0, 500.0
        expected = A * math.exp(-Ea / (R_GAS * T))
        assert abs(arrhenius(T, A, Ea) - expected) / expected < 1e-10

    def test_higher_temperature_increases_rate(self):
        k_low = arrhenius(300.0, 1e10, 50000.0)
        k_high = arrhenius(600.0, 1e10, 50000.0)
        assert k_high > k_low

    def test_zero_activation_energy_returns_A(self):
        """Ea=0 → k = A regardless of temperature."""
        A = 5.0e8
        assert abs(arrhenius(300.0, A, 0.0) - A) / A < 1e-10
        assert abs(arrhenius(1000.0, A, 0.0) - A) / A < 1e-10

    def test_very_high_activation_energy_suppresses_rate(self):
        k = arrhenius(300.0, 1e10, 500000.0)
        assert k < 1e-50  # essentially zero


class TestModifiedArrhenius:
    def test_formula_exact(self):
        """k = A * T^n * exp(-Ea / (R*T))."""
        A, n, Ea, T = 1e10, 0.5, 40000.0, 400.0
        expected = A * (T ** n) * math.exp(-Ea / (R_GAS * T))
        result = modified_arrhenius(T, A, n, Ea)
        assert abs(result - expected) / expected < 1e-10

    def test_zero_n_matches_arrhenius(self):
        """n=0 → modified_arrhenius should equal arrhenius."""
        A, Ea, T = 1e12, 60000.0, 500.0
        assert abs(modified_arrhenius(T, A, 0.0, Ea) - arrhenius(T, A, Ea)) < 1e-8


class TestVantHoff:
    def test_zero_enthalpy_gives_entropy_term(self):
        """delta_H=0 → K_eq = exp(delta_S / R)."""
        delta_S = 50.0  # J/(mol·K)
        expected = math.exp(delta_S / R_GAS)
        assert abs(vant_hoff(500.0, 0.0, delta_S) - expected) / expected < 1e-10

    def test_exothermic_K_decreases_with_T(self):
        """Exothermic reaction (delta_H < 0): K_eq decreases as T rises (Le Chatelier)."""
        K_low = vant_hoff(300.0, -40000.0, 0.0)
        K_high = vant_hoff(600.0, -40000.0, 0.0)
        assert K_high < K_low

    def test_endothermic_K_increases_with_T(self):
        """Endothermic reaction (delta_H > 0): K_eq increases as T rises."""
        K_low = vant_hoff(300.0, 40000.0, 0.0)
        K_high = vant_hoff(600.0, 40000.0, 0.0)
        assert K_high > K_low


class TestReactionRates:
    def _first_order_reaction(self, k_val):
        """A → B, first-order k=k_val."""
        return Reaction(
            reactants={"A": 1},
            products={"B": 1},
            rate_law=RateLaw(type=RateLawType.ARRHENIUS, A=k_val, Ea=0.0),
        )

    def test_first_order_rate_formula(self):
        """d[A]/dt = -k * [A] for A→B first order."""
        k = 0.5
        conc = {"A": 2.0, "B": 0.0}
        rates = reaction_rates(conc, 298.15, [self._first_order_reaction(k)])
        assert abs(rates["A"] - (-k * 2.0)) < 1e-10
        assert abs(rates["B"] - (k * 2.0)) < 1e-10

    def test_second_order_rate(self):
        """A + B → C, rate = k * [A] * [B]."""
        k = 1.0
        reaction = Reaction(
            reactants={"A": 1, "B": 1},
            products={"C": 1},
            rate_law=RateLaw(type=RateLawType.ARRHENIUS, A=k, Ea=0.0),
        )
        conc = {"A": 2.0, "B": 3.0, "C": 0.0}
        rates = reaction_rates(conc, 298.15, [reaction])
        expected_rate = k * 2.0 * 3.0  # = 6.0
        assert abs(rates["A"] - (-expected_rate)) < 1e-10
        assert abs(rates["B"] - (-expected_rate)) < 1e-10
        assert abs(rates["C"] - expected_rate) < 1e-10

    def test_zero_concentration_gives_zero_rate(self):
        """If reactant is depleted, forward rate = 0."""
        k = 10.0
        conc = {"A": 0.0, "B": 0.0}
        rates = reaction_rates(conc, 298.15, [self._first_order_reaction(k)])
        assert rates["A"] == 0.0
        assert rates["B"] == 0.0


class TestBatchReactor:
    def _first_order(self, k):
        return Reaction(
            reactants={"A": 1},
            products={"B": 1},
            rate_law=RateLaw(type=RateLawType.ARRHENIUS, A=k, Ea=0.0),
        )

    def test_first_order_decay(self):
        """[A](t) = [A]₀ · exp(-k·t) for irreversible first-order A→B."""
        k, A0, t = 0.1, 1.0, 10.0
        result = batch_reactor({"A": A0, "B": 0.0}, [self._first_order(k)], 298.15, t)
        expected_A = A0 * math.exp(-k * t)
        assert abs(result["A"] - expected_A) / expected_A < 0.01  # within 1%

    def test_mass_conservation(self):
        """[A] + [B] = [A]₀ + [B]₀ throughout A→B reaction."""
        A0, B0 = 2.0, 0.5
        result = batch_reactor({"A": A0, "B": B0}, [self._first_order(0.2)], 298.15, 5.0)
        total = result["A"] + result["B"]
        assert abs(total - (A0 + B0)) < 0.01

    def test_full_conversion_at_long_time(self):
        """For fast reaction at long time, [A] → 0."""
        result = batch_reactor({"A": 1.0, "B": 0.0}, [self._first_order(10.0)], 298.15, 100.0)
        assert result["A"] < 0.001


class TestCSTR:
    def _first_order(self, k):
        return Reaction(
            reactants={"A": 1},
            products={"B": 1},
            rate_law=RateLaw(type=RateLawType.ARRHENIUS, A=k, Ea=0.0),
        )

    def test_first_order_steady_state(self):
        """CSTR steady-state: c_out = c_in / (1 + τ·k)."""
        k = 0.5     # 1/s
        c_in = 1.0  # mol/L
        Q = 0.1     # L/s
        V = 2.0     # L
        tau = V / Q  # = 20 s

        result = cstr({"A": c_in, "B": 0.0}, Q, V, [self._first_order(k)], 298.15)
        expected_A = c_in / (1.0 + tau * k)
        assert abs(result["A"] - expected_A) / expected_A < 0.01

    def test_mass_balance(self):
        """[A] + [B] at outlet should equal [A]_in (conservation)."""
        Q, V = 0.1, 1.0
        result = cstr({"A": 2.0, "B": 0.0}, Q, V, [self._first_order(0.3)], 298.15)
        assert abs(result["A"] + result["B"] - 2.0) < 0.01

    def test_higher_flow_reduces_conversion(self):
        """Higher feed flow rate → shorter residence time → less conversion."""
        rxn = self._first_order(0.5)
        result_slow = cstr({"A": 1.0, "B": 0.0}, 0.01, 1.0, [rxn], 298.15)
        result_fast = cstr({"A": 1.0, "B": 0.0}, 1.0, 1.0, [rxn], 298.15)
        # Slow flow (long τ) → more conversion → lower [A] out
        assert result_slow["A"] < result_fast["A"]


class TestMassTransferLimited:
    def test_harmonic_mean_formula(self):
        """1/k_eff = 1/k1 + 1/k2."""
        k1, k2 = 2.0, 3.0
        expected = 1.0 / (1.0/k1 + 1.0/k2)
        assert abs(mass_transfer_limited(k1, k2) - expected) < 1e-12

    def test_dominated_by_slower_step(self):
        """k_eff is always less than the smaller of k1, k2."""
        k1, k2 = 100.0, 0.01
        k_eff = mass_transfer_limited(k1, k2)
        assert k_eff < min(k1, k2)

    def test_equal_rates(self):
        """When k1 = k2 = k, k_eff = k/2."""
        k = 5.0
        assert abs(mass_transfer_limited(k, k) - k / 2) < 1e-10


# ============================================================================
# Electrochem
# ============================================================================

class TestButlerVolmer:
    def test_zero_overpotential_zero_current(self):
        """At η=0, the anodic and cathodic terms cancel: i=0."""
        i = butler_volmer(0.0, i0=10.0, alpha=0.5, n=1, temp=298.15)
        assert abs(i) < 1e-10

    def test_positive_overpotential_positive_current(self):
        """Anodic overpotential drives positive (oxidation) current."""
        i = butler_volmer(0.1, i0=1.0, alpha=0.5, n=1, temp=298.15)
        assert i > 0

    def test_negative_overpotential_negative_current(self):
        """Cathodic overpotential drives negative (reduction) current."""
        i = butler_volmer(-0.1, i0=1.0, alpha=0.5, n=1, temp=298.15)
        assert i < 0

    def test_antisymmetry_at_alpha_half(self):
        """At α=0.5: i(-η) = -i(+η)."""
        i_pos = butler_volmer(0.05, i0=1.0, alpha=0.5, n=1, temp=298.15)
        i_neg = butler_volmer(-0.05, i0=1.0, alpha=0.5, n=1, temp=298.15)
        assert abs(i_pos + i_neg) < 1e-10

    def test_larger_i0_scales_current(self):
        """Doubling i0 doubles current at same overpotential."""
        i1 = butler_volmer(0.1, i0=1.0, alpha=0.5, n=1, temp=298.15)
        i2 = butler_volmer(0.1, i0=2.0, alpha=0.5, n=1, temp=298.15)
        assert abs(i2 / i1 - 2.0) < 1e-8

    def test_formula_exact(self):
        """i = i0 * [exp(α·n·F·η/(R·T)) - exp(-(1-α)·n·F·η/(R·T))]."""
        eta, i0, alpha, n, T = 0.1, 5.0, 0.3, 2, 350.0
        beta_a = alpha * n * FARADAY / (R_GAS * T)
        beta_c = (1.0 - alpha) * n * FARADAY / (R_GAS * T)
        expected = i0 * (math.exp(beta_a * eta) - math.exp(-beta_c * eta))
        result = butler_volmer(eta, i0, alpha, n, T)
        assert abs(result - expected) / abs(expected) < 1e-10


class TestTafelEquation:
    def test_zero_overpotential_returns_i0(self):
        """At η=0, Tafel gives i = i0 * exp(0) = i0."""
        i = tafel_equation(0.0, i0=3.0, alpha=0.5, n=1, temp=298.15)
        assert abs(i - 3.0) < 1e-10

    def test_exponential_increase_with_overpotential(self):
        """Current increases exponentially with overpotential."""
        i1 = tafel_equation(0.05, i0=1.0, alpha=0.5, n=1, temp=298.15)
        i2 = tafel_equation(0.10, i0=1.0, alpha=0.5, n=1, temp=298.15)
        assert i2 > i1

    def test_tafel_vs_butler_volmer_high_overpotential(self):
        """At large anodic η, Tafel ≈ anodic term of Butler-Volmer."""
        eta, i0 = 0.5, 1.0
        i_tafel = tafel_equation(eta, i0, alpha=0.5, n=1, temp=298.15, anodic=True)
        i_bv = butler_volmer(eta, i0, alpha=0.5, n=1, temp=298.15)
        # BV anodic term dominates at large η; ratio should be close to 1
        assert abs(i_tafel / i_bv - 1.0) < 0.01


class TestNernst:
    def test_equal_concentrations_gives_standard(self):
        """c_ox = c_red → E = E°."""
        E = nernst(0.77, 1.0, 1.0, n=1, temp=298.15)
        assert abs(E - 0.77) < 1e-4

    def test_higher_ox_concentration_raises_potential(self):
        """Increasing [ox]/[red] ratio → higher E (more oxidizing)."""
        E_low = nernst(0.0, 1.0, 10.0, n=1, temp=298.15)
        E_high = nernst(0.0, 10.0, 1.0, n=1, temp=298.15)
        assert E_high > E_low

    def test_nernst_formula_exact(self):
        """E = E° + (R·T)/(n·F) · ln(c_ox/c_red)."""
        E0, c_ox, c_red, n, T = 0.34, 2.0, 0.5, 2, 298.15
        expected = E0 + (R_GAS * T / (n * FARADAY)) * math.log(c_ox / c_red)
        result = nernst(E0, c_ox, c_red, n, T)
        assert abs(result - expected) < 1e-6

    def test_nernst_temperature_effect(self):
        """Higher temperature amplifies concentration effect."""
        E_low_T = nernst(0.0, 10.0, 1.0, n=1, temp=250.0)
        E_high_T = nernst(0.0, 10.0, 1.0, n=1, temp=400.0)
        # Both positive (c_ox > c_red), higher T → larger deviation from E°
        assert E_high_T > E_low_T


class TestLimitingCurrent:
    def test_formula_exact(self):
        """i_L = n·F·D·c / δ  (area=1)."""
        n, D, c, delta = 2, 1e-9, 0.01, 1e-5
        expected = n * FARADAY * D * c / delta
        result = limiting_current(n, D, c, delta, area=1.0)
        assert abs(result - expected) / expected < 1e-6

    def test_scales_with_area(self):
        """Doubling area doubles limiting current."""
        i1 = limiting_current(1, 1e-9, 0.1, 1e-5, area=1.0)
        i2 = limiting_current(1, 1e-9, 0.1, 1e-5, area=2.0)
        assert abs(i2 / i1 - 2.0) < 1e-8

    def test_scales_with_electrons(self):
        """Doubling n doubles limiting current (more charge per mole)."""
        i1 = limiting_current(1, 1e-9, 0.1, 1e-5, area=1.0)
        i2 = limiting_current(2, 1e-9, 0.1, 1e-5, area=1.0)
        assert abs(i2 / i1 - 2.0) < 1e-8


# ============================================================================
# Catalysis
# ============================================================================

class TestLangmuirAdsorption:
    def test_half_coverage_at_unit_affinity(self):
        """θ = K·P/(1+K·P). When K·P = 1, θ = 0.5."""
        assert abs(langmuir_adsorption(1.0, K_ads=1.0) - 0.5) < 1e-10

    def test_low_pressure_linear_regime(self):
        """K·P << 1: θ ≈ K·P."""
        K, P = 1e-6, 1.0  # KP = 1e-6
        expected = K * P  # linear regime
        result = langmuir_adsorption(P, K)
        assert abs(result - expected) / expected < 0.01

    def test_high_pressure_saturation(self):
        """K·P >> 1: θ → 1."""
        coverage = langmuir_adsorption(1e6, K_ads=1.0)
        assert coverage > 0.999

    def test_coverage_bounded_zero_to_one(self):
        """Coverage must always be in [0, 1]."""
        for P in [0.0, 0.001, 1.0, 1000.0]:
            c = langmuir_adsorption(P, K_ads=0.1)
            assert 0.0 <= c <= 1.0

    def test_formula_exact(self):
        K, P = 0.3, 5.0
        expected = (K * P) / (1.0 + K * P)
        assert abs(langmuir_adsorption(P, K) - expected) < 1e-10


class TestLangmuirHinshelwood:
    def test_proportional_to_coverages(self):
        """rate = k * θ_A * θ_B (no activation energy)."""
        k, theta_A, theta_B = 2.0, 0.4, 0.6
        rate = langmuir_hinshelwood(theta_A, theta_B, k, activation_energy=0.0)
        assert abs(rate - k * theta_A * theta_B) < 1e-10

    def test_zero_coverage_zero_rate(self):
        assert langmuir_hinshelwood(0.0, 0.5, 1.0) == 0.0
        assert langmuir_hinshelwood(0.5, 0.0, 1.0) == 0.0

    def test_activation_energy_suppresses_rate_at_low_T(self):
        """Positive Ea → lower rate at lower temperature."""
        rate_low = langmuir_hinshelwood(0.5, 0.5, 1.0, activation_energy=50000.0, temp=300.0)
        rate_high = langmuir_hinshelwood(0.5, 0.5, 1.0, activation_energy=50000.0, temp=600.0)
        assert rate_high > rate_low

    def test_activation_energy_formula(self):
        """With Ea>0: k_eff = k_surface * exp(-Ea/(R*T)), rate = k_eff * θ_A * θ_B."""
        k_s, Ea, T, theta_A, theta_B = 1e5, 40000.0, 500.0, 0.3, 0.7
        k_eff = k_s * math.exp(-Ea / (R_GAS * T))
        expected = k_eff * theta_A * theta_B
        result = langmuir_hinshelwood(theta_A, theta_B, k_s, Ea, T)
        assert abs(result - expected) / expected < 1e-10


class TestCompetitiveAdsorption:
    def test_coverages_sum_to_less_than_one(self):
        """Competing species share sites: total coverage ≤ 1."""
        pressures = [100.0, 200.0, 50.0]
        K_ads = [0.01, 0.02, 0.005]
        coverages = competitive_adsorption(pressures, K_ads)
        assert sum(coverages) <= 1.0 + 1e-10

    def test_single_species_matches_langmuir(self):
        """With one species, competitive_adsorption matches langmuir_adsorption."""
        P, K = 5.0, 0.2
        comp = competitive_adsorption([P], [K])
        lang = langmuir_adsorption(P, K)
        assert abs(comp[0] - lang) < 1e-10

    def test_coverages_bounded(self):
        """Each coverage is in [0, 1]."""
        coverages = competitive_adsorption([1.0, 2.0], [0.5, 0.3])
        for c in coverages:
            assert 0.0 <= c <= 1.0


# ============================================================================
# Transport
# ============================================================================

class TestConduction:
    def test_fourier_sign(self):
        """Heat flows opposite to temperature gradient: positive dT/dx → q < 0."""
        q = conduction(temp_gradient=10.0, thermal_conductivity=1.0, area=1.0)
        assert q < 0

    def test_negative_gradient_positive_heat(self):
        """dT/dx < 0 → heat flows in +x direction → q > 0."""
        q = conduction(temp_gradient=-10.0, thermal_conductivity=1.0, area=1.0)
        assert q > 0

    def test_formula_exact(self):
        """q = -k * A * dT/dx."""
        k, A, grad = 50.0, 0.02, 100.0
        expected = -k * A * grad
        assert abs(conduction(grad, k, A) - expected) < 1e-10

    def test_proportional_to_conductivity(self):
        """Doubling k doubles |q|."""
        q1 = conduction(10.0, 1.0, 1.0)
        q2 = conduction(10.0, 2.0, 1.0)
        assert abs(q2 / q1 - 2.0) < 1e-10

    def test_zero_gradient_zero_heat(self):
        assert conduction(0.0, 50.0, 1.0) == 0.0


class TestRadiation:
    def test_equal_temperatures_zero_heat(self):
        """T_surface = T_ambient → no net radiation."""
        q = radiation(400.0, 400.0, emissivity=0.9, area=1.0)
        assert abs(q) < 1e-6

    def test_hotter_surface_positive_heat_loss(self):
        """Hotter surface radiates more than it absorbs → positive q."""
        q = radiation(500.0, 300.0, emissivity=0.9, area=1.0)
        assert q > 0

    def test_cooler_surface_negative_heat(self):
        """Cooler-than-ambient surface absorbs net radiation → q < 0."""
        q = radiation(200.0, 300.0, emissivity=0.9, area=1.0)
        assert q < 0

    def test_stefan_boltzmann_formula(self):
        """q = ε·σ·A·(T_s⁴ - T_a⁴)."""
        T_s, T_a, eps, A = 600.0, 300.0, 0.85, 2.0
        expected = eps * STEFAN_BOLTZMANN * A * (T_s**4 - T_a**4)
        result = radiation(T_s, T_a, eps, A)
        assert abs(result - expected) / expected < 1e-8

    def test_zero_emissivity_zero_radiation(self):
        """Perfect mirror (ε=0) has no net radiation."""
        q = radiation(500.0, 300.0, emissivity=0.0, area=1.0)
        assert abs(q) < 1e-10


class TestNusseltCorrelation:
    def test_laminar_pipe_constant(self):
        """Fully developed laminar pipe flow: Nu = 3.66."""
        Nu = nusselt_correlation(Re=1000.0, Pr=7.0, geometry="pipe")
        assert abs(Nu - 3.66) < 1e-10

    def test_turbulent_pipe_dittus_boelter(self):
        """High-Re turbulent pipe: Nu = 0.023 · Re^0.8 · Pr^0.4."""
        Re, Pr = 1e5, 3.0
        expected = 0.023 * Re**0.8 * Pr**0.4
        result = nusselt_correlation(Re=Re, Pr=Pr, geometry="pipe")
        assert abs(result - expected) / expected < 1e-10

    def test_sphere_minimum_nusselt(self):
        """Sphere at very low Re: Nu → 2 (pure conduction limit)."""
        Nu = nusselt_correlation(Re=0.001, Pr=7.0, geometry="sphere")
        assert abs(Nu - 2.0) < 0.1  # approaches 2 from above

    def test_flat_plate_laminar(self):
        """Laminar flat plate: Nu = 0.664 · Re^0.5 · Pr^(1/3)."""
        Re, Pr = 1e4, 5.0
        expected = 0.664 * Re**0.5 * Pr**(1.0/3.0)
        result = nusselt_correlation(Re=Re, Pr=Pr, geometry="flat_plate")
        assert abs(result - expected) / expected < 1e-10

    def test_nusselt_positive(self):
        """Nu must be positive for all valid geometries."""
        for geom in ["pipe", "flat_plate", "sphere", "cylinder"]:
            Nu = nusselt_correlation(Re=5000.0, Pr=3.0, geometry=geom)
            assert Nu > 0, f"Nu not positive for {geom}"

    def test_turbulent_higher_than_laminar_pipe(self):
        """Turbulent flow transfers more heat than laminar in a pipe."""
        Nu_lam = nusselt_correlation(Re=500.0, Pr=7.0, geometry="pipe")
        Nu_turb = nusselt_correlation(Re=1e5, Pr=7.0, geometry="pipe")
        assert Nu_turb > Nu_lam
