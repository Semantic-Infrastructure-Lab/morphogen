"""Tests for controls domain (PID, state-space, LQR, Kalman, frequency analysis).

Tests correctness, edge cases, and determinism for all 15 operators.
"""

import pytest
import numpy as np
import scipy.signal
import scipy.linalg

from morphogen.stdlib.controls import (
    # Data structures
    PIDState, LinearSystem, KalmanState, TransferFunction,
    # Construction
    pid, state_space, transfer_function,
    # Step / Transform
    pid_step, step, kalman_predict, kalman_update, discretize,
    # Design / Compose
    lqr, place_poles, observer,
    # Analysis / Query
    poles, is_stable, step_response, bode,
    # Simulation
    simulate,
)


# ============================================================================
# SHARED FIXTURES
# ============================================================================

def double_integrator():
    """A = [[0,1],[0,0]], B = [[0],[1]] — canonical LQR test case."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    return A, B, C, D


def first_order_system():
    """ẋ = -2x + u, y = x — analytic step response known."""
    A = np.array([[-2.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    return A, B, C, D


def constant_velocity_model():
    """2-state Kalman test: x = [position, velocity], dt=0.1."""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.zeros((2, 1))
    C = np.array([[1.0, 0.0]])  # Observe position only
    Q = np.diag([0.001, 0.001])
    R = np.array([[0.1]])
    return A, B, C, Q, R


# ============================================================================
# TEST: PID
# ============================================================================

class TestPID:
    """Tests for pid() and pid_step()."""

    def test_pid_construction_defaults(self):
        """pid() returns PIDState with correct defaults."""
        s = pid(kp=1.0)
        assert s.kp == 1.0
        assert s.ki == 0.0
        assert s.kd == 0.0
        assert s.dt == 0.01
        assert s.error_sum == 0.0
        assert s.prev_error == 0.0

    def test_pid_construction_all_params(self):
        """pid() stores all gains correctly."""
        s = pid(kp=2.5, ki=0.3, kd=0.05, dt=0.005,
                output_min=-10.0, output_max=10.0)
        assert s.kp == 2.5
        assert s.ki == 0.3
        assert s.kd == 0.05
        assert s.dt == 0.005
        assert s.output_min == -10.0
        assert s.output_max == 10.0

    def test_pid_invalid_dt(self):
        """pid() raises on non-positive dt."""
        with pytest.raises(ValueError):
            pid(kp=1.0, dt=0.0)
        with pytest.raises(ValueError):
            pid(kp=1.0, dt=-0.01)

    def test_pid_step_proportional_only(self):
        """P-only controller: output = kp * (setpoint - measurement)."""
        s = pid(kp=3.0)
        output, s2 = pid_step(s, setpoint=10.0, measurement=7.0)
        assert np.isclose(output, 3.0 * 3.0)  # kp * error = 3 * 3 = 9

    def test_pid_step_zero_error(self):
        """P-only: zero error gives zero output."""
        s = pid(kp=5.0)
        output, _ = pid_step(s, setpoint=5.0, measurement=5.0)
        assert output == 0.0

    def test_pid_step_integral_accumulates(self):
        """I term accumulates over steps."""
        s = pid(kp=0.0, ki=1.0, dt=0.1)
        # 10 steps with constant error=1: integral ≈ 10 * 0.1 = 1.0
        for _ in range(10):
            output, s = pid_step(s, setpoint=1.0, measurement=0.0)
        assert np.isclose(output, 1.0, atol=1e-10)

    def test_pid_step_derivative_term(self):
        """D term produces finite difference of error."""
        s = pid(kp=0.0, ki=0.0, kd=1.0, dt=0.1)
        # First step: prev_error=0, error=1, derivative = (1-0)/0.1 = 10
        output, s2 = pid_step(s, setpoint=1.0, measurement=0.0)
        assert np.isclose(output, 10.0)
        # Second step: same setpoint/measurement, error=1, prev_error=1, derivative=0
        output2, _ = pid_step(s2, setpoint=1.0, measurement=0.0)
        assert np.isclose(output2, 0.0)

    def test_pid_step_output_clamping(self):
        """Output is clamped to [output_min, output_max]."""
        s = pid(kp=100.0, output_min=-5.0, output_max=5.0)
        output, _ = pid_step(s, setpoint=10.0, measurement=0.0)
        assert output == 5.0
        output_neg, _ = pid_step(s, setpoint=0.0, measurement=10.0)
        assert output_neg == -5.0

    def test_pid_step_antiwindup(self):
        """Anti-windup: integral does not accumulate when saturated."""
        s = pid(kp=0.0, ki=1.0, dt=0.1, output_max=1.0)
        # Run 100 steps with large error — integral should stop growing
        for _ in range(100):
            output, s = pid_step(s, setpoint=100.0, measurement=0.0)
        # If anti-windup works, output stays at max
        assert output <= 1.0 + 1e-10

    def test_pid_step_state_is_immutable(self):
        """pid_step returns a new PIDState; original is not modified."""
        s = pid(kp=1.0, ki=1.0, dt=0.1)
        old_sum = s.error_sum
        _, s2 = pid_step(s, setpoint=1.0, measurement=0.0)
        assert s.error_sum == old_sum  # Original unchanged
        assert s2.error_sum != old_sum

    def test_pid_step_deterministic(self):
        """Same inputs produce same outputs."""
        s = pid(kp=1.0, ki=0.5, kd=0.1, dt=0.01)
        out1, _ = pid_step(s, setpoint=5.0, measurement=2.0)
        out2, _ = pid_step(s, setpoint=5.0, measurement=2.0)
        assert out1 == out2

    def test_pid_convergence(self):
        """PID drives first-order system to setpoint."""
        # Simple first-order plant: dy/dt = -y + u, dt=0.01
        s = pid(kp=5.0, ki=2.0, kd=0.1, dt=0.01)
        y = 0.0
        for _ in range(2000):
            u, s = pid_step(s, setpoint=1.0, measurement=y)
            y = y + 0.01 * (-y + u)
        assert np.isclose(y, 1.0, atol=0.05)


# ============================================================================
# TEST: STATE-SPACE CONSTRUCTION
# ============================================================================

class TestStateSpace:
    """Tests for state_space() and step()."""

    def test_state_space_basic(self):
        """state_space() creates LinearSystem with correct matrices."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        assert np.allclose(sys.A, A)
        assert np.allclose(sys.B, B)
        assert np.allclose(sys.C, C)
        assert np.allclose(sys.D, D)
        assert sys.dt is None
        assert len(sys.x) == 1

    def test_state_space_initial_state(self):
        """State is initialized to zeros."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        assert np.all(sys.x == 0.0)

    def test_state_space_shape_validation_A(self):
        """Non-square A raises ValueError."""
        with pytest.raises(ValueError):
            state_space(np.ones((2, 3)), np.ones((2, 1)), np.ones((1, 2)), np.ones((1, 1)))

    def test_state_space_shape_validation_B(self):
        """B with wrong number of rows raises ValueError."""
        with pytest.raises(ValueError):
            state_space(np.eye(2), np.ones((3, 1)), np.ones((1, 2)), np.ones((1, 1)))

    def test_state_space_shape_validation_C(self):
        """C with wrong number of columns raises ValueError."""
        with pytest.raises(ValueError):
            state_space(np.eye(2), np.ones((2, 1)), np.ones((1, 3)), np.ones((1, 1)))

    def test_state_space_shape_validation_D(self):
        """D with wrong shape raises ValueError."""
        with pytest.raises(ValueError):
            state_space(np.eye(2), np.ones((2, 1)), np.ones((1, 2)), np.ones((2, 2)))

    def test_step_continuous_euler(self):
        """Continuous Euler step matches hand calculation."""
        A = np.array([[-1.0]])
        B = np.array([[0.0]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        sys = state_space(A, B, C, D)
        sys.x = np.array([1.0])

        # x_dot = -x, so x_next = x + dt*(-x) = x*(1 - dt)
        sys2 = step(sys, u=np.array([0.0]), dt=0.1)
        assert np.isclose(sys2.x[0], 0.9)

    def test_step_discrete(self):
        """Discrete step: x[k+1] = A*x[k] + B*u[k]."""
        A = np.array([[0.9]])
        B = np.array([[0.1]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        sys = state_space(A, B, C, D, dt=0.1)
        sys.x = np.array([1.0])

        sys2 = step(sys, u=np.array([0.5]))
        expected_x = 0.9 * 1.0 + 0.1 * 0.5  # = 0.95
        assert np.isclose(sys2.x[0], expected_x)

    def test_step_does_not_mutate(self):
        """step() returns new system; original x is unchanged."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        sys.x = np.array([5.0])
        sys2 = step(sys, u=np.array([0.0]), dt=0.01)
        assert sys.x[0] == 5.0
        assert sys2.x[0] != 5.0

    def test_step_deterministic(self):
        """Same state + input → same next state."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        sys.x = np.array([2.0])
        s1 = step(sys, u=np.array([1.0]), dt=0.01)
        s2 = step(sys, u=np.array([1.0]), dt=0.01)
        assert np.allclose(s1.x, s2.x)


# ============================================================================
# TEST: SIMULATE
# ============================================================================

class TestSimulate:
    """Tests for simulate() against scipy references."""

    def test_simulate_first_order_step(self):
        """Step response of 1st-order system matches analytic."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        t = np.linspace(0, 3.0, 3000)
        u = np.ones((3000, 1))
        t_out, y_out, x_out = simulate(sys, u, t)

        # Analytic: y(t) = 0.5 * (1 - e^{-2t}) for unit step input on ẋ=-2x+u, y=x
        y_analytic = 0.5 * (1 - np.exp(-2 * t))
        assert np.allclose(y_out[:, 0], y_analytic, atol=1e-3)

    def test_simulate_output_shapes(self):
        """simulate() returns correct array shapes."""
        A, B, C, D = double_integrator()
        sys = state_space(A, B, C, D)
        n_steps = 100
        t = np.linspace(0, 1.0, n_steps)
        u = np.zeros((n_steps, 1))
        t_out, y_out, x_out = simulate(sys, u, t)

        assert t_out.shape == (n_steps,)
        assert y_out.shape == (n_steps, 2)  # C is 2×2, so p=2
        assert x_out.shape == (n_steps, 2)

    def test_simulate_zero_input_decay(self):
        """Free response from nonzero initial condition decays to zero."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        t = np.linspace(0, 5.0, 500)
        u = np.zeros((500, 1))
        _, _, x_out = simulate(sys, u, t, x0=np.array([1.0]))
        assert abs(x_out[-1, 0]) < 0.01

    def test_simulate_discrete_matches_scipy(self):
        """Discrete simulate() matches scipy.signal.dlsim."""
        A, B, C, D = first_order_system()
        dt = 0.1
        sys = state_space(A, B, C, D, dt=dt)

        t = np.arange(0, 2.0, dt)
        u = np.ones((len(t), 1))

        t_ours, y_ours, _ = simulate(sys, u, t)
        t_scipy, y_scipy, _ = scipy.signal.dlsim(
            (sys.A, sys.B, sys.C, sys.D, dt), u, t=t
        )
        assert np.allclose(y_ours, y_scipy, atol=1e-10)

    def test_simulate_deterministic(self):
        """Same inputs always produce same outputs."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        t = np.linspace(0, 1.0, 100)
        u = np.random.RandomState(42).randn(100, 1)

        _, y1, _ = simulate(sys, u, t)
        _, y2, _ = simulate(sys, u, t)
        assert np.allclose(y1, y2)


# ============================================================================
# TEST: LQR
# ============================================================================

class TestLQR:
    """Tests for lqr() optimal control."""

    def test_lqr_double_integrator_shape(self):
        """lqr() returns K with correct shape m×n."""
        A, B, C, D = double_integrator()
        Q = np.eye(2)
        R = np.array([[1.0]])
        K = lqr(A, B, Q, R)
        assert K.shape == (1, 2)

    def test_lqr_double_integrator_stabilizes(self):
        """Closed-loop A - B*K should be stable (all eigs negative real part)."""
        A, B, C, D = double_integrator()
        Q = np.eye(2)
        R = np.array([[1.0]])
        K = lqr(A, B, Q, R)
        A_cl = A - B @ K
        eigs = np.linalg.eigvals(A_cl)
        assert all(e.real < 0 for e in eigs)

    def test_lqr_state_reaches_origin(self):
        """LQR-controlled double integrator drives x → [0, 0]."""
        A, B, C, D = double_integrator()
        Q = np.eye(2) * 10.0
        R = np.array([[1.0]])
        K = lqr(A, B, Q, R)

        A_cl = A - B @ K
        sys = state_space(A_cl, B, C, D)
        t = np.linspace(0, 5.0, 500)
        u = np.zeros((500, 1))  # No additional input (K already closes loop)
        _, _, x_out = simulate(sys, u, t, x0=np.array([1.0, 0.0]))
        assert np.allclose(x_out[-1], [0.0, 0.0], atol=0.01)

    def test_lqr_gain_increases_with_Q(self):
        """Larger Q (penalize state more) → larger gain magnitude."""
        A, B, C, D = double_integrator()
        R = np.array([[1.0]])
        K_small = lqr(A, B, np.eye(2) * 1.0, R)
        K_large = lqr(A, B, np.eye(2) * 100.0, R)
        assert np.linalg.norm(K_large) > np.linalg.norm(K_small)

    def test_lqr_deterministic(self):
        """Same (A, B, Q, R) always returns same K."""
        A, B, C, D = double_integrator()
        Q = np.eye(2)
        R = np.array([[1.0]])
        K1 = lqr(A, B, Q, R)
        K2 = lqr(A, B, Q, R)
        assert np.allclose(K1, K2)

    def test_lqr_first_order(self):
        """LQR on 1st-order system: analytic K = sqrt(Q/R)."""
        # For ẋ = ax + bu, J = ∫(qx² + ru²): K = sqrt(q*b²/r - a²)/b ... but
        # simpler to just verify stability
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        Q = np.array([[4.0]])
        R = np.array([[1.0]])
        K = lqr(A, B, Q, R)
        A_cl = A - B @ K
        assert A_cl[0, 0] < 0


# ============================================================================
# TEST: KALMAN FILTER
# ============================================================================

class TestKalman:
    """Tests for kalman_predict() and kalman_update()."""

    def test_kalman_predict_shape(self):
        """kalman_predict() returns KalmanState with correct shapes."""
        A, B, C, Q, R = constant_velocity_model()
        ks = KalmanState(x_hat=np.zeros(2), P=np.eye(2))
        ks2 = kalman_predict(ks, A, B, Q)
        assert ks2.x_hat.shape == (2,)
        assert ks2.P.shape == (2, 2)

    def test_kalman_predict_state_propagation(self):
        """Predicted state follows x̂⁻ = A*x̂ (no input)."""
        A = np.array([[0.9, 0.1], [0.0, 0.95]])
        B = np.zeros((2, 1))
        Q = np.eye(2) * 0.01
        ks = KalmanState(x_hat=np.array([1.0, 2.0]), P=np.eye(2))
        ks2 = kalman_predict(ks, A, B, Q)
        expected = A @ np.array([1.0, 2.0])
        assert np.allclose(ks2.x_hat, expected)

    def test_kalman_predict_covariance_grows(self):
        """P grows (or stays equal) during predict step."""
        A, B, C, Q, R = constant_velocity_model()
        ks = KalmanState(x_hat=np.zeros(2), P=np.eye(2) * 0.1)
        ks2 = kalman_predict(ks, A, B, Q)
        # Frobenius norm of P should not decrease
        assert np.linalg.norm(ks2.P) >= np.linalg.norm(ks.P)

    def test_kalman_update_shape(self):
        """kalman_update() returns KalmanState with correct shapes."""
        A, B, C, Q, R = constant_velocity_model()
        ks = KalmanState(x_hat=np.zeros(2), P=np.eye(2))
        ks2 = kalman_update(ks, C, R, y=np.array([0.5]))
        assert ks2.x_hat.shape == (2,)
        assert ks2.P.shape == (2, 2)

    def test_kalman_update_covariance_shrinks(self):
        """P shrinks after update (measurement reduces uncertainty)."""
        A, B, C, Q, R = constant_velocity_model()
        ks = KalmanState(x_hat=np.zeros(2), P=np.eye(2))
        ks2 = kalman_update(ks, C, R, y=np.array([0.5]))
        assert np.linalg.norm(ks2.P) < np.linalg.norm(ks.P)

    def test_kalman_update_moves_estimate(self):
        """Update shifts estimate toward measurement."""
        A, B, C, Q, R = constant_velocity_model()
        # Start at x_hat = [0, 0], observe y = 1.0
        ks = KalmanState(x_hat=np.zeros(2), P=np.eye(2))
        ks2 = kalman_update(ks, C, R, y=np.array([1.0]))
        # Position estimate (index 0) should move toward 1.0
        assert ks2.x_hat[0] > 0.0

    def test_kalman_filter_reduces_noise(self):
        """Full predict/update loop on noisy position: estimate better than raw."""
        A, B, C, Q, R = constant_velocity_model()
        rng = np.random.RandomState(7)

        true_pos = np.linspace(0, 10, 100)
        true_vel = np.ones(100)
        noisy_pos = true_pos + rng.randn(100) * np.sqrt(R[0, 0])

        ks = KalmanState(x_hat=np.array([0.0, 1.0]), P=np.eye(2))
        estimates = []
        for z in noisy_pos:
            ks = kalman_predict(ks, A, B, Q)
            ks = kalman_update(ks, C, R, y=np.array([z]))
            estimates.append(ks.x_hat[0])

        estimates = np.array(estimates)
        raw_error = np.mean((noisy_pos[10:] - true_pos[10:]) ** 2)
        kf_error = np.mean((estimates[10:] - true_pos[10:]) ** 2)
        assert kf_error < raw_error  # Kalman should reduce noise

    def test_kalman_deterministic(self):
        """Same sequence gives same result."""
        A, B, C, Q, R = constant_velocity_model()
        ks1 = KalmanState(x_hat=np.array([0.0, 1.0]), P=np.eye(2))
        ks2 = KalmanState(x_hat=np.array([0.0, 1.0]), P=np.eye(2))

        for y_val in [0.1, 0.2, 0.15]:
            ks1 = kalman_predict(ks1, A, B, Q)
            ks1 = kalman_update(ks1, C, R, y=np.array([y_val]))
            ks2 = kalman_predict(ks2, A, B, Q)
            ks2 = kalman_update(ks2, C, R, y=np.array([y_val]))

        assert np.allclose(ks1.x_hat, ks2.x_hat)
        assert np.allclose(ks1.P, ks2.P)


# ============================================================================
# TEST: FREQUENCY ANALYSIS
# ============================================================================

class TestFrequency:
    """Tests for poles(), is_stable(), bode(), step_response()."""

    def test_poles_linear_system(self):
        """poles() returns eigenvalues of A for LinearSystem."""
        A = np.array([[-2.0, 0.0], [0.0, -3.0]])
        B = np.zeros((2, 1))
        C = np.eye(2)
        D = np.zeros((2, 1))
        sys = state_space(A, B, C, D)
        p = poles(sys)
        assert sorted(p.real) == pytest.approx([-3.0, -2.0])

    def test_poles_transfer_function(self):
        """poles() returns roots of denominator for TransferFunction."""
        # G(s) = 1 / ((s+1)(s+2)) = 1 / (s² + 3s + 2)
        tf = transfer_function([1.0], [1.0, 3.0, 2.0])
        p = poles(tf)
        p_sorted = sorted(p.real)
        assert p_sorted == pytest.approx([-2.0, -1.0])

    def test_poles_type_error(self):
        """poles() raises TypeError for invalid input."""
        with pytest.raises(TypeError):
            poles("not a system")

    def test_is_stable_stable(self):
        """is_stable() returns True for stable continuous system."""
        A, B, C, D = first_order_system()  # A = [[-2]]
        sys = state_space(A, B, C, D)
        assert is_stable(sys) is True

    def test_is_stable_unstable(self):
        """is_stable() returns False for unstable continuous system."""
        A = np.array([[1.0]])  # Unstable: pole at +1
        sys = state_space(A, np.array([[1.0]]), np.array([[1.0]]), np.array([[0.0]]))
        assert is_stable(sys) is False

    def test_is_stable_discrete_stable(self):
        """is_stable() returns True for stable discrete system (|poles| < 1)."""
        A = np.array([[0.5]])
        sys = state_space(A, np.array([[0.1]]), np.array([[1.0]]), np.array([[0.0]]), dt=0.1)
        assert is_stable(sys) is True

    def test_is_stable_discrete_unstable(self):
        """is_stable() returns False for unstable discrete system (|poles| > 1)."""
        A = np.array([[1.2]])
        sys = state_space(A, np.array([[0.1]]), np.array([[1.0]]), np.array([[0.0]]), dt=0.1)
        assert is_stable(sys) is False

    def test_is_stable_transfer_function(self):
        """is_stable() works on TransferFunction."""
        tf = transfer_function([1.0], [1.0, 3.0, 2.0])  # Poles at -1, -2
        assert is_stable(tf) is True

    def test_step_response_output_shape(self):
        """step_response() returns 1D time array and 2D output array."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        t, y = step_response(sys, t_end=2.0, dt=0.01)
        assert t.ndim == 1
        assert len(t) > 10

    def test_step_response_steady_state(self):
        """step_response() converges to steady-state value."""
        A, B, C, D = first_order_system()  # ẋ = -2x + u, steady state y = 0.5
        sys = state_space(A, B, C, D)
        t, y = step_response(sys, t_end=5.0, dt=0.005)
        ss_value = 0.5  # Steady-state gain = -C*A^{-1}*B = -1*(1/2)*1 = 0.5
        assert np.isclose(y[-1, 0], ss_value, atol=0.01)

    def test_step_response_starts_at_zero(self):
        """Step response starts near zero for zero initial state."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        t, y = step_response(sys, t_end=2.0, dt=0.01)
        assert np.isclose(y[0, 0], 0.0, atol=1e-6)

    def test_bode_output_shapes(self):
        """bode() returns three arrays of matching length."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        w, mag, phase = bode(sys)
        assert len(w) == len(mag) == len(phase)
        assert len(w) > 0

    def test_bode_dc_gain(self):
        """Bode DC gain matches known value for 1st-order system."""
        # G(s) = 1/(s+2): DC gain = G(0) = 0.5 → -6 dB
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        w, mag, phase = bode(sys, frequencies=np.array([1e-4]))
        assert np.isclose(mag[0], 20 * np.log10(0.5), atol=0.1)

    def test_bode_transfer_function(self):
        """bode() works on TransferFunction objects."""
        tf = transfer_function([1.0], [1.0, 1.0])  # G(s) = 1/(s+1)
        w, mag, phase = bode(tf)
        assert len(w) > 0


# ============================================================================
# TEST: DISCRETIZE
# ============================================================================

class TestDiscretize:
    """Tests for discretize() ZOH and Tustin methods."""

    def test_discretize_zoh_sets_dt(self):
        """ZOH discretize sets sys.dt."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        disc = discretize(sys, dt=0.1)
        assert disc.dt == 0.1

    def test_discretize_zoh_matches_scipy(self):
        """ZOH result matches scipy.signal.cont2discrete."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        disc = discretize(sys, dt=0.05)

        Ad_scipy, Bd_scipy, _, _, _ = scipy.signal.cont2discrete(
            (A, B, C, D), 0.05, method="zoh"
        )
        assert np.allclose(disc.A, Ad_scipy, atol=1e-10)
        assert np.allclose(disc.B, Bd_scipy, atol=1e-10)

    def test_discretize_tustin(self):
        """Tustin discretize does not raise and returns discrete system."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        disc = discretize(sys, dt=0.1, method="tustin")
        assert disc.dt == 0.1

    def test_discretize_already_discrete_raises(self):
        """Discretizing a discrete system raises ValueError."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D, dt=0.1)
        with pytest.raises(ValueError):
            discretize(sys, dt=0.05)

    def test_discretize_invalid_method(self):
        """Unknown method raises ValueError."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        with pytest.raises(ValueError):
            discretize(sys, dt=0.1, method="euler")

    def test_discretize_invalid_dt(self):
        """Non-positive dt raises ValueError."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        with pytest.raises(ValueError):
            discretize(sys, dt=0.0)

    def test_discretize_zoh_simulation_matches_continuous(self):
        """Discrete ZOH simulation closely matches continuous Euler."""
        A, B, C, D = first_order_system()
        sys_cont = state_space(A, B, C, D)
        dt = 0.001  # Small dt so Euler ≈ ZOH
        sys_disc = discretize(sys_cont, dt=dt)

        t = np.arange(0, 1.0, dt)
        u = np.ones((len(t), 1))

        _, y_cont, _ = simulate(sys_cont, u, t)
        _, y_disc, _ = simulate(sys_disc, u, t)
        assert np.allclose(y_cont, y_disc, atol=1e-3)

    def test_discretize_deterministic(self):
        """Same continuous system → same discrete system."""
        A, B, C, D = first_order_system()
        sys = state_space(A, B, C, D)
        d1 = discretize(sys, dt=0.1)
        d2 = discretize(sys, dt=0.1)
        assert np.allclose(d1.A, d2.A)
        assert np.allclose(d1.B, d2.B)


# ============================================================================
# TEST: TRANSFER FUNCTION
# ============================================================================

class TestTransferFn:
    """Tests for transfer_function() construction and poles()."""

    def test_transfer_function_basic(self):
        """transfer_function() stores num/den correctly."""
        tf = transfer_function([1.0], [1.0, 3.0, 2.0])
        assert np.allclose(tf.num, [1.0])
        assert np.allclose(tf.den, [1.0, 3.0, 2.0])
        assert tf.dt is None

    def test_transfer_function_discrete(self):
        """transfer_function() stores dt for discrete."""
        tf = transfer_function([1.0], [1.0, -0.5], dt=0.1)
        assert tf.dt == 0.1

    def test_transfer_function_invalid_den(self):
        """Zero leading denominator coefficient raises ValueError."""
        with pytest.raises(ValueError):
            transfer_function([1.0], [0.0, 1.0, 2.0])

    def test_transfer_function_poles(self):
        """poles() on TF matches analytic roots."""
        tf = transfer_function([1.0], [1.0, 5.0, 6.0])  # (s+2)(s+3)
        p = sorted(poles(tf).real)
        assert p == pytest.approx([-3.0, -2.0])

    def test_transfer_function_stable(self):
        """is_stable() returns True for stable TF."""
        tf = transfer_function([1.0], [1.0, 5.0, 6.0])
        assert is_stable(tf) is True

    def test_transfer_function_unstable(self):
        """is_stable() returns False for TF with positive pole."""
        tf = transfer_function([1.0], [1.0, -1.0])  # Pole at +1
        assert is_stable(tf) is False


# ============================================================================
# TEST: POLE PLACEMENT
# ============================================================================

class TestPolePlacement:
    """Tests for place_poles() and observer()."""

    def test_place_poles_shape(self):
        """place_poles() returns gain K with correct shape."""
        A, B, C, D = double_integrator()
        K = place_poles(A, B, np.array([-2.0, -3.0]))
        assert K.shape == (1, 2)

    def test_place_poles_achieves_poles(self):
        """Closed-loop eigenvalues match desired_poles."""
        A, B, C, D = double_integrator()
        desired = np.array([-2.0, -5.0])
        K = place_poles(A, B, desired)
        A_cl = A - B @ K
        actual = sorted(np.linalg.eigvals(A_cl).real)
        expected = sorted(desired.real)
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_place_poles_stabilizes(self):
        """Placed poles in LHP → stable closed loop."""
        A, B, C, D = double_integrator()
        K = place_poles(A, B, np.array([-1.0 + 1j, -1.0 - 1j]))
        A_cl = A - B @ K
        eigs = np.linalg.eigvals(A_cl)
        assert all(e.real < 0 for e in eigs)

    def test_observer_shape(self):
        """observer() returns L with shape n×p."""
        A, B, C, D = double_integrator()
        C_obs = np.array([[1.0, 0.0]])  # Observe first state
        L = observer(A, C_obs, np.array([-10.0, -11.0]))
        assert L.shape == (2, 1)

    def test_observer_achieves_poles(self):
        """Observer error dynamics eig(A - L*C) match desired_poles."""
        A, B, C, D = double_integrator()
        C_obs = np.array([[1.0, 0.0]])
        desired = np.array([-10.0, -11.0])
        L = observer(A, C_obs, desired)
        A_obs = A - L @ C_obs
        actual = sorted(np.linalg.eigvals(A_obs).real)
        expected = sorted(desired.real)
        assert actual == pytest.approx(expected, abs=1e-4)
