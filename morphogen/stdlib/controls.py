"""Control Systems Domain for Morphogen.

Implements classical and modern control theory:
- PID control (proportional-integral-derivative)
- State-space representation and simulation
- LQR optimal state feedback (scipy.linalg.solve_continuous_are)
- Kalman filter (predict/update cycle)
- Frequency analysis (Bode, poles, step response)
- Discretization (ZOH, Tustin/bilinear)

Dependencies: numpy, scipy (already in pyproject.toml)
"""

from typing import Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg
import scipy.signal

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PIDState:
    """Carries PID controller state for stateless-style stepping.

    Attributes:
        error_sum: Integral accumulator
        prev_error: Previous error for derivative term
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        dt: Timestep in seconds
        output_min: Lower saturation limit
        output_max: Upper saturation limit
    """
    error_sum: float = 0.0
    prev_error: float = 0.0
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 0.01
    output_min: float = -np.inf
    output_max: float = np.inf


@dataclass
class LinearSystem:
    """State-space system (A, B, C, D) with current state.

    Attributes:
        A: n×n state matrix
        B: n×m input matrix
        C: p×n output matrix
        D: p×m feedthrough matrix
        dt: None = continuous; float = discrete with this timestep
        x: Current state vector (n,)
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    dt: Optional[float] = None
    x: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)


@dataclass
class KalmanState:
    """Kalman filter state: estimate and error covariance.

    Attributes:
        x_hat: State estimate vector (n,)
        P: Error covariance matrix (n×n)
    """
    x_hat: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        self.x_hat = np.asarray(self.x_hat, dtype=float)
        self.P = np.asarray(self.P, dtype=float)


@dataclass
class TransferFunction:
    """Transfer function as ratio of polynomials.

    Attributes:
        num: Numerator polynomial coefficients (highest power first)
        den: Denominator polynomial coefficients (highest power first)
        dt: None = continuous; float = discrete
    """
    num: np.ndarray
    den: np.ndarray
    dt: Optional[float] = None

    def __post_init__(self):
        self.num = np.asarray(self.num, dtype=float)
        self.den = np.asarray(self.den, dtype=float)


# ============================================================================
# LAYER 1 — CONSTRUCTION
# ============================================================================

@operator(
    domain="controls",
    category=OpCategory.CONSTRUCT,
    signature="(kp: float, ki: float, kd: float, dt: float, output_min: float, output_max: float) -> PIDState",
    deterministic=True,
    doc="Create a PID controller state with given gains"
)
def pid(
    kp: float,
    ki: float = 0.0,
    kd: float = 0.0,
    dt: float = 0.01,
    output_min: float = -np.inf,
    output_max: float = np.inf
) -> PIDState:
    """Create a PID controller state with given gains.

    Returns a fresh PIDState ready for use with pid_step().

    Args:
        kp: Proportional gain
        ki: Integral gain (default 0)
        kd: Derivative gain (default 0)
        dt: Timestep in seconds
        output_min: Lower saturation limit
        output_max: Upper saturation limit

    Returns:
        Initialized PIDState

    Example:
        state = controls.pid(kp=2.0, ki=0.5, kd=0.1, dt=0.01)
        output, state = controls.pid_step(state, setpoint=10.0, measurement=0.0)
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    return PIDState(
        kp=kp, ki=ki, kd=kd, dt=dt,
        output_min=output_min, output_max=output_max
    )


@operator(
    domain="controls",
    category=OpCategory.CONSTRUCT,
    signature="(A: ndarray, B: ndarray, C: ndarray, D: ndarray, dt: Optional[float]) -> LinearSystem",
    deterministic=True,
    doc="Create a state-space system (A, B, C, D) with shape validation"
)
def state_space(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    dt: Optional[float] = None
) -> LinearSystem:
    """Create a state-space system with matrix shape validation.

    State-space form:
        ẋ = A*x + B*u    (continuous)  or  x[k+1] = A*x[k] + B*u[k]  (discrete)
        y = C*x + D*u

    Args:
        A: n×n state matrix
        B: n×m input matrix
        C: p×n output matrix
        D: p×m feedthrough matrix
        dt: None for continuous, timestep float for discrete

    Returns:
        LinearSystem with x initialized to zeros

    Example:
        # First-order system: ẋ = -2x + u, y = x
        sys = controls.state_space(
            A=np.array([[-2.0]]),
            B=np.array([[1.0]]),
            C=np.array([[1.0]]),
            D=np.array([[0.0]])
        )
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got shape {A.shape}")
    if B.shape[0] != n:
        raise ValueError(f"B must have {n} rows (states), got {B.shape[0]}")
    if C.shape[1] != n:
        raise ValueError(f"C must have {n} columns (states), got {C.shape[1]}")
    m = B.shape[1]
    p = C.shape[0]
    if D.shape != (p, m):
        raise ValueError(f"D must be ({p}×{m}), got {D.shape}")

    return LinearSystem(A=A, B=B, C=C, D=D, dt=dt, x=np.zeros(n))


@operator(
    domain="controls",
    category=OpCategory.CONSTRUCT,
    signature="(num: ndarray, den: ndarray, dt: Optional[float]) -> TransferFunction",
    deterministic=True,
    doc="Create a transfer function from numerator and denominator polynomial coefficients"
)
def transfer_function(
    num: np.ndarray,
    den: np.ndarray,
    dt: Optional[float] = None
) -> TransferFunction:
    """Create a transfer function from polynomial coefficients.

    Coefficients are ordered highest power first (numpy convention).

    Args:
        num: Numerator coefficients, e.g. [1] for G(s) = 1/(s+2)
        den: Denominator coefficients, e.g. [1, 2]
        dt: None for continuous, float for discrete

    Returns:
        TransferFunction

    Example:
        # G(s) = 10 / (s² + 3s + 10)
        tf = controls.transfer_function([10], [1, 3, 10])
    """
    num = np.atleast_1d(np.asarray(num, dtype=float))
    den = np.atleast_1d(np.asarray(den, dtype=float))
    if len(den) == 0 or den[0] == 0:
        raise ValueError("Denominator must have at least one nonzero leading coefficient")
    return TransferFunction(num=num, den=den, dt=dt)


# ============================================================================
# LAYER 2 — STEP / TRANSFORM
# ============================================================================

@operator(
    domain="controls",
    category=OpCategory.TRANSFORM,
    signature="(state: PIDState, setpoint: float, measurement: float) -> Tuple[float, PIDState]",
    deterministic=True,
    doc="One PID iteration with anti-windup: returns (control_output, updated_state)"
)
def pid_step(
    state: PIDState,
    setpoint: float,
    measurement: float
) -> Tuple[float, PIDState]:
    """One PID iteration with anti-windup clamping.

    Computes: output = kp*e + ki*∫e + kd*(de/dt)
    Anti-windup: only integrates when output is not saturated.

    Args:
        state: Current PIDState
        setpoint: Desired value
        measurement: Actual measured value

    Returns:
        Tuple of (control_output, updated_PIDState)

    Example:
        state = controls.pid(kp=1.0, ki=0.1, dt=0.01)
        for t in range(100):
            measurement = system_output()
            u, state = controls.pid_step(state, setpoint=1.0, measurement=measurement)
    """
    error = setpoint - measurement
    derivative = (error - state.prev_error) / state.dt

    # Tentative integral (for anti-windup check)
    new_error_sum = state.error_sum + error * state.dt

    # Compute output
    output = state.kp * error + state.ki * new_error_sum + state.kd * derivative

    # Clamp output
    output_clamped = float(np.clip(output, state.output_min, state.output_max))

    # Anti-windup: only accumulate integral if not saturated
    if output_clamped == output:
        final_error_sum = new_error_sum
    else:
        final_error_sum = state.error_sum  # Don't accumulate when saturated

    new_state = PIDState(
        error_sum=final_error_sum,
        prev_error=error,
        kp=state.kp,
        ki=state.ki,
        kd=state.kd,
        dt=state.dt,
        output_min=state.output_min,
        output_max=state.output_max
    )
    return output_clamped, new_state


@operator(
    domain="controls",
    category=OpCategory.TRANSFORM,
    signature="(sys: LinearSystem, u: ndarray, dt: Optional[float]) -> LinearSystem",
    deterministic=True,
    doc="Advance state-space system by one timestep given input u"
)
def step(
    sys: LinearSystem,
    u: np.ndarray,
    dt: Optional[float] = None
) -> LinearSystem:
    """Advance state-space system by one timestep.

    For discrete systems: x[k+1] = A*x[k] + B*u[k]
    For continuous systems: uses Euler integration with given dt.

    Args:
        sys: LinearSystem with current state
        u: Input vector (m,)
        dt: Timestep for continuous systems (uses sys.dt if None)

    Returns:
        New LinearSystem with updated state x

    Example:
        sys = controls.state_space(A, B, C, D)
        for u_k in inputs:
            sys = controls.step(sys, u_k, dt=0.01)
            y = sys.C @ sys.x
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))

    if sys.dt is not None:
        # Discrete: exact update
        x_new = sys.A @ sys.x + sys.B @ u
    else:
        # Continuous: Euler integration
        effective_dt = dt if dt is not None else 0.01
        x_dot = sys.A @ sys.x + sys.B @ u
        x_new = sys.x + effective_dt * x_dot

    import copy
    new_sys = copy.copy(sys)
    new_sys.x = x_new
    return new_sys


@operator(
    domain="controls",
    category=OpCategory.TRANSFORM,
    signature="(state: KalmanState, A: ndarray, B: ndarray, Q: ndarray, u: Optional[ndarray]) -> KalmanState",
    deterministic=True,
    doc="Kalman filter predict step: x̂⁻ = A*x̂ + B*u; P⁻ = A*P*Aᵀ + Q"
)
def kalman_predict(
    state: KalmanState,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    u: Optional[np.ndarray] = None
) -> KalmanState:
    """Kalman filter predict (time update) step.

    Propagates state estimate and covariance forward:
        x̂⁻ = A*x̂ + B*u
        P⁻ = A*P*Aᵀ + Q

    Args:
        state: Current KalmanState (x_hat, P)
        A: State transition matrix (n×n)
        B: Input matrix (n×m)
        Q: Process noise covariance (n×n)
        u: Control input (m,), defaults to zeros

    Returns:
        Predicted KalmanState

    Example:
        ks = KalmanState(x_hat=np.zeros(2), P=np.eye(2))
        ks = controls.kalman_predict(ks, A, B, Q)
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))

    if u is None:
        u = np.zeros(B.shape[1])
    u = np.atleast_1d(np.asarray(u, dtype=float))

    x_pred = A @ state.x_hat + B @ u
    P_pred = A @ state.P @ A.T + Q

    return KalmanState(x_hat=x_pred, P=P_pred)


@operator(
    domain="controls",
    category=OpCategory.TRANSFORM,
    signature="(state: KalmanState, C: ndarray, R: ndarray, y: ndarray) -> KalmanState",
    deterministic=True,
    doc="Kalman filter update step: K = P⁻Cᵀ(CP⁻Cᵀ+R)⁻¹; x̂ = x̂⁻ + K(y - Cx̂⁻)"
)
def kalman_update(
    state: KalmanState,
    C: np.ndarray,
    R: np.ndarray,
    y: np.ndarray
) -> KalmanState:
    """Kalman filter update (measurement update) step.

    Incorporates measurement y to refine estimate:
        K = P⁻Cᵀ(CP⁻Cᵀ + R)⁻¹     (Kalman gain)
        x̂ = x̂⁻ + K(y - Cx̂⁻)        (state update)
        P = (I - KC)P⁻               (covariance update)

    Args:
        state: Predicted KalmanState from kalman_predict()
        C: Observation matrix (p×n)
        R: Measurement noise covariance (p×p)
        y: Measurement vector (p,)

    Returns:
        Updated KalmanState with reduced uncertainty

    Example:
        ks = controls.kalman_update(ks, C, R, y_measured)
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))

    n = state.P.shape[0]
    S = C @ state.P @ C.T + R          # Innovation covariance
    K = state.P @ C.T @ np.linalg.inv(S)  # Kalman gain

    innovation = y - C @ state.x_hat
    x_new = state.x_hat + K @ innovation
    P_new = (np.eye(n) - K @ C) @ state.P

    return KalmanState(x_hat=x_new, P=P_new)


@operator(
    domain="controls",
    category=OpCategory.TRANSFORM,
    signature="(sys: LinearSystem, dt: float, method: str) -> LinearSystem",
    deterministic=True,
    doc="Discretize continuous state-space system using ZOH or Tustin method"
)
def discretize(
    sys: LinearSystem,
    dt: float,
    method: str = "zoh"
) -> LinearSystem:
    """Discretize a continuous-time state-space system.

    Converts continuous (A, B, C, D) to discrete equivalents.
    ZOH (zero-order hold) is exact for piecewise-constant inputs.
    Tustin (bilinear) preserves frequency response better.

    Args:
        sys: Continuous LinearSystem (sys.dt must be None)
        dt: Desired sampling period
        method: "zoh" (default) or "tustin"

    Returns:
        Discrete LinearSystem with sys.dt = dt

    Example:
        disc_sys = controls.discretize(cont_sys, dt=0.01)
        # Now sys.dt == 0.01
    """
    if sys.dt is not None:
        raise ValueError("System is already discrete (sys.dt is not None)")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    if method == "zoh":
        Ad, Bd, _, _, _ = scipy.signal.cont2discrete(
            (sys.A, sys.B, sys.C, sys.D), dt, method="zoh"
        )
    elif method == "tustin":
        Ad, Bd, _, _, _ = scipy.signal.cont2discrete(
            (sys.A, sys.B, sys.C, sys.D), dt, method="bilinear"
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'zoh' or 'tustin'.")

    return LinearSystem(A=Ad, B=Bd, C=sys.C.copy(), D=sys.D.copy(), dt=dt, x=sys.x.copy())


# ============================================================================
# LAYER 3 — DESIGN / COMPOSE
# ============================================================================

@operator(
    domain="controls",
    category=OpCategory.COMPOSE,
    signature="(A: ndarray, B: ndarray, Q: ndarray, R: ndarray) -> ndarray",
    deterministic=True,
    doc="LQR optimal state feedback: solve CARE, return gain K such that u = -K*x"
)
def lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """Compute LQR optimal state feedback gain.

    Minimizes the infinite-horizon cost:
        J = ∫₀^∞ (xᵀQx + uᵀRu) dt

    via the continuous algebraic Riccati equation (CARE):
        AᵀP + PA - PBR⁻¹BᵀP + Q = 0

    Optimal control law: u = -K*x, where K = R⁻¹BᵀP

    Args:
        A: n×n state matrix
        B: n×m input matrix
        Q: n×n state cost matrix (positive semidefinite)
        R: m×m input cost matrix (positive definite)

    Returns:
        K: m×n gain matrix

    Example:
        # Double integrator: park at origin
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        K = controls.lqr(A, B, np.eye(2), np.array([[1.0]]))
        # Close loop: A_cl = A - B @ K
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


@operator(
    domain="controls",
    category=OpCategory.COMPOSE,
    signature="(A: ndarray, B: ndarray, desired_poles: ndarray) -> ndarray",
    deterministic=True,
    doc="State feedback pole placement: return K such that eig(A - B*K) = desired_poles"
)
def place_poles(
    A: np.ndarray,
    B: np.ndarray,
    desired_poles: np.ndarray
) -> np.ndarray:
    """Pole placement for state feedback design.

    Computes gain K such that eig(A - B*K) = desired_poles.

    Args:
        A: n×n state matrix
        B: n×m input matrix
        desired_poles: Array of n desired closed-loop pole locations (complex)

    Returns:
        K: m×n gain matrix

    Example:
        K = controls.place_poles(A, B, desired_poles=[-1+0j, -2+0j])
        A_cl = A - B @ K  # Closed-loop system
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    desired_poles = np.asarray(desired_poles)

    result = scipy.signal.place_poles(A, B, desired_poles)
    return result.gain_matrix


@operator(
    domain="controls",
    category=OpCategory.COMPOSE,
    signature="(A: ndarray, C: ndarray, desired_poles: ndarray) -> ndarray",
    deterministic=True,
    doc="Luenberger observer gain: return L such that eig(A - L*C) = desired_poles"
)
def observer(
    A: np.ndarray,
    C: np.ndarray,
    desired_poles: np.ndarray
) -> np.ndarray:
    """Design Luenberger observer gain via pole placement.

    Computes L such that eig(A - L*C) = desired_poles.
    Uses duality: transposes A and C for pole placement on (Aᵀ - Cᵀ*Lᵀ).

    Args:
        A: n×n state matrix
        C: p×n observation matrix
        desired_poles: n desired observer pole locations (faster than plant poles)

    Returns:
        L: n×p observer gain matrix

    Example:
        # Observer poles 5× faster than plant poles
        L = controls.observer(A, C, desired_poles=[-10, -10])
        # Observer: x̂_dot = A*x̂ + B*u + L*(y - C*x̂)
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    desired_poles = np.asarray(desired_poles)

    # Duality: design gain on transposed system
    result = scipy.signal.place_poles(A.T, C.T, desired_poles)
    return result.gain_matrix.T


# ============================================================================
# LAYER 4 — ANALYSIS / QUERY
# ============================================================================

@operator(
    domain="controls",
    category=OpCategory.QUERY,
    signature="(sys: Union[LinearSystem, TransferFunction]) -> ndarray",
    deterministic=True,
    doc="Compute system poles (eigenvalues of A, or roots of denominator)"
)
def poles(sys) -> np.ndarray:
    """Compute the poles of a linear system.

    For LinearSystem: returns eigenvalues of A.
    For TransferFunction: returns roots of the denominator polynomial.

    Args:
        sys: LinearSystem or TransferFunction

    Returns:
        Complex array of pole locations

    Example:
        p = controls.poles(sys)
        print("Stable:", all(p.real < 0))
    """
    if isinstance(sys, LinearSystem):
        return np.linalg.eigvals(sys.A)
    elif isinstance(sys, TransferFunction):
        return np.roots(sys.den)
    else:
        raise TypeError(f"Expected LinearSystem or TransferFunction, got {type(sys)}")


@operator(
    domain="controls",
    category=OpCategory.QUERY,
    signature="(sys: Union[LinearSystem, TransferFunction]) -> bool",
    deterministic=True,
    doc="Check stability: all poles have negative real part (continuous) or magnitude < 1 (discrete)"
)
def is_stable(sys) -> bool:
    """Check if a system is stable.

    Continuous: stable if all pole real parts < 0.
    Discrete: stable if all pole magnitudes < 1.

    Args:
        sys: LinearSystem or TransferFunction

    Returns:
        True if stable, False otherwise

    Example:
        if not controls.is_stable(sys):
            print("WARNING: Unstable system!")
    """
    p = poles(sys)

    if isinstance(sys, LinearSystem) and sys.dt is not None:
        # Discrete
        return bool(np.all(np.abs(p) < 1.0))
    elif isinstance(sys, TransferFunction) and sys.dt is not None:
        return bool(np.all(np.abs(p) < 1.0))
    else:
        # Continuous
        return bool(np.all(p.real < 0))


@operator(
    domain="controls",
    category=OpCategory.QUERY,
    signature="(sys: LinearSystem, t_end: float, dt: float) -> Tuple[ndarray, ndarray]",
    deterministic=True,
    doc="Simulate unit step response: returns (t, y) arrays"
)
def step_response(
    sys: LinearSystem,
    t_end: float,
    dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the unit step response of a linear system.

    Applies a unit step input (u=1 for t≥0) starting from x=0.

    Args:
        sys: LinearSystem (continuous or discrete)
        t_end: Simulation end time
        dt: Timestep (used for continuous systems; ignored for discrete)

    Returns:
        Tuple (t, y) where t is time array and y is output array

    Example:
        t, y = controls.step_response(sys, t_end=5.0)
        # y[-1] is the steady-state value
    """
    if sys.dt is not None:
        # Use scipy for discrete
        t_arr = np.arange(0, t_end, sys.dt)
        m = sys.B.shape[1]
        u = np.ones((len(t_arr), m))
        t_out, y_out, _ = scipy.signal.dlsim(
            (sys.A, sys.B, sys.C, sys.D, sys.dt), u, t=t_arr
        )
        return t_out, y_out
    else:
        # Continuous: use simulate with step input
        n_steps = int(t_end / dt)
        t_arr = np.linspace(0, t_end, n_steps)
        m = sys.B.shape[1]
        u = np.ones((n_steps, m))
        t_out, y_out, _ = simulate(sys, u, t_arr)
        return t_out, y_out


@operator(
    domain="controls",
    category=OpCategory.QUERY,
    signature="(sys: Union[LinearSystem, TransferFunction], frequencies: Optional[ndarray]) -> Tuple[ndarray, ndarray, ndarray]",
    deterministic=True,
    doc="Bode plot data: returns (frequencies, magnitude_dB, phase_deg)"
)
def bode(
    sys,
    frequencies: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bode plot data (frequency response).

    Args:
        sys: LinearSystem or TransferFunction
        frequencies: Angular frequency array (rad/s). If None, auto-determined.

    Returns:
        Tuple (w, mag_dB, phase_deg) where:
        - w: Angular frequencies (rad/s)
        - mag_dB: Magnitude in decibels
        - phase_deg: Phase in degrees

    Example:
        w, mag, phase = controls.bode(sys)
        # Bandwidth ≈ where mag drops 3 dB below DC gain
    """
    if isinstance(sys, LinearSystem):
        scipy_sys = scipy.signal.StateSpace(sys.A, sys.B, sys.C, sys.D)
        if sys.dt is not None:
            scipy_sys = scipy.signal.StateSpace(sys.A, sys.B, sys.C, sys.D, dt=sys.dt)
        w, mag, phase = scipy.signal.bode(scipy_sys, w=frequencies)
    elif isinstance(sys, TransferFunction):
        if sys.dt is not None:
            scipy_tf = scipy.signal.dlti(sys.num, sys.den, dt=sys.dt)
            # Convert to continuous for bode
            w_out, H = scipy.signal.dfreqz(sys.num, sys.den, worN=frequencies if frequencies is not None else 512)
            w = w_out / sys.dt
            mag = 20 * np.log10(np.abs(H) + 1e-300)
            phase = np.degrees(np.angle(H))
            return w, mag, phase
        else:
            scipy_tf = scipy.signal.TransferFunction(sys.num, sys.den)
            w, mag, phase = scipy.signal.bode(scipy_tf, w=frequencies)
    else:
        raise TypeError(f"Expected LinearSystem or TransferFunction, got {type(sys)}")

    return w, mag, phase


# ============================================================================
# LAYER 5 — SIMULATION (INTEGRATE)
# ============================================================================

@operator(
    domain="controls",
    category=OpCategory.INTEGRATE,
    signature="(sys: LinearSystem, u: ndarray, t: ndarray, x0: Optional[ndarray]) -> Tuple[ndarray, ndarray, ndarray]",
    deterministic=True,
    doc="Full time-domain simulation: returns (t, y, x) for input sequence u"
)
def simulate(
    sys: LinearSystem,
    u: np.ndarray,
    t: np.ndarray,
    x0: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a linear system over a full time horizon.

    For discrete systems: uses scipy.signal.dlsim (exact).
    For continuous systems: Euler integration loop.

    Args:
        sys: LinearSystem (continuous or discrete)
        u: Input sequence, shape (n_steps, m_inputs) or (n_steps,) for SISO
        t: Time array, shape (n_steps,)
        x0: Initial state (n,). Defaults to zeros if None.

    Returns:
        Tuple (t, y, x) where:
        - t: Time array (n_steps,)
        - y: Output array (n_steps, p_outputs)
        - x: State trajectory (n_steps, n_states)

    Example:
        t = np.linspace(0, 5, 500)
        u = np.ones((500, 1))  # Step input
        t, y, x = controls.simulate(sys, u, t, x0=np.zeros(2))
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    if u.ndim == 1 or (u.ndim == 2 and u.shape[0] == 1):
        u = u.reshape(-1, 1)
    t = np.asarray(t, dtype=float)
    n_steps = len(t)

    if x0 is None:
        x0 = np.zeros(sys.A.shape[0])
    x0 = np.asarray(x0, dtype=float)

    if sys.dt is not None:
        # Discrete: scipy.signal.dlsim handles this exactly
        t_out, y_out, x_out = scipy.signal.dlsim(
            (sys.A, sys.B, sys.C, sys.D, sys.dt), u, t=t, x0=x0
        )
        return t_out, y_out, x_out
    else:
        # Continuous: Euler integration
        n = sys.A.shape[0]
        p = sys.C.shape[0]

        x_traj = np.zeros((n_steps, n))
        y_traj = np.zeros((n_steps, p))

        x = x0.copy()
        for k in range(n_steps):
            uk = u[k]
            y_k = sys.C @ x + sys.D @ uk
            x_traj[k] = x
            y_traj[k] = y_k

            if k < n_steps - 1:
                dt_k = t[k + 1] - t[k]
                x_dot = sys.A @ x + sys.B @ uk
                x = x + dt_k * x_dot

        return t, y_traj, x_traj


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data structures
    'PIDState',
    'LinearSystem',
    'KalmanState',
    'TransferFunction',

    # Construction
    'pid',
    'state_space',
    'transfer_function',

    # Step / Transform
    'pid_step',
    'step',
    'kalman_predict',
    'kalman_update',
    'discretize',

    # Design / Compose
    'lqr',
    'place_poles',
    'observer',

    # Analysis / Query
    'poles',
    'is_stable',
    'step_response',
    'bode',

    # Simulation
    'simulate',
]
