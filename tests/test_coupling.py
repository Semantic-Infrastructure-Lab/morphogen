"""Tests for the coupling substrate (morphogen.coupling).

Guards the North-Star deliverable: a reusable co-simulation driver that
co-advances rigorous domains with per-timestep feedback. These assert the
driver's *contract* (ordering, multi-rate, determinism, validation) and the
*value claim* (closed-loop coupling rejects a disturbance an open loop can't).
"""
import numpy as np
import pytest

from morphogen.coupling import Subsystem, couple
from morphogen.stdlib.integrators import integrate
from morphogen.stdlib.controls import pid, pid_step


# --------------------------------------------------------------------------- #
# Contract                                                                     #
# --------------------------------------------------------------------------- #
def test_basic_feedback_converges():
    """A trivial plant + P controller drives the measurement toward setpoint."""
    def plant(x, sig, t, dt):
        x = x + dt * (-x + sig.get("u", 0.0))
        return x, {"y": x}

    def ctrl(_, sig, t, dt):
        return None, {"u": 2.0 * (1.0 - sig.get("y", 0.0))}

    res = couple([Subsystem("plant", 0.0, plant), Subsystem("ctrl", None, ctrl)],
                 steps=500, dt=0.01, record=("y",))
    # steady state of x' = -x + 2(1-x) is x = 2/3
    assert res.history["y"][-1] == pytest.approx(2 / 3, abs=1e-3)
    assert res.times.shape == (500,)
    assert res["y"].shape == (500,)  # __getitem__ shortcut


def test_gauss_seidel_ordering():
    """List order is producer->consumer: a consumer sees the producer's fresh
    output from the *same* step, not a step-delayed one."""
    def producer(n, sig, t, dt):
        n += 1
        return n, {"count": n}

    def consumer(_, sig, t, dt):
        return None, {"seen": sig["count"]}   # must see this step's count

    res = couple([Subsystem("p", 0, producer), Subsystem("c", None, consumer)],
                 steps=5, dt=1.0, record=("count", "seen"))
    # consumer runs after producer each step -> seen == count, no lag
    assert list(res.history["count"]) == [1, 2, 3, 4, 5]
    assert list(res.history["seen"]) == [1, 2, 3, 4, 5]


def test_initial_signals_seed_bus():
    """A consumer that runs before its producer on step 0 sees the seed."""
    def consumer(_, sig, t, dt):
        return None, {"echo": sig["u"]}

    def producer(_, sig, t, dt):
        return None, {"u": 42.0}

    res = couple([Subsystem("c", None, consumer), Subsystem("p", None, producer)],
                 steps=2, dt=1.0, record=("echo",), initial_signals={"u": 7.0})
    # step 0: consumer runs first, sees seeded 7.0; producer then sets 42.0
    # step 1: consumer sees the held 42.0
    assert list(res.history["echo"]) == [7.0, 42.0]


def test_multirate_stride():
    """A stride>1 subsystem advances only every N ticks; its output is held."""
    ticks = []

    def fast(x, sig, t, dt):
        return x, {"fast": t}

    def slow(x, sig, t, dt):
        ticks.append(t)
        x += 1
        return x, {"slow": x}

    res = couple([Subsystem("fast", 0, fast), Subsystem("slow", 0, slow, stride=3)],
                 steps=9, dt=1.0, record=("slow",))
    # slow steps at t=0,3,6 only -> 3 advances
    assert ticks == [0.0, 3.0, 6.0]
    assert res.states["slow"] == 3
    # value held between advances: [1,1,1, 2,2,2, 3,3,3]
    assert list(res.history["slow"]) == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_determinism_bitwise():
    """Same inputs -> bit-identical output (no hidden global RNG state)."""
    def plant(x, sig, t, dt):
        x = x + dt * (-x + sig.get("u", 0.0))
        return x, {"y": x}

    def ctrl(_, sig, t, dt):
        return None, {"u": 1.5 * (1.0 - sig.get("y", 0.0))}

    def make():
        return [Subsystem("plant", 0.0, plant), Subsystem("ctrl", None, ctrl)]

    a = couple(make(), steps=300, dt=0.01, record=("y",))
    b = couple(make(), steps=300, dt=0.01, record=("y",))
    assert np.array_equal(a.history["y"], b.history["y"])


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_duplicate_names_rejected():
    s = Subsystem("dup", 0, lambda st, sig, t, dt: (st, {}))
    with pytest.raises(ValueError, match="unique"):
        couple([s, Subsystem("dup", 0, lambda st, sig, t, dt: (st, {}))],
               steps=1, dt=1.0)


def test_bad_steps_dt_rejected():
    s = [Subsystem("a", 0, lambda st, sig, t, dt: (st, {}))]
    with pytest.raises(ValueError, match="steps"):
        couple(s, steps=0, dt=1.0)
    with pytest.raises(ValueError, match="dt"):
        couple(s, steps=1, dt=0.0)


def test_bad_stride_rejected():
    with pytest.raises(ValueError, match="stride"):
        Subsystem("a", 0, lambda st, sig, t, dt: (st, {}), stride=0)


def test_unrecorded_signal_raises_helpful():
    s = [Subsystem("a", 0, lambda st, sig, t, dt: (st, {"x": 1.0}))]
    with pytest.raises(KeyError, match="never published"):
        couple(s, steps=1, dt=1.0, record=("does_not_exist",))


# --------------------------------------------------------------------------- #
# The value claim: closed loop rejects a disturbance the open loop cannot      #
# --------------------------------------------------------------------------- #
def test_coupled_governor_beats_open_loop():
    """Engine idle governor (controls <-> mechanics), coupled vs. open loop.

    This is the substrate's reason to exist, reduced to an assertion: two
    rigorous domains co-advanced with feedback hold idle under a load lurch that
    a fairly-tuned open loop cannot reject.
    """
    INERTIA, DRAG, K_THROTTLE, TARGET_RPM, DT = 0.05, 0.02, 3.0, 800.0, 0.002
    STEPS = 3000
    omega_target = TARGET_RPM * 2 * np.pi / 60.0
    fixed = (0.15 + DRAG * omega_target) / K_THROTTLE

    def load(t):
        return 0.15 + (1.2 if 2.0 <= t < 3.5 else 0.0) + 0.1 * np.sin(2 * np.pi * 1.5 * t)

    def mechanics_step(state, sig, t, dt):
        omega, gov = state
        throttle = sig.get("throttle", fixed)
        deriv = lambda tt, y: np.array(
            [(K_THROTTLE * throttle - load(t) - DRAG * y[0]) / INERTIA])
        omega = integrate(deriv, omega, t, dt, method="rk4")
        rpm = omega[0] * 60.0 / (2 * np.pi)
        return (omega, gov), {"rpm": rpm}

    def make_controller(coupled):
        def ctrl_step(gov, sig, t, dt):
            if not coupled:
                return gov, {"throttle": fixed}
            throttle, gov = pid_step(gov, TARGET_RPM, sig.get("rpm", TARGET_RPM))
            return gov, {"throttle": throttle}
        return ctrl_step

    def run(coupled):
        omega0 = np.array([omega_target])
        gov0 = pid(kp=0.010, ki=0.040, kd=0.0004, dt=DT, output_min=0.0, output_max=1.0)
        # controller first so mechanics reads a fresh throttle each step
        res = couple(
            [Subsystem("ctrl", gov0, make_controller(coupled)),
             Subsystem("mech", (omega0, None), mechanics_step)],
            steps=STEPS, dt=DT, record=("rpm",),
            initial_signals={"throttle": fixed, "rpm": TARGET_RPM})
        rpm = res.history["rpm"]
        return float(np.sqrt(np.mean((rpm - TARGET_RPM) ** 2)))

    rms_coupled = run(coupled=True)
    rms_open = run(coupled=False)
    # coupled feedback should cut idle-speed error by a wide margin (proof showed ~10x)
    assert rms_coupled < rms_open
    assert rms_open / rms_coupled > 5.0


def test_three_domain_warmup_couples():
    """Three mutually-coupled rigorous domains: thermal <-> mechanics <-> control.

    Guards the multi-domain (>2) path and the physical signature of the coupled
    warm-up: cold engine needs more throttle (temperature-dependent drag), the
    block heats up (work-dependent heating), and the governor self-relaxes the
    throttle as drag falls — none of which is expressible without per-step
    feedback across all three domains.
    """
    INERTIA, K_THROTTLE, TARGET_RPM, DT, STEPS = 0.05, 3.0, 800.0, 0.002, 4000
    HEAT_CAP, P_COMB, H_COOL, T_AMB, T_WARM = 8.0, 220.0, 1.85, 20.0, 90.0
    DRAG_WARM, COLD_FACTOR = 0.02, 2.5

    def drag(T):
        frac_cold = np.clip((T_WARM - T) / (T_WARM - T_AMB), 0.0, 1.0)
        return DRAG_WARM * (1.0 + (COLD_FACTOR - 1.0) * frac_cold)

    def controls(gov, sig, t, dt):
        throttle, gov = pid_step(gov, TARGET_RPM, sig.get("rpm", TARGET_RPM))
        return gov, {"throttle": throttle}

    def thermal(T, sig, t, dt):
        deriv = lambda tt, y: np.array(
            [(P_COMB * sig.get("throttle", 0.0) - H_COOL * (y[0] - T_AMB)) / HEAT_CAP])
        T = integrate(deriv, T, t, dt, method="rk4")
        return T, {"temp": T[0]}

    def mechanics(omega, sig, t, dt):
        b = drag(sig.get("temp", T_AMB))
        deriv = lambda tt, y: np.array(
            [(K_THROTTLE * sig.get("throttle", 0.0) - 0.15 - b * y[0]) / INERTIA])
        omega = integrate(deriv, omega, t, dt, method="rk4")
        rpm = omega[0] * 60.0 / (2 * np.pi)
        return omega, {"omega": omega[0], "rpm": rpm}

    omega0 = np.array([TARGET_RPM * 2 * np.pi / 60.0])
    gov0 = pid(kp=0.012, ki=0.05, kd=0.0005, dt=DT, output_min=0.0, output_max=1.0)
    res = couple(
        [Subsystem("controls", gov0, controls),
         Subsystem("thermal", np.array([T_AMB]), thermal),
         Subsystem("mechanics", omega0, mechanics)],
        steps=STEPS, dt=DT, record=("rpm", "throttle", "temp"),
        initial_signals={"throttle": 0.1, "rpm": TARGET_RPM, "temp": T_AMB})

    temp, throttle, rpm = res["temp"], res["throttle"], res["rpm"]
    # the block warms substantially from cold
    assert temp[0] == pytest.approx(T_AMB, abs=1.0)
    assert temp[-1] > T_AMB + 40
    # governor self-relaxes: cold throttle (early) > warm throttle (settled)
    assert throttle[100] > throttle[-100] + 0.1
    # and it holds idle throughout (loose bound; RK4 + PID keeps it close)
    assert float(np.sqrt(np.mean((rpm - TARGET_RPM) ** 2))) < 80.0
