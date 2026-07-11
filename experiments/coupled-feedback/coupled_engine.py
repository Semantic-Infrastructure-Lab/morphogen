"""Three-domain coupled engine — a cold-start warm-up, on the coupling substrate.

Where ``engine_governor.py`` proved the *smallest* honest coupling (2 domains, a
hand-written loop), this shows the same idea generalized onto the reusable
:func:`morphogen.coupling.couple` substrate and scaled to **three mutually-coupled
rigorous domains**:

  THERMAL   (integrators, RK4)  block temperature T:
              C·dT/dt = P_comb·throttle + P_fric(ω) − h·(T − T_amb)
  MECHANICS (integrators, RK4)  crank speed ω:
              I·dω/dt = kτ·throttle − load(t) − b(T)·ω        ← drag depends on T
  CONTROLS  (PID governor)      throttle to hold idle RPM.

The physics that makes it a genuine multiphysics problem: **viscous drag depends
on temperature** (cold oil drags harder), and **temperature depends on how hard
the engine is working** (combustion + friction heat it). So the three states feed
back on each other every timestep — you cannot get this from any single-domain
library, and it is exactly the kind of thing the (retired) one-shot composer never
could express.

The story you see: start the engine cold. Drag is high, so the governor has to run
extra throttle to hold 800 RPM. As the block warms, drag falls, and the governor
backs the throttle off on its own — the classic fast-idle-then-settle warm-up —
all while holding RPM flat against a mid-run load lurch. We plot all three signals
and sonify RPM so you can hear it settle.

Run:  python experiments/coupled-feedback/coupled_engine.py
"""
from pathlib import Path
import numpy as np

from morphogen.coupling import Subsystem, couple           # the substrate
from morphogen.stdlib.integrators import integrate          # THERMAL + MECHANICS
from morphogen.stdlib.controls import pid, pid_step         # CONTROLS
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer

OUT = Path(__file__).parent
SR = 22050

# --- plant parameters ---------------------------------------------------------
INERTIA = 0.05          # kg·m²   crank + flywheel
K_THROTTLE = 3.0        # N·m per unit throttle
TARGET_RPM = 800.0      # idle setpoint
DT = 0.002              # 500 Hz sim/control rate
STEPS = 6000            # 12 s

# thermal
HEAT_CAP = 8.0          # J/K     lumped block heat capacity (small, so it warms fast on-screen)
P_COMB = 220.0          # W per unit throttle (combustion heat into the block)
K_FRIC_HEAT = 0.02      # W per (N·m·s) of drag power -> heat
H_COOL = 1.85           # W/K     convective cooling (tuned so it settles near operating temp)
T_AMB = 20.0            # °C      ambient / cold-start temperature
T_WARM = 90.0           # °C      fully-warm operating temperature

# temperature-dependent drag: cold -> more drag
DRAG_WARM = 0.02        # N·m·s   viscous drag when fully warm
COLD_DRAG_FACTOR = 2.5  # drag multiplier at ambient vs. warm


def drag_coeff(T):
    """Viscous drag rises as the block gets colder (thick cold oil)."""
    frac_cold = np.clip((T_WARM - T) / (T_WARM - T_AMB), 0.0, 1.0)
    return DRAG_WARM * (1.0 + (COLD_DRAG_FACTOR - 1.0) * frac_cold)


def load_torque(t):
    """External load: baseline + an AC-compressor lurch + gentle ripple."""
    lurch = 1.0 if 6.0 <= t < 8.5 else 0.0
    return 0.15 + lurch + 0.08 * np.sin(2 * np.pi * 1.5 * t)


def rpm_of(omega):
    return omega * 60.0 / (2 * np.pi)


# ---- three subsystems, wired by named signals -------------------------------
def controls_step(gov, sig, t, dt):
    """PID governor: read measured RPM, set throttle to hold idle."""
    throttle, gov = pid_step(gov, TARGET_RPM, sig.get("rpm", TARGET_RPM))
    return gov, {"throttle": throttle}


def thermal_step(T, sig, t, dt):
    """Block temperature ODE (RK4). Heats from combustion + friction, cools to ambient."""
    throttle = sig.get("throttle", 0.0)
    omega = sig.get("omega", 0.0)
    p_fric = K_FRIC_HEAT * drag_coeff(T[0]) * omega ** 2
    deriv = lambda tt, y: np.array(
        [(P_COMB * throttle + p_fric - H_COOL * (y[0] - T_AMB)) / HEAT_CAP])
    T = integrate(deriv, T, t, dt, method="rk4")
    return T, {"temp": T[0]}


def mechanics_step(omega, sig, t, dt):
    """Crank-speed ODE (RK4). Drive torque vs. temperature-dependent drag + load."""
    throttle = sig.get("throttle", 0.0)
    temp = sig.get("temp", T_AMB)
    b = drag_coeff(temp)
    deriv = lambda tt, y: np.array(
        [(K_THROTTLE * throttle - load_torque(t) - b * y[0]) / INERTIA])
    omega = integrate(deriv, omega, t, dt, method="rk4")
    return omega, {"omega": omega[0], "rpm": rpm_of(omega[0])}


def run():
    omega0 = np.array([TARGET_RPM * 2 * np.pi / 60.0])   # start already spinning at idle
    T0 = np.array([T_AMB])                               # ...but stone cold
    gov0 = pid(kp=0.012, ki=0.05, kd=0.0005, dt=DT, output_min=0.0, output_max=1.0)

    # order = controls -> thermal -> mechanics, so each reads the freshest upstream signal
    return couple(
        [Subsystem("controls", gov0, controls_step),
         Subsystem("thermal", T0, thermal_step),
         Subsystem("mechanics", omega0, mechanics_step)],
        steps=STEPS, dt=DT,
        record=("rpm", "throttle", "temp"),
        initial_signals={"throttle": 0.1, "rpm": TARGET_RPM,
                         "omega": omega0[0], "temp": T_AMB},
    )


def sonify(rpm_hist, name):
    """RPM(t) -> engine hum (4-cyl 4-stroke: 2 firing events per revolution)."""
    t = np.arange(STEPS) * DT
    fire_hz = rpm_hist / 60.0 * 2.0
    audio_t = np.linspace(0, t[-1], int(SR * t[-1]))
    f = np.interp(audio_t, t, fire_hz)
    phase = np.cumsum(2 * np.pi * f / SR)
    wave = (np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)) * 0.3
    audio.save(AudioBuffer(wave, SR), str(OUT / name))


def plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t = res.times
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax1.axhline(TARGET_RPM, color="#888", ls="--", lw=1, label="target idle")
    ax1.axvspan(6.0, 8.5, color="#e8a0a0", alpha=0.25, label="load lurch")
    ax1.plot(t, res["rpm"], color="#2c7", lw=1.4)
    ax1.set(ylabel="engine speed\n(RPM)",
            title="Three-domain coupled engine warm-up  (thermal ↔ mechanics ↔ control)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.2)

    ax2.plot(t, res["throttle"] * 100, color="#37c", lw=1.4)
    ax2.set(ylabel="throttle\n(%)"); ax2.grid(alpha=0.2)
    ax2.annotate("governor eases off\nas the block warms", xy=(4, res["throttle"][2000] * 100),
                 xytext=(4.5, 60), fontsize=8, color="#37c",
                 arrowprops=dict(arrowstyle="->", color="#37c"))

    ax3.axhline(T_WARM, color="#888", ls=":", lw=1, label="fully warm")
    ax3.plot(t, res["temp"], color="#c63", lw=1.4)
    ax3.set(xlabel="time (s)", ylabel="block temp\n(°C)")
    ax3.legend(loc="lower right", fontsize=8); ax3.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(str(OUT / "coupled_engine.png"), dpi=110)
    return True


def main():
    print("=" * 70)
    print("THREE-DOMAIN COUPLED ENGINE — cold-start warm-up (thermal<->mech<->control)")
    print("=" * 70)
    res = run()
    rpm, throttle, temp = res["rpm"], res["throttle"], res["temp"]

    def at(sec):
        return int(sec / DT)

    rms = float(np.sqrt(np.mean((rpm - TARGET_RPM) ** 2)))
    print(f"\n  idle setpoint {TARGET_RPM:.0f} RPM held across a 12 s cold-start.\n")
    print(f"  {'time':>6}{'RPM':>10}{'throttle %':>13}{'block °C':>12}{'drag':>9}")
    for sec in (0.0, 1.0, 3.0, 6.5, 9.0, 11.9):
        i = min(at(sec), STEPS - 1)
        print(f"  {sec:>5.1f}s{rpm[i]:>10.1f}{throttle[i]*100:>12.1f}%"
              f"{temp[i]:>12.1f}{drag_coeff(temp[i]):>9.4f}")

    warm_throttle = throttle[at(0.2)]
    settled_throttle = throttle[at(5.5)]
    print(f"\n  RMS idle-speed error over the whole run: {rms:.1f} RPM")
    print(f"  cold throttle {warm_throttle*100:.1f}%  ->  warm throttle "
          f"{settled_throttle*100:.1f}%   (governor self-relaxes as drag falls)")

    sonify(rpm, "coupled_engine.wav")
    wrote = plot(res)
    print(f"\n  ✓ wrote coupled_engine.wav" + ("  / coupled_engine.png" if wrote else ""))
    print("  Three rigorous domains, one shared feedback loop, ~90 lines of glue.")
    print("  The substrate is morphogen.coupling.couple; the physics is Morphogen's.")


if __name__ == "__main__":
    main()
