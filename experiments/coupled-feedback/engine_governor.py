"""Coupled-feedback demo — an engine idle-speed governor.

This is the smallest honest proof of the ONE thing Morphogen's docs describe
everywhere and the code never had: a **per-timestep feedback driver that
co-advances two rigorous domains**. Everything else (the domains) already exists
and is validated; the *only* new code here is the ~15-line coupling loop in
`run()` — the missing "coupling substrate."

Two domains, genuinely coupled each step:
  - MECHANICS  : a rotating crank  I·dω/dt = kτ·throttle − load − b·ω,
                 advanced by `morphogen.stdlib.integrators.integrate` (RK4, rigorous).
  - CONTROLS   : a PID governor `morphogen.stdlib.controls.pid_step` (rigorous),
                 reading measured RPM and setting throttle to hold a target RPM.

The loop closes mechanics → controls → mechanics every timestep (zero-order hold
on throttle — a correct sampled-data controller). A time-varying load disturbance
gives the governor something real to reject.

We run it twice — governor ENGAGED (coupled) vs. FIXED throttle (open loop) — and
show the coupling is what holds the engine at idle. Then we sonify RPM(t) to a WAV
so you can *hear* the governor fighting the load: the engine vision in miniature.
"""
from pathlib import Path
import numpy as np

from morphogen.stdlib.integrators import integrate          # MECHANICS domain
from morphogen.stdlib.controls import pid, pid_step          # CONTROLS domain
from morphogen.stdlib.audio import AudioOperations as audio, AudioBuffer  # AUDIO domain

OUT = Path(__file__).parent
SR = 22050

# --- plant (engine crank) parameters ---
INERTIA = 0.05          # kg·m²  crank + flywheel
DRAG = 0.02             # N·m·s  viscous drag coefficient
K_THROTTLE = 3.0        # N·m per unit throttle
TARGET_RPM = 800.0      # idle setpoint
DT = 0.002              # 500 Hz control/sim rate
STEPS = 3000            # 6 seconds


def _load_torque(t):
    """A load disturbance: baseline + a lurch at t=2s (e.g. AC compressor kicks in)."""
    base = 0.15
    lurch = 1.2 if 2.0 <= t < 3.5 else 0.0
    return base + lurch + 0.1 * np.sin(2 * np.pi * 1.5 * t)


def _omega_to_rpm(omega):
    return omega * 60.0 / (2 * np.pi)


# ---- THE COUPLING SUBSTRATE (the only new code) ----------------------------
def run(coupled: bool):
    omega_target = TARGET_RPM * 2 * np.pi / 60.0
    omega = np.array([omega_target])                    # start at idle
    gov = pid(kp=0.010, ki=0.040, kd=0.0004, dt=DT, output_min=0.0, output_max=1.0)
    # FAIR open-loop baseline: the exact steady-state throttle that holds idle with NO
    # disturbance (balances baseline load + drag at target speed). It only fails when the
    # unexpected load lurch arrives — which is precisely what a governor exists to reject.
    fixed_throttle = (0.15 + DRAG * omega_target) / K_THROTTLE
    rpm_hist = np.empty(STEPS)

    for i in range(STEPS):
        t = i * DT
        rpm = _omega_to_rpm(omega[0])                   # MECHANICS -> measurement
        if coupled:
            throttle, gov = pid_step(gov, TARGET_RPM, rpm)   # CONTROLS reads plant
        else:
            throttle = fixed_throttle                        # open loop: no feedback
        load = _load_torque(t)
        deriv = lambda tt, y: np.array([                 # MECHANICS derivative (ZOH throttle)
            (K_THROTTLE * throttle - load - DRAG * y[0]) / INERTIA])
        omega = integrate(deriv, omega, t, DT, method="rk4")  # CONTROLS -> plant, co-advance
        rpm_hist[i] = _omega_to_rpm(omega[0])
    return rpm_hist


def sonify(rpm_hist, name):
    """Map RPM(t) -> engine hum: firing frequency tracks RPM (4-cyl 4-stroke = 2 fires/rev)."""
    t = np.arange(STEPS) * DT
    fire_hz = rpm_hist / 60.0 * 2.0                     # instantaneous fundamental
    audio_t = np.linspace(0, t[-1], int(SR * t[-1]))
    f = np.interp(audio_t, t, fire_hz)
    phase = np.cumsum(2 * np.pi * f / SR)
    wave = (np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)) * 0.3
    audio.save(AudioBuffer(wave, SR), str(OUT / name))


def plot(coupled, open_loop):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t = np.arange(STEPS) * DT
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(TARGET_RPM, color="#888", ls="--", lw=1, label="target idle (800 RPM)")
    ax.axvspan(2.0, 3.5, color="#e8a0a0", alpha=0.25, label="load lurch")
    ax.plot(t, coupled, color="#2c7", lw=1.6, label="governor engaged (coupled)")
    ax.plot(t, open_loop, color="#c33", lw=1.6, label="fixed throttle (open loop)")
    ax.set(xlabel="time (s)", ylabel="engine speed (RPM)",
           title="Coupled feedback rejects the load lurch; open loop droops  (controls ↔ mechanics)")
    ax.legend(loc="center right", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(str(OUT / "engine_governor.png"), dpi=110)
    return True


def main():
    print("=" * 66)
    print("COUPLED-FEEDBACK DEMO — engine idle governor (controls <-> mechanics)")
    print("=" * 66)
    coupled = run(coupled=True)
    open_loop = run(coupled=False)

    def rms_err(h):
        return float(np.sqrt(np.mean((h - TARGET_RPM) ** 2)))
    def max_dev(h):
        return float(np.max(np.abs(h - TARGET_RPM)))

    print(f"\n  target idle: {TARGET_RPM:.0f} RPM, load lurch 2.0-3.5s\n")
    print(f"  {'':16}{'RMS error':>12}{'max deviation':>16}{'final RPM':>12}")
    print(f"  {'GOVERNOR (coupled)':16}{rms_err(coupled):>10.1f}  {max_dev(coupled):>14.1f}  {coupled[-1]:>10.1f}")
    print(f"  {'FIXED (open loop)':16}{rms_err(open_loop):>10.1f}  {max_dev(open_loop):>14.1f}  {open_loop[-1]:>10.1f}")

    sonify(coupled, "engine_governed.wav")
    sonify(open_loop, "engine_openloop.wav")
    wrote_plot = plot(coupled, open_loop)
    print(f"\n  ✓ wrote engine_governed.wav / engine_openloop.wav"
          + ("  / engine_governor.png" if wrote_plot else ""))

    ratio = rms_err(open_loop) / max(rms_err(coupled), 1e-6)
    print(f"\n  Coupling verdict: feedback cuts idle-speed error {ratio:.1f}x.")
    print("  The domains are Morphogen's; the ~15-line co-advance loop is the substrate.")


if __name__ == "__main__":
    main()
