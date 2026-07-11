"""Task 2 — Damped oscillator sonification: simulate a spring, hear its ringdown.

The cross-domain operation is "physics simulation → audio." This is Morphogen's
WEAKEST case: Morphogen owns a rigorous integrator, but the physics→audio mapping
is NOT provided by any bridge — it is hand-written glue in BOTH versions (exactly
as the canonical physics_to_audio demo does). So this task isolates the question:
when the composition itself must be hand-rolled anyway, does using Morphogen's
domain primitive still pay?

  - morphogen_version(): morphogen.integrators.symplectic evolves the spring
  - glue_version():      hand-rolled velocity-Verlet (the whole method is ~4 lines)
"""
import numpy as np

SR = 44100
DUR = 1.0
K = (2 * np.pi * 220.0) ** 2   # stiffness → 220 Hz natural frequency
DAMP = 6.0                     # damping coefficient


def _sonify(displacement):
    """Shared hand-written physics→audio glue (identical in both versions)."""
    sig = displacement / (np.max(np.abs(displacement)) + 1e-9) * 0.8
    return sig


# ---- MORPHOGEN VERSION ------------------------------------------------------
def morphogen_version():
    from morphogen.stdlib.integrators import symplectic

    dt = 1.0 / SR
    n = int(SR * DUR)
    x = np.array([1.0]); v = np.array([0.0])
    disp = np.empty(n)
    grad = lambda pos: -(K * pos)              # -∇V for a spring (force per unit mass)
    for i in range(n):
        x, v = symplectic(i * dt, x, v, grad, dt, order=2)
        v *= np.exp(-DAMP * dt)                # damping (symplectic() is conservative-only)
        disp[i] = x[0]
    return _sonify(disp), "morphogen (integrators.symplectic + hand-rolled damping & sonify)"


# ---- RAW-GLUE VERSION -------------------------------------------------------
def glue_version():
    dt = 1.0 / SR
    n = int(SR * DUR)
    x = 1.0; v = 0.0
    disp = np.empty(n)
    for i in range(n):                          # velocity-Verlet, hand-rolled
        a = -K * x
        v = (v + a * dt) * np.exp(-DAMP * dt)
        x = x + v * dt
        disp[i] = x
    return _sonify(disp), "glue (hand-rolled velocity-Verlet)"


if __name__ == "__main__":
    for fn in (morphogen_version, glue_version):
        sig, label = fn()
        print(f"{label}: peak={np.max(np.abs(sig)):.3f} rms={np.sqrt(np.mean(sig**2)):.4f} n={len(sig)}")
