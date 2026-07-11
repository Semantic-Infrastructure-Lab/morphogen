"""The coupling substrate — a small sequential co-simulation driver.

This is the layer Morphogen's vision always described and the code never had: an
intelligent driver that **co-advances several rigorous domains with per-timestep
feedback**. It is deliberately tiny — the value is not in the driver's cleverness
but in the rigorous domains it couples (integrators, controls, thermo, circuit…),
each already validated against analytic ground truth.

It is the opposite of the (retired) one-shot ``TransformComposer``: that plumbed
data through a domain graph *once*, A→B→C. This closes a *loop* — B reads A's
current output, A then advances using B's response, every timestep — which is what
"model an engine: heat + mechanics + control together" actually requires and what
no single-domain library gives you.

Scheme
------
Sequential (Gauss–Seidel) co-simulation with a zero-order hold on the coupling
signals. Each timestep, subsystems step **in list order**; each one reads the
shared signal bus (holding last step's signals plus any already refreshed by
earlier subsystems this step), advances its own state by ``dt``, and publishes
signals that merge back into the bus. List order is therefore meaningful and lets
you express a sensor → controller → plant chain directly::

    couple([plant, controller], ...)   # controller reads the plant's fresh output

Multi-rate is supported per subsystem via ``stride`` (advance every N ticks),
so a fast plant and a slow controller can share one run.

Example
-------
>>> from morphogen.coupling import Subsystem, couple
>>> import numpy as np
>>> def plant_step(x, sig, t, dt):
...     u = sig.get("u", 0.0)                 # read controller output (ZOH)
...     x = x + dt * (-x + u)                 # trivial 1st-order plant
...     return x, {"y": x}                    # publish measurement
>>> def ctrl_step(_, sig, t, dt):
...     err = 1.0 - sig.get("y", 0.0)         # hold y at 1.0
...     return None, {"u": 2.0 * err}         # publish actuation
>>> res = couple(
...     [Subsystem("plant", 0.0, plant_step),
...      Subsystem("ctrl", None, ctrl_step)],
...     steps=500, dt=0.01, record=("y",))
>>> round(res.history["y"][-1], 2)            # converged near setpoint
0.67
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

# A subsystem's step: (state, signals, t, dt) -> (new_state, published_signals)
StepFunc = Callable[[Any, Mapping[str, float], float, float],
                    Tuple[Any, Mapping[str, float]]]


@dataclass
class Subsystem:
    """One domain participating in a coupled co-simulation.

    A subsystem owns its own state and knows how to advance it by a single
    timestep, reading coupling signals published by the others and publishing its
    own. The state is opaque to the driver — it can be a float, a numpy array, a
    dataclass, whatever the wrapped domain uses.

    Args:
        name: Unique identifier; also namespaces nothing — publish flat signal
            names and wire subsystems by agreeing on those names.
        state: Initial state. Whatever ``step`` accepts and returns.
        step: ``step(state, signals, t, dt) -> (new_state, published)``. Must be
            side-effect free with respect to ``signals`` (treat it as read-only)
            and return the signals it wants on the bus as a plain dict.
        stride: Advance only every ``stride`` ticks (multi-rate co-simulation).
            Between advances the subsystem's last-published signals are held
            (zero-order hold). Default 1 (every tick).
    """

    name: str
    state: Any
    step: StepFunc
    stride: int = 1

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"{self.name}: stride must be >= 1, got {self.stride}")


@dataclass
class CoupledResult:
    """Outcome of a :func:`couple` run.

    Attributes:
        signals: The final signal bus (last timestep's published values).
        history: ``{signal_name: np.ndarray[steps]}`` for each recorded signal.
        states: ``{subsystem_name: final_state}``.
        times: ``np.ndarray[steps]`` of sample times.
        steps: Number of timesteps advanced.
        dt: The timestep used.
    """

    signals: Dict[str, float]
    history: Dict[str, np.ndarray]
    states: Dict[str, Any]
    times: np.ndarray
    steps: int
    dt: float

    def __getitem__(self, key: str) -> np.ndarray:
        """``result["rpm"]`` -> the recorded history for signal ``rpm``."""
        return self.history[key]


def couple(
    subsystems: Sequence[Subsystem],
    *,
    steps: int,
    dt: float,
    t0: float = 0.0,
    record: Iterable[str] = (),
    initial_signals: Mapping[str, float] | None = None,
) -> CoupledResult:
    """Co-advance ``subsystems`` for ``steps`` timesteps with per-step feedback.

    See the module docstring for the scheme (sequential / Gauss–Seidel, ZOH on
    coupling signals, list order = producer→consumer order).

    Args:
        subsystems: The domains to couple, in step order. Names must be unique.
        steps: Number of timesteps to advance.
        dt: Timestep size (seconds).
        t0: Start time.
        record: Signal names to capture per-step into ``result.history``.
        initial_signals: Values to seed the bus with before the first step, so a
            consumer that runs before its producer on step 0 sees a defined input.

    Returns:
        A :class:`CoupledResult` with the final bus, recorded histories, and
        final per-subsystem states.

    Raises:
        ValueError: on duplicate subsystem names, or ``steps``/``dt`` <= 0.
        KeyError: if a recorded signal is never published (reported once, with
            the set of signals that *were* seen, so wiring typos are obvious).
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    names = [s.name for s in subsystems]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"subsystem names must be unique; duplicates: {dupes}")

    record = tuple(record)
    history: Dict[str, np.ndarray] = {k: np.empty(steps) for k in record}
    signals: Dict[str, float] = dict(initial_signals or {})

    for i in range(steps):
        t = t0 + i * dt
        for sub in subsystems:
            if i % sub.stride != 0:
                continue  # multi-rate: hold this subsystem's last outputs
            sub.state, published = sub.step(sub.state, signals, t, dt)
            if published:
                signals.update(published)
        for k in record:
            if k not in signals:
                raise KeyError(
                    f"recorded signal {k!r} was never published by step {i}; "
                    f"signals on the bus: {sorted(signals)}"
                )
            history[k][i] = signals[k]

    times = t0 + np.arange(steps) * dt
    states = {s.name: s.state for s in subsystems}
    return CoupledResult(
        signals=signals, history=history, states=states,
        times=times, steps=steps, dt=dt,
    )
