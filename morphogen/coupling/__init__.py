"""Coupling substrate — co-simulate several rigorous domains with feedback.

The one coordinating layer Morphogen's domains were missing: a small driver that
co-advances rigorous domains (integrators, controls, thermo, circuit…) with
per-timestep feedback. This is what "model an engine — heat + mechanics + control
together" actually needs, and what no single-domain library provides.

Distinct from the retired cross-domain *composer* (one-shot A→B→C data plumbing):
this closes a *loop*, every timestep, across domains that each hold their own
state. See :mod:`morphogen.coupling.substrate`.

>>> from morphogen.coupling import Subsystem, couple
"""
from .substrate import CoupledResult, Subsystem, couple

__all__ = ["Subsystem", "couple", "CoupledResult"]
