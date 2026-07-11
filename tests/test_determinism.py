"""Determinism guard (BACKLOG P0-1).

"Deterministic" is in Morphogen's one-line identity, but `global_seed` used to be
*stored and never applied* — no RNG ever consumed it. These tests assert the fix:
constructing an ExecutionContext with a fixed seed actually seeds the process RNGs
the stdlib domains draw from, so two runs with the same seed reproduce, and two
runs with different seeds diverge.
"""
import random

import numpy as np

from morphogen.runtime.runtime import ExecutionContext


def test_context_seeds_global_numpy_rng():
    """Same seed -> identical np.random.* draws (the 77 stdlib call sites)."""
    ExecutionContext(global_seed=1234)
    a = np.random.rand(50)
    ExecutionContext(global_seed=1234)
    b = np.random.rand(50)
    assert np.array_equal(a, b)


def test_context_seeds_stdlib_random():
    """Same seed -> identical stdlib random draws."""
    ExecutionContext(global_seed=7)
    a = [random.random() for _ in range(20)]
    ExecutionContext(global_seed=7)
    b = [random.random() for _ in range(20)]
    assert a == b


def test_context_rng_is_seeded_generator():
    """The dedicated context.rng Generator is reproducible across contexts."""
    a = ExecutionContext(global_seed=99).rng.standard_normal(30)
    b = ExecutionContext(global_seed=99).rng.standard_normal(30)
    assert np.array_equal(a, b)


def test_different_seeds_diverge():
    """Different seeds must not accidentally coincide."""
    ExecutionContext(global_seed=1)
    a = np.random.rand(50)
    ExecutionContext(global_seed=2)
    b = np.random.rand(50)
    assert not np.array_equal(a, b)


def test_apply_seed_reseeds_midstream():
    """apply_seed() lets a caller reset determinism and updates global_seed."""
    ctx = ExecutionContext(global_seed=5)
    first = np.random.rand(10)
    np.random.rand(10)               # advance the stream
    ctx.apply_seed()                 # reset to global_seed
    assert np.array_equal(np.random.rand(10), first)

    ctx.apply_seed(42)               # reseed to a new value
    assert ctx.global_seed == 42


def test_real_domain_reproducible_under_seed():
    """A real stdlib code path that draws from global np.random reproduces.

    ``flappy.random_controller`` samples ``np.random.rand(...)`` with no internal
    seed — exactly the pattern used across the stdlib. Under a fixed context seed
    it now reproduces; before the fix it never would have.
    """
    from types import SimpleNamespace

    from morphogen.stdlib.flappy import random_controller

    state = SimpleNamespace(n_birds=64)  # random_controller only reads n_birds
    ExecutionContext(global_seed=2024)
    a = random_controller(state, flap_prob=0.1)
    ExecutionContext(global_seed=2024)
    b = random_controller(state, flap_prob=0.1)
    assert np.array_equal(a, b)
