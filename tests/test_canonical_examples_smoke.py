"""Smoke test for the canonical cross-domain Python examples.

The `examples/canonical/` set is Morphogen's designated public showcase — the
"4-6 excellent demos" that STRATEGY.md and the 2026-04-17 project review call the
project's distinctive v1.0 story (typed cross-domain composition: physics ->
audio, circuit -> audio, fluid -> sound, analysis -> instrument).

These ran only by hand before. This test executes each as a subprocess exactly
as a user would (`python examples/canonical/NN_*.py`) and asserts a clean exit,
so a regression in the flagship demos fails CI instead of being discovered later.

Outputs are .wav files under examples/canonical/output/, which is gitignored
(`*.wav`), so running here does not dirty the tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CANONICAL_DIR = Path(__file__).resolve().parent.parent / "examples" / "canonical"

CANONICAL_FILES = sorted(CANONICAL_DIR.glob("[0-9]*.py"))
CANONICAL_IDS = [p.name for p in CANONICAL_FILES]


def test_canonical_examples_exist():
    """Guard against the glob silently matching nothing."""
    assert CANONICAL_FILES, f"No canonical examples found under {CANONICAL_DIR}"


@pytest.mark.parametrize("example", CANONICAL_FILES, ids=CANONICAL_IDS)
def test_canonical_example_runs(example: Path):
    """Each canonical cross-domain demo runs end-to-end with a clean exit."""
    result = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{example.name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
