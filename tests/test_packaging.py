"""Packaging sanity: the project must build an installable wheel.

Guards the regression fixed when setup.py was collapsed into a pyproject-only
shim: setuptools automatic package discovery aborts on a repo with multiple
top-level packages (morphogen/ and tests/), so `pip install .` / wheel builds
fail unless discovery is explicitly scoped in pyproject.toml
([tool.setuptools.packages.find]).

This builds a real wheel (~2s) and asserts it ships the morphogen package and
excludes the test tree. Skipped if the `build` frontend isn't installed.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

build = pytest.importorskip("build", reason="`build` frontend not installed")


def test_wheel_builds_and_scopes_packages(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"wheel build failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )

    wheels = list(tmp_path.glob("morphogen-*.whl"))
    assert wheels, f"no wheel produced; dir has {list(tmp_path.iterdir())}"

    names = zipfile.ZipFile(wheels[0]).namelist()
    assert any(n.startswith("morphogen/") for n in names), (
        "wheel does not contain the morphogen package"
    )
    assert not any(n.startswith("tests/") for n in names), (
        "wheel unexpectedly bundles the tests/ tree"
    )
