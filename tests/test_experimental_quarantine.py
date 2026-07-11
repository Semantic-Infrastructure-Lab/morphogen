"""Guards the qchem quarantine (BACKLOG P0-3).

qchem fabricates physical quantities (a flat placeholder energy per electron). It
must not be reachable as a validated domain: not auto-imported by
morphogen.stdlib, not in the domain registry, and it must announce itself as a
stub when imported explicitly.
"""
import warnings

import pytest


def test_qchem_not_in_stdlib_namespace():
    import morphogen.stdlib as stdlib
    assert not hasattr(stdlib, "qchem")
    assert "qchem" not in getattr(stdlib, "__all__", [])


def test_qchem_not_a_registered_domain():
    from morphogen.core import domain_registry as dr

    # whichever accessor exists, qchem must not appear as a domain name
    text = repr(getattr(dr, "DOMAIN_MODULES", "")) + repr(vars(dr))
    # be robust to the registry's exact shape: no domain entry maps to the qchem module
    assert "morphogen.stdlib.qchem" not in text


def test_experimental_qchem_warns_on_import():
    import importlib
    import sys

    sys.modules.pop("morphogen.stdlib.experimental.qchem", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("morphogen.stdlib.experimental.qchem")
    assert any("NOT implemented" in str(w.message) for w in caught)
