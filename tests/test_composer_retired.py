"""Guards the retirement of the cross-domain auto-composition engine (BACKLOG P0-2).

The engine (TransformComposer/TransformPipeline/auto_compose) never executed — it
passed a source_data= kwarg every built-in transform rejected — so it was retired
2026-07-11. These tests assert the *chosen reality*: the retired names are gone
(with a guiding error), and the honest, working parts remain.
"""
import numpy as np
import pytest

import morphogen.cross_domain as xd


@pytest.mark.parametrize(
    "name", ["TransformComposer", "TransformPipeline", "BatchTransformComposer", "auto_compose"]
)
def test_retired_names_raise_guiding_error(name):
    """Accessing a retired symbol raises AttributeError pointing to the replacements."""
    with pytest.raises(AttributeError) as exc:
        getattr(xd, name)
    msg = str(exc.value)
    assert "retired" in msg
    assert "morphogen.coupling" in msg  # points users to the real coupling layer


def test_retired_names_not_importable():
    """A hard `from ... import auto_compose` also fails (no silent shim)."""
    with pytest.raises(ImportError):
        from morphogen.cross_domain import auto_compose  # noqa: F401


def test_find_transform_path_still_works():
    """The honest path-finder survives: it discovers routes, runs nothing."""
    # source == target is a trivial 0-hop path
    assert xd.find_transform_path("field", "field") == ["field"]
    # a nonexistent domain has no path
    assert xd.find_transform_path("field", "no_such_domain_xyz") is None


def test_compose_chains_explicit_transforms():
    """compose() chains explicitly-built interfaces (no auto-instantiation)."""
    from morphogen.cross_domain import TerrainToFieldInterface, FieldToAudioInterface

    terrain = np.random.rand(32, 32) * 50.0
    pipeline = xd.compose(
        TerrainToFieldInterface(terrain, normalize=True),
        FieldToAudioInterface(
            np.zeros((32, 32)),
            mapping={"mean": "frequency", "std": "amplitude"},
            sample_rate=44100, duration=1.0),
        validate=False,
    )
    out = pipeline(terrain)
    assert "frequency" in out and "amplitude" in out


def test_empty_compose_is_identity():
    ident = xd.compose()
    obj = object()
    assert ident(obj) is obj
