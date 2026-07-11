# Morphogen — Current Status

**Version:** v0.12.0 toward v1.0  
**Updated:** 2026-04-17

---

## At a Glance

Morphogen is currently strongest as a **Python-first cross-domain computation library** with:

- a large multi-domain stdlib
- working cross-domain examples
- a broad automated test suite
- a partial DSL surface that now handles the documented hello-world path
- MLIR/compiler work that exists, but is not yet the primary user surface

## What Works Reliably Today

### Python library

The Python API is the most credible and best-supported interface today.

Examples that were re-run successfully during the April 17 review/repair passes:

- `examples/canonical/01_physics_to_audio.py`
- `examples/showcase/07_physical_instrument.py`
- `examples/showcase/08_digital_twin.py`
- `examples/cross_domain/fluid_acoustics_audio.py`

### `.morph` CLI basics

The documented hello-world path works again:

- `python -m morphogen.cli check examples/01_hello_heat.morph`
- `python -m morphogen.cli run examples/01_hello_heat.morph --steps 1`

The CLI also now reports `flow(...)` execution more accurately and applies `--steps` to flow blocks.

### Test suite

The suite is broad and useful, but some parts are environment-sensitive.

Verified recently:

- `pytest tests/test_cli.py tests/test_runtime.py tests/test_use_statement.py` → `74 passed`
- `pytest tests/test_visual3d.py` → `53 passed, 5 skipped`

The fragile `visual3d` screenshot/render tests are now opt-in via:

- `MORPHOGEN_RUN_VISUAL3D_RENDER_TESTS=1`

This keeps the default suite stable on machines where VTK off-screen rendering is unreliable.

## What Is Primary vs Secondary

### Primary surface

- Python API
- examples and tutorials
- cross-domain composition patterns
- deterministic NumPy-based runtime

### Secondary / in-progress surface

- `.morph` DSL expansion beyond the currently supported/documented path
- MLIR/native compilation as a general user-facing runtime
- fully portable headless `visual3d` screenshot rendering

## Known Limitations

- Packaging metadata is improved but still duplicated across `pyproject.toml` and `setup.py`
- Public docs still contain some stale counts and historical claims
- Real VTK screenshot rendering remains environment-sensitive
- Some advanced roadmap language still reflects aspiration more than current default usage
- PyPI release/distribution work is not yet complete

## Recommended Reading

- **Where the project is going:** [docs/STRATEGY.md](docs/STRATEGY.md)
- **What is being fixed right now:** [docs/PROGRESS_2026-04-17.md](docs/PROGRESS_2026-04-17.md)
- **Roadmap framing:** [docs/ROADMAP.md](docs/ROADMAP.md)
- **Project review:** [docs/PROJECT_REVIEW_2026-04-17.md](docs/PROJECT_REVIEW_2026-04-17.md)
