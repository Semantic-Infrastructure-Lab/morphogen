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

10 of the 24 committed `.morph` examples run cleanly today; the rest use
planned-but-unimplemented DSL syntax and are marked `xfail`. All 24 are now
exercised end-to-end through the CLI by `tests/test_morph_examples_smoke.py`,
so the class of bug that broke the hello-world path cannot silently return.
See [examples/README.md](examples/README.md) for the working-vs-planned list.

### Test suite

The suite is broad and useful, but some parts are environment-sensitive.

Verified recently:

- `pytest tests/test_cli.py tests/test_runtime.py tests/test_use_statement.py` → `74 passed`
- `pytest tests/test_visual3d.py` → `53 passed, 5 skipped`
- `pytest tests/test_morph_examples_smoke.py` → `21 passed, 28 xfailed` (CLI over every example)
- `pytest tests/test_canonical_examples_smoke.py` → `5 passed` (flagship cross-domain demos)

The fragile `visual3d` screenshot/render tests are now opt-in via:

- `MORPHOGEN_RUN_VISUAL3D_RENDER_TESTS=1`

This keeps the default suite stable on machines where VTK off-screen rendering is unreliable.

## What Is Primary vs Secondary

### Primary surface

- Python API
- examples and tutorials
- the rigorous domain library (~11 physics-validated domains) + hand-composed cross-domain demos

> **Correction (2026-07-11):** earlier drafts listed "cross-domain composition patterns" and a
> "deterministic NumPy-based runtime" as primary surfaces. Both are inaccurate: the composition
> *engine* does not execute and is being retired (see the composition-vs-glue experiment), and
> `global_seed` is stored but never applied to any RNG, so the runtime is not yet deterministic
> (BACKLOG P0-1). Cross-domain work today = calling rigorous domains and writing a few lines of
> glue, as the canonical demos do.

### Secondary / in-progress surface

- `.morph` DSL expansion beyond the currently supported/documented path
- MLIR/native compilation as a general user-facing runtime
- fully portable headless `visual3d` screenshot rendering

## Known Limitations

- Packaging metadata now lives solely in `pyproject.toml`; `setup.py` is a thin compatibility shim
- Public docs still contain some stale counts and historical claims
- Real VTK screenshot rendering remains environment-sensitive
- Some advanced roadmap language still reflects aspiration more than current default usage
- `pip install .` now succeeds in a clean virtualenv and installs a working
  `morphogen` CLI (guarded by `tests/test_packaging.py`); git-based install
  is the distribution story for v1.0 — PyPI publication is not a goal

## Recommended Reading

- **The honest, current picture (start here):** [docs/reviews/](docs/reviews/) — reality audit,
  the composition-vs-glue experiment, and the vision-vs-reality grounding, in reading order
- **The actionable work:** [BACKLOG.md](BACKLOG.md)
- **Where the project is going:** [docs/STRATEGY.md](docs/STRATEGY.md)
- **Roadmap framing (self-flagged stale — see BACKLOG):** [docs/ROADMAP.md](docs/ROADMAP.md)
