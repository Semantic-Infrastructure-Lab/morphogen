# Morphogen — Current Status

**Version:** v0.12.0 toward v1.0  
**Updated:** 2026-07-11

---

## At a Glance

Morphogen is currently strongest as a **Python-first library of rigorous, interoperable
scientific domains** with:

- ~11 analytically-validated physics domains (+ ~5 applied, ~24 utilities) in one tested namespace
- a **coupling substrate** (`morphogen.coupling.couple`) that co-advances domains with per-timestep feedback
- ~16 single-hop cross-domain bridges + hand-composed cross-domain examples
- deterministic execution (a fixed seed reproduces)
- a broad automated test suite
- a partial DSL surface that handles the documented hello-world path
- MLIR/compiler work that exists, but is not yet the primary user surface (north-star, not built)

## What Works Reliably Today

### Python library

The Python API is the most credible and best-supported interface today.

Examples that were re-run successfully during the April 17 review/repair passes:

- `examples/canonical/01_physics_to_audio.py`
- `examples/showcase/07_physical_instrument.py`
- `examples/showcase/08_digital_twin.py`
- `examples/cross_domain/fluid_acoustics_audio.py`

### Coupled simulation (`morphogen.coupling`)

`couple(subsystems, steps, dt)` co-advances several rigorous domains with per-timestep
feedback (sequential co-simulation, zero-order hold, multi-rate via `stride`, deterministic).
Runnable demos in `experiments/coupled-feedback/` — a 2-domain engine governor (~10× disturbance
rejection vs. open loop) and a 3-domain thermal ↔ mechanics ↔ control cold-start warm-up.
Guarded by `tests/test_coupling.py` (11 tests). See [docs/usage/coupled_simulation.md](docs/usage/coupled_simulation.md).

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
- the coupling substrate (`morphogen.coupling`) for feedback co-simulation across domains

> **Resolved (2026-07-11, meteoric-star):** the non-functional cross-domain composition *engine*
> has been retired (`compose()` + `find_transform_path()` are what remain; BACKLOG P0-2), and the
> runtime is now genuinely deterministic — `global_seed` is applied to the process RNGs on
> `ExecutionContext` construction, proven by `tests/test_determinism.py` (BACKLOG P0-1). One-shot
> cross-domain work = calling rigorous domains + a few lines of glue (or `compose()`); time-coupled
> work = `morphogen.coupling.couple()`.

### Secondary / in-progress surface

- `.morph` DSL expansion beyond the currently supported/documented path
- MLIR/native compilation as a general user-facing runtime
- fully portable headless `visual3d` screenshot rendering

## Known Limitations

- Packaging metadata now lives solely in `pyproject.toml`; `setup.py` is a thin compatibility shim
- `qchem` fabricates numbers — quarantined to `morphogen.stdlib.experimental.qchem`, not a domain (P0-3)
- Some public docs (e.g. `docs/DOMAINS.md`) still carry a flat domain catalog; reconciling to the
  honest tiered count is remaining P3 doc-debt
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
