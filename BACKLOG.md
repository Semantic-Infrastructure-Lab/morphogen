---
title: "Morphogen Backlog"
type: reference
beth_topics:
  - morphogen
  - backlog
  - roadmap
  - code-quality
  - cross-domain
---

# Morphogen Backlog

**Created:** 2026-07-11
**Basis:** [docs/reviews/2026-07-11-code-reality-and-paths-forward.md](docs/reviews/2026-07-11-code-reality-and-paths-forward.md)
— a five-pass evidence audit of the actual code.

> **Why this file exists.** As of 2026-07-11 there was no live backlog: every concrete
> item in `docs/ROADMAP.md` was checked ✅, `internal/CODE_ISSUES.md` was 100% "RESOLVED,"
> GitHub issues = 0, tia tasks = 0. What remained was aspirational growth boilerplate. This
> is a *real* backlog: prioritized, evidence-linked, with acceptance criteria. ROADMAP.md
> stays as the direction/vision doc; **this file is the work.**

Priorities: **P0** = says-something-false-or-broken that ships today (credibility/correctness).
**P1** = the decisive experiment + the honest reframing. **P2** = code-health regressions.
**P3** = documentation debt. **P4** = genuine feature/research bets (only after P0–P1).

---

## P0 — Correctness & credibility (things currently false or broken)

### P0-1 — Make determinism real, or stop claiming it  ·  ✅ DONE 2026-07-11 (meteoric-star)
"Deterministic" is in Morphogen's one-line identity, but `global_seed`/`--seed` was stored
(`runtime.py:250`) and **never applied** to any RNG.
- **Done:** `ExecutionContext.__init__` now *applies* the seed — seeds the process-global
  RNGs the stdlib actually uses (`np.random`: 77 call sites; stdlib `random`) and exposes a
  dedicated seeded `context.rng` Generator; added `apply_seed(seed)` to reseed midstream.
  `tests/test_determinism.py` (6 tests) proves same-seed→identical output for np.random,
  stdlib random, and context.rng, that different seeds diverge, and that a real stdlib code
  path (`flappy.random_controller`) reproduces. (`strict/repro/live` profiles don't exist in
  the runtime — that part of the old claim was aspirational; not added.)

### P0-2 — Retire the cross-domain composer  ·  ✅ DONE 2026-07-11 (meteoric-star)
- **Done:** `composer.py` gutted to keep only what always worked — `find_transform_path()`
  (registry BFS, discovers routes, runs nothing) and `compose()` (chain explicitly-built
  interfaces). The broken engine (`TransformComposer`/`TransformPipeline`/`BatchTransformComposer`/
  `auto_compose`) removed; `cross_domain.__init__` raises a guiding `AttributeError` on the
  retired names pointing to `compose()`/`find_transform_path()`/`morphogen.coupling`. Example 01
  lost its failure-hiding `try/except` + dead composer-stats section; Part 4 now shows real
  `compose()` chaining. `tests/test_composer_retired.py` (8 tests) asserts the chosen reality.

### P0-3 — Remove or relabel `qchem` (ships fabricated numbers)  ·  ✅ DONE 2026-07-11 (meteoric-star)
- **Done:** `morphogen/stdlib/qchem.py` → `morphogen/stdlib/experimental/qchem.py`, behind a
  loud module docstring + import-time warning ("NOT implemented, not a domain"). Removed from
  `morphogen.stdlib` auto-import and `__all__`, and from the core domain registry (dropped from
  the domain count). Still importable explicitly (`morphogen.stdlib.experimental.qchem`) for the
  interface sketch, but never as a validated domain. `tests/test_experimental_quarantine.py`
  (3 tests) guards it. Peripheral honestly-flagged stubs in real domains left intact as noted.

### P0-4 — Correct the domain-count / "rigorous" claims  ·  ◐ PARTIAL 2026-07-11 (meteoric-star)
"39 rigorous domains" is inflated: ~11 rigorous + ~5 applied + ~24 utilities.
- **Done:** README + PITCH now state the honest tiered count (~11 rigorous + ~5 applied + ~24
  utilities). STATUS/ROADMAP no longer carry an inflated count.
- **Remaining (P3-3 territory):** `docs/DOMAINS.md` still lists a flat catalog — reconcile it
  to the same tiered language when doing the P3 doc pass.

---

## P1 — The decisive experiment + honest reframing

### P1-1 — Head-to-head: does composition beat glue?  ·  ✅ DONE 2026-07-11
**Result: composition thesis FALSE; composer retired.** See
[docs/reviews/2026-07-11-composition-vs-glue-experiment.md](docs/reviews/2026-07-11-composition-vs-glue-experiment.md)
and runnable code in [`experiments/composition-vs-glue/`](experiments/composition-vs-glue/).
Two tasks implemented twice + an engine probe:
- **LOC:** glue was equal-or-fewer lines in *both* tasks (Task 1: glue 7 vs morphogen 12;
  Task 2: glue 10 vs morphogen 11). Composition does not save code.
- **Engine:** `auto_compose` runs 0/5 advertised pairs; used by 1/93 examples.
- **Real value is a *library* property, not composition:** Morphogen wins only where a
  primitive is genuinely hard (Task 1's real diode clipper vs a `tanh` approximation with 2×
  lower spectral centroid; ~1,112 LOC of MNA/Newton-Raphson you'd otherwise rewrite).
- **Decides P0-2 → retire (b); confirms Path A is the whole product; elevates P0-1/Path C.**

### P1-2 — Reframe the front door around the real asset  ·  ✅ DONE 2026-07-11 (meteoric-star)
- **Done:** README top + PITCH rewritten to lead with "a tested Python library of rigorous,
  interoperable scientific domains" + the coupling substrate, with a real Python code sample
  (not fictional `.morph`). MLIR/symbolic/category-theory/`.morph` moved to a clearly-labeled
  "north star, not built" section in both. A new reader now sees today's reality first.

### P1-3 — Make one demo undeniable
Depth on one beats breadth on four. Circuit→audio ("design a guitar pedal, hear it") is the
most visceral and hardest to reproduce elsewhere.
- **Do:** polish one flagship end-to-end demo that uses the *real* rigorous domains, produces
  a compelling artifact (WAV/plot), and is CI-guarded.
- **Accept:** `python examples/canonical/<demo>.py` runs clean, output is impressive, test guards it.

### P1-4 — The coupling substrate: build the layer the vision needs *(new North Star candidate)*
Vision-vs-reality audit ([docs/reviews/2026-07-11-vision-vs-reality.md](docs/reviews/2026-07-11-vision-vs-reality.md))
found that all three founding visions (transform algebra, cross-domain tool reuse, coupled
multiphysics/"model an engine") reduce to **one missing layer**: an intelligent driver that
co-advances/routes across the rigorous domains. It was designed across ~10 spec/ADR docs and
never built. Two faces, very different cost:
- **Feedback co-advance driver (multiphysics/engine) — ✅ DONE 2026-07-11 (meteoric-star).**
  The ~15-line proof (`engine_governor.py`) is now a real library module:
  **`morphogen/coupling/`** — `Subsystem` + `couple(subsystems, steps, dt, ...)`, a sequential
  (Gauss–Seidel) co-simulation driver with zero-order hold, multi-rate via `stride`, and
  deterministic output. Guarded by `tests/test_coupling.py` (11 tests: ordering, multi-rate,
  determinism, validation + the value claim — coupled governor beats open loop >5×, 3-domain
  warm-up couples). Flagship demo `experiments/coupled-feedback/coupled_engine.py` scales it to
  **three** mutually-coupled rigorous domains (thermal ↔ mechanics ↔ control — a cold-start
  warm-up the governor self-relaxes through). Documented in `docs/usage/coupled_simulation.md`.
  - **Next (optional deepening):** a genuinely different flagship (combustion/thermo → pressure
    → mechanics → acoustics) and multi-rate demo; the substrate already supports both.
- **Transform planner (cost-model routing) — RESEARCH BET, do NOT gate the above on it.** e-graph /
  equality-saturation territory; even Poisson-spectral isn't shown. Park in P4.

---

## P2 — Code-health regressions (nobody's re-run health since March)

- **P2-1** — ✅ DONE 2026-07-11 (meteoric-star). `cmd_run` decomposed 27 → **4** (extracted
  `_report_type_errors` / `_execute_program` / `_execute_flow_blocks` / `_execute_step_blocks`);
  nothing in `cli.py` now exceeds 20. Unused `optimize_module` import removed. 39 cli tests green.
- **P2-2** — New circular-import cycle in `mlir/dialects/__init__.py` (5 files). Break it, and
  add a lint rule against the `__init__`-level re-import pattern so it stops recurring.
- **P2-3** — Re-baseline health and wire it into CI: `reveal overview morphogen/` (today 84.7/100,
  10 critical files) as a tracked gate so drift is caught, not discovered months later.
- **P2-4** — God-class decomposition (internal grouping, keep public API): `geometry.py` (2,411),
  `Visual3DOperations` (1,905), `VisualOperations` (1,464), `MLIRCompiler` (1,326). Low urgency;
  do opportunistically. Note `geometry.py` was never in the original audit baseline.
- **P2-5** — Fix the 8 exact-duplicate functions (D001) and 3 bare `except:` clauses (B001).
- **P2-6** — Document the `@operator` dynamic-dispatch false-positive so future dead-code audits
  don't re-litigate the ~284 "uncalled" functions (~15.5%, mostly not dead).

## P3 — Documentation debt (76k lines > 57k code lines)

- **P3-1** — Fix `DOCUMENTATION_INDEX.md` self-contradictions (STATUS.md "1,096 lines" vs actual
  ~94; domain-architecture.md listed live *and* archived; disagreeing counts). Or regenerate it.
- **P3-2** — Collapse the doc set toward the honest core. Aggressively archive the 26
  specification docs describing unbuilt DSL surface and the grandiose philosophy/vision docs
  (or clearly stamp them "aspirational, not implemented").
- **P3-3** — Reconcile README/ROADMAP/STATUS/STRATEGY to one voice (the honest one). No doc
  should assert MLIR/symbolic/category/determinism as present-tense fact until it is.

## P4 — Genuine bets (only after P0–P1 land)

- **P4-1** — Symbolic layer (sympy): the one "aspirational" claim that could become a *real*
  differentiator rather than table stakes. Spike "try symbolic, fall back to numeric" on one
  domain (e.g. field PDE or circuit) and measure whether it earns its keep.
- **P4-2** — Reproducible-science positioning (Path C): once P0-1 is real, add cross-platform
  repro tests and target the computational-methods audience.
- **P4-3** — Revisit MLIR only if a concrete performance need appears with a real user. Until
  then it's inert weight; don't invest.

---

## Explicitly NOT doing (per STRATEGY.md + this audit)

- PyPI publication — git install is the v1.0 story.
- Expanding the `.morph` DSL surface / chasing the 14 xfail examples.
- More domains — depth on the rigorous ~11 beats breadth.
- "50+ examples / community channels / marketing" growth boilerplate — premature; the product
  isn't honestly described yet.
