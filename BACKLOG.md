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

### P0-1 — Make determinism real, or stop claiming it
"Deterministic" is in Morphogen's one-line identity, but `global_seed`/`--seed` is stored
(`runtime.py:250`) and **never applied** to any RNG (no `np.random.seed`/`default_rng`
consumes it; profiles unenforced).
- **Do:** thread a seeded `numpy.random.Generator` from `ExecutionContext.global_seed` through
  execution so stdlib RNG draws are reproducible; make `strict/repro/live` actually branch —
  OR remove the determinism claim from README/PITCH/identity until it's true.
- **Accept:** running the same `.morph`/API program twice with a fixed seed yields
  bit-identical output, proven by a new `tests/test_determinism.py`; OR the claim is gone.
- **Evidence:** `grep -rn global_seed morphogen/`; `03_aspirational_stack` finding #5.

### P0-2 — Retire the cross-domain composer  ·  DECIDED → option (b), by P1-1
`TransformPipeline.__call__` (`composer.py:215`) raises `ValueError` on every registered
built-in (it passes `source_data=` which every `__init__` rejects). The showcase
(`examples/cross_domain/01_transform_composition.py:164-180`) hides the guaranteed failure in
a `try/except` that prints "Pipeline creation note." No test executes the composer.
- **P1-1 settled the fix-or-retire question: RETIRE (option b).** The experiment
  ([composition-vs-glue](docs/reviews/2026-07-11-composition-vs-glue-experiment.md)) showed the
  composer runs 0/5 advertised pairs, is used by 1 of 93 examples, and — even if fixed — solves
  no real problem: hand-glue was equal-or-fewer LOC in every task. There is no case for fixing it.
- **Do:** delete `auto_compose` / `TransformPipeline` / registry-as-engine (or quarantine it,
  clearly labeled "experimental, non-functional"); keep the bridges usable by direct
  instantiation. Remove the failure-hiding `try/except` from the example.
- **Accept:** the composer is gone (or unmistakably quarantined) and no example pretends it
  works; a test asserts the chosen reality. No user loses anything (1/93 examples used it, and
  that one never ran).
- **Evidence:** runtime-confirmed `ValueError`; `01_cross_domain` finding #3; P1-1 experiment.

### P0-3 — Remove or relabel `qchem` (ships fabricated numbers)
`stdlib/qchem.py` — every function `warnings.warn("...stub")` and returns `-0.5 * n_electrons`
placeholder energies, inside a library advertised as "physically rigorous, production-quality."
- **Do:** delete qchem, or move it behind a clearly-labeled `experimental/`-style namespace
  that cannot be mistaken for a validated domain; drop it from the domain count.
- **Accept:** no code path returns fabricated physical quantities under a "domain" banner.
- **Evidence:** `reveal morphogen/stdlib/qchem.py`; `02_stdlib_depth` "the one that is fake."
- **Scope (re-verified chaotic-star-0711):** `qchem` is the *only* wholesale-fabricated
  domain (21 stub markers, every function returns placeholder numbers). Do **not** strip the
  peripheral, honestly-flagged stubs in real domains — `molecular.load_smiles`/`to_smiles`
  (RDKit-replaceable I/O, not the validated MM core) and `multiphase`'s ideal-K fallback
  branch are fine to keep; just don't count them as validated physics.

### P0-4 — Correct the domain-count / "rigorous" claims
"39 rigorous domains" (README/ROADMAP/STATUS) is inflated: ~11 rigorous + ~5 applied + ~24
utilities across 44 modules; flappy is a game.
- **Do:** state honest tiers — rigorous scientific / applied engineering / utilities — with
  the real counts. Keep the utilities (they're good); just stop calling them rigorous physics.
- **Accept:** every doc that gives a count gives the same, honest, tiered count.

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

### P1-2 — Reframe the front door around the real asset
- **Do:** rewrite README top + PITCH to lead with "a tested Python library of rigorous,
  interoperable scientific domains," not "universal deterministic category-theoretic platform."
  Move MLIR/symbolic/category-theory to a clearly-labeled "north-star, not built" section.
- **Accept:** a new reader in 60 seconds understands what actually works today.

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
- **Feedback co-advance driver (multiphysics/engine) — BOUNDED & PROVEN.** The smallest honest
  instance already exists: [`experiments/coupled-feedback/engine_governor.py`](experiments/coupled-feedback/)
  couples `integrators` ↔ `controls` with real per-timestep feedback in **~15 lines**; governor
  holds idle under a load lurch (RMS err 13.7) where a fairly-tuned open loop droops to ~540 RPM
  (136.3) — **~10× win** on honest disturbance-rejection. This is distinct from the (retired)
  one-shot composer.
  - **Do:** generalize the ~15-line loop into a small reusable co-simulation stepper (shared
    typed state/ports, per-step feedback, later multi-rate) over 2–3 rigorous domains; ship
    one engine-flavored coupled demo (e.g. combustion/thermo → pressure → mechanics → acoustics).
  - **Accept:** a documented `couple([...], steps)`-style driver + one coupled demo that a user
    could not trivially get from any single-domain library; a test guards it.
- **Transform planner (cost-model routing) — RESEARCH BET, do NOT gate the above on it.** e-graph /
  equality-saturation territory; even Poisson-spectral isn't shown. Park in P4.

---

## P2 — Code-health regressions (nobody's re-run health since March)

- **P2-1** — `cli.py:cmd_run` complexity regressed to **27** (March "fixed" it to 10).
  Re-extract; `reveal 'ast://morphogen?complexity>20'` should return 0. Also remove duplicate
  imports / unused `optimize_module` import in `cli.py`.
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
