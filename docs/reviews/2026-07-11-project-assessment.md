# Morphogen — Project Assessment & Findings

**Date:** 2026-07-11
**Scope:** Fresh-clone install, smoke-run of the CLI and examples, full local test run, and a
"what's interesting / what's next" review.
**Environment:** Linux, Python 3.x, NumPy 2.4.6, package installed via `pip install -e .`.
**Branch reviewed:** `claude/project-overview-next-steps-tsdc5z` (at the time, even with `main`).

> This is a point-in-time engineering assessment, not a roadmap change. It records reproducible
> findings so they can be triaged. Nothing in the product runtime was modified to produce it.

> **Resolution status (updated 2026-07-11):** All three findings are now closed.
> **#1** (CLI TypeChecker crash) — fixed; `morphogen run`/`check` work on every runnable
> example, guarded by `tests/test_morph_examples_smoke.py` and `tests/test_cli.py`.
> **#2** (no end-to-end CLI test) — closed by that smoke suite over all committed `.morph` files.
> **#3** (undeclared deps / clean out-of-box run) — `pillow` and `scipy` are now runtime
> `dependencies`; `pytest-benchmark` is in the `dev` extra; `tests/test_geometry_benchmarks.py`
> now `importorskip`s the plugin so a bare `pip install -e .` run skips benchmarks cleanly
> instead of erroring; the 10 stray root-level scripts were relocated/removed in a prior session.

---

## TL;DR

- Morphogen is a real, actively developed, ~59k-line cross-domain computation DSL + runtime. The
  core library is healthy: **1,611 tests pass locally** (213 skipped).
- **However, the flagship user command is broken.** `morphogen run <any real .morph program>`
  crashes before execution with `AttributeError: 'TypeChecker' object has no attribute 'visit_use'`.
  The same crash affects `morphogen check`. This includes the README's and examples' documented
  "START HERE" command.
- Root cause is small and contained: the base AST visitor is missing visitor methods for 11 node
  types that every real program uses. The **runtime itself is fine** — bypassing the type-checker,
  programs execute correctly.
- The reason a headline-breaking bug survived a large green test suite: **no test exercises the
  `morphogen run` CLI path over the committed examples.** The tests drive the Python runtime API
  directly.

---

## 1. What Morphogen is (context)

A deterministic DSL and NumPy-backed runtime whose thesis is **cross-domain composition**: audio
synthesis, physics, circuits, chemistry, and geometry in one type system, one scheduler, one
language. Programs use `use <domains>`, `@state` variables, and `flow(dt, steps) { … }`
time-evolution blocks. There is an MLIR compilation path (6 custom dialects) as a long-term backend.

- **Size:** ~58,900 lines of Python under `morphogen/`.
- **Breadth:** 39 domains, ~606 operators (per STATUS.md/ROADMAP.md).
- **Examples:** 19 `.morph` files + many Python demos under `examples/`.

### What is most interesting

1. **The cross-domain thesis is genuinely novel** — typed connections between domains (e.g.
   fluid → acoustics → audio) with dimensional units (`[K]`, `[m/s]`, `[N]`) and multi-rate
   scheduling. This is the project's real differentiator.
2. **Recent engineering is rigorous.** The changelog records *physics-level* fixes, not cosmetic
   ones — e.g. a molecular-dynamics fs→ps unit bug that produced ~2.3×10²⁶ K temperatures,
   Maxwell-Boltzmann NVT velocity initialization, replacing a diverging CSTR Picard solver with
   `scipy.optimize.root`, a Newton-Raphson nonlinear diode circuit solver. Tests assert invariants
   (e.g. `T60 × decay = 6.91`), not just "does not crash."
3. **Determinism as a first-class guarantee** (explicit RNG seeding; `strict`/`repro`/`live`
   profiles) is a meaningful differentiator for reproducible science and audio.

---

## 2. Finding #1 (Critical) — `morphogen run` / `morphogen check` crash on every real program

### Symptom

```
$ morphogen run examples/01_hello_heat.morph      # README + examples "START HERE"
...
AttributeError: 'TypeChecker' object has no attribute 'visit_use'

$ morphogen check examples/01_hello_heat.morph
Type-checking examples/01_hello_heat.morph...
Error: 'TypeChecker' object has no attribute 'visit_use'
```

A program without `use` fails later on the next unhandled node instead — e.g.
`examples/v0_3_1_recursive_factorial.morph` crashes with
`AttributeError: 'TypeChecker' object has no attribute 'visit_function'`.

### Root cause

`morphogen/cli.py` runs the type-checker unconditionally before execution:

- `cmd_run` — `morphogen/cli.py:146-147`: `checker = TypeChecker(); checker.visit(program)`
- `cmd_check` — `morphogen/cli.py:204-205`: same pattern

`TypeChecker` (`morphogen/ast/visitors.py:96`) subclasses `ASTVisitor`
(`morphogen/ast/visitors.py:7`). Dispatch works via `node.accept(visitor)` calling
`visitor.visit_<node>(...)`. But the base `ASTVisitor` **only defines 15 of the 26 visitor methods
the AST nodes call**, so the first unhandled node type raises `AttributeError`, which the broad
`except Exception` in `cmd_run` reports as a "Runtime error" and exits non-zero.

**Missing visitor methods** (node types whose `accept()` in `morphogen/ast/nodes.py` calls a method
that neither `ASTVisitor` nor `TypeChecker` defines):

```
visit_block          visit_flow           visit_function
visit_if_else        visit_lambda         visit_link
visit_output         visit_return         visit_struct
visit_struct_literal visit_use
```

Every non-trivial `.morph` program uses at least one of these (`use`, `flow`, `fn`, `output`,
structs, `if/else`, lambdas), so the crash is effectively universal.

### Proof the runtime is otherwise fine

Bypassing the type-checker and executing directly through the runtime succeeds:

```python
from morphogen.parser.parser import parse
from morphogen.runtime.runtime import Runtime, ExecutionContext
from morphogen.ast.nodes import Step
prog = parse(open('examples/v0_3_1_recursive_factorial.morph').read())
rt = Runtime(ExecutionContext(global_seed=42))
for s in prog.statements:
    if not isinstance(s, Step):
        rt.execute_statement(s)
# -> "INTERPRETER RAN OK (bypassing TypeChecker)"
```

So the defect is isolated to the visitor-dispatch layer used by the CLI's type-check pass, **not**
the interpreter.

### Suggested fix (low risk, ~30 min)

Give `ASTVisitor` safe default implementations for the 11 missing node types (recurse into child
nodes, return `None`) so that `TypeChecker` degrades gracefully instead of raising. A
`generic_visit`-style fallback (as in Python's own `ast.NodeVisitor`) would also prevent this whole
class of bug for future node types. This unblocks `morphogen run` and `morphogen check` for every
example.

---

## 3. Finding #2 (High) — No end-to-end CLI test lets Finding #1 hide

The test suite (1,611 passing) drives the Python runtime API directly and **never invokes the
`morphogen` CLI over the committed `examples/*.morph` files**. That is precisely why a crash in the
user-facing entry point coexists with a green suite.

**Recommendation:** add a smoke test that shells `morphogen run --steps 1 <file>` (or an in-process
equivalent) over each committed `.morph` example and asserts a clean exit. This would have caught
Finding #1 and guards against regressions in the parser → type-check → runtime pipeline as a whole.

---

## 4. Finding #3 (Medium) — Undeclared test/dev dependencies; suite not clean out-of-the-box

On a fresh clone with only the declared install, the suite does not run cleanly until extra packages
are added by hand:

- **`Pillow`** — not installed by default; several visual tests and 3 files that execute image code
  at import time (`tests/test_diagnosis.py`, `tests/test_portfolio_examples.py`,
  `tests/test_proper_patterns.py`) error at collection without it.
- **`pytest-benchmark`** — the `benchmark` fixture is undeclared; all 20 errors in the final run are
  `tests/test_geometry_benchmarks.py` failing at setup because the fixture is missing (these are
  **not** product bugs).
- **`scipy`** — required at runtime by recent molecular/kinetics/circuit fixes; should be an
  explicit dependency, not incidental.

**Recommendation:** declare these in `pyproject.toml` (runtime vs. `[project.optional-dependencies]`
dev/test extras), so `pip install -e .[test]` yields a clean run.

### Related housekeeping

- **10 stray test/validate scripts live in the repo root** (`test_*.py`, `validate_*.py`) outside
  `tests/`. They are not collected by pytest and duplicate/shadow the organized suite. Consider
  moving into `tests/` or removing.

---

## 5. Test run summary (verified locally)

Command (excluding the 3 import-time-erroring files and with a per-test timeout):

```
python -m pytest tests/ -q --timeout=20 \
  --ignore=tests/test_diagnosis.py \
  --ignore=tests/test_portfolio_examples.py \
  --ignore=tests/test_proper_patterns.py
```

Result:

```
1611 passed, 213 skipped, 11 warnings, 20 errors in 78.15s
```

- **20 errors** — all `tests/test_geometry_benchmarks.py`, missing `benchmark` fixture (Finding #3),
  not product defects.
- **213 skipped** — consistent with the documented MLIR-path skips and a couple of pre-existing
  conformer-generation API skips noted in the changelog.

---

## 6. Recommended next steps (ranked)

| # | Action | Effort | Why |
|---|--------|--------|-----|
| 1 | Fix the CLI type-checker: add the 11 missing `visit_*` defaults (+ a `generic_visit` fallback) to `ASTVisitor` | ~30 min | Unblocks `morphogen run` / `check` — the #1 thing a new user hits |
| 2 | Add a CLI smoke test over `examples/*.morph` | ~1 hr | Closes the gap that let #1 ship; guards the whole pipeline |
| 3 | Declare `Pillow` / `scipy` / `pytest-benchmark` (runtime vs. test extras) in `pyproject.toml`; relocate the 10 root-level test scripts | ~1 hr | Clean install + clean test run out of the box |
| 4 | Then the roadmap's own P0 — narrative examples for `fluid_jet`, `audio_analysis`, `instrument_model`, and the `audio_analysis → instrument_model` cross-domain showcase | — | Demonstrates the project's unique thesis |

Items 1–3 are small, self-contained, and squarely prerequisite to the v1.0 goal of
"`getting started guide < 10 minutes`" — which currently fails at the first command.

---

## Appendix — Key file references

- CLI type-check invocation: `morphogen/cli.py:146` (`cmd_run`), `morphogen/cli.py:204` (`cmd_check`)
- Base visitor (missing methods): `morphogen/ast/visitors.py:7` (`ASTVisitor`)
- Type checker: `morphogen/ast/visitors.py:96` (`TypeChecker`)
- Node `accept()` dispatch: `morphogen/ast/nodes.py`
- Runtime (works when type-check is bypassed): `morphogen/runtime/runtime.py`
