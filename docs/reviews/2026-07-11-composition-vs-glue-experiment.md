---
title: "Morphogen — Composition-vs-Glue Experiment (P1-1 result)"
type: analysis
beth_topics:
  - morphogen
  - cross-domain
  - composition
  - experiment
  - composer
  - paths-forward
  - strategy
---

# Composition vs Glue — Does Morphogen's Cross-Domain Machinery Earn Its Keep?

**Date:** 2026-07-11 (chaotic-star-0711)
**Backlog item:** [P1-1](../../BACKLOG.md) — "the decisive experiment," highest information value.
**Basis:** [2026-07-11-code-reality-and-paths-forward.md](2026-07-11-code-reality-and-paths-forward.md).
**Code:** [`experiments/composition-vs-glue/`](../../experiments/composition-vs-glue/) — runnable,
reproducible (`python run_experiment.py`).

> **The question this decides.** Morphogen's entire *platform* premise is one sentence:
> *"typed cross-domain composition beats hand-rolled glue."* Everything else — the composer,
> the DSL-as-language, MLIR, the "universal ontology" framing — rests on it. It had never
> been tested. This experiment tests it with runnable code and numbers instead of a slogan.

## Method

Two real cross-domain tasks, each implemented **twice** — once through Morphogen's domains,
once with the raw-numpy/scipy glue a competent developer would actually write — plus a direct
probe of the advertised composition **engine**. For each version we measured logical LOC (of
the working function body, shared helpers excluded), validity of output, and bit-reproducibility.

- **Task 1 — Circuit soft-clipper** (audio → nonlinear analog circuit → audio). Morphogen's
  *strongest* case: it owns an integrated cross-domain op, `circuit.process_audio`, backed by
  a real MNA + Newton-Raphson diode solver.
- **Task 2 — Damped-oscillator sonification** (physics → audio). Morphogen's *weakest* case:
  it owns a rigorous integrator, but **no bridge provides the physics→audio mapping** — that is
  hand-written glue in both versions (exactly as the canonical `physics_to_audio` demo does).
- **Task 3 — The composer engine** (`auto_compose` / `TransformPipeline`) run on the five
  domain pairs its own registry advertises.

## Results

| Task | Morphogen LOC | Glue LOC | Δ | Both valid? | Both reproducible? |
|---|---|---|---|---|---|
| 1 — circuit clipper | 12 | **7** | **+5** | yes | yes |
| 2 — oscillator sonify | 11 | **10** | **+1** | yes | yes |

**Task 3 — the engine:** `auto_compose` executes **0 of 5** advertised pairs (4 fail at call
with `ValueError` on auto-instantiation, 1 has no registered path). Corroborating context from
the codebase: of **93 example files, exactly 1** imports the composer — and that one hides its
own failure in a `try/except` that prints a reassuring "note."

## What the numbers say

**1. LOC is not Morphogen's advantage — the "composition saves you code" premise is false at
this scale.** In *both* tasks, including the one hand-picked to favor Morphogen, the raw-glue
version was equal or **fewer** lines. Composition did not reduce code.

**2. The composition *engine* adds nothing and doesn't work.** It runs 0/5 pairs, is used by
1/93 examples, and — even if fixed — LOC shows it would not be solving a real problem, because
glue is already short. The thing Morphogen sells hardest is the thing with the least value.

**3. Morphogen's real advantage is narrow, and it is a *library* advantage, not a *composition*
advantage: correctness of hard primitives.** In Task 1 the glue `tanh` clipper is a perceptual
*approximation* — opposite polarity (the op-amp is inverting) and, tellingly, a **2× lower
spectral centroid** (real diode clipping generates markedly richer harmonics). To obtain
Morphogen's physically-correct diode response by hand you would reimplement the ~**1,112-line**
MNA + Newton-Raphson solver in `circuit.py`. So Morphogen wins **only when you need real
physics**, and it wins as *a validated primitive you'd otherwise have to write* — not because
"composition beats glue."

**4. Where the primitive is easy, Morphogen adds ~nothing.** In Task 2 a velocity-Verlet step
is 4 hand-written lines; `integrators.symplectic` saved +1 line net and produced the same
result, and the cross-domain mapping was hand glue either way.

## Verdict

**The composition thesis is false as stated.** "Typed cross-domain composition beats
hand-rolled glue" is not supported: glue is shorter, and the engine meant to deliver the thesis
neither runs nor would help if it did. What *is* real is orthogonal to composition — a library
of **rigorous, validated domain primitives** that save you reimplementing hard physics (Task 1's
1,112-line solver), used by writing a few lines of ordinary glue (as every canonical demo
already does).

### Decisions this resolves

- **P0-2 → option (b): retire the composer.** Delete `auto_compose` / `TransformPipeline` /
  registry-as-engine (or quarantine it, clearly labeled "experimental, non-functional"). Keep
  the bridges usable by direct instantiation. Remove the failure-hiding `try/except` from
  `examples/cross_domain/01_transform_composition.py`. No user loses anything: 1/93 examples
  touched it and that one never ran.
- **Path A (Honest Library) is the whole product**, confirmed by evidence. The pitch is "a
  tested library of rigorous, interoperable scientific domains you compose with a few lines of
  Python," not "a universal composition platform/language."
- **Path C (reproducible science) remains the sharpest positioning** and is now the highest-value
  *forward* bet — but it is gated on **P0-1 (make determinism real)**, which is the one property
  that would give the library a defensible edge glue does not have.

### What does NOT change

The rigorous domains keep their value — Task 1 is exactly the kind of thing they're good at.
This experiment doesn't diminish the library; it relocates the value from the (broken)
composition layer to the (real) primitive layer, and tells us to stop paying rent on the former.
