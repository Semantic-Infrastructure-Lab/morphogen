---
title: "Morphogen reviews — reading order"
type: reference
beth_topics:
  - morphogen
  - reviews
  - assessment
  - vision
  - reality
---

# Morphogen reviews

The honest, evidence-grounded picture of the project. Read in this order — each builds
on the last, and together they supersede the grandiose framing in the legacy front-door
docs (README top, `philosophy/vision-and-value.md`, `PITCH.md`).

1. **[2026-07-11-code-reality-and-paths-forward.md](2026-07-11-code-reality-and-paths-forward.md)**
   — *What is actually real in the code.* Five-pass audit (independently re-verified): ~11
   rigorous domains are genuine; the composer doesn't execute, determinism is decorative,
   MLIR/symbolic/category-theory are unbuilt; qchem is the sole fabricated domain. Three
   paths forward.

2. **[2026-07-11-composition-vs-glue-experiment.md](2026-07-11-composition-vs-glue-experiment.md)**
   — *The decisive experiment (BACKLOG P1-1).* Data-plumbing composition does **not** beat
   hand-glue (glue is equal-or-fewer LOC; composer runs 0/5 pairs). Value is the rigorous
   *library*, not composition. Decides: retire the composer. Code: `experiments/composition-vs-glue/`.

3. **[2026-07-11-vision-vs-reality.md](2026-07-11-vision-vs-reality.md)**
   — *The founding vision, grounded.* All three founding visions (transform algebra,
   cross-domain tool reuse, coupled-multiphysics "engine") reduce to **one missing layer**:
   a coordinating driver over the rigorous domains. Proven small and useful by a runnable
   coupled-feedback demo (`experiments/coupled-feedback/`). Recommends the coupling substrate
   as the buildable North Star.

Superseded: `2026-07-11-project-assessment.md` (its "all findings closed" framing is
corrected by #1 above). The actionable work these generate lives in **[/BACKLOG.md](../../BACKLOG.md)**.
