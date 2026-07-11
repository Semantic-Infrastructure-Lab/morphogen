# Morphogen experiments — reality checks with runnable code

Small, runnable experiments that test Morphogen's core theses with numbers instead of
slogans. Each is self-contained and reproducible.

## `composition-vs-glue/` — does data-plumbing composition beat glue? (BACKLOG P1-1)
**Verdict: no.** Two cross-domain tasks implemented twice (Morphogen vs raw scipy/numpy glue)
plus a probe of the advertised composer engine. Glue was equal-or-fewer LOC in every task; the
composer runs 0/5 pairs. Real value is the rigorous *library*, not composition.
Run: `python composition-vs-glue/run_experiment.py`.
Writeup: [`docs/reviews/2026-07-11-composition-vs-glue-experiment.md`](../docs/reviews/2026-07-11-composition-vs-glue-experiment.md).

## `coupled-feedback/` — the missing coupling substrate, proven and then built
**Verdict: the one layer Morphogen never built is small, it matters, and it now ships.**

- **`engine_governor.py`** — the original proof. An engine idle governor co-advances two
  rigorous domains (`integrators` mechanics ↔ `controls` PID) with real per-timestep feedback
  in a ~15-line hand loop. Governor engaged holds 800 RPM under a load lurch (RMS error 13.7);
  a *fairly-tuned* fixed-throttle open loop droops to ~540 RPM (RMS error 136.3) — feedback
  coupling wins **~10×**. Sonified + plotted. This is what the one-shot composer cannot do.
- **`coupled_engine.py`** — the proof, generalized. Same idea rebuilt on the reusable
  **`morphogen.coupling.couple`** substrate (the ~15-line loop is now a tested library module)
  and scaled to **three** mutually-coupled rigorous domains: thermal ↔ mechanics ↔ control. A
  cold-start warm-up where temperature-dependent drag and work-dependent heating feed back every
  timestep — the governor floors the throttle cold, then self-relaxes to ~60% as the block warms
  and drag falls, holding idle throughout. Plotted (`coupled_engine.png`) + sonified.

Run: `python coupled-feedback/engine_governor.py` · `python coupled-feedback/coupled_engine.py`.
Substrate: [`morphogen/coupling/`](../morphogen/coupling/) · guard: `tests/test_coupling.py`.
Grounding: [`docs/reviews/2026-07-11-vision-vs-reality.md`](../docs/reviews/2026-07-11-vision-vs-reality.md).
