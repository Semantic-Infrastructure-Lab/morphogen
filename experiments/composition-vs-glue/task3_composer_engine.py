"""Task 3 — The advertised composition ENGINE itself.

Morphogen's headline platform claim is "first-class, type-safe *composable*
cross-domain operations." The engine that would deliver that is
`morphogen.cross_domain.composer` (auto_compose / TransformPipeline). Tasks 1-2
never touch it — they use domain methods and hand glue, exactly like all 92 of
93 example files. This task tests the engine directly, on the pairs its own
registry advertises.
"""
import numpy as np


def probe_composer():
    from morphogen.cross_domain.composer import auto_compose
    pairs = [("field", "agent"), ("audio", "visual"), ("fluid", "acoustics"),
             ("agent", "field"), ("visual", "audio")]
    results = []
    for s, t in pairs:
        try:
            pipe = auto_compose(s, t)
            try:
                pipe(np.zeros((8, 8)))
                results.append((s, t, "RUNS"))
            except Exception as e:
                results.append((s, t, f"CALL-FAILS: {type(e).__name__}"))
        except Exception as e:
            results.append((s, t, f"COMPOSE-FAILS: {type(e).__name__}"))
    return results


if __name__ == "__main__":
    runs = sum(1 for *_, r in probe_composer() if r == "RUNS")
    for s, t, r in probe_composer():
        print(f"  {s:>8} -> {t:<8}  {r}")
    print(f"\nengine executes {runs}/5 advertised pairs")
