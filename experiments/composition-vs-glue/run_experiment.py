"""Composition-vs-glue experiment runner (BACKLOG P1-1).

For each task: run both implementations, verify each produces valid finite audio,
measure logical LOC of the two `*_version` function BODIES (the fair unit — shared
signal-gen / sonify helpers are excluded), and check reproducibility (bit-identical
on a second run). Prints a scorecard.
"""
import ast
import inspect
import numpy as np

import task1_circuit_clipper as t1
import task2_oscillator_sonify as t2
import task3_composer_engine as t3


def logical_loc(fn):
    """Count non-blank, non-comment, non-docstring source lines of a function body."""
    src = inspect.getsource(fn)
    tree = ast.parse(src).body[0]
    body = tree.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)):
        body = body[1:]                       # drop docstring
    lines = set()
    for node in body:
        for n in ast.walk(node):
            if hasattr(n, "lineno"):
                lines.add(n.lineno)
    return len(lines)


def valid(sig):
    return sig is not None and len(sig) > 100 and np.all(np.isfinite(sig)) and np.max(np.abs(sig)) > 1e-6


def reproducible(fn):
    a, _ = fn(); b, _ = fn()
    return bool(np.array_equal(a, b))


def score_task(name, mod):
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    rows = []
    for tag, fn in (("morphogen", mod.morphogen_version), ("glue", mod.glue_version)):
        sig, label = fn()
        rows.append((tag, logical_loc(fn), valid(sig), reproducible(fn), label))
        print(f"  {tag:<10} LOC={rows[-1][1]:>3}  valid={rows[-1][2]}  reproducible={rows[-1][3]}  {label}")
    m, g = rows[0], rows[1]
    print(f"  --> LOC delta (morphogen - glue): {m[1] - g[1]:+d}")
    return rows


def main():
    score_task("TASK 1 — Circuit soft-clipper (Morphogen's strong case)", t1)
    score_task("TASK 2 — Oscillator sonification (composition is glue in both)", t2)
    print(f"\n{'='*66}\nTASK 3 — The advertised composition engine\n{'='*66}")
    for s, t, r in t3.probe_composer():
        print(f"  {s:>8} -> {t:<8}  {r}")


if __name__ == "__main__":
    main()
