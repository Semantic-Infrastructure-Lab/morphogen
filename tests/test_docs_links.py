"""Lightweight checks for local links in high-signal project docs."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "STATUS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "DOCUMENTATION_INDEX.md",
    ROOT / "docs" / "ROADMAP.md",
    ROOT / "docs" / "STRATEGY.md",
    ROOT / "docs" / "PITCH.md",
    ROOT / "docs" / "DOMAINS.md",
    ROOT / "docs" / "getting-started.md",
    ROOT / "docs" / "PROGRESS_2026-04-17.md",
]

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _iter_local_targets(doc_path: Path):
    text = doc_path.read_text(encoding="utf-8")
    for match in LINK_RE.finditer(text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        yield target


def test_high_signal_docs_have_valid_local_links():
    """README and top-level docs should not contain broken local links."""
    broken: list[str] = []

    for doc in DOC_FILES:
        for target in _iter_local_targets(doc):
            resolved = (doc.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")

    assert not broken, "Broken local doc links:\n" + "\n".join(sorted(broken))
