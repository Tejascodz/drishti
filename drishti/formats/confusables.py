"""Glyph groups that plate OCR models confuse with each other."""

from __future__ import annotations

CONFUSABLE_GROUPS = ["0OQD", "1IJLT7", "2Z", "5S", "6G", "8B", "4A", "UV", "MN", "KX", "CG", "EF", "PR", "HN", "3B"]


def confusable_map() -> dict[str, set[str]]:
    m: dict[str, set[str]] = {}
    for g in CONFUSABLE_GROUPS:
        for a in g:
            m.setdefault(a, set()).update(c for c in g if c != a)
    return m


CONFUSABLE = confusable_map()
