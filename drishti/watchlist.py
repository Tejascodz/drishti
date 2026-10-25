"""Hotlist matching with OCR-confusion-aware edit distance.

A plain Levenshtein distance treats O->0 the same as O->7. Here substitutions
between visually confusable glyphs cost less, so a watchlisted 'KA05MX3021'
still matches a read of 'KA05MX302I' while unrelated plates stay far apart.
Alternatives from the decoder are checked too.
"""

from __future__ import annotations

from dataclasses import dataclass

from .formats.confusables import CONFUSABLE_GROUPS

_SUB: dict[tuple[str, str], float] = {}
for group in CONFUSABLE_GROUPS:
    for a in group:
        for b in group:
            if a != b:
                _SUB[(a, b)] = min(_SUB.get((a, b), 1.0), 0.35)


def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _SUB.get((a, b), 1.0)


def weighted_distance(a: str, b: str, indel: float = 1.0) -> float:
    a, b = a.upper(), b.upper()
    prev = [j * indel for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [i * indel]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + indel, cur[j - 1] + indel, prev[j - 1] + sub_cost(ca, cb)))
        prev = cur
    return prev[-1]


def normalize_plate(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


@dataclass
class WatchEntry:
    plate: str
    reason: str = ""
    level: str = "alert"  # alert | info | allow


@dataclass
class WatchHit:
    entry: WatchEntry
    matched_text: str
    distance: float

    def to_dict(self) -> dict:
        return {"plate": self.entry.plate, "reason": self.entry.reason, "level": self.entry.level,
                "matched_text": self.matched_text, "distance": round(self.distance, 3)}


class Watchlist:
    def __init__(self, entries: list[WatchEntry] | None = None, max_distance: float = 1.0):
        self.max_distance = max_distance
        self.entries: dict[str, WatchEntry] = {}
        for e in entries or []:
            self.add(e)

    def add(self, entry: WatchEntry) -> None:
        entry.plate = normalize_plate(entry.plate)
        self.entries[entry.plate] = entry

    def remove(self, plate: str) -> bool:
        return self.entries.pop(normalize_plate(plate), None) is not None

    def match(self, text: str | None, alternatives: list[tuple[str, float]] | None = None) -> WatchHit | None:
        if not self.entries:
            return None
        cands = [t for t in [text] if t] + [t for t, _ in (alternatives or [])]
        best: WatchHit | None = None
        for cand in dict.fromkeys(normalize_plate(c) for c in cands):
            exact = self.entries.get(cand)
            if exact:
                return WatchHit(exact, cand, 0.0)
            for plate, e in self.entries.items():
                if abs(len(plate) - len(cand)) > self.max_distance:
                    continue
                d = weighted_distance(plate, cand)
                if d <= self.max_distance and (best is None or d < best.distance):
                    best = WatchHit(e, cand, d)
        return best
