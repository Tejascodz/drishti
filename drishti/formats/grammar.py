"""Format-constrained decoding for fixed-slot plate OCR.

A slot OCR model outputs an (S, V) matrix of per-slot character probabilities,
characters left-aligned and the remainder filled with a pad symbol. Instead of
taking the arg-max of each slot independently, we search for the most probable
string that is a *valid plate* under a set of templates ("region pack").

Patterns are written as whitespace-separated tokens:
    D{2}        two characters of class D
    S{1,3}      one to three characters of class S (expanded into variants)
    {STATE}     one item from the multi-character set STATE (scored jointly)
    'BH'        a literal
Classes allow hard constraints (allowed characters) and soft log-penalties
(e.g. Indian series letters avoid I and O). Every expanded template carries a
log-prior. The decoder returns the best string, its model log-probability,
per-character confidences and ranked alternatives.

Search is exact: within a template the slots are independent given the model
output (sets are maximised jointly over their own slots), so the best string
per template is found by per-token arg-max, and the global best by comparing
templates. Cost is O(#templates * S * V), i.e. microseconds.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .confusables import CONFUSABLE

NEG_INF = -1e9


def _expand_chars(spec: str) -> str:
    out = []
    i = 0
    while i < len(spec):
        if i + 2 < len(spec) and spec[i + 1] == "-":
            a, b = spec[i], spec[i + 2]
            out.extend(chr(c) for c in range(ord(a), ord(b) + 1))
            i += 3
        else:
            out.append(spec[i])
            i += 1
    return "".join(dict.fromkeys(out))


@dataclass
class Token:
    width: int
    bias: np.ndarray  # (V,) for width 1; (P,) for sets
    items: np.ndarray | None = None  # (P, width) alphabet indices for sets


@dataclass
class Template:
    name: str
    tokens: list[Token]
    prior: float
    length: int
    pattern: str
    masks: np.ndarray | None = None  # (length, V) allowed characters per slot


@dataclass
class DecodeResult:
    text: str
    fmt: str
    logprob: float  # model log-probability of the full slot sequence (incl. padding)
    score: float  # logprob + format priors / soft penalties
    char_probs: list[float]
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    raw_text: str = ""
    raw_logprob: float = 0.0
    format_penalty: float = 0.0
    cond_logprob: float | None = None  # log-prob counting grammar-resolved look-alike mass as support

    @property
    def prob(self) -> float:
        return float(np.exp(self.logprob if self.cond_logprob is None else self.cond_logprob))


class FormatGrammar:
    def __init__(self, alphabet: str, pad_char: str, pack: dict | None):
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.pad_idx = alphabet.index(pad_char)
        self.index = {c: i for i, c in enumerate(alphabet)}
        self.pack = pack or {}
        self.name = self.pack.get("name", "generic")
        V = len(alphabet)
        self.lookalike = np.zeros((V, V), dtype=bool)
        for a, others in CONFUSABLE.items():
            for b in others:
                if a in self.index and b in self.index:
                    self.lookalike[self.index[a], self.index[b]] = True
        self.templates: list[Template] = self._build() if pack else []

    # ------------------------------------------------------------ building
    def _class_bias(self, cdef: dict) -> np.ndarray:
        chars = _expand_chars(cdef["chars"])
        bias = np.full(len(self.alphabet), NEG_INF, dtype=np.float64)
        for c in chars:
            if c in self.index and c != self.pad_char:
                bias[self.index[c]] = 0.0
        for c, pen in (cdef.get("soft") or {}).items():
            if c in self.index and bias[self.index[c]] > NEG_INF / 2:
                bias[self.index[c]] = float(pen)
        return bias

    def _set_token(self, sdef) -> Token:
        if isinstance(sdef, dict):
            items, soft = list(sdef.get("items", [])), dict(sdef.get("soft") or {})
            items += [k for k in soft if k not in items]
        else:
            items, soft = list(sdef), {}
        width = len(items[0])
        if any(len(s) != width for s in items):
            raise ValueError("all items of a set must have equal length")
        valid = [s for s in items if all(ch in self.index for ch in s)]
        idx = np.array([[self.index[ch] for ch in s] for s in valid], dtype=np.int64)
        bias = np.array([float(soft.get(s, 0.0)) for s in valid], dtype=np.float64)
        return Token(width=width, bias=bias, items=idx)

    def _build(self) -> list[Template]:
        classes = {k: self._class_bias(v) for k, v in (self.pack.get("classes") or {}).items()}
        sets = {k: self._set_token(v) for k, v in (self.pack.get("sets") or {}).items()}
        tok_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\{(\d+)(?:,(\d+))?\})?$")
        templates: list[Template] = []
        for p in self.pack.get("patterns", []):
            options: list[list[list[Token]]] = []  # per pattern token: list of alternative token-sequences
            for raw in p["pattern"].split():
                if raw.startswith("{") and raw.endswith("}"):
                    options.append([[sets[raw[1:-1]]]])
                elif raw[0] in "'\"":
                    lit = raw.strip("'\"")
                    seq = []
                    for ch in lit:
                        b = np.full(len(self.alphabet), NEG_INF)
                        b[self.index[ch]] = 0.0
                        seq.append(Token(1, b))
                    options.append([seq])
                else:
                    m = tok_re.match(raw)
                    if not m or m.group(1) not in classes:
                        raise ValueError(f"bad token '{raw}' in pattern '{p['pattern']}'")
                    lo = int(m.group(2) or 1)
                    hi = int(m.group(3) or lo)
                    bias = classes[m.group(1)]
                    options.append([[Token(1, bias)] * n for n in range(lo, hi + 1)])
            for combo in itertools.product(*options):
                tokens = [t for seq in combo for t in seq]
                length = sum(t.width for t in tokens)
                masks = []
                for t in tokens:
                    if t.items is None:
                        masks.append(t.bias > NEG_INF / 2)
                    else:
                        for k in range(t.width):
                            m = np.zeros(len(self.alphabet), dtype=bool)
                            m[np.unique(t.items[:, k])] = True
                            masks.append(m)
                templates.append(Template(p.get("name", "fmt"), tokens, float(p.get("prior", 0.0)), length, p["pattern"],
                                          np.array(masks) if masks else np.zeros((0, len(self.alphabet)), bool)))
        return templates

    @classmethod
    def from_pack(cls, pack: str | dict | None, alphabet: str, pad_char: str) -> "FormatGrammar":
        if pack is None or pack in ("generic", "none", "free"):
            return cls(alphabet, pad_char, None)
        if isinstance(pack, dict):
            return cls(alphabet, pad_char, pack)
        path = Path(pack)
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            text = resources.files("drishti.formats").joinpath(f"packs/{pack}.yaml").read_text(encoding="utf-8")
            data = yaml.safe_load(text)
        return cls(alphabet, pad_char, data)

    # ------------------------------------------------------------ decoding
    def _to_text(self, idx: list[int]) -> str:
        return "".join(self.alphabet[i] for i in idx if i != self.pad_idx)

    def free_decode(self, logp: np.ndarray) -> tuple[str, float, list[float]]:
        best = logp.argmax(axis=1)
        lp = logp[np.arange(len(best)), best]
        text = self._to_text(best.tolist())
        probs = [float(np.exp(lp[i])) for i, b in enumerate(best) if b != self.pad_idx]
        return text, float(lp.sum()), probs

    def _score_template(self, logp: np.ndarray, t: Template):
        S = logp.shape[0]
        if t.length > S:
            return None
        pos, score, obs = 0, t.prior, 0.0
        chars: list[int] = []
        alts: list[tuple[float, int, int]] = []  # (score delta, slot, alt char) for single-slot tokens
        for tok in t.tokens:
            if tok.items is None:
                s = logp[pos] + tok.bias
                order = np.argsort(-s)
                j = int(order[0])
                score += s[j]
                obs += logp[pos, j]
                chars.append(j)
                if len(order) > 1 and s[order[1]] > NEG_INF / 2:
                    alts.append((float(s[order[1]] - s[j]), pos, int(order[1])))
                pos += 1
            else:
                s = tok.bias.copy()
                for k in range(tok.width):
                    s += logp[pos + k, tok.items[:, k]]
                order = np.argsort(-s)
                j = int(order[0])
                score += s[j]
                obs += float(sum(logp[pos + k, tok.items[j, k]] for k in range(tok.width)))
                chars.extend(int(c) for c in tok.items[j])
                if len(order) > 1:
                    alts.append((float(s[order[1]] - s[j]), -1 - pos, int(order[1])))
                pos += tok.width
        pad_lp = float(logp[pos:, self.pad_idx].sum())
        score += pad_lp
        obs += pad_lp
        return score, obs, chars, alts

    def decode(self, logp: np.ndarray, mode: str = "strict", fallback_nats: float = 7.0, top_k: int = 5) -> DecodeResult:
        """Decode an (S, V) log-probability matrix.

        mode: 'strict' always returns a valid plate (when the pack can express one),
              'auto' falls back to free decoding when the format costs > fallback_nats,
              'free' ignores the pack.
        """
        raw_text, raw_lp, raw_probs = self.free_decode(logp)

        def _free(penalty: float = 0.0) -> DecodeResult:
            # an empty read is no evidence of a plate, however confident the pad slots are
            cond = raw_lp if raw_text else -50.0
            alts = [(raw_text, float(np.exp(cond)))] if raw_text else []
            return DecodeResult(raw_text, "free", raw_lp, raw_lp, raw_probs, alts, raw_text, raw_lp, penalty, cond)

        if mode == "free" or not self.templates or not raw_text:
            return _free()  # nothing legible: never invent a plate from padding

        scored = []
        for t in self.templates:
            r = self._score_template(logp, t)
            if r is not None:
                scored.append((r[0], r[1], r[2], r[3], t))
        if not scored:
            return _free()
        scored.sort(key=lambda x: -x[0])
        score, obs, chars, alts, tmpl = scored[0]
        penalty = max(0.0, raw_lp - obs)

        if mode == "auto" and penalty > fallback_nats:
            return _free(penalty)

        text = self._to_text(chars)
        probs = np.exp(logp)
        resolved = []
        for i, c in enumerate(chars):
            # mass on look-alikes that the format rules out at this slot supports the chosen glyph
            outside = ~tmpl.masks[i] & self.lookalike[c]
            resolved.append(min(1.0, float(probs[i, c] + probs[i, outside].sum())))
        char_probs = resolved
        pad_lp = float(logp[len(chars):, self.pad_idx].sum())
        cond_lp = float(np.sum(np.log(np.maximum(resolved, 1e-12)))) + pad_lp

        # ranked alternatives: best string of each template + single substitutions of the winner
        cands: dict[str, float] = {}
        for sc, ob, ch, _, _t in scored:
            tx = self._to_text(ch)
            if tx not in cands or ob > cands[tx]:
                cands[tx] = ob
        for delta, pos, alt in sorted(alts, key=lambda a: -a[0])[: 2 * top_k]:
            if pos < 0:
                continue  # set alternatives are covered by other templates often enough
            ch = list(chars)
            old = ch[pos]
            ch[pos] = alt
            tx = self._to_text(ch)
            ob = obs - logp[pos, old] + logp[pos, alt]
            if tx not in cands or ob > cands[tx]:
                cands[tx] = float(ob)
        ranked = sorted(cands.items(), key=lambda kv: -kv[1])[:top_k]
        best_p = float(np.exp(cond_lp))
        alternatives = [(t, best_p * float(np.exp(min(0.0, lp - obs)))) for t, lp in ranked]
        return DecodeResult(text, tmpl.name, float(obs), float(score), char_probs, alternatives, raw_text, raw_lp, penalty, cond_lp)

    def string_logprob(self, logp: np.ndarray, text: str) -> float | None:
        """Log-probability the slot OCR gives to exactly `text` (left-aligned, the rest padding)."""
        if len(text) > logp.shape[0] or any(c not in self.index or c == self.pad_char for c in text):
            return None
        idx = [self.index[c] for c in text]
        return float(sum(logp[i, j] for i, j in enumerate(idx)) + logp[len(idx):, self.pad_idx].sum())

    def is_valid(self, text: str) -> str | None:
        """Return the format name if `text` is expressible by the pack, else None."""
        if not self.templates:
            return "free"
        S = max(len(text), 1)
        logp = np.full((S, len(self.alphabet)), np.log(1e-6))
        for i, c in enumerate(text):
            if c not in self.index:
                return None
            logp[i, self.index[c]] = 0.0
        for t in self.templates:
            if t.length != len(text):
                continue
            r = self._score_template(logp, t)
            if r is not None and r[1] > -1e-3 and r[0] > NEG_INF / 2:
                return t.name
        return None
