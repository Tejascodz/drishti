"""Evidence fusion across crop views and video frames.

Every OCR pass yields an (S, V) probability matrix. Observations that agree on
layout (same decoded length, small edit distance to the weighted-vote winner)
are fused at the log-probability level (weighted geometric mean, renormalised
per slot) and decoded once more under the format grammar. This is the
track-level "logit aggregation + character voting" strategy that the strongest
low-resolution plate systems converge on, combined here with format constraints.

`align_and_stack` adds multi-frame image restoration: the best crops of a track
are registered to a reference with ECC (affine) and averaged, which suppresses
noise and compression artefacts before one extra OCR pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .formats.grammar import DecodeResult, FormatGrammar


@dataclass
class Observation:
    logp: np.ndarray  # (S, V)
    weight: float
    decode: DecodeResult
    source: str = ""
    quality: float = 0.0


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _renorm(logp: np.ndarray) -> np.ndarray:
    m = logp.max(axis=1, keepdims=True)
    return logp - (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True)))


@dataclass
class FusionResult:
    decode: DecodeResult
    agreement: float
    n_used: int
    n_total: int
    vote_text: str
    logp: np.ndarray | None = None  # fused (S, V) log-probabilities


def fuse(observations: list[Observation], grammar: FormatGrammar, mode: str = "strict", fallback_nats: float = 7.0, top_k: int = 5) -> FusionResult:
    obs = [o for o in observations if o.weight > 0 and o.decode.text]
    if not obs:  # nothing legible: return the strongest (empty / weak) observation as-is
        o = max(observations, key=lambda x: x.weight)
        return FusionResult(o.decode, 0.0, 0, len(observations), o.decode.text, o.logp)
    if len(obs) == 1:
        o = obs[0]
        return FusionResult(o.decode, 1.0, 1, 1, o.decode.text, o.logp)

    # 1) weighted vote over decoded strings (weight * confidence^0.5 keeps weak reads from dominating)
    votes: dict[str, float] = {}
    for o in obs:
        votes[o.decode.text] = votes.get(o.decode.text, 0.0) + o.weight * math.sqrt(max(o.decode.prob, 1e-6))
    vote_text = max(votes.items(), key=lambda kv: kv[1])[0]

    # 2) keep observations consistent with the vote winner (same slot layout)
    tol = max(1, int(0.3 * len(vote_text)))
    keep = [o for o in obs if len(o.decode.text) == len(vote_text) and levenshtein(o.decode.text, vote_text) <= tol]
    if not keep:
        keep = obs

    # 3) weighted geometric mean of the per-slot distributions, renormalised, then decode
    wsum = sum(o.weight for o in keep)
    fused = sum(o.weight * o.logp for o in keep) / wsum
    fused = _renorm(fused)
    dec = grammar.decode(fused, mode=mode, fallback_nats=fallback_nats, top_k=top_k)

    total = sum(o.weight for o in obs)
    agree = sum(o.weight for o in obs if o.decode.text == dec.text) / total
    return FusionResult(dec, float(agree), len(keep), len(obs), vote_text, fused)


def combined_confidence(prob: float, agreement: float, n_obs: int, quality: float, penalty: float, calib: dict | None = None) -> float:
    """Plate-level confidence.

    Default: fused string probability scaled by cross-view agreement. With a
    calibration file (drishti calibrate) a logistic model over the same features
    is used instead, which is what tightens the gap between the confidence of
    correct and incorrect reads.
    """
    if calib:
        p = min(max(prob, 1e-6), 1 - 1e-6)
        z = (
            calib.get("bias", 0.0)
            + calib.get("w_logit", 1.0) * math.log(p / (1 - p))
            + calib.get("w_agree", 0.0) * agreement
            + calib.get("w_logn", 0.0) * math.log(max(n_obs, 1))
            + calib.get("w_quality", 0.0) * quality
            + calib.get("w_penalty", 0.0) * penalty
        )
        return 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(prob * (0.5 + 0.5 * agreement), 0.0, 1.0))


def align_and_stack(crops: list[np.ndarray], weights: list[float] | None = None, min_corr: float = 0.55) -> np.ndarray | None:
    """Register crops to the highest-weight one (ECC affine) and average them."""
    if len(crops) < 3:
        return None
    weights = weights or [1.0] * len(crops)
    order = np.argsort(weights)[::-1]
    ref = crops[order[0]]
    th = int(np.clip(ref.shape[0] * 2, 48, 128))
    tw = int(round(ref.shape[1] * th / ref.shape[0]))
    if tw < 16:
        return None

    def prep(c):
        r = cv2.resize(c, (tw, th), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return r.astype(np.float32), g

    ref_c, ref_g = prep(ref)
    acc = ref_c * weights[order[0]]
    wacc = weights[order[0]]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    used = 1
    for i in order[1:]:
        c, g = prep(crops[i])
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            corr, warp = cv2.findTransformECC(ref_g, g, warp, cv2.MOTION_AFFINE, crit, None, 3)
        except cv2.error:
            continue
        if corr < min_corr:
            continue
        aligned = cv2.warpAffine(c, warp, (tw, th), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        acc += aligned * weights[i]
        wacc += weights[i]
        used += 1
    if used < 3:
        return None
    mean = acc / wacc
    blur = cv2.GaussianBlur(mean, (0, 0), 1.0)
    sharp = cv2.addWeighted(mean, 1.6, blur, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
