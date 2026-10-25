import numpy as np

from drishti.formats import FormatGrammar
from drishti.fusion import Observation, align_and_stack, combined_confidence, fuse, levenshtein
from drishti.track import ByteTracker
from drishti.types import BBox, Detection
from drishti.watchlist import WatchEntry, Watchlist, weighted_distance

from .test_grammar import ALPHA, onehot

G = FormatGrammar.from_pack("india", ALPHA, "_")


def obs(text, w=1.0, q=0.5, **kw):
    lp = onehot(text, **kw)
    return Observation(lp, w, G.decode(lp), text, q)


def test_levenshtein():
    assert levenshtein("KA05", "KA05") == 0
    assert levenshtein("KA05", "KA5") == 1
    assert levenshtein("", "ABC") == 3


def test_fusion_majority_and_logit_average():
    o = [obs("KA05MX3021"), obs("KA05MX3021"), obs("KA05MX3027", w=0.5)]
    fr = fuse(o, G)
    assert fr.decode.text == "KA05MX3021"
    assert 0.6 < fr.agreement < 1.0
    assert fr.logp.shape == (10, len(ALPHA))


def test_fusion_ignores_layout_outliers():
    # a truncated read (different length) must not corrupt slot-aligned averaging
    o = [obs("MH12DE1433"), obs("MH12DE1433"), obs("H12DE1433", w=0.8)]
    assert fuse(o, G).decode.text == "MH12DE1433"


def test_fusion_handles_all_empty():
    fr = fuse([obs(""), obs("")], G)
    assert fr.decode.text == "" and fr.agreement == 0.0


def test_combined_confidence_bounds_and_calibration():
    assert combined_confidence(0.9, 1.0, 1, 0.5, 0.0) == 0.9
    assert combined_confidence(0.9, 0.0, 1, 0.5, 0.0) == 0.45
    c = combined_confidence(0.9, 1.0, 5, 0.5, 0.0, {"bias": 0.0, "w_logit": 1.0})
    assert abs(c - 0.9) < 1e-6


def test_align_and_stack_recovers_signal():
    rng = np.random.default_rng(0)
    clean = np.full((40, 160, 3), 220, np.uint8)
    import cv2

    cv2.putText(clean, "KA05MX3021", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    frames = []
    for k in range(6):
        M = np.float32([[1, 0, rng.uniform(-1.5, 1.5)], [0, 1, rng.uniform(-1.5, 1.5)]])
        shifted = cv2.warpAffine(clean, M, (160, 40), borderMode=cv2.BORDER_REPLICATE)
        frames.append(np.clip(shifted + rng.normal(0, 30, shifted.shape), 0, 255).astype(np.uint8))
    stacked = align_and_stack(frames, [1.0] * 6)
    assert stacked is not None
    single = cv2.resize(frames[0], (stacked.shape[1], stacked.shape[0]), interpolation=cv2.INTER_CUBIC)
    # noise in a text-free strip drops when frames are registered and averaged
    flat = (slice(0, stacked.shape[0] // 8), slice(None))
    assert stacked[flat].astype(float).std() < 0.8 * single[flat].astype(float).std()


def _det(x, y, w=60, h=20, s=0.9):
    return Detection(BBox(x, y, x + w, y + h), s)


def test_tracker_keeps_identity_through_motion_and_gap():
    tr = ByteTracker(max_age=5)
    ids = []
    for f in range(10):
        dets = [] if f == 5 else [_det(100 + 12 * f, 200 + 4 * f), _det(600 - 10 * f, 300)]
        m = tr.update(dets, f)
        ids.append(sorted(t.track_id for t, _ in m))
    assert ids[0] == [1, 2]
    assert all(i == [1, 2] for k, i in enumerate(ids) if k != 5)


def test_tracker_low_score_second_stage_and_removal():
    tr = ByteTracker(high=0.5, low=0.1, max_age=2)
    tr.update([_det(100, 100, s=0.9)], 0)
    m = tr.update([_det(103, 101, s=0.3)], 1)  # weak detection keeps the track alive
    assert len(m) == 1 and m[0][0].track_id == 1
    for f in range(2, 6):
        tr.update([], f)
    removed = tr.pop_removed()
    assert [t.track_id for t in removed] == [1]


def test_watchlist_confusion_aware():
    wl = Watchlist([WatchEntry("KA05MX3021", "stolen")], max_distance=1.0)
    assert wl.match("KA05MX3021").distance == 0
    assert wl.match("KA05MX302I") is not None  # 1 vs I is a cheap substitution
    assert wl.match("KA05MX3O21") is not None
    assert wl.match("KA05MX9999") is None
    assert wl.match(None, [("KA05MX3021", 0.2)]) is not None  # found via alternatives
    assert weighted_distance("0", "O") < weighted_distance("0", "7")
