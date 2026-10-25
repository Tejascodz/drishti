
import cv2
import numpy as np

from drishti.config import load_config
from drishti.health import CameraHealth
from drishti.imaging import BadPixelCorrector, edge_enhance, outlier_mask
from drishti.store import EventStore
from drishti.tolling import AVERAGE, GOOD, POOR, confidence_level, toll_decision
from drishti.types import BBox, PlateEvent


def _text_image():
    img = np.full((60, 200, 3), 230, np.uint8)
    cv2.putText(img, "KA05MX3021", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def test_bpc_removes_isolated_and_cluster_defects_but_keeps_text():
    clean = _text_image()
    bad = clean.copy()
    bad[10, 20] = 255; bad[50, 150] = 0; bad[8:10, 100:102] = 0; bad[52, 60] = 0  # hot, dead, 2x2 cluster
    fixed = BadPixelCorrector(learn=False)(bad)
    assert np.abs(fixed.astype(int) - clean.astype(int)).max() < 40
    m = outlier_mask(clean)
    assert m.sum() <= 3  # strokes of characters are not mistaken for defects


def test_bpc_learns_stuck_pixel_over_time():
    rng = np.random.default_rng(0)
    bpc = BadPixelCorrector(learn=True, min_frames=20, threshold=200)  # threshold too high for dynamic: static map must catch it
    for _ in range(40):
        f = rng.integers(60, 200, (40, 40, 3), dtype=np.uint8)
        f[20, 20] = 128  # stuck: never changes while the scene does
        bpc(f)
    assert bpc.static is not None and bpc.static[20, 20]


def test_edge_enhance_increases_sharpness():
    soft = cv2.GaussianBlur(_text_image(), (0, 0), 1.5)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    assert lap(edge_enhance(soft)) > lap(soft)


def test_confidence_levels():
    assert confidence_level(0.97) == GOOD and confidence_level(0.85) == AVERAGE and confidence_level(0.5) == POOR


class _Eng:
    cfg = load_config()


def _ev(text, conf, width=140):
    return PlateEvent("e1", 1, "cam", text, conf, "IN-STD", 0, 1, 0, 10, 10, 10, BBox(0, 0, width, 30), plate_width_px=width)


def test_toll_rules():
    e = _Eng()
    assert toll_decision(e, _ev("KA05MX3021", 0.97))["action"] == "auto"                     # Good, > 95%
    assert toll_decision(e, _ev("KA05MX3021", 0.90))["action"] == "manual_review"            # Average, not > 95%
    assert toll_decision(e, _ev(None, 0.0))["action"] == "manual_review"                     # unreadable
    r = toll_decision(e, _ev("KA05MX3021", 0.97, width=70))
    assert any("px wide" in x for x in r["reasons"])                                       # pixel density warning
    d = _ev("KA05MX3021", 0.99); d.duplicate_of = "e0"
    assert toll_decision(e, d)["action"] == "duplicate"


def test_dedupe_across_cameras(tmp_path):
    st = EventStore(tmp_path / "e.db")
    st.add_event(_ev("KA05MX3021", 0.9), "KA05MX3021")
    assert st.recent_similar("KA05MX3021", 10, exclude_camera="cam2") == "e1"
    assert st.recent_similar("KA05MXO021", 10, exclude_camera="cam2") == "e1"   # O vs 0
    assert st.recent_similar("KA05MX3021", 10, exclude_camera="cam") is None    # same camera
    assert st.recent_similar("MH12DE1433", 10, exclude_camera="cam2") is None


def test_health_alerts_frozen_dark_and_moved():
    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (0, 0), 2)
    h = CameraHealth("lane1", baseline_checks=3, frozen_after_s=3)
    raised = lambda alerts: {a["check"] for a in alerts if a["state"] == "raised"}
    seen = set()
    for t in range(6):  # normal, changing frames
        seen |= raised(h.update(np.clip(scene.astype(int) + rng.integers(-3, 4, scene.shape), 0, 255).astype(np.uint8), t))
    assert not seen
    for t in range(6, 12):  # frozen: identical frames
        seen |= raised(h.update(scene, t))
    assert "frozen_feed" in seen
    moved = np.roll(scene, 40, axis=1)
    for t in range(12, 17):
        seen |= raised(h.update(np.clip(moved.astype(int) + rng.integers(-3, 4, scene.shape), 0, 255).astype(np.uint8), t))
    assert "camera_moved" in seen
    for t in range(17, 21):
        seen |= raised(h.update(np.zeros_like(scene), t))
    assert "too_dark" in seen
