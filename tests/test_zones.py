import cv2
import numpy as np
import pytest

from drishti.config import load_config
from drishti.zones import Zones, grid_frame


def test_parse_and_inside_normalised_and_pixels():
    z = Zones(roi="0.2,0.2 0.8,0.2 0.8,0.8 0.2,0.8")
    assert z.inside(500, 300, 1000, 600)
    assert not z.inside(50, 300, 1000, 600)
    zp = Zones(roi=[[200, 120], [800, 120], [800, 480], [200, 480]])
    assert zp.inside(500, 300, 1000, 600) and not zp.inside(900, 300, 1000, 600)
    with pytest.raises(ValueError):
        Zones(roi="0.1,0.1 0.2,0.2")


def test_line_crossing_and_direction():
    z = Zones(trigger_line="0,0.5 1,0.5", trigger_direction="down")
    assert z.crossed((100, 280), (100, 320), 1000, 600)       # moving down across y=300
    assert not z.crossed((100, 320), (100, 280), 1000, 600)   # moving up: wrong direction
    assert not z.crossed((100, 200), (100, 280), 1000, 600)   # never reaches the line
    assert Zones(trigger_line="0,0.5 1,0.5").crossed((100, 320), (100, 280), 1000, 600)  # any direction


def test_grid_frame_draws():
    img = np.zeros((360, 640, 3), np.uint8)
    out = grid_frame(img, Zones(roi="0.1,0.1 0.9,0.1 0.9,0.9", trigger_line="0,0.6 1,0.6"))
    assert out.shape == img.shape and out.sum() > 0


@pytest.fixture(scope="module")
def engine():
    try:
        from drishti.pipeline import Drishti

        return Drishti(load_config(overrides={"models": {"vehicle_detector": None}}))
    except Exception as exc:
        pytest.skip(f"models unavailable: {exc}")


def _frames(n=14):
    from .test_pipeline import synthetic_scene

    base = synthetic_scene()
    for f in range(n):  # plate moves down the image (vehicle approaching an overhead camera)
        M = np.float32([[1, 0, 0], [0, 1, -120 + 14 * f]])
        yield cv2.warpAffine(base, M, (1280, 720), borderMode=cv2.BORDER_REPLICATE)


def test_trigger_line_emits_once_at_the_line(engine):
    engine.cfg.zones = {"roi": None, "trigger_line": [[0.0, 0.75], [1.0, 0.75]], "trigger_direction": "down",
                        "emit_uncrossed": False}
    try:
        sess = engine.session("lane-1", fps=10)
        events = []
        for f, img in enumerate(_frames()):
            evs = sess.step(img, f)
            for e in evs:
                e.emitted_at = f
            events += evs
        events += sess.finish()
    finally:
        engine.cfg.zones = {"roi": None, "trigger_line": None, "trigger_direction": "any", "emit_uncrossed": True}
    line_events = [e for e in events if e.trigger == "line"]
    assert len(line_events) == 1
    assert line_events[0].emitted_at < 13            # decided before the track ended
    assert all(e.corrects == line_events[0].event_id for e in events if e is not line_events[0])


def test_roi_excludes_other_lane(engine):
    engine.cfg.zones = {"roi": [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]], "trigger_line": None}
    try:
        sess = engine.session("lane-1", fps=10)
        events = []
        for f, img in enumerate(_frames(8)):
            events += sess.step(img, f)
        events += sess.finish()
    finally:
        engine.cfg.zones = {"roi": None, "trigger_line": None, "trigger_direction": "any", "emit_uncrossed": True}
    assert events == []  # the plate is in the middle of the frame, outside this lane
