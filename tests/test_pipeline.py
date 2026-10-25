"""End-to-end tests with the real pretrained models (downloaded on first use).
Skipped automatically when the model hub is unreachable."""

import cv2
import numpy as np
import pytest

from drishti.config import load_config

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    try:
        from drishti.pipeline import Drishti

        cfg = load_config(overrides={"models": {"vehicle_detector": None},
                                     "storage": {"db_path": str(tmp_path_factory.mktemp("db") / "t.db")}})
        return Drishti(cfg)
    except Exception as exc:  # no network / no cache
        pytest.skip(f"models unavailable: {exc}")


def synthetic_scene(text="KA05MX3021", ph=70):
    """A simple rear-of-car scene with a rendered plate (no external images needed)."""
    from PIL import Image, ImageDraw, ImageFont

    img = np.full((720, 1280, 3), 85, np.uint8)
    img[450:] = (60, 60, 62)
    cv2.rectangle(img, (400, 260), (880, 640), (175, 175, 180), -1)
    cv2.rectangle(img, (440, 290), (840, 420), (60, 70, 80), -1)
    cv2.rectangle(img, (410, 450), (480, 500), (30, 30, 200), -1)
    cv2.rectangle(img, (800, 450), (870, 500), (30, 30, 200), -1)
    pw = int(ph * 4.4)
    x0, y0 = 640 - pw // 2, 520
    pil = Image.new("RGB", (pw, ph), (245, 245, 245))
    d = ImageDraw.Draw(pil)
    d.rectangle([1, 1, pw - 2, ph - 2], outline=(10, 10, 10), width=3)
    try:  # the scene was tuned on this font; others render plates the public detector misses
        font = ImageFont.truetype(FONT, int(ph * 0.72))
    except OSError:
        pytest.skip("needs DejaVuSansCondensed-Bold.ttf (apt install fonts-dejavu-extra)")
    d.text((pw * 0.08, ph * 0.1), text, font=font, fill=(10, 10, 10))
    img[y0 : y0 + ph, x0 : x0 + pw] = np.array(pil)[..., ::-1]
    return img


def test_image_recognition_returns_valid_structure(engine):
    res = engine.recognize(synthetic_scene())
    assert res, "plate not detected"
    d = res[0].to_dict()
    assert set(d) >= {"bbox", "plate", "quality", "detection_score"}
    assert 0.0 <= d["plate"]["confidence"] <= 1.0
    if d["plate"]["text"]:
        assert engine.grammar.is_valid(d["plate"]["text"]) is not None or d["plate"]["format"] == "free"


def test_video_session_emits_one_event_per_vehicle(engine):
    sess = engine.session("test", fps=10)
    events = []
    for f in range(12):
        img = synthetic_scene()
        M = np.float32([[1, 0, 6 * f], [0, 1, 2 * f]])
        events += sess.step(cv2.warpAffine(img, M, (1280, 720), borderMode=cv2.BORDER_REPLICATE), f)
    events += sess.finish()
    finals = [e for e in events if not e.provisional]
    assert len(finals) == 1
    assert finals[0].n_frames >= 10
    assert finals[0].direction in ("left_to_right", "stationary", "approaching", "receding")


def test_rest_api(engine):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drishti.server.app import create_app

    with TestClient(create_app(engine=engine)) as c:
        assert c.get("/healthz").json()["status"] == "ok"
        ok, buf = cv2.imencode(".jpg", synthetic_scene())
        r = c.post("/v1/recognize", files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")})
        assert r.status_code == 200 and "plates" in r.json()
        assert c.post("/v1/watchlist", json={"plate": "KA05MX3021", "reason": "test"}).json()["ok"]
        assert c.get("/v1/watchlist").json()["entries"][0]["plate"] == "KA05MX3021"
        assert c.post("/v1/recognize", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
        assert "drishti_requests_total" in c.get("/metrics").text


def test_rest_api_key(engine):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drishti.server.app import create_app

    with TestClient(create_app(engine=engine, api_key="s3cret")) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/watchlist").status_code == 401
        assert c.get("/v1/watchlist", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/watchlist", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/v1/events", headers={"Authorization": "Bearer s3cret"}).status_code == 200
