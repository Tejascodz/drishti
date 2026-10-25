"""Ollama integration and the labelling tool, against a local stand-in server that
speaks Ollama's /api/tags and /api/chat."""

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from drishti.formats import FormatGrammar
from drishti.ocr.vlm import VLMVerifier

from .test_grammar import ALPHA


class _FakeOllama(BaseHTTPRequestHandler):
    answer = {"plate": "KA05MX3021", "readable": True}
    reject_think = False
    calls: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "qwen3-vl:2b"}, {"name": "llama3.2:latest"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(body)
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        if type(self).reject_think and "think" in body:
            return self._send(400, {"error": "\"qwen3-vl:2b\" does not support thinking"})
        msg = body["messages"][0]
        assert msg["images"] and isinstance(msg["images"][0], str)
        self._send(200, {"message": {"role": "assistant", "content": json.dumps(type(self).answer)}, "done": True})


@pytest.fixture()
def ollama():
    _FakeOllama.answer = {"plate": "KA05MX3021", "readable": True}
    _FakeOllama.reject_think = False
    _FakeOllama.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


CROP = np.full((40, 160, 3), 230, np.uint8)
G = FormatGrammar.from_pack("india", ALPHA, "_")


def test_check_and_read(ollama):
    v = VLMVerifier(ollama, "qwen3-vl:2b", region="india")
    info = v.check()
    assert info["reachable"] and info["model_present"]
    assert VLMVerifier(ollama, "qwen3-vl:8b").check()["hint"] == "ollama pull qwen3-vl:8b"
    ans = v.read(CROP)
    assert ans.text == "KA05MX3021" and ans.readable
    sent = _FakeOllama.calls[-1]
    assert sent["think"] is False and sent["stream"] is False
    assert sent["format"]["required"] == ["plate", "readable"]
    assert "India" in sent["messages"][0]["content"]


def test_think_fallback(ollama):
    _FakeOllama.reject_think = True
    v = VLMVerifier(ollama, "qwen3-vl:2b")
    assert v.read(CROP).text == "KA05MX3021"
    assert "think" not in _FakeOllama.calls[-1]


def test_verify_is_constrained(ollama):
    v = VLMVerifier(ollama, "qwen3-vl:2b")
    alts = [("KA05MX3027", 0.4), ("KA05MX3021", 0.3)]
    assert v.verify(CROP, "KA05MX3027", alts, G) == "KA05MX3021"  # a candidate: accepted
    _FakeOllama.answer = {"plate": "TN99ZZ9999", "readable": True}
    assert v.verify(CROP, "KA05MX3027", alts, G) is None  # valid but unrelated: rejected
    _FakeOllama.answer = {"plate": "KA05MX302", "readable": True}
    assert v.verify(CROP, "KA05MX3027", alts, G) is None  # invalid format: rejected
    _FakeOllama.answer = {"plate": "", "readable": False}
    assert v.verify(CROP, "KA05MX3027", alts, G) is None


def test_unreachable_never_breaks_recognition():
    v = VLMVerifier("http://127.0.0.1:9", "qwen3-vl:2b", timeout_s=1)
    assert v.check()["reachable"] is False
    assert v.verify(CROP, "KA05MX3021", [], G) is None


def test_labelling_end_to_end(ollama, tmp_path):
    from drishti.config import load_config
    from drishti.label import run_labelling
    from drishti.pipeline import Drishti

    from .test_pipeline import synthetic_scene

    try:
        eng = Drishti(load_config(overrides={"models": {"vehicle_detector": None}}))
    except Exception as exc:
        pytest.skip(f"models unavailable: {exc}")
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    cv2.imwrite(str(root / "images" / "a.jpg"), synthetic_scene("KA05MX3021"))
    cv2.imwrite(str(root / "images" / "b.jpg"), synthetic_scene("MH12DE1433"))
    # b.jpg has a ground-truth box: used instead of the detector
    (root / "labels" / "b.txt").write_text("0 0.5 0.7708 0.2406 0.0972\n")

    v = VLMVerifier(ollama, "qwen3-vl:2b")
    out = tmp_path / "out"
    summary = run_labelling(eng, [str(root / "images")], out, vlm=v)
    assert summary["plates"] == 2
    rows = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines()]
    by = {r["source"].rsplit("/", 1)[-1]: r for r in rows}
    assert by["b.jpg"]["box_from"] == "labels" and by["a.jpg"]["box_from"] == "detector"
    # the fake VLM always says KA05MX3021: agreement only on a.jpg
    assert by["b.jpg"]["status"] == "review"
    if by["a.jpg"]["ocr_text"] == "KA05MX3021":
        assert by["a.jpg"]["status"] == "auto"
    html = (out / "review.html").read_text()
    assert "Download annotations.csv" in html and "crops/" in html
    with open(out / "annotations_auto.csv") as f:
        assert next(csv.reader(f)) == ["image_path", "plate_text"]
    for r in rows:
        assert (out / r["crop"]).exists()
    # resumable: a second run processes nothing new
    again = run_labelling(eng, [str(root / "images")], out, vlm=v)
    assert again["images_processed_now"] == 0 and again["plates"] == 2
