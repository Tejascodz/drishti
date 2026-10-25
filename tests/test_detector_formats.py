"""Detector output decoding, tested with tiny constant-output ONNX graphs so no
weights are needed: YOLO26 end-to-end pose (corners), YOLO raw head + NMS."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from drishti.detect.plate import YoloDetector  # noqa: E402
from drishti.geometry import corners_from_keypoints, warp_quad  # noqa: E402


def _const_model(path, out: np.ndarray, in_hw=(640, 640)):
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, *in_hw])
    outp = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))
    # output = const + 0 * sum(images): keeps the input in the graph
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["c", "z"], ["output0"]),
    ]
    inits = [numpy_helper.from_array(out.astype(np.float32), "c"), numpy_helper.from_array(np.array(0, np.float32), "zero")]
    g = helper.make_graph(nodes, "g", [inp], [outp], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, str(path))


def test_yolo26_e2e_pose_decoding(tmp_path):
    # frame 1280x640 letterboxed into 640x640: scale 0.5, pad_y 160
    rows = np.zeros((1, 300, 6 + 12), np.float32)
    # plate at source (400, 300)-(600, 350) -> letterbox coords (200, 310)-(300, 335)
    rows[0, 0, :6] = [200, 310, 300, 335, 0.92, 0]
    corners_src = np.array([[402, 302], [598, 298], [600, 350], [400, 352]], np.float32)
    lb = corners_src * 0.5 + np.array([0, 160])
    rows[0, 0, 6:] = np.concatenate([lb, np.ones((4, 1))], axis=1).ravel()
    rows[0, 1, :6] = [10, 170, 20, 175, 0.05, 0]  # below threshold
    p = tmp_path / "pose.onnx"
    _const_model(p, rows)
    det = YoloDetector(str(p), {"providers": "cpu"}, conf=0.3, tiling="off")
    assert det.fmt == "yolo-e2e"
    img = np.zeros((640, 1280, 3), np.uint8)
    out = det(img)
    assert len(out) == 1
    d = out[0]
    np.testing.assert_allclose(d.bbox.as_array(), [400, 300, 600, 350], atol=1.0)
    np.testing.assert_allclose(d.keypoints[:, :2], corners_src, atol=1.0)
    quad = corners_from_keypoints(d.keypoints, d.bbox)
    assert quad is not None
    img[290:360, 390:610] = 200
    warped = warp_quad(img, quad, 0.05, 0.1)
    assert warped.shape[0] >= 48 and warped.shape[1] > warped.shape[0]


def test_yolo_raw_head_with_nms(tmp_path):
    A = 50
    raw = np.zeros((1, 5, A), np.float32)  # cx, cy, w, h, score (1 class, no kpts)
    raw[0, :, 0] = [320, 320, 100, 30, 0.9]
    raw[0, :, 1] = [322, 321, 98, 31, 0.85]  # duplicate -> suppressed
    raw[0, :, 2] = [100, 500, 80, 20, 0.6]
    p = tmp_path / "raw.onnx"
    _const_model(p, raw)
    det = YoloDetector(str(p), {"providers": "cpu"}, conf=0.3, tiling="off")
    assert det.fmt == "yolo-raw"
    out = det(np.zeros((640, 640, 3), np.uint8))
    assert len(out) == 2
    assert out[0].score == pytest.approx(0.9, abs=1e-5)


def test_tiling_merges_duplicates(tmp_path):
    rows = np.zeros((1, 300, 6), np.float32)
    rows[0, 0] = [300, 300, 340, 320, 0.9, 0]
    p = tmp_path / "e2e.onnx"
    _const_model(p, rows)
    det = YoloDetector(str(p), {"providers": "cpu"}, conf=0.3, tiling="on")
    out = det(np.zeros((2160, 3840, 3), np.uint8))
    # every tile reports a box at its own (300,300); NMS keeps distinct ones only
    assert len(out) >= 1
    assert all(d.score == pytest.approx(0.9, abs=1e-5) for d in out)
