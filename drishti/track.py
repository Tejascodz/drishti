"""Multi-object tracker for plates: ByteTrack-style two-stage association with
buffered IoU (C-BIoU) so small, fast-moving plates at low frame rates still match."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import BBox, Detection

try:  # optional, exact assignment
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) as in SORT / ByteTrack."""

    def __init__(self):
        ndim = 4
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = 1.0
        self.Hm = np.eye(ndim, 2 * ndim)
        self.w_pos = 1.0 / 20
        self.w_vel = 1.0 / 160

    def initiate(self, z: np.ndarray):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-2, self.w_pos * h, self.w_vel * h, self.w_vel * h, 1e-5, self.w_vel * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]))
        S = self.Hm @ cov @ self.Hm.T + R
        K = cov @ self.Hm.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.Hm @ mean)
        cov = (np.eye(len(mean)) - K @ self.Hm) @ cov
        return mean, cov


def _xyah(b: BBox) -> np.ndarray:
    return np.array([b.cx, b.cy, b.w / max(b.h, 1e-6), b.h], dtype=np.float64)


def _bbox(m: np.ndarray) -> BBox:
    cx, cy, a, h = m[:4]
    w = a * h
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    bbox: BBox
    score: float
    start_frame: int
    last_frame: int
    hits: int = 1
    time_since_update: int = 0
    state: str = "tracked"  # tracked | lost | removed
    history: list[tuple[int, BBox]] = field(default_factory=list)

    @property
    def predicted_bbox(self) -> BBox:
        return _bbox(self.mean)


def _biou_cost(tracks: list[BBox], dets: list[BBox], buf: float) -> np.ndarray:
    def expand(b: BBox):
        return [b.x1 - buf * b.w, b.y1 - buf * b.h, b.x2 + buf * b.w, b.y2 + buf * b.h]

    if not tracks or not dets:
        return np.ones((len(tracks), len(dets)))
    a = np.array([expand(b) for b in tracks])
    d = np.array([expand(b) for b in dets])
    ix = np.clip(np.minimum(a[:, None, 2], d[None, :, 2]) - np.maximum(a[:, None, 0], d[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], d[None, :, 3]) - np.maximum(a[:, None, 1], d[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    iou = inter / (aa[:, None] + ad[None, :] - inter + 1e-9)
    # discourage matches whose scale changed implausibly (different vehicles side by side)
    ha = np.array([b.h for b in tracks])[:, None]
    hd = np.array([b.h for b in dets])[None, :]
    ratio = np.minimum(ha, hd) / np.maximum(ha, hd)
    return 1.0 - iou * np.clip(ratio * 1.5, 0, 1)


def _assign(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if _lsa is not None:
        r, c = _lsa(cost)
        pairs = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
    else:
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            i, j = np.unravel_index(flat, cost.shape)
            if cost[i, j] > thresh:
                break
            if i in used_r or j in used_c:
                continue
            pairs.append((int(i), int(j)))
            used_r.add(i)
            used_c.add(j)
    mr = {i for i, _ in pairs}
    mc = {j for _, j in pairs}
    return pairs, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class ByteTracker:
    def __init__(self, high: float = 0.45, low: float = 0.1, new_thresh: float = 0.45, match_thresh: float = 0.85,
                 max_age: int = 30, buffer1: float = 0.3, buffer2: float = 0.5):
        self.high, self.low, self.new_thresh = high, low, new_thresh
        self.match_thresh, self.max_age = match_thresh, max_age
        self.b1, self.b2 = buffer1, buffer2
        self.kf = KalmanXYAH()
        self.tracks: list[Track] = []
        self._removed: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection], frame_idx: int) -> list[tuple[Track, int]]:
        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        high = [i for i, d in enumerate(dets) if d.score >= self.high]
        low = [i for i, d in enumerate(dets) if self.low <= d.score < self.high]
        matches: list[tuple[Track, int]] = []

        pool = [t for t in self.tracks if t.state in ("tracked", "lost")]
        cost = _biou_cost([t.predicted_bbox for t in pool], [dets[i].bbox for i in high], self.b1)
        pairs, un_t, un_d = _assign(cost, self.match_thresh)
        for ti, di in pairs:
            matches.append((pool[ti], high[di]))
        rem_tracks = [pool[i] for i in un_t if pool[i].state == "tracked"]
        rem_high = [high[i] for i in un_d]

        cost2 = _biou_cost([t.predicted_bbox for t in rem_tracks], [dets[i].bbox for i in low], self.b2)
        pairs2, un_t2, _ = _assign(cost2, min(0.95, self.match_thresh + 0.05))
        for ti, di in pairs2:
            matches.append((rem_tracks[ti], low[di]))

        matched_ids = set()
        for t, di in matches:
            d = dets[di]
            t.mean, t.cov = self.kf.update(t.mean, t.cov, _xyah(d.bbox))
            t.bbox, t.score = d.bbox, d.score
            t.hits += 1
            t.time_since_update = 0
            t.state = "tracked"
            t.last_frame = frame_idx
            t.history.append((frame_idx, d.bbox))
            matched_ids.add(t.track_id)

        for t in self.tracks:
            if t.track_id not in matched_ids:
                t.time_since_update += 1
                t.state = "lost"
                if t.time_since_update > self.max_age:
                    t.state = "removed"

        for i in rem_high:
            d = dets[i]
            if d.score < self.new_thresh:
                continue
            mean, cov = self.kf.initiate(_xyah(d.bbox))
            t = Track(self._next_id, mean, cov, d.bbox, d.score, frame_idx, frame_idx, history=[(frame_idx, d.bbox)])
            self._next_id += 1
            self.tracks.append(t)
            matches.append((t, i))

        self._removed.extend(t for t in self.tracks if t.state == "removed")
        self.tracks = [t for t in self.tracks if t.state != "removed"]
        return matches

    def pop_removed(self) -> list[Track]:
        out, self._removed = self._removed, []
        return out

    def flush(self) -> list[Track]:
        """End of stream: every remaining track is finished."""
        out = self.tracks + self.pop_removed()
        self.tracks = []
        return out
