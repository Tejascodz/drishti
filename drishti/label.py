"""Semi-automatic plate-text labelling.

For every plate in a folder of images (boxes from YOLO label files when present,
otherwise from the plate detector) two independent readers are run:

  1. Drishti's OCR + format grammar
  2. a local vision-language model (Ollama), asked without seeing Drishti's answer

Plates where both agree on a valid registration mark are accepted automatically.
The rest go to `review.html`, an offline page where each crop is confirmed or
corrected with the keyboard. The result is a fast-plate-ocr dataset
(crops/ + annotations.csv) ready for OCR fine-tuning.

    drishti label /data/indian_plates --out data/ocr_real
    open data/ocr_real/review.html     # fix the disagreements, download annotations.csv
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import cv2

from .geometry import crop_box, warp_quad
from .pipeline import Drishti, iter_images
from .types import BBox, Detection

log = logging.getLogger(__name__)


def _yolo_boxes(img_path: Path, labels_dir: Path | None, W: int, H: int) -> list[BBox] | None:
    """Ground-truth boxes from a YOLO label file next to the image, if one exists."""
    cands = []
    if labels_dir is not None:
        cands.append(labels_dir / (img_path.stem + ".txt"))
    parts = list(img_path.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        cands.append(Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt"))
    cands.append(img_path.parent.parent / "labels" / (img_path.stem + ".txt"))
    for lp in cands:
        if lp.exists():
            boxes = []
            for row in lp.read_text().splitlines():
                v = row.split()
                if len(v) >= 5:
                    cx, cy, w, h = (float(x) for x in v[1:5])
                    boxes.append(BBox((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H).clip(W, H))
            return boxes
    return None


def _load_done(path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows, {r["source"] for r in rows}


def run_labelling(engine: Drishti, inputs: list[str], out_dir: str | Path, vlm=None, labels_dir: str | None = None,
                  trust_ocr: float = 0.985, min_height: float = 12.0, limit: int | None = None,
                  progress_every: int = 25, crops: bool = False) -> dict:
    """crops=True: every image is already a cropped plate (the whole image is the box)."""
    out = Path(out_dir)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out / "labels.jsonl"
    rows, done = _load_done(jsonl)
    ldir = Path(labels_dir) if labels_dir else None
    fx, fy = engine.views[min(1, len(engine.views) - 1)]
    t0, n_img, vlm_errors, vlm_ok_calls = time.time(), 0, 0, 0

    with open(jsonl, "a", encoding="utf-8") as fh:
        for img_path in iter_images(inputs):
            key = str(img_path.resolve())
            if key in done:
                continue
            if limit is not None and n_img >= limit:
                break
            frame = cv2.imread(str(img_path))
            n_img += 1
            if frame is None:
                continue
            H, W = frame.shape[:2]
            gt = [BBox(0, 0, W, H)] if crops else _yolo_boxes(img_path, ldir, W, H)
            dets = [Detection(b, 1.0) for b in gt] if gt is not None else engine.plate_det(frame)
            reads = engine._read_many(frame, dets)
            for k, (det, r) in enumerate(zip(dets, reads)):
                row = {"source": key, "index": k, "bbox": det.bbox.to_list(),
                       "box_from": "crop" if crops else ("labels" if gt is not None else "detector")}
                if r is None or det.bbox.h < min_height:
                    row.update(status="skipped", reason="too small")
                    fh.write(json.dumps(row) + "\n")
                    continue
                crop = None
                if r.quad is not None:
                    try:
                        crop = warp_quad(frame, r.quad, fx, fy)
                    except ValueError:
                        crop = None
                if crop is None:
                    crop = frame if crops else crop_box(frame, det.bbox, fx, fy)
                name = f"{img_path.stem}_{k}.jpg"
                cv2.imwrite(str(crops_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                ocr_text = r.read.text or ""
                vlm_text, vlm_ok = None, False
                if vlm is not None:
                    try:
                        ans = vlm.read(crop)
                        vlm_text, vlm_ok = ans.text, ans.readable
                        vlm_ok_calls += 1
                    except Exception as exc:
                        vlm_errors += 1
                        log.warning("VLM read failed for %s: %s", name, exc)
                        if vlm_ok_calls == 0 and vlm_errors >= 3:  # never worked: stop instead of mislabelling everything
                            raise RuntimeError(f"VLM not usable ({exc}); check with `drishti vlm-check`") from exc
                ocr_valid = bool(ocr_text) and engine.grammar.is_valid(ocr_text) is not None
                vlm_valid = bool(vlm_text) and engine.grammar.is_valid(vlm_text) is not None
                if vlm is not None and vlm_text and vlm_text == ocr_text and ocr_valid:
                    status = "auto"
                elif vlm is None and ocr_valid and r.read.confidence >= trust_ocr:
                    status = "auto"
                else:
                    status = "review"
                suggestion = ocr_text if ocr_valid else (vlm_text if vlm_valid else (ocr_text or vlm_text or ""))
                row.update(
                    crop=f"crops/{name}", status=status, label=suggestion if status == "auto" else None,
                    suggestion=suggestion or "", ocr_text=ocr_text, ocr_conf=round(float(r.read.confidence), 4),
                    ocr_valid=ocr_valid, vlm_text=vlm_text, vlm_readable=vlm_ok, vlm_valid=vlm_valid,
                    alternatives=[t for t, _ in r.read.alternatives][:5], two_line=r.two_line, height=round(det.bbox.h, 1),
                )
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            if progress_every and n_img % progress_every == 0:
                rate = n_img / max(time.time() - t0, 1e-6)
                print(f"  {n_img} images  ({rate:.2f} img/s)", flush=True)

    plates = [r for r in rows if r.get("status") in ("auto", "review")]
    auto = [r for r in plates if r["status"] == "auto"]
    with open(out / "annotations_auto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        for r in auto:
            w.writerow([r["crop"], r["label"]])
    write_review_page(out, plates, title=out.name)
    agree = sum(1 for r in plates if r.get("vlm_text") and r["vlm_text"] == r["ocr_text"])
    return {
        "images_processed_now": n_img,
        "plates": len(plates),
        "auto_accepted": len(auto),
        "needs_review": len(plates) - len(auto),
        "ocr_vlm_agreement": round(agree / len(plates), 3) if plates else None,
        "skipped_small": sum(1 for r in rows if r.get("status") == "skipped"),
        "vlm_errors": vlm_errors,
        "review_page": str(out / "review.html"),
        "auto_csv": str(out / "annotations_auto.csv"),
    }


def write_review_page(out: Path, plates: list[dict], title: str = "plates") -> Path:
    data = json.dumps([
        {k: r.get(k) for k in ("crop", "status", "label", "suggestion", "ocr_text", "ocr_conf", "vlm_text", "alternatives", "two_line", "height")}
        for r in plates
    ]).replace("</", "<\\/")
    html = _REVIEW_HTML.replace("__DATA__", data).replace("__TITLE__", title.replace("<", ""))
    path = out / "review.html"
    path.write_text(html, encoding="utf-8")
    return path


_REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plate review · __TITLE__</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e3e0da;--accent:#1f5fbf;--ok:#1e7a46;--warn:#a5640c;--bad:#b3261e;--chip:#efede8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--panel:#1f1f1d;--ink:#ecebe7;--muted:#a19e97;--line:#34332f;--accent:#7aa7f0;--ok:#5cc38a;--warn:#e0a54a;--bad:#f08079;--chip:#2a2a27}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
.bar{max-width:980px;margin:0 auto;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
h1{font-size:17px;margin:0 12px 0 0;font-weight:650}
.tabs button,.act{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 10px;font:inherit;cursor:pointer}
.tabs button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
.act.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.stats{color:var(--muted);font-size:13px;margin-left:auto}
main{max-width:980px;margin:0 auto;padding:16px}
.help{color:var(--muted);font-size:13px;margin:0 0 14px}
kbd{background:var(--chip);border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
.card{display:grid;grid-template-columns:minmax(0,300px) 1fr;gap:16px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.card.focus{outline:2px solid var(--accent)}
.card.done{opacity:.6}
.imgwrap{background:#000;border-radius:8px;display:flex;align-items:center;justify-content:center;min-height:90px;overflow:hidden}
.imgwrap img{width:100%;height:120px;object-fit:contain}
.meta{font-size:12px;color:var(--muted);margin-bottom:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.chip{border:1px solid var(--line);background:var(--chip);border-radius:999px;padding:3px 10px;font:600 13px ui-monospace,Menlo,monospace;cursor:pointer;color:var(--ink)}
.chip small{font-weight:400;color:var(--muted)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
input.txt{font:600 20px ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);width:min(100%,260px)}
.fmt{font-size:12px}
.fmt.ok{color:var(--ok)}.fmt.warn{color:var(--warn)}
.state{font-size:12px;font-weight:600}
.state.accepted{color:var(--ok)}.state.rejected{color:var(--bad)}
@media (max-width:640px){.card{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>Plate review</h1>
  <div class="tabs" role="group" aria-label="Filter">
    <button data-f="review" aria-pressed="true">Needs review</button>
    <button data-f="auto" aria-pressed="false">Auto-accepted</button>
    <button data-f="all" aria-pressed="false">All</button>
  </div>
  <button class="act primary" id="dl">Download annotations.csv</button>
  <span class="stats" id="stats"></span>
</div></header>
<main>
<p class="help">Type or click a suggestion, then <kbd>Enter</kbd> to accept and move on. <kbd>Esc</kbd> marks a plate unreadable (left out of training).
<kbd>Tab</kbd>/<kbd>Shift</kbd>+<kbd>Tab</kbd> move between plates. Progress is kept in this browser. Save the CSV into the same folder as this page.</p>
<div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const KEY = "drishti-review:" + location.pathname;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const FMT = [/^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$/, /^\d{2}BH\d{4}[A-Z]{1,2}$/, /^[A-Z]{2}VA[A-Z]{2}\d{4}$/, /^\d{2}[A-Z]\d{6}[A-Z]$/, /^\d{1,3}(CD|CC|UN)\d{1,4}$/];
const clean = s => (s || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
const valid = s => FMT.some(r => r.test(s));
let filter = "review";

function decision(i) {
  const d = DATA[i], s = state[i];
  if (s) return s;
  if (d.status === "auto") return { label: d.label, accepted: true, auto: true };
  return null;
}
function counts() {
  let acc = 0, rej = 0, todo = 0;
  DATA.forEach((d, i) => { const s = decision(i); if (!s) todo++; else if (s.accepted) acc++; else rej++; });
  document.getElementById("stats").textContent = `${DATA.length} plates · ${acc} accepted · ${rej} unreadable · ${todo} to review`;
}
function chip(text, note, input) {
  const b = document.createElement("button");
  b.className = "chip"; b.type = "button";
  b.textContent = text + " ";
  if (note) { const sm = document.createElement("small"); sm.textContent = note; b.appendChild(sm); }
  b.onclick = () => { input.value = text; input.dispatchEvent(new Event("input")); input.focus(); };
  return b;
}
function render() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  DATA.forEach((d, i) => {
    if (filter === "review" && d.status !== "review") return;
    if (filter === "auto" && d.status !== "auto") return;
    const s = decision(i);
    const card = document.createElement("div");
    card.className = "card" + (s ? " done" : ""); card.dataset.i = i;
    const left = document.createElement("div");
    const wrap = document.createElement("div"); wrap.className = "imgwrap";
    const img = document.createElement("img"); img.loading = "lazy"; img.alt = "plate crop"; img.src = d.crop;
    wrap.appendChild(img); left.appendChild(wrap);
    const right = document.createElement("div");
    const meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = `${d.crop} · ${d.height}px high${d.two_line ? " · two-line" : ""}`;
    const input = document.createElement("input");
    input.className = "txt"; input.spellcheck = false; input.autocomplete = "off";
    input.value = (s && s.label) || d.suggestion || "";
    const fmt = document.createElement("span"); fmt.className = "fmt";
    const st = document.createElement("span"); st.className = "state";
    const upd = () => {
      const v = clean(input.value);
      fmt.textContent = v ? (valid(v) ? "valid format" : "unusual format") : "";
      fmt.className = "fmt " + (valid(v) ? "ok" : "warn");
      const cur = decision(i);
      st.textContent = cur ? (cur.accepted ? (cur.auto ? "auto-accepted" : "accepted") : "unreadable") : "";
      st.className = "state " + (cur ? (cur.accepted ? "accepted" : "rejected") : "");
    };
    input.addEventListener("input", upd);
    input.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        const v = clean(input.value); if (!v) return;
        state[i] = { label: v, accepted: true }; save(); card.classList.add("done"); upd(); counts(); next(card);
      } else if (e.key === "Escape") {
        state[i] = { label: null, accepted: false }; save(); card.classList.add("done"); upd(); counts(); next(card);
      }
    });
    input.addEventListener("focus", () => card.classList.add("focus"));
    input.addEventListener("blur", () => card.classList.remove("focus"));
    const chips = document.createElement("div"); chips.className = "chips";
    const seen = new Set();
    const add = (t, note) => { if (t && !seen.has(t)) { seen.add(t); chips.appendChild(chip(t, note, input)); } };
    add(d.ocr_text, `OCR ${Number(d.ocr_conf || 0).toFixed(2)}`);
    add(d.vlm_text, "VLM");
    (d.alternatives || []).forEach(t => add(t, ""));
    const row = document.createElement("div"); row.className = "row";
    row.append(input, fmt, st);
    right.append(meta, chips, row);
    card.append(left, right);
    list.appendChild(card);
    upd();
  });
  counts();
  const first = list.querySelector(".card:not(.done) input") || list.querySelector("input");
  if (first) first.focus({ preventScroll: false });
}
function next(card) {
  let n = card.nextElementSibling;
  while (n && n.classList.contains("done")) n = n.nextElementSibling;
  const inp = (n || card.nextElementSibling)?.querySelector("input");
  if (inp) { inp.focus(); inp.scrollIntoView({ block: "center", behavior: "smooth" }); }
}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  filter = b.dataset.f;
  document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-pressed", x === b));
  render();
});
document.getElementById("dl").onclick = () => {
  const lines = ["image_path,plate_text"];
  DATA.forEach((d, i) => { const s = decision(i); if (s && s.accepted && s.label) lines.push(`${d.crop},${s.label}`); });
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "annotations.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
render();
</script>
</body>
</html>
"""
