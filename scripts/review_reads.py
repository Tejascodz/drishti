"""Make a page to check plate reads by eye: each photo annotated, and every plate crop
enlarged next to the text Drishti read from it.

    python scripts/review_reads.py samples my_photos --config models/drishti.yaml --out review
    open review/index.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishti.config import load_config  # noqa: E402
from drishti.pipeline import Drishti, iter_images  # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plate reads</title>
<style>
:root{{--bg:#f6f5f2;--panel:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e3e0da;--good:#1e7a46;--avg:#a5640c;--poor:#b3261e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--panel:#1f1f1d;--ink:#ecebe7;--muted:#a19e97;--line:#34332f;--good:#5cc38a;--avg:#e0a54a;--poor:#f08079}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:16px}}
h1{{font-size:20px}} h2{{font-size:16px;margin:0 0 10px}}
.photo{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:18px}}
.photo>img{{width:100%;max-height:420px;object-fit:contain;background:#000;border-radius:8px}}
.plates{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-top:12px}}
.plate{{border:1px solid var(--line);border-radius:10px;padding:10px}}
.plate img{{width:100%;height:90px;object-fit:contain;background:#000;border-radius:6px;image-rendering:pixelated}}
.text{{font:700 22px ui-monospace,Menlo,monospace;letter-spacing:.05em;margin-top:6px}}
.meta{{color:var(--muted);font-size:13px}}
.Good{{color:var(--good)}}.Average{{color:var(--avg)}}.Poor{{color:var(--poor)}}
</style></head><body><main>
<h1>Plate reads</h1>
<p class="meta">{summary}. Compare each enlarged crop with the text under it.</p>
{body}
</main></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--config", action="append", default=None, help="e.g. models/drishti.yaml")
    ap.add_argument("--out", default="review")
    ap.add_argument("--providers", default="cpu")
    a = ap.parse_args()

    cfg = load_config(a.config, {"runtime": {"providers": a.providers}, "models": {"vehicle_detector": None}})
    eng = Drishti(cfg)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    blocks, n_img, n_plates = [], 0, 0
    for path in iter_images(a.inputs):
        img = cv2.imread(str(path))
        if img is None:
            continue
        n_img += 1
        res = eng.recognize(img)
        stem = f"{n_img:03d}_{path.stem}"
        cv2.imwrite(str(out / "img" / f"{stem}.jpg"), eng.annotate(img, res))
        cards = []
        for k, r in enumerate(res, 1):
            n_plates += 1
            crop_name = f"{stem}_plate{k}.png"
            if r.crop is not None and r.crop.size:
                cv2.imwrite(str(out / "img" / crop_name), r.crop)
            text = r.read.text or "unreadable"
            alts = ", ".join(t for t, _ in r.read.alternatives[1:4] if t != r.read.text)
            cards.append(
                f'<div class="plate"><img src="img/{crop_name}" alt="plate {k} crop">'
                f'<div class="text">{html.escape(text)}</div>'
                f'<div class="meta"><span class="{r.read.level}">{r.read.confidence:.2f} {r.read.level}</span>'
                f' · {int(round(r.bbox.w))} px wide{" · rescued" if r.read.rescued else ""}</div>'
                + (f'<div class="meta">other guesses: {html.escape(alts)}</div>' if alts else "")
                + "</div>"
            )
        plates = f'<div class="plates">{"".join(cards)}</div>' if cards else '<p class="meta">no plate found</p>'
        blocks.append(f'<section class="photo"><h2>{html.escape(path.name)}</h2>'
                      f'<img src="img/{stem}.jpg" alt="{html.escape(path.name)} annotated">{plates}</section>')
        print(f"{path.name}: {', '.join((r.read.text or '?') for r in res) or 'no plate'}", flush=True)
    (out / "index.html").write_text(PAGE.format(summary=f"{n_img} photos, {n_plates} plates", body="\n".join(blocks)),
                                    encoding="utf-8")
    print(f"open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
