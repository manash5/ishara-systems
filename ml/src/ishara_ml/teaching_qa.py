"""QA images for the cut-out styles (git-ignored ml/reports/, they contain dataset frames).

Run from ``ml/`` after make_teaching_media (with ``--also photo`` for the comparison):
    uv run python -m ishara_ml.teaching_qa ka,kha,a,ba,u

- reports/style_compare.png: per letter the real crop at the held frame, then photo and toon posters.
- reports/qa/<slug>.png: 4 moments (start, middle, held, end): the real crop with the landmarks drawn on it next to
  the toon frame (and the photo frame when available).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ishara_ml import config
from ishara_ml.teaching import ffmpeg
from ishara_ml.teaching.cutout import crop_square, to_crop
from ishara_ml.teaching.geometry import FINGERS
from ishara_ml.teaching.pipeline import load_map
from ishara_ml.teaching_check import _font

T = 360
FINGER_COLOURS = {"thumb": "red", "index": "blue", "middle": "green", "ring": "orange", "pinky": "magenta"}


def _style_dir(slug: str, style: str) -> Path:
    main = {r["slug"]: (r.get("style") or "toon") for r in load_map()}.get(slug, "toon")
    return (config.MEDIA_DIR if main == style else config.REPORTS_DIR / "styles" / style) / slug


def _meta(slug: str) -> dict:
    return json.loads((config.TEACHING_LANDMARKS_DIR / f"{slug}.json").read_text(encoding="utf-8"))


def _real_crops(meta: dict, indices: list[int]) -> list[Image.Image]:
    frames = list(ffmpeg.read_frames(config.NSL23_DIR / meta["source"], meta["fps"], meta["start"], meta["end"]))
    b = meta["box"]
    return [
        Image.fromarray(crop_square(frames[meta["firstFrame"] + k], b["x0"], b["y0"], b["side"], T)) for k in indices
    ]


def _with_landmarks(img: Image.Image, meta: dict, k: int) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    b = meta["box"]
    p = to_crop(np.array(meta["landmarks"][k]), b["x0"], b["y0"], b["side"], T)
    for name, chain in FINGERS.items():
        pts = [tuple(p[i]) for i in [0, *chain]]
        d.line(pts, fill=FINGER_COLOURS[name], width=2)
        for x, y in pts[1:]:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=FINGER_COLOURS[name])
    return img


def _anim_frame(path: Path, k: int) -> Image.Image | None:
    if not path.exists():
        return None
    im = Image.open(path)
    im.seek(min(k, im.n_frames - 1))
    return im.convert("RGB")


def style_compare(slugs: list[str], name: str = "style_compare.png") -> Path:
    signs = {s["slug"]: s for s in json.loads(config.SIGNS_JSON.read_text(encoding="utf-8"))["signs"]}
    cols = ["real (held frame)", "photo", "toon"]
    label_w, gap, head = 170, 12, 44
    sheet = Image.new("RGB", (label_w + 3 * (T + gap), head + len(slugs) * (T + gap)), (232, 232, 232))
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((label_w + j * (T + gap), 10), c, fill=(30, 30, 30), font=_font(24))
    for i, slug in enumerate(slugs):
        meta = _meta(slug)
        y = head + i * (T + gap)
        d.rectangle([0, y, label_w - 10, y + T], fill="white")
        d.text((14, y + 30), signs[slug]["glyph"], fill="black", font=_font(90))
        d.text((14, y + 170), slug, fill="black", font=_font(36))
        d.text((14, y + 220), Path(meta["source"]).name, fill=(90, 90, 90), font=_font(16))
        real = _real_crops(meta, [meta["posterFrame"]])[0]
        cells = [real]
        for st in ("photo", "toon"):
            p = _style_dir(slug, st) / "poster.webp"
            cells.append(Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (T, T), "white"))
        for j, im in enumerate(cells):
            sheet.paste(im.resize((T, T)), (label_w + j * (T + gap), y))
    out = config.REPORTS_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def qa_sheet(slug: str) -> Path:
    meta = _meta(slug)
    n = len(meta["landmarks"])
    moments = {"start": 0, "middle": n // 2, "held": meta["posterFrame"], "end": n - 1}
    reals = _real_crops(meta, list(moments.values()))
    toon = _style_dir(slug, "toon") / "animation.webp"
    photo = _style_dir(slug, "photo") / "animation.webp"
    rows = [("real + landmarks", None), ("toon", toon)] + ([("photo", photo)] if photo.exists() else [])
    gap, head, label_w = 10, 40, 150
    sheet = Image.new("RGB", (label_w + 4 * (T + gap), head + len(rows) * (T + gap)), (232, 232, 232))
    d = ImageDraw.Draw(sheet)
    d.text(
        (10, 8),
        f"{slug}: {Path(meta['source']).name}, {meta['kind']}, {n} frames, held frame {meta['posterFrame']}",
        fill=(30, 30, 30),
        font=_font(22),
    )
    for j, (label, k) in enumerate(moments.items()):
        d.text((label_w + j * (T + gap), head - 2), f"{label} (frame {k})", fill=(60, 60, 60), font=_font(16))
    for i, (label, path) in enumerate(rows):
        y = head + 18 + i * (T + gap)
        d.text((10, y + T // 2), label, fill=(30, 30, 30), font=_font(20))
        for j, k in enumerate(moments.values()):
            im = _with_landmarks(reals[j], meta, k) if path is None else _anim_frame(path, k)
            if im is not None:
                sheet.paste(im.resize((T, T)), (label_w + j * (T + gap), y))
    out = config.REPORTS_DIR / "qa" / f"{slug}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = sheet.crop((0, 0, sheet.width, head + 18 + len(rows) * (T + gap)))
    sheet.save(out)
    return out


if __name__ == "__main__":
    slugs = [s for s in (sys.argv[1] if len(sys.argv) > 1 else "ka,kha,a,ba,u").split(",") if s]
    print(style_compare(slugs))
    for s in slugs:
        print(qa_sheet(s))
