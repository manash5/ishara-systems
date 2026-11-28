"""Key-pose check sheet: per letter the real held frame (crop) | the drawn key pose | 6 frames of the animation.

Run from ``ml/`` after make_teaching_media:  ``uv run python -m ishara_ml.teaching_keypose_check ka,kha,a,ba,u``
Writes ml/reports/keypose_check.png (git-ignored: it contains dataset frames).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

from ishara_ml import config
from ishara_ml.teaching.keypose import Track
from ishara_ml.teaching_check import _font, _frame_at

CELL = 300
STRIP = 6
LABEL_W = 170
GAP = 12


def _held_crop(slug: str, kp: dict, src: Path, fps: float) -> Image.Image:
    a, b = kp["sourceFrames"]
    mid = (a + b) // 2
    track = Track.load(config.KEYPOSE_LANDMARKS_DIR / f"{slug}.npz")
    img = _frame_at(src, kp["sourceTimesS"][0] + (mid - a) / fps, fps)
    pts = track.image[mid]
    if np.isnan(pts).any():
        return img.resize((CELL, CELL))
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    side = float((hi - lo).max()) * 1.6
    cx, cy = (lo + hi) / 2
    crop = img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    return crop.resize((CELL, CELL), Image.Resampling.LANCZOS)


def build(slugs: list[str], out: Path = config.REPORTS_DIR / "keypose_check.png") -> Path:
    signs = {s["slug"]: s for s in json.loads(config.SIGNS_JSON.read_text(encoding="utf-8"))["signs"]}
    cols = 2 + STRIP
    width = LABEL_W + cols * (CELL + GAP) + GAP
    head = 44
    sheet = Image.new("RGB", (width, head + len(slugs) * (CELL + GAP)), (225, 225, 225))
    draw = ImageDraw.Draw(sheet)
    small, big = _font(22), _font(90)
    titles = ["real held frame", "drawn key pose"] + [f"anim {i + 1}/{STRIP}" for i in range(STRIP)]
    for c, t in enumerate(titles):
        draw.text((LABEL_W + c * (CELL + GAP) + GAP, 10), t, fill=(30, 30, 30), font=small)
    for r, slug in enumerate(slugs):
        y = head + r * (CELL + GAP)
        doc = json.loads((config.KEYPOSES_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        src = config.NSL23_DIR / doc["source"]["file"]
        draw.text((GAP, y + 10), signs.get(slug, {}).get("glyph", ""), fill=(0, 0, 0), font=big)
        draw.text(
            (GAP, y + 130), f"{slug}\n{doc['kind']}\n{len(doc['keyPoses'])} key pose(s)", fill=(0, 0, 0), font=small
        )
        media = config.MEDIA_DIR / slug
        frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(media / "animation.webp"))]
        picks = np.linspace(0, len(frames) - 1, STRIP).round().astype(int)
        cells = [_held_crop(slug, doc["keyPoses"][-1], src, doc["source"]["fps"]), Image.open(media / "poster.webp")]
        cells += [frames[i] for i in picks]
        for c, cell in enumerate(cells):
            sheet.paste(cell.convert("RGB").resize((CELL, CELL)), (LABEL_W + c * (CELL + GAP) + GAP, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    slugs = [s.strip() for s in (args[0] if args else "ka,kha,a,ba,u").split(",") if s.strip()]
    print(build(slugs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
