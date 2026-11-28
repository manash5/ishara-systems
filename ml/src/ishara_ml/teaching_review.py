"""Review sheets for all letters (phase 4).

Run from ``ml/`` after make_teaching_media:
    uv run python -m ishara_ml.teaching_review            # all sheets
    uv run python -m ishara_ml.teaching_review --only ka,pa  # grid for a subset (fix rounds)

Writes to ml/reports/ (git-ignored, contains dataset frames):
- review_<i>of<n>.png: every letter in signs.json order; per cell glyph, slug, source/signer/style, the real hand
  (crop box) at the poster's held frame, and the illustrated poster.
- review_motion.png: for letters with movement, the real clip and the animation as frame strips.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from ishara_ml import config
from ishara_ml.teaching import ffmpeg
from ishara_ml.teaching.pipeline import MOVEMENT_SIGNS
from ishara_ml.teaching_check import _fit, _font, _frame_at

MOVEMENT_VOWELS = [s for s in ("aa", "u", "uu", "e", "ai", "o", "au", "am", "ah") if s in MOVEMENT_SIGNS]

IMG = 300
GAP = 14
HEAD = 78
COLS = 3
BG = (232, 232, 232)


def _signs() -> list[dict]:
    return json.loads(config.SIGNS_JSON.read_text(encoding="utf-8"))["signs"]


def _meta(slug: str) -> dict | None:
    path = config.TEACHING_LANDMARKS_DIR / f"{slug}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _crop(frame: Image.Image, box: dict) -> Image.Image:
    """The crop box from the real frame; parts of the box outside the frame are white (boxes are not clamped)."""
    x0, y0, side = round(box["x0"]), round(box["y0"]), round(box["side"])
    out = Image.new("RGB", (side, side), (255, 255, 255))
    out.paste(frame, (-x0, -y0))
    return out


def _cell(sign: dict) -> Image.Image:
    w = 2 * IMG + GAP
    cell = Image.new("RGB", (w, HEAD + IMG), (255, 255, 255))
    d = ImageDraw.Draw(cell)
    d.text((10, 0), sign["glyph"], fill=(20, 20, 20), font=_font(54))
    d.text((120, 8), f"{sign['sortOrder']}. {sign['slug']}", fill=(20, 20, 20), font=_font(30))
    meta = _meta(sign["slug"])
    poster = config.MEDIA_DIR / sign["slug"] / "poster.webp"
    if meta is None or not poster.exists():
        d.text((120, 46), "NOT GENERATED", fill=(200, 0, 0), font=_font(22))
        return cell
    info = f"{Path(meta['source']).name} · {meta.get('style', 'illustrated')}"
    d.text((120, 46), info, fill=(100, 100, 100), font=_font(20))
    frame = _frame_at(config.NSL23_DIR / meta["source"], meta["posterTimeS"], meta["fps"])
    cell.paste(_fit(_crop(frame, meta["box"]), IMG, IMG), (0, HEAD))
    cell.paste(_fit(Image.open(poster).convert("RGB"), IMG, IMG), (IMG + GAP, HEAD))
    return cell


def build_grid(signs: list[dict], pages: int, prefix: str = "review") -> list[Path]:
    per_page = math.ceil(len(signs) / pages)
    out = []
    cell_w, cell_h = 2 * IMG + GAP, HEAD + IMG
    for page in range(pages):
        chunk = signs[page * per_page : (page + 1) * per_page]
        if not chunk:
            break
        rows = math.ceil(len(chunk) / COLS)
        sheet = Image.new("RGB", (COLS * (cell_w + GAP) + GAP, rows * (cell_h + GAP) + GAP + 40), BG)
        ImageDraw.Draw(sheet).text(
            (GAP, 8),
            f"Ishara teaching media review {page + 1}/{pages} - per cell: real hand at the held frame | poster",
            fill=(40, 40, 40),
            font=_font(24),
        )
        for i, sign in enumerate(chunk):
            r, c = divmod(i, COLS)
            sheet.paste(_cell(sign), (GAP + c * (cell_w + GAP), 40 + GAP + r * (cell_h + GAP)))
        path = config.REPORTS_DIR / f"{prefix}_{page + 1}of{pages}.png"
        sheet.save(path)
        out.append(path)
    return out


def _strip_frames(slug: str, n: int) -> tuple[list[Image.Image], list[Image.Image]] | None:
    meta = _meta(slug)
    anim_path = config.MEDIA_DIR / slug / "animation.webp"
    if meta is None or not anim_path.exists():
        return None
    total = len(meta["landmarks"])
    idx = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    frames = list(ffmpeg.read_frames(config.NSL23_DIR / meta["source"], meta["fps"], meta["start"], meta["end"]))
    real = [_crop(Image.fromarray(frames[meta["firstFrame"] + k]), meta["box"]) for k in idx]
    anim = Image.open(anim_path)
    drawn = []
    for k in idx:
        anim.seek(k)
        drawn.append(anim.convert("RGB"))
    return real, drawn


def build_motion(slugs: list[str], n: int = 7, name: str = "review_motion.png") -> Path:
    signs = {s["slug"]: s for s in _signs()}
    size, label = 190, 170
    width = label + n * (size + 6) + GAP
    row_h = 2 * size + 6 + GAP
    sheet = Image.new("RGB", (width, 40 + len(slugs) * row_h), BG)
    d = ImageDraw.Draw(sheet)
    d.text(
        (GAP, 8),
        f"Movement check - {n} evenly spaced frames: real clip (top) / animation (bottom)",
        fill=(40, 40, 40),
        font=_font(22),
    )
    for r, slug in enumerate(slugs):
        y = 40 + r * row_h
        d.rectangle([0, y, label - 8, y + 2 * size + 6], fill=(255, 255, 255))
        d.text((12, y + 20), signs[slug]["glyph"], fill=(20, 20, 20), font=_font(80))
        d.text((12, y + 150), slug, fill=(20, 20, 20), font=_font(30))
        strips = _strip_frames(slug, n)
        if strips is None:
            d.text((label, y + 20), "NOT GENERATED", fill=(200, 0, 0), font=_font(26))
            continue
        for k, (real, drawn) in enumerate(zip(*strips, strict=True)):
            x = label + k * (size + 6)
            sheet.paste(_fit(real, size, size), (x, y))
            sheet.paste(_fit(drawn, size, size), (x, y + size + 6))
    path = config.REPORTS_DIR / name
    sheet.save(path)
    return path


def build_side_by_side(slug: str, size: int = 360) -> Path | None:
    """Animated webp: the real clip (same crop, same frames, same timing) next to the animation.

    Written to ml/reports/compare/<slug>.webp, so motion and hand shape can be judged by watching both together.
    """
    meta = _meta(slug)
    anim_path = config.MEDIA_DIR / slug / "animation.webp"
    if meta is None or not anim_path.exists():
        return None
    frames = list(ffmpeg.read_frames(config.NSL23_DIR / meta["source"], meta["fps"], meta["start"], meta["end"]))
    anim = Image.open(anim_path)
    out: list[Image.Image] = []
    for k in range(len(meta["landmarks"])):
        pair = Image.new("RGB", (2 * size + 8, size), (255, 255, 255))
        real = _crop(Image.fromarray(frames[meta["firstFrame"] + k]), meta["box"])
        pair.paste(real.resize((size, size), Image.Resampling.LANCZOS), (0, 0))
        anim.seek(k)
        pair.paste(anim.convert("RGB").resize((size, size)), (size + 8, 0))
        out.append(pair)
    path = config.REPORTS_DIR / "compare" / f"{slug}.webp"
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = meta.get("durationsMs") or [round(1000 / meta["playbackFps"])] * len(out)
    out[0].save(path, save_all=True, append_images=out[1:], loop=0, duration=durations, quality=70, method=4)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated slugs: one grid of just these (file review_fix_*.png)")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--side-by-side", action="store_true", help="also write reports/compare/<slug>.webp")
    args = parser.parse_args(argv)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    signs = _signs()
    if args.side_by_side:
        wanted = [s.strip() for s in args.only.split(",")] if args.only else [s["slug"] for s in signs]
        for slug in wanted:
            print(build_side_by_side(slug))
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        subset = [s for s in signs if s["slug"] in wanted]
        for p in build_grid(subset, 1, prefix="review_fix"):
            print(p)
        moving = [s for s in wanted if s in MOVEMENT_VOWELS]
        if moving:
            print(build_motion(moving, name="review_fix_motion.png"))
        return 0
    for p in build_grid(signs, args.pages):
        print(p)
    print(build_motion(MOVEMENT_VOWELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
