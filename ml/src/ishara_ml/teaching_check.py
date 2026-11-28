"""Side-by-side check sheet: glyph + slug | real S1 frame (full) | same frame cropped | illustrated poster.

Run from ``ml/`` after make_teaching_media:  ``uv run python -m ishara_ml.teaching_check ka,kha``
Writes ml/reports/teaching_check_<slugs>.png (git-ignored: it contains dataset frames).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ishara_ml import config
from ishara_ml.teaching import ffmpeg

CELL = 480
LABEL_W = 260
GAP = 16


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("Nirmala.ttc", "NirmalaUI.ttf", "Nirmala.ttf", "mangal.ttf", "NotoSansDevanagari-Regular.ttf"):
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/noto")):
            try:
                return ImageFont.truetype(str(base / name), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _frame_at(src: Path, t: float, fps: float) -> Image.Image:
    frames = list(ffmpeg.read_frames(src, fps, start=t, end=t + 1.5 / fps))
    if not frames:
        raise RuntimeError(f"no frame at {t}s in {src}")
    return Image.fromarray(frames[0])


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.copy()
    ratio = min(w / img.width, h / img.height)
    img = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (w, h), (255, 255, 255))
    cell.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return cell


def build(slugs: list[str]) -> Path:
    signs = {s["slug"]: s for s in json.loads(config.SIGNS_JSON.read_text(encoding="utf-8"))["signs"]}
    full_w = int(CELL * 16 / 9)
    width = LABEL_W + full_w + 2 * CELL + 4 * GAP
    head = 50
    sheet = Image.new("RGB", (width, head + len(slugs) * (CELL + GAP)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    small, big = _font(26), _font(150)
    xs = [GAP, LABEL_W + GAP, LABEL_W + full_w + 2 * GAP, LABEL_W + full_w + CELL + 3 * GAP]
    for x, title in zip(
        xs, ["sign", "S1 video, middle frame", "same frame, crop box", "illustrated poster"], strict=True
    ):
        draw.text((x, 12), title, fill=(40, 40, 40), font=small)

    for i, slug in enumerate(slugs):
        meta = json.loads((config.TEACHING_LANDMARKS_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        src = config.NSL23_DIR / meta["source"]
        y = head + i * (CELL + GAP)
        frame = _frame_at(src, meta["posterTimeS"], meta["fps"])
        b = meta["box"]
        crop = frame.crop((round(b["x0"]), round(b["y0"]), round(b["x0"] + b["side"]), round(b["y0"] + b["side"])))
        boxed = frame.copy()
        ImageDraw.Draw(boxed).rectangle(
            [b["x0"], b["y0"], b["x0"] + b["side"], b["y0"] + b["side"]], outline=(255, 0, 0), width=6
        )
        poster = Image.open(config.MEDIA_DIR / slug / "poster.webp").convert("RGB")

        draw.rectangle([GAP, y, LABEL_W, y + CELL], fill=(255, 255, 255))
        draw.text((GAP + 40, y + 60), signs[slug]["glyph"], fill=(20, 20, 20), font=big)
        draw.text((GAP + 40, y + 300), slug, fill=(20, 20, 20), font=_font(48))
        draw.text((GAP + 20, y + 380), Path(meta["source"]).name, fill=(90, 90, 90), font=_font(20))
        sheet.paste(_fit(boxed, full_w, CELL), (xs[1], y))
        sheet.paste(_fit(crop, CELL, CELL), (xs[2], y))
        sheet.paste(_fit(poster, CELL, CELL), (xs[3], y))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"teaching_check_{'_'.join(slugs)}.png"
    sheet.save(out)
    return out


def build_compare(
    slugs: list[str], old_dir: Path, name: str = "teaching_check_v2.png", old_label: str = "v1", new_label: str = "v2"
) -> Path:
    """Per letter: glyph + slug | real hand (crop box) at the held frame | old poster | new poster."""
    signs = {s["slug"]: s for s in json.loads(config.SIGNS_JSON.read_text(encoding="utf-8"))["signs"]}
    head = 50
    xs = [GAP] + [LABEL_W + GAP + i * (CELL + GAP) for i in range(3)]
    width = xs[-1] + CELL + GAP
    sheet = Image.new("RGB", (width, head + len(slugs) * (CELL + GAP)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    small = _font(26)
    titles = ["sign", "real hand, held frame", f"old poster ({old_label})", f"new poster ({new_label})"]
    for x, title in zip(xs, titles, strict=True):
        draw.text((x, 12), title, fill=(40, 40, 40), font=small)
    for i, slug in enumerate(slugs):
        meta = json.loads((config.TEACHING_LANDMARKS_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        y = head + i * (CELL + GAP)
        frame = _frame_at(config.NSL23_DIR / meta["source"], meta["posterTimeS"], meta["fps"])
        b = meta["box"]
        crop = frame.crop((round(b["x0"]), round(b["y0"]), round(b["x0"] + b["side"]), round(b["y0"] + b["side"])))
        old_path = old_dir / f"{slug}.webp"
        old = Image.open(old_path).convert("RGB") if old_path.exists() else Image.new("RGB", (CELL, CELL), "white")
        new = Image.open(config.MEDIA_DIR / slug / "poster.webp").convert("RGB")
        draw.rectangle([GAP, y, LABEL_W, y + CELL], fill=(255, 255, 255))
        draw.text((GAP + 40, y + 60), signs[slug]["glyph"], fill=(20, 20, 20), font=_font(150))
        draw.text((GAP + 40, y + 300), slug, fill=(20, 20, 20), font=_font(48))
        ext = [n for n, v in meta.get("extendedFraction", {}).items() if v >= 0.5]
        draw.text((GAP + 20, y + 380), "extended: " + (", ".join(ext) or "none"), fill=(90, 90, 90), font=_font(18))
        draw.text((GAP + 20, y + 410), f"held frame {meta['posterFrame']}", fill=(90, 90, 90), font=_font(18))
        for x, img in zip(xs[1:], [crop, old, new], strict=True):
            sheet.paste(_fit(img, CELL, CELL), (x, y))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / name
    sheet.save(out)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    # --compare vN [slugs]: real hand | posters saved in reports/v<N-1>_posters/ | current posters
    if args and args[0] in ("--v2", "--compare"):
        version = 2 if args[0] == "--v2" else int(args[1].lstrip("v"))
        rest = args[1:] if args[0] == "--v2" else args[2:]
        slugs = [s for s in (rest[0] if rest else "ka,kha,ba,u").split(",") if s]
        old = f"v{version - 1}"
        print(
            build_compare(
                slugs, config.REPORTS_DIR / f"{old}_posters", f"teaching_check_v{version}.png", old, f"v{version}"
            )
        )
    else:
        slugs = [s for s in (args[0] if args else "ka,kha").split(",") if s]
        print(build(slugs))
