"""Movement check sheet, per letter with movement (two rows):

  row 1: sharpest real frame (crop) | handshape key pose | 6 frames of the animation's movement phase
  row 2: (sharpness / scale info)   |                    | the real video's frames at the same moments

Run from ``ml/`` after make_teaching_media:  ``uv run python -m ishara_ml.teaching_movement_check u,o,am``
Writes ml/reports/movement_check.png (git-ignored: it contains dataset frames).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

from ishara_ml import config
from ishara_ml.teaching import ffmpeg, timeline
from ishara_ml.teaching.keypose import KeyPoseDoc, Track
from ishara_ml.teaching.keypose_pipeline import doc_path, load_skeleton
from ishara_ml.teaching.keypose_render import DRAG_STYLE, Frame, render_frames
from ishara_ml.teaching.movement import PathSampler, is_drag, palm_quat
from ishara_ml.teaching_check import _font

CELL = 260
STRIP = 6
LABEL_W = 200
GAP = 10


def _crop(img: np.ndarray, pts: np.ndarray, pad: float = 1.5) -> Image.Image:
    lo, hi = np.nanmin(pts[..., :2].reshape(-1, 2), axis=0), np.nanmax(pts[..., :2].reshape(-1, 2), axis=0)
    side = float((hi - lo).max()) * pad
    cx, cy = (lo + hi) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    return Image.fromarray(img).crop(box).resize((CELL, CELL), Image.Resampling.LANCZOS)


def _frame_at_time(slug: str, ms: int) -> Image.Image:
    """The animation.webp frame on screen at ``ms``. Looked up by time, not index: the WebP encoder merges identical
    consecutive frames (the hold) into one longer frame."""
    t, shown = 0, None
    with Image.open(config.MEDIA_DIR / slug / "animation.webp") as im:
        for frame in ImageSequence.Iterator(im):
            if t > ms:
                break
            shown = frame.convert("RGB").copy()
            t += int(frame.info.get("duration", 0))
    assert shown is not None
    return shown


def _letter(slug: str, skeleton) -> tuple[list[Image.Image], list[Image.Image], list[str]]:
    raw = json.loads(doc_path(slug).read_text(encoding="utf-8"))
    doc = KeyPoseDoc.from_dict(raw, skeleton)
    if doc.path is None:
        raise ValueError(f"{slug} is not a letter with movement")
    track = Track.load(config.KEYPOSE_LANDMARKS_DIR / f"{slug}.npz")
    fps = track.fps
    frames = list(ffmpeg.read_frames(config.NSL23_DIR / track.source, fps))

    # sharpest frame + the handshape drawn with that frame's real orientation
    sharp = np.where(np.isnan(track.sharp), -1, track.sharp)
    best = int(np.argmax(sharp))
    sampler = PathSampler(doc.path, [k.pose for k in doc.key_poses])
    style = DRAG_STYLE if is_drag(sampler, skeleton) else None  # same look as the animation
    used = (doc.analysis or {}).get("sharpFrames") or [best]
    drawn = []
    for kp in doc.key_poses:  # each handshape with the orientation of its own sharpest frame
        group = [f for f in used if kp.source_frames[0] <= f < kp.source_frames[1]] or [best]
        f = max(group, key=lambda i: sharp[i])
        hs = kp.pose.copy()
        hs.wrist_q = palm_quat(track.world[f])
        arm = track.arm[f]
        ang = math.degrees(math.atan2(arm[1] - arm[3], arm[0] - arm[2])) if not np.isnan(arm).any() else 90.0
        drawn.append(render_frames([Frame(hs, ang)], skeleton, CELL, style=style)[0][0])
    handshape = [drawn[0]]
    if len(drawn) > 1:  # two handshapes: side by side
        pair = Image.new("RGB", (CELL, CELL), (255, 255, 255))
        for k, im in enumerate(drawn[:2]):
            pair.paste(im.resize((CELL // 2, CELL // 2)), (k * CELL // 2, CELL // 4))
        handshape = [pair]

    # animation: the movement phase, mapped back to real clip frames
    sampler = PathSampler(doc.path, [k.pose for k in doc.key_poses])
    t = doc.timing
    p, n = timeline.frames_for(t.pauseS), timeline.frames_for(t.transitionS)
    m = timeline.frames_for(sampler.duration * timeline.MOVEMENT_TIME_SCALE)
    picks = np.linspace(0, m, STRIP).round().astype(int)  # 0 = the movement's first state
    strip = [_frame_at_time(slug, (p + n - 1 + i) * round(1000 / timeline.FPS)) for i in picks]
    real_idx = [int(round(doc.path[0].frame + sampler.duration * i / m * fps)) for i in picks]
    real_idx = [min(max(i, 0), len(frames) - 1) for i in real_idx]
    span = track.image[min(real_idx) : max(real_idx) + 1]
    real = [_crop(frames[i], span, 1.3) for i in real_idx]

    top = [_crop(frames[best], track.image[best][None]), handshape[0], *strip]
    a = doc.analysis or {}
    info = [
        f"{slug}  ({len(doc.key_poses)} handshape)",
        f"sharpest frame {best}",
        f"sharp frames {a.get('sharpFrames')}",
        f"sharpness p90 {a.get('sharpness', {}).get('p90')}",
        f"path {a.get('pathLengthHandSizes')} hand sizes",
        f"real frames {real_idx}",
    ]
    return top, real, info


def build(slugs: list[str], out: Path = config.REPORTS_DIR / "movement_check.png") -> Path:
    skeleton = load_skeleton()
    cols = 2 + STRIP
    width = LABEL_W + cols * (CELL + GAP) + GAP
    head = 40
    row_h = 2 * (CELL + GAP) + 2 * GAP
    sheet = Image.new("RGB", (width, head + len(slugs) * row_h), (225, 225, 225))
    draw = ImageDraw.Draw(sheet)
    small = _font(18)
    titles = ["sharpest real frame", "handshape key pose"] + [f"anim / real {i + 1}" for i in range(STRIP)]
    for c, title in enumerate(titles):
        draw.text((LABEL_W + c * (CELL + GAP) + GAP, 10), title, fill=(30, 30, 30), font=small)
    for r, slug in enumerate(slugs):
        y = head + r * row_h
        top, real, info = _letter(slug, skeleton)
        draw.text((GAP, y + 8), "\n".join(info), fill=(0, 0, 0), font=_font(15))
        for c, cell in enumerate(top):
            sheet.paste(cell.convert("RGB").resize((CELL, CELL)), (LABEL_W + c * (CELL + GAP) + GAP, y))
        for c, cell in enumerate(real):
            sheet.paste(cell, (LABEL_W + (c + 2) * (CELL + GAP) + GAP, y + CELL + GAP))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    print(build([s.strip() for s in (args[0] if args else "u,o,am").split(",") if s.strip()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
