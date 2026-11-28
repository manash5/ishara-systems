"""Teaching media from the real hand (styles ``toon`` and ``photo``), brief §5.

video -> landmarks (VIDEO pass + refined pass on the upscaled hand crop) -> trim to the sign -> one tight square crop
per clip -> hand mask (person ∩ hand region ∩ skin) -> toon / photo frames -> animation.webp, poster.webp,
video.mp4 (trimmed the same way) + the landmarks JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ishara_ml import config
from ishara_ml.nsl23 import resolve_video
from ishara_ml.teaching import ffmpeg
from ishara_ml.teaching.cutout import (
    WORK,
    crop_square,
    fit_skin,
    hand_mask,
    hand_only,
    mask_bbox,
    person_masks,
    smooth_alpha,
    temporal_median,
    to_crop,
)
from ishara_ml.teaching.geometry import hand_size, held_frame, sign_segment
from ishara_ml.teaching.landmarker import detect_hands, refine_poses
from ishara_ml.teaching.pose import detect_elbows, fill_and_smooth
from ishara_ml.teaching.render import forearm_direction
from ishara_ml.teaching.stylise import photo_frame, stable_edges, strong_edges, tone_thresholds, toon_frame
from ishara_ml.teaching.track import Box, crop_box, smooth, trim_and_fill

CUTOUT_STYLES = ("toon", "photo")
FPS = 12.0  # frames sampled and played at 12 fps (real time)
HOLD_END_S = 1.0  # the final pose is held this long before the loop restarts
BOX_PAD = 0.15  # tight square crop: the hand fills most of the frame; the forearm runs off the edge
GENEROUS_PAD = 0.9  # first crop around the landmarks, before the real hand pixels are known
FOREARM_FADE = (1.0, 1.6)  # x hand size past the wrist: the forearm is solid, then fades out (never a stump)


def forearm_fade(
    shape: tuple[int, int], pose_xy: np.ndarray, forearm_dir: np.ndarray, size_px: float, hand: np.ndarray
) -> np.ndarray:
    """Multiplier 1 -> 0 along the forearm between ``FOREARM_FADE`` hand sizes past the wrist; 1 on the hand."""
    d = np.asarray(forearm_dir, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    along = (xs - pose_xy[0, 0]) * d[0] + (ys - pose_xy[0, 1]) * d[1]
    a, b = FOREARM_FADE[0] * size_px, FOREARM_FADE[1] * size_px
    fade = 1.0 - np.clip((along - a) / (b - a), 0.0, 1.0)
    return np.where(hand, 1.0, fade).astype(np.float32)


def frame_box(hand_masks: list[np.ndarray], gbox: Box, poses: list[np.ndarray], w: int, h: int) -> Box:
    """Tight square around the union of the hand pixels (in frame coordinates) with ``BOX_PAD`` margin."""
    boxes = [bb for bb in (mask_bbox(m) for m in hand_masks) if bb is not None]
    if not boxes:
        return crop_box(poses, w, h, pad=BOX_PAD, wrist_extra=0.0, clamp=False)
    k = gbox.side / WORK
    x0 = gbox.x0 + min(b[0] for b in boxes) * k
    y0 = gbox.y0 + min(b[1] for b in boxes) * k
    x1 = gbox.x0 + max(b[2] for b in boxes) * k
    y1 = gbox.y0 + max(b[3] for b in boxes) * k
    side = max(x1 - x0, y1 - y0) * (1 + BOX_PAD)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return Box(cx - side / 2, cy - side / 2, side)


def reframe(mask: np.ndarray, src: Box, dst: Box) -> np.ndarray:
    """A mask made on the ``src`` crop, resampled onto the ``dst`` crop (both WORK px)."""
    s = src.side / dst.side
    m = np.array([[s, 0, (src.x0 - dst.x0) * WORK / dst.side], [0, s, (src.y0 - dst.y0) * WORK / dst.side]])
    out = cv2.warpAffine(mask.astype(np.float32), m, (WORK, WORK), flags=cv2.INTER_LINEAR, borderValue=0)
    return out >= 0.5


@dataclass
class CutoutResult:
    slug: str
    frames: int
    animation_kb: float
    total_kb: float
    source: str
    refined: int


def durations_ms(n: int, fps: float = FPS, hold_s: float = HOLD_END_S) -> list[int]:
    """Frame times: uniform, with the final pose held ``hold_s`` longer so the loop doesn't jump."""
    base = round(1000 / fps)
    d = [base] * n
    if n:
        d[-1] += round(hold_s * 1000)
    return d


def save_animation(frames: list[Image.Image], path: Path, durations: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=durations, quality=80, method=6)


def process_cutout(
    row: dict[str, str],
    *,
    moving: bool,
    styles: tuple[str, ...] = ("toon",),
    out_dirs: dict[str, Path] | None = None,
    size: int = 360,
    fps: float = FPS,
    root: Path = config.NSL23_DIR,
    landmarks_dir: Path = config.TEACHING_LANDMARKS_DIR,
    hand_model: Path = config.HAND_LANDMARKER_MODEL,
    pose_model: Path | None = config.POSE_LANDMARKER_MODEL,
) -> CutoutResult:
    """Render ``styles`` for one map row. The first style goes to ``out_dirs[style]`` (default: the media folder),
    together with video.mp4; ``out_dirs`` maps every style to the folder that receives ``<slug>/...``."""
    slug = row["slug"]
    out_dirs = out_dirs or {styles[0]: config.MEDIA_DIR}
    for st in styles:
        if st not in CUTOUT_STYLES:
            raise ValueError(f"style must be one of {CUTOUT_STYLES}")
    src = resolve_video(root, row["signer"], row["file"])
    start = float(row["start"]) if (row.get("start") or "").strip() else None
    end = float(row["end"]) if (row.get("end") or "").strip() else None

    frames = list(ffmpeg.read_frames(src, fps, start, end))
    if not frames:
        raise RuntimeError("no frames decoded")
    h, w = frames[0].shape[:2]

    # 1. Landmarks: VIDEO pass (finds/tracks the hand), then a precise pass on the upscaled hand crop.
    first_pass = detect_hands(frames, fps, hand_model, mode="video")
    refined_raw, refined = refine_poses(frames, first_pass, hand_model)
    poses, first = trim_and_fill(refined_raw)
    if not poses:
        raise RuntimeError(f"no hand detected in {len(frames)} frames")
    poses = smooth(poses, window=3)

    # 2. Trim to the sign (no arm rising from rest / lowering).
    a, b = sign_segment(poses, fps, moving)
    poses = poses[a:b]
    first += a
    seg = frames[first : first + len(poses)]

    # 3. Cut the hand out on a generous crop first (landmarks can be short of the real fingertips) ...
    gbox = crop_box(poses, w, h, pad=GENEROUS_PAD, wrist_extra=0.0, clamp=False)
    gcrops = [crop_square(f, gbox.x0, gbox.y0, gbox.side) for f in seg]
    gpts = [to_crop(p, gbox.x0, gbox.y0, gbox.side) for p in poses]
    gsize = hand_size(poses) * WORK / gbox.side

    # Forearm direction: towards the elbow (MediaPipe Pose), else away from the knuckles.
    elbows = fill_and_smooth(detect_elbows(seg, [p[0] for p in poses], pose_model), 3) if pose_model else None
    dirs = [
        forearm_direction(gpts[i], None if elbows is None else to_crop(elbows[i], gbox.x0, gbox.y0, gbox.side))
        for i in range(len(gpts))
    ]

    # 4. Hand mask = person ∩ hand region ∩ skin, cleaned and stabilised over time.
    people = person_masks(gcrops)
    skin = fit_skin(gcrops, gpts, people)
    gmasks = [hand_mask(c, pm, p, d, gsize, skin) for c, pm, p, d in zip(gcrops, people, gpts, dirs, strict=True)]
    gmasks = temporal_median(gmasks)

    # ... then frame it: one tight square around the real hand pixels (not the forearm) across the clip.
    box = frame_box([hand_only(m, p, gsize) for m, p in zip(gmasks, gpts, strict=True)], gbox, poses, w, h)
    crops = [crop_square(f, box.x0, box.y0, box.side) for f in seg]
    size_px = hand_size(poses) * WORK / box.side
    masks = [reframe(m, gbox, box) for m in gmasks]
    pts = [to_crop(p, box.x0, box.y0, box.side) for p in poses]
    alphas = [
        smooth_alpha(m, size_px) * forearm_fade(m.shape, p, d, size_px, hand_only(m, p, size_px))
        for m, p, d in zip(masks, pts, dirs, strict=True)
    ]

    held = held_frame(poses)
    durations = durations_ms(len(poses), fps)

    # 5./6. Styles.
    rendered: dict[str, list[Image.Image]] = {}
    for st in styles:
        if st == "photo":
            rendered[st] = [photo_frame(c, al, size) for c, al in zip(crops, alphas, strict=True)]
        else:
            thresholds = tone_thresholds(crops, alphas)
            edges = stable_edges([strong_edges(c, al, size_px) for c, al in zip(crops, alphas, strict=True)])
            rendered[st] = [
                toon_frame(c, al, thresholds, e, size_px, size) for c, al, e in zip(crops, alphas, edges, strict=True)
            ]
        base = out_dirs.get(st, config.REPORTS_DIR / "styles" / st) / slug
        save_animation(rendered[st], base / "animation.webp", durations)
        rendered[st][held].save(base / "poster.webp", quality=80, method=6)

    # 7. The real video, trimmed the same way (main style's folder).
    main_dir = out_dirs.get(styles[0], config.MEDIA_DIR) / slug
    t0 = (start or 0.0) + first / fps
    ffmpeg.encode_preview_mp4(src, main_dir / "video.mp4", t0, t0 + len(poses) / fps)

    rel_src = src.relative_to(root).as_posix()
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    (landmarks_dir / f"{slug}.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "source": rel_src,
                "signer": row["signer"],
                "styles": list(styles),
                "kind": "movement" if moving else "static",
                "fps": fps,
                "start": start,
                "end": end,
                "frameSize": [w, h],
                "firstFrame": first,
                "segment": [a, b],
                "posterFrame": held,
                "posterTimeS": round((start or 0) + (first + held) / fps, 4),
                "durationsMs": durations,
                "refinedFrames": refined,
                "box": box.to_dict(),
                "handSizePx": round(hand_size(poses), 2),
                "skin": {"meanAB": [round(float(x), 2) for x in skin.mean]},
                "elbows": None if elbows is None else [np.round(e, 1).tolist() for e in elbows],
                "landmarks": [np.round(p, 2).tolist() for p in poses],
                "coords": "source-frame pixels (x, y, z*frameWidth); refined on the hand crop; only the sign",
            }
        ),
        encoding="utf-8",
    )
    main = out_dirs.get(styles[0], config.MEDIA_DIR) / slug
    total = sum(f.stat().st_size for f in main.iterdir() if f.is_file())
    return CutoutResult(
        slug, len(poses), (main / "animation.webp").stat().st_size / 1024, total / 1024, rel_src, refined
    )
