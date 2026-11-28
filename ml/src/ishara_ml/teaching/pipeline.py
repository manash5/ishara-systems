"""Per-letter teaching-media pipeline (brief §5): video -> landmarks -> one crop box -> drawn frames -> files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ishara_ml import config
from ishara_ml.nsl23 import resolve_video
from ishara_ml.teaching import ffmpeg
from ishara_ml.teaching.geometry import (
    facing_track,
    finger_states,
    hand_size,
    held_frame,
    hold_segment,
    movement_segment,
)
from ishara_ml.teaching.landmarker import detect_hands
from ishara_ml.teaching.photo import photo_frames
from ishara_ml.teaching.pose import detect_elbows, fill_and_smooth
from ishara_ml.teaching.render import draw_hand
from ishara_ml.teaching.track import crop_box, smooth, to_box_coords, trim_and_fill

STYLES = ("illustrated", "photo")
# Frames are sampled at 24 fps (smooth movement) and played back at 16 fps: 1.5x slower than real time.
CAPTURE_FPS = 24.0
PLAYBACK_FPS = 16.0
# Signs whose movement is part of the sign (brief §3). Every other sign is a static hand shape: only its held
# part is animated, so the loop doesn't replay the signer raising the hand into position.
MOVEMENT_SIGNS = frozenset({"aa", "u", "uu", "e", "ai", "o", "au", "am", "ah"})
# Movement signs pause on the final shape before the loop restarts, and briefly on the first one.
END_HOLD_MS = 700
STATIC_SMOOTH_WINDOW = 9  # frames (~0.4 s at 24 fps)
BOX_PAD = 0.25  # crop-box margin as a fraction of the hand's extent (smaller = hand drawn larger)
OVERRIDES = config.LABELS_DIR / "teaching_overrides.json"
START_HOLD_MS = 250
# Static signs: no pause at the start, the held shape stays ~1 s before the loop restarts (brief §5).
STATIC_START_HOLD_MS = 0
STATIC_END_HOLD_MS = 1000


def load_overrides(path: Path = OVERRIDES) -> dict[str, dict]:
    """Per-letter manual corrections where the landmark detector is known to be wrong (reviewed by a person)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def smooth_window(fps: float) -> int:
    """Odd smoothing window covering ~0.15 s (3 frames at 12-20 fps, 5 at 24-33 fps)."""
    n = max(3, round(fps * 0.15))
    return n if n % 2 else n + 1


def frame_durations(n: int, playback_fps: float, moving: bool) -> list[int]:
    """Per-frame display time (ms). The last pose is held so the restart reads as a pause, not as a jump;
    movement signs also pause briefly on the first pose."""
    base = round(1000 / playback_fps)
    durations = [base] * n
    if n > 1:
        durations[0] += START_HOLD_MS if moving else STATIC_START_HOLD_MS
        durations[-1] += END_HOLD_MS if moving else STATIC_END_HOLD_MS
    return durations


@dataclass
class Result:
    slug: str
    frames: int
    animation_kb: float
    total_kb: float
    source: str


def load_map(path: Path = config.LABELS_DIR / "nsl23_map.csv") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _seconds(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def process_row(
    row: dict[str, str],
    *,
    style: str | None = None,  # None: the letter's reviewed choice in teaching_overrides.json, else illustrated
    fps: float = CAPTURE_FPS,
    size: int = 360,
    root: Path = config.NSL23_DIR,
    media_dir: Path = config.MEDIA_DIR,
    landmarks_dir: Path = config.TEACHING_LANDMARKS_DIR,
    model: Path = config.HAND_LANDMARKER_MODEL,
    pose_model: Path | None = config.POSE_LANDMARKER_MODEL,
    handedness: str = "right",  # NSL23 signers are all right-handed (brief §3)
    playback_fps: float = PLAYBACK_FPS,
    running_mode: str = "image",
) -> Result:
    slug = row["slug"]
    fixes = load_overrides().get(slug, {})
    style = style or fixes.get("style", "illustrated")
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    src = resolve_video(root, row["signer"], row["file"])
    start, end = _seconds(row.get("start", "")), _seconds(row.get("end", ""))
    out_dir = media_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg.encode_preview_mp4(src, out_dir / "video.mp4", start, end)

    frames = list(ffmpeg.read_frames(src, fps, start, end))
    if not frames:
        raise RuntimeError("no frames decoded")
    h, w = frames[0].shape[:2]
    raw = detect_hands(frames, fps, model, mode=running_mode)
    poses, first = trim_and_fill(raw)
    if not poses:
        raise RuntimeError(f"no hand detected in {len(frames)} frames")
    poses = smooth(poses, window=smooth_window(fps))

    # Keep only the part of the clip that is the sign itself (see geometry.hold_segment / movement_segment).
    moving = slug in MOVEMENT_SIGNS
    seg_a, seg_b = (movement_segment if moving else hold_segment)(poses, fps)
    poses = poses[seg_a:seg_b]
    first += seg_a
    if not moving:
        # A static hand shape shouldn't wobble: smooth over ~0.4 s so landmark jitter doesn't move fingers around.
        poses = smooth(poses, window=STATIC_SMOOTH_WINDOW)
    detected = sum(p is not None for p in raw[first : first + len(poses)])
    # The box may extend past the frame so it stays centred on the hand (drawings need no pixels; photos pad white).
    box = crop_box(poses, w, h, pad=BOX_PAD, clamp=False)  # outside the frame is white (photo pads too)
    unit = [to_box_coords(p, box) for p in poses]

    seg_frames = frames[first : first + len(poses)]
    elbows_px = (
        fill_and_smooth(detect_elbows(seg_frames, [p[0] for p in poses], pose_model), smooth_window(fps))
        if pose_model
        else None
    )
    elbows = (
        None
        if elbows_px is None
        else [np.array([(e[0] - box.x0) / box.side, (e[1] - box.y0) / box.side]) for e in elbows_px]
    )

    size_unit = hand_size(unit)
    forced = {n: False for n in fixes.get("fold", [])} | {n: True for n in fixes.get("raise", [])}
    extended = finger_states(unit, static=not moving, overrides=forced)
    facing = facing_track(unit, handedness)
    held = held_frame(unit)

    if style == "illustrated":
        drawn = [
            draw_hand(
                u,
                size=size,
                hand_scale=size_unit,
                extended=ext,
                facing=f,
                handedness=handedness,
                elbow=None if elbows is None else elbows[i],
            )
            for i, (u, ext, f) in enumerate(zip(unit, extended, facing, strict=True))
        ]
    else:
        drawn = photo_frames(seg_frames, poses, elbows_px, box, size, size_unit * box.side)

    durations = frame_durations(len(drawn), playback_fps, moving)
    anim = out_dir / "animation.webp"
    drawn[0].save(
        anim,
        save_all=True,
        append_images=drawn[1:],
        loop=0,
        duration=durations,
        quality=80,
        method=6,
    )
    drawn[held].save(out_dir / "poster.webp", quality=80, method=6)

    rel_src = src.relative_to(root).as_posix()
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    (landmarks_dir / f"{slug}.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "source": rel_src,
                "signer": row["signer"],
                "fps": fps,
                "playbackFps": playback_fps,
                "runningMode": running_mode,
                "style": style,
                "kind": "movement" if moving else "static",
                "segment": [seg_a, seg_b],
                "durationsMs": durations,
                "start": start,
                "end": end,
                "frameSize": [w, h],
                "firstFrame": first,
                "posterFrame": held,
                "posterTimeS": round((start or 0) + (first + held) / fps, 4),
                "detectedFrames": detected,
                "box": box.to_dict(),
                "handSizePx": round(size_unit * box.side, 2),
                "handedness": handedness,
                "extendedFraction": {n: round(sum(e[n] for e in extended) / len(extended), 2) for n in extended[0]},
                "palmFacing": facing,
                "overrides": fixes or None,
                "elbows": None if elbows_px is None else [np.round(e, 1).tolist() for e in elbows_px],
                "landmarks": [np.round(p, 2).tolist() for p in poses],
                "coords": "source-frame pixels (x, y, z*frameWidth); smoothed, gaps held; only the sign segment",
            }
        ),
        encoding="utf-8",
    )
    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    return Result(slug, len(drawn), anim.stat().st_size / 1024, total / 1024, rel_src)
