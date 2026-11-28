"""Per-letter key-pose media: video -> key poses (editable JSON) -> animation.webp, poster.webp, video.mp4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ishara_ml import config
from ishara_ml.teaching import ffmpeg, timeline
from ishara_ml.teaching.keypose import (
    SKELETON_ID,
    SKELETON_PATH,
    KeyPoseDoc,
    Track,
    analyse,
    build_skeleton_from_tracks,
    classify,
    extract,
    neutral_pose,
)
from ishara_ml.teaching.keypose_render import (
    DRAG_STYLE,
    Frame,
    fit_path,
    lerp_frame,
    render_frames,
    render_movement,
    render_style,
)
from ishara_ml.teaching.kinematics import Skeleton
from ishara_ml.teaching.movement import PathSampler, is_drag


@dataclass
class KeyPoseResult:
    slug: str
    kind: str
    key_poses: int
    frames: int
    animation_kb: float
    total_kb: float
    source: str


def load_skeleton(path: Path = SKELETON_PATH) -> Skeleton:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run make_teaching_media --build-skeleton first")
    return Skeleton.load(path)


def doc_path(slug: str) -> Path:
    return config.KEYPOSES_DIR / f"{slug}.json"


def loop_frames(doc: KeyPoseDoc, skeleton: Skeleton) -> list[Frame]:
    keys = [Frame(k.pose, k.forearm_deg) for k in doc.key_poses]
    neutral = Frame(neutral_pose(skeleton, keys[0].pose), keys[0].forearm_deg)
    return timeline.build_timeline(neutral, keys, lerp_frame, doc.timing)


def movement_frames(doc: KeyPoseDoc, skeleton: Skeleton) -> tuple[list[Frame], PathSampler]:
    """One loop of a letter with movement: neutral -> movement (handshape + wrist path) -> hold -> neutral."""
    sampler = PathSampler(doc.path or [], [k.pose for k in doc.key_poses])

    def sample(t: float) -> Frame:
        pose, forearm = sampler(t)
        return Frame(pose, forearm)

    start = sample(0.0)
    neutral = Frame(neutral_pose(skeleton, start.pose), start.forearm_deg)
    return timeline.build_movement_timeline(neutral, sample, sampler.duration, lerp_frame, doc.timing), sampler


def render_movement_doc(doc: KeyPoseDoc, skeleton: Skeleton, size: int = 360):
    """(frames, poster, MovementRender) for a letter with movement; the poster is the last frame of the hold."""
    frames, sampler = movement_frames(doc, skeleton)
    drag = (doc.render or {}).get("drag")  # per-letter override, else the automatic rule
    style = DRAG_STYLE if (is_drag(sampler, skeleton) if drag is None else drag) else None  # every dragging letter
    extra, front = render_style(doc.render)
    result = render_movement(frames, skeleton, size, style, extra, front)
    final = len(frames) - 1 - timeline.frames_for(doc.timing.returnS)  # last frame of the hold
    return result.images, result.images[final].copy(), result


def render_doc(doc: KeyPoseDoc, skeleton: Skeleton, size: int = 360) -> tuple[list[Image.Image], Image.Image]:
    """Animation frames + poster (the final key pose, same view)."""
    if doc.path is not None:  # letters with movement only
        images, poster, result = render_movement_doc(doc, skeleton, size)
        kind = "drag style (natural hand)" if result.style else "movement, static look (little travel)"
        print(f"  {doc.slug}: {kind}", flush=True)
        if result.scale < 1.0:
            print(
                f"  {doc.slug}: view shrunk to {result.scale:.0%} of the static scale to fit the movement", flush=True
            )
        return images, poster
    frames, factor = fit_path(loop_frames(doc, skeleton), skeleton)
    if factor < 1.0:
        print(f"  {doc.slug}: wrist path scaled to {factor:.0%} to fit the shared view", flush=True)
    extra, front = render_style(doc.render)  # per-letter options; none = drawn exactly as before
    images, centre = render_frames(frames, skeleton, size, style=extra or None, front=front)
    final = frames[-1 - timeline.frames_for(doc.timing.returnS)]  # last frame of the hold
    poster, _ = render_frames([final], skeleton, size, centre=centre, style=extra or None, front=front)
    return images, poster[0]


def save_media(images: list[Image.Image], poster: Image.Image, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    anim = out_dir / "animation.webp"
    images[0].save(
        anim,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=round(1000 / timeline.FPS),
        quality=80,
        method=6,
    )
    poster.save(out_dir / "poster.webp", quality=80, method=6)
    return anim


def process_keypose(
    row: dict[str, str], *, from_json: bool = False, size: int = 360, media_dir: Path = config.MEDIA_DIR
) -> KeyPoseResult:
    slug = row["slug"]
    skeleton = load_skeleton()
    out_dir = media_dir / slug
    locked = doc_path(slug).exists() and json.loads(doc_path(slug).read_text(encoding="utf-8")).get("locked")
    if from_json or locked:  # never opens the video; "locked": true = hand-edited, never re-analysed
        doc = KeyPoseDoc.load(doc_path(slug), skeleton)
        if locked and not from_json:
            print(f"  {slug}: key poses locked (hand-edited JSON): rendering from JSON", flush=True)
    else:
        track = extract(row)
        if classify(track)[0] and not track.has_movement_fields:  # movement letters need sharpness + Pose arm
            track = extract(row, need_movement=True)
        doc = analyse(track, skeleton)
        doc.save(doc_path(slug), skeleton)
        from ishara_ml.nsl23 import resolve_video

        src = resolve_video(config.NSL23_DIR, row["signer"], row["file"])
        start = float(row["start"]) if (row.get("start") or "").strip() else None
        end = float(row["end"]) if (row.get("end") or "").strip() else None
        ffmpeg.encode_preview_mp4(src, out_dir / "video.mp4", start, end)
    images, poster = render_doc(doc, skeleton, size)
    anim = save_media(images, poster, out_dir)
    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    return KeyPoseResult(
        slug, doc.kind, len(doc.key_poses), len(images), anim.stat().st_size / 1024, total / 1024, doc.source["file"]
    )


def build_skeleton(rows: list[dict[str, str]], signer: str = "S1") -> Skeleton:
    """Extract (cached) landmarks for every ``signer`` row and write the shared skeleton. No media."""
    tracks: list[Track] = []
    for row in rows:
        if row["signer"] != signer:
            continue
        tracks.append(extract(row))
        print(f"  landmarks {row['slug']}", flush=True)
    skeleton = build_skeleton_from_tracks(tracks, SKELETON_ID)
    skeleton.save(SKELETON_PATH)
    return skeleton
