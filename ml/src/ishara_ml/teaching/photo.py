"""Photo style (brief §2 fallback): the real hand cut out onto white, for letters a flat drawing can't show.

rembg separates the person from the wall; that alone would keep the signer's shirt and body in the crop, so its
mask is intersected with a hand + forearm region built from the landmarks (and the elbow direction), with soft
edges and a soft fade along the forearm like the drawings.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ishara_ml import config
from ishara_ml.teaching.geometry import FINGERS
from ishara_ml.teaching.render import PALM, forearm_direction, forearm_polygon
from ishara_ml.teaching.track import Box

U2NET_HOME = config.MODELS_DIR / "u2net"  # keep rembg's model download inside the (git-ignored) project folder
FINGER_HALF_WIDTH = 0.32  # x hand size: generous - rembg decides the true (crisp) edge of the hand
FOREARM_FADE = (1.0, 1.6)  # x hand size past the wrist: forearm solid until the first, gone by the second


def _session():
    os.environ.setdefault("U2NET_HOME", str(U2NET_HOME))
    try:
        from rembg import new_session
    except ImportError as exc:  # optional dependency, only for this style
        raise RuntimeError("--style photo needs rembg: run `uv add rembg[cpu]` in ml/") from exc
    return new_session("u2net")


def crop_white(frame: np.ndarray, box: Box) -> Image.Image:
    """The square box from the frame; parts outside the frame are white (boxes are not clamped)."""
    x0, y0, side = round(box.x0), round(box.y0), round(box.side)
    out = Image.new("RGB", (side, side), (255, 255, 255))
    out.paste(Image.fromarray(frame), (-x0, -y0))
    return out


def hand_region(pose_px: np.ndarray, elbow_px: np.ndarray | None, box: Box, size_px: float) -> Image.Image:
    """Mask (L, crop size) covering the hand generously and the forearm up to a soft fade."""
    side = round(box.side)
    mask = Image.new("L", (side, side), 0)
    d = ImageDraw.Draw(mask)
    xy = pose_px[:, :2] - [box.x0, box.y0]
    r = FINGER_HALF_WIDTH * size_px
    d.polygon([tuple(xy[i]) for i in PALM], fill=255)
    for chain in FINGERS.values():
        pts = [tuple(xy[i]) for i in [0, *chain]]
        d.line(pts, fill=255, width=max(1, round(2 * r)), joint="curve")
        for x, y in pts:
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    elbow = None if elbow_px is None else np.asarray(elbow_px, dtype=np.float64)[:2] - [box.x0, box.y0]
    arm = forearm_polygon(xy, size_px, length=FOREARM_FADE[1] * size_px, elbow=elbow)
    # The arm strip is wide: the person mask from rembg gives its real edges.
    centre = np.mean(arm, axis=0)
    d.polygon([tuple(centre + (p - centre) * 1.6) for p in arm], fill=255)
    region = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32) / 255.0
    # Soft fade only along the forearm, measured from the wrist like the drawings.
    away = forearm_direction(xy, elbow)
    ys, xs = np.mgrid[0:side, 0:side].astype(np.float32)
    along = (xs - xy[0, 0]) * away[0] + (ys - xy[0, 1]) * away[1]
    a, b = FOREARM_FADE[0] * size_px, FOREARM_FADE[1] * size_px
    region *= 1.0 - np.clip((along - a) / (b - a), 0.0, 1.0)
    return Image.fromarray((region * 255).round().astype(np.uint8))


def photo_frames(
    frames: list[np.ndarray],
    poses_px: list[np.ndarray],
    elbows_px: list[np.ndarray] | None,
    box: Box,
    size: int,
    size_px: float,
) -> list[Image.Image]:
    session = _session()
    from rembg import remove

    out = []
    for i, frame in enumerate(frames):
        crop = crop_white(frame, box)
        cut = remove(crop, session=session, only_mask=True)  # L mask of the person
        region = hand_region(poses_px[i], None if elbows_px is None else elbows_px[i], box, size_px)
        alpha = (np.asarray(cut, dtype=np.float32) / 255.0) * (np.asarray(region, dtype=np.float32) / 255.0)
        rgb = np.asarray(crop, dtype=np.float32)
        comp = rgb * alpha[..., None] + 255.0 * (1 - alpha[..., None])
        img = Image.fromarray(comp.round().astype(np.uint8))
        out.append(img.resize((size, size), Image.Resampling.LANCZOS))
    return out


def model_path() -> Path:
    return U2NET_HOME / "u2net.onnx"
