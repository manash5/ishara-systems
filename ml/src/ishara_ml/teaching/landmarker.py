"""MediaPipe Hand Landmarker (Tasks API) over a list of frames, plus a precise second pass on the hand crop.

First pass (default VIDEO mode, tracking between frames) finds the hand; ``refine_poses`` then re-detects it on
an upscaled crop for precise points. (VIDEO-mode tracking alone once bent ka's pointing finger into the fist; the
refined crop pass corrects such cases. ``image`` mode = full detection on every frame remains an option.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ishara_ml.teaching.track import Pose, hand_area

RUNNING_MODES = ("image", "video")


@dataclass
class HandObs:
    """One detected hand: image points (pixels; z = MediaPipe depth x frame width), world points (metres, camera
    axes, origin near the hand's centre) and MediaPipe's handedness label."""

    image: Pose
    world: np.ndarray
    handedness: str
    score: float


def _options(model: Path, mode: str) -> vision.HandLandmarkerOptions:
    return vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
        num_hands=2,
    )


def _observations(result, w: float, h: float, scale: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> list[HandObs]:
    """Every hand in a result; image points mapped back through an optional crop (scale, x0, y0)."""
    out = []
    for i, hand in enumerate(result.hand_landmarks):
        image = np.array([[lm.x * w / scale + x0, lm.y * h / scale + y0, lm.z * w / scale] for lm in hand])
        world = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]], dtype=np.float64)
        cat = result.handedness[i][0] if result.handedness and result.handedness[i] else None
        out.append(HandObs(image, world, cat.category_name if cat else "", float(cat.score) if cat else 0.0))
    return out


def detect_hand_obs(frames: list[np.ndarray], fps: float, model: Path, mode: str = "video") -> list[HandObs | None]:
    """Largest hand per frame with world landmarks + handedness, or None where no hand is found."""
    if mode not in RUNNING_MODES:
        raise ValueError(f"mode must be one of {RUNNING_MODES}")
    out: list[HandObs | None] = []
    with vision.HandLandmarker.create_from_options(_options(model, mode)) as landmarker:
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            if mode == "video":
                result = landmarker.detect_for_video(image, int(round(i * 1000 / fps)))
            else:
                result = landmarker.detect(image)
            hands = _observations(result, w, h)
            out.append(max(hands, key=lambda o: hand_area(o.image)) if hands else None)
    return out


def detect_hands(
    frames: list[np.ndarray], fps: float, model: Path, num_hands: int = 2, mode: str = "image"
) -> list[Pose | None]:
    """Largest hand per frame as a (21, 3) pixel-space pose, or None where no hand is found.

    A new landmarker is created per clip, so VIDEO-mode timestamps restart at 0.
    """
    return [None if o is None else o.image for o in detect_hand_obs(frames, fps, model, mode)]


REFINE_SIZE = 512  # px: the hand crop is upscaled to this before the second pass
REFINE_MARGIN = 1.8  # crop side = this x the hand's larger extent


def _crop_transform(pose: Pose) -> tuple[float, float, float, float]:
    lo, hi = pose[:, :2].min(axis=0), pose[:, :2].max(axis=0)
    side = max(float((hi - lo).max()) * REFINE_MARGIN, 64.0)
    cx, cy = (lo + hi) / 2
    return side, REFINE_SIZE / side, float(cx - side / 2), float(cy - side / 2)


def refine_obs(frames: list[np.ndarray], obs: list[HandObs | None], model: Path) -> tuple[list[HandObs | None], int]:
    """Second pass: re-detect each hand on a square crop around it, upscaled to ``REFINE_SIZE`` px, and map the
    points back to frame coordinates. On a 1080p frame the hand is only ~150-300 px, so the upscaled crop gives the
    landmark model more detail (finger bends, thumb). Keeps the first-pass hand where the crop finds no hand.

    Returns the refined observations and how many frames were refined.
    """
    import cv2

    out: list[HandObs | None] = []
    refined = 0
    with vision.HandLandmarker.create_from_options(_options(model, "image")) as landmarker:
        for frame, ob in zip(frames, obs, strict=True):
            if ob is None:
                out.append(None)
                continue
            side, s, x0, y0 = _crop_transform(ob.image)
            # Affine crop (pads outside the frame with white) straight to REFINE_SIZE.
            m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], dtype=np.float64)
            crop = cv2.warpAffine(
                frame, m, (REFINE_SIZE, REFINE_SIZE), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
            )
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            hands = _observations(result, REFINE_SIZE, REFINE_SIZE, s, x0, y0)
            best, best_d = None, float("inf")
            for cand in hands:
                d = float(np.linalg.norm(cand.image[:, :2] - ob.image[:, :2], axis=1).mean())  # same hand
                if d < best_d:
                    best, best_d = cand, d
            if best is not None and best_d < 0.5 * side:
                out.append(best)
                refined += 1
            else:
                out.append(ob)
    return out, refined


def refine_poses(frames: list[np.ndarray], poses: list[Pose | None], model: Path) -> tuple[list[Pose | None], int]:
    """``refine_obs`` for plain pixel poses (world landmarks dropped)."""
    obs = [None if p is None else HandObs(p, np.zeros((21, 3)), "", 0.0) for p in poses]
    out, refined = refine_obs(frames, obs, model)
    return [None if o is None else o.image for o in out], refined
