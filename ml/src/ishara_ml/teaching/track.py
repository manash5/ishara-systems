"""Pure pose-track helpers: pick the hand, trim/fill gaps, smooth, and compute one crop box per clip.

A pose is a (21, 3) float array: x, y in source-frame pixels, z in the same pixel scale (MediaPipe's relative
depth times the frame width; smaller = nearer the camera).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pose = np.ndarray


def hand_area(pose: Pose) -> float:
    """Area of the hand's 2-D bounding box in pixels²."""
    span = pose[:, :2].max(axis=0) - pose[:, :2].min(axis=0)
    return float(span[0] * span[1])


def pick_largest(hands: list[Pose]) -> Pose | None:
    """The largest detected hand, or None if there are none."""
    return max(hands, key=hand_area) if hands else None


def trim_and_fill(poses: list[Pose | None]) -> tuple[list[Pose], int]:
    """Drop frames before the hand first appears / after it last appears; hold the previous pose over gaps.

    Returns the filled poses and the index of the first kept frame. Empty list if the hand never appears.
    """
    present = [i for i, p in enumerate(poses) if p is not None]
    if not present:
        return [], 0
    first, last = present[0], present[-1]
    filled: list[Pose] = []
    prev: Pose | None = None
    for p in poses[first : last + 1]:
        if p is not None:
            prev = p
        assert prev is not None
        filled.append(prev.copy())
    return filled, first


def smooth(poses: list[Pose], window: int = 3) -> list[Pose]:
    """Centred moving average over ``window`` frames; edge frames average the neighbours that exist."""
    if window <= 1 or len(poses) < 2:
        return [p.copy() for p in poses]
    stack = np.stack(poses).astype(np.float64)
    half = window // 2
    out = []
    for i in range(len(stack)):
        lo, hi = max(0, i - half), min(len(stack), i + half + 1)
        out.append(stack[lo:hi].mean(axis=0))
    return out


@dataclass(frozen=True)
class Box:
    """Square crop box in source-frame pixels."""

    x0: float
    y0: float
    side: float

    def to_dict(self) -> dict[str, float]:
        return {"x0": round(self.x0, 2), "y0": round(self.y0, 2), "side": round(self.side, 2)}


def crop_box(
    poses: list[Pose],
    frame_w: int,
    frame_h: int,
    pad: float = 0.30,
    wrist_extra: float = 0.05,
    clamp: bool = True,
) -> Box:
    """One square box around the union of all poses, so the view never jumps.

    ``pad`` adds that fraction of the hand's larger extent as margin in total (split over both sides);
    ``wrist_extra`` extends the bottom by that fraction of the extent for the wrist/forearm.
    With ``clamp`` (needed when real pixels are cropped) the box is shrunk to fit and shifted to stay inside the
    frame. Without it (illustrations) the box stays centred on the hand, even near the frame edge.
    """
    if not poses:
        raise ValueError("no poses")
    pts = np.concatenate([p[:, :2] for p in poses])
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    extent = max(x_max - x_min, y_max - y_min, 1.0)
    y_max = y_max + wrist_extra * extent
    side = max(x_max - x_min, y_max - y_min) + pad * extent
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    if not clamp:
        return Box(float(cx - side / 2), float(cy - side / 2), float(side))
    side = min(side, float(frame_w), float(frame_h))
    x0 = float(np.clip(cx - side / 2, 0, frame_w - side))
    y0 = float(np.clip(cy - side / 2, 0, frame_h - side))
    return Box(x0, y0, float(side))


def to_box_coords(pose: Pose, box: Box) -> Pose:
    """Pose in unit-box coordinates: x, y in [0, 1] inside the box; z scaled the same way."""
    out = pose.astype(np.float64).copy()
    out[:, 0] = (out[:, 0] - box.x0) / box.side
    out[:, 1] = (out[:, 1] - box.y0) / box.side
    out[:, 2] = out[:, 2] / box.side
    return out
