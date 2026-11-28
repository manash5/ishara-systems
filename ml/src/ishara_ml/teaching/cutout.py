"""Cut the real hand out of a clip (shared by the toon and photo styles).

hand mask = person (rembg) ∩ region around the hand (dilated convex hull of the 21 landmarks + a corridor along
the forearm) ∩ skin colour (calibrated per clip from pixels inside the palm). The NSL23 wall is beige, close to
skin, so skin colour is never used on its own; the sleeve, body and face are excluded by colour and by region.
Then: keep the connected part that contains the hand, fill only small holes (real gaps such as the loop in ठ stay
open), smooth the edges, and take a 3-frame temporal median so the outline doesn't flicker.

Everything works on square crops at ``WORK`` px; the renderers downscale to the output size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from ishara_ml import config

WORK = 720  # px: crops are processed at this size (2x the 360 px output, for clean edges)
PALM_POLY = [0, 1, 5, 9, 13, 17]
HULL_DILATE = 0.30  # x hand size: margin around the landmarks' convex hull
CORRIDOR_HALF_WIDTH = (1.0, 1.4)  # x hand size at the wrist / far end: wider than any forearm (skin + person
# masks give the real edge; the corridor only keeps the body and face out)
SKIN_SIGMAS = 6.0  # Mahalanobis distance in Lab (a, b): beyond this it is never skin (sleeve, clothes)
SMALL_HOLE = 0.35  # x hand size, squared: enclosed holes smaller than this are filled (highlights, nails);
# bigger ones are real gaps between fingers (e.g. the loop in ठ)
U2NET_HOME = config.MODELS_DIR / "u2net"  # rembg's model download stays inside the (git-ignored) project folder


def crop_square(frame: np.ndarray, x0: float, y0: float, side: float, size: int = WORK) -> np.ndarray:
    """The square (x0, y0, side) of the frame resized to ``size``; outside the frame is white."""
    s = size / side
    m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], dtype=np.float64)
    return cv2.warpAffine(
        frame, m, (size, size), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


def to_crop(points: np.ndarray, x0: float, y0: float, side: float, size: int = WORK) -> np.ndarray:
    """Frame coordinates -> crop pixel coordinates (x, y only)."""
    p = np.asarray(points, dtype=np.float64)[..., :2]
    return (p - [x0, y0]) * (size / side)


# --- skin colour -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourModel:
    mean: np.ndarray  # (2,) Lab a, b
    inv_cov: np.ndarray  # (2, 2)

    def d2(self, rgb: np.ndarray) -> np.ndarray:
        ab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32) - self.mean
        return np.einsum("...i,ij,...j->...", ab, self.inv_cov, ab)


@dataclass(frozen=True)
class SkinModel:
    """Skin vs. background colour for one clip (chroma only, so shadows on the hand still count as skin).

    A pixel is skin when it is closer to this clip's skin than to its wall, and not far from skin at all. A pink
    thumb or a shadowed knuckle is still nearer to skin than to the beige wall; the blue sleeve and its white
    flowers are not.
    """

    skin: ColourModel
    wall: ColourModel | None = None
    sigmas: float = SKIN_SIGMAS

    @property
    def mean(self) -> np.ndarray:
        return self.skin.mean

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        ds = self.skin.d2(rgb)
        ok = ds < self.sigmas**2
        if self.wall is not None:
            ok &= ds < self.wall.d2(rgb)
        return ok


def _fit_colour(ab: np.ndarray) -> ColourModel:
    ab = ab.astype(np.float64)
    cov = np.cov(ab.T) + np.eye(2) * 4.0  # floor: never narrower than ~2 Lab units
    return ColourModel(ab.mean(axis=0).astype(np.float32), np.linalg.inv(cov).astype(np.float32))


def palm_mask(shape: tuple[int, int], pose_xy: np.ndarray, shrink: float = 0.4) -> np.ndarray:
    """Inside of the palm polygon, shrunk towards its centre (so samples stay on skin, away from edges)."""
    pts = pose_xy[PALM_POLY]
    centre = pts.mean(axis=0)
    poly = (centre + (pts - centre) * (1 - shrink)).round().astype(np.int32)
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def fit_skin(crops: list[np.ndarray], poses_xy: list[np.ndarray], people: list[np.ndarray] | None = None) -> SkinModel:
    """Skin colour of this clip from pixels inside the palm in every frame; wall colour from pixels the person
    segmentation calls background (when ``people`` is given)."""
    skin, wall = [], []
    for i, (rgb, xy) in enumerate(zip(crops, poses_xy, strict=True)):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        inside = palm_mask(rgb.shape[:2], xy)
        if inside.any():
            skin.append(lab[inside][:, 1:])
        if people is not None:
            bg = (np.asarray(people[i]) < 0.2) & (rgb.min(axis=2) < 250)  # not the white padding outside the frame
            if bg.sum() > 500:
                wall.append(lab[bg][:, 1:][:: max(1, int(bg.sum()) // 5000)])
    if not skin:
        raise RuntimeError("no palm pixels to calibrate skin colour")
    return SkinModel(_fit_colour(np.concatenate(skin)), _fit_colour(np.concatenate(wall)) if wall else None)


# --- region around the hand -------------------------------------------------------------------------------------


def hand_region(shape: tuple[int, int], pose_xy: np.ndarray, forearm_dir: np.ndarray, size_px: float) -> np.ndarray:
    """Dilated convex hull of the landmarks + a corridor from the wrist along the forearm (to the crop edge)."""
    h, w = shape
    m = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(pose_xy.round().astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    k = max(3, int(round(2 * HULL_DILATE * size_px)) | 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    d = np.asarray(forearm_dir, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    n = np.array([-d[1], d[0]])
    wrist = pose_xy[0]
    far = wrist + d * 2 * (h + w)
    w0, w1 = CORRIDOR_HALF_WIDTH[0] * size_px, CORRIDOR_HALF_WIDTH[1] * size_px
    corridor = np.array([wrist + n * w0, far + n * w1, far - n * w1, wrist - n * w0]).round().astype(np.int32)
    cv2.fillConvexPoly(m, corridor, 1)
    return m.astype(bool)


# --- combining, cleaning, stabilising ---------------------------------------------------------------------------


def _keep_hand_component(m: np.ndarray, pose_xy: np.ndarray) -> np.ndarray:
    """The single connected region that contains the hand (most palm/landmark pixels)."""
    n, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    h, w = m.shape
    pts = pose_xy.round().astype(int)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    votes = np.bincount(labels[pts[:, 1], pts[:, 0]], minlength=n) if len(pts) else np.zeros(n)
    votes[0] = 0
    if votes.max() == 0:  # no landmark on the mask: fall back to the biggest region
        votes = np.bincount(labels.ravel(), minlength=n)
        votes[0] = 0
    return labels == int(np.argmax(votes))


def _fill_small_holes(m: np.ndarray, max_area: float) -> np.ndarray:
    inv = (~m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m.copy()
    h, w = m.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area <= max_area:
            out[labels == i] = True
    return out


def clean_mask(m: np.ndarray, pose_xy: np.ndarray, size_px: float) -> np.ndarray:
    """Open/close away specks, keep the hand's connected region, fill small holes."""
    k = max(3, int(round(0.04 * size_px)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, kernel)
    out = _keep_hand_component(u.astype(bool), pose_xy)
    return _fill_small_holes(out, (SMALL_HOLE * size_px) ** 2)


def hand_mask(
    rgb: np.ndarray,
    person: np.ndarray,
    pose_xy: np.ndarray,
    forearm_dir: np.ndarray,
    size_px: float,
    skin: SkinModel,
) -> np.ndarray:
    """Binary hand mask for one crop. ``person`` is rembg's foreground (bool or 0..1)."""
    person_b = np.asarray(person) > 0.5 if np.asarray(person).dtype != bool else np.asarray(person)
    m = person_b & hand_region(rgb.shape[:2], pose_xy, forearm_dir, size_px) & skin.mask(rgb)
    return clean_mask(m, pose_xy, size_px)


def temporal_median(masks: list[np.ndarray], window: int = 3) -> list[np.ndarray]:
    """Per-pixel majority over ``window`` neighbouring frames, so the outline doesn't flicker.

    At the clip's ends the window shifts inwards (frames 0-2 for frame 0) so every vote has ``window`` frames; a
    2-frame window would turn "majority" into "either frame" and grow the first/last mask.
    """
    n = len(masks)
    if n < window:
        return [m.copy() for m in masks]
    stack = np.stack(masks).astype(np.uint8)
    half = window // 2
    out = []
    for i in range(n):
        lo = min(max(0, i - half), n - window)
        out.append(stack[lo : lo + window].sum(axis=0) * 2 > window)
    return out


def smooth_alpha(m: np.ndarray, size_px: float) -> np.ndarray:
    """Smooth the mask's edge (blur + re-threshold), then a 1 px soft edge: float alpha 0..1."""
    sigma = max(1.0, 0.025 * size_px)
    blurred = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    hard = (blurred >= 0.5).astype(np.float32)
    return cv2.GaussianBlur(hard, (0, 0), 0.8)


# --- rembg ------------------------------------------------------------------------------------------------------


def person_masks(crops: list[np.ndarray]) -> list[np.ndarray]:
    """rembg (u2net) foreground per crop, as float 0..1."""
    os.environ.setdefault("U2NET_HOME", str(U2NET_HOME))
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError("the toon/photo styles need rembg: run `uv add rembg[cpu]` in ml/") from exc
    from PIL import Image

    session = new_session("u2net")
    out = []
    for rgb in crops:
        mask = remove(Image.fromarray(rgb), session=session, only_mask=True)
        out.append(np.asarray(mask, dtype=np.float32) / 255.0)
    return out


def hand_only(m: np.ndarray, pose_xy: np.ndarray, size_px: float) -> np.ndarray:
    """The mask without the forearm: only what lies near the landmarks (for framing the hand)."""
    near = np.zeros(m.shape, dtype=np.uint8)
    hull = cv2.convexHull(pose_xy.round().astype(np.int32))
    cv2.fillConvexPoly(near, hull, 1)
    k = max(3, int(round(2 * HULL_DILATE * size_px)) | 1)
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m & near.astype(bool)


def mask_bbox(m: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
