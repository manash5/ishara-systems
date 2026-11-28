"""Letters with movement: the HANDSHAPE comes only from sharp, confident, slow frames; the MOVEMENT is the wrist path
and orientation over time, smoothed strongly, reduced to a few control points and replayed with a spline (position,
forearm angle) and slerp (orientation). Static letters never go through this module.

Why: during the movement NSL23 frames are motion-blurred (u: Laplacian variance falls from ~290 to ~20), so the
landmarks of those frames are wrong; used as key poses they gave squashed palms and detached fingers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from ishara_ml.teaching.kinematics import (
    CHAINS,
    PALM_POINTS,
    HandPose,
    Skeleton,
    palm_frame,
    q_from_matrix,
    q_normalize,
    slerp,
)

# --- sharp-frame selection (handshape) ---
SHARP_MAX_SPEED = 2.0  # hand sizes/s: faster frames are motion-blurred
SHARP_MIN_SCORE = 0.8  # MediaPipe hand confidence
SHARP_REL = 0.5  # sharpness must reach this x the clip's 90th percentile ...
SHARP_ABS_MIN = 15.0  # ... and this absolute Laplacian variance (256 px hand crop; motion-blurred NSL23 frames: 6-9)
MIN_SHARP_FRAMES = 3
SHAPE_CHANGE = 0.50  # hand sizes: a second handshape only if some joint's median position moves this much ...
SHAPE_CONSISTENT = 0.5  # ... and each group's spread is below this x the change
MIN_GROUP = 3
# --- movement path ---
PATH_SMOOTH_S = 0.5  # Savitzky-Golay window for the wrist path and the arm
RELIABLE_MAX_SPEED = 4.0  # orientation samples: frames slower than this ...
RELIABLE_MIN_SCORE = 0.5  # ... and at least this confident
ORIENT_SIGMA_S = 0.12  # Gaussian weight window around each control point
CONTROL_EVERY_S = 0.25  # one control point per this much movement, 3-6 in total
MIN_CONTROLS, MAX_CONTROLS = 3, 6
# --- drawing / checks ---
MAX_WRIST_BEND_DEG = 70.0  # the forearm is at least 180 - this degrees from the finger direction (never along it)
MOVE_FIT = 0.92  # the hand plus its drawn thickness fills at most this fraction of the view
JOINT_MARGIN = 0.26  # x hand size: drawn finger radius + outline around every joint
MIN_PALM_AREA = 0.15  # visible palm area >= this x the palm seen face-on
# --- drag style ---
# A letter whose wrist really travels (a "drag", like u and o) is drawn with a short fading forearm and the natural
# hand drawing, so the eye follows the hand. Measured on the replayed path: u 2.64, o 3.75, am 1.43 hand sizes.
DRAG_MIN_TRAVEL = 2.0  # skeleton hand sizes of net wrist travel


class MovementCheckError(RuntimeError):
    """A rendered frame failed a sanity check."""


def hand_sharpness(frame: np.ndarray, pts: np.ndarray, out: int = 256) -> float:
    """Variance of the Laplacian of the grey hand crop (square, 1.4 x the hand), resized to ``out`` px."""
    import cv2

    h, w = frame.shape[:2]
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    side = max(float((hi - lo).max()) * 1.4, 16.0)
    cx, cy = (lo + hi) / 2
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    grey = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
    grey = cv2.resize(grey, (out, out), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def select_sharp(
    speed: np.ndarray, score: np.ndarray, sharp: np.ndarray, present: np.ndarray
) -> tuple[np.ndarray, str]:
    """Mask of frames usable for the handshape: slow, confident and sharp. Relaxes step by step (and says so) when
    fewer than ``MIN_SHARP_FRAMES`` qualify."""
    valid = present & np.isfinite(sharp)
    if not valid.any():
        return np.zeros(len(speed), dtype=bool), "no detected frames"
    thr = max(SHARP_ABS_MIN, SHARP_REL * float(np.percentile(sharp[valid], 90)))
    tiers = (
        (SHARP_MAX_SPEED, SHARP_MIN_SCORE, thr, "strict"),
        (3.0, 0.6, thr, "relaxed speed/confidence"),
        (3.0, 0.5, 0.7 * thr, "relaxed sharpness"),
    )
    for max_speed, min_score, min_sharp, label in tiers:
        m = valid & (speed < max_speed) & (score >= min_score) & (sharp >= min_sharp)
        if m.sum() >= MIN_SHARP_FRAMES:
            return m, f"{label}: sharpness >= {min_sharp:.0f}, speed < {max_speed}, confidence >= {min_score}"
    # No sharp slow frames (o: every frame of its end hold is soft): the slowest confident frames, sharpest half.
    slow = valid & (score >= 0.5) & (speed < SHARP_MAX_SPEED)
    if slow.sum() < MIN_SHARP_FRAMES:
        slow = valid & (speed <= np.quantile(speed[valid], 0.3))
    m = slow & (sharp >= np.median(sharp[slow]))
    if m.sum() < MIN_SHARP_FRAMES:
        m = slow
    return m, f"fallback: no sharp slow frames, used the sharper half of the slowest frames ({int(m.sum())})"


def _joint_change(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-joint distance between two shapes (one finger bending counts fully)."""
    return (
        float(np.linalg.norm(a - b, axis=-1).max(axis=-1).mean())
        if a.ndim == 3
        else float(np.linalg.norm(a - b, axis=-1).max())
    )


def split_handshapes(local: np.ndarray, size: float) -> int | None:
    """Index splitting time-ordered sharp shapes (palm-local) into two clearly different, consistent groups, or
    None when one handshape explains them. Change = the largest per-joint move between the two groups' medians;
    spread = each frame's largest per-joint distance to its own group's median, averaged. Among the splits with
    change > SHAPE_CHANGE and spread < SHAPE_CONSISTENT x change, the one with the smallest spread wins."""
    n = len(local)
    if n < 2 * MIN_GROUP:
        return None
    best, best_spread = None, float("inf")
    for k in range(MIN_GROUP, n - MIN_GROUP + 1):
        ma, mb = np.median(local[:k], axis=0), np.median(local[k:], axis=0)
        d = _joint_change(ma, mb) / size
        sa, sb = _joint_change(local[:k], ma) / size, _joint_change(local[k:], mb) / size
        spread = (k * sa + (n - k) * sb) / n
        if d > SHAPE_CHANGE and max(sa, sb) < SHAPE_CONSISTENT * d and spread < best_spread:
            best, best_spread = k, spread
    return best


@dataclass
class PathPoint:
    t: float  # seconds from the start of the movement
    frame: float  # clip frame (for the check sheet)
    pos: np.ndarray  # wrist position, metres from the movement's start
    q: np.ndarray  # wrist (palm frame) orientation
    forearm_deg: float  # image angle wrist -> elbow
    shape: float  # 0 = first handshape, 1 = second


def control_frames(sa: int, sb: int, fps: float) -> np.ndarray:
    dur = max(sb - 1 - sa, 1) / fps
    k = int(np.clip(1 + round(dur / CONTROL_EVERY_S), MIN_CONTROLS, MAX_CONTROLS))
    return np.linspace(sa, sb - 1, k)


def weighted_orientation(qs: np.ndarray, frames: np.ndarray, weights: np.ndarray, at: float, fps: float) -> np.ndarray:
    """Gaussian-weighted mean orientation around frame ``at`` (sign-aligned quaternion average)."""
    sigma = ORIENT_SIGMA_S * fps
    for _ in range(5):
        w = weights * np.exp(-0.5 * ((frames - at) / sigma) ** 2)
        if w.sum() >= 0.5:
            break
        sigma *= 2
    ref = qs[int(np.argmax(w))]
    aligned = np.where((qs @ ref)[:, None] < 0, -qs, qs)
    return q_normalize((aligned * w[:, None]).sum(axis=0))


def arm_angles(arm: np.ndarray, fps: float, smooth) -> np.ndarray | None:
    """Per-frame image angle (degrees, unwrapped) of the Pose elbow seen from the Pose wrist, smoothed."""
    known = ~np.isnan(arm[:, 0])
    if known.sum() < 2:
        return None
    filled = arm.copy()
    idx = np.where(known)[0]
    for c in range(4):
        filled[:, c] = np.interp(np.arange(len(arm)), idx, arm[idx, c])
    filled = smooth(filled, fps, PATH_SMOOTH_S)
    vec = filled[:, :2] - filled[:, 2:]
    return np.degrees(np.unwrap(np.arctan2(vec[:, 1], vec[:, 0])))


# --- playback ---------------------------------------------------------------------------------------------------


class PathSampler:
    """Continuous movement: position + forearm angle by clamped cubic spline (zero velocity at both ends),
    orientation by slerp between control points, handshape by slerp between the two handshapes' joints."""

    def __init__(self, points: list[PathPoint], shapes: list[HandPose]):
        self.points = points
        self.shapes = shapes
        self.times = np.array([p.t for p in points])
        if len(points) == 1:
            self.times = np.array([0.0, 1e-3])
            points = [points[0], points[0]]
        pos = np.stack([p.pos for p in points])
        ang = np.array([p.forearm_deg for p in points], dtype=float)
        ang = ang[0] + np.concatenate([[0.0], np.cumsum((np.diff(ang) + 180) % 360 - 180)])
        self._pos = CubicSpline(self.times, pos, bc_type="clamped")
        self._ang = CubicSpline(self.times, ang, bc_type="clamped")
        self._qs = [p.q for p in points]
        self._shape = np.array([p.shape for p in points])

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def __call__(self, t: float) -> tuple[HandPose, float]:
        t = float(np.clip(t, self.times[0], self.times[-1]))
        i = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2))
        u = (t - self.times[i]) / max(self.times[i + 1] - self.times[i], 1e-9)
        q = slerp(self._qs[i], self._qs[i + 1], u)
        blend = float(np.interp(t, self.times, self._shape))
        a = self.shapes[0]
        b = self.shapes[-1]
        bones = {n: np.stack([slerp(a.bones[n][k], b.bones[n][k], blend) for k in range(3)]) for n in CHAINS}
        return HandPose(q, np.asarray(self._pos(t), dtype=np.float64), bones), float(self._ang(t))


def wrist_travel(sampler: PathSampler, skeleton: Skeleton, samples: int = 200) -> float:
    """Net wrist travel of the replayed movement, in skeleton hand sizes."""
    ts = np.linspace(0.0, sampler.duration, samples)
    start, end = sampler(float(ts[0]))[0].wrist_pos, sampler(float(ts[-1]))[0].wrist_pos
    return float(np.linalg.norm(end - start)) / skeleton.hand_size()


def is_drag(sampler: PathSampler, skeleton: Skeleton) -> bool:
    """True for a dragging movement (the wrist travels at least DRAG_MIN_TRAVEL hand sizes)."""
    return wrist_travel(sampler, skeleton) >= DRAG_MIN_TRAVEL


# --- framing, forearm, checks -----------------------------------------------------------------------------------


def joint_margin(skeleton: Skeleton) -> float:
    return JOINT_MARGIN * skeleton.hand_size()


def movement_view(points: list[np.ndarray], skeleton: Skeleton, base_side: float) -> tuple[np.ndarray, float]:
    """(centre, side) of one fixed square view holding every frame's hand plus its drawn thickness, with padding.
    ``side`` stays ``base_side`` (the static letters' scale) when that fits, else grows as little as needed."""
    r = joint_margin(skeleton)
    xy = np.concatenate([p[:, :2] for p in points])
    lo, hi = xy.min(axis=0) - r, xy.max(axis=0) + r
    extent = float((hi - lo).max())
    return (lo + hi) / 2, max(base_side, extent / MOVE_FIT)


def constrain_forearm(points: np.ndarray, away: np.ndarray) -> tuple[np.ndarray, bool]:
    """Keep the forearm at least 180 - MAX_WRIST_BEND_DEG degrees from the fingers (2-D), so it never points along
    them. Skipped when the fingers point at/away from the camera (their 2-D direction is meaningless)."""
    f = points[9, :2] - points[0, :2]
    f3 = float(np.linalg.norm(points[9] - points[0]))
    nf = float(np.linalg.norm(f))
    if nf < 0.35 * f3:
        return away, False
    back = -f / nf
    ang = math.atan2(float(back[0] * away[1] - back[1] * away[0]), float(back @ away))
    limit = math.radians(MAX_WRIST_BEND_DEG)
    if abs(ang) <= limit:
        return away, False
    a = math.copysign(limit, ang)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * back[0] - s * back[1], s * back[0] + c * back[1]]), True


def _area(xy: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    try:
        return float(ConvexHull(xy).volume)
    except Exception:  # degenerate (collinear) hull
        return 0.0


def face_on_palm_area(skeleton: Skeleton) -> float:
    return _area(np.stack([skeleton.palm[i][:2] for i in PALM_POINTS]))


def check_frame(i: int, points: np.ndarray, unit: np.ndarray, skeleton: Skeleton, side: float) -> None:
    """Raise MovementCheckError (with the frame number) if a rendered frame is broken."""
    r = joint_margin(skeleton) / side
    xy = unit[:, :2]
    if (xy < r - 1e-9).any() or (xy > 1 - r + 1e-9).any():
        raise MovementCheckError(f"frame {i}: hand leaves the frame (joint at {xy.min():.3f}..{xy.max():.3f})")
    for name, c in CHAINS.items():
        for k in range(3):
            length = float(np.linalg.norm(points[c[k + 1]] - points[c[k]]))
            if abs(length - skeleton.lengths[name][k]) > 1e-6:
                want = skeleton.lengths[name][k]
                raise MovementCheckError(f"frame {i}: {name} bone {k} is {length:.5f} m, skeleton {want:.5f}")
    rot = palm_frame(points)
    for p in PALM_POINTS:
        drift = float(np.linalg.norm(points[p] - (points[0] + rot @ skeleton.palm[p])))
        if drift > 1e-6:  # finger bases (knuckles / thumb base) must sit on the rigid palm
            raise MovementCheckError(f"frame {i}: point {p} detached from the palm by {drift * 1000:.2f} mm")
    area = _area(points[list(PALM_POINTS), :2]) / max(face_on_palm_area(skeleton), 1e-12)
    if area < MIN_PALM_AREA:
        raise MovementCheckError(f"frame {i}: visible palm area {area:.2f} x face-on < {MIN_PALM_AREA}")


def palm_quat(points: np.ndarray) -> np.ndarray:
    return q_from_matrix(palm_frame(points))
