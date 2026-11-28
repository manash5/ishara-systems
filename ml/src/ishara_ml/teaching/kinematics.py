"""Hand kinematics for the key-pose animation: palm frame, joint rotations, one fixed skeleton, forward kinematics.

Coordinates are MediaPipe world landmarks: metres, axes aligned with the camera (x right, y down, z away from the
camera), which is a right-handed system. Quaternions are (w, x, y, z) numpy arrays.

Palm frame (from points 0, 5, 9, 17): origin = wrist, y = wrist -> middle knuckle, z = palm normal
(p5 - p0) x (p17 - p0) made orthogonal to y (points out of the palm side, the side fingers curl towards), x = y x z.
Points 0, 1, 5, 9, 13, 17 are rigid in this frame (the "palm"). The thumb (1-2-3-4) and every finger
(MCP-PIP-DIP-TIP) are 3-bone chains. Each bone stores a local swing quaternion relative to its parent: the
parent's frame rotated so the bone's rest direction points along the bone. The rest direction is the same vector
for every bone of a chain (the chain's straight continuation), so a pose with all-identity bones is a flat hand.

Editable form: each bone's swing as (flex, abduct) degrees - flex bends towards the palm side, abduct sideways.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ishara_ml.teaching.geometry import FINGERS, SIZE_BONES

CHAINS: dict[str, list[int]] = dict(FINGERS)  # thumb [1,2,3,4], index [5,6,7,8], ...
CHAIN_NAMES = tuple(CHAINS)
PALM_POINTS = (0, 1, 5, 9, 13, 17)
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
FOREARM_OF_PALM = 0.85  # forearm width at the wrist = this x the palm width (index knuckle -> pinky knuckle)

# --- quaternions --------------------------------------------------------------------------------------------


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return IDENTITY.copy() if n < 1e-12 else q / n


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])


def q_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def q_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion with w >= 0 (Shepperd's method)."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = q_normalize(np.array(q))
    return q if q[0] >= 0 else -q


def q_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking direction u onto direction v."""
    u = np.asarray(u, dtype=np.float64) / max(float(np.linalg.norm(u)), 1e-12)
    v = np.asarray(v, dtype=np.float64) / max(float(np.linalg.norm(v)), 1e-12)
    d = float(np.clip(u @ v, -1.0, 1.0))
    if d < -1 + 1e-9:  # opposite: rotate 180 degrees about any axis perpendicular to u
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        return q_from_axis_angle(axis, math.pi)
    return q_normalize(np.concatenate([[1.0 + d], np.cross(u, v)]))


def q_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle (radians) between two orientations."""
    d = abs(float(np.clip(q_normalize(a) @ q_normalize(b), -1.0, 1.0)))
    return 2 * math.acos(min(d, 1.0))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation along the shorter arc (b's sign is flipped when needed)."""
    a, b = q_normalize(a), q_normalize(b)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:  # nearly equal: lerp + normalise is exact enough and avoids dividing by ~0
        return q_normalize(a + t * (b - a))
    theta = math.acos(d)
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / s


def q_mean(qs: np.ndarray) -> np.ndarray:
    """Mean orientation of nearby quaternions: sign-align to the first, average, normalise."""
    qs = np.asarray(qs, dtype=np.float64)
    ref = qs[0]
    aligned = np.where((qs @ ref)[:, None] < 0, -qs, qs)
    return q_normalize(aligned.mean(axis=0))


# --- palm frame ----------------------------------------------------------------------------------------------


def palm_frame(points: np.ndarray) -> np.ndarray:
    """3x3 rotation whose columns are the palm frame's x, y, z axes in camera coordinates."""
    p = np.asarray(points, dtype=np.float64)
    y = p[9] - p[0]
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    n = np.cross(p[5] - p[0], p[17] - p[0])
    x = np.cross(y, n)
    x = x / max(float(np.linalg.norm(x)), 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def to_palm_local(points: np.ndarray) -> np.ndarray:
    """Points expressed in their own palm frame (wrist at the origin)."""
    p = np.asarray(points, dtype=np.float64)
    return (p - p[0]) @ palm_frame(p)


# --- skeleton ------------------------------------------------------------------------------------------------


@dataclass
class Skeleton:
    """One fixed hand: rigid palm points + bone lengths + rest directions, in palm-frame metres."""

    id: str
    palm: dict[int, np.ndarray]  # PALM_POINTS -> local position
    lengths: dict[str, np.ndarray]  # chain -> (3,) bone lengths
    rest: dict[str, np.ndarray]  # chain -> unit rest direction (local)
    forearm_width: float  # metres
    frames_used: int = 0
    letters: list[str] = field(default_factory=list)

    @property
    def palm_width(self) -> float:
        return float(np.linalg.norm(self.palm[5] - self.palm[17]))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "units": "metres, palm frame (origin wrist, y wrist->middle knuckle, z palm normal, x = y cross z)",
            "palm": {str(k): np.round(v, 6).tolist() for k, v in self.palm.items()},
            "boneLengths": {k: np.round(v, 6).tolist() for k, v in self.lengths.items()},
            "rest": {k: np.round(v, 6).tolist() for k, v in self.rest.items()},
            "forearmWidth": round(self.forearm_width, 6),
            "framesUsed": self.frames_used,
            "letters": self.letters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skeleton:
        return cls(
            id=d["id"],
            palm={int(k): np.array(v, dtype=np.float64) for k, v in d["palm"].items()},
            lengths={k: np.array(v, dtype=np.float64) for k, v in d["boneLengths"].items()},
            rest={k: np.array(v, dtype=np.float64) for k, v in d["rest"].items()},
            forearm_width=float(d["forearmWidth"]),
            frames_used=int(d.get("framesUsed", 0)),
            letters=list(d.get("letters", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Skeleton:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def hand_size(self) -> float:
        """``geometry.hand_size`` of this skeleton (constant whatever the pose: palm + proximal bones only)."""
        return float(np.mean([np.linalg.norm(p[a] - p[b]) for p in [flat_points(self)] for a, b in SIZE_BONES]))


def build_skeleton(world_frames: list[np.ndarray], skeleton_id: str, letters: list[str] | None = None) -> Skeleton:
    """Median palm shape and bone lengths over many frames (any poses) of one signer.

    The median palm is re-expressed in its own palm frame, so the skeleton's palm frame is exactly the identity.
    """
    if not world_frames:
        raise ValueError("no frames")
    local = np.stack([to_palm_local(p) for p in world_frames])
    med = np.median(local, axis=0)
    palm_pts = to_palm_local(med)  # re-canonicalise the median palm
    palm = {i: palm_pts[i] for i in PALM_POINTS}
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in world_frames])
    lengths = {
        name: np.array([np.median(np.linalg.norm(stack[:, c[k + 1]] - stack[:, c[k]], axis=1)) for k in range(3)])
        for name, c in CHAINS.items()
    }
    rest = {}
    for name, c in CHAINS.items():
        d = palm[c[0]] - palm[0]  # straight continuation of the metacarpal (wrist -> knuckle / wrist -> thumb base)
        rest[name] = d / max(float(np.linalg.norm(d)), 1e-12)
    width = FOREARM_OF_PALM * float(np.linalg.norm(palm[5] - palm[17]))
    return Skeleton(skeleton_id, palm, lengths, rest, width, len(world_frames), sorted(letters or []))


# --- poses ---------------------------------------------------------------------------------------------------


@dataclass
class HandPose:
    """Wrist orientation (palm frame in camera coords) + wrist position (m) + per-bone local swings."""

    wrist_q: np.ndarray
    wrist_pos: np.ndarray
    bones: dict[str, np.ndarray]  # chain -> (3, 4) local quaternions

    def copy(self) -> HandPose:
        return HandPose(self.wrist_q.copy(), self.wrist_pos.copy(), {k: v.copy() for k, v in self.bones.items()})


def pose_to_rotations(points: np.ndarray, skeleton: Skeleton) -> HandPose:
    """Joint rotations of a 21-point hand (camera coords). Only bone *directions* are used, so the pose can be
    rebuilt on any skeleton."""
    p = np.asarray(points, dtype=np.float64)
    r_palm = palm_frame(p)
    bones = {}
    for name, c in CHAINS.items():
        rest = skeleton.rest[name]
        frame = r_palm
        qs = []
        for k in range(3):
            d = p[c[k + 1]] - p[c[k]]
            q = q_between(rest, frame.T @ d)
            qs.append(q)
            frame = frame @ q_to_matrix(q)
        bones[name] = np.stack(qs)
    return HandPose(q_from_matrix(r_palm), p[0].copy(), bones)


def forward_kinematics(pose: HandPose, skeleton: Skeleton) -> np.ndarray:
    """(21, 3) camera-coordinate points of ``pose`` built on ``skeleton``: every bone has its skeleton length."""
    r_palm = q_to_matrix(pose.wrist_q)
    pts = np.zeros((21, 3))
    for i in PALM_POINTS:
        pts[i] = pose.wrist_pos + r_palm @ skeleton.palm[i]
    for name, c in CHAINS.items():
        frame = r_palm
        rest = skeleton.rest[name]
        for k in range(3):
            frame = frame @ q_to_matrix(pose.bones[name][k])
            pts[c[k + 1]] = pts[c[k]] + frame @ rest * skeleton.lengths[name][k]
    return pts


def flat_points(skeleton: Skeleton) -> np.ndarray:
    """The skeleton's rest pose (all bones straight) in palm coordinates."""
    rest = HandPose(IDENTITY.copy(), np.zeros(3), {n: np.tile(IDENTITY, (3, 1)) for n in CHAINS})
    return forward_kinematics(rest, skeleton)


def interpolate(a: HandPose, b: HandPose, t: float) -> HandPose:
    """Slerp every joint and the wrist orientation, lerp the wrist position. Never interpolates points."""
    return HandPose(
        slerp(a.wrist_q, b.wrist_q, t),
        (1 - t) * a.wrist_pos + t * b.wrist_pos,
        {n: np.stack([slerp(a.bones[n][k], b.bones[n][k], t) for k in range(3)]) for n in CHAINS},
    )


# --- editable angles -----------------------------------------------------------------------------------------


def _basis(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flex, abduct) unit axes for a bone with this rest direction: flex leans towards the palm side (+z)."""
    z = np.array([0.0, 0.0, 1.0])
    e_f = z - (z @ rest) * rest
    e_f = e_f / max(float(np.linalg.norm(e_f)), 1e-12)
    return e_f, np.cross(rest, e_f)


def swing_to_angles(q: np.ndarray, rest: np.ndarray) -> tuple[float, float]:
    """Local swing -> (flex, abduct) degrees."""
    v = q_to_matrix(q) @ rest
    e_f, e_a = _basis(rest)
    abd = math.asin(float(np.clip(v @ e_a, -1.0, 1.0)))
    flex = math.atan2(float(v @ e_f), float(v @ rest))
    return math.degrees(flex), math.degrees(abd)


def angles_to_swing(flex_deg: float, abduct_deg: float, rest: np.ndarray) -> np.ndarray:
    """(flex, abduct) degrees -> the shortest-arc swing from ``rest`` to that direction."""
    f, a = math.radians(flex_deg), math.radians(abduct_deg)
    e_f, e_a = _basis(rest)
    v = math.cos(a) * (math.cos(f) * rest + math.sin(f) * e_f) + math.sin(a) * e_a
    return q_between(rest, v)


def pose_from_angles(
    joints: dict[str, list[list[float]]], wrist_q: np.ndarray, wrist_pos: np.ndarray, skeleton: Skeleton
) -> HandPose:
    bones = {n: np.stack([angles_to_swing(f, a, skeleton.rest[n]) for f, a in joints[n]]) for n in CHAINS}
    return HandPose(q_normalize(wrist_q), np.asarray(wrist_pos, dtype=np.float64), bones)


def pose_angles(pose: HandPose, skeleton: Skeleton) -> dict[str, list[list[float]]]:
    return {n: [[round(v, 1) for v in swing_to_angles(q, skeleton.rest[n])] for q in pose.bones[n]] for n in CHAINS}
