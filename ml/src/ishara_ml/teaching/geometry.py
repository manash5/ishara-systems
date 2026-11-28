"""Pure 3-D hand geometry used by the renderer and the pipeline.

Poses are (21, 3) arrays with x, y, z on one common scale (pixels or unit-box); smaller z = nearer the camera.
"""

from __future__ import annotations

import numpy as np

FINGERS: dict[str, list[int]] = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
# Bones that barely change length in 3-D whatever the hand does (palm frame + proximal phalanges).
SIZE_BONES = [(0, 5), (0, 17), (5, 17), (0, 9), (5, 6), (9, 10), (13, 14), (17, 18)]
EXTENDED_THRESHOLD = 0.85


def hand_size(poses: list[np.ndarray] | np.ndarray) -> float:
    """One size per clip: median over frames of the mean 3-D length of ``SIZE_BONES``.

    3-D lengths don't shrink when the hand turns, so widths derived from this stay steady.
    """
    stack = np.asarray(poses, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[None]
    lengths = np.stack([np.linalg.norm(stack[:, a] - stack[:, b], axis=1) for a, b in SIZE_BONES], axis=1)
    return float(np.median(lengths.mean(axis=1)))


def straightness(pose: np.ndarray, chain: list[int]) -> float:
    """|tip - base| / sum of segment lengths, in 3-D: 1.0 = straight, lower = curled."""
    pts = np.asarray(pose, dtype=np.float64)[chain]
    total = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return float(np.linalg.norm(pts[-1] - pts[0]) / total) if total > 1e-9 else 1.0


def classify_fingers(
    poses: list[np.ndarray], threshold: float = EXTENDED_THRESHOLD, window: int = 5
) -> list[dict[str, bool]]:
    """Per frame, whether each finger is extended, from 3-D straightness moving-averaged over ``window`` frames."""
    if not poses:
        return []
    raw = {name: np.array([straightness(p, chain) for p in poses]) for name, chain in FINGERS.items()}
    half = window // 2
    out: list[dict[str, bool]] = []
    for i in range(len(poses)):
        lo, hi = max(0, i - half), min(len(poses), i + half + 1)
        out.append({name: bool(values[lo:hi].mean() >= threshold) for name, values in raw.items()})
    return out


PALM_CENTRE = [0, 1, 5, 9, 13, 17]
RAISED_THRESHOLD = 1.0  # tip-to-palm-centre distance (hand sizes) above which a finger is drawn on its own


def tip_reach(pose: np.ndarray, chain: list[int], size: float) -> float:
    """3-D distance from the fingertip to the palm centre, in hand sizes.

    Folded fingers (in a fist) end near the palm (<= ~0.9 on NSL23); raised ones - straight *or* hooked, like
    ah's bent index and middle - end farther away (>= ~1.05). Straightness alone misses hooked fingers.
    """
    p = np.asarray(pose, dtype=np.float64)
    return float(np.linalg.norm(p[chain[-1]] - p[PALM_CENTRE].mean(axis=0)) / max(size, 1e-9))


def finger_states(
    poses: list[np.ndarray],
    static: bool,
    window: int = 5,
    threshold: float = RAISED_THRESHOLD,
    overrides: dict[str, bool] | None = None,
) -> list[dict[str, bool]]:
    """Per frame, True if a (non-thumb) finger is raised (drawn on its own, on top) or False if folded into the fist.

    Static signs get one decision for the whole clip (the hand shape doesn't change, so fingers mustn't flicker
    between states); movement signs are decided per frame on a ``window``-frame average. ``overrides`` forces a
    finger's state where the landmark detector is known to be wrong. The thumb is always drawn on its own.
    """
    if not poses:
        return []
    size = hand_size(poses)
    names = [n for n in FINGERS if n != "thumb"]
    reach = {n: np.array([tip_reach(p, FINGERS[n], size) for p in poses]) for n in names}
    out: list[dict[str, bool]] = []
    half = window // 2
    for i in range(len(poses)):
        state = {"thumb": True}
        for n in names:
            vals = reach[n] if static else reach[n][max(0, i - half) : i + half + 1]
            state[n] = bool(np.median(vals) >= threshold)
        if overrides:
            state.update({k: v for k, v in overrides.items() if k != "thumb"})
        out.append(state)
    return out


def palm_normal_z(pose: np.ndarray) -> float:
    """z component of the unit normal (p5 - p0) x (p17 - p0)."""
    p = np.asarray(pose, dtype=np.float64)
    n = np.cross(p[5] - p[0], p[17] - p[0])
    norm = float(np.linalg.norm(n))
    return float(n[2] / norm) if norm > 1e-9 else 0.0


def palm_facing(pose: np.ndarray, handedness: str = "right", margin: float = 0.35) -> int:
    """+1 if the back of the hand faces the camera, -1 if the palm does, 0 if unsure (hand side-on).

    Image coordinates (y down, z away from camera): for a right hand with the palm towards the camera the
    normal's z is negative. NSL23 signers are all right-handed (brief §3).
    """
    nz = palm_normal_z(pose)
    if abs(nz) < margin:
        return 0
    back = nz > 0 if handedness == "right" else nz < 0
    return 1 if back else -1


def facing_track(
    poses: list[np.ndarray], handedness: str = "right", window: int = 5, margin: float = 0.35
) -> list[int]:
    """``palm_facing`` per frame with the palm normal moving-averaged over ``window`` frames (no flicker)."""
    nz = np.array([palm_normal_z(p) for p in poses])
    half = window // 2
    out = []
    for i in range(len(poses)):
        v = float(nz[max(0, i - half) : i + half + 1].mean())
        if abs(v) < margin:
            out.append(0)
        else:
            out.append(1 if (v > 0) == (handedness == "right") else -1)
    return out


def motion(poses: list[np.ndarray]) -> np.ndarray:
    """Centred per-frame motion: mean 2-D landmark displacement to the neighbouring frames."""
    stack = np.asarray(poses, dtype=np.float64)[:, :, :2]
    n = len(stack)
    if n < 2:
        return np.zeros(n)
    step = np.linalg.norm(np.diff(stack, axis=0), axis=2).mean(axis=1)  # between frame i and i+1
    out = np.empty(n)
    out[0], out[-1] = step[0], step[-1]
    out[1:-1] = (step[:-1] + step[1:]) / 2
    return out


def speed_profile(poses: list[np.ndarray], fps: float) -> np.ndarray:
    """Centred per-frame hand speed in hand-sizes per second (scale-free, so comparable across clips)."""
    if len(poses) < 2:
        return np.zeros(len(poses))
    return motion(poses) / max(hand_size(poses), 1e-9) * fps


def _stillest_window(v: np.ndarray, w: int) -> int:
    costs = np.convolve(v, np.ones(w) / w, mode="valid")  # mean speed of every window of length w
    return int(np.argmin(costs))


def hold_segment(
    poses: list[np.ndarray],
    fps: float,
    min_s: float = 0.5,
    max_s: float = 1.5,
    fallback_s: float = 1.0,
    still: float = 1.0,
) -> tuple[int, int]:
    """[start, end) of the held hand shape.

    The longest run where the hand moves slower than ``still`` hand-sizes/s; if longer than ``max_s`` the stillest
    ``max_s`` of it. If no run lasts ``min_s``, the stillest ``fallback_s`` window instead.

    For static signs this drops the signer raising the hand into position (and lowering it), which otherwise
    plays as a very fast swoop followed by a jump back when the animation loops.
    """
    n = len(poses)
    if n < 3:
        return 0, n
    v = speed_profile(poses, fps)
    best = (0, 0)
    start = None
    for i, slow in enumerate(list(v < still) + [False]):
        if slow and start is None:
            start = i
        elif not slow and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    a, b = best
    if b - a >= round(min_s * fps):
        limit = round(max_s * fps)
        if b - a > limit:
            a = a + _stillest_window(v[a:b], limit)
            b = a + limit
        return a, b
    w = min(n, max(2, round(fallback_s * fps)))
    a = _stillest_window(v, w)
    return a, a + w


def movement_segment(poses: list[np.ndarray], fps: float, approach: float = 3.0) -> tuple[int, int]:
    """[start, end) for a sign with movement: drop the opening approach (hand raised into position).

    The approach is a speed peak (> ``approach`` hand-sizes/s) in the first 35 % of the clip that falls to a local
    minimum before the sign's own movement; the clip starts at that minimum. Without such a peak nothing is cut.
    """
    n = len(poses)
    v = speed_profile(poses, fps)
    head = max(2, int(n * 0.35))
    peak = int(np.argmax(v[:head]))
    if v[peak] <= approach:
        return 0, n
    i = peak
    while i + 1 < n and v[i + 1] <= v[i]:  # walk down (through flat stretches) to the local minimum
        i += 1
    settled = v[i] < 0.5 * v[peak]
    if not settled or i >= head + round(0.25 * fps):  # no real settle near the start: it is all movement
        return 0, n
    return i, n


SETTLE_MAX_SPEED = 2.0  # hand sizes/s: slower than this counts as "settling into the shape" (faster = arm rising,
# and motion-blurred frames that cut out badly)
SETTLE_MAX_S = 0.5  # at most this much of the move into the shape is shown before the hold
STILL_SPEED = 1.0  # hand sizes/s


def sign_segment(poses: list[np.ndarray], fps: float, moving: bool) -> tuple[int, int]:
    """[start, end) of the sign itself: without the arm rising from rest at the start or lowering at the end.

    Static signs: the short settle into the shape (the last <= 0.5 s before the hold where the hand is already
    slower than 3 hand-sizes/s) + the hold. Movement signs: from the end of the opening approach to the last still
    moment before the hand speeds off again (lowering).
    """
    n = len(poses)
    if n < 3:
        return 0, n
    v = speed_profile(poses, fps)
    if not moving:
        a, b = hold_segment(poses, fps)
        start = a
        while start > 0 and a - start < round(SETTLE_MAX_S * fps) and v[start - 1] < SETTLE_MAX_SPEED:
            start -= 1
        return start, b
    a, b = movement_segment(poses, fps)
    still = [i for i in range(a, b) if v[i] < STILL_SPEED]
    if still:
        last_still = still[-1]
        if last_still + 1 < b and v[last_still + 1 :].max() > SETTLE_MAX_SPEED:  # the hand leaves: lowering
            b = last_still + 1
    return a, b


def held_frame(poses: list[np.ndarray], middle: float = 0.70) -> int:
    """Index of the stillest frame inside the middle ``middle`` fraction of the clip (the held hand shape)."""
    n = len(poses)
    if n <= 2:
        return n // 2
    cut = (1 - middle) / 2
    lo = min(int(np.floor(n * cut)), n - 1)
    hi = max(int(np.ceil(n * (1 - cut))), lo + 1)
    m = motion(poses)[lo:hi]
    return lo + int(np.argmin(m))
