"""Draw key-pose animation frames with the illustrated renderer.

One fixed orthographic camera per letter, looking along the video camera's axis (MediaPipe world axes are the
camera's), the same scale for every letter (``VIEW_SIDE_M``), centred on everything the hand covers in the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ishara_ml.teaching.geometry import FINGERS, tip_reach
from ishara_ml.teaching.kinematics import HandPose, Skeleton, forward_kinematics, interpolate, palm_frame
from ishara_ml.teaching.render import draw_hand, parts_behind_palm

VIEW_SIDE_M = 0.26  # metres of the scene shown across the square image (same for every letter)
FIT = 0.85  # the hand's joints (not the forearm) must stay inside this fraction of the view (outline room) ...
# ... so a wrist path too long for the shared scale is scaled down (never the hand): see fit_path().
RAISED = 1.0  # tip reach (hand sizes) above which a finger is drawn on its own ...
HYSTERESIS = 0.08  # ... with this dead band, so fingers don't flicker during a transition
SIZE = 360
STATIC_STYLE = {"hand_style": "classic", "forearm": "edge"}
# Every letter (user decision): no forearm, the hand is what matters; a bigger palm that ends in a rounded heel.
# A letter's own "render" options may change the palm, but never bring a forearm back.
HAND_DEFAULTS = {"forearm": "none", "palm_scale": 1.2, "palm_heel": 1.0}
# Dragging letters (movement.is_drag: u, o, ...): the hand's travel is the point, so a short fading forearm and the
# natural hand drawing (seamless knuckles, rigid palm shape, thumb from its base). Other movement letters (am: a
# finger bends, the hand barely moves) keep the static look.
DRAG_STYLE = {"hand_style": "natural", "forearm": "short"}
# Per-letter options a key-pose JSON may set under "render" (letters without it are drawn exactly as before):
#   seamlessKnuckles: bool  - fingers/thumb behind the palm join it with no outline line across them
#   palmWidth: float        - widen the palm across (e.g. 1.25), for a hand that looks too skinny
#   palmHeel: float         - palm meets the forearm this many forearm-widths wide (1.0 = flush), not at a point
#   forearm: "edge"|"short" - "short" = a stub that fades out right after the wrist
#   frontParts: [names]     - draw these parts in front of the palm even if the 3-D pose puts them behind
#   drag: bool              - movement letters: force the drag style on/off (default: movement.is_drag)
#   thumbLoop: finger|"auto"- thumb tip meets this fingertip: keep the opening between them open (O / beak shapes)
#   loopThrough: [names]    - fingers the thumbLoop opening may cut through (they bunch at the thumb, e.g. ttha)
RENDER_OPTIONS = {
    "seamlessKnuckles": "seamless_behind",
    "palmWidth": "palm_scale",
    "palmHeel": "palm_heel",
    "forearm": "forearm",
    "thumbLoop": "thumb_loop",
    "loopThrough": "loop_through",
}


def render_style(render: dict | None) -> tuple[dict, frozenset[str]]:
    """(draw_hand keyword options, parts forced in front) from a key-pose JSON's "render" block."""
    render = render or {}
    unknown = set(render) - set(RENDER_OPTIONS) - {"frontParts", "drag"}
    if unknown:
        raise ValueError(f"unknown render options {sorted(unknown)} (allowed: {sorted(RENDER_OPTIONS)} + frontParts)")
    kwargs = {RENDER_OPTIONS[k]: v for k, v in render.items() if k in RENDER_OPTIONS}
    return kwargs, frozenset(render.get("frontParts", ()))


ARM_THICKNESS = 0.7  # forearm cross-section: thickness (back-palm) = this x its width (thumb side-pinky side)


@dataclass
class Frame:
    pose: HandPose
    forearm_deg: float


def lerp_frame(a: Frame, b: Frame, t: float) -> Frame:
    d = (b.forearm_deg - a.forearm_deg + 180) % 360 - 180
    return Frame(interpolate(a.pose, b.pose, t), a.forearm_deg + t * d)


def view_centre(points: list[np.ndarray]) -> np.ndarray:
    xy = np.concatenate([p[:, :2] for p in points])
    return (xy.min(axis=0) + xy.max(axis=0)) / 2


def to_unit(points: np.ndarray, centre: np.ndarray, side: float = VIEW_SIDE_M) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, :2] = (out[:, :2] - centre) / side + 0.5
    out[:, 2] = out[:, 2] / side
    return out


def _extent(pts: list[np.ndarray]) -> float:
    xy = np.concatenate([p[:, :2] for p in pts])
    return float((xy.max(axis=0) - xy.min(axis=0)).max())


def fit_path(frames: list[Frame], skeleton: Skeleton, side: float = VIEW_SIDE_M) -> tuple[list[Frame], float]:
    """Scale the wrist path (positions are relative to key pose 1) until the hand fits the fixed view.

    Returns the frames and the factor used (1.0 = the real path)."""

    def scaled(f: float) -> list[Frame]:
        out = []
        for fr in frames:
            pose = fr.pose.copy()
            pose.wrist_pos = pose.wrist_pos * f
            out.append(Frame(pose, fr.forearm_deg))
        return out

    def fits(f: float) -> bool:
        return _extent([forward_kinematics(fr.pose, skeleton) for fr in scaled(f)]) <= FIT * side

    if fits(1.0):
        return frames, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return scaled(lo), lo


def visible_arm_width(points: np.ndarray, away: np.ndarray, width: float) -> float:
    """Image width of an elliptical forearm (``width`` x ARM_THICKNESS*width) that turns with the palm."""
    r = palm_frame(points)
    s3 = np.array([-away[1], away[0], 0.0])
    a = width / 2 * float(r[:, 0] @ s3)
    b = ARM_THICKNESS * width / 2 * float(r[:, 2] @ s3)
    return 2 * float(np.hypot(a, b))


def finger_states(units: list[np.ndarray], size_unit: float) -> list[dict[str, bool]]:
    """Raised/folded per finger per frame, with hysteresis (state changes only past RAISED +- HYSTERESIS)."""
    names = [n for n in FINGERS if n != "thumb"]
    state: dict[str, bool] = {}
    out = []
    for u in units:
        for n in names:
            r = tip_reach(u, FINGERS[n], size_unit)
            if n not in state:
                state[n] = r >= RAISED
            elif state[n] and r < RAISED - HYSTERESIS:
                state[n] = False
            elif not state[n] and r > RAISED + HYSTERESIS:
                state[n] = True
        out.append({"thumb": True, **state})
    return out


def render_frames(
    frames: list[Frame],
    skeleton: Skeleton,
    size: int = SIZE,
    centre: np.ndarray | None = None,
    *,
    side: float = VIEW_SIDE_M,
    aways: list[np.ndarray] | None = None,
    style: dict[str, str | bool] | None = None,
    front: frozenset[str] = frozenset(),
) -> tuple[list[Image.Image], np.ndarray]:
    """RGB images of ``frames`` and the view centre used (metres). ``side``: metres across the view (static letters
    always use VIEW_SIDE_M); ``aways``: per-frame 2-D forearm directions overriding ``Frame.forearm_deg``."""
    pts = [forward_kinematics(f.pose, skeleton) for f in frames]
    if centre is None:
        centre = view_centre(pts)
    units = [to_unit(p, centre, side) for p in pts]
    size_unit = skeleton.hand_size() / side
    states = finger_states(units, size_unit)
    images = []
    for i, (u, p, f, ext) in enumerate(zip(units, pts, frames, states, strict=True)):
        rad = math.radians(f.forearm_deg)
        away = np.array([math.cos(rad), math.sin(rad)]) if aways is None else aways[i]
        arm = visible_arm_width(p, away, skeleton.forearm_width) / side
        images.append(
            draw_hand(
                u,
                size=size,
                hand_scale=size_unit,
                extended=ext,
                handedness="right",
                **{**STATIC_STYLE, **HAND_DEFAULTS, **(style or {}), "forearm": "none"},
                forearm_dir=away,
                arm_width=arm,
                behind_palm=parts_behind_palm(u) - front,
            ).convert("RGB")
        )
    return images, centre


@dataclass
class MovementRender:
    images: list[Image.Image]
    centre: np.ndarray
    side: float
    scale: float  # VIEW_SIDE_M / side: 1.0 = same scale as the static letters
    forearm_clamped: int  # frames whose forearm had to be turned away from the fingers
    style: dict[str, str] | None = None


def render_movement(
    frames: list[Frame],
    skeleton: Skeleton,
    size: int = SIZE,
    style: dict[str, str] | None = None,
    extra: dict | None = None,
    front: frozenset[str] = frozenset(),
) -> MovementRender:
    """Letters with movement: one view over ALL frames (hand fully inside with padding, static scale if it fits),
    forearm kept away from the fingers, every frame sanity-checked before it is drawn."""
    from ishara_ml.teaching import movement as mv

    pts = [forward_kinematics(f.pose, skeleton) for f in frames]
    centre, side = mv.movement_view(pts, skeleton, VIEW_SIDE_M)
    aways, clamped = [], 0
    for i, (p, f) in enumerate(zip(pts, frames, strict=True)):
        rad = math.radians(f.forearm_deg)
        away, hit = mv.constrain_forearm(p, np.array([math.cos(rad), math.sin(rad)]))
        aways.append(away)
        clamped += hit
        mv.check_frame(i, p, to_unit(p, centre, side), skeleton, side)
    full_style = {**(style or {}), "rigid_palm_if_thumb_behind": True, **(extra or {})}  # no spike (ddha)
    images, _ = render_frames(frames, skeleton, size, centre, side=side, aways=aways, style=full_style, front=front)
    return MovementRender(images, centre, side, VIEW_SIDE_M / side, clamped, style)
