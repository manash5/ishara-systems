"""Draw a flat illustrated hand from MediaPipe's 21 landmarks (brief §5, step 7).

Coordinates come in unit-box space (x, y in [0, 1] inside the crop box, z on the same scale, smaller = nearer).
Everything is drawn at ``scale``x and downscaled with LANCZOS for smooth edges.

Layers, bottom to top:
0. An outline underlay for every part (so no white background shows where parts meet).
1. One merged silhouette: forearm + palm + every *folded* finger. All fills, then crease lines between the folded
   fingers (nearer fingers hide the creases of farther ones), so the fist reads as one shape with visible fingers.
2. The thumb and every *raised* finger (straight or hooked), each with its own outline, farthest-first by depth,
   slightly thicker than folded ones. A pointing finger is never hidden by the fist.
Nails are drawn only on straight raised fingers when the back of the hand faces the camera.
All widths scale with one per-clip hand size (3-D bone lengths, see ``geometry.hand_size``); the forearm points at
the elbow from MediaPipe Pose when available.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from ishara_ml.teaching.geometry import EXTENDED_THRESHOLD, FINGERS, hand_size, palm_facing, straightness

# Cartoon "emoji hand" look (user reference, 2026-11-26): light peach fill, a darker peach shade along the
# lower-right inside edge of every shape, a thick dark-brown outline and brown crease lines between fingers.
SKIN = (0xF9, 0xD2, 0xA9)
SHADE = (0xEE, 0xB0, 0x7E)  # cel shade: inside each skin region, along its edge facing away from the light
OUTLINE = (0x8E, 0x45, 0x22)
CREASE = (0x9E, 0x55, 0x2E)  # between folded fingers: almost the outline colour, drawn a little thinner
NAIL = (0xEE, 0xB8, 0x8E)
DRAW_NAILS = False  # the reference style has no nails
SHADE_OFFSET = 0.11  # x S: how deep the cel shade reaches in from a region's lower-right edge
SHADE_DIR = (0.55, 0.835)  # unit image direction pointing away from the light (light comes from the upper left)
WHITE = (255, 255, 255)

PALM = [0, 1, 2, 5, 9, 13, 17]
# "natural" hand style (letters with movement): the palm is built from rigid points only, so a folding thumb can't
# pinch it; the thumb is drawn from its base (point 1) with a fleshy, thicker start (thenar).
PALM_NATURAL = [0, 1, 5, 9, 13, 17]
PALM_ROUNDING_NATURAL = 0.20
THUMB_NATURAL_RADII = (1.0, 1.1, 1.0, 0.85)  # x the thumb radius at points 1, 2, 3, 4 (thenar = widest at 2)
HAND_STYLES = ("classic", "natural")
# "short" forearm (letters with movement): a stub that fades out, so the hand's movement is what the eye follows.
SHORT_ARM_FADE_START = 0.5  # x S past the wrist: solid up to here ...
SHORT_ARM_FADE_END = 1.4  # ... faded into the background by here
# Drawn chains: the thumb starts at 2 because 1-2 is part of the palm polygon.
CHAINS = {"thumb": [2, 3, 4], **{k: v for k, v in FINGERS.items() if k != "thumb"}}

# All widths are fractions of the per-clip hand size S (mean 3-D length of palm + proximal bones, ~0.6 x palm length).
# chunky cartoon fingers
FINGER_RADIUS = {"thumb": 0.205, "index": 0.185, "middle": 0.190, "ring": 0.180, "pinky": 0.160}
CURLED_FACTOR = 1.15  # folded fingers press against each other: a compact fist, separated by creases
EXTENDED_FACTOR = 1.10
TIP_TAPER = 0.92  # fat, rounded fingertips
PALM_ROUNDING = 0.16  # softer, rounder cartoon palm corners
OUTLINE_WIDTH = 0.068  # thick cartoon outline
CREASE_WIDTH = 0.75  # x outline width

FOREARM_MAX_OF_PALM = 0.85  # forearm width <= this x the palm's visible (2-D) width ...
FOREARM_MIN = 0.50  # ... but at least this x S, for a side-on hand
FOREARM_MAX = 1.00  # ... and at most this x S
FOREARM_MAX_ALONG_EDGE = 0.6  # top corners sit at most this far along the wrist edges (0-17, 0-1)
FOREARM_FLARE = 0.06  # half-width grows by this x S per S of length (tapers towards the wrist)
FOREARM_FADE_START = 1.0  # x S past the wrist: the forearm is solid up to here ...
FOREARM_FADE_END = 1.7  # ... then fades into the white background by here (no hard cut, no long arm)


def forearm_width(pose_xy: np.ndarray, size: float) -> float:
    """Full forearm width at the wrist: <= 85 % of the palm's visible width, clamped to [0.5, 1.0] x size."""
    visible = float(np.hypot(*(pose_xy[5] - pose_xy[17])))
    return float(np.clip(FOREARM_MAX_OF_PALM * visible, FOREARM_MIN * size, FOREARM_MAX * size))


def forearm_direction(pose_xy: np.ndarray, elbow: np.ndarray | None = None) -> np.ndarray:
    """Unit 2-D direction from the wrist along the forearm: towards the elbow when known (MediaPipe Pose),
    otherwise away from the middle knuckle (wrong when the wrist is bent)."""
    xy = np.asarray(pose_xy, dtype=np.float64)
    for vec in ((np.asarray(elbow, dtype=np.float64) - xy[0]) if elbow is not None else None, xy[0] - xy[9]):
        if vec is not None:
            norm = float(np.hypot(*vec))
            if norm > 1e-6:
                return vec / norm
    return np.array([0.0, 1.0])


def forearm_polygon(
    pose_xy: np.ndarray, size: float, length: float, elbow: np.ndarray | None = None
) -> list[np.ndarray]:
    """Forearm quad whose top corners lie on the palm's wrist edges (0->17 and 0->1), so the arm continues the
    palm outline without a step, running ``length`` from the wrist towards the elbow."""
    xy = np.asarray(pose_xy, dtype=np.float64)
    away = forearm_direction(xy, elbow)
    side = np.array([-away[1], away[0]])
    width = forearm_width(xy, size)
    e_pinky, e_thumb = xy[17] - xy[0], xy[1] - xy[0]
    spread = abs(float((e_pinky - e_thumb) @ side))
    t = min(width / spread, FOREARM_MAX_ALONG_EDGE) if spread > 1e-6 else 0.0
    a, b = xy[0] + t * e_pinky, xy[0] + t * e_thumb
    if spread <= 1e-6 or t * spread < 0.5 * width:  # edges too close together (side-on): centre on the wrist
        a, b = xy[0] + side * width / 2, xy[0] - side * width / 2
    mid = (a + b) / 2
    sign = 1.0 if float((a - mid) @ side) >= 0 else -1.0
    far_half = width / 2 + FOREARM_FLARE * length
    far = xy[0] + away * length  # measured from the wrist, like the fade
    return [a, far + sign * side * far_half, far - sign * side * far_half, b]


# Key-pose style: the forearm is one tapered shape that always runs off the frame (no fade, no stump).
EDGE_FLARE = 0.18  # half-width grows by this fraction from the wrist ...
EDGE_FLARE_LEN = 2.5  # ... over this many wrist widths (curved sides: smoothstep)
EDGE_CAP = 0.45  # x wrist width: the arm starts this far inside the hand with a rounded top the palm hides
EDGE_SAMPLES = 32


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def forearm_edge_polygon(pose_xy: np.ndarray, width: float, away: np.ndarray, length: float) -> list[np.ndarray]:
    """Forearm as one tapered polygon with curved sides. It starts inside the hand with a rounded top (covered by
    the palm, so palm and arm read as one shape), is ``width`` wide at the wrist, widens slightly towards the
    elbow and runs ``length`` from the wrist (pass more than the image diagonal, so it always leaves the frame)."""
    xy = np.asarray(pose_xy, dtype=np.float64)
    away = np.asarray(away, dtype=np.float64)
    away = away / max(float(np.hypot(*away)), 1e-12)
    side = np.array([-away[1], away[0]])
    cap = EDGE_CAP * width
    s = np.concatenate([-cap * np.cos(np.linspace(0, np.pi / 2, 9)), np.linspace(0, 4 * width, EDGE_SAMPLES)[1:]])
    s = np.concatenate([s, [length]])
    half = width / 2 * (1 + EDGE_FLARE * _smoothstep(s / (EDGE_FLARE_LEN * width)))
    top = s < 0
    half[top] = width / 2 * np.sqrt(np.clip(1 - (s[top] / cap) ** 2, 0.0, 1.0))
    centre = xy[0] + away[None] * s[:, None]
    left = centre + side[None] * half[:, None]
    right = centre - side[None] * half[:, None]
    return [*left, *right[::-1]]


def parts_behind_palm(pose: np.ndarray, margin: float = 0.15) -> set[str]:
    """Thumb/fingers lying behind the palm as seen from the camera (larger z = farther), from the 3-D pose.

    Each chain's points past its base are compared with the palm plane's depth at the same image position; a
    side-on palm (no usable plane) falls back to the palm centre's depth. ``margin`` is in hand sizes.
    """
    p = np.asarray(pose, dtype=np.float64)
    size = max(hand_size(p), 1e-9)
    n = np.cross(p[5] - p[0], p[17] - p[0])
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    centre = p[[0, 5, 9, 13, 17]].mean(axis=0)
    out = set()
    for name, chain in FINGERS.items():
        pts = p[chain[1:]]
        if abs(n[2]) > 0.3:
            plane_z = centre[2] - (n[0] * (pts[:, 0] - centre[0]) + n[1] * (pts[:, 1] - centre[1])) / n[2]
            depth = float((pts[:, 2] - plane_z).mean())
        else:
            depth = float(pts[:, 2].mean() - centre[2])
        if depth > margin * size:
            out.add(name)
    return out


def _disc(draw: ImageDraw.ImageDraw, c: np.ndarray, r: float, fill) -> None:
    draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill)


def _segment(draw: ImageDraw.ImageDraw, a: np.ndarray, b: np.ndarray, ra: float, rb: float, fill) -> None:
    """Tapered capsule body from a to b plus the round cap at b."""
    d = b - a
    length = float(np.hypot(*d))
    if length > 1e-6:
        n = np.array([-d[1], d[0]]) / length
        draw.polygon([tuple(a + n * ra), tuple(b + n * rb), tuple(b - n * rb), tuple(a - n * ra)], fill=fill)
    _disc(draw, b, rb, fill)


def _rounded_polygon(draw: ImageDraw.ImageDraw, pts: list[np.ndarray], r: float, fill) -> None:
    draw.polygon([tuple(p) for p in pts], fill=fill)
    for i, p in enumerate(pts):
        _segment(draw, p, pts[(i + 1) % len(pts)], r, r, fill)


def _chain(draw: ImageDraw.ImageDraw, pts: list[np.ndarray], radii: np.ndarray, grow: float, fill, base_cap: bool):
    """A finger as joined segments. Without ``base_cap`` the body starts a little past the knuckle (no arc on the
    palm); with it the knuckle gets a full round cap."""
    a = pts[0]
    if base_cap:
        _disc(draw, a, radii[0] + grow, fill)
    else:
        d = pts[1] - a
        length = float(np.hypot(*d))
        if length > 1e-6:
            a = a + d * min(0.5, 0.9 * radii[0] / length)
    for k in range(len(pts) - 1):
        _segment(draw, a if k == 0 else pts[k], pts[k + 1], radii[k] + grow, radii[k + 1] + grow, fill)


def _ellipse(c: np.ndarray, u: np.ndarray, a: float, b: float, steps: int = 24) -> list[tuple[float, float]]:
    v = np.array([-u[1], u[0]])
    ts = np.linspace(0, 2 * math.pi, steps, endpoint=False)
    return [tuple(c + u * a * math.cos(t) + v * b * math.sin(t)) for t in ts]


def draw_hand(
    pose: np.ndarray,
    size: int = 360,
    scale: int = 4,
    *,
    hand_scale: float | None = None,
    extended: dict[str, bool] | None = None,
    facing: int | None = None,
    handedness: str = "right",
    only: Iterable[str] | None = None,
    part_colors: dict[str, tuple[int, int, int]] | None = None,
    elbow: np.ndarray | None = None,
    forearm: str = "fade",
    forearm_dir: np.ndarray | None = None,
    arm_width: float | None = None,
    behind_palm: Iterable[str] | None = None,
    hand_style: str = "classic",
    rigid_palm_if_thumb_behind: bool = False,
    seamless_behind: bool = False,
    palm_scale: float = 1.0,
    palm_heel: float = 0.0,
    thumb_loop: str | None = None,
    loop_through: Iterable[str] = (),
) -> Image.Image:
    """Render one pose (21 x 3, unit-box coords) to a ``size`` x ``size`` RGB image on white.

    ``hand_scale``: per-clip hand size in unit-box coords (``geometry.hand_size``); defaults to this pose's.
    ``extended``: per-finger raised (True: drawn on its own, on top) or folded into the fist (False), e.g. from
    ``geometry.finger_states``; defaults to this pose's straightness.
    ``elbow``: elbow position in unit-box coords (MediaPipe Pose) for the forearm's direction.
    ``facing``: ``geometry.palm_facing`` result; defaults to this pose's.
    ``only`` / ``part_colors``: draw a subset of parts ("forearm", "palm", finger names) / override a part's
    fill colour (used by tests to measure what stays visible).
    ``forearm``: "fade" (short arm fading into white) or "edge" (key-pose style: one tapered shape running off
    the frame, ``forearm_dir`` = 2-D image direction wrist -> elbow, ``arm_width`` = width in unit-box coords).
    ``behind_palm``: thumb/fingers behind the palm (``parts_behind_palm``), drawn before the palm.
    ``forearm="short"``: the edge-style shape, but only a stub that fades out SHORT_ARM_FADE_START..END x S.
    ``forearm="none"``: no forearm (use with ``palm_heel`` so the palm ends in a rounded heel, not a point).
    ``hand_style``: "classic" or "natural" (rigid palm shape, thumb from its base, softer palm corners, and parts
    behind the palm join it without an outline across the knuckles).
    ``thumb_loop``: a finger name (or "auto" = the fingertip nearest the thumb tip) whose tip meets the thumb tip:
    the opening the two enclose is cut back out of the drawing and outlined (O / beak handshapes), see
    ``_cut_thumb_loop``. Off by default. ``loop_through``: other fingers the opening may cut through (fingers that
    bunch at the thumb but would otherwise fill the opening, ttha).
    """
    if hand_style not in HAND_STYLES:
        raise ValueError(f"hand_style must be one of {HAND_STYLES}")
    natural = hand_style == "natural"
    chains = {**CHAINS, "thumb": [1, 2, 3, 4]} if natural else CHAINS
    big = size * scale
    p = np.asarray(pose, dtype=np.float64) * big
    xy = p[:, :2]
    S = max((hand_scale if hand_scale else hand_size(np.asarray(pose))) * big, 1.0)
    ow = max(OUTLINE_WIDTH * S, float(scale))
    if extended is None:
        extended = {n: straightness(pose, c) >= EXTENDED_THRESHOLD for n, c in FINGERS.items()}
    if facing is None:
        facing = palm_facing(pose, handedness)
    parts = set(only) if only is not None else {"forearm", "palm", *CHAINS}
    colors = part_colors or {}

    def radii(name: str) -> np.ndarray:
        factor = EXTENDED_FACTOR if extended.get(name, False) else CURLED_FACTOR
        r0 = FINGER_RADIUS[name] * S * factor
        if natural and name == "thumb":
            rr = r0 * np.array(THUMB_NATURAL_RADII)
            rr[0] = min(rr[0], PALM_ROUNDING_NATURAL * S)  # the base cap stays inside the palm's rounded corner
            return rr
        return np.linspace(r0, r0 * TIP_TAPER, len(chains[name]))

    img = Image.new("RGB", (big, big), WHITE)
    draw = ImageDraw.Draw(img)

    # --- Layer 1: forearm + palm + curled fingers as one silhouette -----------------------------------------
    # Drawn a little past the fade end; _fade_forearm() then blends its far part into the background.
    elbow_px = None if elbow is None else np.asarray(elbow, dtype=np.float64)[:2] * big
    away = forearm_direction(xy, elbow_px) if forearm_dir is None else np.asarray(forearm_dir, dtype=np.float64)
    arm = None
    if forearm == "none":  # no forearm at all: the hand ends at the palm's heel
        away = forearm_direction(xy, None)  # the heel is square to the hand's own axis
    elif forearm in ("edge", "short"):
        width = arm_width * big if arm_width else forearm_width(xy, S)
        length = 3.0 * big if forearm == "edge" else (SHORT_ARM_FADE_END + 0.3) * S
        arm = forearm_edge_polygon(xy, width, away, length=length)
    else:
        arm = forearm_polygon(xy, S, length=(FOREARM_FADE_END + 0.2) * S, elbow=elbow_px)
    behind = sorted(
        (n for n in (behind_palm or ()) if n in CHAINS and n in parts), key=lambda n: -p[chains[n], 2].mean()
    )
    # The classic palm includes the thumb's base knuckle (2). When the thumb is behind the palm that corner has
    # nothing to join and can stick out as a spike (ddha); letters with movement then use the rigid palm.
    # (Opt-in, so the approved static letters stay pixel-identical.)
    rigid = natural or (rigid_palm_if_thumb_behind and "thumb" in behind)
    palm = [xy[i] for i in (PALM_NATURAL if rigid else PALM)]
    if palm_scale != 1.0:  # widen the palm across (thumb side <-> pinky side), keeping its length (jha)
        across = xy[5] - xy[17]
        across = across / max(float(np.hypot(*across)), 1e-9)
        centre = np.mean(palm, axis=0)
        palm = [centre + (q - centre) + across * float((q - centre) @ across) * (palm_scale - 1.0) for q in palm]
    if palm_heel > 0:  # the palm meets the forearm at (palm_heel x) the forearm's width instead of at one point
        w_arm = width if forearm in ("edge", "short") else forearm_width(xy, S)  # "none": a wrist-wide heel
        across_arm = np.array([-away[1], away[0]]) / max(float(np.hypot(*away)), 1e-9)
        # inset by the palm's corner rounding, so the rounded heel ends flush with the forearm's sides (no knob)
        rounding = (PALM_ROUNDING_NATURAL if natural else PALM_ROUNDING) * S
        half = max(palm_heel * w_arm / 2 - rounding, 0.0)
        a, b = xy[0] + across_arm * half, xy[0] - across_arm * half
        near1, near17 = (a, b) if np.linalg.norm(a - xy[1]) <= np.linalg.norm(b - xy[1]) else (b, a)
        palm = [near17, near1, *palm[1:]]  # polygon order: ... 17 -> heel -> 1 ...
    curled = [n for n in CHAINS if n != "thumb" and not extended.get(n, False) and n in parts and n not in behind]
    polys = []
    if "forearm" in parts and arm is not None:
        polys.append((arm, 0.06 * S, "forearm"))
    if "palm" in parts:
        polys.append((palm, (PALM_ROUNDING_NATURAL if natural else PALM_ROUNDING) * S, "palm"))

    on_top = sorted(  # thumb + raised fingers, farthest first, so depth decides what overlaps what
        (n for n in CHAINS if (n == "thumb" or extended.get(n, False)) and n in parts and n not in behind),
        key=lambda n: -p[chains[n], 2].mean(),
    )

    # Outline underlay for every part (with round knuckle caps). Fills drawn later cover it inside the hand; around
    # the base of the thumb and raised fingers it closes the gap that otherwise showed white background there.
    for pts, r, _ in polys:
        _rounded_polygon(draw, pts, r + ow, OUTLINE)
    for n in curled + on_top:
        _chain(draw, [xy[i] for i in chains[n]], radii(n), ow, OUTLINE, base_cap=True)
    # Parts behind the palm: drawn now (outline + fill), then the palm's outline and fill go over them.
    for n in behind:
        pts, rr = [xy[i] for i in chains[n]], radii(n)
        # seamless: the knuckle cap gets its own outline, so where it pokes past the palm edge it reads as a knuckle
        # bump (inside the palm the palm's fill covers it), instead of a pale blob with no edge.
        _chain(draw, pts, rr, ow, OUTLINE, base_cap=seamless_behind and not natural)
        _chain(draw, pts, rr, 0.0, colors.get(n, SKIN), base_cap=True)
    if behind and not natural and not seamless_behind:  # else: parts behind join the palm with no line
        for pts, r, _ in polys:
            _rounded_polygon(draw, pts, r + ow, OUTLINE)
    for pts, r, name in polys:
        _rounded_polygon(draw, pts, r, colors.get(name, SKIN))
    for n in curled:
        _chain(draw, [xy[i] for i in chains[n]], radii(n), 0.0, colors.get(n, SKIN), base_cap=True)
    # Crease lines between folded fingers: painter's order, farthest first, so a nearer one hides the crease behind.
    crease = CREASE_WIDTH * ow
    for n in sorted(curled, key=lambda n: -p[chains[n], 2].mean()):
        pts, rr = [xy[i] for i in chains[n]], radii(n)
        _chain(draw, pts, rr, 0.0, CREASE, base_cap=False)
        _chain(draw, pts, rr - crease, 0.0, colors.get(n, SKIN), base_cap=False)

    # --- Layer 2: thumb and raised fingers, farthest first -------------------------------------------------
    for n in on_top:
        pts, rr = [xy[i] for i in chains[n]], radii(n)
        # Outline starts past the knuckle (no arc on the palm); the fill keeps its round knuckle cap so the
        # finger joins the hand without a gap.
        _chain(draw, pts, rr, ow, OUTLINE, base_cap=False)
        _chain(draw, pts, rr, 0.0, colors.get(n, SKIN), base_cap=True)
        if DRAW_NAILS and facing == 1 and straightness(pose, FINGERS[n]) >= EXTENDED_THRESHOLD:
            d = pts[-1] - pts[-2]
            dl = float(np.hypot(*d))
            if dl > 0.3 * rr[-1]:
                u = d / dl
                draw.polygon(_ellipse(pts[-1] - u * 0.35 * rr[-1], u, 0.55 * rr[-1], 0.45 * rr[-1]), fill=NAIL)

    if thumb_loop:
        through = set(loop_through)
        drawn = {n: ([xy[i] for i in chains[n]], radii(n)) for n in CHAINS if n in parts and n not in through}
        img = _cut_thumb_loop(img, xy, S, ow, drawn, thumb_loop)
    img = _cel_shade(img, SHADE_OFFSET * S)  # before the fade, so fading the arm never changes the hand's shading
    if "forearm" in parts and forearm in ("fade", "short"):
        hand_parts = [(pts, r) for pts, r, name in polys if name != "forearm"]
        fingers = [(chains[n], radii(n)) for n in CHAINS if n in parts]
        fade = (
            (SHORT_ARM_FADE_START, SHORT_ARM_FADE_END) if forearm == "short" else (FOREARM_FADE_START, FOREARM_FADE_END)
        )
        direction = away if forearm == "short" else forearm_direction(xy, elbow_px)
        img = _fade_forearm(img, xy, S, ow, arm, hand_parts, fingers, direction, fade)
    return img.resize((size, size), Image.Resampling.LANCZOS)


LOOP_WEB = 0.30  # x S: skin kept along the straight web between the thumb base and the finger's knuckle
LOOP_CONTACT = 0.6  # x S: the cut only applies while the fingertip is this close to the thumb tip (touching) ...
# ... so transition frames, where the hand opens up, never get a line drawn across an open gap
LOOP_MIN_AREA = 0.008  # x S^2: smaller openings are not cut (they would read as specks)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by about ``r`` px (octagon), numpy only."""
    out = mask.copy()
    for k in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        if k % 2:  # alternate with diagonal steps, so the growth is round-ish, not a diamond
            grown[1:, 1:] |= out[:-1, :-1]
            grown[:-1, :-1] |= out[1:, 1:]
            grown[1:, :-1] |= out[:-1, 1:]
            grown[:-1, 1:] |= out[1:, :-1]
        out = grown
    return out


def _cut_thumb_loop(
    img: Image.Image,
    xy: np.ndarray,
    S: float,
    ow: float,
    drawn: dict[str, tuple[list[np.ndarray], np.ndarray]],
    finger: str,
) -> Image.Image:
    """Open the hole a thumb and a fingertip enclose (thumb 2-3-4, the finger tip -> knuckle, back along the web).

    Inside that loop everything stays that is a drawn thumb/finger (with its outline), the real palm (landmarks,
    not the widened, rounded palm shape) or the web next to the thumb base; the rest becomes background with an
    outline ring around it. Drawn chains are always kept, so a finger lying in front of the loop is never cut."""
    if finger == "auto":
        finger = min(("index", "middle", "ring", "pinky"), key=lambda n: float(np.hypot(*(xy[FINGERS[n][-1]] - xy[4]))))
    if finger not in FINGERS or finger == "thumb":
        raise ValueError(f"thumb_loop must be a finger name or 'auto', got {finger!r}")
    chain = FINGERS[finger]
    if float(np.hypot(*(xy[chain[-1]] - xy[4]))) > LOOP_CONTACT * S:  # not touching: nothing is enclosed
        return img
    loop_img = Image.new("L", img.size, 0)
    ImageDraw.Draw(loop_img).polygon([tuple(xy[i]) for i in (2, 3, 4, *chain[::-1])], fill=255)
    keep_img = Image.new("L", img.size, 0)
    kd = ImageDraw.Draw(keep_img)
    for pts, rr in drawn.values():
        _chain(kd, pts, rr, ow, 255, base_cap=True)
    kd.polygon([tuple(xy[i]) for i in PALM], fill=255)
    web = LOOP_WEB * S
    _segment(kd, xy[2], xy[chain[0]], web, web, 255)
    _disc(kd, xy[2], web, 255)
    hole = (np.asarray(loop_img) > 0) & (np.asarray(keep_img) == 0)
    if hole.sum() < LOOP_MIN_AREA * S * S:
        return img
    ring = _dilate(hole, max(1, int(round(ow)))) & ~hole
    a = np.asarray(img).copy()
    a[ring] = OUTLINE
    a[hole] = WHITE
    return Image.fromarray(a)


def _cel_shade(img: Image.Image, depth: float) -> Image.Image:
    """Cartoon shading: every SKIN pixel whose neighbour ``depth`` px towards SHADE_DIR is not SKIN (an outline,
    a crease or the background) becomes SHADE, so each finger and the palm get a darker band along their
    lower-right inside edge. Runs before the downscale, where fills are exact colours (no anti-aliasing)."""
    a = np.asarray(img).copy()
    skin = np.all(a == np.array(SKIN, dtype=np.uint8), axis=-1)
    dx, dy = (int(round(depth * c)) for c in SHADE_DIR)
    if not skin.any() or (dx == 0 and dy == 0):
        return img
    h, w = skin.shape
    ahead = np.zeros_like(skin)
    ahead[: h - dy, : w - dx] = skin[dy:, dx:]
    a[skin & ~ahead] = SHADE
    return Image.fromarray(a)


def _fade_forearm(
    img: Image.Image,
    xy: np.ndarray,
    S: float,
    ow: float,
    forearm: list[np.ndarray],
    hand_parts: list[tuple[list[np.ndarray], float]],
    fingers: list[tuple[list[int], np.ndarray]],
    away: np.ndarray,
    fade: tuple[float, float] = (FOREARM_FADE_START, FOREARM_FADE_END),
) -> Image.Image:
    """Blend the forearm into white between ``fade`` = (start, end) x S past the wrist.

    Only forearm pixels fade; anything belonging to the palm or a finger (e.g. a hand pointing down over its own
    arm) is left untouched.
    """
    big = img.size[0]
    arm = Image.new("L", img.size, 0)
    _rounded_polygon(ImageDraw.Draw(arm), forearm, 0.06 * S + ow, 255)
    hand = Image.new("L", img.size, 0)
    hd = ImageDraw.Draw(hand)
    for pts, r in hand_parts:
        _rounded_polygon(hd, pts, r + ow, 255)
    for chain, rr in fingers:
        _chain(hd, [xy[i] for i in chain], rr, ow, 255, base_cap=True)

    ys, xs = np.mgrid[0:big, 0:big].astype(np.float32)
    along = (xs - xy[0, 0]) * away[0] + (ys - xy[0, 1]) * away[1]
    start, end = fade[0] * S, fade[1] * S
    alpha = np.clip((along - start) / (end - start), 0.0, 1.0)
    alpha *= (np.asarray(arm) > 0) & (np.asarray(hand) == 0)
    if not alpha.any():
        return img
    a = alpha[..., None]
    out = np.asarray(img, dtype=np.float32) * (1 - a) + 255.0 * a
    return Image.fromarray(out.round().astype(np.uint8))
