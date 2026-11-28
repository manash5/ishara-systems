"""Tests for the v2 hand drawing: forearm limit, finger classification, layering, nails, held frame."""

import numpy as np
import pytest
from PIL import Image

from ishara_ml.teaching import render
from ishara_ml.teaching.geometry import (
    classify_fingers,
    facing_track,
    hand_size,
    held_frame,
    palm_facing,
)
from ishara_ml.teaching.pipeline import CAPTURE_FPS, PLAYBACK_FPS
from ishara_ml.teaching.render import (
    FOREARM_FADE_END,
    FOREARM_FADE_START,
    FOREARM_MAX_OF_PALM,
    FOREARM_MIN,
    draw_hand,
    forearm_polygon,
    forearm_width,
)
from ishara_ml.teaching.track import crop_box, to_box_coords

BLUE = (0, 0, 255)


def open_hand(cx: float = 500, cy: float = 500, s: float = 100) -> np.ndarray:
    """Upright open hand, flat (z = 0). Index on the image left of the pinky = back of a right hand."""
    pts = np.zeros((21, 3))
    pts[0] = [cx, cy + s, 0]
    for base, dx in {1: -0.55, 5: -0.3, 9: -0.05, 13: 0.18, 17: 0.4}.items():
        x0, y0 = cx + dx * s, cy + (0.45 * s if base == 1 else 0)
        length = 0.55 if base == 1 else (0.75 if base in (9, 13) else 0.65)
        for k in range(4):
            if base == 1:
                pts[base + k] = [x0 - 0.18 * s * k, y0 - length * s * k / 3, 0]
            else:
                pts[base + k] = [x0, y0 - length * s * k * 0.4, 0]
    return pts


def fist_with_index(s: float = 100) -> np.ndarray:
    """Middle/ring/pinky curl towards the camera *and across the index*; the index stays straight but sits
    farther from the camera in z than the curled fingers (the ka situation)."""
    p = open_hand(s=s)
    ix = p[5, 0]
    for base in (9, 13, 17):
        mcp = p[base].copy()
        p[base + 1] = mcp + [-0.05 * s, -0.35 * s, -0.15 * s]
        p[base + 2] = [ix + 0.05 * s, mcp[1] - 0.45 * s, -0.40 * s]
        p[base + 3] = [ix - 0.05 * s, mcp[1] - 0.15 * s, -0.35 * s]
    p[5:9, 2] += 0.10 * s
    return p


def mirrored(p: np.ndarray) -> np.ndarray:
    """Mirror x: the back of a right hand becomes the palm of a right hand."""
    q = p.copy()
    q[:, 0] = 1000 - q[:, 0]
    return q


def unit(p: np.ndarray) -> np.ndarray:
    return to_box_coords(p, crop_box([p], 1920, 1080))


def rows_width(img: np.ndarray, y: int) -> int:
    xs = np.where((img[y] < 245).any(axis=1))[0]
    return int(xs.max() - xs.min() + 1) if len(xs) else 0


# --- forearm ---------------------------------------------------------------------------------------------


def test_forearm_width_limited_by_visible_palm() -> None:
    p = unit(open_hand())
    visible = float(np.hypot(*(p[5, :2] - p[17, :2])))
    assert forearm_width(p[:, :2], hand_size(p)) <= FOREARM_MAX_OF_PALM * visible + 1e-9


def test_forearm_side_on_minimum() -> None:
    p = unit(open_hand())
    p[17, :2] = p[5, :2] + [0.001, 0]  # palm seen edge-on: almost no visible width
    size = hand_size(p)
    assert forearm_width(p[:, :2], size) == pytest.approx(FOREARM_MIN * size)


def test_forearm_top_corners_sit_on_palm_wrist_edges() -> None:
    p = unit(open_hand())
    a, _, _, b = forearm_polygon(p[:, :2], hand_size(p), length=2.0)

    def on_segment(q: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> bool:
        d = s1 - s0
        t = float((q - s0) @ d / (d @ d))
        return 0 <= t <= 1 and np.linalg.norm(s0 + t * d - q) < 1e-9

    assert on_segment(a, p[0, :2], p[17, :2]) and on_segment(b, p[0, :2], p[1, :2])


def test_forearm_is_short_and_fades_out() -> None:
    p = unit(open_hand())
    p = p * 0.5 + [0.25, 0.05, 0]  # smaller hand, high in the frame: room below for the whole forearm
    size_px = hand_size(p) * 360
    wrist_y, cx = p[0, 1] * 360, int(round(p[0, 0] * 360))
    img = np.asarray(draw_hand(p, only={"forearm"})).astype(int)

    def darkness(y: float) -> int:
        row = img[int(round(y)), cx - 5 : cx + 6]
        return int((255 - row).sum())

    solid = darkness(wrist_y + 0.5 * size_px)
    fading = darkness(wrist_y + (FOREARM_FADE_START + FOREARM_FADE_END) / 2 * size_px)
    assert solid > 0
    assert 0 < fading < solid
    assert (img[int(wrist_y + (FOREARM_FADE_END + 0.1) * size_px) :] >= 254).all()  # gone, well above the edge


def test_fade_never_touches_the_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hand pointing down over its own forearm: fingers lie inside the fade zone.
    p = unit(open_hand())
    p[:, 1] = 1.0 - p[:, 1]
    p = p * 0.6 + [0.2, 0.05, 0]
    monkeypatch.setattr(render, "SHADE_OFFSET", 0.0)
    hand_px = (np.asarray(draw_hand(p, only={"palm", "thumb", "index", "middle", "ring", "pinky"})) < 245).any(axis=2)
    faded = np.asarray(draw_hand(p))
    monkeypatch.setattr(render, "FOREARM_FADE_START", 1e6)
    monkeypatch.setattr(render, "FOREARM_FADE_END", 2e6)
    unfaded = np.asarray(draw_hand(p))
    assert not np.array_equal(faded, unfaded)  # the fade did something to the arm ...
    # ... but nothing visible to the hand. LANCZOS blends the dark outline with whatever lies just outside it, so
    # the 2-px rim of the hand mask is left out; the arm's drawn length also follows the fade end, which can change
    # what borders the fingers and so the cel shade: switch the shade off for this comparison.
    inner = hand_px.copy()
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        inner &= np.roll(hand_px, (dy, dx), axis=(0, 1))
    hand_px = inner
    diff = np.abs(faded[hand_px].astype(int) - unfaded[hand_px].astype(int)).sum(axis=1)
    # The arm's side slope depends on its drawn length (FOREARM_FLARE), so a few outline pixels where palm meets arm
    # shift by a pixel; the dark cartoon outline makes those visible, hence a share limit, not a max.
    assert (diff > 20).mean() < 0.005 and (diff > 0).mean() < 0.01


def test_drawn_forearm_narrower_than_drawn_palm() -> None:
    p = unit(open_hand())
    forearm = np.asarray(draw_hand(p, only={"forearm"}))
    palm = np.asarray(draw_hand(p, only={"palm"}))
    wrist_row = int(round(p[0, 1] * 360)) + 4
    palm_w = max(rows_width(palm, y) for y in range(360))
    assert 0 < rows_width(forearm, wrist_row) <= 0.85 * palm_w


# --- extended / curled ------------------------------------------------------------------------------------


def test_classify_open_hand_and_fist() -> None:
    assert all(classify_fingers([open_hand()])[0].values())
    fist = classify_fingers([fist_with_index()])[0]
    assert fist["index"] and fist["thumb"]
    assert not (fist["middle"] or fist["ring"] or fist["pinky"])


def test_classify_smooths_a_one_frame_dip() -> None:
    frames = [open_hand()] * 5
    bent = open_hand()
    bent[7] += [30, 0, -40]
    bent[8] += [60, 40, -60]
    seq = frames[:2] + [bent] + frames[2:]
    assert classify_fingers([bent])[0]["index"] is False
    assert all(f["index"] for f in classify_fingers(seq))


# --- layering ---------------------------------------------------------------------------------------------


def test_extended_finger_stays_visible_over_fist() -> None:
    p = unit(fist_with_index())
    colors = {"index": BLUE}
    alone = np.asarray(draw_hand(p, only={"index"}, part_colors=colors)).astype(int)
    full = np.asarray(draw_hand(p, part_colors=colors)).astype(int)

    def is_blue(img: np.ndarray) -> np.ndarray:
        return (img[..., 2] > 180) & (img[..., 0] < 90)

    mask = is_blue(alone)
    assert mask.sum() > 500
    assert is_blue(full)[mask].mean() >= 0.80


# --- nails ------------------------------------------------------------------------------------------------


def test_no_nails_on_palm_facing_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "DRAW_NAILS", True)  # the cartoon style has nails off; the facing logic stays
    palm_side = unit(mirrored(open_hand()))
    assert palm_facing(palm_side) == -1
    auto = np.asarray(draw_hand(palm_side))
    assert np.array_equal(auto, np.asarray(draw_hand(palm_side, facing=-1)))
    assert np.array_equal(auto, np.asarray(draw_hand(palm_side, facing=0)))  # unsure -> no nails either
    assert not np.array_equal(auto, np.asarray(draw_hand(palm_side, facing=1)))  # nails would change it


def test_back_facing_hand_gets_nails_only_on_extended_fingers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "DRAW_NAILS", True)
    back = unit(fist_with_index())
    assert palm_facing(back) == 1
    no_nails = np.asarray(draw_hand(back, facing=0)).astype(int)
    nails = np.asarray(draw_hand(back, facing=1)).astype(int)
    changed = np.argwhere(np.abs(nails - no_nails).sum(axis=2) > 0)
    tip_ys = [back[i, 1] * 360 for i in (4, 8)]  # extended thumb + index tips
    assert len(changed) > 0
    for y, x in changed:
        assert min(np.hypot(y - ty, x - back[i, 0] * 360) for ty, i in zip(tip_ys, (4, 8), strict=True)) < 30


def test_facing_track_smooths() -> None:
    back, palm = unit(open_hand()), unit(mirrored(open_hand()))
    assert facing_track([back] * 4 + [palm] + [back] * 4) == [1] * 9


# --- held frame -------------------------------------------------------------------------------------------


def test_held_frame_picks_still_segment_in_middle() -> None:
    steps = [0, 0, 0] + [20] * 8 + [0.5, 0.5, 0.5] + [20] * 6  # still at 0-2 (outside the middle) and 11-13
    xs = np.cumsum(steps)
    poses = [open_hand() + [x, 0, 0] for x in xs]
    assert 11 <= held_frame(poses) <= 13


def test_held_frame_ignores_still_frames_outside_middle() -> None:
    steps = [0, 0, 0, 0] + list(np.linspace(30, 5, 16))
    poses = [open_hand() + [x, 0, 0] for x in np.cumsum(steps)]
    n = len(poses)
    h = held_frame(poses)
    assert int(np.floor(n * 0.15)) <= h < int(np.ceil(n * 0.85))


def test_playback_is_slower_than_capture() -> None:
    assert PLAYBACK_FPS < CAPTURE_FPS


# --- thumb loop (O / beak handshapes) ------------------------------------------------------------------


def o_hand(s: float = 100) -> np.ndarray:
    """Index curls over to touch the thumb tip, leaving a round opening between them (va-like)."""
    p = open_hand(s=s)
    p[6] = [p[5, 0] - 0.40 * s, p[5, 1] - 0.62 * s, 0]
    p[7] = [p[4, 0] - 0.10 * s, p[5, 1] - 0.62 * s, 0]
    p[8] = [p[4, 0] - 0.06 * s, p[4, 1] - 0.08 * s, 0]
    return p


def test_thumb_loop_cuts_what_covers_the_opening() -> None:
    # Anything drawn over the loop that is not a thumb/finger (e.g. the widened palm shape) is cut away ...
    p = unit(o_hand()) * 1440
    xy = p[:, :2]
    img = Image.new("RGB", (1440, 1440), render.SKIN)
    out = np.asarray(render._cut_thumb_loop(img, xy, S=300.0, ow=20.0, drawn={}, finger="index"))
    cx, cy = (int(round(v)) for v in xy[[2, 3, 4, 8, 7, 6, 5]].mean(axis=0))
    assert (out[cy, cx] == render.WHITE).all()
    # ... and outlined, while the real palm stays skin.
    palm_centre = xy[[0, 5, 17]].mean(axis=0).round().astype(int)
    assert (out[palm_centre[1], palm_centre[0]] == render.SKIN).all()
    assert (out == np.array(render.OUTLINE, dtype=np.uint8)).all(axis=2).any()
    # A loop too small to read is left alone.
    tiny = np.asarray(render._cut_thumb_loop(img, xy * 0.05, S=300.0, ow=20.0, drawn={}, finger="index"))
    assert (tiny == render.SKIN).all()


def test_thumb_loop_keeps_fingers_and_is_off_by_default() -> None:
    p = unit(o_hand())
    plain = np.asarray(draw_hand(p))
    assert np.array_equal(plain, np.asarray(draw_hand(p, thumb_loop=None)))
    cut = np.asarray(draw_hand(p, thumb_loop="index"))
    assert np.array_equal(cut, np.asarray(draw_hand(p, thumb_loop="auto")))  # index = fingertip nearest the thumb
    fingers = (np.asarray(draw_hand(p, only={"thumb", "index"})) < 245).any(axis=2)
    # never cuts into the drawn thumb/index (a few anti-aliased edge pixels aside)
    newly_white = (cut >= 250).all(axis=2) & ~(plain >= 250).all(axis=2)
    assert (newly_white & fingers).sum() < 10
    with pytest.raises(ValueError):
        draw_hand(p, thumb_loop="thumb")


def test_thumb_loop_needs_the_fingertip_to_touch_the_thumb() -> None:
    # Transition frames: the index opens away from the thumb, so there is no loop and nothing is cut or outlined.
    q = o_hand()
    q[8] = q[7] + [0, -60, 0]
    p = unit(q)
    assert np.array_equal(np.asarray(draw_hand(p)), np.asarray(draw_hand(p, thumb_loop="index")))


def test_loop_through_lets_the_opening_cut_a_finger_lying_across_it() -> None:
    q = o_hand()
    # the middle finger (knuckle 9 stays on the palm) curls over so its last segments lie across the opening
    q[10:13] = q[[2, 3, 4, 8]].mean(axis=0) + np.array([[-10, 0, 0], [5, 0, 0], [20, 0, 0]])
    p = unit(q)

    def white(**kw) -> int:
        return int((np.asarray(draw_hand(p, thumb_loop="index", **kw)) >= 250).all(axis=2).sum())

    assert white(loop_through=["middle"]) > white() + 50  # the opening grows where the middle lay across it
