"""Raised vs folded fingers, overrides, no white gaps at finger bases, elbow-driven forearm, centred crop box."""

import json
from pathlib import Path

import numpy as np

from ishara_ml.teaching import pipeline
from ishara_ml.teaching.geometry import finger_states, hand_size, tip_reach
from ishara_ml.teaching.pose import fill_and_smooth
from ishara_ml.teaching.render import (
    EXTENDED_FACTOR,
    FINGER_RADIUS,
    OUTLINE_WIDTH,
    draw_hand,
    forearm_direction,
)
from ishara_ml.teaching.track import crop_box, to_box_coords


def open_hand(cx: float = 500, cy: float = 500, s: float = 100) -> np.ndarray:
    pts = np.zeros((21, 3))
    pts[0] = [cx, cy + s, 0]
    for base, dx in {1: -0.55, 5: -0.3, 9: -0.05, 13: 0.18, 17: 0.4}.items():
        x0, y0 = cx + dx * s, cy + (0.45 * s if base == 1 else 0)
        for k in range(4):
            if base == 1:
                pts[base + k] = [x0 - 0.18 * s * k, y0 - 0.55 * s * k / 3, 0]
            else:
                pts[base + k] = [x0, y0 - 0.3 * s * k, 0]
    return pts


def hooked(p: np.ndarray, base: int, s: float = 100) -> None:
    """Raise a finger, then hook its last two segments forward (towards the camera), like ah."""
    mcp = p[base]
    p[base + 1] = mcp + [0, -0.4 * s, 0]
    p[base + 2] = mcp + [0, -0.6 * s, -0.25 * s]
    p[base + 3] = mcp + [0, -0.5 * s, -0.45 * s]


def folded(p: np.ndarray, base: int, s: float = 100) -> None:
    mcp = p[base]
    p[base + 1] = mcp + [0, -0.2 * s, -0.2 * s]
    p[base + 2] = mcp + [0, 0.05 * s, -0.3 * s]
    p[base + 3] = mcp + [0, 0.2 * s, -0.15 * s]


def unit(p: np.ndarray) -> np.ndarray:
    return to_box_coords(p, crop_box([p], 1920, 1080))


def test_hooked_fingers_count_as_raised_folded_ones_do_not() -> None:
    p = open_hand()
    hooked(p, 5)
    folded(p, 13)
    size = hand_size([p])
    assert tip_reach(p, [5, 6, 7, 8], size) > tip_reach(p, [13, 14, 15, 16], size)
    state = finger_states([p], static=True)[0]
    assert state["index"] and not state["ring"] and state["thumb"]


def test_static_states_are_constant_over_the_clip() -> None:
    a, b = open_hand(), open_hand()
    folded(b, 9)  # one odd frame
    states = finger_states([a, a, b, a, a], static=True)
    assert len({s["middle"] for s in states}) == 1 and states[0]["middle"]


def test_overrides_force_a_finger_state() -> None:
    states = finger_states([open_hand()], static=True, overrides={"middle": False, "thumb": False})
    assert states[0]["middle"] is False and states[0]["thumb"] is True  # the thumb can't be folded away


def test_tta_override_file_is_valid() -> None:
    data = pipeline.load_overrides()
    fingers = {"index", "middle", "ring", "pinky"}
    for slug, fix in data.items():
        assert set(fix.get("fold", [])) <= fingers and set(fix.get("raise", [])) <= fingers
        assert fix.get("reason"), f"{slug}: every override needs a reason"
    raw = json.loads(Path(pipeline.OVERRIDES).read_text(encoding="utf-8"))
    assert "_doc" in raw and "_doc" not in data


def test_no_white_gap_beside_raised_finger_bases() -> None:
    """Just beside the base of a raised finger (inside its outline band, outside the palm's) must not be white.

    Before the outline underlay, the finger's outline started a little past the knuckle and left a white notch there.
    """
    p = open_hand()
    for b in (9, 13, 17):
        folded(p, b)
    u = unit(p)
    img = np.asarray(draw_hand(u, size=360)).astype(int)
    S = hand_size([u]) * 360
    ow = OUTLINE_WIDTH * S
    for name, (k0, k1) in {"index": (5, 6), "thumb": (2, 3)}.items():
        r = FINGER_RADIUS[name] * S * EXTENDED_FACTOR
        a, b = u[k0, :2] * 360, u[k1, :2] * 360
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        for side in (1, -1):
            x, y = (a + d * 0.3 * r + side * n * (r + 0.5 * ow)).round().astype(int)
            assert img[y, x].min() < 245, f"white gap beside the {name} base"


def test_forearm_points_at_the_elbow() -> None:
    xy = open_hand()[:, :2]
    elbow = xy[0] + [300, 50]  # off to the right, unlike the hand's own axis (straight down)
    d = forearm_direction(xy, elbow)
    assert d[0] > 0.9
    assert forearm_direction(xy)[1] > 0.9  # without an elbow: away from the knuckles


def test_elbow_track_fills_gaps_and_smooths() -> None:
    pts = [None, np.array([10.0, 0.0]), None, np.array([20.0, 0.0])]
    out = fill_and_smooth(pts, window=1)
    assert [float(p[0]) for p in out] == [10.0, 10.0, 10.0, 20.0]
    assert fill_and_smooth([None, None], window=3) is None


def test_unclamped_box_stays_centred_near_the_frame_edge() -> None:
    p = open_hand(30, 1040)
    box = crop_box([p], 1920, 1080, clamp=False)
    hand_cx = (p[:, 0].min() + p[:, 0].max()) / 2
    assert abs(box.x0 + box.side / 2 - hand_cx) < 1e-6
    clamped = crop_box([p], 1920, 1080)
    assert clamped.x0 >= 0 and abs(clamped.x0 + clamped.side / 2 - hand_cx) > 1


def test_photo_region_covers_hand_and_fades_the_forearm() -> None:
    from ishara_ml.teaching.photo import crop_white, hand_region
    from ishara_ml.teaching.track import Box

    p = open_hand(500, 400)
    box = crop_box([p], 1920, 1080, clamp=False)
    size_px = hand_size([p])
    region = np.asarray(hand_region(p, np.array([500.0, 900.0]), box, size_px)).astype(int)
    for i in (0, 4, 8, 12, 20):  # wrist and fingertips are inside
        x, y = (p[i, :2] - [box.x0, box.y0]).round().astype(int)
        assert region[y, x] > 200
    x, y = (np.array([500.0, 500 + 100 + 1.7 * size_px]) - [box.x0, box.y0]).round().astype(int)
    if 0 <= y < region.shape[0]:
        assert region[y, x] == 0  # forearm faded out past the fade end

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = np.asarray(crop_white(frame, Box(-20, -20, 60)))
    assert (out[:20, :20] == 255).all() and (out[30:, 30:] == 0).all()  # outside the frame is white


def test_per_letter_style_comes_from_overrides() -> None:
    for slug, fix in pipeline.load_overrides().items():
        assert fix.get("style", "illustrated") in pipeline.STYLES, slug
