"""Real-hand cut-out: mask, rendering, trimming to the sign, held frame (synthetic data, no rembg / MediaPipe)."""

import numpy as np
from PIL import Image

from ishara_ml.teaching.cutout import fit_skin, hand_mask, hand_region, smooth_alpha, temporal_median
from ishara_ml.teaching.geometry import hand_size, held_frame, sign_segment
from ishara_ml.teaching.realhand import durations_ms
from ishara_ml.teaching.stylise import photo_frame, tone_thresholds, toon_frame

SKIN = (205, 145, 115)
SLEEVE = (45, 55, 95)  # patterned blue top
WALL = (222, 212, 188)  # beige wall, close to skin in brightness
FPS = 12.0


def open_hand(cx: float = 200, cy: float = 190, s: float = 70) -> np.ndarray:
    pts = np.zeros((21, 3))
    pts[0] = [cx, cy + s, 0]
    for base, dx in {1: -0.55, 5: -0.3, 9: -0.05, 13: 0.18, 17: 0.4}.items():
        x0, y0 = cx + dx * s, cy + (0.45 * s if base == 1 else 0)
        for k in range(4):
            pts[base + k] = [x0 - (0.18 * s * k if base == 1 else 0), y0 - (0.18 if base == 1 else 0.3) * s * k, 0]
    return pts


def scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A 400 px crop: beige wall, a skin-coloured hand + forearm, a blue sleeve on the forearm, and a wall patch that
    the person segmentation wrongly includes next to the hand."""
    import cv2

    img = np.full((400, 400, 3), WALL, dtype=np.uint8)
    pose = open_hand()
    hand = np.zeros((400, 400), dtype=np.uint8)
    cv2.fillConvexPoly(hand, cv2.convexHull(pose[:, :2].round().astype(np.int32)), 1)
    for chain in ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
        cv2.polylines(hand, [pose[chain, :2].round().astype(np.int32)], False, 1, thickness=16)
    cv2.rectangle(hand, (175, 255), (225, 330), 1, -1)  # forearm (skin)
    img[hand > 0] = SKIN
    sleeve = np.zeros_like(hand)
    cv2.rectangle(sleeve, (165, 330), (235, 400), 1, -1)
    img[sleeve > 0] = SLEEVE
    person = (hand > 0) | (sleeve > 0)
    # Beige wall right next to the hand, inside the hand region, that the person segmentation wrongly includes:
    # only the skin colour can exclude it.
    region = hand_region((400, 400), pose[:, :2], np.array([0.0, 1.0]), hand_size([pose]))
    wall_patch = region & (hand == 0) & (sleeve == 0)
    person |= wall_patch
    return img, person, pose, wall_patch


def test_mask_excludes_sleeve_and_beige_wall() -> None:
    img, person, pose, wall_patch = scene()
    xy = pose[:, :2]
    skin = fit_skin([img], [xy])
    m = hand_mask(img, person, xy, np.array([0.0, 1.0]), hand_size([pose]), skin)
    assert wall_patch.sum() > 2000  # the trap is big and sits inside person ∩ hand region
    assert m[190, 200] and m[300, 200]  # palm and forearm kept
    assert not m[360:, 170:230].any()  # sleeve excluded
    assert m[wall_patch].mean() < 0.02  # beige wall excluded although "person" said yes
    assert not m[:40].any()  # far wall excluded


def test_temporal_median_removes_one_frame_flicker() -> None:
    a = np.zeros((10, 10), dtype=bool)
    b = a.copy()
    b[5, 5] = True
    assert not temporal_median([a, b, a])[1][5, 5]
    # At the ends, too: a blob only in the first frame must not survive (no 2-frame "either" vote).
    out = temporal_median([b, a, a, a])
    assert not out[0][5, 5] and not out[1][5, 5]


def test_outputs_are_rgb_with_white_corners() -> None:
    img, person, pose, _ = scene()
    xy = pose[:, :2]
    size = hand_size([pose])
    m = hand_mask(img, person, xy, np.array([0.0, 1.0]), size, fit_skin([img], [xy]))
    alpha = smooth_alpha(m, size)
    toon = toon_frame(img, alpha, tone_thresholds([img], [alpha]), np.zeros_like(m), size, 180)
    photo = photo_frame(img, alpha, 180)
    for out in (toon, photo):
        assert isinstance(out, Image.Image) and out.mode == "RGB" and out.size == (180, 180)
        a = np.asarray(out)
        for y, x in [(0, 0), (0, 179), (179, 0)]:
            assert tuple(a[y, x]) == (255, 255, 255)
        assert (a < 250).any(axis=2).mean() > 0.05  # the hand is there


def clip(heights: list[float]) -> list[np.ndarray]:
    return [open_hand(200, y) for y in heights]


def rest_sign_hold_rest() -> tuple[list[np.ndarray], int, int]:
    rest = [900.0] * 6
    rise = list(np.linspace(900, 260, 6))[1:]  # arm rising fast from rest
    settle = [250.0, 244.0, 241.0, 240.0]  # slowing into the shape
    hold = [240.0 + 0.3 * np.sin(i) for i in range(14)]
    lower = list(np.linspace(240, 900, 6))[1:]
    heights = rest + rise + settle + hold + lower + [900.0] * 6
    hold_start = len(rest) + len(rise) + len(settle)
    return clip(heights), hold_start, hold_start + len(hold)


def test_trim_static_sign_drops_rise_and_lowering() -> None:
    poses, hold_start, hold_end = rest_sign_hold_rest()
    a, b = sign_segment(poses, FPS, moving=False)
    assert hold_start - 6 <= a <= hold_start  # at most a short settle before the hold, never the rise from rest
    assert a >= 11  # rest (6) + rise (5) are gone
    assert hold_end - 2 <= b <= hold_end + 1  # the lowering is gone


def test_trim_movement_keeps_movement_drops_lowering() -> None:
    approach = list(np.linspace(900, 300, 5))
    pause = [300.0] * 3
    movement = list(np.linspace(300, 200, 8))  # the sign's own (slower) movement
    final = [200.0] * 6
    lower = list(np.linspace(200, 900, 5))
    poses = clip(approach + pause + movement + final + lower)
    a, b = sign_segment(poses, FPS, moving=True)
    end_of_final = len(approach + pause + movement + final)
    assert a <= len(approach + pause)  # starts before the movement
    assert end_of_final - 2 <= b <= end_of_final + 1


def test_held_frame_is_inside_the_hold() -> None:
    poses, hold_start, hold_end = rest_sign_hold_rest()
    a, b = sign_segment(poses, FPS, moving=False)
    k = a + held_frame(poses[a:b])
    assert hold_start <= k < hold_end


def test_final_pose_is_held_about_a_second() -> None:
    d = durations_ms(10)
    assert d[:-1] == [83] * 9 and d[-1] == 83 + 1000
