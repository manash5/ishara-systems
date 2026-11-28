"""Tests for the pure parts of the teaching-media pipeline (no video, no MediaPipe)."""

from pathlib import Path

import numpy as np
import pytest

from ishara_ml.nsl23 import resolve_video
from ishara_ml.teaching.render import OUTLINE, SKIN, draw_hand
from ishara_ml.teaching.track import Box, crop_box, pick_largest, smooth, to_box_coords, trim_and_fill


def pose(value: float) -> np.ndarray:
    return np.full((21, 3), value, dtype=np.float64)


def open_hand(cx: float = 500, cy: float = 500, s: float = 100) -> np.ndarray:
    """A flat, upright open right hand (pixel coords, palm facing the camera)."""
    pts = np.zeros((21, 3))
    pts[0] = [cx, cy + s, 0]
    bases = {1: -0.55, 5: -0.3, 9: -0.05, 13: 0.18, 17: 0.4}
    for base, dx in bases.items():
        length = 0.55 if base == 1 else (0.75 if base in (9, 13) else 0.65)
        x0 = cx + dx * s
        y0 = cy + (0.45 * s if base == 1 else 0)
        for k in range(4):
            if base == 1:
                pts[base + k] = [x0 - 0.18 * s * k, y0 - length * s * k / 3, 0]
            else:
                pts[base + k] = [x0, y0 - length * s * k / 3 * 1.2, 0]
    return pts


# --- trim / fill / smooth ---------------------------------------------------------------------------------


def test_trim_and_fill_drops_edges_and_holds_gaps() -> None:
    seq = [None, None, pose(1), None, None, pose(4), None]
    filled, first = trim_and_fill(seq)
    assert first == 2
    assert [p[0, 0] for p in filled] == [1, 1, 1, 4]


def test_trim_and_fill_returns_copies() -> None:
    p = pose(1)
    filled, _ = trim_and_fill([p, None])
    filled[0][0, 0] = 99
    assert p[0, 0] == 1


def test_trim_and_fill_no_hand() -> None:
    assert trim_and_fill([None, None]) == ([], 0)


def test_smooth_three_frame_window() -> None:
    out = smooth([pose(0), pose(3), pose(6), pose(30)], window=3)
    assert [round(p[0, 0], 6) for p in out] == [1.5, 3.0, 13.0, 18.0]


def test_smooth_single_frame_unchanged() -> None:
    assert smooth([pose(5)])[0][0, 0] == 5


def test_pick_largest() -> None:
    small, large = open_hand(s=50), open_hand(s=200)
    assert pick_largest([small, large]) is large
    assert pick_largest([]) is None


# --- crop box ---------------------------------------------------------------------------------------------


def test_crop_box_is_square_padded_and_contains_all_points() -> None:
    poses = [open_hand(900, 500, 100), open_hand(950, 520, 100)]
    box = crop_box(poses, 1920, 1080)
    pts = np.concatenate([p[:, :2] for p in poses])
    extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    assert box.side > extent * 1.25
    assert (pts[:, 0] >= box.x0).all() and (pts[:, 0] <= box.x0 + box.side).all()
    assert (pts[:, 1] >= box.y0).all() and (pts[:, 1] <= box.y0 + box.side).all()


def test_crop_box_extra_room_below() -> None:
    p = open_hand(900, 500, 100)
    box = crop_box([p], 1920, 1080)
    below = box.y0 + box.side - p[:, 1].max()
    above = p[:, 1].min() - box.y0
    assert below > above


def test_crop_box_clamped_to_frame() -> None:
    box = crop_box([open_hand(30, 1040, 100)], 1920, 1080)
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x0 + box.side <= 1920 and box.y0 + box.side <= 1080


def test_crop_box_never_larger_than_frame() -> None:
    box = crop_box([open_hand(320, 240, 400)], 640, 480)
    assert box.side <= 480 and box.y0 == 0


def test_to_box_coords() -> None:
    p = pose(150)
    u = to_box_coords(p, Box(100, 50, 200))
    assert u[0].tolist() == [0.25, 0.5, 0.75]


# --- renderer ---------------------------------------------------------------------------------------------


@pytest.fixture
def rendered() -> np.ndarray:
    hand = open_hand(500, 500, 100)
    box = crop_box([hand], 1920, 1080)
    return np.asarray(draw_hand(to_box_coords(hand, box), size=360))


def test_render_size_and_white_corners(rendered: np.ndarray) -> None:
    assert rendered.shape == (360, 360, 3)
    for y, x in [(0, 0), (0, 359), (359, 0)]:
        assert tuple(rendered[y, x]) == (255, 255, 255)


def test_render_not_blank_and_uses_palette(rendered: np.ndarray) -> None:
    non_white = (rendered < 250).any(axis=2).mean()
    assert 0.05 < non_white < 0.8
    flat = rendered.reshape(-1, 3).astype(int)
    for colour in (SKIN, OUTLINE):
        assert (np.abs(flat - colour).sum(axis=1) < 12).any()


def test_render_respects_size_option() -> None:
    hand = open_hand()
    img = draw_hand(to_box_coords(hand, crop_box([hand], 1920, 1080)), size=200)
    assert img.size == (200, 200)


# --- file finding used by the pipeline (S1 vs S10) ---------------------------------------------------------


def test_pipeline_lookup_keeps_s1_apart_from_s10(tmp_path: Path) -> None:
    for f in ["P1/S1_NSL_Consonant_Bright/S1_KA.MOV", "P2/S10_NSL_Consonant_Prepared/S1_KA.MOV"]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes(b"")
    found = resolve_video(tmp_path, "S1", "S1_NSL_Consonant_Bright/S1_KA.MOV")
    assert found.parent.name == "S1_NSL_Consonant_Bright"
    assert resolve_video(tmp_path, "S10", "S1_KA.MOV").parent.name == "S10_NSL_Consonant_Prepared"
