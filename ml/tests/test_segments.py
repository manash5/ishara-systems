"""Which part of a clip is animated: the held shape (static signs) or the movement without the approach."""

import numpy as np

from ishara_ml.teaching.geometry import hold_segment, movement_segment, speed_profile
from ishara_ml.teaching.pipeline import (
    END_HOLD_MS,
    MOVEMENT_SIGNS,
    START_HOLD_MS,
    STATIC_END_HOLD_MS,
    STATIC_START_HOLD_MS,
    frame_durations,
    smooth_window,
)

FPS = 24.0


def hand(x: float, y: float) -> np.ndarray:
    """A rigid 21-point hand (size ~100 px) translated to (x, y)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(-50, 50, size=(21, 3))
    base[:, 2] = 0
    return base + [x, y, 0]


def path(xs: list[float]) -> list[np.ndarray]:
    return [hand(x, 500) for x in xs]


def approach_then_hold(approach: int = 8, hold: int = 30) -> list[np.ndarray]:
    xs = list(np.linspace(0, 800, approach)) + [800.0 + 0.3 * np.sin(i) for i in range(hold)]
    return path(xs)


def test_hold_segment_drops_the_approach() -> None:
    poses = approach_then_hold()
    a, b = hold_segment(poses, FPS)
    assert a >= 8 and b - a >= round(0.5 * FPS)
    assert speed_profile(poses[a:b], FPS).max() < 1.0


def test_hold_segment_is_capped() -> None:
    poses = approach_then_hold(hold=100)
    a, b = hold_segment(poses, FPS, max_s=1.5)
    assert b - a == round(1.5 * FPS)


def test_hold_segment_loop_jump_is_small() -> None:
    poses = approach_then_hold()
    a, b = hold_segment(poses, FPS)
    jump = np.linalg.norm(poses[b - 1][:, :2] - poses[a][:, :2], axis=1).mean()
    assert jump < 2.0  # px, on a ~100 px hand


def test_hold_segment_falls_back_when_never_still() -> None:
    poses = path(list(np.linspace(0, 3000, 40)))
    a, b = hold_segment(poses, FPS, fallback_s=1.0)
    assert b - a == round(FPS)


def test_movement_segment_cuts_approach_keeps_movement() -> None:
    approach = list(np.linspace(0, 600, 6))  # fast
    settle = [600.0] * 4  # pause: the local minimum
    movement = list(np.linspace(600, 900, 12))  # the sign's own movement
    poses = path(approach + settle + movement + [900.0] * 6)
    a, b = movement_segment(poses, FPS)
    assert 5 <= a <= 10 and b == len(poses)
    kept = speed_profile(poses[a:b], FPS)
    assert kept.max() > 3.0  # the sign's movement is still there


def test_movement_segment_keeps_clip_without_approach() -> None:
    poses = path([500.0] * 6 + list(np.linspace(500, 800, 12)) + [800.0] * 6)
    assert movement_segment(poses, FPS) == (0, len(poses))


def test_frame_durations() -> None:
    static = frame_durations(10, 16, moving=False)
    assert static[0] == 62 + STATIC_START_HOLD_MS and static[-1] == 62 + STATIC_END_HOLD_MS
    assert static[1:-1] == [62] * 8
    moving = frame_durations(10, 16, moving=True)
    assert moving[0] == 62 + START_HOLD_MS and moving[-1] == 62 + END_HOLD_MS and moving[1:-1] == [62] * 8


def test_smooth_window_is_odd_and_scales() -> None:
    assert smooth_window(12) == 3 and smooth_window(24) == 5
    assert all(smooth_window(f) % 2 for f in (8, 12, 16, 24, 30, 60))


def test_movement_signs_are_the_brief_vowels() -> None:
    assert MOVEMENT_SIGNS == {"aa", "u", "uu", "e", "ai", "o", "au", "am", "ah"}
