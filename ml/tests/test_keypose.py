"""Key-pose animation: kinematics, slerp, hold detection, robust averaging, forearm and output format."""

import math

import numpy as np
import pytest

from ishara_ml.teaching import kinematics as K
from ishara_ml.teaching.keypose import (
    KeyPose,
    KeyPoseDoc,
    longest_hold,
    motion_speed,
    neutral_pose,
    robust_mean_pose,
)
from ishara_ml.teaching.keypose_pipeline import loop_frames, render_doc
from ishara_ml.teaching.keypose_render import Frame, lerp_frame
from ishara_ml.teaching.render import draw_hand, forearm_edge_polygon
from ishara_ml.teaching.timeline import Timing, build_timeline

FPS = 24.0


def world_hand(curl: float = 0.0, thumb_in: float = 0.0) -> np.ndarray:
    """A right hand in metres (camera axes), fingers up (-y), palm towards the camera. ``curl`` bends every finger
    joint by that many radians towards the camera; ``thumb_in`` swings the thumb across the palm."""
    p = np.zeros((21, 3))
    bases = {1: (0.025, -0.02), 5: (0.022, -0.085), 9: (0.0, -0.09), 13: (-0.02, -0.085), 17: (-0.038, -0.075)}
    lengths = {
        1: (0.04, 0.032, 0.028),
        5: (0.04, 0.025, 0.022),
        9: (0.045, 0.028, 0.024),
        13: (0.042, 0.026, 0.022),
        17: (0.032, 0.02, 0.019),
    }
    for base, (x, y) in bases.items():
        p[base] = [x, y, 0.0]
        d = np.array([0.6, -0.8, 0.0]) if base == 1 else np.array([0.0, -1.0, 0.0])
        if base == 1 and thumb_in:
            d = np.array([math.cos(thumb_in) * 0.6 - 0.8 * math.sin(thumb_in), -0.8, -0.3 * math.sin(thumb_in)])
        for k in range(3):
            ang = curl * (k + 1) if base != 1 else 0.3 * curl * (k + 1)
            dk = np.array([d[0], d[1] * math.cos(ang), -math.sin(ang)])  # bend towards the camera (-z)
            dk = dk / np.linalg.norm(dk)
            p[base + k + 1] = p[base + k] + dk * lengths[base][k]
    return p


@pytest.fixture(scope="module")
def skeleton() -> K.Skeleton:
    return K.build_skeleton([world_hand(c) for c in (0.0, 0.3, 0.6, 1.0)], "test")


def bone_lengths(pts: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(pts[c[k + 1]] - pts[c[k]]) for c in K.CHAINS.values() for k in range(3)])


def rotated(points: np.ndarray, axis, angle: float, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    r = K.q_to_matrix(K.q_from_axis_angle(np.array(axis, dtype=float), angle))
    return points @ r.T + np.array(shift)


# --- kinematics ----------------------------------------------------------------------------------------------


def test_fk_keeps_bone_lengths_during_interpolation(skeleton: K.Skeleton) -> None:
    a = K.pose_to_rotations(world_hand(0.0), skeleton)
    b = K.pose_to_rotations(rotated(world_hand(1.2, thumb_in=1.0), [1, 2, 0.5], 1.3, (0.05, 0.02, 0)), skeleton)
    expected = np.concatenate([skeleton.lengths[n] for n in K.CHAINS])
    palm = {i: np.linalg.norm(skeleton.palm[i] - skeleton.palm[0]) for i in K.PALM_POINTS}
    for t in np.linspace(0, 1, 21):
        pts = K.forward_kinematics(K.interpolate(a, b, float(t)), skeleton)
        np.testing.assert_allclose(bone_lengths(pts), expected, atol=1e-9)
        for i, d in palm.items():
            assert abs(np.linalg.norm(pts[i] - pts[0]) - d) < 1e-9


def test_rotations_round_trip_through_fk(skeleton: K.Skeleton) -> None:
    src = rotated(world_hand(0.7, thumb_in=0.6), [0.3, 1, 0.2], 0.8)
    pose = K.pose_to_rotations(src, skeleton)
    again = K.pose_to_rotations(K.forward_kinematics(pose, skeleton), skeleton)
    for n in K.CHAINS:
        for qa, qb in zip(pose.bones[n], again.bones[n], strict=True):
            assert K.q_angle(qa, qb) < 1e-6
    # bone directions of the rebuilt hand match the source hand
    fk = K.forward_kinematics(pose, skeleton)
    for c in K.CHAINS.values():
        for k in range(3):
            u, v = src[c[k + 1]] - src[c[k]], fk[c[k + 1]] - fk[c[k]]
            assert float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)) > 0.9999


def test_angles_round_trip(skeleton: K.Skeleton) -> None:
    pose = K.pose_to_rotations(rotated(world_hand(0.9, thumb_in=0.8), [1, 0, 0], 0.4), skeleton)
    rebuilt = K.pose_from_angles(K.pose_angles(pose, skeleton), pose.wrist_q, pose.wrist_pos, skeleton)
    a, b = K.forward_kinematics(pose, skeleton), K.forward_kinematics(rebuilt, skeleton)
    assert np.abs(a - b).max() < 0.002  # angles are stored to 0.1 degree


def test_slerp_midpoint() -> None:
    a = K.IDENTITY
    b = K.q_from_axis_angle(np.array([0, 0, 1.0]), math.pi / 2)
    mid = K.slerp(a, b, 0.5)
    assert abs(np.linalg.norm(mid) - 1) < 1e-12
    assert abs(K.q_angle(mid, a) - math.pi / 4) < 1e-9
    assert abs(K.q_angle(mid, b) - math.pi / 4) < 1e-9
    # -b is the same rotation: slerp must still take the short (45 degree) path
    mid2 = K.slerp(a, -b, 0.5)
    assert abs(K.q_angle(mid2, a) - math.pi / 4) < 1e-9
    np.testing.assert_allclose(K.slerp(a, b, 0.0), a, atol=1e-12)
    assert K.q_angle(K.slerp(a, b, 1.0), b) < 1e-9


# --- hold and averaging --------------------------------------------------------------------------------------


def test_hold_detection_move_hold_move() -> None:
    rng = np.random.default_rng(1)
    base = world_hand(0.5)
    xs = list(np.linspace(0, 0.3, 12)) + [0.3] * 20 + list(np.linspace(0.3, 0.0, 12))
    frames = np.stack([base + [x, 0, 0] + rng.normal(0, 0.0003, base.shape) for x in xs])
    speed = motion_speed(frames, 0.04, FPS)
    hold = longest_hold(speed, 1.0, 7)
    assert hold is not None
    a, b = hold
    assert abs(a - 12) <= 1 and abs(b - 32) <= 1


def test_hold_detection_none_when_always_moving() -> None:
    frames = np.stack([world_hand() + [x, 0, 0] for x in np.linspace(0, 1, 30)])
    assert longest_hold(motion_speed(frames, 0.04, FPS), 1.0, 7) is None


def test_robust_average_ignores_outlier_frame() -> None:
    clean = rotated(world_hand(0.6), [0, 1, 0], 0.3, (0.1, 0.0, 0.0))
    frames = np.stack([clean] * 6 + [rotated(world_hand(1.4, thumb_in=1.2), [0, 1, 0], 0.3, (0.1, 0, 0))])
    out, keep = robust_mean_pose(frames)
    assert keep.tolist() == [True] * 6 + [False]
    np.testing.assert_allclose(out, clean, atol=1e-9)


# --- timeline -------------------------------------------------------------------------------------------------


def test_timeline_starts_and_ends_on_neutral() -> None:
    frames = build_timeline(0.0, [1.0, 2.0], lambda a, b, t: a + (b - a) * t, Timing(), 24)
    assert frames[0] == frames[-1] == 0.0
    assert max(frames) == 2.0
    assert len(frames) == 6 + 10 + 10 + 24 + 7
    steps = np.abs(np.diff(frames))
    assert steps.max() < 0.5  # eased: no jumps (the 2-unit return over 7 frames peaks at ~0.45)
    assert steps[6] < steps[10]  # ease-in: the first step of a transition is the smallest


def test_loop_has_no_seam(skeleton: K.Skeleton) -> None:
    kp = K.pose_to_rotations(world_hand(1.0), skeleton)
    doc = KeyPoseDoc("t", "static", {"fps": 24, "file": "x"}, "right", [KeyPose("kp1", kp, 90.0, (0, 5), 5)])
    frames = loop_frames(doc, skeleton)
    a, b = K.forward_kinematics(frames[0].pose, skeleton), K.forward_kinematics(frames[-1].pose, skeleton)
    assert np.abs(a - b).max() < 1e-9
    n = neutral_pose(skeleton, kp)
    assert K.q_angle(n.wrist_q, kp.wrist_q) < 1e-9


# --- drawing --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("wrist", [(0.5, 0.75), (0.2, 0.9), (0.85, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("deg", [90, 60, 135, 0, 180, -90])
def test_forearm_never_ends_inside_the_frame(wrist, deg) -> None:
    big = 1440
    hand = world_hand()
    xy = (hand[:, :2] - hand[0, :2]) * 3000 + np.array(wrist) * big
    away = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    poly = np.array(forearm_edge_polygon(xy, 120.0, away, length=3.0 * big))
    far = poly[len(poly) // 2 - 1 : len(poly) // 2 + 1]  # the two far-end points
    for pt in far:
        assert not (0 <= pt[0] <= big and 0 <= pt[1] <= big)
    # and the drawn arm reaches the image border along its direction
    u = np.zeros((21, 3))
    u[:, :2] = xy / big
    img = np.asarray(
        draw_hand(u, size=360, hand_scale=0.08, forearm="edge", forearm_dir=away, arm_width=0.08, only={"forearm"})
    )
    t = np.linspace(0, 2, 2000)
    line = np.array(wrist)[None] + t[:, None] * away[None]
    inside = line[(line >= 0).all(axis=1) & (line < 1).all(axis=1)]
    ex, ey = (inside[-1] * 360).astype(int).clip(0, 359)
    lo_x, hi_x, lo_y, hi_y = max(ex - 40, 0), min(ex + 41, 360), max(ey - 40, 0), min(ey + 41, 360)
    border = np.concatenate(  # the frame edge near where the arm's centre line leaves the image
        [img[0, lo_x:hi_x], img[359, lo_x:hi_x], img[lo_y:hi_y, 0], img[lo_y:hi_y, 359]]
    )
    assert (border < 200).any(axis=1).any()


def test_rendered_frames_rgb_white_corners(skeleton: K.Skeleton) -> None:
    kp = K.pose_to_rotations(world_hand(1.0), skeleton)
    doc = KeyPoseDoc("t", "static", {"fps": 24, "file": "x"}, "right", [KeyPose("kp1", kp, 90.0, (0, 5), 5)])
    images, poster = render_doc(doc, skeleton)
    for img in [images[0], images[len(images) // 2], poster]:
        assert img.mode == "RGB" and img.size == (360, 360)
        a = np.asarray(img)
        for y, x in ((0, 0), (0, 359)):  # top corners: the forearm leaves through the bottom
            assert (a[y, x] == 255).all()


def test_lerp_frame_turns_the_short_way(skeleton: K.Skeleton) -> None:
    kp = K.pose_to_rotations(world_hand(), skeleton)
    f = lerp_frame(Frame(kp, 170.0), Frame(kp, -170.0), 0.5)
    assert abs(((f.forearm_deg + 180) % 360) - 180 - 180) < 1e-9 or abs(abs(f.forearm_deg) - 180) < 1e-9
