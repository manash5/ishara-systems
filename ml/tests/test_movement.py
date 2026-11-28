"""Letters with movement: sharp-frame handshape, framing over all frames, forearm from the elbow, per-frame checks,
and that static letters never touch the movement code path."""

import math

import numpy as np
import pytest
from test_keypose import world_hand

from ishara_ml.teaching import keypose as kp_mod
from ishara_ml.teaching import keypose_pipeline, keypose_render
from ishara_ml.teaching import kinematics as K
from ishara_ml.teaching import movement as mv
from ishara_ml.teaching.keypose import KeyPose, KeyPoseDoc, Track, analyse
from ishara_ml.teaching.keypose_pipeline import movement_frames, render_doc
from ishara_ml.teaching.keypose_render import VIEW_SIDE_M, Frame, render_movement, to_unit
from ishara_ml.teaching.render import draw_hand

FPS = 24.0
PX_PER_M = 2500.0


@pytest.fixture(scope="module")
def skeleton() -> K.Skeleton:
    return K.build_skeleton([world_hand(c) for c in (0.0, 0.3, 0.6, 1.0)], "test")


def synthetic_track(slug: str, xs_px: list[float], *, sharp=None, arm_deg: float | None = 30.0) -> Track:
    """A hand (fingers up) translating along x in the image; MediaPipe-like world points; Pose arm at arm_deg."""
    n = len(xs_px)
    world = np.stack([world_hand(0.2) for _ in range(n)])
    image = np.zeros((n, 21, 3))
    for i, x in enumerate(xs_px):
        image[i, :, :2] = (world[i, :, :2] - world[i, 0, :2]) * PX_PER_M + [x, 700.0]
        image[i, :, 2] = world[i, :, 2] * PX_PER_M
    arm = np.full((n, 4), np.nan)
    if arm_deg is not None:
        d = np.array([math.cos(math.radians(arm_deg)), math.sin(math.radians(arm_deg))]) * 400
        arm[:, 2:] = image[:, 0, :2]  # Pose wrist
        arm[:, :2] = image[:, 0, :2] + d  # Pose elbow
    return Track(
        slug, "x.mov", "S1", FPS, (1920, 1080), None, image, world, np.full((n, 2), np.nan),
        ["Right"] * n, np.full(n, 0.99), np.full(n, 250.0) if sharp is None else np.asarray(sharp, float), arm,
    )  # fmt: skip


# --- sharp-frame selection -------------------------------------------------------------------------------------


def test_sharp_selection_ignores_blurred_frames() -> None:
    n = 30
    speed = np.full(n, 0.5)
    sharp = np.full(n, 280.0)
    blurred = [3, 8, 9, 10, 20, 27]
    sharp[blurred] = 12.0  # motion/defocus blur: Laplacian variance collapses
    speed[[15, 16]] = 9.0  # fast frames, sharp-looking or not, are excluded too
    score = np.full(n, 0.99)
    score[22] = 0.3  # low-confidence detection
    mask, how = mv.select_sharp(speed, score, sharp, np.ones(n, dtype=bool))
    assert how.startswith("strict")
    assert not mask[blurred].any() and not mask[[15, 16, 22]].any()
    assert mask.sum() == n - len(blurred) - 3


def test_sharp_selection_falls_back_to_slowest_frames() -> None:
    speed = np.array([12.0, 11.0, 10.0, 1.0, 1.2, 0.8, 1.1, 0.9])
    sharp = np.array([120.0, 130.0, 125.0, 12.0, 11.0, 13.0, 12.5, 10.0])  # sharp only while moving fast
    mask, how = mv.select_sharp(speed, np.full(8, 0.99), sharp, np.ones(8, dtype=bool))
    assert how.startswith("fallback")
    assert not mask[:3].any() and mask.sum() >= 3


def test_second_handshape_only_for_a_clear_consistent_change() -> None:
    rng = np.random.default_rng(0)
    a = K.to_palm_local(world_hand(0.0))
    b = a.copy()
    b[6:9] = K.to_palm_local(world_hand(1.4))[6:9]  # only the index finger bends: one finger counts fully
    noisy = lambda p: p + rng.normal(0, 0.001, p.shape)  # noqa: E731
    size = 0.04
    same = np.stack([noisy(a) for _ in range(12)])
    assert mv.split_handshapes(same, size) is None
    change = np.stack([noisy(a) for _ in range(6)] + [noisy(b) for _ in range(6)])
    assert mv.split_handshapes(change, size) == 6


# --- framing ----------------------------------------------------------------------------------------------------


def _moving_doc(skeleton: K.Skeleton, dx: float) -> KeyPoseDoc:
    hs = K.pose_to_rotations(world_hand(0.3), skeleton)
    hs.wrist_pos = np.zeros(3)
    path = [mv.PathPoint(t, t * FPS, np.array([dx * t / 1.0, -0.3 * dx * t, 0.0]), hs.wrist_q, 100.0, 0.0)
            for t in (0.0, 0.5, 1.0)]  # fmt: skip
    return KeyPoseDoc(
        "m", "movement", {"fps": FPS, "file": "x"}, "right", [KeyPose("hs1", hs, 0.0, (0, 5), 5)], path=path
    )


@pytest.mark.parametrize("dx", [0.02, 0.15, 0.40])
def test_framing_keeps_a_moving_hand_fully_inside(skeleton: K.Skeleton, dx: float) -> None:
    frames, _ = movement_frames(_moving_doc(skeleton, dx), skeleton)
    pts = [K.forward_kinematics(f.pose, skeleton) for f in frames]
    centre, side = mv.movement_view(pts, skeleton, VIEW_SIDE_M)
    r = mv.joint_margin(skeleton) / side
    for p in pts:
        u = to_unit(p, centre, side)[:, :2]
        assert (u >= r).all() and (u <= 1 - r).all()
    result = render_movement(frames, skeleton, size=120)  # runs every per-frame check
    if dx == 0.02:
        assert result.scale == 1.0  # fits: same scale as the static letters
    else:
        assert result.scale < 1.0  # shrunk as little as needed
        extent = max(np.ptp(np.concatenate([p[:, 0] for p in pts])), np.ptp(np.concatenate([p[:, 1] for p in pts])))
        assert side == pytest.approx((extent + 2 * mv.joint_margin(skeleton)) / mv.MOVE_FIT, rel=0.02)


# --- forearm ----------------------------------------------------------------------------------------------------


def test_forearm_direction_comes_from_the_elbow(skeleton: K.Skeleton) -> None:
    xs = [800.0] * 6 + list(np.linspace(800, 1100, 10)) + [1100.0] * 14
    doc = kp_mod.analyse_movement(synthetic_track("u", xs, arm_deg=30.0), skeleton)
    assert doc.analysis["forearmSource"] == "pose elbow -> pose wrist"
    for pt in doc.path:  # the hand's own axis would say 90 (fingers up -> arm straight down)
        assert abs(pt.forearm_deg - 30.0) < 2.0


def test_forearm_never_points_along_the_fingers() -> None:
    hand = world_hand()  # fingers up (-y)
    for deg in range(0, 360, 15):
        away = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        out, _ = mv.constrain_forearm(hand, away)
        fingers = hand[9, :2] - hand[0, :2]
        fingers /= np.linalg.norm(fingers)
        angle = math.degrees(math.acos(float(np.clip(out @ fingers, -1, 1))))
        assert angle >= 180 - mv.MAX_WRIST_BEND_DEG - 1e-6


# --- per-frame checks -------------------------------------------------------------------------------------------


def test_frame_checks_fail_loudly_with_the_frame_number(skeleton: K.Skeleton) -> None:
    pose = K.pose_to_rotations(world_hand(0.3), skeleton)
    pts = K.forward_kinematics(pose, skeleton)
    centre = pts[:, :2].mean(axis=0)
    mv.check_frame(0, pts, to_unit(pts, centre, 0.3), skeleton, 0.3)  # fine
    with pytest.raises(mv.MovementCheckError, match="frame 7: hand leaves the frame"):
        mv.check_frame(7, pts, to_unit(pts, centre + [0.12, 0], 0.3), skeleton, 0.3)
    stretched = pts.copy()
    stretched[8] += [0.01, 0, 0]
    with pytest.raises(mv.MovementCheckError, match="frame 3: index bone 2"):
        mv.check_frame(3, stretched, to_unit(stretched, centre, 0.3), skeleton, 0.3)
    side_on = K.HandPose(K.q_from_axis_angle(np.array([0, 1.0, 0]), math.pi / 2), np.zeros(3), pose.bones)
    sp = K.forward_kinematics(side_on, skeleton)
    with pytest.raises(mv.MovementCheckError, match="frame 4: visible palm area"):
        mv.check_frame(4, sp, to_unit(sp, sp[:, :2].mean(axis=0), 0.3), skeleton, 0.3)


def test_sampler_hits_control_points_and_blends_handshapes(skeleton: K.Skeleton) -> None:
    a = K.pose_to_rotations(world_hand(0.0), skeleton)
    b = K.pose_to_rotations(world_hand(1.2), skeleton)
    q2 = K.q_from_axis_angle(np.array([0, 0, 1.0]), 0.6)
    pts = [
        mv.PathPoint(0.0, 0, np.zeros(3), a.wrist_q, 90.0, 0.0),
        mv.PathPoint(0.5, 12, np.array([0.05, 0.0, 0.0]), q2, 60.0, 0.5),
        mv.PathPoint(1.0, 24, np.array([0.08, 0.02, 0.0]), q2, 45.0, 1.0),
    ]
    s = mv.PathSampler(pts, [a, b])
    for p in pts:
        pose, ang = s(p.t)
        np.testing.assert_allclose(pose.wrist_pos, p.pos, atol=1e-12)
        assert abs(ang - p.forearm_deg) < 1e-9 and K.q_angle(pose.wrist_q, p.q) < 1e-9
    end, _ = s(1.0)
    for n in K.CHAINS:
        for k in range(3):
            assert K.q_angle(end.bones[n][k], b.bones[n][k]) < 1e-9


# --- static letters are untouched -------------------------------------------------------------------------------


def test_static_letters_never_use_the_movement_path(skeleton: K.Skeleton, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise AssertionError("movement code path used for a static letter")

    monkeypatch.setattr(kp_mod, "analyse_movement", boom)
    monkeypatch.setattr(keypose_pipeline, "render_movement", boom)
    monkeypatch.setattr(keypose_render, "render_movement", boom)
    still = synthetic_track("ka", [900.0] * 30, sharp=None, arm_deg=None)
    still.score = still.sharp = still.arm = None  # an old cache without movement fields is fine for static letters
    doc = analyse(still, skeleton)
    assert doc.kind == "static" and doc.path is None
    d = doc.to_dict(skeleton)
    assert "path" not in d and "analysis" not in d  # static JSON layout unchanged
    images, poster = render_doc(doc, skeleton)
    assert len(images) > 1 and poster.size == (360, 360)


def test_movement_frames_start_and_end_on_neutral(skeleton: K.Skeleton) -> None:
    frames, sampler = movement_frames(_moving_doc(skeleton, 0.1), skeleton)
    a = K.forward_kinematics(frames[0].pose, skeleton)
    b = K.forward_kinematics(frames[-1].pose, skeleton)
    assert np.abs(a - b).max() < 1e-9
    assert isinstance(frames[0], Frame) and sampler.duration == pytest.approx(1.0)


# --- drag style (u, o: the wrist travels) ---------------------------------------------------------------------


def test_drag_rule_uses_wrist_travel(skeleton: K.Skeleton) -> None:
    hs = K.pose_to_rotations(world_hand(0.3), skeleton)
    S = skeleton.hand_size()
    for travel, drag in ((0.5, False), (1.9, False), (2.1, True), (4.0, True)):
        pts = [
            mv.PathPoint(0.0, 0, np.zeros(3), hs.wrist_q, 90.0, 0.0),
            mv.PathPoint(1.0, 24, np.array([travel * S, 0.0, 0.0]), hs.wrist_q, 90.0, 0.0),
        ]
        assert mv.is_drag(mv.PathSampler(pts, [hs]), skeleton) is drag


def test_drag_letters_get_the_drag_style_others_do_not(skeleton: K.Skeleton, monkeypatch) -> None:
    seen = []
    real = keypose_pipeline.render_movement

    def spy(frames, sk, size=360, style=None, *args):
        seen.append(style)
        return real(frames, sk, 96, style, *args)

    monkeypatch.setattr(keypose_pipeline, "render_movement", spy)
    S = skeleton.hand_size()
    render_doc(_moving_doc(skeleton, 3.0 * S), skeleton, 96)  # travels ~3 hand sizes: a drag
    render_doc(_moving_doc(skeleton, 0.5 * S), skeleton, 96)  # barely moves
    assert seen == [keypose_render.DRAG_STYLE, None]


def _hand_unit(skeleton: K.Skeleton, curl: float = 0.0, thumb_in: float = 0.0) -> np.ndarray:
    pose = K.pose_to_rotations(world_hand(curl, thumb_in), skeleton)
    pts = K.forward_kinematics(pose, skeleton)
    return to_unit(pts, pts[0, :2] + [0.0, -0.06], 0.3)  # centred on the wrist: the thumb can't shift the view


def _palm_pixels(u: np.ndarray, hand_style: str) -> int:
    img = np.asarray(
        draw_hand(u, size=200, hand_scale=0.13, only={"palm"}, part_colors={"palm": (0, 0, 255)}, hand_style=hand_style)
    )
    return int(((img[..., 2] > 200) & (img[..., 0] < 60)).sum())


def test_natural_palm_keeps_its_shape_when_the_thumb_folds(skeleton: K.Skeleton) -> None:
    open_thumb, folded = _hand_unit(skeleton), _hand_unit(skeleton, thumb_in=1.4)
    assert _palm_pixels(open_thumb, "natural") == _palm_pixels(folded, "natural")  # rigid palm: no pinch
    assert _palm_pixels(open_thumb, "classic") != _palm_pixels(folded, "classic")  # the old palm followed the thumb


def test_natural_style_has_no_outline_across_the_knuckles(skeleton: K.Skeleton) -> None:
    from ishara_ml.teaching.render import OUTLINE

    u = _hand_unit(skeleton, curl=0.4)
    behind = {"index", "middle", "ring", "pinky"}
    ext = {"thumb": True, **dict.fromkeys(behind, True)}

    def knuckle_line(style: str) -> int:
        from ishara_ml.teaching.render import OUTLINE_WIDTH, PALM_ROUNDING, PALM_ROUNDING_NATURAL

        size, hs = 400, 0.13
        img = np.asarray(draw_hand(u, size=size, hand_scale=hs, extended=ext, behind_palm=behind, hand_style=style))
        rounding = PALM_ROUNDING_NATURAL if style == "natural" else PALM_ROUNDING
        offset = (rounding + OUTLINE_WIDTH / 2) * hs  # where the palm's outline ring runs, just outside its edge
        hits = 0
        for k in (5, 9, 13, 17):  # on each finger's base, just past the palm's edge: the line crossing the finger
            along = u[k + 1, :2] - u[k, :2]
            along /= np.linalg.norm(along)
            for extra in (0.0, 0.01, 0.02):
                x, y = (u[k, :2] + along * (offset + extra * hs)) * size
                px = img[int(round(y)), int(round(x))].astype(int)
                hits += int(np.abs(px - np.array(OUTLINE)).sum() < 60)
        return hits

    assert knuckle_line("classic") > 0  # the old drawing has the line ...
    assert knuckle_line("natural") == 0  # ... the natural style does not


def test_short_forearm_fades_out_near_the_wrist(skeleton: K.Skeleton) -> None:
    u = _hand_unit(skeleton)
    away = np.array([0.0, 1.0])
    img = np.asarray(draw_hand(u, size=400, hand_scale=0.13, forearm="short", forearm_dir=away, arm_width=0.12))
    wx, wy = (u[0, :2] * 400).round().astype(int)
    s_px = 0.13 * 400
    near = img[min(wy + int(0.3 * s_px), 399), wx]
    far = img[min(wy + int(1.8 * s_px), 399), wx]
    assert (near < 245).any()  # solid just past the wrist
    assert (far >= 250).all()  # gone by 1.8 hand sizes: no long arm, no stump


# --- per-letter render options (key-pose JSON "render" block) ---------------------------------------------------


def test_render_options_map_to_drawing_flags_and_reject_unknown_ones() -> None:
    kw, front = keypose_render.render_style(
        {"seamlessKnuckles": True, "palmWidth": 1.3, "forearm": "short", "frontParts": ["thumb"], "drag": True}
    )
    assert kw == {"seamless_behind": True, "palm_scale": 1.3, "forearm": "short"} and front == {"thumb"}
    assert keypose_render.render_style(None) == ({}, frozenset())
    with pytest.raises(ValueError, match="unknown render options"):
        keypose_render.render_style({"glow": True})


def test_seamless_knuckles_removes_the_line_on_classic_hands(skeleton: K.Skeleton) -> None:
    from ishara_ml.teaching.render import OUTLINE, OUTLINE_WIDTH, PALM_ROUNDING

    u = _hand_unit(skeleton, curl=0.4)
    behind = {"index", "middle", "ring", "pinky"}
    ext = {"thumb": True, **dict.fromkeys(behind, True)}
    size, hs = 400, 0.13
    offset = (PALM_ROUNDING + OUTLINE_WIDTH / 2) * hs

    def hits(seamless: bool) -> int:
        img = np.asarray(
            draw_hand(u, size=size, hand_scale=hs, extended=ext, behind_palm=behind, seamless_behind=seamless)
        )
        n = 0
        for k in (5, 9, 13, 17):
            along = (u[k + 1, :2] - u[k, :2]) / np.linalg.norm(u[k + 1, :2] - u[k, :2])
            x, y = (u[k, :2] + along * offset) * size
            n += int(np.abs(img[int(round(y)), int(round(x))].astype(int) - np.array(OUTLINE)).sum() < 60)
        return n

    assert hits(False) > 0 and hits(True) == 0


def test_letters_without_render_options_are_drawn_as_before(skeleton: K.Skeleton) -> None:
    kp = K.pose_to_rotations(world_hand(1.0), skeleton)
    doc = KeyPoseDoc("t", "static", {"fps": 24, "file": "x"}, "right", [KeyPose("kp1", kp, 90.0, (0, 5), 5)])
    a, pa = render_doc(doc, skeleton)
    doc.render = {}
    b, pb = render_doc(doc, skeleton)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b, strict=True))
    assert "render" not in KeyPoseDoc(
        "t", "static", {"fps": 24, "file": "x"}, "right", [KeyPose("kp1", kp, 90.0, (0, 5), 5)]
    ).to_dict(skeleton)
