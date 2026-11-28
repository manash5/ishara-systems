"""Key poses from a sign video: landmarks -> smoothed track -> hold / movement key frames -> joint rotations.

A letter becomes 1 key pose (static hand shape) or 2-4 (letters with movement), each a robust average over a few
video frames, stored as editable joint angles on the shared skeleton (``kinematics``). Nothing here draws.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from ishara_ml import config
from ishara_ml.teaching import movement as mv
from ishara_ml.teaching.geometry import hand_size, sign_segment
from ishara_ml.teaching.kinematics import (
    CHAINS,
    HandPose,
    Skeleton,
    palm_frame,
    pose_angles,
    pose_from_angles,
    pose_to_rotations,
    q_from_matrix,
    q_mean,
    q_to_matrix,
    to_palm_local,
)
from ishara_ml.teaching.movement import PathPoint
from ishara_ml.teaching.timeline import Timing

DOC_VERSION = 1
SKELETON_ID = "s1-median-v1"
SKELETON_PATH = config.KEYPOSES_DIR / "skeleton_s1.json"
CAPTURE_FPS = 24.0
SMOOTH_S = 0.3  # Savitzky-Golay window (order 2)
STILL = 1.0  # hand sizes/s: slower than this = holding
MIN_HOLD_S = 0.3  # a hold must last this long to count
MIN_STATIC_HOLD_S = 0.4  # a letter not listed as moving, whose longest hold is shorter, is treated as moving
OUTLIER_MADS = 3.0
# Signs whose movement is part of the sign (brief §3); same set as the old pipeline.
MOVEMENT_SIGNS = frozenset({"aa", "u", "uu", "e", "ai", "o", "au", "am", "ah"})

# Neutral: relaxed, slightly curved open hand. (flex, abduct) degrees per bone. Abduct is measured from the
# metacarpal's line, which already fans out; ``parallel_abduct`` gives the angle that makes a finger parallel to
# the middle finger, and NEUTRAL_SPREAD opens the fingers slightly from there (away from the middle finger).
NEUTRAL_FLEX = {"index": (12, 15, 8), "middle": (14, 18, 10), "ring": (16, 20, 10), "pinky": (18, 20, 10)}
NEUTRAL_SPREAD = {"index": 3.0, "middle": 0.0, "ring": 3.0, "pinky": 6.0}
NEUTRAL_THUMB = ((15.0, -10.0), (10.0, 0.0), (10.0, 0.0))  # flex, abduct; tuned visually (see report)


# --- cached landmarks ----------------------------------------------------------------------------------------


@dataclass
class Track:
    """All frames of one clip: image points (px, NaN where no hand), world points (m), elbow (px), handedness."""

    slug: str
    source: str
    signer: str
    fps: float
    frame_size: tuple[int, int]
    start: float | None
    image: np.ndarray  # (N, 21, 3)
    world: np.ndarray  # (N, 21, 3)
    elbow: np.ndarray  # (N, 2)
    handedness: list[str] = field(default_factory=list)
    # Movement letters only (older caches lack them; ``extract(..., need_movement=True)`` adds them):
    score: np.ndarray | None = None  # (N,) MediaPipe handedness confidence of the detected hand (0 = no hand)
    sharp: np.ndarray | None = None  # (N,) variance of the Laplacian of the hand crop (higher = sharper)
    arm: np.ndarray | None = None  # (N, 4) MediaPipe Pose elbow x, y and wrist x, y (px), NaN where not found

    @property
    def has_movement_fields(self) -> bool:
        return self.score is not None and self.sharp is not None and self.arm is not None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "slug": self.slug,
            "source": self.source,
            "signer": self.signer,
            "fps": self.fps,
            "frameSize": list(self.frame_size),
            "start": self.start,
            "handedness": self.handedness,
        }
        extra = {k: v for k, v in (("score", self.score), ("sharp", self.sharp), ("arm", self.arm)) if v is not None}
        np.savez_compressed(path, image=self.image, world=self.world, elbow=self.elbow, meta=json.dumps(meta), **extra)

    @classmethod
    def load(cls, path: Path) -> Track:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                meta["slug"],
                meta["source"],
                meta["signer"],
                float(meta["fps"]),
                tuple(meta["frameSize"]),
                meta["start"],
                z["image"],
                z["world"],
                z["elbow"],
                list(meta["handedness"]),
                *(z[k] if k in z.files else None for k in ("score", "sharp", "arm")),
            )


def extract(
    row: dict[str, str],
    *,
    fps: float = CAPTURE_FPS,
    cache: Path | None = None,
    refresh: bool = False,
    need_movement: bool = False,
) -> Track:
    """Landmarks for one map row (cached as ``<cache>/<slug>.npz``): VIDEO-mode pass + 512 px crop pass + elbow,
    plus detection confidence, hand-crop sharpness and the Pose arm (elbow + wrist). ``need_movement`` re-extracts
    a cache made before those fields existed."""
    from ishara_ml.nsl23 import resolve_video
    from ishara_ml.teaching import ffmpeg
    from ishara_ml.teaching.landmarker import detect_hand_obs, refine_obs
    from ishara_ml.teaching.movement import hand_sharpness
    from ishara_ml.teaching.pose import detect_arms, detect_elbows

    cache = cache or config.KEYPOSE_LANDMARKS_DIR
    path = cache / f"{row['slug']}.npz"
    if path.exists() and not refresh:
        cached = Track.load(path)
        if cached.has_movement_fields or not need_movement:
            return cached
    src = resolve_video(config.NSL23_DIR, row["signer"], row["file"])
    start = float(row["start"]) if (row.get("start") or "").strip() else None
    end = float(row["end"]) if (row.get("end") or "").strip() else None
    frames = list(ffmpeg.read_frames(src, fps, start, end))
    if not frames:
        raise RuntimeError("no frames decoded")
    h, w = frames[0].shape[:2]
    obs = detect_hand_obs(frames, fps, config.HAND_LANDMARKER_MODEL, mode="video")
    obs, _ = refine_obs(frames, obs, config.HAND_LANDMARKER_MODEL)
    n = len(frames)
    image = np.full((n, 21, 3), np.nan)
    world = np.full((n, 21, 3), np.nan)
    elbow = np.full((n, 2), np.nan)
    score = np.zeros(n)
    sharp = np.full(n, np.nan)
    arm = np.full((n, 4), np.nan)
    present = [i for i, o in enumerate(obs) if o is not None]
    for i in present:
        image[i], world[i] = obs[i].image, obs[i].world
        score[i] = obs[i].score
        sharp[i] = hand_sharpness(frames[i], obs[i].image)
    if present and config.POSE_LANDMARKER_MODEL.exists():
        wrists = [obs[i].image[0] for i in present]
        found = detect_elbows([frames[i] for i in present], wrists, config.POSE_LANDMARKER_MODEL)
        for i, e in zip(present, found, strict=True):
            if e is not None:
                elbow[i] = e
        arms = detect_arms([frames[i] for i in present], wrists, config.POSE_LANDMARKER_MODEL)
        for i, a in zip(present, arms, strict=True):
            if a is not None:
                arm[i] = a
    track = Track(
        row["slug"],
        src.relative_to(config.NSL23_DIR).as_posix(),
        row["signer"],
        fps,
        (w, h),
        start,
        image,
        world,
        elbow,
        [o.handedness if o else "" for o in obs],
        score,
        sharp,
        arm,
    )
    track.save(path)
    return track


# --- smoothing and motion -----------------------------------------------------------------------------------


def _fill(a: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Hold the last known value over gaps (and back-fill the start)."""
    out = a.copy()
    idx = np.where(present)[0]
    last = idx[0]
    for i in range(len(out)):
        if present[i]:
            last = i
        else:
            out[i] = out[last]
    return out


def smooth_track(a: np.ndarray, fps: float, seconds: float = SMOOTH_S) -> np.ndarray:
    """Savitzky-Golay (order 2) along time; the window shrinks for short clips."""
    n = len(a)
    win = max(3, round(seconds * fps) | 1)
    win = min(win, n if n % 2 else n - 1)
    if win < 3:
        return a.copy()
    return savgol_filter(a, win, 2, axis=0, mode="interp")


@dataclass
class Prepared:
    """A trimmed, gap-filled, smoothed clip. ``points``: camera-frame metres (world shape + wrist translation)."""

    first: int  # index of the first kept frame in the Track
    image: np.ndarray  # (n, 21, 3) px
    points: np.ndarray  # (n, 21, 3) m
    wrist_m: np.ndarray  # (n, 3) wrist translation in metres
    px_per_m: float
    size_m: float  # hand size (geometry.hand_size) in metres
    speed: np.ndarray  # (n,) hand sizes / s
    forearm_deg: np.ndarray  # (n,) image angle wrist -> elbow (degrees, y down)
    detected: np.ndarray  # (n,) bool


def px_per_metre(image: np.ndarray, world: np.ndarray) -> float:
    """Image scale of the world landmarks (weak perspective): median ratio of 2-D palm bone lengths."""
    ratios = []
    for a, b in ((0, 5), (0, 17), (5, 17), (0, 9)):
        li = np.linalg.norm(image[:, a, :2] - image[:, b, :2], axis=1)
        lw2 = np.linalg.norm(world[:, a, :2] - world[:, b, :2], axis=1)
        lw3 = np.linalg.norm(world[:, a] - world[:, b], axis=1)
        ok = lw2 > 0.5 * lw3  # skip bones foreshortened in the image
        ratios.extend((li[ok] / lw2[ok]).tolist())
    if not ratios:
        raise RuntimeError("cannot estimate image scale")
    return float(np.median(ratios))


def motion_speed(points: np.ndarray, size_m: float, fps: float) -> np.ndarray:
    """Centred per-frame speed: mean 3-D displacement of all 21 points, in hand sizes per second."""
    n = len(points)
    if n < 2:
        return np.zeros(n)
    step = np.linalg.norm(np.diff(points, axis=0), axis=2).mean(axis=1)
    out = np.empty(n)
    out[0], out[-1] = step[0], step[-1]
    out[1:-1] = (step[:-1] + step[1:]) / 2
    return out / max(size_m, 1e-9) * fps


def prepare(track: Track) -> Prepared:
    present = ~np.isnan(track.image[:, 0, 0])
    idx = np.where(present)[0]
    if len(idx) == 0:
        raise RuntimeError("no hand detected")
    a, b = int(idx[0]), int(idx[-1]) + 1
    det = present[a:b]
    image = smooth_track(_fill(track.image[a:b], det), track.fps)
    world = smooth_track(_fill(track.world[a:b], det), track.fps)
    scale = px_per_metre(image, world)
    wrist_m = np.zeros((b - a, 3))
    wrist_m[:, :2] = (image[:, 0, :2] - image[0, 0, :2]) / scale
    points = world - world[:, :1] + wrist_m[:, None]
    size_m = hand_size(world)
    speed = motion_speed(points, size_m, track.fps)
    elbow = track.elbow[a:b]
    known = ~np.isnan(elbow[:, 0])
    if known.any():
        elbow = smooth_track(_fill(elbow, known), track.fps)
        vec = elbow - image[:, 0, :2]
    else:  # no person found: fall back to the hand's own axis (middle knuckle -> wrist)
        vec = image[:, 0, :2] - image[:, 9, :2]
    ang = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
    return Prepared(a, image, points, wrist_m, scale, size_m, speed, np.degrees(ang), det)


# --- hold / key frames ----------------------------------------------------------------------------------------


def longest_hold(speed: np.ndarray, still: float, min_len: int) -> tuple[int, int] | None:
    """[start, end) of the longest run with speed < ``still``, or None if no run lasts ``min_len`` frames."""
    best, start = (0, 0), None
    for i, slow in enumerate([*(speed < still), False]):
        if slow and start is None:
            start = i
        elif not slow and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best if best[1] - best[0] >= max(min_len, 1) else None


def stillest_window(speed: np.ndarray, w: int) -> tuple[int, int]:
    w = max(1, min(w, len(speed)))
    costs = np.convolve(speed, np.ones(w) / w, mode="valid")
    a = int(np.argmin(costs))
    return a, a + w


def robust_mean_pose(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust average of several 21-point hands (camera frame).

    Per-joint median of the palm-local shapes; frames farther than ``OUTLIER_MADS`` x MAD from it are dropped and
    the median is taken again. Orientation = quaternion mean of the kept frames, position = their mean wrist.
    Returns the averaged points and the kept-frame mask.
    """
    frames = np.asarray(frames, dtype=np.float64)
    local = np.stack([to_palm_local(p) for p in frames])
    med = np.median(local, axis=0)
    d = np.linalg.norm(local - med, axis=2).mean(axis=1)
    mad = float(np.median(np.abs(d - np.median(d))))
    keep = d <= np.median(d) + OUTLIER_MADS * max(mad, 0.05 * float(np.median(d)), 1e-12)
    med = np.median(local[keep], axis=0)
    rot = q_to_matrix(q_mean(np.stack([q_from_matrix(palm_frame(p)) for p in frames[keep]])))
    wrist = frames[keep, 0].mean(axis=0)
    return wrist + med @ rot.T, keep


# --- key-pose document ----------------------------------------------------------------------------------------


@dataclass
class KeyPose:
    name: str
    pose: HandPose
    forearm_deg: float
    source_frames: tuple[int, int]  # [a, b) in the clip's frame numbering (at CAPTURE_FPS)
    frames_used: int


@dataclass
class KeyPoseDoc:
    slug: str
    kind: str
    source: dict
    handedness: str
    key_poses: list[KeyPose]
    timing: Timing = field(default_factory=Timing)
    notes: list[str] = field(default_factory=list)
    locked: bool = False  # hand-edited: the pipeline renders it as is and never re-analyses the video
    # Movement letters only: key_poses are the HANDSHAPES (joints; their wrist is informational) and ``path`` is
    # the movement (wrist position + orientation + forearm over time, as control points).
    path: list[PathPoint] | None = None
    analysis: dict | None = None
    # Optional per-letter drawing options (see keypose_render.RENDER_OPTIONS), e.g. {"seamlessKnuckles": true}.
    render: dict | None = None

    def to_dict(self, skeleton: Skeleton) -> dict:
        d = self._base_dict(skeleton)
        if self.path is not None:
            d["path"] = [
                {
                    "tS": round(pt.t, 4),
                    "frame": round(pt.frame, 2),
                    "positionM": [round(float(v), 4) for v in pt.pos],
                    "rotationDeg": _euler(pt.q),
                    "forearmDeg": round(pt.forearm_deg, 1),
                    "shape": round(pt.shape, 3),
                }
                for pt in self.path
            ]
        if self.analysis is not None:
            d["analysis"] = self.analysis
        if self.render is not None:
            d["render"] = self.render
        return d

    def _base_dict(self, skeleton: Skeleton) -> dict:
        return {
            "version": DOC_VERSION,
            "slug": self.slug,
            "skeletonId": skeleton.id,
            "kind": self.kind,
            "handedness": self.handedness,
            "source": self.source,
            "timing": self.timing.to_dict(),
            "units": (
                "wrist.rotationDeg: palm frame in camera axes (x right, y down, z away), extrinsic xyz degrees; "
                "wrist.positionM: metres from key pose 1; joints: per bone [flex, abduct] degrees relative to the "
                "parent bone (flex = towards the palm); forearmDeg: image angle wrist->elbow (0 = right, 90 = down)"
            ),
            "keyPoses": [
                {
                    "name": k.name,
                    "sourceFrames": list(k.source_frames),
                    "sourceTimesS": [round(frame_time(self, f), 3) for f in k.source_frames],
                    "framesUsed": k.frames_used,
                    "forearmDeg": round(k.forearm_deg, 1),
                    "wrist": {
                        "rotationDeg": [
                            round(float(v), 2)
                            for v in Rotation.from_matrix(q_to_matrix(k.pose.wrist_q)).as_euler("xyz", degrees=True)
                        ],
                        "positionM": [round(float(v), 4) for v in k.pose.wrist_pos],
                    },
                    "joints": pose_angles(k.pose, skeleton),
                }
                for k in self.key_poses
            ],
            "notes": self.notes,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, d: dict, skeleton: Skeleton) -> KeyPoseDoc:
        if d.get("skeletonId") != skeleton.id:
            raise ValueError(f"{d.get('slug')}: key poses are for skeleton {d.get('skeletonId')}, not {skeleton.id}")
        kps = []
        for k in d["keyPoses"]:
            m = Rotation.from_euler("xyz", k["wrist"]["rotationDeg"], degrees=True).as_matrix()
            pose = pose_from_angles(k["joints"], q_from_matrix(m), np.array(k["wrist"]["positionM"]), skeleton)
            kps.append(KeyPose(k["name"], pose, float(k["forearmDeg"]), tuple(k["sourceFrames"]), int(k["framesUsed"])))
        path = None
        if d.get("path") is not None:
            path = [
                PathPoint(
                    float(pt["tS"]),
                    float(pt["frame"]),
                    np.array(pt["positionM"], dtype=np.float64),
                    q_from_matrix(Rotation.from_euler("xyz", pt["rotationDeg"], degrees=True).as_matrix()),
                    float(pt["forearmDeg"]),
                    float(pt["shape"]),
                )
                for pt in d["path"]
            ]
        return cls(
            d["slug"],
            d["kind"],
            d["source"],
            d["handedness"],
            kps,
            Timing.from_dict(d.get("timing")),
            list(d.get("notes", [])),
            bool(d.get("locked", False)),
            path,
            d.get("analysis"),
            d.get("render"),
        )

    def save(self, path: Path, skeleton: Skeleton) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(skeleton), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, skeleton: Skeleton) -> KeyPoseDoc:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), skeleton)


def _snap(pose: HandPose, skeleton: Skeleton) -> HandPose:
    """Round-trip through the stored angles so a freshly analysed pose renders exactly like its JSON."""
    angles = pose_angles(pose, skeleton)
    return pose_from_angles(angles, pose.wrist_q, pose.wrist_pos, skeleton)


def classify(track: Track) -> tuple[bool, list[str]]:
    """(moving, notes): listed movement vowels, or any letter whose clip has no hold of MIN_STATIC_HOLD_S."""
    if track.slug in MOVEMENT_SIGNS:
        return True, []
    prep = prepare(track)
    hold = longest_hold(prep.speed, STILL, round(MIN_HOLD_S * track.fps))
    if hold is None or hold[1] - hold[0] < round(MIN_STATIC_HOLD_S * track.fps):
        return True, ["no long hold found: treated as a letter with movement"]
    return False, []


def _source(track: Track, prep: Prepared) -> dict:
    labels = [h for h in track.handedness if h]
    return {
        "file": track.source,
        "signer": track.signer,
        "fps": track.fps,
        "start": track.start,
        "frameSize": list(track.frame_size),
        "framesWithHand": int(prep.detected.sum()),
        "pxPerMetre": round(prep.px_per_m, 1),
        "mediapipeHandedness": max(set(labels), key=labels.count) if labels else None,
    }


def analyse(track: Track, skeleton: Skeleton) -> KeyPoseDoc:
    """Static letters: one robust key pose over the hold. Letters with movement: ``analyse_movement``."""
    moving, notes = classify(track)
    if moving:
        return analyse_movement(track, skeleton, notes)
    prep = prepare(track)
    fps = track.fps
    hold = longest_hold(prep.speed, STILL, round(MIN_HOLD_S * fps))
    windows: list[tuple[int, int]] = [hold]  # type: ignore[list-item]

    origin = None
    key_poses = []
    for i, (a, b) in enumerate(windows):
        pts, keep = robust_mean_pose(prep.points[a:b])
        pose = pose_to_rotations(pts, skeleton)
        if origin is None:
            origin = pose.wrist_pos.copy()
        pose.wrist_pos = pose.wrist_pos - origin
        forearm = float(np.median(prep.forearm_deg[a:b]))
        key_poses.append(
            KeyPose(f"kp{i + 1}", _snap(pose, skeleton), forearm, (prep.first + a, prep.first + b), int(keep.sum()))
        )
    return KeyPoseDoc(track.slug, "static", _source(track, prep), "right", key_poses, notes=notes)


def analyse_movement(track: Track, skeleton: Skeleton, notes: list[str] | None = None) -> KeyPoseDoc:
    """Handshape(s) from sharp frames only + the movement as a smoothed wrist path (see teaching/movement.py)."""
    if not track.has_movement_fields:
        raise RuntimeError(f"{track.slug}: landmark cache lacks sharpness/confidence/arm: extract(need_movement=True)")
    notes = list(notes or [])
    prep = prepare(track)
    fps = track.fps
    n = len(prep.speed)
    sl = slice(prep.first, prep.first + n)
    present = prep.detected
    score, sharp, world, arm = track.score[sl], track.sharp[sl], track.world[sl], track.arm[sl]
    img_raw = track.image[sl]
    sa, sb = sign_segment(list(prep.image), fps, moving=True)

    # 1. HANDSHAPE: sharp, confident, slow frames of the sign segment (whole clip if too few there)
    in_seg = np.zeros(n, dtype=bool)
    in_seg[sa:sb] = True
    mask, how = mv.select_sharp(np.where(in_seg, prep.speed, np.inf), score, sharp, present)
    if mask.sum() < mv.MIN_SHARP_FRAMES:
        mask, how = mv.select_sharp(prep.speed, score, sharp, present)
        how += " (whole clip: too few in the sign segment)"
    idx = np.where(mask)[0]
    local = np.stack([to_palm_local(world[i]) for i in idx])
    split = mv.split_handshapes(local, prep.size_m)
    groups = [idx] if split is None else [idx[:split], idx[split:]]
    if split is not None:
        notes.append(
            f"two handshapes: sharp frames {[int(prep.first + i) for i in groups[0]]} vs "
            f"{[int(prep.first + i) for i in groups[1]]}"
        )
    shapes, key_poses = [], []
    for g, grp in enumerate(groups):
        pts, keep = robust_mean_pose(world[grp])
        pose = _snap(pose_to_rotations(pts, skeleton), skeleton)
        pose.wrist_pos = np.zeros(3)
        shapes.append(pose)
        frames = (int(prep.first + grp[0]), int(prep.first + grp[-1] + 1))
        key_poses.append(KeyPose(f"hs{g + 1}", pose, 0.0, frames, int(keep.sum())))

    # 2. MOVEMENT: wrist position (strongly smoothed), orientation (reliable frames), forearm (Pose arm)
    known = present & ~np.isnan(img_raw[:, 0, 0])
    wrist_px = smooth_track(_fill(img_raw[:, 0, :2], known), fps, mv.PATH_SMOOTH_S)
    px_m = px_per_metre(img_raw[mask], world[mask]) if mask.sum() >= 3 else prep.px_per_m
    pos = np.zeros((n, 3))
    pos[:, :2] = (wrist_px - wrist_px[sa]) / px_m
    reliable = present & (score >= mv.RELIABLE_MIN_SCORE) & (prep.speed < mv.RELIABLE_MAX_SPEED)
    if reliable.sum() < 2:
        reliable = present.copy()
        notes.append("orientation: too few slow frames, used every detected frame")
    rel_idx = np.where(reliable)[0]
    qs = np.stack([mv.palm_quat(world[i]) for i in rel_idx])
    p90 = max(float(np.nanpercentile(sharp, 90)), 1e-9)
    weights = score[rel_idx] * np.clip(sharp[rel_idx] / p90, 0.1, 1.0)
    angles = mv.arm_angles(arm, fps, smooth_track)
    forearm_source = "pose elbow -> pose wrist"
    if angles is None:
        angles = prep.forearm_deg
        forearm_source = "hand axis (no Pose arm found)"
        notes.append("forearm: MediaPipe Pose found no arm, fell back to the hand axis")
    frame_axis = np.arange(n)
    path = []
    prev_q = None
    for c in mv.control_frames(sa, sb, fps):
        q = mv.weighted_orientation(qs, rel_idx.astype(float), weights, float(c), fps)
        if prev_q is not None and float(q @ prev_q) < 0:
            q = -q
        prev_q = q
        blend = 0.0
        if split is not None:
            ta, tb = float(groups[0][-1]), float(groups[1][0])
            blend = float(np.clip((c - ta) / max(tb - ta, 1e-9), 0.0, 1.0))
        point = np.array([np.interp(c, frame_axis, pos[:, k]) for k in range(3)])
        forearm = float(np.interp(c, frame_axis, angles))
        path.append(mv.PathPoint((c - sa) / fps, float(prep.first + c), point, q, forearm, blend))
    seg_len = float(np.linalg.norm(np.diff(pos[sa:sb], axis=0), axis=1).sum())
    analysis = {
        "segmentFrames": [int(prep.first + sa), int(prep.first + sb)],
        "sharpSelection": how,
        "sharpFrames": [int(prep.first + i) for i in idx],
        "sharpness": {
            "median": round(float(np.nanmedian(sharp)), 1),
            "p90": round(p90, 1),
            "perFrame": [None if not np.isfinite(v) else round(float(v), 1) for v in sharp],
        },
        "handshapes": len(groups),
        "pathLengthHandSizes": round(seg_len / prep.size_m, 2),
        "forearmSource": forearm_source,
    }
    doc = KeyPoseDoc(track.slug, "movement", _source(track, prep), "right", key_poses, notes=notes, path=path)
    doc.analysis = analysis
    return doc


def _spread_sign(skeleton: Skeleton, name: str) -> float:
    """+1 if abduct > 0 moves this finger away from the middle finger, else -1."""
    from ishara_ml.teaching.kinematics import _basis

    c = CHAINS[name]
    _, e_a = _basis(skeleton.rest[name])
    side = skeleton.palm[c[0]] - skeleton.palm[9]
    return 1.0 if float(side @ e_a) >= 0 else -1.0


def parallel_abduct(skeleton: Skeleton, name: str) -> float:
    """MCP abduct (degrees) that makes this finger parallel to the middle finger (fingers together)."""
    from ishara_ml.teaching.kinematics import _basis

    _, e_a = _basis(skeleton.rest[name])
    target = skeleton.rest["middle"]
    return math.degrees(math.asin(float(np.clip(target @ e_a, -1.0, 1.0))))


def neutral_pose(skeleton: Skeleton, first: HandPose) -> HandPose:
    """Relaxed, slightly curved open hand with ``first``'s wrist orientation and position."""
    joints: dict[str, list[list[float]]] = {"thumb": [list(v) for v in NEUTRAL_THUMB]}
    for name, flex in NEUTRAL_FLEX.items():
        spread = parallel_abduct(skeleton, name) + NEUTRAL_SPREAD[name] * _spread_sign(skeleton, name)
        joints[name] = [[flex[0], spread], [flex[1], 0.0], [flex[2], 0.0]]
    return pose_from_angles(joints, first.wrist_q, first.wrist_pos, skeleton)


def skeleton_frames(track: Track) -> np.ndarray:
    """Held (low-motion) world frames of a clip, for the shared skeleton."""
    prep = prepare(track)
    keep = prep.speed < STILL
    if keep.sum() < 3:
        keep = prep.speed <= np.quantile(prep.speed, 0.3)
    return prep.points[keep]


def build_skeleton_from_tracks(tracks: list[Track], skeleton_id: str = SKELETON_ID) -> Skeleton:
    from ishara_ml.teaching.kinematics import build_skeleton

    frames = [f for t in tracks for f in skeleton_frames(t)]
    return build_skeleton(frames, skeleton_id, [t.slug for t in tracks])


def _euler(q: np.ndarray) -> list[float]:
    return [round(float(v), 2) for v in Rotation.from_matrix(q_to_matrix(q)).as_euler("xyz", degrees=True)]


def frame_time(doc: KeyPoseDoc, frame: float) -> float:
    return (doc.source.get("start") or 0) + frame / doc.source["fps"]


def angle_lerp(a: float, b: float, t: float) -> float:
    d = (b - a + 180) % 360 - 180
    return a + t * d
