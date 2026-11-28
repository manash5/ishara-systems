"""Elbow positions from MediaPipe Pose Landmarker, so the drawn forearm points where the real forearm points.

The hand landmarks have no elbow; guessing the forearm from the hand's own axis fails whenever the wrist bends
(o, ai: the forearm swung round and looked oversized). Only the elbow position is used - never stored beyond the
per-letter landmark JSON, like the hand landmarks.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Pose landmark indices: 13/14 = left/right elbow, 15/16 = left/right wrist (the person's left/right).
ARMS = ((13, 15), (14, 16))


def detect_elbows(frames: list[np.ndarray], hand_wrists: list[np.ndarray], model: Path) -> list[np.ndarray | None]:
    """Per frame, the (x, y) pixel position of the elbow on the arm whose wrist is nearest the hand's wrist.

    None where no person is found or the chosen arm's wrist is far from the hand (> 25 % of the frame width).
    """
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    out: list[np.ndarray | None] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame, hand_wrist in zip(frames, hand_wrists, strict=True):
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect(image)
            if not result.pose_landmarks:
                out.append(None)
                continue
            lms = result.pose_landmarks[0]
            best, best_d = None, float("inf")
            for elbow_i, wrist_i in ARMS:
                wrist = np.array([lms[wrist_i].x * w, lms[wrist_i].y * h])
                d = float(np.linalg.norm(wrist - hand_wrist[:2]))
                if d < best_d:
                    best, best_d = np.array([lms[elbow_i].x * w, lms[elbow_i].y * h]), d
            out.append(best if best_d < 0.25 * w else None)
    return out


def fill_and_smooth(points: list[np.ndarray | None], window: int) -> list[np.ndarray] | None:
    """Hold the last known elbow over gaps (and back-fill the start), then moving-average. None if never found."""
    known = [p for p in points if p is not None]
    if not known:
        return None
    filled: list[np.ndarray] = []
    last = known[0]
    for p in points:
        last = p if p is not None else last
        filled.append(last)
    stack = np.stack(filled)
    half = window // 2
    return [stack[max(0, i - half) : i + half + 1].mean(axis=0) for i in range(len(stack))]


def detect_arms(frames: list[np.ndarray], hand_wrists: list[np.ndarray], model: Path) -> list[np.ndarray | None]:
    """Per frame, the real arm from MediaPipe Pose: (elbow_x, elbow_y, wrist_x, wrist_y) in pixels, on the arm whose
    *pose* wrist is nearest the hand's wrist. The forearm direction is then elbow -> pose wrist, independent of the
    hand landmarks (which smear when the hand is motion-blurred). None where no person / no matching arm is found.
    """
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    out: list[np.ndarray | None] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame, hand_wrist in zip(frames, hand_wrists, strict=True):
            h, w = frame.shape[:2]
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
            if not result.pose_landmarks:
                out.append(None)
                continue
            lms = result.pose_landmarks[0]
            best, best_d = None, float("inf")
            for elbow_i, wrist_i in ARMS:
                wrist = np.array([lms[wrist_i].x * w, lms[wrist_i].y * h])
                d = float(np.linalg.norm(wrist - hand_wrist[:2]))
                if d < best_d:
                    best = np.array([lms[elbow_i].x * w, lms[elbow_i].y * h, wrist[0], wrist[1]])
                    best_d = d
            out.append(best if best_d < 0.25 * w else None)
    return out
