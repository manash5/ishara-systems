"""Render a cut-out hand: ``toon`` (real hand repainted as a flat illustration) or ``photo`` (real pixels).

toon: inside the mask, brightness is quantised into 3 flat tones of the app's skin palette, with thresholds fixed
per clip (so tones don't flicker); a clean outline follows the mask edge; thin soft internal lines appear only at
strong edges (finger gaps, knuckles), kept only where they persist over neighbouring frames, so bends stay
readable without noise.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

SHADOW = np.array([0xE8, 0xAA, 0x8C], dtype=np.float32)
BASE = np.array([0xF6, 0xC4, 0xA8], dtype=np.float32)
HIGHLIGHT = np.array([0xFC, 0xDD, 0xC9], dtype=np.float32)
OUTLINE = np.array([0xD6, 0x96, 0x78], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)

TONE_PERCENTILES = (8, 88)  # per clip: darkest 8 % of the hand -> shadow, brightest 12 % -> highlight
OUTLINE_PX = 2.0  # at the 360 px output
LINE_ALPHA = 0.9  # internal lines: a little softer than the outline
LINE_MIN_LENGTH = 0.12  # x hand size: shorter crease pieces are skin texture, not finger gaps / knuckles
CREASE_WIDTH = 0.12  # x hand size: how wide a dark valley (finger gap) can be
MIN_TONE_AREA = 0.02  # x hand size squared: smaller shadow/highlight islands are merged into the base tone


def _smooth_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance with skin texture removed but large shapes kept."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    for _ in range(2):
        lum = cv2.bilateralFilter(lum, 15, 25, 15)
    return lum


def tone_thresholds(crops: list[np.ndarray], alphas: list[np.ndarray]) -> tuple[float, float]:
    """Brightness thresholds for the 3 tones, from all frames of the clip at once (stable over time)."""
    vals = np.concatenate([_smooth_gray(c)[a > 0.5] for c, a in zip(crops, alphas, strict=True)])
    if vals.size == 0:
        return 100.0, 200.0
    lo, hi = np.percentile(vals, TONE_PERCENTILES)
    return float(lo), float(hi)


def _drop_small(region: np.ndarray, min_area: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def tone_map(rgb: np.ndarray, alpha: np.ndarray, thresholds: tuple[float, float], size_px: float) -> np.ndarray:
    """0 = shadow, 1 = base, 2 = highlight: large flat areas only."""
    g = cv2.GaussianBlur(_smooth_gray(rgb), (0, 0), max(1.0, 0.03 * size_px))
    inside = alpha > 0.5
    k = max(3, int(round(0.06 * size_px)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    min_area = MIN_TONE_AREA * size_px**2
    shadow = cv2.morphologyEx(((g < thresholds[0]) & inside).astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    light = cv2.morphologyEx(((g > thresholds[1]) & inside).astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    tone = np.ones(g.shape, dtype=np.uint8)
    tone[_drop_small(shadow, min_area)] = 0
    tone[_drop_small(light, min_area)] = 2
    return tone


def strong_edges(rgb: np.ndarray, alpha: np.ndarray, size_px: float) -> np.ndarray:
    """Internal lines: narrow dark valleys in brightness (finger gaps, knuckle creases), thinned to single-pixel
    lines, inside the hand (not along its outline) and without short specks."""
    g = _smooth_gray(rgb).astype(np.float32)
    k = max(3, int(round(CREASE_WIDTH * size_px)) | 1)
    valley = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inner = cv2.erode((alpha > 0.5).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if not inner.any():
        return np.zeros(g.shape, dtype=bool)
    thr = max(5.0, float(np.percentile(valley[inner], 80)))
    lines = ((valley > thr) & inner).astype(np.uint8) * 255
    lines = cv2.ximgproc.thinning(lines) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= LINE_MIN_LENGTH * size_px
    return keep[labels]


def stable_edges(edge_maps: list[np.ndarray]) -> list[np.ndarray]:
    """Keep a line pixel only if it is (near) a line in at least 2 of 3 neighbouring frames: no flicker."""
    grown = [cv2.dilate(e.astype(np.uint8), np.ones((5, 5), np.uint8)) for e in edge_maps]
    out = []
    for i, e in enumerate(edge_maps):
        votes = sum(grown[j] for j in range(max(0, i - 1), min(len(edge_maps), i + 2)))
        need = 2 if len(edge_maps) > 1 else 1
        out.append(e & (votes >= need))
    return out


def toon_frame(
    rgb: np.ndarray,
    alpha: np.ndarray,
    thresholds: tuple[float, float],
    edges: np.ndarray,
    size_px: float,
    out_size: int = 360,
) -> Image.Image:
    scale = rgb.shape[0] / out_size
    palette = np.stack([SHADOW, BASE, HIGHLIGHT])
    colour = palette[tone_map(rgb, alpha, thresholds, size_px)]

    # Internal lines: thin (~1.5 px at the output) and soft-edged.
    line_px = max(1, int(round(1.5 * scale)))
    lines = cv2.dilate(edges.astype(np.uint8), np.ones((line_px, line_px), np.uint8)).astype(np.float32)
    lines = np.clip(cv2.GaussianBlur(lines, (0, 0), 0.6 * scale), 0, 1) * LINE_ALPHA
    colour = colour * (1 - lines[..., None]) + OUTLINE * lines[..., None]

    # Outline: a band just outside the mask edge, anti-aliased.
    r = OUTLINE_PX * scale
    hard = (alpha > 0.5).astype(np.uint8)
    dist_out = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
    outline_a = np.clip(r + 0.5 - dist_out, 0.0, 1.0) * (1 - alpha)

    img = WHITE * (1 - outline_a[..., None]) + OUTLINE * outline_a[..., None]
    img = img * (1 - alpha[..., None]) + colour * alpha[..., None]
    return _downscale(img, out_size)


def photo_frame(rgb: np.ndarray, alpha: np.ndarray, out_size: int = 360) -> Image.Image:
    img = WHITE * (1 - alpha[..., None]) + rgb.astype(np.float32) * alpha[..., None]
    return _downscale(img, out_size)


def _downscale(img: np.ndarray, out_size: int) -> Image.Image:
    pil = Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8), mode="RGB")
    return pil.resize((out_size, out_size), Image.Resampling.LANCZOS)
