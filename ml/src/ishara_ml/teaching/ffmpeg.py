"""Thin ffmpeg wrappers: preview MP4 encoding and frame extraction to numpy."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import numpy as np


@cache
def ffmpeg_exe() -> str:
    """ffmpeg from PATH if installed, else the binary bundled with ``imageio-ffmpeg`` inside the venv."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _trim_args(start: float | None, end: float | None) -> list[str]:
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    if end is not None:
        args += ["-to", f"{end:.3f}"]
    return args


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", *args], capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg failed: " + " | ".join(tail))
    return proc


def encode_preview_mp4(src: Path, dst: Path, start: float | None = None, end: float | None = None) -> None:
    """Muted H.264, 480 px high, 24 fps, CRF 28, yuv420p, +faststart."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "-y",
            "-i",
            str(src),
            *_trim_args(start, end),
            "-an",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-vf",
            "scale=-2:480,fps=24",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def video_size(src: Path) -> tuple[int, int]:
    """Display (width, height) of the first video stream, after ffmpeg's automatic rotation."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(src)], capture_output=True)
    info = proc.stderr.decode(errors="replace")
    match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    if not match:
        raise RuntimeError(f"could not read video size of {src}")
    w, h = int(match.group(1)), int(match.group(2))
    rot = re.search(r"rotation of (-?\d+(?:\.\d+)?)", info)
    if rot and round(abs(float(rot.group(1)))) % 180 == 90:
        w, h = h, w
    return w, h


def read_frames(src: Path, fps: float, start: float | None = None, end: float | None = None) -> Iterator[np.ndarray]:
    """Frames (H, W, 3 uint8 RGB) sampled at ``fps`` at full resolution."""
    w, h = video_size(src)
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(src),
        *_trim_args(start, end),
        "-an",
        "-vf",
        f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    frame_bytes = w * h * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg frame extraction failed: {stderr.strip()[-300:]}")
