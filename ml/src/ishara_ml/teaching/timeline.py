"""Animation timeline for the key-pose style: every timing constant lives here.

neutral (short pause) -> key pose 1 -> ... -> key pose N (eased transitions) -> hold the final pose -> ease back to
neutral. The loop starts and ends on the same neutral pose, so there is no jump at the seam.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

FPS = 24  # animation frame rate
TRANSITION_S = 0.4  # neutral -> first key pose, and between key poses
HOLD_S = 1.0  # the final key pose is held this long
RETURN_S = 0.3  # final key pose -> neutral
PAUSE_S = 0.25  # neutral is held this long before the next loop


@dataclass(frozen=True)
class Timing:
    transitionS: float = TRANSITION_S  # noqa: N815 - JSON field names
    holdS: float = HOLD_S  # noqa: N815
    returnS: float = RETURN_S  # noqa: N815
    pauseS: float = PAUSE_S  # noqa: N815

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> Timing:
        d = d or {}
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


def ease_in_out(t: float) -> float:
    """Cosine ease: zero velocity at both ends."""
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def frames_for(seconds: float, fps: int = FPS) -> int:
    return max(1, round(seconds * fps))


def build_timeline[T](
    neutral: T, keys: list[T], lerp: Callable[[T, T, float], T], timing: Timing = Timing(), fps: int = FPS
) -> list[T]:
    """Frames of one loop. ``lerp(a, b, t)`` interpolates two states (joint slerp for hand poses)."""
    if not keys:
        raise ValueError("need at least one key pose")
    out: list[T] = [neutral] * frames_for(timing.pauseS, fps)
    prev = neutral
    for key in keys:
        n = frames_for(timing.transitionS, fps)
        out += [lerp(prev, key, ease_in_out(i / n)) for i in range(1, n + 1)]
        prev = key
    out += [prev] * frames_for(timing.holdS, fps)
    n = frames_for(timing.returnS, fps)
    out += [lerp(prev, neutral, ease_in_out(i / n)) for i in range(1, n + 1)]
    return out


MOVEMENT_TIME_SCALE = 1.25  # letters with movement replay their real movement this much slower (easier to follow)


def build_movement_timeline[T](
    neutral: T,
    sample: Callable[[float], T],
    duration: float,
    lerp: Callable[[T, T, float], T],
    timing: Timing = Timing(),
    fps: int = FPS,
) -> list[T]:
    """Frames of one loop for a letter with movement: pause on neutral -> ease into the movement's first state ->
    replay the movement (``sample(t)``, t in [0, duration] seconds, at MOVEMENT_TIME_SCALE x real time) -> hold the
    final state -> ease back to neutral. Starts and ends on ``neutral``."""
    out: list[T] = [neutral] * frames_for(timing.pauseS, fps)
    start = sample(0.0)
    n = frames_for(timing.transitionS, fps)
    out += [lerp(neutral, start, ease_in_out(i / n)) for i in range(1, n + 1)]
    n = frames_for(duration * MOVEMENT_TIME_SCALE, fps)
    out += [sample(duration * i / n) for i in range(1, n + 1)]
    end = out[-1]
    out += [end] * frames_for(timing.holdS, fps)
    n = frames_for(timing.returnS, fps)
    out += [lerp(end, neutral, ease_in_out(i / n)) for i in range(1, n + 1)]
    return out
