"""Generate teaching media for each letter in ml/labels/nsl23_map.csv.

Run from ``ml/``:
    uv run python -m ishara_ml.make_teaching_media --build-skeleton          # once: shared S1 hand skeleton
    uv run python -m ishara_ml.make_teaching_media --only ka,kha [--from-json] [--style keypose|photo|illustrated]

Style per letter comes from the map's optional ``style`` column (keypose | photo; empty or the dropped "toon" =
keypose); ``--style`` forces one for every letter. keypose: key poses + joint-rotation interpolation, see
teaching/keypose.py; its editable key poses live in ml/data/processed/keyposes/<slug>.json and ``--from-json``
re-renders from them without reading the video.

Writes backend/content/media/<slug>/{video.mp4, animation.webp, poster.webp}. Continues past failures and lists
them at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ishara_ml import config
from ishara_ml.teaching.keypose_pipeline import build_skeleton, process_keypose
from ishara_ml.teaching.pipeline import MOVEMENT_SIGNS, load_map, process_row
from ishara_ml.teaching.realhand import process_cutout

CUTOUT_STYLES = ("photo",)  # real hand cut out onto white: per-letter fallback ("toon" was dropped)
STYLES = ("keypose", *CUTOUT_STYLES, "illustrated")  # illustrated = the old per-frame landmark drawing
DEFAULT_STYLE = "keypose"
LEGACY_STYLES = {"toon": "keypose"}


def row_style(row: dict[str, str], forced: str | None = None) -> str:
    style = forced or (row.get("style") or "").strip() or DEFAULT_STYLE
    style = LEGACY_STYLES.get(style, style)
    if style not in STYLES:
        raise ValueError(f"{row['slug']}: unknown style {style!r} (expected one of {STYLES})")
    return style


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated slugs (default: all rows)")
    parser.add_argument("--style", choices=STYLES, default=None, help="force a style for every letter")
    parser.add_argument("--also", default="", help="extra cut-out styles for comparison, e.g. photo")
    parser.add_argument("--size", type=int, default=360)
    parser.add_argument("--build-skeleton", action="store_true", help="extract all S1 letters' landmarks -> skeleton")
    parser.add_argument("--from-json", action="store_true", help="keypose: re-render from the key-pose JSON only")
    args = parser.parse_args(argv)

    os.environ.setdefault("GLOG_minloglevel", "2")
    rows = load_map()
    if args.build_skeleton:
        sk = build_skeleton(rows)
        print(
            f"skeleton {sk.id}: {len(sk.letters)} letters, {sk.frames_used} frames, palm width "
            f"{sk.palm_width * 1000:.1f} mm, forearm {sk.forearm_width * 1000:.1f} mm"
        )
        if not args.only:
            return 0
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = sorted(set(wanted) - {r["slug"] for r in rows})
        if unknown:
            print(f"unknown slugs: {unknown}")
            return 2
        rows = [r for r in rows if r["slug"] in wanted]
    also = tuple(s.strip() for s in args.also.split(",") if s.strip())
    for s in also:
        if s not in CUTOUT_STYLES:
            parser.error(f"--also accepts {CUTOUT_STYLES}")

    failures: list[tuple[str, str]] = []
    for row in rows:
        t0 = time.perf_counter()
        try:
            style = row_style(row, args.style)
            if style == "keypose":
                r = process_keypose(row, from_json=args.from_json, size=args.size)
                extra = f"  {r.kind}, {r.key_poses} key pose(s)"
            elif style == "illustrated":
                r = process_row(row, style="illustrated", size=args.size)
                extra = ""
            else:
                styles = (style, *(s for s in also if s != style))
                dirs = {style: config.MEDIA_DIR, **{s: config.REPORTS_DIR / "styles" / s for s in styles[1:]}}
                r = process_cutout(
                    row, moving=row["slug"] in MOVEMENT_SIGNS, styles=styles, out_dirs=dirs, size=args.size
                )
                extra = f"  refined {r.refined}/{r.frames}"
        except Exception as exc:  # keep going; report at the end
            failures.append((row["slug"], f"{type(exc).__name__}: {exc}"))
            print(f"{row['slug']:<6} FAILED  {exc}", flush=True)
            if os.environ.get("ISHARA_DEBUG"):
                raise
            continue
        print(
            f"{r.slug:<6} {style:<11} {r.frames:>3} frames  animation {r.animation_kb:6.1f} KB  "
            f"total {r.total_kb:6.1f} KB  {time.perf_counter() - t0:5.1f}s{extra}  {r.source}",
            flush=True,
        )

    print(f"\n{len(rows) - len(failures)}/{len(rows)} letters done")
    for slug, err in failures:
        print(f"  FAILED {slug}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
