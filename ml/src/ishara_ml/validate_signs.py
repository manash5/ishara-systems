"""Validate backend/content/signs.json against ml/labels/nsl23_map.csv and the extracted NSL23 videos.

Run from ``ml/``:  ``uv run python -m ishara_ml.validate_signs``  (exit code 1 on any problem).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from ishara_ml import config
from ishara_ml.nsl23 import find_video

EXPECTED_COUNT = 49
SLUG_RE = re.compile(r"^[a-z0-9_]+$")
CSV_COLUMNS = ["slug", "glyph", "nsl23_label", "signer", "file", "start", "end"]


def check_signs(signs_path: Path) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    signs = json.loads(signs_path.read_text(encoding="utf-8")).get("signs")
    if not isinstance(signs, list):
        return [f"{signs_path.name}: top-level 'signs' array missing"], {}

    if len(signs) != EXPECTED_COUNT:
        errors.append(f"signs.json: expected {EXPECTED_COUNT} entries, found {len(signs)}")
    slugs = [s.get("slug") for s in signs]
    for slug in slugs:
        if not isinstance(slug, str) or not SLUG_RE.match(slug):
            errors.append(f"signs.json: invalid slug {slug!r}")
    dupes = sorted({s for s in slugs if slugs.count(s) > 1})
    if dupes:
        errors.append(f"signs.json: duplicate slugs {dupes}")
    orders = [s.get("sortOrder") for s in signs]
    if not all(isinstance(o, int) for o in orders) or sorted(orders) != list(range(1, len(signs) + 1)):
        errors.append(f"signs.json: sortOrder must be 1..{len(signs)} without gaps or duplicates")
    for s in signs:
        if not isinstance(s.get("glyph"), str) or not s["glyph"].strip():
            errors.append(f"signs.json: {s.get('slug')}: empty glyph")
        if s.get("howTo") is not None:
            errors.append(f"signs.json: {s.get('slug')}: howTo must be null")
    return errors, {s["slug"]: s.get("glyph") for s in signs if isinstance(s.get("slug"), str)}


def check_map(csv_path: Path, glyphs: dict[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != CSV_COLUMNS:
            errors.append(f"{csv_path.name}: columns {reader.fieldnames} != {CSV_COLUMNS}")
        rows = list(reader)

    csv_slugs = [r["slug"] for r in rows]
    if len(csv_slugs) != len(set(csv_slugs)):
        errors.append(f"{csv_path.name}: duplicate slugs")
    missing, extra = sorted(set(glyphs) - set(csv_slugs)), sorted(set(csv_slugs) - set(glyphs))
    if missing:
        errors.append(f"{csv_path.name}: slugs missing vs signs.json: {missing}")
    if extra:
        errors.append(f"{csv_path.name}: slugs not in signs.json: {extra}")
    for r in rows:
        if r["slug"] in glyphs and r["glyph"] != glyphs[r["slug"]]:
            errors.append(f"{csv_path.name}: {r['slug']}: glyph {r['glyph']!r} != signs.json {glyphs[r['slug']]!r}")
        for col in ("nsl23_label", "signer", "file"):
            if not r[col].strip():
                errors.append(f"{csv_path.name}: {r['slug']}: empty {col}")
    return errors, rows


def check_files(rows: list[dict[str, str]], root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    resolved = 0
    if not root.is_dir():
        return [f"dataset folder not found: {root}"], 0
    for r in rows:
        if not (r["signer"].strip() and r["file"].strip()):
            continue
        hits = find_video(root, r["signer"], r["file"])
        if len(hits) == 1:
            resolved += 1
            continue
        if hits:
            msg = f"{r['slug']}: {len(hits)} files match {r['file']!r}:"
        else:
            # Show what a bare-name search would find, to help pick the right folder.
            name = r["file"].replace("\\", "/").rsplit("/", 1)[-1]
            hits = find_video(root, r["signer"], name)
            msg = f"{r['slug']}: no file {r['file']!r}; same name elsewhere for {r['signer']}:"
        lines = [msg] + ([f"    {h.relative_to(root).as_posix()}" for h in hits] or ["    (none)"])
        if hits:
            lines.append(f"    proposed file: {hits[0].parent.name}/{hits[0].name}")
        errors.append("\n".join(lines))
    return errors, resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signs", type=Path, default=config.SIGNS_JSON)
    parser.add_argument("--map", type=Path, default=config.LABELS_DIR / "nsl23_map.csv")
    parser.add_argument("--root", type=Path, default=config.NSL23_DIR)
    args = parser.parse_args(argv)

    sign_errors, glyphs = check_signs(args.signs)
    map_errors, rows = check_map(args.map, glyphs)
    file_errors, resolved = check_files(rows, args.root)

    print(f"signs.json : {len(glyphs)} signs, {len(sign_errors)} problem(s)")
    print(f"nsl23 map  : {len(rows)} rows, {len(map_errors)} problem(s)")
    print(f"videos     : {resolved}/{len(rows)} resolved to exactly one file")
    errors = sign_errors + map_errors + file_errors
    for e in errors:
        print(f"  - {e}")
    print("OK" if not errors else f"FAILED: {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
