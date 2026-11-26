"""Find NSL23 videos by signer and file name.

NSL23 folders start with the signer id (``S1_NSL_Consonant_Bright``). Matching must keep ``S1`` apart from
``S10``-``S14``, which contain files with the same names. File names are matched case-insensitively because the
dataset mixes ``.MOV`` and ``.mov``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path, PurePosixPath


class VideoNotFoundError(LookupError):
    """No file matched the signer and file name."""


class AmbiguousVideoError(LookupError):
    """More than one file matched; ``candidates`` lists them."""

    def __init__(self, message: str, candidates: list[Path]) -> None:
        super().__init__(message)
        self.candidates = candidates


def is_signer_dir(name: str, signer: str) -> bool:
    """True if a folder name belongs to ``signer``: exactly ``S1`` or ``S1_...`` (never ``S10``)."""
    name, signer = name.casefold(), signer.casefold()
    return name == signer or name.startswith(signer + "_")


def signer_dirs(root: Path, signer: str) -> Iterator[Path]:
    """All folders under ``root`` (any depth) that belong to ``signer``."""
    for path in sorted(root.rglob("*")):
        if path.is_dir() and is_signer_dir(path.name, signer):
            yield path


def find_video(root: Path, signer: str, file: str) -> list[Path]:
    """Every file matching ``file`` inside ``signer``'s folders.

    ``file`` is either a bare name (``S1_KA.MOV``) or ``<folder>/<name>``
    (``S1_NSL_Consonant_Bright/S1_KA.MOV``), which also requires the immediate parent folder to match.
    Matching is case-insensitive. Files are only searched directly inside signer folders.
    """
    parts = PurePosixPath(file.replace("\\", "/")).parts
    if not parts or len(parts) > 2:
        raise ValueError(f"file must be '<name>' or '<folder>/<name>', got {file!r}")
    name = parts[-1].casefold()
    folder = parts[0].casefold() if len(parts) == 2 else None

    matches: set[Path] = set()
    for directory in signer_dirs(root, signer):
        if folder is not None and directory.name.casefold() != folder:
            continue
        for child in directory.iterdir():
            if child.is_file() and child.name.casefold() == name:
                matches.add(child)
    return sorted(matches)


def resolve_video(root: Path, signer: str, file: str) -> Path:
    """The single file for ``signer``/``file``; raises if there are none or several."""
    candidates = find_video(root, signer, file)
    if not candidates:
        raise VideoNotFoundError(f"no file {file!r} for signer {signer} under {root}")
    if len(candidates) > 1:
        raise AmbiguousVideoError(f"{len(candidates)} files match {file!r} for signer {signer}", candidates)
    return candidates[0]
