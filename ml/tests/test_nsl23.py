"""Tests for the NSL23 video resolver on a small fake dataset tree."""

from pathlib import Path

import pytest

from ishara_ml.nsl23 import (
    AmbiguousVideoError,
    VideoNotFoundError,
    find_video,
    is_signer_dir,
    resolve_video,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    files = [
        "Part_1/S1_NSL_Consonant_Bright/S1_KA.MOV",
        "Part_1/S1_NSL_Consonant_Dark/S1_KA.MOV",
        "Part_1/S1_NSL_Consonant_Bright/S1_GA.MOV",
        "Part_2/S10_NSL_Consonant_Prepared/S1_GA.MOV",
        "Part_2/S10_NSL_Consonant_Prepared/S1_HA.MOV",
        "Part_3/S1/S1_YA.mov",
    ]
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize(
    ("name", "expected"),
    [("S1", True), ("S1_NSL_Vowel", True), ("s1_x", True), ("S10_NSL", False), ("S14", False), ("S2_x", False)],
)
def test_is_signer_dir(name: str, expected: bool) -> None:
    assert is_signer_dir(name, "S1") is expected


def test_s1_never_matches_s10(root: Path) -> None:
    assert [p.parent.name for p in find_video(root, "S1", "S1_GA.MOV")] == ["S1_NSL_Consonant_Bright"]
    assert find_video(root, "S1", "S1_HA.MOV") == []


def test_case_insensitive_and_exact_signer_folder(root: Path) -> None:
    assert resolve_video(root, "S1", "s1_ya.MOV").name == "S1_YA.mov"


def test_folder_narrows_to_one(root: Path) -> None:
    assert len(find_video(root, "S1", "S1_KA.MOV")) == 2
    path = resolve_video(root, "S1", "s1_nsl_consonant_bright/S1_KA.MOV")
    assert path.parent.name == "S1_NSL_Consonant_Bright"


def test_ambiguous_lists_candidates(root: Path) -> None:
    with pytest.raises(AmbiguousVideoError) as exc:
        resolve_video(root, "S1", "S1_KA.MOV")
    assert len(exc.value.candidates) == 2


def test_not_found(root: Path) -> None:
    with pytest.raises(VideoNotFoundError):
        resolve_video(root, "S1", "S1_NSL_Consonant_Dark/S1_GA.MOV")
