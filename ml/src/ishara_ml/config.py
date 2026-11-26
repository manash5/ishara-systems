"""Shared constants, paths and seeding for the ML pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 42

# Normalisation spec (docs/landmark-spec.md). Bump on any spec change, regenerate the parity fixture, retrain.
LANDMARK_SPEC_VERSION = 1
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63

ML_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_ROOT.parent
DATA_DIR = ML_ROOT / "data"  # git-ignored: public datasets only
RAW_DIR = DATA_DIR / "raw"
NSL23_DIR = RAW_DIR / "nsl23"  # the 4 NSL23 zips extracted with their folder structure untouched
LABELS_DIR = ML_ROOT / "labels"  # versioned: dataset label maps (e.g. nsl23_map.csv)
SIGNS_JSON = REPO_ROOT / "backend" / "content" / "signs.json"  # content source of truth
LANDMARKS_DIR = DATA_DIR / "landmarks"
ARTIFACTS_DIR = ML_ROOT / "artifacts"  # git-ignored: .tflite, labels.json, metadata.json, templates
PARITY_FIXTURE = REPO_ROOT / "shared" / "fixtures" / "landmark_parity.json"


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if imported) TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
