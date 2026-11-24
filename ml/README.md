# Ishara ML

Python pipeline that turns public Nepali Sign Language (NSL) data into the on-device models used by the
`ishara-app`:

- `fingerspelling.tflite` + `labels.json` + `metadata.json`: a small classifier over normalised MediaPipe hand
  landmarks (21 × (x, y, z) = 63 features).
- Greeting DTW templates: one JSON file per sign with calibrated `correct` and `almost` thresholds. These give
  practice feedback only; they don't verify recognition.

Artefacts are delivered to `ishara-app/assets/ml/` by pull request into that repo.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 automatically (pinned in `.python-version`,
because MediaPipe and TensorFlow wheels are most reliable on 3.12).

```bash
cd ml
uv sync                      # creates .venv and installs everything from uv.lock
uv run pytest                # smoke tests: dependencies import, constants are consistent
uv run ruff check .          # lint
uv run ruff format .         # format
uv run jupyter lab           # notebooks (they only call code in src/)
```

Add a dependency with `uv add <pkg>` (or `uv add --dev <pkg>`), then commit `pyproject.toml` and `uv.lock`.

## Layout

```
ml/
├── src/ishara_ml/
│   ├── config.py        # seeds, LANDMARK_SPEC_VERSION, feature size, paths
│   └── (planned) landmarks, normalize, dataset, train, evaluate, export, templates
├── tests/               # pytest
├── notebooks/           # exploration only; import from ishara_ml
├── data/                # git-ignored: downloaded public datasets + extracted landmarks
└── artifacts/           # git-ignored: trained models, labels, metadata, templates
```

## Key packages

| Package | Used for |
|---|---|
| `mediapipe` | Hand landmark extraction (same Hand Landmarker model version as the app) |
| `tensorflow` | Small MLP classifier + TFLite export |
| `scikit-learn` | RandomForest baseline, signer-grouped splits, metrics |
| `numpy`, `pandas` | Feature arrays and tables |
| `matplotlib` | Confusion matrices and plots for `docs/evaluation.md` |
| `dtaidistance` | DTW for greeting templates |
| `opencv-contrib-python` (via mediapipe) | Reading images and video frames; don't add a second OpenCV package |

## Method rules

- Landmark normalisation follows spec v1 (`docs/landmark-spec.md`): wrist moved to the origin, scaled by the
  wrist → middle-MCP distance, x mirrored for left hands, then flattened. Any spec change bumps
  `LANDMARK_SPEC_VERSION`, regenerates `shared/fixtures/landmark_parity.json` and requires retraining.
- Splits are **signer-independent**: whole signers are held out.
- Report accuracy, macro-F1, per-class and per-signer results, and a confusion matrix. Compare the MLP with a
  RandomForest baseline. Thesis target: ≥ 85% accuracy on held-out signers.
- Fixed seeds (`ishara_ml.config.set_seeds()`).

## Data (public only)

| Dataset | Notes | Citation and licence |
|---|---|---|
| NSL23 (Zenodo) | Consonants and vowels, 14 signers including 5 native | TODO: add citation and licence |
| Poudel et al. image set (optional) | ~54k images, 36 classes | TODO: add citation and licence |

Datasets, extracted landmarks, model artefacts and videos are never committed.

## Privacy

This pipeline uses public datasets and developer recordings only. User camera frames, video or hand landmarks are
never collected, stored or used for training.
