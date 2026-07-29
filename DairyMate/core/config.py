"""
Paths, constants and pipeline discovery.

The app never copies model weights — it reads them in place from the two
pipeline folders. Point it at them with environment variables:

    TEAT_PIPELINE_DIR=/path/to/teat_pipeline
    UDDER_PIPELINE_DIR=/path/to/udder_pipeline

or drop the folders next to this app, or set the paths in the sidebar.

Layout is not assumed. A pipeline folder is anything containing the model
file, at any depth — `models/teat_classifier.keras` and a loose
`teat_classifier.keras` both resolve. Report files are located the same way,
so a folder that has been flattened, renamed or partly unpacked still works.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

APP_NAME = "Dairy Mate"
APP_SUBTITLE = "Mastitis screening console — teat and udder models"
APP_VERSION = "1.1"

# ---------------------------------------------------------------- teat model
TEAT_IMG_SIZE = (224, 224)
# train_teat_model.py bakes mobilenet_v2.preprocess_input INTO the graph,
# so the model is fed raw 0-255 RGB. Do not rescale before calling predict.
TEAT_THRESHOLD = 0.50
TEAT_CLASSES = ("healthy", "mastitis")

# --------------------------------------------------------------- udder model
UDDER_IMG_SIZE = 128          # features.py resizes to 128x128
UDDER_FEATURE_DIM = 223
UDDER_FALLBACK_THRESHOLD = 0.60   # overridden by decision_threshold.json
UDDER_CLASSES = ("healthy", "mastitis")

CLASS_LABELS = {
    "healthy": "Healthy",
    "mastitis": "Mastitis indicated",
}

# --------------------------------------------------------------- discovery
MODEL_FILE = {
    "teat": "teat_classifier.keras",
    "udder": "udder_svm_model.joblib",
}

# Folder names worth trying, in order. Spelling varies between the two repos
# ("teat_pipeline" vs "teats_pipeline"), so check both.
FOLDER_NAMES = {
    "teat": ("teat_pipeline", "teats_pipeline", "teat-pipeline", "Teat-Pipeline"),
    "udder": ("udder_pipeline", "udders_pipeline", "udder-pipeline", "Udder-Pipeline"),
}

_MAX_DEPTH = 6
_SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env"}


def _walk(base: Path, max_depth: int = _MAX_DEPTH):
    """Yield files under base, depth-limited, skipping noise directories."""
    base = base.resolve()
    base_depth = len(base.parts)
    for root, dirs, files in os.walk(base, topdown=True):
        root_path = Path(root)
        if len(root_path.parts) - base_depth >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            yield root_path / f


def _find_named(base: Path, filename: str, max_depth: int = _MAX_DEPTH) -> Path | None:
    """Shallowest file called `filename` under base."""
    best: Path | None = None
    best_depth = 10 ** 6
    try:
        for f in _walk(base, max_depth):
            if f.name.lower() == filename.lower():
                depth = len(f.parts)
                if depth < best_depth:
                    best, best_depth = f, depth
    except OSError:
        return None
    return best


def _root_from_model(model_file: Path) -> Path:
    """The pipeline root is the model's folder, or its parent if that folder is models/."""
    parent = model_file.parent
    return parent.parent if parent.name.lower() == "models" else parent


def _search_bases():
    """Where to look, nearest first."""
    seen = set()
    for base in (APP_DIR, APP_DIR.parent, APP_DIR.parent.parent,
                 APP_DIR.parent.parent.parent, Path.cwd(), Path.cwd().parent):
        try:
            r = base.resolve()
        except OSError:
            continue
        if r not in seen and r.is_dir():
            seen.add(r)
            yield r


@lru_cache(maxsize=8)
def find_pipeline(kind: str) -> Path | None:
    """kind is 'teat' or 'udder'. Returns the pipeline root, or None."""
    marker = MODEL_FILE[kind]

    # 1. Explicit path wins. Accept it whether it is the root, the models/
    #    folder, or an ancestor of either.
    env = os.environ.get(f"{kind.upper()}_PIPELINE_DIR")
    if env:
        p = Path(env).expanduser()
        if p.is_file() and p.name.lower() == marker.lower():
            return _root_from_model(p)
        if p.is_dir():
            hit = _find_named(p, marker)
            if hit is not None:
                return _root_from_model(hit)
        return None

    # 2. A folder with a name we recognise, near the app. `pipelines/` is the
    #    layout this project ships with; the bare names cover looser setups.
    for base in _search_bases():
        for parent in (base / "pipelines", base):
            for name in FOLDER_NAMES[kind]:
                cand = parent / name
                if cand.is_dir():
                    hit = _find_named(cand, marker)
                    if hit is not None:
                        return _root_from_model(hit)

    # 3. Give up on names and look for the model file itself.
    for base in _search_bases():
        hit = _find_named(base, marker)
        if hit is not None:
            return _root_from_model(hit)

    return None


# --------------------------------------------------------------- artifacts
@lru_cache(maxsize=64)
def _resolve(root_str: str, filename: str) -> Path:
    """
    Locate an artifact inside a pipeline. Returns a real path if found, or a
    plausible one if not — callers test .exists() and degrade gracefully.
    """
    root = Path(root_str)
    hit = _find_named(root, filename)
    return hit if hit is not None else root / filename


@lru_cache(maxsize=16)
def _resolve_dir(root_str: str, dirname: str) -> Path:
    root = Path(root_str)
    try:
        for path in root.rglob(dirname):
            if path.is_dir():
                return path
    except OSError:
        pass
    return root / dirname


def teat_paths(root: Path) -> dict:
    r = str(root)
    return {
        "model": _resolve(r, "teat_classifier.keras"),
        "evaluation": _resolve(r, "teat_evaluation_report.json"),
        "history": _resolve(r, "teat_training_history.json"),
        "confusion_png": _resolve(r, "teat_confusion_matrix.png"),
        "gradcam_grid": _resolve(r, "gradcam_summary_grid.png"),
        "hand_report": _resolve(r, "hand_detection_report.csv"),
        "readme": _resolve(r, "README.md"),
    }


def udder_paths(root: Path) -> dict:
    r = str(root)
    return {
        "model": _resolve(r, "udder_svm_model.joblib"),
        "threshold": _resolve(r, "decision_threshold.json"),
        "best_config": _resolve(r, "best_config.json"),
        "test_metrics": _resolve(r, "test_metrics.txt"),
        "model_comparison": _resolve(r, "model_comparison.csv"),
        "tuning_log": _resolve(r, "hyperparameter_tuning_log.csv"),
        "error_analysis": _resolve(r, "error_analysis.txt"),
        "splits": _resolve_dir(r, "splits"),
        "segmented": _resolve_dir(r, "segmented"),
        "readme": _resolve(r, "README.md"),
    }
