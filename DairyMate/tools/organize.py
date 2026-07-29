"""
organize.py — put a scattered pipeline folder into the standard layout.

Target shape, per pipeline:

    <name>/
      models/     weights, thresholds, fitted configs
      reports/    metrics, plots, logs, Grad-CAM contact sheets
      scripts/    the training and evaluation code
      data/       left exactly as found
      README.md, requirements.txt stay at the root

Dry run by default — it prints what it would do and changes nothing. Add
--apply once the plan looks right, and --move if you want the originals gone
rather than copied.

    python tools/organize.py --source "C:\\path\\to\\messy" --dest "C:\\DairyMate"
    python tools/organize.py --source "C:\\path\\to\\messy" --dest "C:\\DairyMate" --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ------------------------------------------------------------ classification
MARKER = {
    "teat_pipeline": "teat_classifier.keras",
    "udder_pipeline": "udder_svm_model.joblib",
}

MODEL_EXT = {".keras", ".joblib", ".h5", ".hdf5", ".pkl", ".pickle",
             ".task", ".pb", ".onnx", ".tflite"}
# Small JSONs that are part of the fitted model, not a report about it.
MODEL_NAMES = {"decision_threshold.json", "best_config.json", "label_map.json",
               "class_indices.json"}
CODE_EXT = {".py", ".ipynb", ".sh", ".ps1"}
ROOT_NAMES = {"readme.md", "requirements.txt", "license", "license.md",
              ".gitignore", "makefile", "pyproject.toml"}
REPORT_EXT = {".csv", ".txt", ".png", ".jpg", ".jpeg", ".svg", ".html", ".pdf",
              ".log", ".tsv", ".md"}
REPORT_HINTS = ("report", "metric", "history", "confusion", "comparison",
                "tuning", "error", "gradcam", "overlay", "summary", "log",
                "curve", "roc", "matrix", "analysis", "eval")

DIR_MAP = {
    "models": "models", "model": "models", "weights": "models", "checkpoints": "models",
    "reports": "reports", "report": "reports", "results": "reports", "figures": "reports",
    "outputs": "reports", "logs": "reports", "plots": "reports",
    "scripts": "scripts", "src": "scripts", "code": "scripts", "notebooks": "scripts",
    "data": "data", "datasets": "data",
}
PASSTHROUGH = {"data"}          # copied wholesale, never rearranged


def classify_file(path: Path) -> str:
    name = path.name.lower()
    ext = path.suffix.lower()

    if name in ROOT_NAMES:
        return "."
    if ext in ARCHIVE_EXT:
        return "archive"
    if name in MODEL_NAMES or ext in MODEL_EXT:
        return "models"
    if ext in CODE_EXT:
        return "scripts"
    if ext == ".json":
        return "reports" if any(h in name for h in REPORT_HINTS) else "models"
    if ext in REPORT_EXT or any(h in name for h in REPORT_HINTS):
        return "reports"
    return "misc"


ARCHIVE_EXT = {".bak", ".old", ".orig", ".backup"}


# ---------------------------------------------------------------- discovery
def find_pipelines(source: Path) -> dict[str, Path]:
    """Locate each pipeline by its marker file, shallowest match wins."""
    found: dict[str, Path] = {}
    for kind, marker in MARKER.items():
        best, best_depth = None, 10 ** 6
        for f in source.rglob(marker):
            parts = {p.lower() for p in f.parts}
            if parts & {"__pycache__", ".git", "venv", ".venv"}:
                continue
            depth = len(f.parts)
            if depth < best_depth:
                best, best_depth = f, depth
        if best is not None:
            root = best.parent
            if root.name.lower() == "models":
                root = root.parent
            found[kind] = root
    return found


# -------------------------------------------------------------------- plan
def build_plan(root: Path, dest_root: Path) -> list[tuple[Path, Path, str]]:
    """(source, destination, note) for everything directly under a pipeline."""
    plan: list[tuple[Path, Path, str]] = []

    for entry in sorted(root.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue

        if entry.is_dir():
            key = entry.name.lower()
            if key in DIR_MAP:
                bucket = DIR_MAP[key]
                if bucket == "data":
                    plan.append((entry, dest_root / "data", "copied as found"))
                else:
                    # Already a standard folder — merge its contents in rather
                    # than nesting reports/reports/.
                    for child in sorted(entry.rglob("*")):
                        if child.is_file() and "__pycache__" not in child.parts:
                            rel = child.relative_to(entry)
                            plan.append((child, dest_root / bucket / rel,
                                         f"from {entry.name}/"))
            else:
                # Anything else keeps its own name and internal structure —
                # gradcam/healthy/ must not collapse into one flat folder.
                plan.append((entry, dest_root / "reports" / entry.name,
                             "folder kept whole"))
        else:
            bucket = classify_file(entry)
            target = dest_root if bucket == "." else dest_root / bucket
            plan.append((entry, target / entry.name, ""))

    return plan


def render(plan, root: Path, dest_root: Path) -> None:
    buckets: dict[str, list[str]] = {}
    for src, dst, note in plan:
        try:
            key = str(dst.relative_to(dest_root).parent)
        except ValueError:
            key = "."
        label = dst.name + (f"   ({note})" if note else "")
        buckets.setdefault(key or ".", []).append(label)

    print(f"\n  {dest_root.name}/")
    for key in sorted(buckets, key=lambda k: (k == ".", k)):
        shown = sorted(set(buckets[key]))
        if key == ".":
            for item in shown:
                print(f"    {item}")
        else:
            print(f"    {key}/")
            for item in shown[:8]:
                print(f"      {item}")
            if len(shown) > 8:
                print(f"      … {len(shown) - 8} more")


def execute(plan, move: bool) -> int:
    done = 0
    for src, dst, _ in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            # dst already carries the final folder name, so copy straight onto it.
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        done += 1
    if move:
        # Only after every copy landed — never delete before the write succeeds.
        for src, _, _ in plan:
            try:
                shutil.rmtree(src) if src.is_dir() else src.unlink()
            except OSError:
                pass
    return done


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Folder to search for pipelines.")
    ap.add_argument("--dest", required=True, help="Project root to build into.")
    ap.add_argument("--apply", action="store_true", help="Actually write files.")
    ap.add_argument("--move", action="store_true",
                    help="Remove originals after a successful copy.")
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()

    if not source.is_dir():
        print(f"Source is not a folder: {source}")
        return 1

    print(f"Searching {source} …")
    found = find_pipelines(source)

    for kind in MARKER:
        if kind in found:
            print(f"  found {kind:15s} {found[kind]}")
        else:
            print(f"  MISSING {kind:14s} (no {MARKER[kind]} anywhere under source)")

    if not found:
        print("\nNothing to organise.")
        return 1

    all_plans = []
    for kind, root in found.items():
        dest_root = dest / "pipelines" / kind
        plan = build_plan(root, dest_root)
        all_plans.extend(plan)
        render(plan, root, dest_root)

    total = len(all_plans)
    if not args.apply:
        print(f"\nDry run — {total} item(s) would be "
              f"{'moved' if args.move else 'copied'}. Nothing has changed.")
        print("Re-run with --apply to do it.")
        return 0

    print(f"\n{'Moving' if args.move else 'Copying'} {total} item(s)…")
    done = execute(all_plans, args.move)
    print(f"Done — {done} item(s) written to {dest}")
    print("\nStart the app with:")
    print(f"    cd {dest}")
    print("    streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
