"""Score a whole folder or a multi-file upload and export the results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from PIL import Image

from core import auth, config, models, state, theme, ui

theme.inject("Batch run · Dairy Mate")
auth.require_login()

SLOT = "batch"

teat_root = config.find_pipeline("teat")
udder_root = config.find_pipeline("udder")
ui.sidebar_status(teat_root, udder_root)
auth.sidebar_account()

ui.masthead(
    "Batch",
    "Score a set of images",
    "One model, many images, one CSV out. Sorted by score so the animals that "
    "need looking at first are at the top.",
)

model_choice = st.radio("Model", ["Teat", "Udder"], horizontal=True)
root = teat_root if model_choice == "Teat" else udder_root
folder_name = "teat_pipeline" if model_choice == "Teat" else "udder_pipeline"

if root is None:
    ui.missing_pipeline(model_choice.lower(), folder_name)
    st.stop()

paths = config.teat_paths(root) if model_choice == "Teat" else config.udder_paths(root)

if model_choice == "Teat":
    threshold = st.slider("Decision cut-off", 0.05, 0.95, config.TEAT_THRESHOLD, 0.01)
else:
    calibrated = models.load_udder_threshold(str(paths["threshold"]))
    threshold = st.slider("Decision cut-off", 0.20, 0.85, float(calibrated), 0.01)

source = st.radio("Source", ["Upload files", "Folder on this machine"], horizontal=True)

images: list[tuple[str, Image.Image]] = []
if source == "Upload files":
    ups = st.file_uploader(
        "Images", type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True, key="batch_uploader",
    )
    if ups:
        images = [(u.name, Image.open(u)) for u in ups]
else:
    folder = st.text_input("Folder path", placeholder="/path/to/images", key="batch_folder")
    if folder:
        p = Path(folder).expanduser()
        if not p.is_dir():
            st.error("That path is not a folder.")
        else:
            files = sorted(
                f for f in p.rglob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
            st.caption(f"{len(files)} image(s) found.")
            images = [(str(f.relative_to(p)), Image.open(f)) for f in files[:500]]

run = st.button(f"Score {len(images)} image(s)", type="primary", disabled=not images)

# ------------------------------------------------------------------- run
if run and images:
    rows, failures = [], 0
    bar = st.progress(0.0, text="Scoring…")

    for i, (name, pil) in enumerate(images, start=1):
        try:
            if model_choice == "Teat":
                pred = models.predict_teat(
                    pil, str(paths["model"]), threshold, with_gradcam=False
                )
            else:
                pred = models.predict_udder(
                    models.pil_to_bgr(pil), str(paths["model"]), threshold,
                    use_segmentation=False,
                )
            rows.append({
                "image": name,
                "p_mastitis": round(pred.probability, 4),
                "verdict": config.CLASS_LABELS[pred.label],
                "margin": round(abs(pred.probability - threshold), 4),
                "review": "yes" if abs(pred.probability - threshold) < 0.08 else "",
            })
        except Exception as exc:      # noqa: BLE001
            failures += 1
            rows.append({"image": name, "p_mastitis": None,
                         "verdict": f"failed: {exc}", "margin": None, "review": ""})
        bar.progress(i / len(images), text=f"Scoring… {i}/{len(images)}")

    bar.empty()
    df = pd.DataFrame(rows).sort_values("p_mastitis", ascending=False, na_position="last")
    # Kept under a fixed key so the table survives a trip to another page.
    state.store_result(SLOT, "last", df,
                       extra={"model": model_choice, "threshold": threshold,
                              "failures": failures})

# --------------------------------------------------------------- results
df = state.cached_result(SLOT, "last")
if df is None:
    st.caption("Pick a source, then score. Results stay on this page until you replace them.")
    st.stop()

meta = state.cached_extra(SLOT) or {}
failures = meta.get("failures", 0)

head, wipe = st.columns([5, 1])
with head:
    st.caption(
        f"Last run · {meta.get('model', '—')} model · cut-off "
        f"{meta.get('threshold', float('nan')):.2f} · {len(df)} image(s)"
    )
with wipe:
    if st.button("Clear", key="batch_clear"):
        state.forget(SLOT)
        st.rerun()

flagged = int((df["verdict"] == config.CLASS_LABELS["mastitis"]).sum())
borderline = int((df["review"] == "yes").sum())

m1, m2, m3, m4 = st.columns(4)
with m1:
    ui.stat("Scored", str(len(df) - failures))
with m2:
    ui.stat("Flagged", str(flagged))
with m3:
    ui.stat("Borderline", str(borderline), "Within 0.08 of the cut-off")
with m4:
    ui.stat("Failed", str(failures))

st.dataframe(df, hide_index=True, use_container_width=True, height=460)
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode(),
    file_name=f"dairymate_{str(meta.get('model', 'batch')).lower()}_batch.csv",
    mime="text/csv",
)
