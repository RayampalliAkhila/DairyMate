"""Udder classifier — GrabCut segmentation, 223-dim features, calibrated SVM."""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core import auth, config, models, state, theme, ui
from core.udder_features import FEATURE_BLOCKS
from core.udder_segment import grabcut_segment, mask_outline

theme.inject("Udder analysis · Dairy Mate")
auth.require_login()

SLOT = "udder"

udder_root = config.find_pipeline("udder")
ui.sidebar_status(config.find_pipeline("teat"), udder_root)
auth.sidebar_account()

ui.masthead(
    "Udder model · SVM on 223 features",
    "Udder analysis",
    "Segmentation, feature extraction and a threshold-calibrated SVM score, "
    "with each stage shown separately so you can see where a reading comes from.",
)

if udder_root is None:
    ui.missing_pipeline("udder", "udder_pipeline")
    st.stop()

paths = config.udder_paths(udder_root)
if not paths["model"].exists():
    st.error(f"Model file missing: {paths['model']}")
    st.stop()

calibrated = models.load_udder_threshold(str(paths["threshold"]))

# ------------------------------------------------------------------ controls
with st.sidebar:
    st.markdown("**Udder settings**")
    threshold = st.slider(
        "Decision cut-off", 0.20, 0.85, float(calibrated), 0.01,
        help=f"Calibrated value is {calibrated:.2f}, chosen on 13 validation images.",
    )
    use_seg = st.toggle(
        "Feed GrabCut output to the model", value=False,
        help="Off by default — the shipped SVM was fitted on unsegmented images.",
    )
    show_seg = st.toggle("Show the segmentation", value=True)

ui.note(
    "<b>Segmentation is in this pipeline but not in the model's input path.</b> "
    "<code>src/segment.py</code> runs GrabCut over all 95 images and writes them to "
    "<code>data/processed/segmented/</code>, but the split manifests still point at "
    "<code>data/raw/</code>, so <code>features.py</code> reads the unsegmented "
    "originals. Leave the toggle off to reproduce the reported metrics; turn it on "
    "to preview what a segmentation-fed model would see.",
    kind="warn",
)
st.write("")

upload = st.file_uploader(
    "Udder image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="udder_uploader"
)
restored = state.is_restored(SLOT, upload)
img_bytes, img_name = state.remember_image(SLOT, upload)

if img_bytes is None:
    st.stop()

if restored:
    held, clear = st.columns([4, 1])
    with held:
        st.caption(f"Showing **{img_name}** from earlier. Upload another to replace it.")
    with clear:
        if st.button("Clear", key="udder_clear"):
            state.forget(SLOT)
            st.rerun()

pil = Image.open(io.BytesIO(img_bytes))
bgr = models.pil_to_bgr(pil)

sig = state.signature(img_bytes, threshold, use_seg, show_seg)
cached = state.cached_result(SLOT, sig)

if cached is not None:
    pred, seg = cached, state.cached_extra(SLOT)
else:
    try:
        with st.spinner("Segmenting and extracting features…"):
            pred = models.predict_udder(
                bgr, str(paths["model"]), threshold, use_segmentation=use_seg
            )
            seg = pred.segmentation if pred.segmentation is not None else (
                grabcut_segment(bgr) if show_seg else None
            )
        state.store_result(SLOT, sig, pred, extra=seg)
    except Exception as exc:      # noqa: BLE001
        st.error(f"Scoring failed: {exc}")
        st.stop()

# ------------------------------------------------------------------ result
ui.verdict(pred.label, pred.probability, pred.threshold)
ui.decision_strip(pred.probability, pred.threshold)
st.write("")

cols = st.columns([1, 1, 1, 0.9], gap="medium")

with cols[0]:
    ui.caption("Submitted")
    st.image(pil, use_container_width=True)
    st.caption(img_name or "")

if show_seg and seg is not None:
    with cols[1]:
        ui.caption("Foreground boundary")
        if seg.mask is not None:
            st.image(models.bgr_to_rgb(mask_outline(bgr, seg.mask)), use_container_width=True)
        else:
            st.image(pil, use_container_width=True)
            st.caption("GrabCut rejected — original kept.")
    with cols[2]:
        ui.caption("Segmented")
        st.image(models.bgr_to_rgb(seg.image), use_container_width=True)
        st.caption(seg.reason)
else:
    with cols[1]:
        ui.caption("Segmentation")
        st.info("Switched off in the sidebar.")

with cols[-1]:
    ui.caption("Detail")
    ui.stat("p(mastitis)", f"{pred.probability:.4f}")
    ui.stat("Cut-off", f"{threshold:.2f}",
            "calibrated" if abs(threshold - calibrated) < 1e-9 else f"calibrated: {calibrated:.2f}")
    ui.stat("Fed to model", "Segmented" if pred.used_segmentation else "Raw image",
            "Raw matches training")
    ui.stat("Feature vector", f"{pred.features.shape[0]}-dim")

st.divider()

# ------------------------------------------------------------- features
st.markdown("### What the model actually measured")
st.markdown(
    '<div class="dm-sub">The SVM never sees pixels. It sees 223 numbers, grouped '
    "as follows. The bars are per-block magnitude after the pipeline's own "
    "standardisation — a long bar means this image is unusual on that block "
    "relative to the training set.</div>",
    unsafe_allow_html=True,
)
st.write("")

pipe = models.load_udder_model(str(paths["model"]))
scaled = pipe.named_steps["scaler"].transform(pred.features.reshape(1, -1))[0]

rows, idx = [], 0
for name, size, desc in FEATURE_BLOCKS:
    block = scaled[idx: idx + size]
    rows.append({
        "Block": name,
        "Dims": size,
        "Deviation": float(np.abs(block).mean()),
        "Max |z|": float(np.abs(block).max()),
        "Measures": desc,
    })
    idx += size

df = pd.DataFrame(rows)
st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Deviation": st.column_config.ProgressColumn(
            "Mean |z|", min_value=0.0, max_value=float(max(3.0, df["Deviation"].max())),
            format="%.2f",
        ),
        "Max |z|": st.column_config.NumberColumn(format="%.2f"),
    },
)

with st.expander("Model card"):
    import json
    cfg = {}
    if paths["best_config"].exists():
        cfg = json.loads(paths["best_config"].read_text())
    st.markdown(
        f"""
| | |
|---|---|
| Estimator | `SVC` — RBF kernel, `class_weight="balanced"`, `probability=True` |
| Hyperparameters | C = {cfg.get('C', '5.0')}, gamma = {cfg.get('gamma', '0.001')}, kernel = {cfg.get('kernel', 'rbf')} |
| Preprocessing | `StandardScaler` inside an `imblearn` Pipeline |
| Trained on | 66 images, from 95 confidently labelled (55 mastitis / 40 healthy) |
| Selection | 3-fold **group** CV — near-duplicate frames stay in the same fold |
| Cut-off | {calibrated:.2f}, swept on 13 validation images |

Why SVM won over RandomForest and GradientBoosting: with 66 training rows, the
stronger regularisation generalises where tree ensembles memorise.

The honest caveat from the pipeline's own README — roughly 328 further udder
images have no reliable label and were excluded rather than guessed at.
Labelling them is worth more than any change to this model.
"""
    )
