"""Teat classifier — score plus Grad-CAM evidence."""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from core import auth, config, models, state, theme, ui

theme.inject("Teat analysis · Dairy Mate")
auth.require_login()

SLOT = "teat"

teat_root = config.find_pipeline("teat")
ui.sidebar_status(teat_root, config.find_pipeline("udder"))
auth.sidebar_account()

ui.masthead(
    "Teat model · MobileNetV2",
    "Teat analysis",
    "A binary healthy/mastitis classifier with a Grad-CAM overlay, so you can "
    "check the model looked at the teat and not at a glove, a bucket, or the light.",
)

if teat_root is None:
    ui.missing_pipeline("teat", "teat_pipeline")
    st.stop()

paths = config.teat_paths(teat_root)
if not paths["model"].exists():
    st.error(f"Model file missing: {paths['model']}")
    st.stop()

# ------------------------------------------------------------------ controls
with st.sidebar:
    st.markdown("**Teat settings**")
    threshold = st.slider(
        "Decision cut-off", 0.05, 0.95, config.TEAT_THRESHOLD, 0.01,
        help="0.50 is the value evaluate.py reported against. Lower it to catch "
             "more mastitis at the cost of false alarms.",
    )
    show_cam = st.toggle("Grad-CAM overlay", value=True)
    alpha = st.slider("Overlay strength", 0.1, 0.9, 0.45, 0.05, disabled=not show_cam)

upload = st.file_uploader(
    "Teat image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="teat_uploader"
)
restored = state.is_restored(SLOT, upload)
img_bytes, img_name = state.remember_image(SLOT, upload)

if img_bytes is None:
    ui.note(
        "The training set carried a real shortcut: gloves appeared in mastitis "
        "images roughly ten times more often than in healthy ones, and the model "
        "learned it before <code>detect_hands_gloves.py</code> quarantined those "
        "frames. Keep hands out of shot and use the overlay to confirm the model "
        "is on the lesion.",
        kind="info",
    )
    st.stop()

if restored:
    held, clear = st.columns([4, 1])
    with held:
        st.caption(f"Showing **{img_name}** from earlier. Upload another to replace it.")
    with clear:
        if st.button("Clear", key="teat_clear"):
            state.forget(SLOT)
            st.rerun()

pil = Image.open(io.BytesIO(img_bytes))

# Alpha only repaints the overlay, so it stays out of the signature.
sig = state.signature(img_bytes, threshold, show_cam)
pred = state.cached_result(SLOT, sig)

if pred is None:
    try:
        with st.spinner("Scoring and computing gradients…"):
            pred = models.predict_teat(
                pil, str(paths["model"]), threshold, with_gradcam=show_cam
            )
        state.store_result(SLOT, sig, pred)
    except Exception as exc:      # noqa: BLE001
        st.error(f"Scoring failed: {exc}")
        st.caption("This page needs TensorFlow. Install the teat pipeline's requirements.txt.")
        st.stop()

# ------------------------------------------------------------------ result
ui.verdict(pred.label, pred.probability, pred.threshold)
ui.decision_strip(pred.probability, pred.threshold)
st.write("")

col_img, col_cam, col_meta = st.columns([1, 1, 0.85], gap="large")

with col_img:
    ui.caption("Submitted")
    st.image(pil, use_container_width=True)
    st.caption(img_name or "")

with col_cam:
    ui.caption("Grad-CAM")
    if pred.heatmap is not None:
        st.image(
            models.overlay_heatmap(models.teat_input(pil), pred.heatmap, alpha=alpha),
            use_container_width=True,
        )
        st.caption("Hot regions push the score toward mastitis.")
    else:
        st.info("Overlay is switched off.")

with col_meta:
    ui.caption("Detail")
    ui.stat("p(mastitis)", f"{pred.probability:.4f}")
    ui.stat("Cut-off", f"{threshold:.2f}")
    ui.stat("Margin", f"{abs(pred.probability - threshold):.3f}",
            "Under 0.08 is a coin-flip; get a second image.")
    ui.stat("Input", "224 × 224 RGB", "Rescaling happens inside the graph")

st.divider()

with st.expander("How to read the overlay"):
    st.markdown(
        """
**Good** — heat concentrated on the teat barrel or teat end, tracking visible
swelling, redness or a lesion.

**Suspect** — heat on a hand, glove, bucket, the cow's leg, a bright specular
highlight, or the image border. That is the model reading the photograph rather
than the animal, and the score should be discarded.

The overlay is computed by splitting the saved model into the MobileNetV2
backbone and the trained head, taking gradients of the sigmoid output with
respect to the last convolutional feature map, and weighting the channels by
their pooled gradient. It is the same routine `gradcam_visualize.py` runs over
the test split, so the overlays here and the ones in
`reports/gradcam/` are directly comparable.
"""
    )

if paths["gradcam_grid"].exists():
    with st.expander("Contact sheet from the last pipeline run"):
        st.image(str(paths["gradcam_grid"]), use_container_width=True)
