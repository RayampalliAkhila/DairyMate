"""
Dairy Mate — mastitis screening console.

Landing page: what the two models are and how they are built. Scoring lives on
the Teat analysis and Udder analysis pages.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from core import auth, config, theme, ui

theme.inject("Dairy Mate")
auth.require_login()

teat_root = config.find_pipeline("teat")
udder_root = config.find_pipeline("udder")
ui.sidebar_status(teat_root, udder_root)
auth.sidebar_account()

ui.masthead(
    "Dairy Mate · screening console",
    "Two models, two body regions, one workflow",
    "A convolutional classifier for teat images and a feature-based classifier "
    "for udder images. This page is what they are; the analysis pages are where "
    "you use them.",
)

# ------------------------------------------------------------------ summary
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        ui.caption("Teat model")
        ui.stat("Architecture", "MobileNetV2", "Transfer learning, two-stage fine-tune")
        ui.stat("Test accuracy", "0.972", "216 held-out images")
        ui.stat("Segmentation", "Not integrated", "Classification only")
with c2:
    with st.container(border=True):
        ui.caption("Udder model")
        ui.stat("Architecture", "SVM (RBF)", "223 handcrafted features")
        ui.stat("Test ROC-AUC", "0.905", "16 held-out images")
        ui.stat("Segmentation", "Present, not wired in", "GrabCut runs, model reads raw")
with c3:
    with st.container(border=True):
        ui.caption("Read before use")
        ui.note(
            "Neither model detects <b>subclinical</b> mastitis — both were trained on "
            "visually obvious cases. A clean score is not a clean quarter.",
            kind="warn",
        )

st.write("")
st.divider()

# ------------------------------------------------------------- architecture
st.markdown("### Architecture")
st.markdown(
    '<div class="dm-sub">The two pipelines solve the same problem from opposite '
    "directions. One learns its own features from a large image set; the other is "
    "told which features to measure because its image set is too small to learn "
    "from.</div>",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        ui.caption("Teat · end-to-end CNN")
        st.markdown(
            """
**Graph**

```
input 224×224×3 RGB (0–255)
  └─ augmentation      flip · rotate · zoom · contrast · brightness
  └─ preprocess_input  scaled to [-1,1] inside the graph
  └─ MobileNetV2       ImageNet weights, include_top=False
  └─ GlobalAveragePooling2D
  └─ Dropout 0.3
  └─ Dense 1, sigmoid  →  p(mastitis)
```

**Training** — two stages. The backbone is frozen while the head learns, then
the top layers unfreeze at a low learning rate. Class weights carry a 28.5 : 1
imbalance.

**Preprocessing note** — rescaling happens *inside* the graph, so the model is
fed raw 0–255 pixels. Normalising before `predict` would silently halve accuracy.

**Explainability** — Grad-CAM on the last convolutional feature map.
"""
        )

with right:
    with st.container(border=True):
        ui.caption("Udder · features + SVM")
        st.markdown(
            """
**Pipeline**

```
input image (BGR)
  └─ resize 128×128 · CLAHE on the grey channel
  └─ feature extraction → 223 dims
       HSV histograms      48
       channel statistics  12
       LBP texture         10
       GLCM texture         8
       edge density         1
       HOG shape          144
  └─ StandardScaler
  └─ SVC, RBF kernel, class_weight balanced
  └─ predict_proba → p(mastitis), cut-off 0.60
```

**Selection** — 3-fold *group* cross-validation so near-duplicate frames stay in
the same fold. SVM beat RandomForest and GradientBoosting; on 66 training rows
the stronger regularisation generalises where ensembles memorise.

**Segmentation** — GrabCut exists and runs, but the split manifests point at
`data/raw/`, so the trained model never saw its output.
"""
        )

st.write("")
ui.note(
    "The teat number is higher because its dataset is roughly fifteen times "
    "larger, not because the architecture is better. Judge the udder model on "
    "ROC-AUC, which says its ranking is sound, rather than on accuracy at a "
    "cut-off fitted to thirteen validation images.",
    kind="info",
)

st.divider()
st.markdown(
    '<div class="dm-kv">Pages · <b>Teat analysis</b> — score an image, Grad-CAM, '
    "shortcut checks · <b>Udder analysis</b> — segmentation view and feature "
    "breakdown · <b>Batch run</b> — a folder at a time · <b>Model reports</b> — "
    "metrics as they were measured</div>",
    unsafe_allow_html=True,
)
