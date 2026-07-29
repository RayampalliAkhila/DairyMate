"""Metrics as the pipelines measured them, read straight from reports/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from core import auth, config, theme, ui

theme.inject("Model reports · Dairy Mate")
auth.require_login()

teat_root = config.find_pipeline("teat")
udder_root = config.find_pipeline("udder")
ui.sidebar_status(teat_root, udder_root)
auth.sidebar_account()

ui.masthead(
    "Reports",
    "Measured performance",
    "Nothing on this page is recomputed. It is read from the report files each "
    "pipeline wrote on its last run.",
)

tab_teat, tab_udder, tab_diff = st.tabs(["Teat", "Udder", "Side by side"])

# ------------------------------------------------------------------- teat
with tab_teat:
    if teat_root is None:
        ui.missing_pipeline("teat", "teat_pipeline")
    else:
        p = config.teat_paths(teat_root)
        if p["evaluation"].exists():
            rep = json.loads(p["evaluation"].read_text())
            cr = rep["classification_report"]
            a, b, c, d = st.columns(4)
            with a:
                ui.stat("Accuracy", f"{cr['accuracy']:.3f}")
            with b:
                ui.stat("Macro F1", f"{cr['macro avg']['f1-score']:.3f}")
            with c:
                ui.stat("Healthy recall", f"{cr['healthy']['recall']:.3f}")
            with d:
                ui.stat("Mastitis recall", f"{cr['mastitis']['recall']:.3f}")

            per_class = pd.DataFrame({
                k: v for k, v in cr.items() if k in rep["class_names"]
            }).T.round(3)
            st.dataframe(per_class, use_container_width=True)

            cm = rep["confusion_matrix"]
            st.markdown("**Confusion matrix** — rows are truth, columns are prediction")
            st.dataframe(
                pd.DataFrame(cm, index=rep["class_names"], columns=rep["class_names"]),
                use_container_width=True,
            )
        else:
            st.info("No evaluation report found. Run `scripts/evaluate.py`.")

        if p["confusion_png"].exists():
            st.image(str(p["confusion_png"]), width=460)

        if p["history"].exists():
            hist = json.loads(p["history"].read_text())
            with st.expander("Training curves"):
                for stage in ("head", "finetune"):
                    if stage in hist:
                        st.markdown(f"**{stage}**")
                        st.line_chart(pd.DataFrame(hist[stage]))

        if p["hand_report"].exists():
            with st.expander("Glove / hand quarantine log"):
                hd = pd.read_csv(p["hand_report"])
                st.caption(
                    "Mastitis images contained gloves far more often than healthy ones, "
                    "so the classifier had a shortcut available. These rows are what "
                    "the detector pulled out before training."
                )
                st.dataframe(hd.head(300), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ udder
with tab_udder:
    if udder_root is None:
        ui.missing_pipeline("udder", "udder_pipeline")
    else:
        p = config.udder_paths(udder_root)
        if p["test_metrics"].exists():
            st.code(p["test_metrics"].read_text(), language="text")
        if p["model_comparison"].exists():
            st.markdown("**Model comparison** — 3-fold group CV on 66 training images")
            st.dataframe(pd.read_csv(p["model_comparison"]), use_container_width=True,
                         hide_index=True)
        if p["tuning_log"].exists():
            with st.expander("Hyperparameter sweep"):
                st.dataframe(pd.read_csv(p["tuning_log"]), use_container_width=True,
                             hide_index=True)
        if p["error_analysis"].exists():
            with st.expander("Error analysis"):
                st.code(p["error_analysis"].read_text(), language="text")

        seg_dir = p["segmented"]
        if seg_dir.exists():
            counts = {
                d.name: len(list(d.glob("*")))
                for d in sorted(seg_dir.iterdir()) if d.is_dir()
            }
            ui.note(
                "Segmentation artefacts exist on disk — "
                + ", ".join(f"<b>{k}</b>: {v} images" for k, v in counts.items())
                + " — but the split manifests in <code>data/splits/</code> reference "
                "<code>data/raw/</code>, so these files were never read during training.",
                kind="warn",
            )

# ------------------------------------------------------------ side by side
with tab_diff:
    st.markdown("### The two pipelines are not comparable, and that is the point")
    st.markdown(
        """
| | Teat | Udder |
|---|---|---|
| Approach | End-to-end CNN, MobileNetV2 transfer learning | Handcrafted features + SVM |
| Input | 224 × 224 RGB, augmented | 128 × 128, 223-dim vector |
| Labelled images | ~1,440 | 95 |
| Class imbalance | 28.5 : 1 | 1.36 : 1 |
| Held-out test size | 216 | 16 |
| Headline metric | 0.972 accuracy | 0.905 ROC-AUC / 0.722 balanced accuracy |
| Segmentation | Not implemented | Implemented, not connected to training |
| Explainability | Grad-CAM | Per-feature-block deviation |
| Main risk | Shortcut features (gloves), subclinical blind spot | Too little labelled data to calibrate a cut-off |

The teat number is high because the dataset is fifteen times larger, not
because the architecture is better. Judge the udder model on ROC-AUC, which
says its ranking is sound, rather than on accuracy at a cut-off fitted to
thirteen validation images.
"""
    )

    st.markdown("### Next steps carried over from both READMEs")
    st.markdown(
        """
1. **Label the ~328 unlabelled udder images.** Roughly a 4x increase in usable
   data, and the single change most likely to move the udder numbers.
2. **Wire segmentation into the udder training path**, or drop it. Right now
   `segment.py` produces output nothing reads. Re-pointing the split manifests
   at `data/processed/segmented/` and re-running `train_final.py` gives the
   first honest read on whether GrabCut helps at all.
3. **Add segmentation to the teat pipeline**, measured against the 0.972
   baseline that is already recorded.
4. **Source subclinical cases.** Neither model can see what neither dataset
   contains.
"""
    )
