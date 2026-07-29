"""
Model loading and inference.

TensorFlow is imported lazily so the udder pages still run in an environment
that only has scikit-learn and OpenCV installed — the two pipelines were built
under separate virtualenvs and there is no reason to force them together here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from core import config
from core.udder_features import extract_features
from core.udder_segment import SegmentResult, grabcut_segment


# =============================================================== teat model
@dataclass
class TeatPrediction:
    probability: float          # p(mastitis)
    label: str
    threshold: float
    heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None


def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in str(v).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def keras_version_of(model_path: str) -> str | None:
    """
    A .keras file is a zip with a metadata.json naming the Keras that wrote it.
    Reading it costs nothing and turns an unreadable deserialisation traceback
    into a one-line version mismatch.
    """
    import json as _json
    import zipfile

    try:
        with zipfile.ZipFile(model_path) as z:
            return _json.loads(z.read("metadata.json")).get("keras_version")
    except Exception:      # noqa: BLE001
        return None


class TeatModelVersionError(RuntimeError):
    """Raised when the installed Keras is too old to read the saved model."""


@st.cache_resource(show_spinner="Loading the teat classifier…")
def load_teat_model(model_path: str):
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(model_path)
    except Exception as exc:      # noqa: BLE001
        saved = keras_version_of(model_path)
        try:
            import keras
            installed = keras.__version__
        except Exception:      # noqa: BLE001
            installed = getattr(tf.keras, "__version__", None)

        if saved and installed and _version_tuple(installed) < _version_tuple(saved):
            raise TeatModelVersionError(
                f"This model was saved with Keras {saved}, but Keras {installed} "
                f"is installed, so the saved layer settings cannot be read.\n\n"
                f"Fix it with:\n\n    pip install --upgrade \"keras>={saved}\"\n\n"
                f"then restart the app."
            ) from exc
        raise


def _teat_input(pil_img: Image.Image) -> np.ndarray:
    """224x224 RGB, values left at 0-255 — preprocess_input lives in the graph."""
    im = pil_img.convert("RGB").resize(config.TEAT_IMG_SIZE)
    return np.asarray(im).astype("float32")


@st.cache_resource(show_spinner=False)
def _teat_gradcam_parts(model_path: str):
    """
    Split the trained model into backbone -> head so gradients flow cleanly.

    The augmentation block is a Sequential and therefore also a Model, and it
    sits before the real backbone in model.layers, so pick the nested model
    with the most sublayers rather than the first one.
    """
    import tensorflow as tf

    model = load_teat_model(model_path)
    nested = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not nested:
        raise ValueError("No nested backbone found — is this the model train_teat_model.py saved?")
    backbone = max(nested, key=lambda l: len(l.layers))

    last_conv = None
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if shape is not None and len(shape) == 4:
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("No 4D convolutional output found in the backbone.")

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer(last_conv).output,
        name="feature_extractor",
    )

    head_layers = model.layers[model.layers.index(backbone) + 1:]
    feat_input = tf.keras.Input(shape=feature_extractor.output.shape[1:], name="feat_input")
    x = feat_input
    for layer in head_layers:
        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D,
                              tf.keras.layers.GlobalMaxPooling2D)):
            x = layer(x)
        elif isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)
        elif isinstance(layer, tf.keras.layers.Dense):
            x = layer(x)
    head_model = tf.keras.Model(inputs=feat_input, outputs=x, name="head_model")
    return feature_extractor, head_model, backbone.name, last_conv


def _gradcam(raw224: np.ndarray, model_path: str):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    feature_extractor, head_model, _, _ = _teat_gradcam_parts(model_path)
    batch = preprocess_input(np.expand_dims(raw224, 0).astype("float32"))
    conv_output = feature_extractor(batch)

    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        prediction = head_model(conv_output, training=False)
        target = prediction[:, 0]

    grads = tape.gradient(target, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_output[0] * pooled, axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy(), float(prediction.numpy()[0, 0])


def _overlay(raw224: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib
    resized = np.array(
        Image.fromarray((heatmap * 255).astype("uint8")).resize(
            config.TEAT_IMG_SIZE, Image.BILINEAR)
    )
    try:
        jet = matplotlib.colormaps["jet"]
    except AttributeError:            # matplotlib < 3.9
        import matplotlib.cm as cm
        jet = cm.get_cmap("jet")
    colours = (jet(np.arange(256))[:, :3] * 255).astype("uint8")
    return (raw224.astype("uint8") * (1 - alpha) + colours[resized] * alpha).astype("uint8")


def predict_teat(pil_img: Image.Image, model_path: str,
                 threshold: float = config.TEAT_THRESHOLD,
                 with_gradcam: bool = True) -> TeatPrediction:
    raw224 = _teat_input(pil_img)

    if with_gradcam:
        heatmap, prob = _gradcam(raw224, model_path)
        overlay = _overlay(raw224, heatmap)
    else:
        model = load_teat_model(model_path)
        prob = float(model.predict(np.expand_dims(raw224, 0), verbose=0).flatten()[0])
        heatmap, overlay = None, None

    label = "mastitis" if prob >= threshold else "healthy"
    return TeatPrediction(prob, label, threshold, heatmap, overlay)


# ============================================================== udder model
@dataclass
class UdderPrediction:
    probability: float
    label: str
    threshold: float
    features: np.ndarray
    segmentation: SegmentResult | None = None
    used_segmentation: bool = False
    top_features: list = field(default_factory=list)


@st.cache_resource(show_spinner="Loading the udder classifier…")
def load_udder_model(model_path: str):
    import joblib
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_udder_threshold(threshold_path: str) -> float:
    try:
        return float(json.loads(Path(threshold_path).read_text())["threshold"])
    except Exception:
        return config.UDDER_FALLBACK_THRESHOLD


def predict_udder(bgr: np.ndarray, model_path: str, threshold: float,
                  use_segmentation: bool = False) -> UdderPrediction:
    """
    bgr: OpenCV-order image.

    use_segmentation defaults to False on purpose. The shipped SVM was fitted
    on features from data/raw, not from data/processed/segmented, so running
    GrabCut first puts the model off-distribution. Turn it on only to see what
    a segmentation-fed model would receive.
    """
    seg = grabcut_segment(bgr) if use_segmentation else None
    source = seg.image if (seg is not None and seg.ok) else bgr

    vec = extract_features(source)
    if vec.shape[0] != config.UDDER_FEATURE_DIM:
        raise ValueError(
            f"Feature vector is {vec.shape[0]}-dim, the model expects "
            f"{config.UDDER_FEATURE_DIM}. Check your scikit-image version."
        )

    pipe = load_udder_model(model_path)
    prob = float(pipe.predict_proba(vec.reshape(1, -1))[0, 1])
    label = "mastitis" if prob >= threshold else "healthy"

    return UdderPrediction(
        probability=prob,
        label=label,
        threshold=threshold,
        features=vec,
        segmentation=seg,
        used_segmentation=bool(seg is not None and seg.ok),
    )


# ================================================================= helpers
teat_input = _teat_input          # 224x224 raw RGB array, for the overlay
overlay_heatmap = _overlay        # (raw224, heatmap, alpha) -> uint8 RGB


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, ::-1].copy()
