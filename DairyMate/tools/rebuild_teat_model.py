"""
rebuild_teat_model.py

Rebuilds the teat classifier from scratch and loads the trained weights.

Why this exists: `teat_classifier.keras` was saved by Keras 3.15.0, and older
Keras cannot read its layer configuration — the saved settings mention
arguments that did not exist yet. Weights are far more portable than
configuration, so this rebuilds the architecture exactly as
`train_teat_model.py` defined it and pours the trained weights into it.

The rebuilt model is then written back out in your own Keras version's format,
so the app loads it normally afterwards.

Usage:
    python rebuild_teat_model.py \
        --weights teat_classifier.weights.h5 \
        --out     teat_classifier_rebuilt.keras

Then point the app at the rebuilt file (rename it over the original, or set
the model path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)


def build_augmentation():
    """Identical to train_teat_model.py. Carries no weights, but it occupies a
    slot in the saved weight tree, so it has to be here for the paths to line up."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="augmentation")


def build_model():
    # weights=None, not "imagenet" — the trained weights are about to be loaded
    # over the top, and this avoids a needless 14 MB download.
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = build_augmentation()(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="teat_classifier.weights.h5")
    ap.add_argument("--out", default="teat_classifier_rebuilt.keras")
    ap.add_argument("--check_dir", default=None,
                    help="Optional data/processed/test folder to verify accuracy against.")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"Weights file not found: {weights}")

    import keras
    print(f"Keras {keras.__version__} / TensorFlow {tf.__version__}")

    print("Building architecture…")
    model = build_model()

    print(f"Loading weights from {weights}…")
    model.load_weights(str(weights))

    model.save(args.out)
    print(f"Saved -> {args.out}")

    # ---- verification -----------------------------------------------------
    import numpy as np
    from PIL import Image

    if args.check_dir:
        root = Path(args.check_dir)
        y_true, y_prob = [], []
        for label, cls in ((0, "healthy"), (1, "mastitis")):
            files = sorted((root / cls).glob("*.jpg"))
            if not files:
                continue
            batch = np.stack([
                np.asarray(Image.open(f).convert("RGB").resize(IMG_SIZE)).astype("float32")
                for f in files
            ])
            probs = model.predict(batch, verbose=0).flatten()
            y_prob.extend(probs)
            y_true.extend([label] * len(files))

        if y_true:
            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            y_pred = (y_prob >= 0.5).astype(int)
            print(f"\nTest accuracy: {(y_pred == y_true).mean():.4f}   (expected ~0.97)")
            for label, cls in ((0, "healthy"), (1, "mastitis")):
                m = y_true == label
                if m.any():
                    print(f"  {cls:9s} recall {(y_pred[m] == label).mean():.4f}  "
                          f"mean p={y_prob[m].mean():.3f}")
    else:
        probe = np.random.rand(1, *IMG_SIZE, 3).astype("float32") * 255
        print(f"\nSmoke test on random noise: p={float(model.predict(probe, verbose=0)[0, 0]):.4f}")
        print("Re-run with --check_dir to verify accuracy against your test split.")


if __name__ == "__main__":
    main()
