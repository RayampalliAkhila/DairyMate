"""
223-dim feature vector — a byte-for-byte port of udder_pipeline/src/features.py.

Do not "improve" anything in here. The saved StandardScaler was fitted on
vectors produced by this exact sequence; changing a bin count, a resize order
or a colour conversion silently shifts every feature and the SVM scores
become meaningless rather than merely wrong.

Input is BGR (OpenCV order), as it was during training.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

IMG_SIZE = 128


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = apply_clahe(gray)

    feats: list[float] = []

    # 48 — HSV histograms, 16 bins per channel
    for ch in range(3):
        h_ = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        h_ = cv2.normalize(h_, h_).flatten()
        feats.extend(h_)

    # 12 — per-channel mean/std in BGR and HSV
    for img_space in (img, hsv):
        for ch in range(3):
            channel = img_space[:, :, ch].astype(np.float32)
            feats.append(channel.mean())
            feats.append(channel.std())

    # 10 — uniform LBP histogram
    lbp = local_binary_pattern(gray_eq, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    feats.extend(lbp_hist)

    # 8 — GLCM properties, mean and std over 2 distances x 4 angles
    gray_q = (gray_eq / 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop)
        feats.append(vals.mean())
        feats.append(vals.std())

    # 1 — Canny edge density
    edges = cv2.Canny(gray_eq, 100, 200)
    feats.append(edges.mean() / 255.0)

    # 144 — HOG
    hog_feats = hog(
        gray_eq,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(1, 1),
        feature_vector=True,
    )
    feats.extend(hog_feats)

    return np.array(feats, dtype=np.float32)


FEATURE_BLOCKS = [
    ("HSV histograms", 48, "Colour distribution — redness and saturation shifts."),
    ("Channel statistics", 12, "Mean and spread per BGR/HSV channel."),
    ("LBP texture", 10, "Local micro-texture; skin roughness and scabbing."),
    ("GLCM texture", 8, "Contrast, homogeneity, energy, correlation."),
    ("Edge density", 1, "Canny edge fraction — lesion boundaries."),
    ("HOG shape", 144, "Gradient orientation over a 4x4 cell grid."),
]
