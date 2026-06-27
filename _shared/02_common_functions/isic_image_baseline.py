from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 20260626


def download_one(row: pd.Series, image_dir: Path) -> dict:
    image_dir.mkdir(parents=True, exist_ok=True)
    dest = image_dir / f"{row['isic_id']}.jpg"
    error = ""
    if not dest.exists():
        try:
            request = urllib.request.Request(
                row["files.thumbnail_256.url"],
                headers={"User-Agent": "OpenSpecialtyRiskAtlas/0.1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
                handle.write(response.read())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            error = str(exc)
    return {
        "isic_id": row["isic_id"],
        "path": str(dest),
        "downloaded": dest.exists(),
        "bytes": dest.stat().st_size if dest.exists() else 0,
        "error": error,
    }


def download_thumbnails(metadata_csv: Path, image_dir: Path, status_csv: Path, workers: int = 16) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)
    status = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, row, image_dir) for _, row in metadata.iterrows()]
        for i, future in enumerate(as_completed(futures), start=1):
            status.append(future.result())
            if i % 500 == 0:
                print(f"Downloaded/checked {i}/{len(metadata)} ISIC thumbnails")
    out = pd.DataFrame(status)
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(status_csv, index=False)
    return out


def image_features(path: Path) -> dict:
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype(np.float32) / 255.0
    hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    feats = {
        "rgb_mean_r": float(arr[:, :, 0].mean()),
        "rgb_mean_g": float(arr[:, :, 1].mean()),
        "rgb_mean_b": float(arr[:, :, 2].mean()),
        "rgb_sd_r": float(arr[:, :, 0].std()),
        "rgb_sd_g": float(arr[:, :, 1].std()),
        "rgb_sd_b": float(arr[:, :, 2].std()),
        "hsv_mean_h": float(hsv[:, :, 0].mean()),
        "hsv_mean_s": float(hsv[:, :, 1].mean()),
        "hsv_mean_v": float(hsv[:, :, 2].mean()),
        "edge_fraction": float((edges > 0).mean()),
    }
    for channel, name in [(arr[:, :, 0], "r"), (arr[:, :, 1], "g"), (arr[:, :, 2], "b")]:
        hist, _ = np.histogram(channel, bins=8, range=(0, 1), density=True)
        for i, value in enumerate(hist):
            feats[f"hist_{name}_{i}"] = float(value)
    return feats


def build_feature_table(status_csv: Path, cohort_csv: Path, output_csv: Path) -> pd.DataFrame:
    status = pd.read_csv(status_csv)
    cohort = pd.read_csv(cohort_csv)
    rows = []
    for _, row in status.loc[status["downloaded"].eq(True)].iterrows():
        try:
            feats = image_features(Path(row["path"]))
        except Exception:
            continue
        feats["isic_id"] = row["isic_id"]
        rows.append(feats)
    features = pd.DataFrame(rows)
    merged = cohort.merge(features, on="isic_id", how="inner")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def train_image_feature_model(feature_csv: Path, models_dir: Path, tables_dir: Path) -> None:
    data = pd.read_csv(feature_csv)
    feature_cols = [c for c in data.columns if c.startswith(("rgb_", "hsv_", "edge_", "hist_"))]
    X = data[feature_cols]
    y = data["outcome_malignant_or_high_risk"].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "model": "thumbnail_image_features_hist_gradient_boosting",
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_valid)),
        "events_validation": int(y_valid.sum()),
        "auroc": roc_auc_score(y_valid, prob),
        "auprc": average_precision_score(y_valid, prob),
        "brier": brier_score_loss(y_valid, prob),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "thumbnail_image_feature_model.joblib")
    pd.DataFrame([metrics]).to_csv(tables_dir / "table_image_model_performance.csv", index=False)

