from __future__ import annotations

import json
import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
sys.path.append(str(REPO_ROOT / "_shared" / "02_common_functions"))

from isic_image_baseline import build_feature_table, download_thumbnails  # noqa: E402
from isic_metadata import build_isic_analysis, write_isic_summary  # noqa: E402


def fetch_collection_metadata(collection_id: int, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    next_url = "https://api.isic-archive.com/api/v2/images/search/?" + urllib.parse.urlencode(
        {"collections": collection_id, "limit": 100}
    )
    while next_url:
        req = urllib.request.Request(next_url, headers={"Accept": "application/json", "User-Agent": "OpenSpecialtyRiskAtlas/0.1"})
        with urllib.request.urlopen(req, timeout=60) as response:
            page = json.loads(response.read().decode("utf-8"))
        rows.extend(page.get("results", []))
        print(f"ISIC external collection {collection_id}: {len(rows)} records")
        next_url = page.get("next")
        time.sleep(0.05)
    raw = dest_dir / f"collection_{collection_id}_metadata_raw.json"
    raw.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat = {
            "isic_id": row.get("isic_id"),
            "copyright_license": row.get("copyright_license"),
            "attribution": row.get("attribution"),
            "public": row.get("public"),
        }
        for group, values in row.get("files", {}).items():
            for key, value in values.items():
                flat[f"files.{group}.{key}"] = value
        for group, values in row.get("metadata", {}).items():
            for key, value in values.items():
                flat[f"metadata.{group}.{key}"] = value
        flat_rows.append(flat)
    csv_path = dest_dir / f"collection_{collection_id}_metadata.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    return csv_path


def train_and_external_eval(train_csv: Path, external_csv: Path, output_csv: Path, model_path: Path) -> None:
    train = pd.read_csv(train_csv)
    external = pd.read_csv(external_csv)
    overlap = set(train["isic_id"]).intersection(set(external["isic_id"]))
    external = external.loc[~external["isic_id"].isin(overlap)].copy()
    metadata = ["age", "sex", "anatom_site", "melanocytic", "image_megapixels", "file_size"]
    image = [c for c in train.columns if c.startswith(("rgb_", "hsv_", "edge_", "hist_"))]
    predictors = [p for p in metadata + image if p in train.columns and p in external.columns]
    X_train = train[predictors]
    y_train = train["outcome_malignant_or_high_risk"].astype(int)
    X_external = external[predictors]
    y_external = external["outcome_malignant_or_high_risk"].astype(int)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), X_train.select_dtypes(include=["number", "bool"]).columns.tolist()),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), [c for c in X_train.columns if c not in X_train.select_dtypes(include=["number", "bool"]).columns]),
        ]
    )
    model = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_external)[:, 1]
    metrics = {
        "model": "combined_metadata_thumbnail_features_external_validation",
        "train_n": int(len(train)),
        "external_n": int(len(external)),
        "external_events": int(y_external.sum()),
        "overlap_removed": int(len(overlap)),
        "auroc": roc_auc_score(y_external, prob) if y_external.nunique() == 2 else np.nan,
        "auprc": average_precision_score(y_external, prob) if y_external.nunique() == 2 else np.nan,
        "brier": brier_score_loss(y_external, prob) if y_external.nunique() == 2 else np.nan,
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)
    joblib.dump(model, model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-id", type=int, default=73)
    args = parser.parse_args()
    collection_id = args.collection_id
    external_dir = PROJECT_ROOT / "data_raw" / f"isic_collection_{collection_id}"
    metadata_csv = external_dir / f"collection_{collection_id}_metadata.csv"
    if not metadata_csv.exists():
        metadata_csv = fetch_collection_metadata(collection_id, external_dir)
    external_cohort = PROJECT_ROOT / "data_processed" / f"isic_collection_{collection_id}_metadata_cohort.csv"
    analysis = build_isic_analysis(metadata_csv)
    analysis.to_csv(external_cohort, index=False)
    write_isic_summary(analysis, PROJECT_ROOT / "tables" / f"external_collection_{collection_id}")
    status_csv = PROJECT_ROOT / "tables" / f"table_isic_collection_{collection_id}_thumbnail_download_status.csv"
    thumb_dir = external_dir / "thumbnails_256"
    feature_csv = PROJECT_ROOT / "data_processed" / f"isic_collection_{collection_id}_thumbnail_image_feature_cohort.csv"
    download_thumbnails(metadata_csv, thumb_dir, status_csv)
    build_feature_table(status_csv, external_cohort, feature_csv)
    train_and_external_eval(
        PROJECT_ROOT / "data_processed" / "isic_thumbnail_image_feature_cohort.csv",
        feature_csv,
        PROJECT_ROOT / "tables" / f"table_external_validation_isic_collection_{collection_id}.csv",
        PROJECT_ROOT / "models" / f"combined_external_validation_collection_{collection_id}_model.joblib",
    )
    print(f"Wrote ISIC external validation against collection {collection_id}")


if __name__ == "__main__":
    main()
