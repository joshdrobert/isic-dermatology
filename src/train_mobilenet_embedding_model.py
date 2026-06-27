from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 20260626


def make_backbone():
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    return base


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def embeddings(status_csv: Path, cohort_csv: Path, output_npz: Path, batch_size: int = 64) -> None:
    status = pd.read_csv(status_csv)
    cohort = pd.read_csv(cohort_csv)
    df = cohort.merge(status.loc[status["downloaded"].eq(True), ["isic_id", "path"]], on="isic_id", how="inner")
    model = make_backbone()
    vectors, ids, labels = [], [], []
    batch, batch_ids, batch_labels = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=output_npz.name):
        try:
            batch.append(load_image(row["path"]))
            batch_ids.append(row["isic_id"])
            batch_labels.append(int(row["outcome_malignant_or_high_risk"]))
        except Exception:
            continue
        if len(batch) >= batch_size:
            z = model.predict(np.stack(batch), verbose=0)
            vectors.append(z)
            ids.extend(batch_ids)
            labels.extend(batch_labels)
            batch, batch_ids, batch_labels = [], [], []
    if batch:
        z = model.predict(np.stack(batch), verbose=0)
        vectors.append(z)
        ids.extend(batch_ids)
        labels.extend(batch_labels)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, X=np.vstack(vectors), isic_id=np.array(ids), y=np.array(labels))


def evaluate(train_npz: Path, external_npzs: dict[str, Path], output_csv: Path) -> None:
    train = np.load(train_npz, allow_pickle=True)
    X = train["X"]
    y = train["y"].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2)
    clf.fit(X_train, y_train)
    rows = []
    for name, X_eval, y_eval in [("internal", X_valid, y_valid)]:
        prob = clf.predict_proba(X_eval)[:, 1]
        rows.append(
            {
                "model": f"mobilenetv2_embedding_{name}",
                "n_validation": int(len(y_eval)),
                "events_validation": int(y_eval.sum()),
                "auroc": roc_auc_score(y_eval, prob),
                "auprc": average_precision_score(y_eval, prob),
                "brier": brier_score_loss(y_eval, prob),
            }
        )
    train_ids = set(train["isic_id"].astype(str))
    for name, path in external_npzs.items():
        if not path.exists():
            continue
        ext = np.load(path, allow_pickle=True)
        mask = np.array([str(i) not in train_ids for i in ext["isic_id"]])
        X_ext = ext["X"][mask]
        y_ext = ext["y"].astype(int)[mask]
        if len(y_ext) and len(set(y_ext)) == 2:
            prob = clf.predict_proba(X_ext)[:, 1]
            rows.append(
                {
                    "model": f"mobilenetv2_embedding_external_{name}",
                    "n_validation": int(len(y_ext)),
                    "events_validation": int(y_ext.sum()),
                    "auroc": roc_auc_score(y_ext, prob),
                    "auprc": average_precision_score(y_ext, prob),
                    "brier": brier_score_loss(y_ext, prob),
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    joblib.dump(clf, PROJECT_ROOT / "models" / "mobilenetv2_embedding_logistic.joblib")


def main() -> None:
    configs = {
        "collection_66": (
            PROJECT_ROOT / "tables" / "table_isic_thumbnail_download_status.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_66_metadata_cohort.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_66_mobilenetv2_embeddings.npz",
        ),
        "collection_73": (
            PROJECT_ROOT / "tables" / "table_isic_collection_73_thumbnail_download_status.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_73_metadata_cohort.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_73_mobilenetv2_embeddings.npz",
        ),
        "collection_67": (
            PROJECT_ROOT / "tables" / "table_isic_collection_67_thumbnail_download_status.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_67_metadata_cohort.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_67_mobilenetv2_embeddings.npz",
        ),
    }
    for _, (status, cohort, out) in configs.items():
        if status.exists() and cohort.exists() and not out.exists():
            embeddings(status, cohort, out)
    evaluate(
        configs["collection_66"][2],
        {"collection_73": configs["collection_73"][2], "collection_67": configs["collection_67"][2]},
        PROJECT_ROOT / "tables" / "table_mobilenetv2_embedding_performance.csv",
    )
    print("Wrote MobileNetV2 transfer-learning embedding performance")


if __name__ == "__main__":
    main()

