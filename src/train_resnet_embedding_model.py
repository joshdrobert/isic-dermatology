from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 20260626


def load_backbone(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    return model, transform


def embeddings_for(status_csv: Path, cohort_csv: Path, output_npz: Path, batch_size: int = 128) -> pd.DataFrame:
    status = pd.read_csv(status_csv)
    cohort = pd.read_csv(cohort_csv)
    df = cohort.merge(status.loc[status["downloaded"].eq(True), ["isic_id", "path"]], on="isic_id", how="inner")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_backbone(device)
    vectors = []
    ids = []
    labels = []
    batch = []
    batch_ids = []
    batch_labels = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"embeddings {output_npz.name}"):
            try:
                img = Image.open(row["path"]).convert("RGB")
                batch.append(transform(img))
                batch_ids.append(row["isic_id"])
                batch_labels.append(int(row["outcome_malignant_or_high_risk"]))
            except Exception:
                continue
            if len(batch) >= batch_size:
                x = torch.stack(batch).to(device)
                z = model(x).cpu().numpy()
                vectors.append(z)
                ids.extend(batch_ids)
                labels.extend(batch_labels)
                batch, batch_ids, batch_labels = [], [], []
        if batch:
            x = torch.stack(batch).to(device)
            z = model(x).cpu().numpy()
            vectors.append(z)
            ids.extend(batch_ids)
            labels.extend(batch_labels)
    X = np.vstack(vectors)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, X=X, isic_id=np.array(ids), y=np.array(labels))
    return pd.DataFrame({"isic_id": ids, "outcome_malignant_or_high_risk": labels})


def train_internal_external(train_npz: Path, external_npz: Path | None, table_csv: Path) -> None:
    train = np.load(train_npz, allow_pickle=True)
    X = train["X"]
    y = train["y"].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_valid)[:, 1]
    rows = [
        {
            "model": "resnet18_thumbnail_embedding_internal",
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_valid)),
            "events_validation": int(y_valid.sum()),
            "auroc": roc_auc_score(y_valid, prob),
            "auprc": average_precision_score(y_valid, prob),
            "brier": brier_score_loss(y_valid, prob),
        }
    ]
    if external_npz and external_npz.exists():
        ext = np.load(external_npz, allow_pickle=True)
        train_ids = set(train["isic_id"].astype(str))
        mask = np.array([str(i) not in train_ids for i in ext["isic_id"]])
        X_ext = ext["X"][mask]
        y_ext = ext["y"].astype(int)[mask]
        if len(y_ext) and len(set(y_ext)) == 2:
            p_ext = clf.predict_proba(X_ext)[:, 1]
            rows.append(
                {
                    "model": "resnet18_thumbnail_embedding_external_collection_470",
                    "n_train": int(len(X)),
                    "n_validation": int(len(X_ext)),
                    "events_validation": int(y_ext.sum()),
                    "auroc": roc_auc_score(y_ext, p_ext),
                    "auprc": average_precision_score(y_ext, p_ext),
                    "brier": brier_score_loss(y_ext, p_ext),
                }
            )
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_csv, index=False)
    joblib.dump(clf, PROJECT_ROOT / "models" / "resnet18_embedding_logistic.joblib")


def main() -> None:
    train_npz = PROJECT_ROOT / "data_processed" / "isic_collection_66_resnet18_embeddings.npz"
    external_npz = PROJECT_ROOT / "data_processed" / "isic_collection_470_resnet18_embeddings.npz"
    if not train_npz.exists():
        embeddings_for(
            PROJECT_ROOT / "tables" / "table_isic_thumbnail_download_status.csv",
            PROJECT_ROOT / "data_processed" / "isic_collection_66_metadata_cohort.csv",
            train_npz,
        )
    external_status = PROJECT_ROOT / "tables" / "table_isic_collection_470_thumbnail_download_status.csv"
    external_cohort = PROJECT_ROOT / "data_processed" / "isic_collection_470_metadata_cohort.csv"
    if external_status.exists() and external_cohort.exists() and not external_npz.exists():
        embeddings_for(external_status, external_cohort, external_npz)
    train_internal_external(train_npz, external_npz, PROJECT_ROOT / "tables" / "table_resnet18_embedding_performance.csv")
    print("Wrote ResNet18 embedding model performance")


if __name__ == "__main__":
    main()

