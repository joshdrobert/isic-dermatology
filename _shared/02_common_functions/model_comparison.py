from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 20260626


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def compare_feature_sets(
    data: pd.DataFrame,
    outcome: str,
    feature_sets: dict[str, list[str]],
    output_csv: Path,
) -> pd.DataFrame:
    y = data[outcome].astype(int)
    train_idx, valid_idx = train_test_split(data.index, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    rows = []
    for name, predictors in feature_sets.items():
        predictors = [p for p in predictors if p in data.columns]
        X = data[predictors]
        model = Pipeline(
            [
                ("preprocess", preprocessor(X.loc[train_idx])),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        model.fit(X.loc[train_idx], y.loc[train_idx])
        prob = model.predict_proba(X.loc[valid_idx])[:, 1]
        rows.append(
            {
                "model": name,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(valid_idx)),
                "events_validation": int(y.loc[valid_idx].sum()),
                "auroc": roc_auc_score(y.loc[valid_idx], prob),
                "auprc": average_precision_score(y.loc[valid_idx], prob),
                "brier": brier_score_loss(y.loc[valid_idx], prob),
                "predictor_count": len(predictors),
            }
        )
    out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out

