from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def train_logistic_score(
    data: pd.DataFrame,
    predictors: list[str],
    outcome: str,
    models_dir: Path,
    tables_dir: Path,
    *,
    random_state: int = 20260626,
    min_events: int = 10,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = data[predictors + [outcome]].dropna(subset=[outcome]).copy()
    df[outcome] = df[outcome].astype(int)
    event_counts = df[outcome].value_counts()
    if event_counts.min() < min_events:
        (tables_dir / "model_not_fit_reason.txt").write_text(
            f"Outcome {outcome} has insufficient class counts for a first-pass model: {event_counts.to_dict()}",
            encoding="utf-8",
        )
        return

    X = df[predictors]
    y = df[outcome]
    stratify = y if y.nunique() == 2 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, stratify=stratify, random_state=random_state
    )
    model = Pipeline(
        [
            ("preprocess", _preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "model": "logistic_interpretable",
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_valid)),
        "events_validation": int(y_valid.sum()),
        "auroc": roc_auc_score(y_valid, prob),
        "auprc": average_precision_score(y_valid, prob),
        "brier": brier_score_loss(y_valid, prob),
    }
    pd.DataFrame([metrics]).to_csv(tables_dir / "table_model_performance.csv", index=False)
    joblib.dump(model, models_dir / "logistic_interpretable.joblib")

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_.ravel()
    coef_table = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    nonzero = coef_table.loc[coef_table["coefficient"].abs() > 1e-8].copy()
    scale = np.nanmedian(nonzero["coefficient"].abs()) if not nonzero.empty else 1.0
    coef_table["points"] = np.round(coef_table["coefficient"] / scale).astype(int)
    coef_table.sort_values("coefficient", key=lambda s: s.abs(), ascending=False).to_csv(
        tables_dir / "table_integer_score.csv", index=False
    )
    (models_dir / "model_metadata.json").write_text(
        json.dumps({"predictors": predictors, "outcome": outcome, "scaling_factor": float(scale)}, indent=2),
        encoding="utf-8",
    )

