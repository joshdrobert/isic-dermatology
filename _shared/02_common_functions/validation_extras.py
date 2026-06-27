from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def train_eval(data: pd.DataFrame, predictors: list[str], outcome: str, model_name: str) -> tuple[dict, pd.DataFrame]:
    predictors = [p for p in predictors if p in data.columns]
    df = data[predictors + [outcome]].dropna(subset=[outcome]).copy()
    y = df[outcome].astype(int)
    train_idx, valid_idx = train_test_split(df.index, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X = df[predictors]
    model = Pipeline([("preprocess", preprocessor(X.loc[train_idx])), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    model.fit(X.loc[train_idx], y.loc[train_idx])
    prob = model.predict_proba(X.loc[valid_idx])[:, 1]
    metrics = {
        "model": model_name,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(valid_idx)),
        "events_validation": int(y.loc[valid_idx].sum()),
        "auroc": roc_auc_score(y.loc[valid_idx], prob),
        "auprc": average_precision_score(y.loc[valid_idx], prob),
        "brier": brier_score_loss(y.loc[valid_idx], prob),
        "predictor_count": len(predictors),
    }
    predictions = pd.DataFrame({"row_index": valid_idx, "observed": y.loc[valid_idx].values, "predicted": prob})
    return metrics, predictions


def cross_validated_metrics(data: pd.DataFrame, predictors: list[str], outcome: str, model_name: str, folds: int = 5) -> pd.DataFrame:
    predictors = [p for p in predictors if p in data.columns]
    df = data[predictors + [outcome]].dropna(subset=[outcome]).copy()
    y = df[outcome].astype(int)
    X = df[predictors]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        model = Pipeline([("preprocess", preprocessor(X.iloc[train_idx])), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "n_validation": int(len(valid_idx)),
                "events_validation": int(y.iloc[valid_idx].sum()),
                "auroc": roc_auc_score(y.iloc[valid_idx], prob),
                "auprc": average_precision_score(y.iloc[valid_idx], prob),
                "brier": brier_score_loss(y.iloc[valid_idx], prob),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(data: pd.DataFrame, predictions: pd.DataFrame, subgroup_cols: list[str], output_csv: Path) -> pd.DataFrame:
    joined = predictions.merge(data.reset_index().rename(columns={"index": "row_index"}), on="row_index", how="left")
    rows = []
    for col in subgroup_cols:
        if col not in joined.columns:
            continue
        for level, group in joined.groupby(col, dropna=False):
            if len(group) < 30 or group["observed"].nunique() < 2:
                continue
            rows.append(
                {
                    "subgroup": col,
                    "level": "Missing" if pd.isna(level) else str(level),
                    "n": int(len(group)),
                    "events": int(group["observed"].sum()),
                    "auroc": roc_auc_score(group["observed"], group["predicted"]),
                    "auprc": average_precision_score(group["observed"], group["predicted"]),
                    "brier": brier_score_loss(group["observed"], group["predicted"]),
                }
            )
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def simple_km_table(data: pd.DataFrame, time_col: str, event_col: str, group_col: str | None, output_csv: Path) -> pd.DataFrame:
    df = data[[time_col, event_col] + ([group_col] if group_col else [])].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col])
    if group_col is not None:
        df[group_col] = df[group_col].fillna("Missing").astype(str)
    horizons = [365, 730, 1095]
    rows = []
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col))
    for label, group in groups:
        for horizon in horizons:
            at_risk = group.loc[group[time_col] >= 0]
            observed = at_risk.loc[at_risk[time_col] <= horizon]
            event_n = int(((observed[event_col] == 1)).sum())
            n = int(len(at_risk))
            rows.append(
                {
                    "group": "overall" if group_col is None else ("Missing" if pd.isna(label) else str(label)),
                    "horizon_days": horizon,
                    "n": n,
                    "events_by_horizon": event_n,
                    "naive_event_fraction": event_n / n if n else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
