from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 20260626


def build_survival_design(data: pd.DataFrame, predictors: list[str], duration_col: str, event_col: str) -> pd.DataFrame:
    predictors = [p for p in predictors if p in data.columns]
    df = data[predictors + [duration_col, event_col]].copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[duration_col, event_col])
    df = df.loc[df[duration_col] > 0].copy()
    X = df[predictors]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        verbose_feature_names_out=False,
    )
    Xp = pd.DataFrame(pre.fit_transform(X), columns=pre.get_feature_names_out(), index=df.index)
    Xp[duration_col] = df[duration_col]
    Xp[event_col] = df[event_col].astype(int)
    # CoxPH is sensitive to all-zero or duplicate columns.
    feature_cols = [c for c in Xp.columns if c not in [duration_col, event_col]]
    keep = [c for c in feature_cols if Xp[c].std() > 1e-8]
    return Xp[keep + [duration_col, event_col]]


def fit_cox_model(data: pd.DataFrame, predictors: list[str], duration_col: str, event_col: str, output_dir: Path, prefix: str) -> pd.DataFrame:
    design = build_survival_design(data, predictors, duration_col, event_col)
    train, valid = train_test_split(design, test_size=0.25, stratify=design[event_col], random_state=RANDOM_STATE)
    penalizers = [0.01, 0.05, 0.1, 0.5, 1.0]
    rows = []
    best = None
    for penalizer in penalizers:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(train, duration_col=duration_col, event_col=event_col)
            risk = cph.predict_partial_hazard(valid)
            c_index = concordance_index(valid[duration_col], -risk, valid[event_col])
            rows.append({"model": f"cox_penalized_{penalizer}", "penalizer": penalizer, "n_train": len(train), "n_validation": len(valid), "events_validation": int(valid[event_col].sum()), "c_index": c_index})
            if best is None or c_index > best[0]:
                best = (c_index, cph, penalizer)
        except Exception as exc:
            rows.append({"model": f"cox_penalized_{penalizer}", "penalizer": penalizer, "error": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / f"table_{prefix}_cox_performance.csv", index=False)
    if best is not None:
        best[1].summary.reset_index().to_csv(output_dir / f"table_{prefix}_cox_coefficients.csv", index=False)
    return results


def write_km_plot(data: pd.DataFrame, duration_col: str, event_col: str, group_col: str, output_png: Path) -> None:
    import matplotlib.pyplot as plt

    df = data[[duration_col, event_col, group_col]].copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df[group_col] = df[group_col].fillna("Missing").astype(str)
    df = df.dropna(subset=[duration_col, event_col])
    df = df.loc[df[duration_col] > 0].copy()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    kmf = KaplanMeierFitter()
    for group, subset in df.groupby(group_col):
        if len(subset) < 10:
            continue
        kmf.fit(subset[duration_col], subset[event_col], label=group)
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Kaplan-Meier curves by {group_col}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

