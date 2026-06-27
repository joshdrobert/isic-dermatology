from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from model_comparison import preprocessor


RANDOM_STATE = 20260626


def _metric_intervals(y: np.ndarray, p: np.ndarray, repetitions: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    estimates = {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }
    sampled: dict[str, list[float]] = {name: [] for name in estimates}
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        sampled["auroc"].append(roc_auc_score(y[idx], p[idx]))
        sampled["auprc"].append(average_precision_score(y[idx], p[idx]))
        sampled["brier"].append(brier_score_loss(y[idx], p[idx]))
    result: dict[str, float] = {}
    for name, estimate in estimates.items():
        low, high = np.quantile(sampled[name], [0.025, 0.975])
        result[name] = estimate
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    return result


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _decision_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    prevalence = float(y.mean())
    rows = []
    for threshold in np.arange(0.05, 0.76, 0.05):
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "model_net_benefit": tp / len(y) - fp / len(y) * weight,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * weight,
                "treat_none_net_benefit": 0.0,
                "sensitivity": tp / int(y.sum()),
                "specificity": int(np.sum(~predicted & (y == 0))) / int(np.sum(y == 0)),
            }
        )
    return pd.DataFrame(rows)


def validate_feature_sets(
    data: pd.DataFrame,
    outcome: str,
    feature_sets: dict[str, list[str]],
    output_dir: Path,
    figure_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    y = data[outcome].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    summary_rows = []
    prediction_frames = []
    decision_frames = []

    for name, requested in feature_sets.items():
        predictors = [column for column in requested if column in data.columns]
        X = data[predictors]
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor(X)),
                ("model", LogisticRegression(max_iter=3000)),
            ]
        )
        probability = cross_val_predict(
            pipeline, X, y, cv=splitter, method="predict_proba", n_jobs=1
        )[:, 1]
        metrics = _metric_intervals(y, probability)
        intercept, slope = _calibration(y, probability)
        summary_rows.append(
            {
                "model": name,
                "n": len(y),
                "events": int(y.sum()),
                "predictor_count": len(predictors),
                **metrics,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": data.index,
                    "observed": y,
                    "predicted": probability,
                    "model": name,
                }
            )
        )
        curve = _decision_curve(y, probability)
        curve.insert(0, "model", name)
        decision_frames.append(curve)

    summary = pd.DataFrame(summary_rows).sort_values("auroc", ascending=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    decisions = pd.concat(decision_frames, ignore_index=True)
    summary.to_csv(output_dir / f"table_{prefix}_rigorous_validation.csv", index=False)
    predictions.to_csv(output_dir / f"table_{prefix}_oof_predictions.csv", index=False)
    decisions.to_csv(output_dir / f"table_{prefix}_decision_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, group in predictions.groupby("model"):
        group = group.assign(bin=pd.qcut(group["predicted"], 10, duplicates="drop"))
        calibration = group.groupby("bin", observed=True).agg(
            predicted=("predicted", "mean"), observed=("observed", "mean")
        )
        ax.plot(calibration["predicted"], calibration["observed"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event proportion", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{prefix}_calibration.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, group in decisions.groupby("model"):
        ax.plot(group["threshold"], group["model_net_benefit"], label=name)
    reference = decisions.drop_duplicates("threshold")
    ax.plot(reference["threshold"], reference["treat_all_net_benefit"], linestyle="--", color="gray", label="Treat all")
    ax.axhline(0, linestyle=":", color="black", label="Treat none")
    ax.set(xlabel="Risk threshold", ylabel="Net benefit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{prefix}_decision_curve.png", dpi=300)
    plt.close(fig)
