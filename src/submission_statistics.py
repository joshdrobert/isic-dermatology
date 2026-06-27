from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
sys.path.append(str(REPO_ROOT / "_shared" / "02_common_functions"))

from validation_extras import preprocessor, train_eval  # noqa: E402


RANDOM_STATE = 20260626
BOOTSTRAPS = 2000
COLLECTIONS = (67, 73, 470)


def bootstrap_metrics(
    y: np.ndarray, probability: np.ndarray, groups: np.ndarray | None = None
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    values = {"auroc": [], "auprc": [], "brier": []}
    unique_groups = np.unique(groups) if groups is not None else None
    group_indices = (
        {group: np.flatnonzero(groups == group) for group in unique_groups}
        if unique_groups is not None
        else None
    )
    for _ in range(BOOTSTRAPS):
        if unique_groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_indices[group] for group in sampled_groups])
        if np.unique(y[idx]).size < 2:
            continue
        values["auroc"].append(roc_auc_score(y[idx], probability[idx]))
        values["auprc"].append(average_precision_score(y[idx], probability[idx]))
        values["brier"].append(brier_score_loss(y[idx], probability[idx]))
    result: dict[str, float] = {}
    for metric, samples in values.items():
        low, high = np.quantile(samples, [0.025, 0.975])
        result[metric] = {
            "auroc": roc_auc_score,
            "auprc": average_precision_score,
            "brier": brier_score_loss,
        }[metric](y, probability)
        result[f"{metric}_ci_low"] = float(low)
        result[f"{metric}_ci_high"] = float(high)
    return result


def predictors(data: pd.DataFrame) -> list[str]:
    metadata = ["age", "sex", "anatom_site", "melanocytic", "image_megapixels", "file_size"]
    image = [c for c in data.columns if c.startswith(("rgb_", "hsv_", "edge_", "hist_"))]
    return metadata + image


def grouped_logistic_predictions(
    data: pd.DataFrame, feature_set: list[str], splits: int
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    y = data["outcome_malignant_or_high_risk"].astype(int)
    groups = data["lesion_id"].astype(str)
    X = data[feature_set]
    splitter = StratifiedGroupKFold(
        n_splits=splits, shuffle=True, random_state=RANDOM_STATE
    )
    prediction_parts = []
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = Pipeline(
            [
                ("preprocess", preprocessor(X.iloc[train_idx])),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = model.predict_proba(X.iloc[valid_idx])[:, 1]
        observed = y.iloc[valid_idx].to_numpy(dtype=int)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "row_index": data.index[valid_idx],
                    "lesion_id": groups.iloc[valid_idx].to_numpy(),
                    "observed": observed,
                    "predicted": probability,
                    "fold": fold,
                }
            )
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_validation": len(valid_idx),
                "lesions_validation": groups.iloc[valid_idx].nunique(),
                "events_validation": int(observed.sum()),
                "auroc": roc_auc_score(observed, probability),
                "auprc": average_precision_score(observed, probability),
                "brier": brier_score_loss(observed, probability),
            }
        )
    return pd.concat(prediction_parts, ignore_index=True), fold_rows


def internal_analysis(train: pd.DataFrame) -> None:
    metadata = ["age", "sex", "anatom_site", "melanocytic", "image_megapixels", "file_size"]
    image = [c for c in train.columns if c.startswith(("rgb_", "hsv_", "edge_", "hist_"))]
    random_split_summaries = []
    for name, feature_set in {
        "metadata_only_logistic": metadata,
        "thumbnail_features_logistic": image,
        "combined_metadata_thumbnail_logistic": metadata + image,
    }.items():
        _, predictions = train_eval(
            train, feature_set, "outcome_malignant_or_high_risk", name
        )
        y = predictions["observed"].to_numpy(dtype=int)
        probability = predictions["predicted"].to_numpy(dtype=float)
        random_split_summaries.append(
            {
                "model": name,
                "cohort": "collection_66_held_out",
                "n": len(y),
                "events": int(y.sum()),
                **bootstrap_metrics(y, probability),
            }
        )
    pd.DataFrame(random_split_summaries).to_csv(
        PROJECT_ROOT / "tables" / "table_random_image_split_sensitivity.csv", index=False
    )

    grouped_rows = []
    grouped_predictions: dict[str, pd.DataFrame] = {}
    grouped_fold_rows = []
    for name, feature_set in {
        "metadata_only_logistic": metadata,
        "thumbnail_features_logistic": image,
        "combined_metadata_thumbnail_logistic": metadata + image,
    }.items():
        predictions, fold_rows = grouped_logistic_predictions(train, feature_set, splits=5)
        grouped_predictions[name] = predictions
        y = predictions["observed"].to_numpy(dtype=int)
        probability = predictions["predicted"].to_numpy(dtype=float)
        groups = predictions["lesion_id"].to_numpy()
        grouped_rows.append(
            {
                "model": name,
                "n": len(y),
                "lesions": int(pd.Series(groups).nunique()),
                "events": int(y.sum()),
                **bootstrap_metrics(y, probability, groups),
            }
        )
        for row in fold_rows:
            grouped_fold_rows.append({"model": name, **row})
    pd.DataFrame(grouped_rows).to_csv(
        PROJECT_ROOT / "tables" / "table_lesion_grouped_model_comparison.csv", index=False
    )
    pd.DataFrame(grouped_fold_rows).to_csv(
        PROJECT_ROOT / "tables" / "table_lesion_grouped_cross_validation.csv", index=False
    )

    combined_predictions = grouped_predictions["combined_metadata_thumbnail_logistic"]
    combined_predictions.to_csv(
        PROJECT_ROOT / "validation" / "isic_lesion_grouped_oof_predictions.csv", index=False
    )
    joined = combined_predictions.merge(
        train.reset_index()
        .rename(columns={"index": "row_index"})[["row_index", "sex", "anatom_site"]],
        on="row_index",
        how="left",
    )
    rows = []
    for column in ("sex", "anatom_site"):
        for level, group in joined.groupby(column, dropna=False):
            if len(group) < 30 or group["observed"].nunique() < 2:
                continue
            rows.append(
                {
                    "subgroup": column,
                    "level": "Missing" if pd.isna(level) else str(level),
                    "n": len(group),
                    "events": int(group["observed"].sum()),
                    **bootstrap_metrics(
                        group["observed"].to_numpy(dtype=int),
                        group["predicted"].to_numpy(dtype=float),
                        group["lesion_id"].to_numpy(),
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        PROJECT_ROOT / "tables" / "table_lesion_grouped_subgroup_intervals.csv", index=False
    )


def raw_metadata(collection_id: int) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT
        / "data_raw"
        / f"isic_collection_{collection_id}"
        / f"collection_{collection_id}_metadata.csv"
    )


def external_analysis(train: pd.DataFrame) -> None:
    train_ids = set(train["isic_id"])
    x_columns = predictors(train)
    compatibility_rows = [
        {
            "collection": 66,
            "role": "development",
            "raw_n": len(train),
            "overlap_removed": 0,
            "evaluation_n": len(train),
            "events": int(train["outcome_malignant_or_high_risk"].sum()),
            "event_prevalence": train["outcome_malignant_or_high_risk"].mean(),
            "dominant_attribution": "HAM10000 / collection 66",
            "label_mapping": "Melanoma, BCC, or SCC = malignant",
            "compatibility": "development reference",
            "rationale": "Prespecified development collection",
        }
    ]
    metric_rows = []
    audit_rows = []
    source_rows = []

    for collection_id in COLLECTIONS:
        external = pd.read_csv(
            PROJECT_ROOT
            / "data_processed"
            / f"isic_collection_{collection_id}_thumbnail_image_feature_cohort.csv"
        )
        overlap = external["isic_id"].isin(train_ids)
        evaluation = external.loc[~overlap].copy()
        raw = raw_metadata(collection_id)
        attribution = raw["attribution"].fillna("Missing").value_counts()

        model_path = (
            PROJECT_ROOT
            / "models"
            / f"combined_external_validation_collection_{collection_id}_model.joblib"
        )
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            from external_isic_validation import train_and_external_eval

            temp_metrics = PROJECT_ROOT / "tables" / f"_temporary_collection_{collection_id}.csv"
            train_and_external_eval(
                PROJECT_ROOT / "data_processed" / "isic_thumbnail_image_feature_cohort.csv",
                PROJECT_ROOT
                / "data_processed"
                / f"isic_collection_{collection_id}_thumbnail_image_feature_cohort.csv",
                temp_metrics,
                model_path,
            )
            model = joblib.load(model_path)
            temp_metrics.unlink(missing_ok=True)

        y = evaluation["outcome_malignant_or_high_risk"].to_numpy(dtype=int)
        external_groups = (
            evaluation["lesion_id"].fillna(evaluation["isic_id"]).astype(str).to_numpy()
        )
        probability = model.predict_proba(evaluation[x_columns])[:, 1]
        metrics = bootstrap_metrics(y, probability, external_groups)
        metric_rows.append(
            {
                "collection": collection_id,
                "n": len(y),
                "events": int(y.sum()),
                "overlap_removed": int(overlap.sum()),
                **metrics,
            }
        )
        compatible = collection_id in (67, 73)
        compatibility_rows.append(
            {
                "collection": collection_id,
                "role": "external evaluation" if compatible else "stress test",
                "raw_n": len(external),
                "overlap_removed": int(overlap.sum()),
                "evaluation_n": len(evaluation),
                "events": int(y.sum()),
                "event_prevalence": y.mean(),
                "dominant_attribution": f"{attribution.index[0]} ({attribution.iloc[0]})",
                "label_mapping": "Melanoma, BCC, or SCC = malignant",
                "compatibility": "label-compatible" if compatible else "domain/endpoint stress test",
                "rationale": (
                    "Challenge-era MILK cohort with the same binary diagnosis mapping"
                    if compatible
                    else "Multi-source balanced collection; overlapping MILK images removed"
                ),
            }
        )
        if collection_id == 470:
            audit_rows.extend(
                [
                    {"audit_item": "Evaluated images", "value": len(evaluation)},
                    {"audit_item": "Exact development overlaps removed", "value": int(overlap.sum())},
                    {"audit_item": "Malignant labels", "value": int(y.sum())},
                    {"audit_item": "Benign/indeterminate labels", "value": int((y == 0).sum())},
                    {"audit_item": "Mean predicted risk, malignant labels", "value": probability[y == 1].mean()},
                    {"audit_item": "Mean predicted risk, nonmalignant labels", "value": probability[y == 0].mean()},
                    {"audit_item": "Observed-label AUROC", "value": metrics["auroc"]},
                    {"audit_item": "Outcome-inverted AUROC", "value": roc_auc_score(1 - y, probability)},
                    {
                        "audit_item": "Diagnosis-family mapping disagreements",
                        "value": int(
                            (
                                (evaluation["diagnosis_family"].eq("Malignant").astype(int) != y)
                            ).sum()
                        ),
                    },
                ]
            )
            by_source = evaluation.assign(
                attribution=evaluation["isic_id"].map(
                    raw.set_index("isic_id")["attribution"].to_dict()
                )
            )
            for source, group in by_source.groupby("attribution", dropna=False):
                if len(group) < 30 or group["outcome_malignant_or_high_risk"].nunique() < 2:
                    continue
                source_probability = model.predict_proba(group[x_columns])[:, 1]
                source_y = group["outcome_malignant_or_high_risk"].to_numpy(dtype=int)
                source_rows.append(
                    {
                        "source": "Missing" if pd.isna(source) else str(source),
                        "n": len(group),
                        "events": int(source_y.sum()),
                        "event_prevalence": source_y.mean(),
                        "mean_predicted_risk": source_probability.mean(),
                        "auroc": roc_auc_score(source_y, source_probability),
                    }
                )

    pd.DataFrame(metric_rows).to_csv(
        PROJECT_ROOT / "tables" / "table_external_bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(compatibility_rows).to_csv(
        PROJECT_ROOT / "tables" / "table_collection_compatibility.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        PROJECT_ROOT / "tables" / "table_collection_470_label_audit.csv", index=False
    )
    pd.DataFrame(source_rows).sort_values("n", ascending=False).to_csv(
        PROJECT_ROOT / "tables" / "table_collection_470_source_audit.csv", index=False
    )


def main() -> None:
    train = pd.read_csv(
        PROJECT_ROOT / "data_processed" / "isic_thumbnail_image_feature_cohort.csv"
    )
    internal_analysis(train)
    external_analysis(train)
    print("Wrote bootstrap intervals, compatibility table, and collection 470 audit")


if __name__ == "__main__":
    main()
