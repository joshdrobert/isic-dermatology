from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
sys.path.append(str(REPO_ROOT / "_shared" / "02_common_functions"))

from validation_extras import cross_validated_metrics, subgroup_metrics, train_eval  # noqa: E402


def main() -> None:
    data = pd.read_csv(PROJECT_ROOT / "data_processed" / "isic_thumbnail_image_feature_cohort.csv")
    metadata = ["age", "sex", "anatom_site", "melanocytic", "image_megapixels", "file_size"]
    image = [c for c in data.columns if c.startswith(("rgb_", "hsv_", "edge_", "hist_"))]
    rows = []
    for name, predictors in {
        "metadata_only": metadata,
        "thumbnail_image_features": image,
        "combined_metadata_thumbnail_features": metadata + image,
    }.items():
        metrics, predictions = train_eval(data, predictors, "outcome_malignant_or_high_risk", name)
        rows.append(metrics)
        if name == "combined_metadata_thumbnail_features":
            predictions.to_csv(PROJECT_ROOT / "validation" / "isic_combined_validation_predictions.csv", index=False)
            subgroup_metrics(
                data,
                predictions,
                ["sex", "anatom_site", "diagnosis"],
                PROJECT_ROOT / "tables" / "table_random_image_split_subgroup_sensitivity.csv",
            )
            cross_validated_metrics(
                data, predictors, "outcome_malignant_or_high_risk", name
            ).to_csv(
                PROJECT_ROOT / "tables" / "table_random_image_split_cross_validation_sensitivity.csv",
                index=False,
            )
    pd.DataFrame(rows).sort_values("auroc", ascending=False).to_csv(
        PROJECT_ROOT / "tables" / "table_random_image_split_model_comparison.csv",
        index=False,
    )
    print("Wrote ISIC random-image split sensitivity analyses")


if __name__ == "__main__":
    main()
