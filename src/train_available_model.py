from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
sys.path.append(str(REPO_ROOT / "_shared" / "02_common_functions"))

from tabular_risk_model import train_logistic_score  # noqa: E402


def main() -> None:
    data = pd.read_csv(PROJECT_ROOT / "data_processed" / "isic_collection_66_metadata_cohort.csv")
    predictors = ["age", "sex", "anatom_site", "melanocytic", "image_megapixels", "file_size"]
    train_logistic_score(
        data,
        predictors,
        "outcome_malignant_or_high_risk",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "tables",
    )
    print("Trained ISIC metadata escalation score")


if __name__ == "__main__":
    main()

