from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, label: str, tolerance: float = 5e-4) -> None:
    if not np.isclose(actual, expected, atol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def main() -> None:
    cohort = pd.read_csv(
        PROJECT_ROOT / "data_processed" / "isic_thumbnail_image_feature_cohort.csv"
    )
    grouped = pd.read_csv(
        PROJECT_ROOT / "tables" / "table_lesion_grouped_model_comparison.csv"
    ).set_index("model")
    external = pd.read_csv(
        PROJECT_ROOT / "tables" / "table_external_bootstrap_intervals.csv"
    ).set_index("collection")
    audit = pd.read_csv(
        PROJECT_ROOT / "tables" / "table_collection_470_label_audit.csv"
    ).set_index("audit_item")["value"]
    manuscript = (PROJECT_ROOT / "manuscript" / "manuscript.tex").read_text(
        encoding="utf-8"
    )

    assert len(cohort) == 10015
    assert cohort["lesion_id"].notna().all()
    assert cohort["lesion_id"].nunique() == 7470
    assert int(cohort["outcome_malignant_or_high_risk"].sum()) == 1824
    assert int(cohort["diagnosis"].eq("Melanoma, NOS").sum()) == 1113
    assert int(cohort["diagnosis"].eq("Basal cell carcinoma").sum()) == 514
    assert int(cohort["diagnosis"].eq("Squamous cell carcinoma, NOS").sum()) == 197

    combined = grouped.loc["combined_metadata_thumbnail_logistic"]
    metadata = grouped.loc["metadata_only_logistic"]
    close(metadata["auroc"], 0.762274, "Grouped metadata AUROC")
    close(combined["auroc"], 0.828750, "Grouped combined AUROC")
    close(combined["auprc"], 0.431405, "Grouped combined AUPRC")
    close(combined["brier"], 0.175069, "Grouped combined Brier")
    close(combined["auroc_ci_low"], 0.817826, "Grouped AUROC CI low")
    close(combined["auroc_ci_high"], 0.839529, "Grouped AUROC CI high")

    expected_external = {
        67: (1512, 288, 0, 0.809014),
        73: (193, 44, 0, 0.852807),
        470: (7900, 4112, 1910, 0.270164),
    }
    for collection, (n, events, overlap, auroc) in expected_external.items():
        row = external.loc[collection]
        assert int(row["n"]) == n
        assert int(row["events"]) == events
        assert int(row["overlap_removed"]) == overlap
        close(row["auroc"], auroc, f"Collection {collection} AUROC")

    assert int(audit["Diagnosis-family mapping disagreements"]) == 0
    close(audit["Outcome-inverted AUROC"], 0.729836, "Outcome-inverted AUROC")

    required_phrases = [
        "10,015 image records representing 7,470 lesions",
        "Lesion-grouped out-of-fold AUROC was 0.762",
        "0.829 (95\\% lesion-cluster bootstrap CI, 0.818--0.840)",
        "challenge validation partition",
        "challenge test partition",
        "not represent two independent external institutions",
        "balanced class weights",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in manuscript]
    if missing:
        raise AssertionError(f"Required manuscript phrases missing: {missing}")

    forbidden_phrases = [
        "two compatible external collections",
        "patient and lesion identifiers were unavailable",
        "prespecified 75/25",
    ]
    present = [phrase for phrase in forbidden_phrases if phrase in manuscript]
    if present:
        raise AssertionError(f"Superseded manuscript phrases remain: {present}")

    print("ISIC manuscript claim verification passed")


if __name__ == "__main__":
    main()
