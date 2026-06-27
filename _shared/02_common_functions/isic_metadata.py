from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_isic_analysis(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    out = pd.DataFrame(
        {
            "isic_id": df["isic_id"],
            "age": pd.to_numeric(df.get("metadata.clinical.age_approx"), errors="coerce"),
            "sex": df.get("metadata.clinical.sex"),
            "anatom_site": df.get("metadata.clinical.anatom_site_1"),
            "diagnosis_family": df.get("metadata.clinical.diagnosis_1"),
            "diagnosis": df.get("metadata.clinical.diagnosis_3"),
            "diagnosis_confirm_type": df.get("metadata.clinical.diagnosis_confirm_type"),
            "lesion_id": df.get("metadata.clinical.lesion_id"),
            "melanocytic": df.get("metadata.clinical.melanocytic"),
            "pixels_x": pd.to_numeric(df.get("metadata.acquisition.pixels_x"), errors="coerce"),
            "pixels_y": pd.to_numeric(df.get("metadata.acquisition.pixels_y"), errors="coerce"),
            "file_size": pd.to_numeric(df.get("files.full.size"), errors="coerce"),
            "image_url": df.get("files.full.url"),
        }
    )
    malignant_terms = "melanoma|malignant|carcinoma|basal cell|squamous cell"
    out["outcome_malignant_or_high_risk"] = (
        out[["diagnosis_family", "diagnosis"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
        .str.contains(malignant_terms, regex=True)
        .astype(int)
    )
    out["image_megapixels"] = (out["pixels_x"] * out["pixels_y"]) / 1_000_000
    return out


def write_isic_summary(analysis: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    missing = (
        analysis.isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    missing["nonmissing_n"] = [int(analysis[col].notna().sum()) for col in analysis.columns]
    missing.to_csv(tables_dir / "table_missingness.csv", index=False)
    rows = []
    for column in ["sex", "anatom_site", "diagnosis_family", "diagnosis", "diagnosis_confirm_type", "melanocytic"]:
        for value, count in analysis[column].fillna("Missing").value_counts().items():
            rows.append({"variable": column, "level": value, "n": int(count)})
    pd.DataFrame(rows).to_csv(tables_dir / "table_metadata_characteristics.csv", index=False)
    analysis.groupby("diagnosis", dropna=False)["outcome_malignant_or_high_risk"].agg(["count", "sum", "mean"]).reset_index().to_csv(
        tables_dir / "table_outcome_by_diagnosis.csv", index=False
    )
