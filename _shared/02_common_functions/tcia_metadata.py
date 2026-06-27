from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_nsclc_metadata(patient_csv: Path, series_csv: Path) -> pd.DataFrame:
    patients = pd.read_csv(patient_csv)
    series = pd.read_csv(series_csv)
    series["SeriesDate"] = pd.to_datetime(series["SeriesDate"], errors="coerce")
    series["ImageCount"] = pd.to_numeric(series["ImageCount"], errors="coerce")
    series["FileSize"] = pd.to_numeric(series["FileSize"], errors="coerce")
    patient_summary = (
        series.groupby("PatientID")
        .agg(
            series_n=("SeriesInstanceUID", "nunique"),
            total_image_count=("ImageCount", "sum"),
            total_file_size_bytes=("FileSize", "sum"),
            modality_set=("Modality", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
            manufacturer_set=("Manufacturer", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
            first_series_date=("SeriesDate", "min"),
        )
        .reset_index()
        .rename(columns={"PatientID": "PatientId"})
    )
    return patients.merge(patient_summary, on="PatientId", how="left")


def write_nsclc_summary(analysis: pd.DataFrame, series_csv: Path, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(series_csv)
    analysis.to_csv(tables_dir / "table_patient_imaging_inventory.csv", index=False)
    series.groupby(["Modality", "Manufacturer"], dropna=False).size().reset_index(name="series_n").to_csv(
        tables_dir / "table_series_by_modality_manufacturer.csv", index=False
    )
    missing = (
        analysis.isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    missing.to_csv(tables_dir / "table_missingness.csv", index=False)
