from __future__ import annotations

import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import http.client
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError

import pandas as pd


GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def tcga_submitter_from_name(name: str) -> str | None:
    match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", name, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def clinical_xml_rows(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    mask = (
        manifest["data_category"].eq("Clinical")
        & manifest["data_type"].eq("Clinical Supplement")
        & manifest["file_name"].astype(str).str.lower().str.endswith(".xml")
    )
    rows = manifest.loc[mask].copy()
    rows["submitter_id"] = rows["file_name"].map(tcga_submitter_from_name)
    rows["supplement_kind"] = rows["file_name"].str.extract(r"org_([^.]*)\.TCGA", expand=False).fillna("clinical")
    return rows


def download_gdc_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{GDC_DATA_URL}/{file_id}",
        headers={"User-Agent": "OpenSpecialtyRiskAtlas/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        handle.write(response.read())


def download_clinical_xml(manifest_csv: Path, xml_dir: Path, sleep_seconds: float = 0.05) -> pd.DataFrame:
    rows = clinical_xml_rows(manifest_csv)
    xml_dir.mkdir(parents=True, exist_ok=True)
    status = []
    for _, row in rows.iterrows():
        dest = xml_dir / str(row["file_name"])
        error = ""
        if not dest.exists():
            try:
                download_gdc_file(str(row["file_id"]), dest)
                time.sleep(sleep_seconds)
            except (HTTPError, URLError, TimeoutError, http.client.RemoteDisconnected, OSError) as exc:
                error = str(exc)
        status.append(
            {
                "file_id": row["file_id"],
                "file_name": row["file_name"],
                "submitter_id": row["submitter_id"],
                "supplement_kind": row["supplement_kind"],
                "path": str(dest),
                "downloaded": dest.exists(),
                "bytes": dest.stat().st_size if dest.exists() else 0,
                "error": error,
            }
        )
    return pd.DataFrame(status)


def parse_clinical_xml(path: Path) -> dict:
    tree = ET.parse(path)
    values: dict[str, list[str]] = defaultdict(list)
    for elem in tree.iter():
        text = (elem.text or "").strip()
        if text and not list(elem):
            key = _local_name(elem.tag)
            if text not in values[key]:
                values[key].append(text)
    row = {key: "|".join(items) for key, items in values.items()}
    row["source_file"] = path.name
    row["submitter_id"] = tcga_submitter_from_name(path.name)
    row["supplement_kind"] = re.search(r"org_([^.]*)\.TCGA", path.name)
    row["supplement_kind"] = row["supplement_kind"].group(1) if row["supplement_kind"] else "clinical"
    return row


def parse_xml_directory(xml_dir: Path) -> pd.DataFrame:
    rows = [parse_clinical_xml(path) for path in sorted(xml_dir.glob("*.xml"))]
    return pd.DataFrame(rows)


def first_present(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    result = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for column in candidates:
        if column in df.columns:
            result = result.fillna(df[column])
    return result


def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"[Not Available]": pd.NA, "[Not Applicable]": pd.NA, "[Unknown]": pd.NA}), errors="coerce")


def build_tcga_analysis_table(parsed: pd.DataFrame, project_id: str) -> pd.DataFrame:
    clinical = parsed.loc[parsed["supplement_kind"].str.contains("clinical", case=False, na=False)].copy()
    if clinical.empty:
        clinical = parsed.copy()
    clinical = clinical.drop_duplicates("submitter_id", keep="first")
    age_raw = to_numeric_clean(first_present(clinical, ["age_at_initial_pathologic_diagnosis", "age_at_diagnosis"]))
    age_years = age_raw.where(age_raw <= 200, age_raw / 365.25)
    out = pd.DataFrame(
        {
            "submitter_id": clinical["submitter_id"],
            "project_id": project_id,
            "sex": first_present(clinical, ["gender", "sex"]),
            "vital_status": first_present(clinical, ["vital_status"]),
            "days_to_death": to_numeric_clean(first_present(clinical, ["days_to_death"])),
            "days_to_last_followup": to_numeric_clean(first_present(clinical, ["days_to_last_followup", "days_to_last_follow_up"])),
            "age_at_diagnosis_raw": age_raw,
            "pathologic_stage": first_present(clinical, ["pathologic_stage", "stage_event_pathologic_stage"]),
            "pathologic_t": first_present(clinical, ["pathologic_t", "pathologic_t_stage"]),
            "pathologic_n": first_present(clinical, ["pathologic_n", "pathologic_n_stage"]),
            "pathologic_m": first_present(clinical, ["pathologic_m", "pathologic_m_stage"]),
            "histologic_grade": first_present(clinical, ["histologic_grade", "neoplasm_histologic_grade"]),
            "primary_site": first_present(clinical, ["tissue_or_organ_of_origin", "anatomic_neoplasm_subdivision"]),
            "smoking_history": first_present(clinical, ["tobacco_smoking_history", "number_pack_years_smoked"]),
            "gleason_score": first_present(clinical, ["gleason_score", "primary_gleason_grade", "secondary_gleason_grade"]),
        }
    )
    out["age_at_diagnosis_years"] = age_years
    out["event_death"] = out["vital_status"].astype(str).str.lower().str.contains("dead|deceased").astype(int)
    out["overall_survival_days"] = out["days_to_death"].fillna(out["days_to_last_followup"])
    return out


def write_summary_tables(analysis: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for column in analysis.columns:
        missing = analysis[column].isna().mean()
        summary.append({"variable": column, "missing_fraction": missing, "nonmissing_n": int(analysis[column].notna().sum())})
    pd.DataFrame(summary).to_csv(tables_dir / "table_missingness.csv", index=False)
    categorical = ["sex", "vital_status", "pathologic_stage", "pathologic_t", "pathologic_n", "pathologic_m", "histologic_grade", "primary_site"]
    rows = []
    for column in categorical:
        if column in analysis.columns:
            for value, count in analysis[column].fillna("Missing").value_counts(dropna=False).items():
                rows.append({"variable": column, "level": value, "n": int(count)})
    pd.DataFrame(rows).to_csv(tables_dir / "table_clinical_characteristics.csv", index=False)
