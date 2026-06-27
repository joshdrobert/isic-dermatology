from __future__ import annotations

import ast
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd


GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


CANCER_PANELS = {
    "HNSC": ["TP53", "CDKN2A", "PIK3CA", "NOTCH1", "FAT1", "EGFR", "CCND1", "SOX2", "KRT5", "KRT14", "EPCAM", "CD274", "PDCD1LG2"],
    "COAD_READ": ["TP53", "APC", "KRAS", "BRAF", "PIK3CA", "SMAD4", "SMAD2", "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM", "MKI67"],
    "PRAD": ["AR", "KLK3", "ERG", "PTEN", "TP53", "SPOP", "FOXA1", "NKX3-1", "MYC", "BRCA1", "BRCA2", "ATM", "CHEK2"],
}


def parse_case_submitter(cases_value: str) -> str | None:
    try:
        cases = json.loads(cases_value)
    except Exception:
        try:
            cases = ast.literal_eval(cases_value)
        except Exception:
            return None
    if not cases:
        return None
    return cases[0].get("submitter_id")


def expression_manifest(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    mask = (
        manifest["access"].eq("open")
        & manifest["data_category"].eq("Transcriptome Profiling")
        & manifest["data_type"].eq("Gene Expression Quantification")
        & manifest["file_name"].astype(str).str.contains("star_gene_counts", case=False, na=False)
    )
    out = manifest.loc[mask].copy()
    out["submitter_id"] = out["cases"].map(parse_case_submitter)
    out = out.dropna(subset=["submitter_id"]).drop_duplicates("submitter_id", keep="first")
    return out


def download_one_expression(row: pd.Series, dest_dir: Path) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / str(row["file_name"])
    error = ""
    if not dest.exists():
        try:
            request = urllib.request.Request(
                f"{GDC_DATA_URL}/{row['file_id']}",
                headers={"User-Agent": "OpenSpecialtyRiskAtlas/0.1"},
            )
            with urllib.request.urlopen(request, timeout=180) as response, dest.open("wb") as handle:
                handle.write(response.read())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            error = str(exc)
    return {
        "submitter_id": row["submitter_id"],
        "file_id": row["file_id"],
        "file_name": row["file_name"],
        "path": str(dest),
        "downloaded": dest.exists(),
        "bytes": dest.stat().st_size if dest.exists() else 0,
        "error": error,
    }


def download_expression_files(manifest_csv: Path, dest_dir: Path, workers: int = 8) -> pd.DataFrame:
    rows = expression_manifest(manifest_csv)
    status = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one_expression, row, dest_dir) for _, row in rows.iterrows()]
        for i, future in enumerate(as_completed(futures), start=1):
            status.append(future.result())
            if i % 25 == 0:
                print(f"Downloaded/checked {i}/{len(rows)} expression files")
    return pd.DataFrame(status)


def parse_target_genes(path: Path, genes: list[str]) -> dict:
    wanted = set(genes)
    row = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        header = None
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                try:
                    gene_idx = header.index("gene_name")
                    tpm_idx = header.index("tpm_unstranded")
                except ValueError:
                    return row
                continue
            if len(parts) <= max(gene_idx, tpm_idx):
                continue
            gene = parts[gene_idx]
            if gene in wanted:
                try:
                    value = float(parts[tpm_idx])
                except ValueError:
                    value = math.nan
                row[f"expr_{gene}"] = math.log1p(value) if not math.isnan(value) else math.nan
    return row


def build_expression_matrix(status_csv: Path, genes: list[str], output_csv: Path) -> pd.DataFrame:
    status = pd.read_csv(status_csv)
    rows = []
    for _, item in status.loc[status["downloaded"].eq(True)].iterrows():
        path = Path(item["path"])
        values = parse_target_genes(path, genes)
        values["submitter_id"] = item["submitter_id"]
        rows.append(values)
    matrix = pd.DataFrame(rows).drop_duplicates("submitter_id", keep="first")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, index=False)
    return matrix


def merge_expression(analysis_csv: Path, expression_csv: Path, output_csv: Path) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_csv)
    expression = pd.read_csv(expression_csv)
    merged = analysis.merge(expression, on="submitter_id", how="left")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged

