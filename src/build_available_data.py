from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
sys.path.append(str(REPO_ROOT / "_shared" / "02_common_functions"))

from isic_metadata import build_isic_analysis, write_isic_summary  # noqa: E402


def main() -> None:
    metadata = PROJECT_ROOT / "data_raw" / "isic_collection_66" / "images_metadata.csv"
    processed = PROJECT_ROOT / "data_processed"
    tables = PROJECT_ROOT / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    analysis = build_isic_analysis(metadata)
    analysis.to_csv(processed / "isic_collection_66_metadata_cohort.csv", index=False)
    write_isic_summary(analysis, tables)
    print(f"Wrote ISIC metadata cohort: {len(analysis):,} images")


if __name__ == "__main__":
    main()

