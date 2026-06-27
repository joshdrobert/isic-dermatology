from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
sys.path.append(str(REPO_ROOT / "_shared" / "02_common_functions"))

from isic_image_baseline import build_feature_table, download_thumbnails, train_image_feature_model  # noqa: E402


def main() -> None:
    metadata = PROJECT_ROOT / "data_raw" / "isic_collection_66" / "images_metadata.csv"
    cohort = PROJECT_ROOT / "data_processed" / "isic_collection_66_metadata_cohort.csv"
    image_dir = PROJECT_ROOT / "data_raw" / "isic_collection_66" / "thumbnails_256"
    status_csv = PROJECT_ROOT / "tables" / "table_isic_thumbnail_download_status.csv"
    feature_csv = PROJECT_ROOT / "data_processed" / "isic_thumbnail_image_feature_cohort.csv"
    download_thumbnails(metadata, image_dir, status_csv)
    build_feature_table(status_csv, cohort, feature_csv)
    train_image_feature_model(feature_csv, PROJECT_ROOT / "models", PROJECT_ROOT / "tables")
    print("Wrote ISIC thumbnail image-feature baseline")


if __name__ == "__main__":
    main()

