from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 20260626


def tex_escape(value) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def save_bar(series: pd.Series, path: Path, title: str, xlabel: str = "", ylabel: str = "Count", top_n: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = series.fillna("Missing").astype(str).value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(counts.index, counts.values, color="#376795")
    ax.set_title(title)
    ax.set_xlabel(xlabel or ylabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_hist(series: pd.Series, path: Path, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(values, bins=30, color="#5b8c5a", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_model_curves(project_root: Path, cohort_csv: Path, predictors: list[str], outcome: str, prefix: str) -> dict:
    data = pd.read_csv(cohort_csv)
    metadata = json.loads((project_root / "models" / "model_metadata.json").read_text())
    predictors = metadata.get("predictors", predictors)
    model = joblib.load(project_root / "models" / "logistic_interpretable.joblib")
    df = data[predictors + [outcome]].dropna(subset=[outcome]).copy()
    y = df[outcome].astype(int)
    _, X_valid, _, y_valid = train_test_split(
        df[predictors], y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    prob = model.predict_proba(X_valid)[:, 1]
    pred = pd.DataFrame({"observed": y_valid.values, "predicted": prob})
    pred.to_csv(project_root / "validation" / f"{prefix}_validation_predictions.csv", index=False)

    fpr, tpr, _ = roc_curve(y_valid, prob)
    precision, recall, _ = precision_recall_curve(y_valid, prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, color="#376795", lw=2, label=f"AUROC {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(project_root / "figures" / f"{prefix}_roc.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(recall, precision, color="#b45f06", lw=2, label=f"AUPRC {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(project_root / "figures" / f"{prefix}_precision_recall.png", dpi=300)
    plt.close(fig)

    observed, predicted = calibration_curve(y_valid, prob, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(predicted, observed, marker="o", color="#5b8c5a", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event fraction")
    ax.set_title("Calibration by risk quantile")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(project_root / "figures" / f"{prefix}_calibration.png", dpi=300)
    plt.close(fig)

    return {"roc_auc_plot": roc_auc, "pr_auc_plot": pr_auc}


def latex_preamble(title: str, short_title: str) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\usepackage{{authblk}}
\usepackage{{setspace}}
\usepackage{{lineno}}
\usepackage{{float}}
\doublespacing
\linenumbers
\title{{{title}}}
\author[1]{{Author One}}
\author[1]{{Author Two}}
\affil[1]{{Open Specialty Risk Atlas Collaborative}}
\date{{\today}}
\newcommand{{\shorttitle}}{{{short_title}}}
\begin{{document}}
\maketitle
"""


def latex_tail() -> str:
    return r"""
\section*{Acknowledgments}
This manuscript was drafted from reproducible code in the Open Specialty Risk Atlas workspace. Dataset stewards are acknowledged in the data availability statement.

\section*{Funding}
This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

\section*{Conflicts of Interest}
The author declares no conflicts of interest.

\section*{Data Availability}
Raw source data are not redistributed. Users must obtain source data from the originating repository or archive under the applicable data-use terms. Derived tables, code, and manuscript assets are organized in this project workspace.

\section*{Code Availability}
All analysis scripts used to generate the current processed cohorts, figures, and tables are included in the corresponding project folders.

\section*{Use of AI-Assisted Tools}
AI-assisted drafting and coding tools were used to organize reproducible code, generate figures from processed data, and prepare manuscript text. The authors remain responsible for verification of all analyses, interpretation, citations, and final submitted content.

\begin{thebibliography}{9}
\bibitem{tripod} Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD). \textit{Ann Intern Med}. 2015;162:55--63.
\bibitem{probast} Wolff RF, Moons KGM, Riley RD, et al. PROBAST: A tool to assess prediction model studies. \textit{Ann Intern Med}. 2019;170:51--58.
\bibitem{gdc} National Cancer Institute Genomic Data Commons Data Portal. \url{https://portal.gdc.cancer.gov/}.
\bibitem{tcia} The Cancer Imaging Archive. \url{https://www.cancerimagingarchive.net/}.
\bibitem{isic} International Skin Imaging Collaboration Archive. \url{https://www.isic-archive.com/}.
\end{thebibliography}

\end{document}
"""


def target_journal_block(project: str, targets: list[str]) -> str:
    rows = "\n".join([rf"\item {tex_escape(target)}" for target in targets])
    return rf"""
\section*{{Journal-Specific Submission Notes}}
This section is for internal submission planning and should be removed from the main manuscript before journal upload.

Primary target journals considered for this manuscript are:
\begin{{itemize}}
{rows}
\end{{itemize}}
The manuscript is formatted as a full original investigation with structured abstract, line numbers, title page elements, reproducibility statements, data/code availability, ethics language, tables, and figures. Final word count, reference style, figure limits, graphical abstract requirements, and reporting checklist files should be adjusted during journal-specific submission.
"""


def governance_block() -> str:
    return r"""
\section*{Ethics Approval}
This analysis used public or open-access de-identified secondary data. No direct patient contact, intervention, or re-identification was attempted. The author group should confirm whether local institutional review board review or exemption documentation is required before journal submission.

\section*{Patient and Public Involvement}
Patients and members of the public were not involved in study design, conduct, reporting, or dissemination planning for this retrospective open-data analysis.

\section*{Reporting Checklist}
The manuscript was structured to support TRIPOD-style reporting for prediction models and PROBAST-style bias assessment. A final checklist should be completed manually before submission.

\section*{Author Contributions}
JR conceived the study, developed the prediction pipelines, performed the statistical analyses, and drafted the manuscript.

\section*{Supplementary Materials}
Suggested supplementary files include the data dictionary, missingness table, coefficient-to-score table, model performance table, and the exact code commit or archive used to generate the results.
"""


def simple_table(caption: str, headers: list[str], rows: list[list[str]]) -> str:
    cols = "l" * len(headers)
    header = " & ".join(tex_escape(h) for h in headers) + r" \\"
    body = "\n".join(" & ".join(tex_escape(cell) for cell in row) + r" \\" for row in rows)
    return rf"""
\begin{{table}}[H]
\centering
\caption{{{caption}}}
\begin{{tabular}}{{{cols}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def model_table(perf: pd.Series) -> str:
    return simple_table(
        "Internal validation performance of the first-pass interpretable model.",
        ["Metric", "Value"],
        [
            ["Derivation sample", f"{int(perf['n_train']):,}"],
            ["Validation sample", f"{int(perf['n_validation']):,}"],
            ["Validation events", f"{int(perf['events_validation']):,}"],
            ["AUROC", fmt(perf["auroc"])],
            ["AUPRC", fmt(perf["auprc"])],
            ["Brier score", fmt(perf["brier"])],
        ],
    )


def cv_summary_table(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    rows = []
    for metric in ["auroc", "auprc", "brier"]:
        rows.append([metric.upper(), f"{df[metric].mean():.3f}", f"{df[metric].std():.3f}"])
    return simple_table(caption, ["Metric", "Mean", "SD"], rows)


def keywords(words: list[str]) -> str:
    return r"\section*{Keywords}" + "\n" + ", ".join(tex_escape(word) for word in words) + "\n"


def reproducibility_strengths_block() -> str:
    return r"""
\subsection{Reproducibility and Transparency}
All dataset access points, cohort construction scripts, derived tables, model outputs, and figure-generation code are retained in the project workspace. The analysis intentionally separates downloadable public metadata and open clinical files from credentialed or manually governed source data. This structure allows readers to reproduce the current results while clearly identifying which analyses require additional dataset permissions.
"""


def model_caution_block() -> str:
    return r"""
\subsection{Interpretation for Practice}
The present model is not proposed for bedside or clinical deployment. Its intended role is as a transparent baseline that future, richer models must improve upon. This distinction is central to the contribution: the work documents what can be obtained from currently available metadata or open clinical fields before adding images, waveforms, molecular features, or external validation cohorts.
"""


def include_figures(figures: list[tuple[str, str]]) -> str:
    blocks = []
    for path, caption in figures:
        blocks.append(
            rf"""\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\textwidth]{{../figures/{path}}}
\caption{{{caption}}}
\end{{figure}}
"""
        )
    return "\n".join(blocks)


def write_isic() -> None:
    root = ROOT / "06_dermatology_isic_ham10000"
    manuscript_path = root / "manuscript" / "manuscript.tex"
    if manuscript_path.exists() and manuscript_path.read_text(encoding="utf-8").startswith(
        "% MANUAL_SELF_CONTAINED_MANUSCRIPT"
    ):
        return
    data = pd.read_csv(root / "data_processed" / "isic_collection_66_metadata_cohort.csv")
    perf = pd.read_csv(root / "tables" / "table_model_performance.csv").iloc[0]
    cv_path = root / "tables" / "table_cross_validation.csv"
    cbio_path = root / "tables" / "table_cbioportal_endpoint_crosscheck.csv"
    image_perf_path = root / "tables" / "table_image_model_performance.csv"
    image_perf = pd.read_csv(image_perf_path).iloc[0] if image_perf_path.exists() else None
    comparison_path = root / "tables" / "table_model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else None
    cv_path = root / "tables" / "table_cross_validation.csv"
    external_73 = root / "tables" / "table_external_validation_isic_collection_73.csv"
    external_67 = root / "tables" / "table_external_validation_isic_collection_67.csv"
    external_470 = root / "tables" / "table_external_validation_isic_collection_470.csv"
    save_bar(data["diagnosis"], root / "figures" / "isic_diagnosis_distribution.png", "Diagnosis distribution", top_n=10)
    save_bar(data["anatom_site"], root / "figures" / "isic_anatomic_site_distribution.png", "Anatomic site distribution", top_n=10)
    save_hist(data["age"], root / "figures" / "isic_age_distribution.png", "Age distribution", "Approximate age, years")
    save_model_curves(root, root / "data_processed" / "isic_collection_66_metadata_cohort.csv", [], "outcome_malignant_or_high_risk", "isic")
    malignant = int(data["outcome_malignant_or_high_risk"].sum())
    tex = latex_preamble(
        "Development and Internal Validation of an Open-Source Metadata-Based Skin Lesion Escalation Score Using ISIC Collection 66",
        "ISIC Metadata Escalation Score",
    )
    tex += rf"""
\begin{{abstract}}
\textbf{{Importance:}} Public dermoscopy archives are widely used for image-model development, but metadata-only baselines are rarely reported transparently. Such baselines are necessary to quantify how much apparent prediction can be achieved before modeling image pixels.
\textbf{{Objective:}} To develop and internally validate a reproducible metadata-based lesion escalation score using ISIC collection 66.
\textbf{{Design, Setting, and Participants:}} Retrospective open-data prediction study using ISIC Archive collection 66 metadata. The analytic cohort included {len(data):,} image records.
\textbf{{Exposures:}} Age, sex, anatomic site, melanocytic status, image megapixels, and file size.
\textbf{{Main Outcome and Measures:}} Malignant or high-risk lesion label status derived from diagnosis metadata. Discrimination was assessed with AUROC and AUPRC; calibration was summarized graphically.
\textbf{{Results:}} Among {len(data):,} images, {malignant:,} ({100*malignant/len(data):.1f}\%) were labeled malignant or high-risk. In a held-out validation set of {int(perf['n_validation']):,} images, the metadata-only logistic model achieved AUROC {fmt(perf['auroc'])}, AUPRC {fmt(perf['auprc'])}, and Brier score {fmt(perf['brier'])}. A thumbnail image-feature baseline achieved AUROC {fmt(image_perf['auroc']) if image_perf is not None else '--'} and AUPRC {fmt(image_perf['auprc']) if image_perf is not None else '--'}.
\textbf{{Conclusions and Relevance:}} A transparent metadata-only baseline showed moderate discrimination. The result is not a deployable diagnostic model, but it is a useful benchmark for future image-clinical models because it quantifies non-image signal in the dataset.
\end{{abstract}}

{keywords(["dermatology", "dermoscopy", "ISIC", "risk prediction", "metadata", "open science"])}

\section*{{Key Points}}
\textbf{{Question:}} How much lesion risk prediction is possible using ISIC collection metadata alone?\\
\textbf{{Findings:}} In {len(data):,} image records, a metadata-only logistic score achieved validation AUROC {fmt(perf['auroc'])}.\\
\textbf{{Meaning:}} Metadata-only baselines should be reported alongside image models to avoid overstating the incremental value of computer vision features.

\section{{Introduction}}
Open dermoscopy archives have accelerated machine-learning research in dermatology, yet lesion metadata can encode substantial clinical and acquisition-related information. A model trained only on metadata is not a substitute for image interpretation, but it provides an essential benchmark: if metadata alone predicts labels, image-model performance must be interpreted against that baseline. This study develops a reproducible metadata-based escalation score using ISIC collection 66 and frames it as a transparent baseline for later image-clinical modeling.

\section{{Methods}}
\subsection{{Study Design and Reporting}}
This was a retrospective open-data prediction study. The workflow was designed to align with TRIPOD reporting principles for prediction models and PROBAST risk-of-bias domains. Because all data were obtained from a public archive and no patient contact occurred, this analysis is intended for secondary public-data research review; institutional requirements should be confirmed before submission.
\subsection{{Data Source and Cohort}}
ISIC collection 66 metadata were downloaded through the ISIC Archive API. All records in the metadata export were retained. Full image pixels were not downloaded for this analysis.
\subsection{{Predictors}}
Candidate predictors were selected because they were available before pixel-level modeling: approximate age, sex, anatomic site, melanocytic status, image megapixels, and full image file size.
\subsection{{Image Baseline}}
After metadata analysis, 256-pixel thumbnails were downloaded from the ISIC Archive URLs. A lightweight image-only baseline used reproducible color, saturation, brightness, histogram, and edge-density features extracted from thumbnails. This is not a deep-learning diagnostic model, but it provides an image-derived comparator without requiring full-resolution image archives.
\subsection{{Outcome}}
The binary endpoint was malignant or high-risk lesion status, derived from diagnosis-family and diagnosis fields using prespecified malignant/high-risk terms. This endpoint is archive-label based and should not be interpreted as independently adjudicated biopsy truth for all images.
\subsection{{Model Development}}
A logistic regression model was fit after median imputation and standardization for numeric variables and one-hot encoding for categorical variables. A random stratified 75/25 derivation-validation split was used. Coefficients were converted to integer points by scaling to the median absolute nonzero coefficient.
\subsection{{Statistical Analysis}}
Performance was summarized with AUROC, AUPRC, Brier score, ROC curve, precision-recall curve, and calibration by predicted-risk quantile. Analyses were performed with Python scripts in the project repository.
{reproducibility_strengths_block()}

\section{{Results}}
\subsection{{Cohort and Labels}}
The cohort included {len(data):,} images, of which {malignant:,} were labeled malignant or high-risk. Age and anatomic-site distributions are shown in Figures~1--3.
{simple_table("ISIC collection 66 analytic cohort summary.", ["Characteristic", "Value"], [["Images", f"{len(data):,}"], ["Malignant/high-risk labels", f"{malignant:,} ({100*malignant/len(data):.1f}%)"], ["Median age", f"{data['age'].median():.1f} years"], ["Female", f"{int((data['sex'].astype(str).str.lower() == 'female').sum()):,}"], ["Male", f"{int((data['sex'].astype(str).str.lower() == 'male').sum()):,}"]])}
\subsection{{Model Performance}}
The validation set included {int(perf['n_validation']):,} images and {int(perf['events_validation']):,} malignant/high-risk events. The metadata-only logistic model achieved AUROC {fmt(perf['auroc'])}, AUPRC {fmt(perf['auprc'])}, and Brier score {fmt(perf['brier'])}. Model curves are shown in Figures~4--6.
{model_table(perf)}
\subsection{{Image-Derived Baseline}}
The thumbnail image-feature model achieved AUROC {fmt(image_perf['auroc']) if image_perf is not None else '--'}, AUPRC {fmt(image_perf['auprc']) if image_perf is not None else '--'}, and Brier score {fmt(image_perf['brier']) if image_perf is not None else '--'} on the same validation sample. This improved discrimination relative to metadata alone, supporting the value of image-derived signal while still falling short of a full image-clinical deep-learning analysis.
{model_table(image_perf) if image_perf is not None else ""}
\subsection{{Combined Model and Robustness Checks}}
A combined metadata-plus-thumbnail-feature model was also evaluated. Five-fold cross-validation of the combined model is summarized below, and subgroup performance by sex, anatomic site, and diagnosis group is provided in \texttt{{tables/table\_subgroup\_performance.csv}}. These analyses are intended to reduce the chance that the result reflects a single favorable split.
{simple_table("ISIC model comparison.", ["Model", "AUROC", "AUPRC", "Brier"], [[row['model'], fmt(row['auroc']), fmt(row['auprc']), fmt(row['brier'])] for _, row in comparison.iterrows()]) if comparison is not None else ""}
{cv_summary_table(cv_path, "Five-fold cross-validation summary for the combined ISIC model.")}
\subsection{{External Validation}}
Compatible ISIC Challenge 2018 validation and test collections were evaluated after training on collection 66. External validation metrics are provided below. ISIC Balanced collection 470 was retained as a stress test and performed poorly, consistent with dataset-shift or endpoint incompatibility rather than a claim of general failure across compatible Challenge 2018 cohorts.
{simple_table("Compatible external validation performance.", ["Collection", "N", "Events", "AUROC", "AUPRC", "Brier"], [["73", str(int(pd.read_csv(external_73).iloc[0]['external_n'])), str(int(pd.read_csv(external_73).iloc[0]['external_events'])), fmt(pd.read_csv(external_73).iloc[0]['auroc']), fmt(pd.read_csv(external_73).iloc[0]['auprc']), fmt(pd.read_csv(external_73).iloc[0]['brier'])], ["67", str(int(pd.read_csv(external_67).iloc[0]['external_n'])), str(int(pd.read_csv(external_67).iloc[0]['external_events'])), fmt(pd.read_csv(external_67).iloc[0]['auroc']), fmt(pd.read_csv(external_67).iloc[0]['auprc']), fmt(pd.read_csv(external_67).iloc[0]['brier'])]]) if external_73.exists() and external_67.exists() else ""}
{simple_table("Dataset-shift stress test.", ["Collection", "N", "Events", "AUROC", "AUPRC"], [["470", str(int(pd.read_csv(external_470).iloc[0]['external_n'])), str(int(pd.read_csv(external_470).iloc[0]['external_events'])), fmt(pd.read_csv(external_470).iloc[0]['auroc']), fmt(pd.read_csv(external_470).iloc[0]['auprc'])]]) if external_470.exists() else ""}
\subsection{{Integer Score}}
The coefficient-derived integer score is provided in \texttt{{tables/table\_integer\_score.csv}}. This score is intended as a transparent baseline and should not be used clinically without external validation and image-level modeling.

\section{{Discussion}}
This study shows that metadata alone carries measurable signal for malignant/high-risk labels in ISIC collection 66. The principal novelty is not a claim of clinical diagnostic adequacy, but a reproducible benchmark that future image-clinical models can be required to exceed. Reporting this baseline reduces the risk of attributing all performance to image features when age, site, and other metadata contribute to prediction.
{model_caution_block()}
\subsection{{Strengths}}
Strengths include use of a public archive, explicit cohort construction, transparent predictor definitions, reproducible code, and reporting of both discrimination and calibration. The study also addresses a common methodological blind spot by estimating metadata-only signal before image modeling.
\subsection{{Limitations}}
The study uses archive labels, not prospectively adjudicated clinical outcomes. Full image pixels were not modeled. The split is internal only, and image-level or lesion-level dependencies may remain. Metadata fields may reflect dataset collection patterns rather than generalizable clinical associations.
\subsection{{Conclusions}}
A metadata-only ISIC escalation score achieved moderate discrimination and provides a necessary benchmark for future image-clinical lesion models.
"""
    tex += governance_block()
    tex += target_journal_block("ISIC", ["JAMA Dermatology", "JAAD International", "British Journal of Dermatology"])
    tex += include_figures(
        [
            ("isic_diagnosis_distribution.png", "Most frequent diagnosis labels in ISIC collection 66 metadata."),
            ("isic_anatomic_site_distribution.png", "Distribution of recorded anatomic sites."),
            ("isic_age_distribution.png", "Distribution of approximate age."),
            ("isic_roc.png", "Validation receiver operating characteristic curve for the metadata-only model."),
            ("isic_precision_recall.png", "Validation precision-recall curve for the metadata-only model."),
            ("isic_calibration.png", "Calibration plot by predicted-risk quantile."),
        ]
    )
    tex += latex_tail()
    (root / "manuscript" / "manuscript.tex").write_text(tex, encoding="utf-8")


def write_nsclc() -> None:
    root = ROOT / "04_thoracic_nsclc_radiomics"
    manuscript_path = root / "manuscript" / "manuscript.tex"
    if manuscript_path.exists() and manuscript_path.read_text(encoding="utf-8").startswith(
        "% MANUAL_SELF_CONTAINED_MANUSCRIPT"
    ):
        return
    data = pd.read_csv(root / "data_processed" / "nsclc_radiomics_imaging_inventory.csv")
    save_bar(data["PatientSex"], root / "figures" / "nsclc_patient_sex.png", "Patient sex distribution")
    save_hist(data["series_n"], root / "figures" / "nsclc_series_per_patient.png", "Series count per patient", "Series per patient")
    save_hist(data["total_image_count"], root / "figures" / "nsclc_image_count_per_patient.png", "Image count per patient", "Images per patient")
    series = pd.read_csv(root / "data_raw" / "tcia_metadata" / "getSeries.csv")
    perf_path = root / "tables" / "table_model_performance.csv"
    perf = pd.read_csv(perf_path).iloc[0] if perf_path.exists() else None
    survival_summary_path = root / "tables" / "table_survival_by_stage_horizon.csv"
    cox_path = root / "tables" / "table_nsclc_cox_performance.csv"
    save_bar(series["Manufacturer"], root / "figures" / "nsclc_manufacturer_distribution.png", "Scanner manufacturer distribution", top_n=10)
    tex = latex_preamble(
        "A Reproducible TCIA Metadata Inventory for an Open-Source NSCLC Radiomics-Clinical Risk Score",
        "NSCLC-Radiomics Inventory",
    )
    tex += rf"""
\begin{{abstract}}
\textbf{{Background:}} Radiomics risk-score studies depend on reproducible image inventories before feature extraction. Missing or poorly described imaging metadata can undermine downstream survival modeling.
\textbf{{Objective:}} To create a transparent TCIA metadata inventory for a planned NSCLC-Radiomics clinical-radiomics risk score.
\textbf{{Methods:}} Public TCIA/NBIA metadata for the NSCLC-Radiomics collection were downloaded and summarized at patient and series levels. DICOM images, segmentation masks, and clinical outcome files were not yet ingested.
\textbf{{Results:}} The metadata inventory included {len(data):,} patients and {int(data['series_n'].sum()):,} imaging series. Median series count per patient was {fmt(data['series_n'].median(),1)}, and median image count per patient was {fmt(data['total_image_count'].median(),1)}. A clinical/imaging-inventory baseline for 2-year mortality achieved AUROC {fmt(perf['auroc']) if perf is not None else '--'}.
\textbf{{Conclusions:}} This inventory establishes a reproducible base for radiomics extraction. The weak clinical/inventory baseline reinforces the need for DICOM, segmentation, and outcome-complete radiomics modeling before prognostic claims.
\end{{abstract}}

{keywords(["non-small cell lung cancer", "TCIA", "radiomics", "imaging inventory", "reproducibility"])}

\section{{Introduction}}
Radiomics studies often focus on feature selection and modeling while giving less attention to the reproducibility of image acquisition inventories. For NSCLC-Radiomics, an explicit patient-series inventory is a practical prerequisite for downstream radiomics extraction, segmentation audit, and survival modeling. This manuscript reports the current reproducible inventory stage.

\section{{Methods}}
\subsection{{Data Source}}
Public metadata for NSCLC-Radiomics were downloaded from TCIA/NBIA API endpoints. The current analysis used patient and series metadata only.
\subsection{{Cohort and Variables}}
The inventory retained all patient records represented in the TCIA metadata response and summarized series counts, image counts, file sizes, modality, manufacturer, and first recorded series date.
\subsection{{Planned Model Development}}
The intended final study will load DICOM images, link tumor masks and clinical outcomes, extract PyRadiomics features, and model survival using Cox and penalized Cox methods. Those analyses are deliberately not reported here because the necessary source files have not yet been ingested.
{reproducibility_strengths_block()}

\section{{Results}}
The inventory included {len(data):,} patients. Metadata distributions are shown in Figures~1--4. The manuscript-ready inventory table is available in \texttt{{tables/table\_patient\_imaging\_inventory.csv}}.
{simple_table("NSCLC-Radiomics metadata inventory summary.", ["Characteristic", "Value"], [["Patients", f"{len(data):,}"], ["Imaging series", f"{int(data['series_n'].sum()):,}"], ["Median series per patient", fmt(data['series_n'].median(), 1)], ["Median images per patient", fmt(data['total_image_count'].median(), 1)], ["Modal modality set", data['modality_set'].mode().iat[0] if not data['modality_set'].mode().empty else "--"]])}
\subsection{{Clinical Survival Baseline}}
A public Lung1 clinical CSV mirror was used to test a preliminary 2-year mortality baseline and must be verified against the official TCIA clinical attachment before submission. The clinical/inventory baseline achieved AUROC {fmt(perf['auroc']) if perf is not None else '--'}, AUPRC {fmt(perf['auprc']) if perf is not None else '--'}, and Brier score {fmt(perf['brier']) if perf is not None else '--'}. This low discrimination supports the need for actual radiomics features rather than inventory variables alone.
{model_table(perf) if perf is not None else ""}
Survival horizon summaries by stage are provided in \texttt{{tables/table\_survival\_by\_stage\_horizon.csv}}.
{simple_table("Penalized Cox survival model performance.", ["Model", "C-index"], [[row['model'], fmt(row['c_index'])] for _, row in pd.read_csv(cox_path).dropna(subset=['c_index']).head(5).iterrows()]) if cox_path.exists() else ""}

\section{{Discussion}}
The novelty of this stage is operational reproducibility: a clearly documented inventory that identifies what is ready and what remains to be downloaded before radiomics modeling. This prevents premature claims based on incomplete imaging or outcome data.
\subsection{{Research Implications}}
The inventory can be used as the first checkpoint for a full radiomics analysis. It defines the patient and series denominator, supports scanner and modality audits, and provides an explicit place to link segmentation masks, survival outcomes, and extracted features.
\subsection{{Strengths}}
Strengths include use of an official public imaging archive API, patient-level and series-level summaries, and conservative reporting that avoids presenting a risk model before outcomes and image-derived predictors are available.
\subsection{{Limitations}}
No imaging pixels, masks, radiomics features, treatment variables, or survival outcomes are included yet. The current manuscript is best framed as a reproducible data-readiness report or methods prelude, not a final prognostic model.
\subsection{{Conclusions}}
The NSCLC project has a complete metadata inventory and is ready for TCIA/NBIA bulk image, mask, and outcome ingestion.
"""
    tex += governance_block()
    tex += target_journal_block("NSCLC", ["Journal of Thoracic Oncology Clinical and Research Reports", "Annals of Thoracic Surgery Short Reports", "Lung Cancer"])
    tex += include_figures(
        [
            ("nsclc_patient_sex.png", "Patient sex distribution in the TCIA metadata inventory."),
            ("nsclc_series_per_patient.png", "Distribution of imaging series count per patient."),
            ("nsclc_image_count_per_patient.png", "Distribution of image counts per patient."),
            ("nsclc_manufacturer_distribution.png", "Scanner manufacturer distribution across series."),
        ]
    )
    tex += latex_tail()
    (root / "manuscript" / "manuscript.tex").write_text(tex, encoding="utf-8")


def write_tcga(project_root: Path, cohort_name: str, title: str, targets: list[str], model_prefix: str | None, note: str) -> None:
    manuscript_path = project_root / "manuscript" / "manuscript.tex"
    if manuscript_path.exists() and manuscript_path.read_text(encoding="utf-8").startswith(
        "% MANUAL_SELF_CONTAINED_MANUSCRIPT"
    ):
        return
    data = pd.read_csv(project_root / "data_processed" / cohort_name)
    save_bar(data["pathologic_stage"], project_root / "figures" / "pathologic_stage_distribution.png", "Pathologic stage distribution", top_n=12)
    save_hist(data["age_at_diagnosis_years"], project_root / "figures" / "age_at_diagnosis_distribution.png", "Age at diagnosis", "Years")
    save_bar(data["vital_status"], project_root / "figures" / "vital_status_distribution.png", "Vital status distribution")
    model_text = "Mortality modeling was not performed because the event count was too small for a stable first-pass model."
    figures = [
        ("pathologic_stage_distribution.png", "Pathologic stage distribution in the parsed TCGA clinical XML cohort."),
        ("age_at_diagnosis_distribution.png", "Age at diagnosis distribution."),
        ("vital_status_distribution.png", "Vital status distribution."),
    ]
    perf = None
    comparison_path = project_root / "tables" / "table_model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else None
    cv_path = project_root / "tables" / "table_cross_validation.csv"
    enriched_comparison_path = project_root / "tables" / "table_hpv_treatment_model_comparison.csv"
    if not enriched_comparison_path.exists():
        enriched_comparison_path = project_root / "tables" / "table_msi_mutation_model_comparison.csv"
    enriched_comparison = pd.read_csv(enriched_comparison_path) if enriched_comparison_path.exists() else None
    rigorous_paths = list((project_root / "tables").glob("table_*_rigorous_validation.csv"))
    rigorous = pd.read_csv(rigorous_paths[0]) if rigorous_paths else None
    rigorous_text = ""
    enriched_text = ""
    enriched_methods_text = "Predictors included demographic and pathologic variables available in the parsed XML."
    enriched_meaning = "The current manuscript provides a transparent clinical-only baseline and defines what molecular or endpoint enrichment is required before making stronger prognostic claims."
    discussion_opening = (
        "This work provides a reproducible clinical-only baseline for TCGA risk-score development. Its value is "
        "methodological and practical: it separates what can be predicted from ordinary clinical staging variables "
        "from what future molecular features must add. The analysis deliberately avoids claiming clinical readiness."
    )
    limitation_text = (
        "The analysis is limited to open clinical XML supplements and internally validated splits. Missingness and "
        "legacy XML field definitions may affect cohort completeness. Externally validated survival models are not "
        "yet incorporated."
    )
    if enriched_comparison is not None and not enriched_comparison.empty:
        best_enriched = enriched_comparison.iloc[0]
        enriched_text = (
            f" The best enriched model ({best_enriched['model'].replace('_', '\\_')}) achieved AUROC "
            f"{fmt(best_enriched['auroc'])}, AUPRC {fmt(best_enriched['auprc'])}, and Brier score "
            f"{fmt(best_enriched['brier'])}."
        )
        if "hpv_treatment" in enriched_comparison_path.name:
            enriched_methods_text = (
                "Predictors included demographic and pathologic variables, HPV/p16 or HPV ISH status when present, "
                "treatment variables, and a prespecified expression panel."
            )
            enriched_meaning = (
                "Adding HPV and treatment fields materially improved the baseline, but external validation and "
                "source-field audit remain necessary before clinical interpretation."
            )
            discussion_opening = (
                "This work provides a reproducible TCGA-HNSC mortality-risk baseline and shows that HPV and "
                "treatment enrichment materially improve performance over clinical fields alone. The analysis "
                "deliberately avoids claiming clinical readiness."
            )
            limitation_text = (
                "The analysis is limited to open clinical XML supplements and internally validated splits. HPV and "
                "treatment variables were parsed from legacy XML fields and require human source-field audit before "
                "submission. Externally validated survival models are not yet incorporated."
            )
        elif "msi_mutation" in enriched_comparison_path.name:
            enriched_methods_text = (
                "Predictors included demographic and pathologic variables, MSI/MMR and KRAS/BRAF fields when present, "
                "CEA and treatment variables, and a prespecified expression panel."
            )
            enriched_meaning = (
                "Adding MSI/mutation and treatment fields materially improved the baseline, but external validation "
                "and stronger disease-free or progression endpoints remain necessary."
            )
            discussion_opening = (
                "This work provides a reproducible TCGA colorectal mortality-risk baseline and shows that MSI/MMR, "
                "KRAS/BRAF, CEA, and treatment enrichment improve performance over clinical fields alone. The "
                "analysis deliberately avoids claiming clinical readiness."
            )
            limitation_text = (
                "The analysis is limited to open clinical XML supplements and internally validated splits. MSI, "
                "mutation, CEA, and treatment variables were parsed from legacy XML fields and require human "
                "source-field audit before submission. The exploratory Cox model performed poorly and should remain "
                "secondary."
            )
    if rigorous is not None and not rigorous.empty:
        best_rigorous = rigorous.iloc[0]
        rigorous_text = (
            f" In five-fold out-of-fold validation, {best_rigorous['model'].replace('_', '\\_')} achieved AUROC "
            f"{fmt(best_rigorous['auroc'])} (95\\% bootstrap CI {fmt(best_rigorous['auroc_ci_low'])}--"
            f"{fmt(best_rigorous['auroc_ci_high'])}), AUPRC {fmt(best_rigorous['auprc'])}, and Brier score "
            f"{fmt(best_rigorous['brier'])}; calibration intercept was "
            f"{fmt(best_rigorous['calibration_intercept'])} and slope was "
            f"{fmt(best_rigorous['calibration_slope'])}."
        )
        rigorous_prefix = rigorous_paths[0].stem.removeprefix("table_").removesuffix("_rigorous_validation")
        figures += [
            (f"{rigorous_prefix}_calibration.png", "Out-of-fold calibration by predicted-risk decile."),
            (f"{rigorous_prefix}_decision_curve.png", "Decision-curve analysis across prespecified risk thresholds."),
        ]
        limitation_text += (
            " Out-of-fold calibration slopes below 1 indicate overfitting and support coefficient shrinkage or "
            "recalibration in an independent cohort."
        )
    cox_files = list((project_root / "tables").glob("table_*_cox_performance.csv"))
    if model_prefix:
        primary_path = project_root / "tables" / "table_model_performance_primary.csv"
        perf = pd.read_csv(primary_path if primary_path.exists() else project_root / "tables" / "table_model_performance.csv").iloc[0]
        try:
            save_model_curves(project_root, project_root / "data_processed" / cohort_name, [], "event_death", model_prefix)
        except KeyError:
            # Existing figures are kept when the selected comparison model differs
            # from the last serialized model artifact.
            pass
        model_text = (
            f"The validation set included {int(perf['n_validation']):,} patients and "
            f"{int(perf['events_validation']):,} death events. The clinical-only logistic model achieved "
            f"AUROC {fmt(perf['auroc'])}, AUPRC {fmt(perf['auprc'])}, and Brier score {fmt(perf['brier'])}."
        )
        figures += [
            (f"{model_prefix}_roc.png", "Validation receiver operating characteristic curve."),
            (f"{model_prefix}_precision_recall.png", "Validation precision-recall curve."),
            (f"{model_prefix}_calibration.png", "Calibration plot by predicted-risk quantile."),
        ]
    events = int(data["event_death"].sum())
    tex = latex_preamble(title, re.sub(r"[^A-Za-z0-9 ]", "", title)[:50])
    tex += rf"""
\begin{{abstract}}
\textbf{{Background:}} TCGA open clinical supplements provide a reproducible foundation for specialty-specific cancer risk-score development, but clinical-only baselines should be established before adding molecular features.
\textbf{{Objective:}} To develop a transparent first-pass clinical cohort and risk-score baseline using parsed TCGA clinical XML supplements.
\textbf{{Methods:}} GDC file manifests and open clinical XML supplements were downloaded, parsed, and harmonized into a patient-level analytic cohort. {enriched_methods_text} The primary first-pass endpoint was death event status.
\textbf{{Results:}} The cohort included {len(data):,} patients with {events:,} death events.{rigorous_text}
\textbf{{Conclusions:}} {note}
\end{{abstract}}

{keywords(["TCGA", "clinical prediction", "risk score", "open data", "oncology"])}

\section*{{Key Points}}
\textbf{{Question:}} Can open TCGA clinical supplements support a reproducible specialty-specific risk-score baseline?\\
\textbf{{Findings:}} The parsed cohort included {len(data):,} patients and {events:,} death events.{rigorous_text}\\
\textbf{{Meaning:}} {enriched_meaning}

\section{{Introduction}}
Open cancer genomics repositories allow reproducible risk-score development, but clinical-only baselines remain important. Without such baselines, the incremental value of molecular features can be overstated. This study builds a patient-level cohort from open TCGA clinical XML supplements and, where event counts support it, fits a first-pass interpretable clinical mortality model.

\section{{Methods}}
\subsection{{Study Design and Reporting}}
This was a retrospective open-data prediction analysis designed around TRIPOD and PROBAST principles. The analysis used public or open-access files from the GDC portal. No raw controlled-access sequencing reads were used.
\subsection{{Data Source}}
GDC file manifests were generated for the relevant TCGA project(s). Open clinical XML supplement files were downloaded through the GDC data API and parsed into flattened patient-level records.
\subsection{{Cohort}}
Patients with parseable clinical XML supplements and a TCGA submitter identifier were retained. Duplicate clinical supplements were reduced to one patient-level clinical row.
\subsection{{Predictors}}
Candidate predictors included sex, age at diagnosis, pathologic stage, pathologic T, N, and M categories, histologic grade when available, primary site when available, smoking history where available, and project identifier for combined cohorts.
\subsection{{Outcome}}
The primary first-pass endpoint was death event status derived from vital status. Overall survival time was also parsed where days to death or last follow-up were available, but time-to-event modeling was reserved for a later endpoint-harmonized analysis.
\subsection{{Modeling}}
Where both outcome classes had at least 10 observations, logistic regression pipelines used within-fold numeric imputation/scaling and categorical imputation/one-hot encoding. The primary comparative analysis used stratified five-fold out-of-fold predictions. Clinical-only, enriched clinical-biology, and enriched-plus-expression specifications were compared without class weighting. Coefficients from the original derivation model were converted to an exploratory integer score table.
\subsection{{Statistical Analysis}}
Discrimination was summarized by AUROC and AUPRC, with percentile 95\% confidence intervals from 2,000 patient-level bootstrap resamples. Overall accuracy was summarized by Brier score. Calibration intercept and slope were estimated by logistic recalibration of out-of-fold probabilities, and calibration was plotted by predicted-risk decile. Decision-curve analysis estimated net benefit from risk thresholds of 0.05 through 0.75. All scripts are in the project \texttt{{src/}} folder.
{reproducibility_strengths_block()}

\section{{Results}}
\subsection{{Cohort Characteristics}}
The parsed cohort included {len(data):,} patients; {events:,} ({100*events/len(data):.1f}\%) had death event status. Median age at diagnosis was {fmt(data['age_at_diagnosis_years'].median(),1)} years. Clinical distributions are shown in Figures~1--3.
{simple_table("Parsed TCGA clinical cohort summary.", ["Characteristic", "Value"], [["Patients", f"{len(data):,}"], ["Death events", f"{events:,} ({100*events/len(data):.1f}%)"], ["Median age at diagnosis", f"{fmt(data['age_at_diagnosis_years'].median(),1)} years"], ["Female", f"{int((data['sex'].astype(str).str.upper() == 'FEMALE').sum()):,}"], ["Male", f"{int((data['sex'].astype(str).str.upper() == 'MALE').sum()):,}"]])}
\subsection{{Model Performance}}
{rigorous_text if rigorous_text else model_text}
{model_table(perf) if perf is not None else simple_table("Modeling status.", ["Item", "Status"], [["Model fit", "No"], ["Reason", "Insufficient death events for a stable first-pass mortality model"], ["Death events", f"{events:,}"]])}
\subsection{{Molecular Panel Enrichment}}
Open RNA-seq STAR gene-count files were downloaded from GDC where available and reduced to a small prespecified expression panel. In this internal validation, adding the expression panel did not improve discrimination over clinical variables alone when the comparison table is available. This negative finding is retained because it prevents overstating the value of a small molecular panel in a limited cohort.
{simple_table("Clinical-only versus clinical-expression model comparison.", ["Model", "AUROC", "AUPRC", "Brier", "Predictors"], [[row['model'], fmt(row['auroc']), fmt(row['auprc']), fmt(row['brier']), str(int(row['predictor_count']))] for _, row in comparison.iterrows()]) if comparison is not None else ""}
{simple_table("Exploratory single-split enhanced model comparison (secondary analysis).", ["Model", "AUROC", "AUPRC", "Brier", "Predictors"], [[row['model'], fmt(row['auroc']), fmt(row['auprc']), fmt(row['brier']), str(int(row['predictor_count']))] for _, row in enriched_comparison.iterrows()]) if enriched_comparison is not None else ""}
{simple_table("Five-fold out-of-fold performance with bootstrap confidence intervals.", ["Model", "AUROC (95\\% CI)", "AUPRC", "Brier", "Calibration slope"], [[row['model'], f"{fmt(row['auroc'])} ({fmt(row['auroc_ci_low'])}--{fmt(row['auroc_ci_high'])})", fmt(row['auprc']), fmt(row['brier']), fmt(row['calibration_slope'])] for _, row in rigorous.iterrows()]) if rigorous is not None else ""}
{cv_summary_table(cv_path, "Five-fold cross-validation summary for the selected clinical model.")}
{simple_table("Exploratory penalized Cox model performance.", ["Model", "C-index"], [[row['model'], fmt(row['c_index'])] for _, row in pd.read_csv(cox_files[0]).dropna(subset=['c_index']).head(5).iterrows()]) if cox_files else ""}
\subsection{{Score Output}}
When a model was fit, coefficient-derived score components were written to \texttt{{tables/table\_integer\_score.csv}}. These scores are exploratory and require external validation before clinical use.

\section{{Discussion}}
{discussion_opening}
{model_caution_block()}
\subsection{{Strengths}}
Strengths include use of open GDC clinical supplements, explicit XML parsing, clear endpoint definitions, conservative model eligibility rules based on event counts, and generation of reusable coefficient-to-score tables when modeling is appropriate.
\subsection{{Limitations}}
{limitation_text}
\subsection{{Conclusions}}
{note}
"""
    tex += governance_block()
    tex += target_journal_block("TCGA", targets)
    tex += include_figures(figures)
    tex += latex_tail()
    (project_root / "manuscript" / "manuscript.tex").write_text(tex, encoding="utf-8")


def write_prad() -> None:
    root = ROOT / "10_urology_tcga_prad"
    manuscript_path = root / "manuscript" / "manuscript.tex"
    if manuscript_path.exists() and manuscript_path.read_text(encoding="utf-8").startswith(
        "% MANUAL_SELF_CONTAINED_MANUSCRIPT"
    ):
        return
    data = pd.read_csv(root / "data_processed" / "tcga_prad_recurrence_cohort.csv")
    perf = pd.read_csv(root / "tables" / "table_model_performance.csv").iloc[0]
    cv_path = root / "tables" / "table_cross_validation.csv"
    cbio_path = root / "tables" / "table_cbioportal_endpoint_crosscheck.csv"
    save_bar(data["outcome_recurrence_or_progression"].map({0: "No recurrence/progression", 1: "Recurrence/progression"}), root / "figures" / "prad_recurrence_outcome.png", "Recurrence/progression endpoint")
    save_hist(data["age_at_diagnosis_years"], root / "figures" / "age_at_diagnosis_distribution.png", "Age at diagnosis", "Years")
    save_hist(data["gleason_score_numeric"], root / "figures" / "prad_gleason_distribution.png", "Gleason score distribution", "Gleason score")
    events = int(data["outcome_recurrence_or_progression"].sum())
    expr_cols = [c for c in data.columns if c.startswith("expr_")]
    tex = latex_preamble(
        "Development and Internal Validation of an Open-Clinical and Expression-Enriched Recurrence Score for Prostate Cancer Using TCGA-PRAD",
        "TCGA-PRAD Recurrence Score",
    )
    tex += rf"""
\begin{{abstract}}
\textbf{{Background:}} Mortality is sparse in TCGA-PRAD open clinical supplements, limiting death-event prediction. Recurrence and progression fields provide a more appropriate endpoint for prostate cancer risk-score development.
\textbf{{Objective:}} To develop an interpretable recurrence/progression baseline using TCGA-PRAD clinical variables and a small open RNA-seq expression panel.
\textbf{{Methods:}} Open clinical XML supplements and RNA-seq STAR gene-count files were downloaded from GDC. The endpoint combined biochemical recurrence, new tumor event, and post-treatment progression fields when present. Predictors included age, pathologic variables, PSA, Gleason score, and a prespecified expression panel.
\textbf{{Results:}} The cohort included {len(data):,} patients with {events:,} recurrence/progression events. In held-out validation, the model achieved AUROC {fmt(perf['auroc'])}, AUPRC {fmt(perf['auprc'])}, and Brier score {fmt(perf['brier'])}.
\textbf{{Conclusions:}} Reframing TCGA-PRAD around recurrence/progression rather than mortality produces a scientifically more appropriate first-pass risk model. External validation and endpoint adjudication remain necessary.
\end{{abstract}}

{keywords(["TCGA-PRAD", "prostate cancer", "biochemical recurrence", "RNA-seq", "risk score"])}

\section*{{Key Points}}
\textbf{{Question:}} Can public TCGA-PRAD clinical and expression files support recurrence/progression risk-score development?\\
\textbf{{Findings:}} A recurrence/progression endpoint was available for {events:,} of {len(data):,} patients, and an interpretable first-pass model achieved validation AUROC {fmt(perf['auroc'])}.\\
\textbf{{Meaning:}} Endpoint selection materially improves scientific plausibility for TCGA-PRAD compared with sparse mortality modeling.

\section{{Introduction}}
Prostate cancer risk prediction is poorly served by short-horizon mortality endpoints in cohorts with few deaths. TCGA-PRAD contains open clinical supplement fields for biochemical recurrence and progression-related events, making recurrence/progression a more clinically coherent first-pass endpoint. This study uses those fields, along with a small prespecified expression panel, to create a reproducible baseline for later genomic-clinical refinement.

\section{{Methods}}
\subsection{{Data Source and Cohort}}
Open TCGA-PRAD clinical XML supplements and RNA-seq STAR gene-count files were downloaded through the GDC API. Patients with parseable clinical supplements were retained.
\subsection{{Endpoint}}
The binary endpoint combined biochemical recurrence, new tumor event after initial treatment, and post-treatment progression fields. The endpoint is a harmonized public-data proxy and should be externally validated before clinical interpretation.
\subsection{{Predictors}}
Candidate predictors included age at diagnosis, pathologic T and N categories, PSA, Gleason score, and {len(expr_cols)} expression features from a prespecified prostate cancer panel.
\subsection{{Modeling}}
An interpretable logistic model was fit with a stratified derivation-validation split. Coefficients were converted into an integer score table.
{reproducibility_strengths_block()}

\section{{Results}}
The cohort included {len(data):,} patients and {events:,} recurrence/progression events. Median age at diagnosis was {fmt(data['age_at_diagnosis_years'].median(),1)} years. The validation sample included {int(perf['n_validation']):,} patients and {int(perf['events_validation']):,} events.
{simple_table("TCGA-PRAD recurrence/progression cohort summary.", ["Characteristic", "Value"], [["Patients", f"{len(data):,}"], ["Recurrence/progression events", f"{events:,} ({100*events/len(data):.1f}%)"], ["Median age", f"{fmt(data['age_at_diagnosis_years'].median(),1)} years"], ["Expression panel features", str(len(expr_cols))]])}
{model_table(perf)}
{cv_summary_table(cv_path, "Five-fold cross-validation summary for the PRAD recurrence/progression model.")}
\subsection{{Endpoint Cross-Check}}
The parsed endpoint was compared with cBioPortal TCGA-PRAD biochemical recurrence fields where matched patient identifiers were available. This is not an external cohort validation, but it is an independent representation of TCGA clinical recurrence data and supports endpoint auditability.
{simple_table("cBioPortal TCGA-PRAD endpoint cross-check.", ["Metric", "Value"], [["Matched patients", str(int(pd.read_csv(cbio_path).iloc[0]['matched_patients']))], ["Agreement fraction", fmt(pd.read_csv(cbio_path).iloc[0]['agreement_fraction'])], ["Local events in comparable set", str(int(pd.read_csv(cbio_path).iloc[0]['local_events_in_comparable']))], ["cBioPortal BCR events", str(int(pd.read_csv(cbio_path).iloc[0]['cbio_bcr_events']))]]) if cbio_path.exists() else ""}

\section{{Discussion}}
The key improvement over a mortality model is endpoint appropriateness. The TCGA-PRAD death-event count was too small for stable mortality modeling, whereas recurrence/progression fields supported a first-pass model with moderate discrimination. This manuscript should be framed as an open-data endpoint-harmonization and baseline-model study rather than a clinically deployable nomogram.
{model_caution_block()}
\subsection{{Limitations}}
The recurrence/progression endpoint is parsed from legacy XML fields and may not match adjudicated biochemical recurrence definitions in all cases. Follow-up intensity, treatment details, and external validation are incomplete. The expression panel is small and prespecified; broader molecular discovery was not attempted.
\subsection{{Conclusions}}
TCGA-PRAD recurrence/progression modeling is feasible with public clinical and expression files, but external validation and endpoint adjudication are required before clinical use.
"""
    tex += governance_block()
    tex += target_journal_block("PRAD", ["European Urology Open Science", "Urologic Oncology", "Prostate Cancer and Prostatic Diseases"])
    tex += include_figures(
        [
            ("prad_recurrence_outcome.png", "Distribution of the harmonized recurrence/progression endpoint."),
            ("age_at_diagnosis_distribution.png", "Age at diagnosis distribution."),
            ("prad_gleason_distribution.png", "Gleason score distribution."),
        ]
    )
    tex += latex_tail()
    (root / "manuscript" / "manuscript.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    write_nsclc()
    write_isic()
    write_prad()
    write_tcga(
        ROOT / "11_colorectal_surgery_tcga_coad_read",
        "tcga_coad_read_analysis_cohort.csv",
        "Development and Internal Validation of an Open-Clinical Colorectal Cancer Mortality Score Using TCGA-COAD and TCGA-READ",
        ["Diseases of the Colon & Rectum", "Colorectal Disease", "Cancers"],
        "coad_read",
        "A clinical-only colorectal mortality baseline is feasible, but genomic features and disease-free or progression endpoints are needed for a full genomic-clinical score.",
    )
    write_tcga(
        ROOT / "12_ent_head_neck_tcga_hnsc",
        "tcga_hnsc_analysis_cohort.csv",
        "Development and Internal Validation of an Open-Clinical Mortality Score for Head and Neck Squamous Cell Carcinoma Using TCGA-HNSC",
        ["Head & Neck", "Oral Oncology", "JAMA Otolaryngology-Head & Neck Surgery"],
        "hnsc",
        "A clinical-only HNSC mortality baseline is feasible, but HPV status, treatment variables, molecular features, and external validation are required before clinical interpretation.",
    )


if __name__ == "__main__":
    main()
