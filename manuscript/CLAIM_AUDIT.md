# ISIC Manuscript Claim Audit

Last verified: 2026-06-26

## Verified Directly From Local Data

- Collection 66 contains 10,015 image records, 7,470 unique nonmissing lesion identifiers, and 1,824 malignant labels.
- Malignant labels comprise 1,113 melanoma, 514 basal cell carcinoma, and 197 squamous cell carcinoma records.
- Age, sex, anatomic-site, diagnosis-confirmation, and missingness counts in the manuscript match the processed cohort.
- Primary internal results come from five-fold lesion-grouped out-of-fold predictions.
- Combined grouped AUROC is 0.828750; AUPRC is 0.431405; Brier score is 0.175069.
- Internal and external confidence intervals use 2,000 lesion-cluster resamples of fixed predictions.
- Collection 67 contains 1,512 evaluated images and 288 events; collection 73 contains 193 and 44.
- Collection 470 contains 7,900 evaluated images after 1,910 exact overlaps were removed, with 4,112 malignant labels.
- Collection 470 has zero diagnosis-family/outcome mapping disagreements.
- Every native LaTeX chart coordinate was checked against a derived table.

## Verified From Source Code

- Numeric median imputation and scaling.
- Categorical most-frequent imputation and one-hot encoding.
- Balanced logistic-regression class weights.
- Five-fold `StratifiedGroupKFold` using lesion identifier and seed 20260626.
- Thumbnail resize to 128 by 128 pixels.
- RGB and HSV summaries, eight-bin RGB histograms, and Canny thresholds 80/160.
- External models are trained on collection 66 and applied without adaptation.

## Verified Against Primary Sources

- ISIC collection 66 is the 2018 Task 3 training set; 67 is test; 73 is validation.
- HAM10000 contains 10,015 images from Austrian and Australian sources and includes repeated lesion views.
- Collection 470 is named ISIC Balanced and points to Cassidy et al.'s duplicate-removal/dataset-analysis paper.
- TRIPOD+AI and PROBAST+AI bibliographic details match the BMJ publications.

## Corrections Made During Audit

- Replaced false language saying lesion identifiers were unavailable. They were present in raw metadata but omitted by the original parser.
- Replaced random-image validation as the primary analysis with lesion-grouped validation.
- Reclassified collections 67 and 73 as ISIC challenge test/validation partitions from one MILK attribution source, not two independent external cohorts.
- Disclosed balanced class weighting.
- Clarified that bootstrap intervals are conditional on fixed predictions and omit model-development and patient-cluster uncertainty.
- Softened the collection 470 mechanism from asserted cross-source reversal to severe source/composition shift with unresolved causal mechanism.

## Claims That Remain Interpretive

- The source/composition-shift explanation for collection 470 is supported by prevalence, attribution, saturation, and source-stratified results, but its precise causal mechanism is not identified.
- Clinical implications are reasoned recommendations, not measured clinical outcomes.
- Acceptance estimates are subjective strategy estimates and are not manuscript evidence.

## Unresolved Before Submission

- Patient-level grouping remains impossible with the current export.
- No independent clinical or institutional validation cohort is implemented.
- No calibration intercept/slope or recalibration analysis is reported for ISIC.
- No skin-tone variable is available.
- No public DOI, institutional ethics determination, dermatologist approval, or statistician sign-off exists yet.

Run:

```powershell
python src\verify_manuscript_claims.py
```
