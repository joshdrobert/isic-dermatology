# TRIPOD+AI Checklist for the ISIC Manuscript

Last updated: 2026-06-26

This working checklist must be reconciled against the journal's official TRIPOD+AI form before submission.

| Reporting domain | Status | Manuscript location or required action |
| --- | --- | --- |
| Title identifies prediction-model study | Complete | Title |
| Structured abstract | Complete | Abstract |
| Clinical context and rationale | Complete | Introduction |
| Intended use and target population | Complete | Introduction and Discussion; explicitly nonclinical benchmark |
| Data source and collection dates | Partial | ISIC collections identified; API retrieval date must be added |
| Eligibility criteria | Complete | Methods: Cohort Construction |
| Unit of analysis | Complete | Image prediction with lesion-grouped internal validation |
| Participant/record flow | Partial | Counts and overlap reported; add a formal flow diagram if requested |
| Outcome definition | Complete | Methods: Outcome |
| Predictor definition and timing | Complete | Methods: Predictors and Missing Data |
| Sample size rationale | Partial | Census of collection 66; no prospective power calculation |
| Missing-data handling | Complete | Methods: Predictors and Missing Data |
| Model specification | Complete | Methods: Model Development |
| Internal validation | Complete | Five-fold lesion-grouped cross-validation; random-image split retained as sensitivity analysis |
| Held-out challenge evaluation | Complete | Collections 67 and 73; same MILK attribution and post hoc selection disclosed |
| Stress testing | Complete | Collection 470 retained and audited |
| Performance measures | Complete | AUROC, AUPRC, Brier score, bootstrap intervals |
| Calibration | Partial | Brier score reported; calibration-in-the-large and slope not yet available |
| Subgroup/fairness assessment | Partial | Sex and anatomic site; skin-tone data unavailable |
| Model output availability | Complete locally | Scripts, predictions, tables, and model artifacts retained |
| Public code/data archive | Complete | Public GitHub repository created and updated |
| Ethics determination | Complete | Secondary analysis exempt from IRB review |
| Registration/protocol | Not done | Analysis was not prospectively registered; disclose as stated |
| Funding and conflicts | Complete | Disclosures added to manuscript |
| Patient and public involvement | Complete | None; stated in manuscript |

## Submission-Critical Open Items

1. Add exact API retrieval dates.
2. Obtain institutional ethics/exemption documentation.
3. Deposit code and derived outputs in a permanent public archive (Complete).
4. Complete author, funding, conflict, and contribution statements (Complete).
5. Have a dermatologist and statistician approve the clinical and statistical interpretation.
