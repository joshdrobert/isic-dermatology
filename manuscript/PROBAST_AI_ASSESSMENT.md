# PROBAST+AI-Oriented Risk-of-Bias Assessment

Last updated: 2026-06-26

This is an author self-assessment, not an independent PROBAST+AI rating.

| Domain | Judgment | Basis |
| --- | --- | --- |
| Participants and data sources | High concern | Archive sample; lesion grouping is available but patient grouping is not; heterogeneous evaluation sources |
| Predictors | Moderate concern | Predictors are reproducible, but file size and image dimensions may encode acquisition shortcuts |
| Outcome | High concern | Archive-derived binary label with heterogeneous confirmation methods; not a clinical decision or prospective endpoint |
| Analysis | Moderate-to-high concern | Primary validation is lesion-grouped with lesion-cluster bootstrap, but patient grouping and full model-development uncertainty are unavailable; calibration is incomplete |
| External evaluation applicability | High concern | Challenge validation/test partitions share one attribution source and are not equivalent to routine dermatology; collection selection was post hoc |
| Overall applicability to clinical diagnosis | High concern | No prospective workflow, dermatologist comparison, threshold analysis, skin-tone assessment, or clinical-impact evaluation |

## Sensitivity Language Required in the Manuscript

- Performance is lesion-disjoint but not patient-disjoint and may remain optimistic.
- Bootstrap confidence intervals account for lesion clustering but not unknown patient clustering or model-development variation.
- Collection 470 is a stress test, not a label-compatible confirmatory cohort.
- curation, attribution, and prevalence shifts can dominate aggregate discrimination.
- The model is not intended for diagnosis, triage, or biopsy decisions.

## Analyses Needed to Lower Risk of Bias

1. Patient-grouped resampling in a cohort with patient identifiers.
2. Institutionally independent, prospectively selected validation.
3. Full calibration assessment with uncertainty.
4. Prespecified subgroup analysis including skin tone where available.
5. Clinical comparator and decision-threshold study.
