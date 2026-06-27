# Development of an Open-Source Image-Clinical Skin Lesion Escalation Score Using ISIC and HAM10000

## Abstract

### Background
Dermatoscopic image archives enable reproducible development of lesion risk stratification tools.

### Objective
To develop an interpretable lesion escalation score using ISIC collection 66 metadata, with future extension to image-derived features.

### Methods
We built a 10,015-image metadata cohort from ISIC collection 66. The primary first-pass outcome was malignant or high-risk lesion label status. Predictors included age, sex, anatomic site, melanocytic status, image megapixels, and file size. A logistic model was fit with a held-out validation split and converted to an integer score table.

### Results
The metadata-only model was evaluated on 2,504 validation images and achieved AUROC 0.777, AUPRC 0.396, and Brier score 0.197.

### Conclusions
The metadata-only score provides a reproducible baseline. Full image download and CNN-derived risk features are the next step.

## Introduction

## Methods

### Data Source
ISIC collection 66 metadata were downloaded through the ISIC Archive API. Full image files were not downloaded in this pass.

### Study Population
All images represented in the collection 66 metadata export were included.

### Predictors and Outcome
Predictors included age, sex, anatomic site, melanocytic status, and image metadata. The binary outcome was malignant/high-risk label status derived from diagnosis fields.

### Model Development
A logistic regression model with preprocessing for numeric and categorical variables was trained and validated using a random held-out split.

## Results

Generated outputs include the processed cohort, missingness table, metadata characteristics, outcome-by-diagnosis table, model performance, and integer score table.

## Discussion

This baseline should be interpreted as metadata-only. It is useful for benchmarking but not a final clinical image model.

## Limitations

Diagnosis labels are archive-derived, image pixels are not yet modeled, and external validation is not yet performed.

## Conclusion

The dermatology project now has a reproducible metadata cohort and first-pass lesion escalation model.

