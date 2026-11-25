
# Explaining Kernel-Based Group Outlier Detectors with Shapley Values

Research conducted at **CISPA Helmholtz Center for Information Security**, Saarbrücken, Germany.

**Researcher:** Swathi Suhas  
**Supervisors:** Dr. Siu Lun Chau, Dr. Krikamol Muandet

## Overview

Understanding why a machine learning model identifies an observation as anomalous is essential for building interpretable and trustworthy systems. While existing explainability methods primarily focus on individual data points, explaining anomalies at the **group or distribution level** remains challenging.

This project introduces **OCSMM-SHAP**, a Shapley value-based explanation framework for kernel-based group outlier detection using One-Class Support Measure Machines (OCSMM).

The method extends RKHS-based SHAP explanations to distributional inputs, enabling feature attribution directly within the Reproducing Kernel Hilbert Space (RKHS), without relying on surrogate models.

## Key Contributions

- **OCSVM-SHAP:** Applied the formulation to instance-level anomaly detection using One-Class Support Vector Machines.
- **OCSMM-SHAP:** Developed a distribution-level Shapley explanation framework using kernel mean embeddings and conditional mean embeddings to explain group-level anomalies.
- **OCSMM-SHAP-IQ:** Incorporated Shapley Interaction Indices to explain higher-order feature interactions contributing to anomalous group behavior.

## Experiments

The framework was evaluated on the following datasets:

- Forest CoverType
- Handwritten Letter Recognition
- Wisconsin Breast Cancer
- Synthetic Gaussian Distributions
- Synthetic School Grades
- Iris
- Wine Quality

The experiments demonstrate that the proposed methods can identify relevant anomalous features and capture feature interactions, including cases where groups differ primarily in their correlation structure.

## Repository Structure

```text
Outlier-Explanation/
├── src/
├── OCSMM-SHAP-IQ_SyntheticData.ipynb
├── OCSMM-SHAP_AdditionalExp1.ipynb
├── OCSMM-SHAP_AdditionalExp2.ipynb
├── OCSMM-SHAP_AdditionalExp3.ipynb
├── OCSMM-SHAP_ForestCoverType.ipynb
├── OCSMM-SHAP_SchoolGrade.ipynb
└── OCSVM-SHAP_AllExperiements.ipynb
```

The notebooks contain experiments for instance-level and group-level anomaly explanations, including controlled synthetic evaluations.


