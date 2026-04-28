# 🧬 Growth-Based Classification of COVID-19 Using *E. coli* Metabolic Sensors

This repository contains the full codebase used for **prognosis and diagnosis of COVID-19** based on *Escherichia coli* growth dynamics, as well as **statistical analyses using Generalized Additive Mixed Models (GAMMs)**.

---

## 📄 Reference

**Preprint**
*Deep learning-based prognosis and diagnosis using* *Escherichia coli* *growth-coupled metabolic sensors*

---

## 🧠 Overview

This project explores multiple machine learning and statistical approaches to classify biological conditions from microbial growth data:

* Growth parameter-based models
* Time-series deep learning models
* Context-aware models (FiLM)
* Multistrain fusion strategies
* Statistical inference using GAMMs

All methods are evaluated under a **nested cross-validation framework** to ensure robust and unbiased performance estimates.

---

## 🧪 Python Models (Classification)

### 1. Growth Parameter-Based Classification

📂 `Growth_parameter_based_classification/`

* Input: extracted growth parameters

* Models:

  * Support Vector Machine (SVM)
  * Logistic Regression
  * XGBoost
  * Soft-voting ensemble

* Main script:
  `growth_parameters_main.py`

---

### 2. Time-Series Classification

📂 `Time_series_classification/`

* Input: raw growth curves

* Models:

  * 1D Convolutional Neural Network (CNN1D)
  * Temporal Convolutional Network (TCN)

* Optional features:

  * First & second derivatives
  * Channel-wise normalization

* Scripts:

  * `time_series_main.py`
  * `time_series_model.py`

---

### 3. FiLM-Based Context-Aware Models

📂 `FiLM/`

* Combines:

  * Growth parameters → conditioning signal
  * Growth curves → primary input

* Mechanism:

  * Feature-wise Linear Modulation (FiLM)
  * Parameters generated via MLP

* Models:

  * FiLM-CNN1D
  * FiLM-TCN

---

### 4. Multistrain Models

📂 `Multistrain_models/`

#### Early Fusion

* Combines multiple strains as multi-channel input
* Optional FiLM conditioning per strain

#### Late Fusion

* Independent models per strain
* Weighted soft-voting ensemble
* Weights learned per fold to avoid leakage

---

## 📊 Model Evaluation Strategy

* Nested cross-validation:

  * 5 outer folds
  * 3 inner folds
* Patient-level splitting
* Shared splits across all models
* Optimization target: **balanced accuracy**
* Hyperparameter tuning:

  * Optuna (for time-series models)

---

## 📈 Statistical Analysis (R - GAMMs)

📂 `GAMM models/`

### Model Selection

* Based on **Akaike Information Criterion (AIC)**
* Basis dimension tuning via `k.check()`

### Scripts

* `GAMM_all_mutants_model_comparison.R`
* `GAMM_selected_mutants_model_comparison.R`

→ Select best model per strain and classification task

* `GAMM_all_mutants.R`
* `GAMM_selected_mutants.R`

→ Perform statistical testing:

* Detect differences between conditions
* Identify significant time windows
* Compute confidence intervals

### Technical details

* Fitted using `mgcv::bam()` (v1.9.3)
* Includes:

  * Simultaneous 95% confidence bands
  * Pointwise variance estimation

---

## 📁 Repository Structure

```
FiLM/
GAMM models/
Growth_parameter_based_classification/
Multistrain_models/
Time_series_classification/
```

Each folder contains:

* Model scripts
* Data inputs (parameters or time series)
* Cross-validation splits

---

## ⚠️ Notes

* RStudio-generated files (`.Rproj.user`, `.RData`, `.Rhistory`) are excluded from version control
* Excel files correspond to experimental datasets used in the study
* Splits are shared across all models to ensure fair comparison

---

## 📬 Contact

**Paul Ahavi**
📧 [paul.ahavi@inrae.fr](mailto:paul.ahavi@inrae.fr)
