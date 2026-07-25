# 🧬 Engineering growth-coupled metabolic biosensors for disease prognosis and diagnosis using full growth trajectories

This repository contains the full codebase used for **prognosis and diagnosis of COVID-19** based on *Escherichia coli* growth dynamics, as well as **statistical analyses using Generalized Additive Mixed Models (GAMMs)**.

---

## 📄 Reference

**Preprint**.
*Engineering growth-coupled metabolic biosensors for disease prognosis and diagnosis using full growth trajectories*. Ahavi P., Hoang A., Meyer P., Epaulard O., Le Gouellec A., Faulon J.-L.

---

## 🧠 Overview

This project explores multiple machine learning and statistical approaches to classify biological conditions from microbial growth data.

The repository includes:

* Growth parameter-based classification models
* Time-series deep learning models
* Context-aware models using Feature-wise Linear Modulation (FiLM)
* Multistrain fusion strategies
* Statistical inference using Generalized Additive Mixed Models (GAMMs)

All classification methods are evaluated under a **nested cross-validation framework** to ensure robust and unbiased performance estimates.

---

## 🧪 Python Models for Classification

### 1. Growth Parameter-Based Classification

📂 `Growth_parameter_based_classification/Two parameters/`

This module performs classification using extracted growth parameters.

#### Input

* Extracted growth parameters from *E. coli* growth curves
* Two-parameter representations of growth dynamics

#### Models

* Support Vector Machine (SVM)
* Logistic Regression
* XGBoost
* Soft-voting ensemble

#### Main script

```bash
Growth_parameter_based_classification/Two parameters/growth_parameters_main.py
```

---

### 2. Time-Series Classification

📂 `Time_series_classification/`

This module performs classification directly from raw growth curves.

#### Input

* Raw microbial growth time series

#### Models

* 1D Convolutional Neural Network (CNN1D)
* Temporal Convolutional Network (TCN)

#### Optional features

* First derivatives of growth curves
* Second derivatives of growth curves
* Channel-wise normalization

#### Scripts

```bash
Time_series_classification/time_series_main.py
Time_series_classification/time_series_model.py
```

---

### 3. FiLM-Based Context-Aware Models

📂 `FiLM/`

This module implements context-aware neural networks using **Feature-wise Linear Modulation (FiLM)**.

#### Principle

FiLM models combine:

* Growth curves as the primary input
* Growth parameters as contextual conditioning variables

The conditioning signal is used to generate feature-wise modulation parameters through a multilayer perceptron.

#### Models

* FiLM-CNN1D
* FiLM-TCN

---

### 4. Multistrain Models

📂 `Multistrain_models/`

This module implements classification strategies combining information from multiple *E. coli* strains.

### Early Fusion

* Combines multiple strains as a multi-channel input
* Supports optional FiLM conditioning per strain

### Late Fusion

* Trains independent models for each strain
* Combines predictions using a weighted soft-voting ensemble
* Learns weights within each fold to avoid data leakage

---

## 📊 Model Evaluation Strategy

All machine learning models are evaluated using a nested cross-validation strategy.

### Cross-validation design

* 5 outer folds
* 3 inner folds
* Patient-level splitting
* Shared splits across models whenever applicable

### Optimization

* Main optimization target: **balanced accuracy**
* Hyperparameter tuning performed with Optuna for time-series models

This strategy ensures that model selection and final performance estimation remain properly separated.

---

## 📈 Statistical Analysis with GAMMs

📂 `GAMM models/`

This folder contains the R scripts used for statistical analysis with **Generalized Additive Mixed Models (GAMMs)**.

---

### Model Selection

Model selection is based on:

* Akaike Information Criterion (AIC)
* Basis dimension diagnostics using `k.check()`

#### Scripts

```bash
GAMM_all_mutants_model_comparison.R
GAMM_selected_mutants_model_comparison.R
```

These scripts select the best model for each strain and classification task.

---

### Statistical Testing

#### Scripts

```bash
GAMM_all_mutants.R
GAMM_selected_mutants.R
```

These scripts are used to:

* Detect differences between biological conditions
* Identify significant time windows
* Compute confidence intervals

---

### Technical Details

Models are fitted using:

```r
mgcv::bam()
```

with `mgcv` version 1.9.3.

The statistical analysis includes:

* Simultaneous 95% confidence bands
* Pointwise variance estimation
* Time-dependent comparison between experimental conditions

---

## 📁 Repository Structure

```text
FiLM/
GAMM models/
Growth_parameter_based_classification/
Multistrain_models/
Time_series_classification/
```

Each folder contains the corresponding scripts, input data, and cross-validation split files required to reproduce the analyses.

---

## ⚠️ Notes

* RStudio-generated files such as `.Rproj.user`, `.RData`, and `.Rhistory` are excluded from version control.
* Excel files correspond to experimental datasets used in the study.
* Cross-validation splits are shared across models whenever applicable to ensure fair comparison.
* Growth parameter-based classification currently uses the `Two parameters` implementation.

---

## 📬 Contact

* **Paul Ahavi**
  📧 [paul.ahavi228@gmail.com](mailto:paul.ahavi228@gmail.com)

* **Jean-Loup Faulon**
  📧 [jean-loup.faulon@inrae.fr](mailto:jean-loup.faulon@inrae.fr)
