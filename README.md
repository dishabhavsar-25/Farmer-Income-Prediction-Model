# KDAG Intra Hackathon '26 - Farmer Income Prediction (Bronze)

This repository contains the code and methodology for our **Bronze-winning** solution at the Kharagpur Data Analytics Group (KDAG) Intra Hackathon 2026.

## Problem Overview

The objective of this competition was to develop a robust predictive model to estimate the annual income of farmers. This task addresses a critical real-world challenge: evaluating the economic potential and creditworthiness of the **"unbanked" population** who lack traditional credit histories.

Our dataset spanned 106 features across diverse domains, including demographics, land & agriculture profiles, environmental factors (rainfall, temperature), and local economic indicators.

## Repository Structure

- `team_3_eda.ipynb`: Exploratory Data Analysis highlighting data distribution, feature correlation, and target skewness.
- `team_3_datapreprocessing_fixed.ipynb`: Complete data cleaning, feature engineering, and data transformation pipeline.
- `team_3_model.ipynb`: Model training, hyperparameter optimization using Optuna, and final ensemble generation.
- `Team-3.pdf`: Our final presentation detailing insights, methodology, and future roadmaps.

## Model Architecture & "Real-World" Tuning

Real-world agricultural and economic data is notoriously noisy, incomplete, and highly skewed. To move beyond baseline models and build a solution closely aligned with real-world complexities, we implemented several specific tuning strategies:

### 1. Handling Extreme Target Skewness

During EDA, we discovered a highly skewed income distribution (Median: 9.5L vs. Max: 8Cr). Because our primary evaluation metric was **MAPE (Mean Absolute Percentage Error)**, failing to address this would cause the model to overly penalize predictions on lower incomes while being distorted by extreme outliers.

- **Tuning:** We applied **Log Transformations** (`np.log1p`) to the target variable before training to compress the scale and stabilize variance. Predictions were later reversed using exponential transformations (`np.expm1`).

### 2. Robust Feature Engineering

- **Categorical Handling:** Instead of standard one-hot encoding which would explode dimensionality given the geographic variables (Villages, Mandis), we utilized **Target Encoding** to map categorical impact directly to historical income trends.
- **Dimensionality Reduction:** Applied **PCA (Principal Component Analysis)** alongside Z-score scaling to handle highly collinear environmental variables (like localized temperature and rainfall metrics), isolating the core signals.

### 3. Hyperparameter Optimization via Optuna

- **Tuning:** We utilized **Optuna** to perform Bayesian optimization across dozens of trials. Instead of optimizing for standard RMSE, we specifically designed the Optuna objective function to minimize our target metric (**MAPE**) on the validation folds, ensuring the model learned to optimize the exact metric it would be judged on.

### 4. XGBoost + CatBoost Ensembling

- No single model captures all complexities. Tree-based models handle tabular data best, but they make different types of localized errors.
- **Tuning:** We built a weighted ensemble of **XGBoost** and **CatBoost**. CatBoost inherently excelled at handling the remaining categorical data combinations, while XGBoost aggressively minimized the continuous feature errors. The weighted average of their predictions smoothed out variance, resulting in a much stronger generalization on unseen data.

## Final Results

Our rigorous approach to handling real-world data skewness and metric-aligned optimization yielded the following results, securing us a **Podium Finish (Bronze)** out of numerous competing teams:

- **Validation MAPE (Total Income):** 17.24%
- **Test MAPE (Total Income):** 23.24%
- **Test MAE:** ~3.59 Lakhs

## Team 3

- Disha Bhavsar
- Ahan Porwal
- Sabal Agarwal
- Dhruv Bisht
- Jatin Dhiman
- Priyanshu Bansal

*Developed for the KDAG Intra Hackathon 2026 at IIT Kharagpur.*
