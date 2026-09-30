# Motor Insurance Claim Risk Modelling

Risk-based pricing pipeline for motor insurance claim frequency using GLM, GAM, penalised regression and XGBoost.

## Overview

This project models the probability of a motor insurance claim on a portfolio of **58,592 policies** (claim rate 6.4%).  
It answers six research questions covering safety features, policy duration, vehicle characteristics, regional effects and the construction of an individual risk score for premium differentiation.

**Best results (hold-out test set)**  
- GLM / GAM ≈ **AUC 0.60** (Gini 0.21)  
- XGBoost (with engineered features) ≈ **AUC 0.68** (Gini 0.35)

## Research Questions

1. Impact of vehicle safety features and engine characteristics on claim probability  
2. Identification of high-risk policy segments  
3. Effect of policy duration (subscription length)  
4. Construction of an individual risk score for premium differentiation  
5. Which vehicle characteristics contribute most to claim risk  
6. Regional factors vs vehicle characteristics

## Data

- 58,592 policies, 41 original variables  
- Target: `claim_status` (binary)  
- Key drivers: subscription length, vehicle age, region density, safety flags, vehicle specifications  

Place the file `Insurance claims data.csv` in the project root.

## Feature Engineering

- PCA vehicle score (max power, displacement, gross weight)  
- Binary safety flags (`is_esc`, `is_tpms`, `is_brake_assist`, …)  
- Lumped high-cardinality factors (fuel, segment, model, region, engine type)  
- Median / mode imputation to retain complete cases  

## Models

| Model              | Test AUC | Gini  | Notes                          |
|--------------------|----------|-------|--------------------------------|
| GLM                | ~0.61    | 0.22  | Interpretable baseline         |
| Ridge / Lasso / EN | ~0.61    | 0.22  | Penalised                      |
| GAM                | ~0.61    | 0.23  | Smooth effects                 |
| XGBoost            | ~0.68    | 0.35  | Best discrimination            |
| Ensemble           | ~0.66    | 0.32  | Average of GLM + GAM + Ridge + XGB |

## Validation

- Stratified 70/30 train-test split  
- Metrics: AUC, Gini, Brier score, lift tables, calibration  
- SHAP values for feature importance (XGBoost)  
- Risk bands with 2.7× difference in observed claim rates (3.6% → 9.6%)

## Project Structure
├── Insurance claims data.csv
├── motor_insurance_pipeline.R      # main modelling script


## How to Run
# Required packages
install.packages(c("tidyverse", "pROC", "glmnet", "mgcv", 
                   "xgboost", "scales", "forcats"))

# Run the full pipeline
source("motor_insurance_pipeline.R")

Key Findings

Subscription length is the strongest single predictor
Vehicle age is the dominant vehicle characteristic (negative association)
Safety features add only marginal lift once collinearity is controlled
Vehicle characteristics explain more than region density; both explain < 1% of deviance
XGBoost improves ranking power; GLM / ensemble remain better calibrated for pricing

Limitations

Frequency model only (no severity)
No mileage, claims history or behavioural data
Modest overall discrimination (ceiling ≈ 0.68 on this dataset)




