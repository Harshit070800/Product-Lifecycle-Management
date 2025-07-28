# Predictive Change Impact Analyzer in PLM

Objective

The goal of this project is to build a machine learning system that predicts the impact of Engineering Change Requests (ECRs) in a Product Lifecycle Management (PLM) system.
It helps product managers and engineers estimate:

    Approval Time (in days)

    Parts Affected (BOM propagation impact)

    Risk Score (likelihood of delay/rejection)



# Project Architecture

PLM_Impact_Analyzer/
├── data/                  # Synthetic ECR dataset
├── models/                # Saved trained models
├── outputs/               # Charts & metrics
├── notebooks/             # Jupyter notebooks for EDA & training
└── src/                   # Source code (modular)


# Features

Synthetic dataset with realistic correlations between features.
EDA Notebook for data exploration and visualization.
Multiple ML Models (Random Forest, XGBoost, Gradient Boosting, Linear, Ridge, Lasso, Decision Tree).
Automatic Model Selection – saves the best-performing model.
Explainability – feature importance plots for interpretability.
Prediction Script – takes new ECR input and predicts outcomes.


# Workflow

Exploratory Data Analysis (EDA)

    Run notebooks/01_EDA.ipynb to analyze patterns in data.

Model Training

    Train individual models using:

    python3 src/train_model.py

Compare multiple models and auto-save the best:

    python3 src/model_comparison.py

Feature Importance

    Generate feature importance plot for the best model:

    python3 src/feature_importance.py

Prediction for New ECRs

    Use the saved model to predict:

    python3 src/predict.py


### How to Run

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # (on Linux/Mac)
venv\Scripts\activate      # (on Windows)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training & evaluation
python3 src/model_comparison.py

# 4. Predict new ECR impact
python3 src/predict.py