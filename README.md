# Predictive Change Impact Analyzer for PLM

A machine learning system to **predict the impact of Engineering Change Requests (ECRs)** in Product Lifecycle Management (PLM).  
Empowers product managers and engineers to estimate:
- **Approval Time (Days)**
- **Parts Affected** (BOM propagation)
- **Risk Score** (likelihood of delay/rejection)

---

## Project Structure

```
Product Lifecycle Management/
├── data/           # ECR dataset (CSV)
├── models/         # Saved trained models (.pkl)
├── outputs/        # Charts & metrics (e.g., model_results.png)
├── notebooks/      # Jupyter notebooks (EDA, prototyping)
└── src/            # Source code (modular Python scripts)
```

---

## Features

- **Full ML Pipeline:** Preprocessing, model training, evaluation, and prediction.
- **Multiple Regression Models:** Lasso, Ridge, Gradient Boosting, XGBoost, Stacking, etc.
- **Automated Hyperparameter Tuning:** Uses Optuna for joint and dedicated tuning.
- **Model Selection & Saving:** Automatically saves the best model(s) based on R².
- **Visual Results:** Generates comparison plots for MAE, RMSE, and R² scores.
- **Streamlit Dashboard:** Interactive UI for ECR impact prediction.
- **Explainability:** Feature importance visualization.
- **Clean, Modular Codebase:** Easy to extend and maintain.

---

## Workflow

### 1. Data Exploration
- Run EDA notebook:
  ```bash
  jupyter notebook notebooks/01_EDA.ipynb
  ```

### 2. Model Training & Tuning
- **Optuna-based tuning and evaluation:**
  ```bash
  python src/optuna_tuning.py
  ```
  - Trains and tunes Lasso, Ridge, GBR, XGB, and Stacking models.
  - Saves best model(s) to `models/`
  - Saves comparison chart to `outputs/model_results.png`

- **Legacy grid search (optional):**
  ```bash
  python src/hyperparameter_tuning.py
  ```

### 3. Model Comparison (optional)
- Compare multiple models and auto-save the best:
  ```bash
  python src/train_and_compare.py
  ```

### 4. Prediction for New ECRs
- **CLI Prediction:**
  ```bash
  python src/predict.py
  ```
- **Streamlit Dashboard:**
  ```bash
  streamlit run src/predict_dashboard.py
  ```
  - Interactive sliders for all input features
  - Outputs predictions for Approval Time, Parts Affected, and Risk Score

### 5. Feature Importance (optional)
- Generate feature importance plot:
  ```bash
  python src/feature_importance.py
  ```

---

## Setup & Installation

1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd Product\ Lifecycle\ Management
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Outputs

- **Best model(s):** `models/`
- **Results charts:** `outputs/model_results.png` (side-by-side MAE, RMSE, R²)
- **Feature importance:** `outputs/feature_importance.png`
- **Streamlit predictions:** via browser UI

---

## Key Files & Scripts

- `src/optuna_tuning.py` — Main script for model tuning, evaluation, and visualization
- `src/predict_dashboard.py` — Streamlit dashboard for interactive predictions
- `src/preprocessing.py` — Data loading and pipeline construction
- `src/train_and_compare.py` — Compare and save best models (optional)
- `src/hyperparameter_tuning.py` — Grid search tuning (legacy, optional)
- `data/ecr_data.csv` — Main dataset

---

## Notes

- All outputs and models are saved relative to the project root.
- The system uses a synthetic dataset; swap in your real ECR data as needed.
- The Streamlit dashboard expects pre-trained models in the `models/` directory.
- For troubleshooting, check the console output for detailed logs and error messages.

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the maintainer.

---