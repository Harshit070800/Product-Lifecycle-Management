import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from preprocessing import load_data, build_pipeline, features

# ===========================
# Load Dataset
# ===========================
df = load_data()
X = df[features]
y = df['Approval_Time_Days']  # tune for approval time prediction

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# Define Models & Params
# ===========================
param_grids = {
    "RandomForest": {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "regressor__n_estimators": [100, 300],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5, 7],
        "regressor__subsample": [0.8, 1.0]
    },
    "GradientBoosting": {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5]
    }
}

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

# ===========================
# Run Grid Search for Each Model
# ===========================
results = []
for name, model in models.items():
    print(f"\n🔍 Tuning {name}...")
    
    pipeline = build_pipeline(regressor=model)
    grid = GridSearchCV(pipeline, param_grids[name], cv=3, n_jobs=-1, scoring='r2')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Best {name}: {grid.best_params_}")
    print(f"📊 MAE: {mae:.2f}, R²: {r2:.2f}")
    
    results.append((name, mae, r2, grid.best_params_, best_model))

# ===========================
# Save Best Model
# ===========================
results_df = pd.DataFrame(results, columns=["Model", "MAE", "R²", "BestParams", "ModelObj"]).sort_values(by="R²", ascending=False)
print("\n=== Hyperparameter Tuning Results ===\n", results_df[["Model", "MAE", "R²", "BestParams"]])

best_model = results_df.iloc[0]["ModelObj"]
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/best_tuned_model.pkl")
print(f"\n🏆 Best Tuned Model saved to ../models/best_tuned_model.pkl")
