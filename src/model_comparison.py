import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from preprocessing import load_data, build_pipeline, features


# Load Dataset

df = load_data()
X, y = df[features], df['Approval_Time_Days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define Models to Compare

models = {
    "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42), False),
    "Extra Trees": (ExtraTreesRegressor(n_estimators=200, random_state=42), False),
    "XGBoost": (XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42), False),
    "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42), False),
    "Decision Tree": (DecisionTreeRegressor(max_depth=6, random_state=42), False),
    "Linear Regression": (LinearRegression(), True),
    "Ridge Regression": (Ridge(alpha=1.0), True),
    "Lasso Regression": (Lasso(alpha=0.01), True)
}


# Train & Evaluate Models

results = []
os.makedirs("../models", exist_ok=True)

for name, (regressor, scale_numerical) in models.items():
    print(f"\nTraining {name} ...")
    model = build_pipeline(regressor=regressor, scale_numerical=scale_numerical)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Save individual model
    model_path = f"../models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_path)

    results.append((name, mae, rmse, r2, model, model_path))

# Convert to DataFrame and sort
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²", "ModelObj", "Path"]).sort_values(by="R²", ascending=False)


# Display Results

print("\nModel Comparison:\n", results_df[["Model", "MAE", "RMSE", "R²"]])


# Visualization (MAE, RMSE, R²)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = plt.cm.tab10.colors

for i, metric in enumerate(["MAE", "RMSE", "R²"]):
    axes[i].bar(results_df["Model"], results_df[metric], color=colors)
    axes[i].set_title(f"{metric} Comparison")
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=40)

plt.tight_layout()
os.makedirs("../outputs", exist_ok=True)
plt.savefig("../outputs/model_comparison_metrics.png")
plt.close()
print("Comparison plot saved to outputs/model_comparison_metrics.png")


# Auto-select Best Model

best_model_row = results_df.iloc[0]
best_model = best_model_row["ModelObj"]
best_model_name = best_model_row["Model"]
best_model_path = "../models/best_model.pkl"

joblib.dump(best_model, best_model_path)
print(f"\nBest Model: {best_model_name} (R²={best_model_row['R²']:.2f}) saved to {best_model_path}")
