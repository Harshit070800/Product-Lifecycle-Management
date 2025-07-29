# import os
# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
# from xgboost import XGBRegressor

# from preprocessing import load_data, build_pipeline, features, target_columns


# # Core Training Function
# def train_and_evaluate(regressor, X_train, y_train, X_test, y_test, scale_numerical=False, model_name="model"):
#     """Trains a given regressor (with pipeline) and returns metrics & trained model."""
#     model = build_pipeline(regressor=MultiOutputRegressor(regressor), scale_numerical=scale_numerical)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     mae = mean_absolute_error(y_test, preds)
#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     r2 = r2_score(y_test, preds)

#     # Per-target MAE and RMSE
#     target_names = target_columns
#     for i, target in enumerate(target_names):
#         target_mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
#         target_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
#         print(f"    {target}: MAE={target_mae:.4f}, RMSE={target_rmse:.4f}")

#     model_path = f"../models/{model_name.replace(' ', '_').lower()}.pkl"
#     joblib.dump(model, model_path)

#     return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R²": r2, "ModelObj": model, "Path": model_path}


# # Main Execution
# def main():
#     os.makedirs("../models", exist_ok=True)
#     os.makedirs("../outputs", exist_ok=True)

#     # Load Data
#     df = load_data()
#     X, y = df[features], df[target_columns]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Define Models
#     models = {
#         "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42), False),
#         "Extra Trees": (ExtraTreesRegressor(n_estimators=200, random_state=42), False),
#         "XGBoost": (XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42), False),
#         "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42), False),
#         "Decision Tree": (DecisionTreeRegressor(max_depth=6, random_state=42), False),
#         "Linear Regression": (LinearRegression(), True),
#         "Ridge Regression": (Ridge(alpha=1.0), True),
#         "Lasso Regression": (Lasso(alpha=0.01), True)
#     }

#     # Train & Compare
#     results = []
#     for name, (regressor, scale) in models.items():
#         print(f"\nTraining {name}...")
#         res = train_and_evaluate(regressor, X_train, y_train, X_test, y_test, scale_numerical=scale, model_name=name)
#         results.append(res)

#     # Create Results DataFrame
#     results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
#     print("\nModel Comparison:\n", results_df[["Model", "MAE", "RMSE", "R²"]])

#     # Plot Comparison
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     colors = plt.cm.tab10.colors
#     for i, metric in enumerate(["MAE", "RMSE", "R²"]):
#         axes[i].bar(results_df["Model"], results_df[metric], color=colors)
#         axes[i].set_title(f"{metric} Comparison")
#         axes[i].tick_params(axis='x', rotation=40)
#     plt.tight_layout()
#     plt.savefig("../outputs/model_comparison_metrics.png")
#     plt.close()
#     print("Comparison plot saved to ../outputs/model_comparison_metrics.png")

#     # Save Best Model
#     best_model_row = results_df.iloc[0]
#     joblib.dump(best_model_row["ModelObj"], "../models/best_model.pkl")
#     print(f"\nBest Model: {best_model_row['Model']} (R²={best_model_row['R²']:.2f}) saved to ../models/best_model.pkl")


# if __name__ == "__main__":
#     main()


import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from preprocessing import load_data, build_pipeline, features, target_columns


# Core Training Function
def train_and_evaluate(regressor, X_train, y_train, X_test, y_test, scale_numerical=False, model_name="model"):
    """Trains a given regressor (with pipeline) and returns metrics & trained model."""
    model = build_pipeline(regressor=MultiOutputRegressor(regressor), scale_numerical=scale_numerical)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Per-target metrics
    for i, target in enumerate(target_columns):
        target_mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        target_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
        print(f"    {target}: MAE={target_mae:.4f}, RMSE={target_rmse:.4f}")

    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R²": r2, "ModelObj": model}


# Main Execution
def main():
    os.makedirs("../outputs", exist_ok=True)

    # Load Data
    df = load_data()
    X, y = df[features], df[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Models
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

    # Train & Compare
    results = []
    for name, (regressor, scale) in models.items():
        print(f"\nTraining {name}...")
        res = train_and_evaluate(regressor, X_train, y_train, X_test, y_test, scale_numerical=scale, model_name=name)
        results.append(res)

    # Create Results DataFrame
    results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
    print("\nModel Comparison:\n", results_df[["Model", "MAE", "RMSE", "R²"]])

    # Plot Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.cm.tab10.colors
    for i, metric in enumerate(["MAE", "RMSE", "R²"]):
        axes[i].bar(results_df["Model"], results_df[metric], color=colors)
        axes[i].set_title(f"{metric} Comparison")
        axes[i].tick_params(axis='x', rotation=40)
    plt.tight_layout()
    plt.savefig("../outputs/model_comparison_metrics.png")
    plt.close()
    print("Comparison plot saved to ../outputs/model_comparison_metrics.png")

    

if __name__ == "__main__":
    main()
