import optuna
import numpy as np
import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from preprocessing import load_data, build_pipeline, features, target_columns
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("../outputs", exist_ok=True)

def save_results_image(results, filename="../outputs/model_results.png"):
    df = pd.DataFrame(results, columns=["Model", "R²", "MAE", "RMSE"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = range(len(df["Model"]))

    # MAE plot
    axes[0].bar(x, df["MAE"], color="skyblue")
    axes[0].set_title("MAE Comparison")
    axes[0].set_ylabel("MAE")

    # RMSE plot
    axes[1].bar(x, df["RMSE"], color="orange")
    axes[1].set_title("RMSE Comparison")
    axes[1].set_ylabel("RMSE")

    # R² plot
    axes[2].bar(x, df["R²"], color="green")
    axes[2].set_title("R² Comparison")
    axes[2].set_ylabel("R²")

    # Set x-ticks and labels for all plots
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Model results image saved to {os.path.abspath(filename)}")

# Objective Function 
def objective(trial, X, y):
    model_name = trial.suggest_categorical("model", ["lasso", "ridge", "gbr", "xgb"])

    if model_name == "lasso":
        alpha = trial.suggest_float("lasso_alpha", 1e-4, 100.0, log=True)
        reg = MultiOutputRegressor(Lasso(alpha=alpha, max_iter=10000, random_state=42))

    elif model_name == "ridge":
        alpha = trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)
        reg = MultiOutputRegressor(Ridge(alpha=alpha, max_iter=10000, random_state=42))

    elif model_name == "gbr":
        n_estimators = trial.suggest_int("gbr_n_estimators", 50, 200)
        max_depth = trial.suggest_int("gbr_max_depth", 2, 5)
        learning_rate = trial.suggest_float("gbr_learning_rate", 1e-4, 0.1, log=True)
        subsample = trial.suggest_float("gbr_subsample", 0.6, 1.0)
        min_samples_split = trial.suggest_int("gbr_min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("gbr_min_samples_leaf", 1, 8)
        reg = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        ))

    elif model_name == "xgb":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
        max_depth = trial.suggest_int("xgb_max_depth", 2, 5)
        learning_rate = trial.suggest_float("xgb_learning_rate", 1e-4, 0.1, log=True)
        subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
        reg_alpha = trial.suggest_float("xgb_reg_alpha", 0.0, 2.0)
        reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.0, 2.0)
        min_child_weight = trial.suggest_int("xgb_min_child_weight", 1, 8)
        reg = MultiOutputRegressor(XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            verbosity=0,
            n_jobs=1
        ))
    else:
        raise ValueError("Unknown model type.")

    pipeline = build_pipeline(regressor=reg)
    scores = cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=5)
    return np.mean(scores)


# ===================== Hyperparameter Tuning =====================
def tune_and_select_best(X, y, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    print("Best trial:", study.best_trial)
    print("Best model and parameters:", study.best_trial.params)
    return study.best_trial


# ===================== Main Execution =====================
if __name__ == "__main__":
    print("\n===== Optuna Model Selection: Lasso, Ridge, GBR, XGB =====\n")

    df = load_data()
    X = df[features]
    y = df[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Joint search for all models
    best_trial = tune_and_select_best(X_train, y_train, n_trials=50)
    print("\nBest model:", best_trial.params['model'])
    print("Best parameters:", best_trial.params)

    # 2. Explicit Optuna for Ridge
    def ridge_objective(trial):
        alpha = trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)
        reg = MultiOutputRegressor(Ridge(alpha=alpha, max_iter=10000, random_state=42))
        pipeline = build_pipeline(regressor=reg)
        scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
        return np.mean(scores)

    ridge_study = optuna.create_study(direction="maximize")
    ridge_study.optimize(ridge_objective, n_trials=30)
    print("\nOptuna Ridge best parameters:", ridge_study.best_params)

    # 3. Explicit Optuna for GradientBoosting
    def gbr_objective(trial):
        n_estimators = trial.suggest_int("gbr_n_estimators", 50, 200)
        max_depth = trial.suggest_int("gbr_max_depth", 2, 5)
        learning_rate = trial.suggest_float("gbr_learning_rate", 1e-4, 0.1, log=True)
        subsample = trial.suggest_float("gbr_subsample", 0.6, 1.0)
        min_samples_split = trial.suggest_int("gbr_min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("gbr_min_samples_leaf", 1, 8)
        reg = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
        pipeline = build_pipeline(regressor=reg)
        scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
        return np.mean(scores)

    gbr_study = optuna.create_study(direction="maximize")
    gbr_study.optimize(gbr_objective, n_trials=30)
    print("\nOptuna GradientBoosting best parameters:", gbr_study.best_params)

    # ===================== Train Models =====================
    lasso_reg = MultiOutputRegressor(Lasso(alpha=best_trial.params.get('lasso_alpha', 0.01), max_iter=10000))
    ridge_reg = MultiOutputRegressor(Ridge(alpha=ridge_study.best_params['ridge_alpha'], max_iter=10000))
    gbr_reg = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=gbr_study.best_params['gbr_n_estimators'],
        max_depth=gbr_study.best_params['gbr_max_depth'],
        learning_rate=gbr_study.best_params['gbr_learning_rate'],
        subsample=gbr_study.best_params['gbr_subsample'],
        min_samples_split=gbr_study.best_params['gbr_min_samples_split'],
        min_samples_leaf=gbr_study.best_params['gbr_min_samples_leaf'],
        random_state=42
    ))
    xgb_reg = MultiOutputRegressor(XGBRegressor(
        n_estimators=best_trial.params.get('xgb_n_estimators', 100),
        max_depth=best_trial.params.get('xgb_max_depth', 3),
        learning_rate=best_trial.params.get('xgb_learning_rate', 0.1),
        subsample=best_trial.params.get('xgb_subsample', 1.0),
        reg_alpha=best_trial.params.get('xgb_reg_alpha', 0.0),
        reg_lambda=best_trial.params.get('xgb_reg_lambda', 1.0),
        min_child_weight=best_trial.params.get('xgb_min_child_weight', 1),
        verbosity=0,
        n_jobs=1,
        random_state=42
    ))

    # ===================== Stacking Ensemble =====================
    print("\nFitting Stacking Ensemble (Ridge + GBR + XGB)...")
    stacking = MultiOutputRegressor(StackingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=ridge_study.best_params['ridge_alpha'], max_iter=10000)),
            ("gbr", GradientBoostingRegressor(
                n_estimators=gbr_study.best_params['gbr_n_estimators'],
                max_depth=gbr_study.best_params['gbr_max_depth'],
                learning_rate=gbr_study.best_params['gbr_learning_rate'],
                subsample=gbr_study.best_params['gbr_subsample'],
                min_samples_split=gbr_study.best_params['gbr_min_samples_split'],
                min_samples_leaf=gbr_study.best_params['gbr_min_samples_leaf'],
                random_state=42
            )),
            ("xgb", XGBRegressor(
                n_estimators=best_trial.params.get('xgb_n_estimators', 100),
                max_depth=best_trial.params.get('xgb_max_depth', 3),
                learning_rate=best_trial.params.get('xgb_learning_rate', 0.1),
                subsample=best_trial.params.get('xgb_subsample', 1.0),
                reg_alpha=best_trial.params.get('xgb_reg_alpha', 0.0),
                reg_lambda=best_trial.params.get('xgb_reg_lambda', 1.0),
                min_child_weight=best_trial.params.get('xgb_min_child_weight', 1),
                verbosity=0,
                n_jobs=1,
                random_state=42
            ))
        ],
        final_estimator=Ridge(alpha=ridge_study.best_params['ridge_alpha'], max_iter=10000),
        passthrough=True,
        n_jobs=-1
    ))

    # ===================== Evaluation & Model Saving 
    models = [
        ("lasso", lasso_reg, "../models/lasso.pkl"),
        ("ridge", ridge_reg, "../models/ridge.pkl"),
        ("gbr", gbr_reg, "../models/gbr.pkl"),
        ("xgb", xgb_reg, "../models/xgb.pkl"),
        ("stacking", stacking, "../models/stacking_ensemble.pkl")
    ]

    best_r2, best_name, best_pipeline, best_path = float('-inf'), None, None, None
    results = []

    for name, reg, path in models:
        pipeline = build_pipeline(regressor=reg)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        results.append([name.capitalize(), r2, mae, rmse])
        print(f"{name.capitalize()} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2, best_name, best_pipeline, best_path = r2, name, pipeline, path

    if best_pipeline:
        joblib.dump(best_pipeline, best_path)
        print(f"\n{best_name.capitalize()} model saved to {best_path} (R² = {best_r2:.4f})")

    save_results_image(results, "model_results.png")



