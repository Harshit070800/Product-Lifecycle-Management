import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from preprocessing import features, cat_features

def plot_feature_importance(model_path="../models/best_model.pkl",
                            save_path="../outputs/feature_importance_plots/best_model_importance.png"):
    """
    Plots feature importance for the saved best model.
    Supports tree-based models (feature_importances_) and linear models (coef_).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model = joblib.load(model_path)
    reg = model.named_steps['regressor']

    # Get feature names
    encoded = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(cat_features)
    all_features = np.concatenate([encoded, features[4:]])

    # Determine importance
    if hasattr(reg, "feature_importances_"):  
        importances = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        importances = np.abs(np.ravel(reg.coef_))
    else:
        print(f"Model {type(reg).__name__} does not support feature importance.")
        return

    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices], color="skyblue")
    plt.xticks(range(len(importances)), all_features[indices], rotation=90)
    plt.title(f"Feature Importance - {os.path.basename(model_path)}")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

if __name__ == "__main__":
    plot_feature_importance()
