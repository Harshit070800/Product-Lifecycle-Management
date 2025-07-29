import joblib
import pandas as pd
import os
from sklearn.metrics import r2_score
from preprocessing import target_columns
MODEL_PATH = "../models/xgb.pkl"

def load_best_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Best model not found. Run model_comparison.py first.")
    return joblib.load(MODEL_PATH)

def predict_new_ecr(new_data: pd.DataFrame):
    """Predicts Approval Time, Parts Affected, and Risk Score for new change requests."""
    model = load_best_model()
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    target = target_columns
    test_data = pd.read_csv("/home/sid/Documents/Product Lifecycle Management/src/ecr_data.csv")


    preds = predict_new_ecr(test_data)
    r2 = r2_score(test_data[target], preds)
    print(f"The r2 score is {r2}")
    print("\nPredicted Output for New ECR:")
    print(preds)
