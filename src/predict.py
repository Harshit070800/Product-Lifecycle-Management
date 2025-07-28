import joblib
import pandas as pd
import os

MODEL_PATH = "../models/best_model.pkl"

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
    # Example input (must match training features)
    sample_input = pd.DataFrame([{
        "Product_Type": "Electronics",
        "Component_Name": "PCB_Module",
        "Change_Type": "Design",
        "Urgency": "High",
        "Change_Complexity": 7,
        "Supplier_Criticality": 8,
        "Past_Similar_Changes": 3,
        "Team_Experience_Level": 4,
        "BOM_Depth": 5
    }])

    preds = predict_new_ecr(sample_input)
    print("\nPredicted Output for New ECR:")
    print(preds)
