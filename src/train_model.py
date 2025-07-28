import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import load_data, build_pipeline, features

def train_and_save_model(target, save_path, regressor=None, scale_numerical=False):
    """
    Train a model for a given target and save it.
    - Uses build_pipeline() from preprocessing.py for consistency.
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load dataset
    df = load_data()
    X, y = df[features], df[target]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline using preprocessing.py
    model = build_pipeline(regressor=regressor, scale_numerical=scale_numerical)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"{target} → MAE: {mean_absolute_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")

    # Save
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

# If run directly, trains default RandomForest models
if __name__ == "__main__":
    train_and_save_model('Approval_Time_Days', '../models/rf_approval_time.pkl')
    train_and_save_model('Parts_Affected', '../models/rf_parts.pkl')
    train_and_save_model('Risk_Score', '../models/rf_risk.pkl')
