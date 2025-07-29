import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Global Configuration  
features = [
    'Product_Type', 'Component_Name', 'Change_Type', 'Urgency',
    'Change_Complexity', 'Supplier_Criticality', 'Past_Similar_Changes',
    'Team_Experience_Level', 'BOM_Depth'
]

cat_features = ['Product_Type', 'Component_Name', 'Change_Type', 'Urgency']
num_features = [f for f in features if f not in cat_features]

target_columns = ['Approval_Time_Days', 'Parts_Affected', 'Risk_Score']

# Load Data
def load_data(path="../data/ecr_data.csv"):
    """Loads dataset from the given path."""
    return pd.read_csv(path)

def get_features_and_targets(df):
    """Splits dataset into features (X) and multiple targets (y)."""
    X = df[features]
    y = df[target_columns]
    return X, y

# Build Preprocessing + Model Pipeline
def build_pipeline(regressor, scale_numerical=False):
    """
    Creates a pipeline with OneHotEncoder for categorical features
    and optional StandardScaler for numerical features.
    """
    transformers = [
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
    if scale_numerical:
        transformers.append(('scaler', StandardScaler(), num_features))

    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

