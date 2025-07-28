import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

features = ['Product_Type','Component_Name','Change_Type','Urgency',
            'Change_Complexity','Supplier_Criticality','Past_Similar_Changes',
            'Team_Experience_Level','BOM_Depth']
cat_features = ['Product_Type','Component_Name','Change_Type','Urgency']

def load_data(path="../data/ecr_data.csv"):
    return pd.read_csv(path)

def build_pipeline(regressor=None, scale_numerical=False):
    """Builds a preprocessing pipeline with the given regressor"""
    if regressor is None:
        regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    
    transformers = [('encoder', OneHotEncoder(handle_unknown='ignore'), cat_features)]
    if scale_numerical:
        transformers.append(('scaler', StandardScaler(), features[4:]))
    
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    return Pipeline(steps=[('preprocessor', ct), ('regressor', regressor)])
