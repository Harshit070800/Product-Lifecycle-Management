import streamlit as st
import joblib
import pandas as pd
import os
from preprocessing import features, target_columns

# Possible categories for dropdowns (from dataset.py)
product_types = ["Engine", "Battery", "PCB", "Chassis", "Display"]
components = ["Valve", "Cell", "Connector", "Frame", "Sensor", "Controller"]
change_types = ["Design", "Supplier", "Process"]
urgency_levels = ["Low", "Medium", "High"]

st.set_page_config(page_title="ECR Risk & Impact Dashboard", layout="centered")
st.title("ECR Risk & Impact Prediction Dashboard")
st.write("Enter details for a new Engineering Change Request (ECR) to get a predictive risk and impact summary.")

# Model loader
@st.cache_resource
def load_model():
    model_paths = [
        "../models/stacking_ensemble.pkl",
        "../models/xgb.pkl",
        "../models/gbr.pkl",
        "../models/ridge.pkl",
        "../models/lasso.pkl"
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

model, model_path = load_model()
if not model:
    st.error("No trained model found. Please train a model first.")
    st.stop()
else:
    st.success(f"Loaded model: {model_path}")

# Input form
with st.form("ecr_form"):
    col1, col2 = st.columns(2)
    with col1:
        Product_Type = st.selectbox("Product Type", product_types)
        Component_Name = st.selectbox("Component Name", components)
        Change_Type = st.selectbox("Change Type", change_types)
        Urgency = st.selectbox("Urgency", urgency_levels)
    with col2:
        Change_Complexity = st.slider("Change Complexity", 1, 10, 5)
        Supplier_Criticality = st.slider("Supplier Criticality", 1, 10, 5)
        Past_Similar_Changes = st.slider("Past Similar Changes", 0, 10, 2)
        Team_Experience_Level = st.slider("Team Experience Level", 1, 10, 5)
        BOM_Depth = st.slider("BOM Depth", 1, 10, 5)
    submitted = st.form_submit_button("Predict Risk & Impact")

if submitted:
    input_dict = {
        "Product_Type": Product_Type,
        "Component_Name": Component_Name,
        "Change_Type": Change_Type,
        "Urgency": Urgency,
        "Change_Complexity": Change_Complexity,
        "Supplier_Criticality": Supplier_Criticality,
        "Past_Similar_Changes": Past_Similar_Changes,
        "Team_Experience_Level": Team_Experience_Level,
        "BOM_Depth": BOM_Depth
    }
    input_df = pd.DataFrame([input_dict])
    preds = model.predict(input_df)
    summary = dict(zip(target_columns, preds[0]))
    st.subheader("Risk/Impact Summary")
    st.write(":star: **Approval Time (Days):** {}".format(int(round(summary['Approval_Time_Days']))))
    st.write(":star: **Parts Affected:** {}".format(int(round(summary['Parts_Affected']))))
    st.write(":star: **Risk Score:** {:.2f}%".format(summary['Risk_Score'] * 100))
    st.success("Prediction complete! Review the summary above.")

st.markdown("---")
st.caption("Built with Streamlit. Model and dashboard by your ML team.")
