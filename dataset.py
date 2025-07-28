import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 2000

# Possible categories
product_types = ["Engine", "Battery", "PCB", "Chassis", "Display"]
components = ["Valve", "Cell", "Connector", "Frame", "Sensor", "Controller"]
change_types = ["Design", "Supplier", "Process"]
urgency_levels = ["Low", "Medium", "High"]

# Generate features
ECR_ID = [f"ECR_{1000+i}" for i in range(n_samples)]
Product_Type = np.random.choice(product_types, n_samples)
Component_Name = np.random.choice(components, n_samples)
Change_Type = np.random.choice(change_types, n_samples)
Change_Complexity = np.random.randint(1, 11, n_samples)  # 1 to 10
Supplier_Criticality = np.random.randint(1, 6, n_samples)  # 1 to 5
Past_Similar_Changes = np.random.poisson(2, n_samples)    # avg around 2
Team_Experience_Level = np.random.randint(1, 6, n_samples) # 1 to 5
BOM_Depth = np.random.randint(1, 8, n_samples)            # 1 to 7
Urgency = np.random.choice(urgency_levels, n_samples)

# Generate correlated targets
Approval_Time_Days = (
    5
    + Change_Complexity * np.random.uniform(1.2, 1.6, n_samples)
    + (Supplier_Criticality * 2)
    - (Team_Experience_Level * np.random.uniform(0.5, 1.0, n_samples))
    - np.array([3 if u == "High" else 0 for u in Urgency])
    + np.random.normal(0, 2, n_samples)
).astype(int)

Parts_Affected = (
    BOM_Depth * np.random.uniform(1.5, 2.5, n_samples)
    + Change_Complexity * 0.5
    + np.random.normal(0, 1, n_samples)
).astype(int)

Risk_Score = (
    (0.05 * Change_Complexity)
    + (0.1 * Supplier_Criticality)
    + (0.05 * (BOM_Depth - 3))
    + np.random.normal(0, 0.05, n_samples)
)
Risk_Score = np.clip(Risk_Score / 10 + 0.3, 0, 1)  # normalize 0-1

# Create DataFrame
df = pd.DataFrame({
    "ECR_ID": ECR_ID,
    "Product_Type": Product_Type,
    "Component_Name": Component_Name,
    "Change_Type": Change_Type,
    "Change_Complexity": Change_Complexity,
    "Supplier_Criticality": Supplier_Criticality,
    "Past_Similar_Changes": Past_Similar_Changes,
    "Team_Experience_Level": Team_Experience_Level,
    "BOM_Depth": BOM_Depth,
    "Urgency": Urgency,
    "Approval_Time_Days": Approval_Time_Days,
    "Parts_Affected": Parts_Affected,
    "Risk_Score": Risk_Score
})

# Save to CSV
df.to_csv("ecr_data.csv", index=False)
print("Synthetic ECR dataset created: ecr_data.csv")
print(df.head())
