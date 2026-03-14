import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load processed dataset
data = pd.read_csv("dataset/processed_sensor_data.csv")

# Split features and target
X = data.drop("activity", axis=1)
y = data["activity"]

# Select top 50 best features
selector = SelectKBest(score_func=f_classif, k=50)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]

# Create dataframe with selected features
selected_data = pd.DataFrame(X_new, columns=selected_features)

# Add target column
selected_data["activity"] = y

# Save dataset
selected_data.to_csv("dataset/selected_features_data.csv", index=False)

print("Original feature shape:", X.shape)
print("Selected feature shape:", X_new.shape)
print("Selected feature dataset saved to dataset/selected_features_data.csv")