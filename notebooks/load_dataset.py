import pandas as pd

# Load feature names
features = pd.read_csv(
    "dataset/UCI_HAR_Dataset/features.txt",
    sep=r"\s+",
    header=None
)

feature_names = features[1].values

# Load training data
X_train = pd.read_csv(
    "dataset/UCI_HAR_Dataset/train/X_train.txt",
    sep=r"\s+",
    header=None
)

X_train.columns = feature_names

# Load training labels
y_train = pd.read_csv(
    "dataset/UCI_HAR_Dataset/train/y_train.txt",
    header=None
)

# Combine features + label
train_data = X_train.copy()
train_data["activity"] = y_train

print(train_data.head())
print("\nDataset shape:", train_data.shape)

# Load activity label mapping
activity_labels = pd.read_csv(
    "dataset/UCI_HAR_Dataset/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity_name"]
)

# Create mapping dictionary
activity_map = dict(zip(activity_labels.id, activity_labels.activity_name))

# Apply mapping
train_data["activity"] = train_data["activity"].map(activity_map)

print("\nActivity mapping applied")
print(train_data["activity"].value_counts())

# Load test features
X_test = pd.read_csv(
    "dataset/UCI_HAR_Dataset/test/X_test.txt",
    sep=r"\s+",
    header=None
)

X_test.columns = feature_names

# Load test labels
y_test = pd.read_csv(
    "dataset/UCI_HAR_Dataset/test/y_test.txt",
    header=None
)

# Combine test features + label
test_data = X_test.copy()
test_data["activity"] = y_test

# Apply activity mapping
test_data["activity"] = test_data["activity"].map(activity_map)

print("\nTest dataset shape:", test_data.shape)

# Combine train + test
full_data = pd.concat([train_data, test_data])

print("\nFull dataset shape:", full_data.shape)


# Save processed dataset
full_data.to_csv("dataset/processed_sensor_data.csv", index=False)

print("\nProcessed dataset saved successfully")