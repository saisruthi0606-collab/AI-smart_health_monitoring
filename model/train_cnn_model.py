import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("dataset/selected_features_data.csv")

# Activity mapping
activity_map = {
    "WALKING":0,
    "WALKING_UPSTAIRS":1,
    "WALKING_DOWNSTAIRS":2,
    "SITTING":3,
    "STANDING":4,
    "LAYING":5
}

# Convert labels to numbers
data["activity"] = data["activity"].map(activity_map)

# Split features and labels
X = data.drop("activity", axis=1).values
y = data["activity"].values

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# One-hot encode labels
y = to_categorical(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(50,1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save model
model.save("model/cnn_model.h5")

print("CNN model trained and saved")