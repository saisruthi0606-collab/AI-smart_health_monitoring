import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("dataset/processed_sensor_data.csv")

# Split features and labels
X = data.drop("activity", axis=1)
y = data["activity"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Convert to numpy
X = X.values

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Build LSTM model
model = Sequential()

model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(32, activation="relu"))
model.add(Dense(y_cat.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model 
model.save("model/lstm_model.h5")

print("LSTM model saved successfully")