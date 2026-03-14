from fastapi import FastAPI
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models

rf_model = joblib.load("model/activity_model.pkl")
lstm_model = load_model("model/lstm_model.h5")
cnn_model = load_model("model/cnn_model.h5")

# Activity labels
activity_map = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

@app.get("/")
def home():
    return {"message": "Smart Health Monitoring API running"}


@app.post("/predict")
def predict_activity(data: dict):

    values = data["data"]

    features = np.array(values).reshape(1, -1)

    # RandomForest already returns activity name
    rf_pred = rf_model.predict(features)[0]

    # LSTM prediction
    lstm_input = np.array(values).reshape(1, 50, 1)
    lstm_prediction = lstm_model.predict(lstm_input)
    lstm_pred_index = int(np.argmax(lstm_prediction, axis=1)[0])

    # CNN prediction
    cnn_prediction = cnn_model.predict(lstm_input)
    cnn_pred_index = int(np.argmax(cnn_prediction, axis=1)[0])

    # Convert DL predictions to activity names
    lstm_activity = activity_map[lstm_pred_index]
    cnn_activity = activity_map[cnn_pred_index]

    # Ensemble voting
    votes = [rf_pred, lstm_activity, cnn_activity]
    final_activity = max(set(votes), key=votes.count)

    return {
        "rf_prediction": rf_pred,
        "lstm_prediction": lstm_activity,
        "cnn_prediction": cnn_activity,
        "final_prediction": final_activity
    }