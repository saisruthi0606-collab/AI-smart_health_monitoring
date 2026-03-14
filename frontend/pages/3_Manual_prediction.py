import streamlit as st
import numpy as np
import pandas as pd
import requests

st.title("🧠 AI Activity Detection")

st.write("Simulate wearable sensor data and detect human activity using AI.")

# Generate sensor data
if st.button("Generate Sensor Data"):
    sensor_data = np.random.uniform(0,1,50)
    st.session_state.sensor_data = sensor_data

# Show sensor signal
if "sensor_data" in st.session_state:

    data = st.session_state.sensor_data

    st.subheader("📡 Sensor Signal")

    df = pd.DataFrame(data, columns=["Sensor Value"])

    st.line_chart(df)

    # Send to API
    if st.button("Run AI Prediction"):

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"data": list(data)}
            )

            result = response.json()

            st.subheader("🤖 AI Model Predictions")

            col1,col2,col3 = st.columns(3)

            with col1:
                st.metric("RandomForest", result["rf_prediction"])

            with col2:
                st.metric("LSTM", result["lstm_prediction"])

            with col3:
                st.metric("CNN", result["cnn_prediction"])

            st.success(f"Final AI Decision → {result['final_prediction']}")

            # Simple health alert
            if result["final_prediction"] == "LAYING":
                st.warning("⚠ Possible inactivity detected")

            if result["final_prediction"] == "WALKING":
                st.success("✅ Healthy movement detected")

        except:
            st.error("API connection failed")