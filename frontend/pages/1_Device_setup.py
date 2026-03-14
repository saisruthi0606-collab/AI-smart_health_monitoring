import streamlit as st

st.title("⚙ Device Setup")

st.write("Configure your wearable health monitoring device.")

device = st.selectbox(
    "Select Device Type",
    ["Smartwatch","Fitness Band","Medical Sensor","Simulated Device"]
)

sampling = st.slider(
    "Sampling Rate (Hz)",
    10,200,50
)

if st.button("Connect Device"):
    st.success(f"{device} connected successfully")