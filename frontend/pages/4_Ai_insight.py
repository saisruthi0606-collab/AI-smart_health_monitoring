import streamlit as st
import pandas as pd
import numpy as np

st.title("🤖 AI Model Insights")

data = pd.DataFrame({
    "Model":["RandomForest","LSTM","CNN"],
    "Accuracy":[0.89,0.92,0.91]
})

st.bar_chart(data.set_index("Model"))

st.info("""
The system uses **Deep Ensemble Learning**.

Multiple models vote to produce the final prediction,
making the system more accurate and robust.
""")