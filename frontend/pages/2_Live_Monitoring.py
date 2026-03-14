import streamlit as st
import numpy as np
import pandas as pd
import time

st.title("📡 Live Sensor Monitoring")

chart = st.line_chart()

for i in range(50):
    new_data = pd.DataFrame(
        np.random.randn(1,1),
        columns=["Sensor Signal"]
    )

    chart.add_rows(new_data)
    time.sleep(0.1)