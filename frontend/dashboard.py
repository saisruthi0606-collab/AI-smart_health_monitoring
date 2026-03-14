import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import time

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Smart Health Monitoring",
    layout="wide",
    page_icon="🧠"
)

# ----------------------------
# SESSION STATE
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

# ----------------------------
# SIMPLE USER DATABASE
# ----------------------------
users = {
    "admin":"admin123"
}

# ----------------------------
# LOGIN PAGE
# ----------------------------
def login_page():

    st.markdown(
    """
    <style>
    .login-box{
        background: linear-gradient(135deg,#1c1c2b,#0f3460);
        padding:40px;
        border-radius:15px;
        color:white;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.title("🧠 AI Smart Health Monitoring System")

    tab1,tab2 = st.tabs(["Login","Create Account"])

    # LOGIN
    with tab1:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if username in users and users[username]==password:

                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()

            else:
                st.error("Invalid credentials")

    # REGISTER
    with tab2:

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Account"):

            users[new_user] = new_pass
            st.success("Account created! Please login.")

# ----------------------------
# REAL TIME HEALTH DATA
# ----------------------------
def generate_health_data():

    heart_rate = np.random.randint(60,110)
    steps = np.random.randint(1000,12000)
    oxygen = np.random.randint(90,100)
    temperature = round(np.random.uniform(36,38.5),2)

    return heart_rate,steps,oxygen,temperature

# ----------------------------
# LOAD DATASET
# ----------------------------
try:
    dataset = pd.read_csv("dataset/mhealth_dataset.csv")
except:
    dataset = pd.DataFrame({
        "heart_rate":np.random.randint(60,100,50),
        "steps":np.random.randint(2000,10000,50),
        "oxygen":np.random.randint(90,100,50),
        "temperature":np.random.uniform(36,38,50)
    })

# ----------------------------
# MAIN APP
# ----------------------------
def main_dashboard():

    st.sidebar.title("🧠 HealthSense AI")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Health Analytics",
            "AI Health Prediction",
            "Health Insights"
        ]
    )

    # ----------------------------
    # DASHBOARD OVERVIEW
    # ----------------------------
    if page == "Dashboard Overview":

        st.title("📊 Health Dashboard")

        hr,steps,oxygen,temp = generate_health_data()

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("❤️ Heart Rate", f"{hr} bpm")
        col2.metric("👟 Steps", steps)
        col3.metric("🫁 Oxygen", f"{oxygen}%")
        col4.metric("🌡 Temperature", f"{temp}°C")

        activity_score = int((steps/12000)*100)

        st.subheader("🏆 Daily Activity Score")

        st.progress(activity_score)
        st.write(f"Score: {activity_score}/100")

        # HEALTH STATUS
        if hr>100 or oxygen<92 or temp>38:
            st.error("🚨 Health Status: Critical")
        elif hr>90:
            st.warning("⚠ Health Status: Warning")
        else:
            st.success("✅ Health Status: Normal")

    # ----------------------------
    # HEALTH ANALYTICS
    # ----------------------------
    elif page == "Health Analytics":

        st.title("📈 Health Analytics")

        fig_hr = px.line(dataset,y="heart_rate",title="Heart Rate Trend")
        st.plotly_chart(fig_hr,use_container_width=True)

        fig_steps = px.bar(dataset,y="steps",title="Daily Steps")
        st.plotly_chart(fig_steps,use_container_width=True)

        oxygen = np.random.randint(90,100)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=oxygen,
            title={'text':"Oxygen Level"},
            gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(fig_gauge,use_container_width=True)

        fig_temp = px.line(dataset,y="temperature",title="Temperature Trend")
        st.plotly_chart(fig_temp,use_container_width=True)

    # ----------------------------
    # AI HEALTH PREDICTION
    # ----------------------------
    elif page == "AI Health Prediction":

        st.title("🤖 AI Health Prediction")

        heart_rate = st.slider("Heart Rate",50,140,80)
        steps = st.slider("Steps",0,15000,5000)
        temperature = st.slider("Temperature",35.0,40.0,36.8)
        oxygen = st.slider("Oxygen",85,100,96)

        if st.button("Run AI Prediction"):

            try:

                sensor_values = np.random.uniform(0,1,50).tolist()

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"data": sensor_values}
                )

                result = response.json()

                st.success("AI Prediction Results")

                st.write("RandomForest:", result["rf_prediction"])
                st.write("LSTM:", result["lstm_prediction"])
                st.write("CNN:", result["cnn_prediction"])
                st.write("Final:", result["final_prediction"])

            except Exception as e:
                st.error(f"API Error: {e}")

    # ----------------------------
    # HEALTH INSIGHTS
    # ----------------------------
    elif page == "Health Insights":

        st.title("💡 Health Recommendations")

        hr,steps,oxygen,temp = generate_health_data()

        insights=[]

        if hr>100:
            insights.append("⚠ Heart rate high. Rest recommended.")

        if oxygen<92:
            insights.append("⚠ Oxygen low. Consult doctor.")

        if steps<4000:
            insights.append("🚶 Increase activity today.")

        if temp>38:
            insights.append("🌡 Fever detected.")

        if len(insights)==0:
            insights.append("✅ Health metrics look good!")

        for i in insights:
            st.info(i)

    # ----------------------------
    # ALERTS PANEL
    # ----------------------------
    st.sidebar.title("🔔 Alerts")

    hr,steps,oxygen,temp = generate_health_data()

    if hr>100:
        st.sidebar.error("High heart rate detected")

    if oxygen<92:
        st.sidebar.error("Low oxygen level")

    if temp>38:
        st.sidebar.error("High body temperature")

    # ----------------------------
    # GAMIFICATION
    # ----------------------------
    st.sidebar.title("🏆 Wellness Tracker")

    st.sidebar.write("🔥 Streak: 5 days")
    st.sidebar.write("🥇 Badge: Active User")

# ----------------------------
# APP FLOW
# ----------------------------
if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()