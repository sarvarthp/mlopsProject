import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
CONF_MATRIX_PATH = Path(__file__).resolve().parents[1] / "models" / "confusion_matrix.npy"

st.set_page_config(page_title="Airline Passenger Satisfaction", layout="wide")
st.title("✈️ Airline Passenger Satisfaction Predictor")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

def load_conf_matrix():
    if CONF_MATRIX_PATH.exists():
        return np.load(CONF_MATRIX_PATH)
    return None

# Load model
try:
    model = load_model()
except:
    st.error("❌ Model not found. Run training first.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Passenger Details")
age = st.sidebar.number_input("Age", 0, 120, 30)
flight_distance = st.sidebar.number_input("Flight Distance", 0, 10000, 500)
dep_delay = st.sidebar.number_input("Departure Delay (minutes)", -60, 600, 0)
arr_delay = st.sidebar.number_input("Arrival Delay (minutes)", -60, 600, 0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
cust_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])

st.sidebar.header("Flight Experience Ratings (1–5)")

# Service rating sliders
onboard_service = st.sidebar.slider("On-board service", 1, 5, 3)
inflight_service = st.sidebar.slider("Inflight service", 1, 5, 3)
online_boarding = st.sidebar.slider("Online boarding", 1, 5, 3)
inflight_entertainment = st.sidebar.slider("Inflight entertainment", 1, 5, 3)
departure_arrival_convenient = st.sidebar.slider("Departure/Arrival time convenient", 1, 5, 3)
leg_room_service = st.sidebar.slider("Leg room service", 1, 5, 3)
checkin_service = st.sidebar.slider("Checkin service", 1, 5, 3)
ease_online_booking = st.sidebar.slider("Ease of Online booking", 1, 5, 3)
seat_comfort = st.sidebar.slider("Seat comfort", 1, 5, 3)
inflight_wifi = st.sidebar.slider("Inflight wifi service", 1, 5, 3)
food_drink = st.sidebar.slider("Food and drink", 1, 5, 3)
gate_location = st.sidebar.slider("Gate location", 1, 5, 3)
baggage_handling = st.sidebar.slider("Baggage handling", 1, 5, 3)
cleanliness = st.sidebar.slider("Cleanliness", 1, 5, 3)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay,
        "Gender": gender,
        "Customer Type": cust_type,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "On-board service": onboard_service,
        "Inflight service": inflight_service,
        "Online boarding": online_boarding,
        "Inflight entertainment": inflight_entertainment,
        "Departure/Arrival time convenient": departure_arrival_convenient,
        "Leg room service": leg_room_service,
        "Checkin service": checkin_service,
        "Ease of Online booking": ease_online_booking,
        "Seat comfort": seat_comfort,
        "Inflight wifi service": inflight_wifi,
        "Food and drink": food_drink,
        "Gate location": gate_location,
        "Baggage handling": baggage_handling,
        "Cleanliness": cleanliness
    }])
    pred = model.predict(input_df)[0]
    st.success("Prediction: " + ("✅ Satisfied" if pred == 1 else "❌ Dissatisfied"))

# Show evaluation metrics
st.markdown("---")
st.subheader("📊 Model Performance (Test Data)")

metrics = load_metrics()
conf_matrix = load_conf_matrix()

if metrics:
    # Show metrics table
    df_metrics = pd.DataFrame([metrics]).T.reset_index()
    df_metrics.columns = ["Metric", "Value"]
    st.dataframe(df_metrics, use_container_width=True)

    # Bar chart
    st.subheader("📈 Metrics Overview")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Metric", y="Value", data=df_metrics, ax=ax, palette="Blues_d")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # Confusion matrix
    if conf_matrix is not None:
        st.subheader("🔍 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Dissatisfied", "Satisfied"],
                    yticklabels=["Dissatisfied", "Satisfied"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("Run training again to generate metrics.json and confusion_matrix.npy.")
