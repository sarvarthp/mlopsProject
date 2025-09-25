import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONF_MATRIX_PATH = MODEL_DIR / "confusion_matrices.npy"

st.set_page_config(page_title="Airline Passenger Satisfaction", layout="wide")
st.title("✈️ Airline Passenger Satisfaction Predictor")

@st.cache_resource
def load_model(name):
    path = MODEL_DIR / f"{name.lower()}_model.pkl"
    return joblib.load(path)

def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

def load_conf_matrices():
    if CONF_MATRIX_PATH.exists():
        return np.load(CONF_MATRIX_PATH, allow_pickle=True).item()
    return None

# Sidebar for algorithm selection
algorithm = st.sidebar.selectbox("Select Algorithm", ["RandomForest", "DecisionTree"])

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
ratings_cols = [
    "On-board service", "Inflight service", "Online boarding", "Inflight entertainment",
    "Departure/Arrival time convenient", "Leg room service", "Checkin service",
    "Ease of Online booking", "Seat comfort", "Inflight wifi service", "Food and drink",
    "Gate location", "Baggage handling", "Cleanliness"
]

rating_values = {}
for col in ratings_cols:
    rating_values[col] = st.sidebar.slider(col, 1, 5, 3)

# Load model
try:
    model = load_model(algorithm)
except:
    st.error("❌ Model not found. Run training first.")
    st.stop()

# Predict
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
        **rating_values
    }])
    pred = model.predict(input_df)[0]
    st.success("Prediction: " + ("✅ Satisfied" if pred == 1 else "❌ Dissatisfied"))

# Show evaluation metrics
st.markdown("---")
st.subheader(f"📊 Model Performance ({algorithm})")

metrics = load_metrics()
conf_matrices = load_conf_matrices()

if metrics and algorithm in metrics:
    df_metrics = pd.DataFrame([metrics[algorithm]]).T.reset_index()
    df_metrics.columns = ["Metric", "Value"]
    st.dataframe(df_metrics, use_container_width=True)

    # Bar chart
    st.subheader("📈 Metrics Overview")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Metric", y="Value", data=df_metrics, ax=ax, palette="Blues_d")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # Confusion matrix
    if conf_matrices and algorithm in conf_matrices:
        cm = conf_matrices[algorithm]
        st.subheader("🔍 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Dissatisfied", "Satisfied"],
                    yticklabels=["Dissatisfied", "Satisfied"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("Run training again to generate metrics and confusion matrices.")
