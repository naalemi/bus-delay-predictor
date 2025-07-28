# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load('xgb_model.pkl')
encoder = joblib.load('target_encoder.pkl')
scaler = joblib.load('scaler.pkl')
threshold = joblib.load('optimal_threshold.pkl')

st.title("🚌 Predicting School Bus Delays and Breakdowns in NYC")

st.markdown("Enter the following information:")

# === Input fields ===
school_year = st.selectbox("School Year", ['2024-2025', '2023-2024', '2022-2023'])
num_students = st.slider("Number of Students on the Bus", 0, 80, 25)
run_type = st.selectbox("Run Type", [
    'Pre-K/EI', 'Special Ed AM Run', 'General Ed AM Run',
    'Special Ed PM Run', 'General Ed PM Run',
    'General Ed Field Trip', 'Special Ed Field Trip'
])
reason = st.selectbox("Reason", [
    'Heavy Traffic', 'Other', 'Mechanical Problem', "Won`t Start", 'Flat Tire',
    'Problem Run', 'Accident', 'Late return from Field Trip',
    'Weather Conditions', 'Delayed by School'
])
borough = st.selectbox("Borough", ['Brooklyn', 'Bronx', 'Staten Island', 'Queens', 'Manhattan'])
hour = st.slider("Hour of Day", 0, 23, 8)
month = st.slider("Month", 1, 12, 5)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = int(day_of_week in [5, 6])
is_rush_hour = int(hour in [7, 8, 9, 15, 16, 17])
school_age_or_prek = st.selectbox("Student Type", ['Pre-K', 'School-Age'])

bus_company = st.selectbox("Bus Company", [
    'L & M BUS CORP', 'BORO TRANSIT INC', 'PIONEER TRANSPORTATION CORP',
    'PRIDE TRANSPORTATION INC', 'ALLIED TRANSIT CORP',
    'CONSOLIDATED BUS TRANSIT INC', 'LOGAN BUS COMPANY INC',
    'LITTLE RICHIE BUS SERVICE', 'SNT BUS INC', 'DON THOMAS BUSES INC',
    'Other'
])

contract_notified_schools = st.checkbox("Contractor Notified Schools?")
contract_notified_parents = st.checkbox("Contractor Notified Parents?")
alerted_opt = st.checkbox("Alerted OPT?")

# === Assemble feature input ===
input_dict = {
    'School_Year': school_year,
    'Number_Of_Students_On_The_Bus': num_students,
    'Run_Type': run_type,
    'Reason': reason,
    'Borough': borough,
    'Hour': hour,
    'Month': month,
    'Day_of_Week': day_of_week,
    'Is_Weekend': is_weekend,
    'Is_Rush_Hour': is_rush_hour,
    'School_Age_or_PreK': school_age_or_prek,
    'Bus_Company_Name_grouped': bus_company,
    'Contract_Notified_Schools': int(contract_notified_schools),
    'Contract_Notified_Parents': int(contract_notified_parents),
    'Alerted_OPT': int(alerted_opt)
}

input_df = pd.DataFrame([input_dict])

# === Encode, scale, and predict ===
input_encoded = encoder.transform(input_df)
input_scaled = scaler.transform(input_encoded)
proba = model.predict_proba(input_scaled)[0][1]  # Probability of Breakdown
pred = int(proba >= threshold)

label = "🚨 Breakdown" if pred == 1 else "⏱️ Running Late"
confidence = proba if pred == 1 else 1 - proba

# === Display result ===
st.markdown(f"### Prediction: **{label}**")
st.progress(float(confidence))
st.caption(f"Confidence: {confidence*100:.1f}%")
