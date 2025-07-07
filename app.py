import streamlit as st
import joblib
import numpy as np

model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

st.title("⚽ Player Market Value Predictor")

age = st.number_input("Age", 16, 40, 24)
position_cat = st.selectbox("Position Category", [1, 2, 3, 4])
page_views = st.number_input("Page Views", 0, 100000, 5000)
fpl_value = st.number_input("FPL Value", 0.0, 15.0, 9.5)
fpl_sel = st.number_input("FPL Selection %", 0.0, 100.0, 25.3)
fpl_points = st.number_input("FPL Points", 0, 300, 180)
region = st.selectbox("Region", [1, 2, 3, 4])
new_foreign = st.selectbox("New Foreign?", [0, 1])
big_club = st.selectbox("Big Club?", [0, 1])
new_signing = st.selectbox("New Signing?", [0, 1])

if st.button("Predict"):
    input_data = {
        "age": age,
        "position_cat": position_cat,
        "page_views": page_views,
        "fpl_value": fpl_value,
        "fpl_sel": fpl_sel,
        "fpl_points": fpl_points,
        "region": region,
        "new_foreign": new_foreign,
        "big_club": big_club,
        "new_signing": new_signing
    }

    import pandas as pd
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns
    df = df[feature_columns]  # Ensure correct column order

    df_scaled = scaler.transform(df)  # Scale the input
    prediction = model.predict(df_scaled)  # Predict using the model

    st.success(f"💰 Predicted Market Value: €{round(prediction[0], 2)} million")
