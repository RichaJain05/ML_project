import streamlit as st
import pickle
import pandas as pd
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load model
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_crop = data["le_crop"]
le_soil = data["le_soil"]
le_stage = data["le_stage"]

# UI
st.set_page_config(page_title="Crop Prediction", page_icon="🌱")

st.title("🌱 Crop Health Prediction System")
st.markdown("Predict whether your crop is **Healthy or Unhealthy**")

# Inputs
crop = st.selectbox(" Select Crop", le_crop.classes_)
soil = st.selectbox(" Soil Type", le_soil.classes_)
stage = st.selectbox(" Seedling Stage", le_stage.classes_)

moi = st.number_input(" Moisture Index (MOI)", min_value=0.0)
temp = st.number_input(" Temperature")
humidity = st.number_input(" Humidity")

# Prediction
if st.button(" Predict"):
    # Encode inputs
    crop_enc = le_crop.transform([crop])[0]
    soil_enc = le_soil.transform([soil])[0]
    stage_enc = le_stage.transform([stage])[0]

    # Use DataFrame (FIXED WARNING)
    input_data = pd.DataFrame([{
        "crop ID": crop_enc,
        "soil_type": soil_enc,
        "Seedling Stage": stage_enc,
        "MOI": moi,
        "temp": temp,
        "humidity": humidity
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    # Output
    st.subheader(" Result")

    if prediction == 1:
        st.success(f" Healthy Crop ({prob[1]*100:.2f}% confidence)")
    else:
        st.error(f" Unhealthy Crop ({prob[0]*100:.2f}% confidence)")