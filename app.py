import streamlit as st
import numpy as np
import pickle

# Load the trained model from the models folder
with open("models/endo_model.pkl", "rb") as f:
    model = pickle.load(f)

# App layout and title
st.set_page_config(page_title="Endometriosis Prediction Tool", layout="centered")
st.title("🌸 Endometriosis Prediction Tool")
st.markdown("""
Enter your blood test values below. These are common markers related to inflammation and reproductive health.  
We’ll let you know if your profile suggests **possible endometriosis**.
""")

st.subheader("🧬 Blood Biomarker Inputs")

# CA-125
ca_125 = st.number_input(
    "🩸 CA-125 – Uterine & ovarian health marker\n*(Normal: 0–35 U/mL | Patient range: 0–500)*",
    min_value=0.0, max_value=500.0, value=25.0, step=1.0
)

# CA19-9
ca_199 = st.number_input(
    "🩸 CA19-9 – Digestive/reproductive marker\n*(Normal: 0–37 U/mL | Patient range: 0–300)*",
    min_value=0.0, max_value=300.0, value=15.0, step=1.0
)

# Interleukin-6
il6 = st.number_input(
    "🔥 Interleukin-6 (IL-6) – Inflammation signal\n*(Normal: <7 pg/mL | Patient range: 0–80)*",
    min_value=0.0, max_value=80.0, value=5.0, step=0.1
)

# Glycodelin
glycodelin = st.number_input(
    "🌸 Glycodelin – Fertility & menstrual protein\n*(Normal: 10–60 ng/mL | Patient range: 0–100)*",
    min_value=0.0, max_value=100.0, value=30.0, step=0.5
)

# TNF-alpha
tnfa = st.number_input(
    "🔥 TNF-α – Major inflammation marker\n*(Normal: <8 pg/mL | Patient range: 0–40)*",
    min_value=0.0, max_value=40.0, value=4.0, step=0.5
)

# Predict button
if st.button("🧠 Predict"):
    try:
        # Prepare input data
        input_data = np.array([[ca_125, ca_199, il6, glycodelin, tnfa]])
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Display result
        st.subheader("🩺 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **Likely Endometriosis Detected**\n\n**Confidence:** {probability:.2%}")
        else:
            st.success(f"✅ **No Endometriosis Detected**\n\n**Confidence:** {1 - probability:.2%}")

    except Exception as e:
        st.error(f"🚨 Prediction error: {e}")
