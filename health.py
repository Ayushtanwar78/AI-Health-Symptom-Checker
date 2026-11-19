import stremlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AI Health Checker", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1f3a, #0f2a44, #081424);s
    color: #e3f2fd;
}

.block-container {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.2);
}

h1 {
    text-align: center;
    color: #00e5ff;
    text-shadow: 0 0 15px #00e5ff;
}

h3 {
    text-align: center;
    color: #81d4fa;
}


label {
    color: #b3e5fc !important;
    font-weight: bold;
}

div[data-baseweb="select"] {
    background-color: #112d4e;
    border-radius: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #00e5ff, #00bcd4);
    color: black;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 0 0 15px #00e5ff;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #00e5ff;
}

</style>
""", unsafe_allow_html=True)


data = pd.DataFrame({
    "fever":      [1,0,1,1,0,1,0],
    "cough":      [1,1,0,1,0,0,1],
    "headache":   [0,1,1,1,1,0,0],
    "fatigue":    [1,0,1,1,0,1,0],
    "body_pain":  [1,0,1,1,0,1,0],
    "nausea":     [0,0,1,1,0,1,0],
    "disease": ["Flu","Cold","Dengue","Covid","Migraine","Malaria","Allergy"]
})

X = data.drop("disease", axis=1)
y = data["disease"]

model = DecisionTreeClassifier()
model.fit(X, y)


st.title("🧠 AI Health Symptom Checker")
st.markdown("### 🤖 Smart Medical Diagnosis System")

st.success("✔ System Ready for Diagnosis")
st.warning("⚠ Please select symptoms carefully")

st.write("Select your symptoms (1 = Yes, 0 = No):")

fever = st.selectbox("🌡 Fever", [0, 1])
cough = st.selectbox("😷 Cough", [0, 1])
headache = st.selectbox("🤕 Headache", [0, 1])
fatigue = st.selectbox("😴 Fatigue", [0, 1])
body_pain = st.selectbox("💪 Body Pain", [0, 1])
nausea = st.selectbox("🤢 Nausea", [0, 1])


if st.button("🚀 Run AI Diagnosis"):
    user_input = [[fever, cough, headache, fatigue, body_pain, nausea]]

    prediction = model.predict(user_input)
    prob = model.predict_proba(user_input)
    confidence = max(prob[0]) * 100

    st.success(f"🩺 Predicted Disease: {prediction[0]}")
    st.info(f"📊 Confidence Level: {confidence:.2f}%")


    if confidence < 80:
        st.warning("⚠️ AI needs more data. Answer additional symptoms.")

        chills = st.selectbox("🥶 Chills", [0, 1])
        sweating = st.selectbox("💦 Sweating", [0, 1])

        new_prediction = model.predict(user_input)
        new_prob = model.predict_proba(user_input)
        new_confidence = max(new_prob[0]) * 100

        st.success(f"🔁 Updated Diagnosis: {new_prediction[0]}")
        st.info(f"📊 New Confidence: {new_confidence:.2f}%")
