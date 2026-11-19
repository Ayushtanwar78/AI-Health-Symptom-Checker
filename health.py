import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Health Checker",
    page_icon="🧠",
    layout="centered"
)

# CUSTOM CSS

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1f3a, #0f2a44, #081424);
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

# DATASET CREATION

data = pd.DataFrame({
    "fever":      [1, 0, 1, 1, 0, 1, 0],
    "cough":      [1, 1, 0, 1, 0, 0, 1],
    "headache":   [0, 1, 1, 1, 1, 0, 0],
    "fatigue":    [1, 0, 1, 1, 0, 1, 0],
    "body_pain":  [1, 0, 1, 1, 0, 1, 0],
    "nausea":     [0, 0, 1, 1, 0, 1, 0],
    "chills":     [1, 0, 1, 1, 0, 1, 0],
    "sweating":   [1, 0, 1, 1, 0, 1, 0],

    "disease": [
        "Flu",
        "Cold",
        "Dengue",
        "Covid",
        "Migraine",
        "Malaria",
        "Allergy"
    ]
})

# MACHINE LEARNING MODEL

X = data.drop("disease", axis=1)
y = data["disease"]

model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)

# STREAMLIT APPLICATION

st.title("🧠 AI Health Symptom Checker")

st.markdown("### 🤖 Smart Medical Diagnosis System")

st.success("✔ System Ready for Diagnosis")

st.warning("⚠ Please select your symptoms carefully.")

st.write("### Select Your Symptoms")

st.write("Choose Yes (1) or No (0) for each symptom.")

# SYMPTOM INPUT

symptoms = {}

symptom_list = [
    ("fever", "🌡 Fever"),
    ("cough", "😷 Cough"),
    ("headache", "🤕 Headache"),
    ("fatigue", "😴 Fatigue"),
    ("body_pain", "💪 Body Pain"),
    ("nausea", "🤢 Nausea"),
    ("chills", "🥶 Chills"),
    ("sweating", "💦 Sweating")
]

for symptom, label in symptom_list:

    symptoms[symptom] = st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        key=symptom
    )

# AI DISEASE PREDICTION

if st.button("🚀 Run AI Diagnosis", use_container_width=True):

    user_input = pd.DataFrame([symptoms], columns=X.columns)

    # Predict disease
    prediction = model.predict(user_input)

    # Calculate model confidence
    probabilities = model.predict_proba(user_input)

    confidence = max(probabilities[0]) * 100

    st.markdown("---")

    st.subheader("📊 Prediction Results")

    st.success(f"🩺 Predicted Disease: {prediction[0]}")

    st.info(f"📈 Model Confidence: {confidence:.2f}%")

    # Display additional information
    st.write("### Selected Symptoms")

    selected_symptoms = [
        name.replace("_", " ").title()
        for name, value in symptoms.items()
        if value == 1
    ]

    if selected_symptoms:
        st.write(", ".join(selected_symptoms))
    else:
        st.write("No symptoms selected.")

    st.markdown("---")

    st.warning(
        "⚠ This project is developed for educational purposes only. "
        "The prediction is not a medical diagnosis and should not "
        "replace professional medical advice."
    )


# FOOTER

st.markdown("---")

st.markdown(
    "<center>Developed using Python, Streamlit, Pandas "
    "and Scikit-learn 🧠</center>",
    unsafe_allow_html=True
)