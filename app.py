import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import joblib
import sklearn.datasets as datasets


model = joblib.load("breast_model.pkl")

feature_names = [
    "worst concave points",
    "mean concave points",
    "worst area",
    "worst perimeter",
    "concavity error",
    "worst radius"
]

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🩺 Breast Cancer Prediction App")
st.markdown("### Enter tumor details below to predict cancer type")
st.markdown("---")

st.subheader("📊 Input Features")

worst_concave_points = st.number_input("Worst Concave Points", value=0.1)
mean_concave_points = st.number_input("Mean Concave Points", value=0.05)
worst_area = st.number_input("Worst Area", value=500.0)
worst_perimeter = st.number_input("Worst Perimeter", value=100.0)
concavity_error = st.number_input("Concavity Error", value=0.01)
worst_radius = st.number_input("Worst Radius", value=15.0)

if st.button("🔍 Predict"):
    input_data = np.array([[
    worst_concave_points,
    mean_concave_points,
    worst_area,
    worst_perimeter,
    concavity_error,
    worst_radius
]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = probability[0][prediction[0]] * 100

    st.markdown("---")

    if prediction[0] == 1:
        st.error(f"⚠️ Malignant ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ Benign ({confidence:.2f}% confidence)")

st.subheader("Feature Importance")

importance = model.feature_importances_
indices = np.argsort(importance)  # top 10 features

plt.figure()
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance Score")

st.pyplot(plt)