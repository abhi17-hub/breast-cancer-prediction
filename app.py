import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import joblib
import sklearn.datasets as datasets

model_choice=st.selectbox("Select Model",["Naive Bayes","XGBoost"])


if model_choice == "XGBoost":
    model = joblib.load("breast_model.pkl")

elif model_choice == "Naive Bayes":
    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()

# Select only required features
    feature_indices = [7, 20, 22, 23, 26, 27]  # indexes for your 6 features

    X = data.data[:, feature_indices]
    y = data.target

    model = GaussianNB()
    model.fit(X, y)

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
st.write(f"🔍 Selected Model: {model_choice}")
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
        mean_concave_points,
        worst_radius,
        worst_perimeter,
        worst_area,
        concavity_error,
        worst_concave_points
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = probability[0][prediction[0]] * 100

    st.markdown("---")

    if prediction[0] == 1:
        st.error(f"⚠️ Malignant ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ Benign ({confidence:.2f}% confidence)")

if model_choice == "XGBoost":
    st.subheader("Feature Importance")

    importance = model.feature_importances_
    indices = np.argsort(importance)

    plt.figure()
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")

    st.pyplot(plt)

else:
    st.info("Feature importance not available for Naive Bayes")