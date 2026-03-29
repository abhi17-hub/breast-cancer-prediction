import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer 

model = joblib.load("breast_model.pkl")

data = load_breast_cancer()
feature_names = data.feature_names

st.title("Breast Cancer Prediction App")
st.write("Enter tumor details to predict cancer type")

mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_area = st.number_input("Mean Area")

if st.button("Predict"):
    input_data = np.array([[mean_radius, mean_texture, mean_area]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = probability[0][prediction[0]] * 100

    if prediction[0] == 1:
            st.error(f"Malignant ⚠️ ({confidence:.2f}% confidence)")
    else:
            st.success(f"Benign ✅ ({confidence:.2f}% confidence)")

st.subheader("Feature Importance")

importance = model.feature_importances_
indices = np.argsort(importance)[-10:]  # top 10 features

plt.figure()
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance Score")

st.pyplot(plt)