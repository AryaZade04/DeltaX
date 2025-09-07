import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Generate synthetic ad dataset
# -----------------------------
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    "Ad_Budget": np.random.randint(1000, 10000, n_samples),
    "Ad_Platform": np.random.choice(["Google", "Meta", "LinkedIn"], n_samples),
    "Ad_Type": np.random.choice(["Video", "Image", "Text"], n_samples),
    "Audience_Age": np.random.choice(["18-24", "25-34", "35-44", "45+"], n_samples),
    "Audience_Interest": np.random.choice(["Tech", "Fashion", "Sports", "Finance"], n_samples),
    "Clicks": np.random.randint(0, 500, n_samples),
    "Impressions": np.random.randint(1000, 10000, n_samples)
})

# Calculate CTR and target variable
data["CTR"] = (data["Clicks"] / data["Impressions"]) * 100
data["High_CTR"] = (data["CTR"] > 5).astype(int)

# -----------------------------
# Encode categorical variables
# -----------------------------
encoders = {}
for col in ["Ad_Platform", "Ad_Type", "Audience_Age", "Audience_Interest"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Features and labels
X = data[["Ad_Budget", "Ad_Platform", "Ad_Type", "Audience_Age", "Audience_Interest", "Impressions"]]
y = data["High_CTR"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Ad Performance Prediction", page_icon="📊", layout="centered")

st.title("📊 Ad Performance Prediction Dashboard")
st.write("Adjust the parameters below and predict whether your ad campaign will have **High CTR (Good Performance)** or **Low CTR (Poor Performance)**.")

# Input widgets
budget = st.slider("Ad Budget ($)", 1000, 10000, 5000, step=500)
platform = st.selectbox("Ad Platform", ["Google", "Meta", "LinkedIn"])
ad_type = st.selectbox("Ad Type", ["Video", "Image", "Text"])
age = st.selectbox("Audience Age", ["18-24", "25-34", "35-44", "45+"])
interest = st.selectbox("Audience Interest", ["Tech", "Fashion", "Sports", "Finance"])
impressions = st.slider("Expected Impressions", 1000, 10000, 4000, step=500)

# Prepare input for prediction
input_data = pd.DataFrame({
    "Ad_Budget": [budget],
    "Ad_Platform": [encoders["Ad_Platform"].transform([platform])[0]],
    "Ad_Type": [encoders["Ad_Type"].transform([ad_type])[0]],
    "Audience_Age": [encoders["Audience_Age"].transform([age])[0]],
    "Audience_Interest": [encoders["Audience_Interest"].transform([interest])[0]],
    "Impressions": [impressions]
})

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# Prediction button
if st.button("Predict CTR"):
    result = "🔥 High CTR (Good Performance)" if prediction[0] == 1 else "⚠️ Low CTR (Poor Performance)"
    st.subheader(result)
