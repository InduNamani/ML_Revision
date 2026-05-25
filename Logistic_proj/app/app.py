import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #b30059;
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #b30059;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🩺 Breast Cancer Prediction Dashboard")

st.write(
    "Predict whether the tumor is Malignant or Benign using Logistic Regression"
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(r"C:\Users\indun\Downloads\archive (15)\data.csv")

# =====================================================
# REMOVE UNNECESSARY COLUMNS
# =====================================================

df = df.drop(["id", "Unnamed: 32"], axis=1)

# =====================================================
# CONVERT TARGET COLUMN
# =====================================================

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =====================================================
# MODEL TRAINING
# =====================================================

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Enter Tumor Details")

radius_mean = st.sidebar.slider(
    "Radius Mean",
    0.0,
    30.0,
    14.0
)

texture_mean = st.sidebar.slider(
    "Texture Mean",
    0.0,
    40.0,
    19.0
)

perimeter_mean = st.sidebar.slider(
    "Perimeter Mean",
    0.0,
    200.0,
    90.0
)

area_mean = st.sidebar.slider(
    "Area Mean",
    0.0,
    3000.0,
    600.0
)

smoothness_mean = st.sidebar.slider(
    "Smoothness Mean",
    0.0,
    1.0,
    0.1
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.sidebar.button("Predict Cancer"):

    # Create full input array with 30 features
    input_data = np.zeros((1, 30))

    input_data[0][0] = radius_mean
    input_data[0][1] = texture_mean
    input_data[0][2] = perimeter_mean
    input_data[0][3] = area_mean
    input_data[0][4] = smoothness_mean

    # Scale data
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:

        st.error(
            f"""
            ⚠️ Malignant Tumor Detected

            Probability: {probability[0][1] * 100:.2f}%
            """
        )

    else:

        st.success(
            f"""
            ✅ Benign Tumor Detected

            Probability: {probability[0][0] * 100:.2f}%
            """
        )

# =====================================================
# MODEL ACCURACY
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")

st.metric(
    label="Accuracy",
    value=f"{accuracy:.2f}"
)

# =====================================================
# DATASET PREVIEW
# =====================================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())

# =====================================================
# COUNT PLOT
# =====================================================

st.subheader("📈 Diagnosis Count")

fig1, ax1 = plt.subplots(figsize=(6,4))

sns.countplot(
    x='diagnosis',
    data=df,
    ax=ax1
)

st.pyplot(fig1)

# =====================================================
# HEATMAP
# =====================================================

st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    ax=ax2
)

st.pyplot(fig2)

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("📋 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig3, ax3 = plt.subplots(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax3
)

st.pyplot(fig3)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

st.subheader("⭐ Feature Importance")

fig4, ax4 = plt.subplots(figsize=(10,8))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance,
    ax=ax4
)

st.pyplot(fig4)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "Developed using Streamlit and Logistic Regression"
)