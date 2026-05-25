# ================================
# HOUSE PRICE PREDICTION APP
# STREAMLIT FULL PROJECT CODE
# ================================

# Importing required libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# CUSTOM CSS STYLING
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.title("🏠 House Price Prediction Dashboard")

st.write("Predict house prices using Machine Learning")


# ==========================================
# LOADING DATASET
# ==========================================

# Loading dataset

df = pd.read_csv(r"C:\Users\indun\Downloads\archive (13)\Housing.csv")


# ==========================================
# DATA PREPROCESSING
# ==========================================

# Encoding categorical columns

encoder = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])


# ==========================================
# SPLITTING DATA
# ==========================================

X = df.drop("price", axis=1)

y = df["price"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# MODEL TRAINING
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Enter House Details")

area = st.sidebar.number_input(
    "Area",
    min_value=500,
    max_value=20000,
    value=5000
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    10,
    2
)

stories = st.sidebar.slider(
    "Stories",
    1,
    5,
    2
)

mainroad = st.sidebar.selectbox(
    "Main Road Access",
    ["Yes", "No"]
)

guestroom = st.sidebar.selectbox(
    "Guest Room",
    ["Yes", "No"]
)

basement = st.sidebar.selectbox(
    "Basement",
    ["Yes", "No"]
)

hotwaterheating = st.sidebar.selectbox(
    "Hot Water Heating",
    ["Yes", "No"]
)

airconditioning = st.sidebar.selectbox(
    "Air Conditioning",
    ["Yes", "No"]
)

parking = st.sidebar.slider(
    "Parking Spaces",
    0,
    5,
    1
)

prefarea = st.sidebar.selectbox(
    "Preferred Area",
    ["Yes", "No"]
)

furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)


# ==========================================
# CONVERTING INPUTS TO NUMBERS
# ==========================================

mainroad = 1 if mainroad == "Yes" else 0

guestroom = 1 if guestroom == "Yes" else 0

basement = 1 if basement == "Yes" else 0

hotwaterheating = 1 if hotwaterheating == "Yes" else 0

airconditioning = 1 if airconditioning == "Yes" else 0

prefarea = 1 if prefarea == "Yes" else 0


# Furnishing encoding

if furnishingstatus == "Furnished":
    furnishingstatus = 0

elif furnishingstatus == "Semi-Furnished":
    furnishingstatus = 1

else:
    furnishingstatus = 2


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.sidebar.button("Predict House Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]])

    prediction = model.predict(input_data)

    st.success(f"🏡 Predicted House Price: ₹ {prediction[0]:,.2f}")


# ==========================================
# MODEL PERFORMANCE
# ==========================================

r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")

st.metric(
    label="R2 Score",
    value=f"{r2:.2f}"
)


# ==========================================
# DATASET PREVIEW
# ==========================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())


# ==========================================
# PRICE DISTRIBUTION PLOT
# ==========================================

st.subheader("📈 House Price Distribution")

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.histplot(df['price'], kde=True, ax=ax1)

ax1.set_title("House Price Distribution")

st.pyplot(fig1)


# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12,8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax2
)

ax2.set_title("Correlation Heatmap")

st.pyplot(fig2)


# ==========================================
# ACTUAL VS PREDICTED PLOT
# ==========================================

st.subheader("📌 Actual vs Predicted Prices")

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.scatter(y_test, y_pred)

ax3.set_xlabel("Actual Prices")

ax3.set_ylabel("Predicted Prices")

ax3.set_title("Actual vs Predicted Prices")

st.pyplot(fig3)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

fig4, ax4 = plt.subplots(figsize=(10,6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance,
    ax=ax4
)

ax4.set_title("Feature Importance")

st.pyplot(fig4)


# ==========================================
# COMPARISON TABLE
# ==========================================

st.subheader("📋 Actual vs Predicted Table")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

st.dataframe(comparison.head(20))


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "Developed using Streamlit and Machine Learning"
)