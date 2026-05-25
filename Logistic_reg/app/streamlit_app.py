import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


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


st.title("🩺 Diabetes Prediction Dashboard")

st.write("Predict whether a person has diabetes using Logistic Regression")

df = pd.read_csv(r"C:\Users\indun\Downloads\archive (14)\diabetes.csv")



X = df.drop("Outcome", axis=1)

y = df["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = LogisticRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


st.sidebar.header("Enter Patient Details")


pregnancies = st.sidebar.slider(
    "Pregnancies",
    0,
    20,
    1
)

glucose = st.sidebar.slider(
    "Glucose",
    0,
    200,
    120
)

bloodpressure = st.sidebar.slider(
    "Blood Pressure",
    0,
    140,
    70
)

skinthickness = st.sidebar.slider(
    "Skin Thickness",
    0,
    100,
    20
)

insulin = st.sidebar.slider(
    "Insulin",
    0,
    900,
    80
)

bmi = st.sidebar.slider(
    "BMI",
    0.0,
    70.0,
    25.0
)

dpf = st.sidebar.slider(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5
)

age = st.sidebar.slider(
    "Age",
    1,
    100,
    30
)


if st.sidebar.button("Predict Diabetes"):

    input_data = np.array([[
        pregnancies,
        glucose,
        bloodpressure,
        skinthickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(
            f"⚠️ Person is likely to have Diabetes\n\nProbability: {probability[0][1]*100:.2f}%"
        )

    else:
        st.success(
            f"✅ Person is not likely to have Diabetes\n\nProbability: {probability[0][0]*100:.2f}%"
        )


accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")

st.metric(
    label="Accuracy Score",
    value=f"{accuracy:.2f}"
)


st.subheader("📁 Dataset Preview")

st.dataframe(df.head())


st.subheader("📈 Diabetes Outcome Count")

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.countplot(
    x='Outcome',
    data=df,
    ax=ax1
)

ax1.set_title("Diabetes Outcome Count")

st.pyplot(fig1)


st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12,8))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    ax=ax2
)

ax2.set_title("Correlation Heatmap")

st.pyplot(fig2)


st.subheader("📌 Glucose Distribution")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.histplot(
    df['Glucose'],
    kde=True,
    ax=ax3
)

ax3.set_title("Glucose Distribution")

st.pyplot(fig3)


st.subheader("📋 Actual vs Predicted")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(comparison.head(20))


st.subheader("🧾 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig4, ax4 = plt.subplots(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax4
)

ax4.set_title("Confusion Matrix")

ax4.set_xlabel("Predicted")

ax4.set_ylabel("Actual")

st.pyplot(fig4)


st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)


importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)


st.subheader("⭐ Feature Importance")

fig5, ax5 = plt.subplots(figsize=(10,6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance,
    ax=ax5
)

ax5.set_title("Feature Importance")

st.pyplot(fig5)


st.markdown("---")

st.markdown(
    "Developed using Streamlit and Logistic Regression"
)