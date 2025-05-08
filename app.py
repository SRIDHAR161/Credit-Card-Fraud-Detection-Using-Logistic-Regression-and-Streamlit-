import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Google Generative AI & .env loading
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyBkb3gPk-D9K-gijIe61MMcGjHNgFBcfn0"))

# Initialize Generative AI model
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# Function to generate AI insight
def generate_ai_insight(features, prediction):
    prompt = f"Explain why this transaction is {'Fraudulent' if prediction == 1 else 'Legitimate'} based on the following feature values:\n{features.tolist()}"

    try:
        response = ai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating explanation: {e}"

# Basic page config
st.set_page_config(page_title="PaySure: Real-Time Fraud Prevention System", layout="centered")

# Sidebar
st.sidebar.title("💳 PaySure: Real-Time Fraud Prevention System")
st.sidebar.info("""This application detects if a transaction is **Legitimate** or **Fraudulent** based on the input features.  
🔹 Model used: Random Forest Classifier  
🔹 Data source: Credit Card Transactions Dataset
""")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by: *SRIDHAR C*")
st.sidebar.write("📅 Date: 2025")

# Load data
data = pd.read_csv('creditcard.csv')

# Add New Features
data['Transaction_Speed'] = data['Amount'] / (data['Time'] + 1)
data['V_Sum'] = data[[f'V{i}' for i in range(1, 29)]].sum(axis=1)
data['Is_Late_Night'] = data['Time'].apply(lambda x: 1 if (x % 86400) / 3600 < 6 else 0)

# Prepare data
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns="Class")
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model - Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=2)
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Main title
st.title("🚀 PaySure: Real-Time Fraud Prevention System")

# Section: Model Performance
st.header("📈 Model Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Train Accuracy", value=f"{train_acc * 100:.2f}%")
with col2:
    st.metric(label="Test Accuracy", value=f"{test_acc * 100:.2f}%")

st.divider()

# Section: Feature Importance
st.header("🔎 Feature Importance")
feature_names = X.columns
importance = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(y='Feature', x='Importance', data=importance_df, palette='coolwarm', ax=ax)
ax.set_title('Feature Importance (Higher value = More influence)')
st.pyplot(fig)

st.divider()

# Section: User Prediction
st.header("📝 Predict a Transaction")
st.write("🔹 **Please enter 33 feature values (Time, V1, V2, ..., V28, Amount, Transaction_Speed, V_Sum, Is_Late_Night) separated by commas (,).**")

input_df = st.text_input('📥 Input All 33 Features:')
input_df_lst = input_df.split(',')

submit = st.button("🔍 Predict Transaction")

if submit:
    try:
        features = np.array(input_df_lst, dtype=np.float64)

        if features.shape[0] != 33:
            st.warning(f"⚠️ You entered {features.shape[0]} features. Please enter exactly **33 features**.")
        else:
            # Predict transaction type
            prediction = model.predict(features.reshape(1, -1))
            prediction_proba = model.predict_proba(features.reshape(1, -1))

            # Display Prediction
            if prediction[0] == 0:
                st.success("✅ **Legitimate Transaction Detected**")
                st.balloons()
            else:
                st.error("🚨 **Fraudulent Transaction Detected!**")
                st.snow()

            # Show confidence
            st.info(f"🧠 Model Confidence: **{np.max(prediction_proba) * 100:.2f}%**")

            # Generate AI-powered explanation
            ai_explanation = generate_ai_insight(features, prediction[0])
            st.write("💡 **AI Explanation**:")
            st.write(ai_explanation)

    except ValueError:
        st.error("⚠️ Please ensure all inputs are numeric and separated by commas.")
