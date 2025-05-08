readme: |
  # 💳 Credit Card Fraud Detection using Logistic Regression & Streamlit

  A Streamlit web app for credit card fraud detection using Logistic Regression. It uses a real-world imbalanced dataset, applies preprocessing and model training, and offers an interactive UI for predictions. Ideal for demonstrating ML workflows and fraud detection basics.

  ---

  ## 📌 Project Overview

  This project demonstrates how logistic regression can be used to identify fraudulent credit card transactions. It incorporates data preprocessing, model training, evaluation, and a user-friendly interface using Streamlit.

  ---

  ## 🧠 Machine Learning Pipeline

  - **Dataset**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  - **Algorithm**: Logistic Regression (binary classification)
  - **Preprocessing**:
    - Handling class imbalance
    - Feature scaling
  - **Evaluation Metrics**:
    - Accuracy
    - Confusion Matrix
    - ROC-AUC Score

  ---

  ## 🚀 Features

  - Exploratory Data Analysis (EDA) and visualization
  - Fraud prediction using logistic regression
  - Interactive user input through Streamlit
  - Clear metrics and model performance output

  ---

  ## 📂 Project Structure

credit-card-fraud-detection/
│
├── data/ # Dataset (CSV file)
├── notebooks/ # Jupyter Notebooks for EDA and modeling
├── model/ # Serialized model (Pickle file)
├── streamlit_app.py # Streamlit app script
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2.Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the Web App
streamlit run streamlit_app.py

