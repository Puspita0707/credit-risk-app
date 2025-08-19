# Credit Risk Survival Analysis & Predictive Modeling

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](
credit-risk-app-mhzzflgzyb3pbc9gdka2ze.streamlit.app)
An end-to-end data science project that predicts not just *if* a loan will default, but *when*, using survival analysis. The final model is deployed as an interactive web application with Streamlit for real-time risk assessment.

![Streamlit App Screenshot](<img width="1407" height="797" alt="Screenshot 2025-08-19 at 11 11 42 PM" src="https://github.com/user-attachments/assets/41fbe15b-669d-471a-95f5-c13ad1359d0e" />
)

---
## 📋 Project Overview

This project addresses a key challenge in the financial sector: understanding the dynamics of credit risk over time. While traditional classification models predict the probability of default, this project implements a **Cox Proportional Hazards survival model** to forecast the timing of default. This provides a more nuanced view of risk, enabling better financial planning and proactive risk management.

The analysis is based on a large-scale dataset from LendingClub, and the final, validated model is deployed as a user-friendly web application for real-time risk assessment of new loan applicants.

---
## ✨ Key Features

* **Survival Analysis:** Implements a Cox Proportional Hazards model to predict loan survival probability over time.
* **Model Evaluation:** Achieves a **Concordance Index of 0.71**, indicating strong predictive accuracy in ranking borrower risk.
* **Key Driver Analysis:** Quantifies the impact of key features like loan grade, purpose, and interest rate on default risk.
* **Interactive Web App:** A deployed Streamlit application allows for real-time prediction and scenario analysis for new loan applicants.

---
## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Lifelines, Matplotlib
* **Deployment:** Streamlit, GitHub, Streamlit Community Cloud

---
## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
The application will be available at `http://localhost:8501`.

---
## 📂 File Structure

* `app.py`: The main script for the Streamlit web application.
* `requirements.txt`: A list of the Python packages required to run the app.
* `cph_model.pkl`: The saved, trained Cox Proportional Hazards model.
* `scaler.pkl`: The saved Scikit-learn scaler object.
* `training_columns.pkl`: The list of training columns for consistent preprocessing.
* `notebook/`: (Optional) Folder containing the Jupyter Notebook used for the analysis and model training.
