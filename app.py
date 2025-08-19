# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- Load Saved Objects ---
# This function loads the trained model, scaler, and training columns.
# The @st.cache_resource decorator caches these objects so they don't reload on every interaction.
@st.cache_resource
def load_objects():
    try:
        with open('cph_model.pkl', 'rb') as file:
            cph_model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('training_columns.pkl', 'rb') as file:
            training_columns = pickle.load(file)
        return cph_model, scaler, training_columns
    except FileNotFoundError:
        return None, None, None

cph_model, scaler, training_columns = load_objects()

# --- App Structure ---
if cph_model is None:
    st.error("Model or necessary files not found. Please ensure `cph_model.pkl`, `scaler.pkl`, and `training_columns.pkl` are in the same folder.")
else:
    # --- App Title and Description ---
    st.title("Credit Risk Survival Analysis 📈")
    st.write(
        "This interactive app uses a Cox Proportional Hazards model to predict loan survival probability. "
        "Enter an applicant's details in the sidebar to generate their personalized risk curve over time."
    )

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Applicant Information")

    # Define lists for categorical features
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 
                'major_purchase', 'medical', 'small_business', 'car', 'moving', 
                'vacation', 'house', 'wedding', 'renewable_energy', 'educational']

    # Create input fields
    loan_amnt = st.sidebar.slider("Loan Amount ($)", 500, 40000, 15000, 500)
    int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 12.5, 0.5)
    annual_inc = st.sidebar.slider("Annual Income ($)", 10000, 1000000, 75000, 1000)
    dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 60.0, 20.0, 0.5)
    grade = st.sidebar.selectbox("Loan Grade", grades)
    purpose = st.sidebar.selectbox("Loan Purpose", sorted(purposes))

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Survival Probability"):

        # 1. Create a DataFrame from the user's inputs
        raw_input = pd.DataFrame({
            'loan_amnt': [loan_amnt], 'int_rate': [int_rate],
            'annual_inc': [annual_inc], 'dti': [dti],
            'grade': [grade], 'purpose': [purpose]
        })

        # 2. Preprocess the input data (the same way as in training)
        processed_input = pd.get_dummies(raw_input)
        processed_input = processed_input.reindex(columns=training_columns, fill_value=0)
        
        numeric_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
        processed_input[numeric_cols] = scaler.transform(processed_input[numeric_cols])

        # 3. Make the prediction
        predicted_survival = cph_model.predict_survival_function(processed_input)
        
        # --- Display the Output ---
        st.subheader("Prediction Results")
        st.write("The chart below shows the predicted probability that the loan will **not** have defaulted over time.")
        
        # Plot the survival curve
        fig, ax = plt.subplots()
        predicted_survival.plot(ax=ax, legend=False)
        ax.set_title("Predicted Loan Survival Curve")
        ax.set_xlabel("Time (in months)")
        ax.set_ylabel("Probability of Survival")
        ax.grid(True)
        st.pyplot(fig)

        # Display a specific metric
        try:
            prob_at_36_months = predicted_survival.loc[36].iloc[0]
            st.success(f"The predicted probability of survival at 36 months is: **{prob_at_36_months:.2%}**")
        except KeyError:
            st.warning("Prediction timeline is shorter than 36 months.")