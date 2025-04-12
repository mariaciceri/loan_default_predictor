import streamlit as st
import pandas as pd
from src.data_management import load_loan_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_status


def display_predict_status():
    """
    Display the prediction status page of the application.
    """
    path = "outputs/ml_pipeline/predict_status/v1/"
    status_pipeline_model = load_pkl_file(
        path + "pipeline_optimized_model.pkl")
    status_pipeline_cleaning = load_pkl_file(
        path + "pipeline_optimized_cleaning.pkl")

    st.header("Loan Status Prediction")
    st.info("""The client is interested in predicting whether a future loan
    applicant is likely to default, in order to support more informed and
    proactive lending decisions.""")
    st.divider()

    X_live = DrawInputsWidgets()
    if st.button("Run Analysis"):
        status_prediction = predict_status(
            X_live,
            status_pipeline_cleaning,
            status_pipeline_model
        )

        if status_prediction == 1:
            st.warning("The applicant is likely to default.")
        else:
            st.success("The applicant is unlikely to default.")


def DrawInputsWidgets():
    """Create input widgets for the user to enter loan data."""
    df = load_loan_data()

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = 'Upfront_charges'
        st_widget = st.number_input(
            "Enter the value for Upfront Charges",
            min_value=0.0,
            max_value=99_000_000.0,
            value=None, placeholder="Type a number..."
        )
    X_live[feature] = st_widget

    with col2:
        feature = 'Interest_rate_spread'
        st_widget = st.slider(
            "Select the Interest Rate Spread",
            min_value=-4.00,
            max_value=4.00,
            value=2.00
        )
    X_live[feature] = st_widget

    with col3:
        feature = 'credit_type'
        st_widget = st.selectbox(
            label="Select the Credit Type",
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col4:
        feature = 'rate_of_interest'
        st_widget = st.slider(
            "Select the Rate of Interest",
            min_value=0.00,
            max_value=10.00,
            value=3.00
        )
    X_live[feature] = st_widget

    st.write("Debt to Income Ratio")

    col5, col6 = st.columns(2)
    with col5:
        monthly_debt = st.number_input("Monthly Debt Payments", min_value=0.0)

    with col6:
        monthly_income = st.number_input("Monthly Income", min_value=0.0)

    if monthly_income > 0:
        dti = (monthly_debt / monthly_income) * 100
    else:
        dti = None

    X_live['dtir1'] = dti

    return X_live
