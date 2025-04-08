import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_loan_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_status


def display_predict_status():
    """
    Display the prediction status page of the application.
    """
    #load the data
    path = "outputs/ml_pipeline/prefict_status/v1"
    status_pipeline_model = load_pkl_file(path + "pipeline_optimized_model.pkl")

    status_features = (pd.read_csv(path + "X_train.csv")
                       .columns
                       .to_list()
                       )
    
    st.header("Loan Status Prediction")
    st.info("""The client is interested in predicting whether a future loan
    applicant is likely to default, in order to support more informed and
    proactive lending decisions.""")
    st.divider()


def DrawInputsWidgets():

    df = load_loan_data()

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

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
            value= 2.00
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
            min_value=0,
            max_value=10.00,
            value= 3.00
        )
    X_live[feature] = st_widget

    with col5:
        feature = 'dtir1'
        st.write("Debt to Income Ratio")
        monthly_debt = st.number_input("Monthly Debt Payments", min_value=0.0)
        monthly_income = st.number_input("Monthly Income", min_value=0.0)

        if monthly_income > 0:
            dti = (monthly_debt / monthly_income) * 100
        else:
            dti = None
        
    X_live[feature] = dti

    # st.write(X_live)

    return X_live