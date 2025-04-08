import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
from src.data_management import load_loan_data


def display_status_study():
    """
    Display the status study page of the application.
    """
    #load data
    df = load_loan_data()

    #Most important features
    important_vars = ['Upfront_charges','Interest_rate_spread','credit_type',
                      'rate_of_interest','dtir1']
    
    st.header("Loan Status Study")
    st.subheader("Data Inspection")
    st.info("""The client wants to identify the top features that influence 
whether a customer defaults.""")
    
    st.badge("Check all the features and first rows below",
             icon=":material/info:", color="orange")

    if st.checkbox("Inspect loan default"):
        st.write(f"""The dataset has {df.shape[0]} rows and 
{df.shape[1]} columns. Check the first rows below""")
        st.write(df.head())

    st.divider()

    st.subheader("Feature Importance")
    st.write("""A correlation study was initially performed to explore 
relationships between variables and default, but all correlations were very 
weak (below 0.2). As a result, feature selection was carried out using a 
pipeline and a tree-based model. The most influential features identified 
through `.get_support()` and `feature_importances_` were:
**{important_vars}**.""")

    st.info("""we can see that...
""")
    
    df_imp = df.filter(important_vars + ["Status"])