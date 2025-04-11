import streamlit as st
import seaborn as sns
import pandas as pd
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
    st.write(f"""A correlation study was initially performed to explore 
relationships between variables and default, but all correlations were very 
weak (below 0.2). As a result, feature selection was carried out using a 
pipeline and a tree-based model. The most influential features identified 
through `.get_support()` and `feature_importances_` were:
**{important_vars}**.""")

    st.info("""Analysing the graphs for the most importante features,
we can see that:
* The default status appears to occur at lower upfront charges.
* There is no data available for defaulted loans in the Interest Rate Spread 
feature, as it is missing for those instances. This absence may suggest that
the individuals who defaulted did not provide this value, which could explain
why the plot shows data only for non-defaulted individuals. Despite this,
the model still considers this feature important, possibly because it is 
handled differently during training (such as imputation or exclusion).
* All credit types from EQUI (Equifax) have defaulted, in contrast to the 
other three categories, where approximately 1/5 of the cases are default.
* The defaulted loans tend to have a slightly higher median rate of 
interest than non-defaulted loans, between 3 and 5.
* The defaulted loans tend to have a larger IOR (between 30 and 50) compared 
to non-defaulted loans, indicating a wider spread of values, 
which might suggest a more varied debt-to-income ratio among those who default.
""")

    df_imp = df.filter(important_vars + ["Status"])

    if st.checkbox("Status Levels per Variable"):
        plot_default_levels_per_variable(df_imp)

    st.divider()
    st.subheader("Project Hypothesis and Validation")
    st.write(f"""**->** It is believed that key factors such as the interest
rate, income, loan-to-value ratio, debt-to-income ratio, and credit score are
critical in predicting whether a loan applicant is likely to default.""")
    st.success("""A correlation analysis was conducted, but it did not reveal
any strong relationships between the variables and the default status.
Consequently, feature selection was carried out using a pipeline with a
tree-based model. The key features identified for the model were upfront
charges, interest rate spread, rate of interest, credit type, and
debt-to-income ratio. This process confirmed that some of the initial
assumptions were correct, but not all.""")
    
    st.write(f"""**->** Applicants who apply for loans through online or
automated systems may have a higher likelihood of defaulting, as these methods
could attract borrowers who may not be able to assess the loan’s long-term
impact.""")
    st.warning("""The analysis did not show any significant impact of the
submission type on the likelihood of defaulting.""")

def plot_default_levels_per_variable(df):
    """
    Plot the default levels per variable.
    """
    list_of_variables = df.columns[df.columns != 'Status']
    target_var = 'Status'

    for var in list_of_variables:
        if df[var].dtype == 'object':
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.countplot(data=df, x=var, hue=target_var,
                        order=df[var].value_counts().index, stat="percent")
            plt.title(f"{var}", fontsize=20, y=1.05)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.boxplot(data=df, x="Status", y=var, ax=ax)
            ax.set_title(f'{var} by Default Status',fontsize=20, y=1.05)
            ax.set_xlabel("Default Status (0 = No, 1 = Yes)")
            ax.set_ylabel(var)
            st.pyplot(fig)
    