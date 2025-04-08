import streamlit as st
import pandas as pd

@st.cache_data
def load_loan_data():
    df = pd.read_csv("outputs/datasets/collection/LoanDefault.csv")
    return df

