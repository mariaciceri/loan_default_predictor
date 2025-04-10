import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_loan_data():
    df = pd.read_csv("outputs/datasets/collection/LoanDefault.csv")
    return df

def load_sets():
    X_train = pd.read_csv("outputs/ml_pipeline/predict_status/v1/X_train.csv")
    y_train = pd.read_csv("outputs/ml_pipeline/predict_status/v1/y_train.csv")

    return X_train, y_train

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)

