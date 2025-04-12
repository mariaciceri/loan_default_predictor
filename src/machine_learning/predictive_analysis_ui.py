import streamlit as st
import pandas as pd


def predict_status(X_live, status_pipeline_cleaning, status_pipeline_model):
    # Preprocess the input data
    X_live_cleaned = status_pipeline_cleaning.transform(X_live)

    # Make predictions using the trained model
    status_prediction = status_pipeline_model.predict(X_live_cleaned)
    status_prediction_proba = status_pipeline_model.predict_proba(
        X_live_cleaned
        )

    # Display the prediction results
    status_probability = status_prediction_proba[0][status_prediction]*100
    if status_prediction == 1:
        status_result = 'will'
    else:
        status_result = 'will not'

    st.write(f"""There is a {status_probability[0].round(2)}% chance that the
applicant {status_result} default.""")

    return status_prediction[0]
