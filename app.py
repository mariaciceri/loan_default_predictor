import streamlit as st
import os


from app_pages.multipage import MultiPage
from app_pages.page_summary import display_summary
from app_pages.page_predict_status import display_predict_status
from app_pages.page_status_study import display_status_study
from app_pages.page_ml_status import display_ml_status


app = MultiPage("Loan Default Predictor")

app.add_page("Summary", display_summary)
app.add_page("Status Study", display_status_study)
app.add_page("Predict Status", display_predict_status)
app.add_page("ML Status", display_ml_status)

if __name__ == "__main__":
    port = os.getenv("PORT", 8501)
    st.run(port=port)