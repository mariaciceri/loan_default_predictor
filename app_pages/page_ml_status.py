import streamlit as st
import matplotlib.pyplot as plt
import time
from src.data_management import load_sets, load_pkl_file
from src.machine_learning.evaluate_pipeline import evaluate_pipeline


def display_ml_status():
    """
    Display the machine learning status page of the application.
    """
    path = "outputs/ml_pipeline/predict_status/v1/"
    status_pipeline_model = load_pkl_file(
        path + "pipeline_optimized_model.pkl")
    status_pipeline_cleaning = load_pkl_file(
        path + "pipeline_optimized_cleaning.pkl")
    feature_importance = plt.imread(path + "feature_importance.png")
    confusion_matrix_train = plt.imread(path + "conf_matrix_Trainset.png")
    confusion_matrix_test = plt.imread(path + "conf_matrix_Testset.png")
    X_train, y_train = load_sets()

    st.header("ML Pipeline for Loan Status Prediction")
    st.info("""The project aimed to achieve a recall of at least 85% and an
    F1 score of 80%, as these metrics are most aligned with the business
    objectives. In the context of predicting loan defaults.""")

    scores_explained = """-> Recall is critical because it measures
    the model’s ability to correctly identify customers who are likely to
    default. Missing such cases could result in significant financial losses.
    -> The F1 score, which balances both precision and recall, ensures the
    model performs well across both metrics, avoiding too many false positives
    while still catching the true defaulters."""

    def stream_data():
        """
        Simulate streaming data by yielding one word at a time.
        """
        for word in scores_explained.split(" "):
            if word == "->":
                yield "\n" + word + " "
            else:
                yield word + " "
            time.sleep(0.05)

    st.success("""The project successfully surpassed these targets, achieving
    a recall and F1 score of 100%, indicating that the model was highly
    effective in identifying all potential defaults without compromising
    overall performance.""")

    if st.checkbox("Learn About Recall and F1 Scores"):
        st.write_stream(stream_data)

    st.divider()

    st.subheader("ML Pipeline Overview")
    st.write(f"""The machine learning pipelines for predictions are divided
    into two distinct stages:
* **Cleaning Pipeline**: This pipeline handles data preprocessing, including
    imputation of missing values, encoding categorical variables, and applying
    Yeo-Johnson transformation for normalization.
* **Model Pipeline**: This pipeline focuses on model training, which includes
    scaling the data using StandardScaler and training an Extra Trees
    Classifier for prediction.""")

    st.code(status_pipeline_cleaning, language="python")
    st.code(status_pipeline_model, language="python")

    st.divider()

    st.subheader("Feature Importance")
    st.write("""The model was trained based on the most important features
    identified during the analysis. The feature importance plot below
    illustrates the significance of each feature in the model's predictions.
    """)
    st.image(feature_importance, caption="Feature Importance Plot")

    st.divider()

    st.subheader("Confusion Matrix")

    st.write("""The confusion matrix provides a visual representation of the
    model's performance, showing the true positives, true negatives,
    false positives, and false negatives. This allows for a better
    understanding of the model's accuracy and its ability to correctly classify
    loan applicants.""")

    st.badge("Train Set Confusion Matrix", icon=":material/rubric:")
    st.image(confusion_matrix_train, caption="Train Set Confusion Matrix")

    st.badge("Test Set Confusion Matrix",
             icon=":material/rubric:",
             color="green")
    st.image(confusion_matrix_test, caption="Test Set Confusion Matrix")

    with st.spinner("Evaluating model performance..."):
        evaluate_pipeline(status_pipeline_model, X_train, y_train)
