import streamlit as st
from sklearn.model_selection import cross_val_score


def evaluate_pipeline(pipeline, X_train, y_train):
    """
    Evaluate the machine learning pipeline for loan status prediction.
    """
    scores_recall = cross_val_score(pipeline,
                                    X_train,
                                    y_train,
                                    cv=5,
                                    scoring="recall")
    scores_f1 = cross_val_score(pipeline,
                                X_train,
                                y_train,
                                cv=5,
                                scoring="f1")

    mean_recall = scores_recall.mean().round(4)
    std_recall = scores_recall.std().round(4)
    mean_f1 = scores_f1.mean().round(4)
    std_f1 = scores_f1.std().round(4)

    st.success("✅ Evaluation complete! 🎯")
    st.write(f""" 💡 **Recall**: {mean_recall} ± {std_recall} \n
    📊 **F1 Score**: {mean_f1} ± {std_f1}
    """)
