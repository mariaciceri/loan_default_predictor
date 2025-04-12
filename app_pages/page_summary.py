import streamlit as st


def display_summary():
    """
    Display the summary page of the application.
    """
    st.title("Project Summary")
    st.info(
        f"**Project Terms & Jargon**\n\n"
        f"""* A **loan default** occurs when a borrower fails to make the
required payments on their loan. This can happen for various reasons, including
financial difficulties, job loss, or unexpected expenses.\n"""
        f"""* The **status** of a loan refers to its current state,
such as whether it is defaulted or not.\n\n"""
        f"**Project Dataset**\n\n"
        f"""* The dataset used in this project is the **Loan Default Dataset**
from **Kaggle**. It contains information about loan applicants and their loan
applications, including various features such as demographic information,
financial history, and loan details.\n"""
    )

    st.write(f"""* For more information, please check the
[README file](https://github.com/mariaciceri/loan_default_predictor).""")

    st.success(
       f"**Project Business Requirements**\n\n"
       f"""-> The client is interested in understanding the key factors
that contribute to a loan default. By identifying the most important features,
the organization can gain valuable insights into customer behavior and
risk profiles. This information will support decision-making processes, help
refine lending criteria, and enable the development of targeted strategies to
minimize financial risk.\n\n"""
       f"""-> The client is interested in implementing a predictive model
capable of determining the likelihood of a customer defaulting on a loan.
With this tool, the organization can proactively assess loan applications,
flag high-risk borrowers, and take preventative actions."""
    )
