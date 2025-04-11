# Loan Default Predictor

[Loan Default Predictor](https://loan-default-predictor-0ouz.onrender.com) is a machine learning project that aims to predict whether a loan applicant is likely to default. Using a dataset of financial and demographic features, the model can predict the probability of defaulting. The project involves data preprocessing, feature selection, model training, and evaluation. It’s built using Python with tools like Pandas, Scikit-learn, and Matplotlib. The goal is to support more informed lending decisions with a simple, explainable prediction interface.

## Table of Content

+ [Dataset Content](#dataset-content)
+ [Business Requirements](#business-requirements)
+ [Hypothesis](#hypothesis)
+ [Map of Business Requirements to Data Analytics Tasks](#map-of-business-requirements-to-data-analytics-tasks)
+ [ML Business Case](#ml-business-case)
+ [Dashboard Design](#dashboard-design)
+ [Technologies Used](#technologies-used)
+ [Deployment](#deployment)
+ [Testing](#testing)
+ [Credits](#credits)

## Dataset Content

The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset), includes approximately 149,000 loan cases across 34 columns. Each row represents an individual loan and contains detailed information such as borrower demographics, loan characteristics, and payment details.

| Attribute      | Information          | Unit         |
|----------------|----------------------|--------------|
| ID | Client's loan application ID | Unique numerical identifier |
| year | Year of the application | 2019 |
| loan_limit | Whether the loan meets specific standards | conforming (cf) or non-conforming (ncf) |
| Gender | Gender of the applicant | male, female, joint, sex not available |
| approv_in_adv | Whether the loan is pre-approved | pre, nopre |
| loan_type | Type of the loan | type1, type2, type3 |
| loan_purpose | Purpose of the loan | p1, p2, p3, p4 |
| Credit_Worthiness | How likely is to the loan to be repaid | l1, l2 |
| open_credit | Indicates if the applicant has any open credit accounts | opc, nopc |
| business_or_commercial | Indicates if the loan if for business or personal purposes | business (ob/c) or personal (nob/c)
| loan_amount | The amount of money being borrowed | Numerical value |
| rate_of_interest | Interest rate on the loan | Numerical value |
| Interest_rate_spread | The difference between the loan's interest rate and a benchmark rate | Numerical value |
| Upfront_charges | Initial charges/down payment | Numerical value |
| term | Duration of the loan in months | Numerical value |
| Neg_ammortization | Indicates if the loan allows negative ammortization or not | neg_amm or not_neg |
| interest_only | Indicates if the loan has a interest-only payment | int_only or not_int |
| lump_sum_payment | Indicates whether a lump sum payment is due at the end of the loan term | lpsm or not_lpsm |
| property_value | The value of the property | Numerical value |
| construction_type | The type of the construction | site build (sb) or manufactured home (mh) |
| occupancy_type | The type of occupancy | primary residence (pr), secondary residence (sr) or investment property (ir) | 
| Secured_by | Indicates the type of asset used as collateral for the loan | home or land |
| total_units | Number of units being financed | 1U, 2U, 3U, 4U |
| income | Applicant's annual income | Numerical value |
| credit_type | Applicant's credit information source | Credit Information Bureau (CIB), CRIF Credit Information Bureau (CRIF), Experian (EXP) or Equifax (EQUI) |
| Credit_Score | Applicant's credit score | Numerical value |
| co-applicant_credit_type | Co-applicant's credit information | Credit Information Bureau (CIB) or Experian (EXP) |
| age | Applicant's age | <25, 25-34, 35-44, 45-54, 55-64, 65-74 and >74 |
| submission_of_application | Indicates the method through which the application was submitted | To institution (to_isnt) or not to institution (not_inst) |
| LTV | Loan-to-value ratio | Numerical value |
| Region | Geographic region of the property | North, south, central, North-East |
| Security_Type | Type of collateral backing the loan | direct ot indirect |
| Status | If the loan was defaulted (1) or not (0) | Numerical value |
| dtir1 | Debt-to-income ratio | Numerical value |

## Business Requirements


## Hypothesis


## Map of Business Requirements to Data Analytics Tasks


## ML Business Case


## Dashboard Design


## Technologies Used


## Deployment


## Testing


## Credits