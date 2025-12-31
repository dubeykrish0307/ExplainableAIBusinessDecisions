import pandas as pd
import numpy as np

NUMERICAL_FEATURES = [
    "loan_duration_months",
    "credit_amount",
    "installment_rate",
    "residence_duration",
    "age",
    "existing_credits",
    "num_dependents"
]

ORDINAL_CATEGORICAL_FEATURES = {
    "employment_duration": {
        "A71": 0,  # unemployed
        "A72": 1,  # < 1 year
        "A73": 2,  # 1–4 years
        "A74": 3,  # 4–7 years
        "A75": 4   # >= 7 years
    },
    "savings_account": {
        "A61": 0,
        "A62": 1,
        "A63": 2,
        "A64": 3,
        "A65": 0
    }
}

NOMINAL_CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "loan_purpose",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

def transform_target(df : pd.DataFrame) -> pd.DataFrame:
    """
    Convert credit_risk to binary:
    0 -> Good Credit
    1 -> Bad Credit (default risk)
    """
    df = df.copy()
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})
    return df

def ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for feature, mapping in ORDINAL_CATEGORICAL_FEATURES.items():
        df[feature] = df[feature].map(mapping)
    
    return df

def encode_nominal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df = pd.get_dummies(
        df,
        columns=NOMINAL_CATEGORICAL_FEATURES,
        drop_first=True
    ) 
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df = transform_target(df)
    df = encode_ordinal_features(df)
    df = encode_nominal_features(df)
    
    return df
