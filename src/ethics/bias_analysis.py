import pandas as pd
import numpy as np

SENSITIVE_FEATURES = [
    "age",
    "personal_status_sex",
    "foreign_worker"
]

def outcome_rate_by_group(df, feature, target="credit_risk"):
    """
    Compute default rate by group.
    """
    rates = (
        df.groupby(feature)[target]
        .mean()
        .sort_values(ascending=False)
    )
    return rates

def prediction_rate_by_group(df, feature, prediction_col):
    """
    Compare predicted risk by group.
    """
    rates = (
        df.groupby(feature)[prediction_col]
        .mean()
        .sort_values(ascending=False)
    )
    return rates

def generate_bias_report(df, prediction_col="predicted_risk"):
    """
    Generate a textual bias analysis summary.
    """
    report = []
    report.append("Bias & Fairness Assessment\n")

    for feature in SENSITIVE_FEATURES:
        report.append(f"\nFeature: {feature}")
        report.append("Observed Default Rate:")
        report.append(str(outcome_rate_by_group(df, feature)))
        report.append("\nModel Predicted Risk Rate:")
        report.append(str(prediction_rate_by_group(df, feature, prediction_col)))

    return "\n".join(report)
