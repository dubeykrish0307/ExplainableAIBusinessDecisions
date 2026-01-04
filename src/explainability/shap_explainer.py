import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def create_shap_explainer(model, X_train):
    """
    Create SHAP explainer for a trained model.
    """
    explainer = shap.LinearExplainer(
        model,
        X_train,
        feature_perturbation="interventional"
    )
    return explainer

def compute_shap_values(explainer, X):
    """
    Compute SHAP values for given samples.
    """
    shap_values = explainer.shap_values(X)
    return shap_values

def plot_global_feature_importance(shap_values, X):
    """
    Global explanation: which features drive predictions overall.
    """
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()

def plot_local_explanation(explainer, shap_values, X, index):
    """
    Local explanation for a single prediction.
    """
    shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X.iloc[index],
        matplotlib=True,
        show=False
    )
    plt.title(f"Local Explanation for Applicant #{index}")
    plt.show()

def generate_decision_report(X, shap_values, index, threshold=0.0):
    """
    Generate a human-readable decision justification.
    """
    row = X.iloc[index]
    contributions = shap_values[index]

    report = []
    report.append("Decision Justification:\n")

    for feature, value, shap_val in zip(row.index, row.values, contributions):
        if abs(shap_val) > threshold:
            direction = "increases" if shap_val > 0 else "reduces"
            report.append(
                f"- {feature}: value={value} {direction} risk (impact={shap_val:.3f})"
            )

    return "\n".join(report)
