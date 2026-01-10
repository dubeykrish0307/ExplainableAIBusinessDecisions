# Explainable Credit Risk Decision System

An end-to-end **explainable machine learning decision system** for credit risk assessment, built with a strong focus on **transparency, auditability, and responsible AI**.

This project goes beyond prediction accuracy and demonstrates how machine learning can be used to support **real business decisions** in **regulated domains** such as banking and finance.

---

## 🚀 Live Application

🔗 **Public Demo (Streamlit App)**  
👉 <PASTE YOUR STREAMLIT URL HERE>

The deployed application allows users to:
- Inspect individual loan applicants
- View default risk probabilities
- See global and local explanations (SHAP)
- Generate audit-ready decision reports

---

## 📌 Problem Statement

In financial institutions, machine learning models are often **distrusted** because:

- Predictions are hard to explain
- Decisions cannot be justified to humans
- Regulatory and ethical risks are ignored
- No audit trail exists for automated decisions

**Goal of this project:**  
Build a system that not only predicts credit default risk, but clearly explains **why a decision is made**, evaluates **bias**, and leaves a **persistent audit trail**.

---

## 🧠 Decision Context (Business Logic)

Model outputs are mapped to real-world business actions:

| Default Risk Probability | Decision |
|--------------------------|----------|
| < 30%                    | APPROVE |
| 30% – 60%                | MANUAL REVIEW |
| > 60%                    | REJECT |

This mirrors how credit decisions are handled in real financial institutions.

---

## 🏗️ System Capabilities

- Interpretable baseline model (Logistic Regression)
- High-performance comparison model (Random Forest)
- Feature engineering with business justification
- Global and local explainability using SHAP
- Human-readable decision justifications
- Bias & fairness analysis on sensitive attributes
- Audit-ready decision report generation
- Interactive Streamlit web application
- Public cloud deployment

---

## 🧩 Architecture Overview

