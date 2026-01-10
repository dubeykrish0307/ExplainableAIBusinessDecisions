# Explainable Credit Risk Decision System

An end-to-end **explainable machine learning decision system** for credit risk assessment, built with a strong focus on **transparency, auditability, and responsible AI**.

This project goes beyond prediction accuracy and demonstrates how machine learning can be used to support **real business decisions** in **regulated domains** such as banking and finance.

---

## 🚀 Live Application

🔗 **Public Demo (Streamlit App)**  
👉 <PASTE https://explainable-credit-risk-decision-system.streamlit.app/>

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
```
explainable-credit-risk-decision-system/
├── app/
│ └── main.py # Streamlit application
│
├── src/
│ ├── features/ # Feature engineering pipeline
│ ├── models/ # Training & model comparison
│ ├── explainability/ # SHAP explainers
│ ├── ethics/ # Bias & fairness analysis
│ └── reports/ # Decision report generator
│
├── data/
│ └── raw/ # Dataset (ignored in git)
│
├── reports/
│ └── decision_reports/ # Generated audit artifacts
│
├── scripts/
│ └── download_dataset.py # Reproducible dataset download
│
├── requirements.txt
└── README.md
```

## 🔍 Explainability (Not a Black Box)

### Global Explanations
- Identify which features generally drive default risk
- Useful for policy review and stakeholder communication

### Local Explanations
- Feature-level contribution for individual applicants
- Explains *why* a specific decision was made
- Supports human review and accountability

SHAP (Shapley Additive Explanations) is used to provide consistent, theoretically grounded explanations.

---

## ⚖️ Responsible AI & Fairness

The system explicitly evaluates bias across sensitive attributes such as:

- Age
- Gender proxies
- Nationality proxies

Bias is **measured before mitigation**, aligning with EU and German regulatory thinking.  
This avoids hiding bias and instead enables **informed governance decisions**.

---

## 🧾 Audit-Ready Decision Reports

Each decision can generate a persistent, timestamped report containing:

- Default risk probability
- Final decision (Approve / Review / Reject)
- Feature-level justification
- UTC timestamp
- Notes for audit and compliance

These reports are stored under:
This ensures decisions are **traceable, reviewable, and auditable**.

---

## 📊 Model Comparison & Trade-offs

Two models are evaluated under identical conditions:

| Model | Strength | Trade-off |
|------|---------|-----------|
| Logistic Regression | High interpretability, strong recall for defaults | Slightly lower accuracy |
| Random Forest | Higher raw accuracy | Lower recall for risky applicants, higher governance cost |

**Key insight:**  
Higher accuracy does not necessarily lead to better business outcomes when error costs are asymmetric.

---

## 📦 Dataset

This project uses the **German Credit Dataset** from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- License: Public / Research Use
- Raw data is intentionally excluded from version control

## ▶️ Run Locally
1. Create virtual environment
   ```
    python -m venv venv
    source venv/bin/activate        # Linux/Mac
    venv\Scripts\activate           # Windows
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run the application
   ```
   streamlit run app/main.py
   ```

## 🌍 Deployment
The application is deployed on Streamlit Community Cloud to make explainable decision logic accessible to non-technical stakeholders without unnecessary infrastructure complexity.

### 👤 Author
Krish Dubey
Bachelor’s Student — Computer Science
Focus: Machine Learning Systems, Explainable AI, Responsible Decision-Making

### 📄 License
This project is intended for educational and demonstration purposes.
