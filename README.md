# 🏦 Bank Customer Churn Prediction & Analytics

This repository features an end-to-end Machine Learning solution designed to predict customer attrition (churn) for a retail bank. Utilizing an **Optimized XGBoost Classifier**, the project identifies high-risk customers and employs **Explainable AI (SHAP)** to provide transparent financial drivers behind every prediction.

---

## 🎯 Business Objective
In banking, customer retention is a primary driver of long-term profitability. As a Finance Associate, I developed this model to:
* **Minimize Revenue Leakage:** Identify at-risk customers with high sensitivity (**Recall**).
* **Strategic Allocation:** Optimize marketing spend by targeting customers with the highest probability of churn.
* **Transparency:** Provide "Black Box" model interpretability for audit and stakeholder review.

---

## 🛠️ Data Engineering & Synthetic Features
To move beyond basic demographics, I engineered **4 synthetic features** that capture the "financial stickiness" of a customer:

* **BalanceSalaryRatio:** Measures a customer's liquid wealth relative to income.
* **TenureByAge:** Proportion of a customer's life spent with the bank.
* **CreditScoreByAge:** Normalizes creditworthiness based on life stage.
* **EngagementScore:** A composite metric aggregating product usage and activity.

---

## 📈 Performance Results

| Metric | Score | Impact for the Bank |
| :--- | :--- | :--- |
| **Accuracy** | **82%** | High overall system reliability. |
| **Recall** | **71%** | **Strategic Success:** We identify 71% of all churners. |
| **Precision** | **55%** | 55% of alerts are true churners, reducing "Marketing Fatigue." |
| **F1-Score** | **0.62** | A robust balance between sensitivity and reliability. |

### **Strategic Optimization: The Precision-Recall Curve**
In banking, a **False Negative** (failing to catch a churner) is significantly more expensive than a **False Positive**. Consequently, I prioritized **Recall** during hyperparameter tuning.

![Precision-Recall Curve](precision_recall_curve.png)

---

## 🔍 Explainable AI (SHAP)
Regulatory compliance is paramount in finance. I integrated **SHAP** to visualize why a specific customer is flagged as a risk.

![SHAP Force Plot](shap_force_plot.png)

* **Red features** push the risk higher (e.g., higher Age).
* **Blue features** pull the risk lower (e.g., active membership).

---

## 🧠 Technical Challenges & Solutions
* **Imbalanced Classes:** Used `scale_pos_weight` in XGBoost and optimized for **F1-Score** rather than Accuracy.
* **Feature Pruning:** Validated that raw data + engineered ratios provided the best context.
* **Hyperparameter Tuning:** Conducted `RandomizedSearchCV` (25 iterations) to optimize depth and learning rates.

---

## 🚀 Deployment
The solution is deployed via a **Streamlit Web Application**, providing an intuitive dashboard for bank managers to receive instant risk assessments.

### **How to Run**
1. **Clone:** `git clone https://github.com/yourusername/bank-churn-prediction.git`
2. **Install:** `pip install -r requirements.txt`
3. **Run:** `streamlit run app.py`