# Interview Guide

Use this project to tell a production ML story, not only a model training story.

## 60-Second Pitch

This is a Customer Intelligence Platform for telecom churn. It uses the IBM Telco dataset to predict churn, estimate
lifetime value, segment customers, generate personas, recommend retention actions, and simulate financial impact.

The important point is that the system does not stop at "customer X may churn." It answers the business questions that
retention teams actually need:

- Which customers are at risk?
- Which customers are worth saving?
- Why are they at risk?
- What action should the business take?
- What ROI can the business expect?

## What To Show First

1. Open the live Streamlit demo.
2. Show the Churn and LTV page.
3. Move to the Retention Simulator and explain why the LTV-v2 ROI is now conservative after removing revenue leakage.
4. Show the architecture diagram in the README.
5. Mention that the system includes FastAPI, MLflow, Docker, CI, tests, and monitoring hooks.

## Strong Technical Answers

**Why Logistic Regression won?**  
The strongest churn signals in this dataset are structured and mostly monotonic: contract type, tenure, payment method,
monthly charges, and support services. Logistic Regression is interpretable, stable, and performs slightly better than
more complex models here.

**Why is this a platform, not a model?**  
The project includes ingestion, validation, feature engineering, multiple model families, segmentation, explainability,
LTV, recommendations, revenue simulation, API serving, dashboarding, Docker, MLflow, tests, CI, and monitoring.

**How is business impact calculated?**  
The simulator ranks customers by churn risk and predicted LTV. It estimates saved value using churn probability,
margin-adjusted predicted LTV, intervention success rate, and campaign cost.

**What would you improve next?**  
Add uplift modeling, A/B testing, feature store support, model registry promotion, fairness checks, and scheduled
retraining with drift-triggered alerts.

## Risks And Tradeoffs

- IBM Telco is a small public dataset, so the architecture matters more than raw model complexity.
- Some engagement and RFM variables are transparent proxies because the dataset is subscription-level, not event-level.
- The ROI simulator uses configurable assumptions and should be calibrated with real campaign results in production.
- More complex models are included, but the simplest high-performing model is preferred for maintainability.
