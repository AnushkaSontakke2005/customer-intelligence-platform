# Model Card

## Intended Use

The churn model ranks customers by near-term churn risk so retention teams can prioritize proactive intervention. It is
not intended to deny service, change contractual terms, or make irreversible customer decisions.

## Training Data

The repository uses the public IBM Telco Customer Churn dataset with 7,043 subscription customers. In production,
training data should come from governed customer, billing, engagement, support, and lifecycle tables.

## Features

- Lifecycle: tenure, contract type, customer age proxy, recency proxy.
- Billing and value: monthly charges, total charges, payment method, paperless billing, annualized revenue, RFM score.
- Service adoption: phone, multiple lines, internet service, streaming, online backup, device protection.
- Risk drivers: month-to-month contract, fiber, electronic check, lack of tech support, lack of online security.
- Household attributes: senior citizen, partner, dependents, gender.

## Algorithms

- Logistic Regression: interpretable baseline.
- Random Forest: nonlinear robust baseline.
- XGBoost, LightGBM, CatBoost: high-performing boosted tree candidates for tabular data.

## Evaluation

Primary metric: ROC-AUC. Secondary metrics: average precision and F1. On the current IBM Telco run, Logistic Regression
is selected with ROC-AUC of 0.847, average precision of 0.665, and F1 of 0.618. Business evaluation happens through the
retention simulator because a model that improves targeting economics is more valuable than a model that only improves
leaderboard metrics.

The production decision threshold is selected with a simple business cost matrix rather than defaulting to 0.5. The
current best threshold is 0.132, using a $12 false-positive outreach cost and a $150 false-negative missed-churn cost.

The LTV v2 model uses an expected margin target based on churn hazard:
`monthly_spend * 0.62 / (churn_probability + 1e-6)`. Direct revenue-derived inputs such as monthly charges, total
charges, annual revenue, monetary value, RFM score, revenue-per-purchase, and price-derived engagement proxies are
excluded from the LTV feature set to reduce target leakage. The v2 holdout metrics are R2 0.869, MAE 45.49, and RMSE
114.36. No encoded LTV input feature has Pearson correlation above 0.95 with the v2 target.

## Risks

- Historical intervention bias can make past retained customers look naturally low-risk.
- Acquisition channels may encode marketing mix changes rather than customer quality.
- NPS and support data can be sparse or delayed.
- Models should be monitored for drift after pricing, product, or onboarding changes.
