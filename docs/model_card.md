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

## Risks

- Historical intervention bias can make past retained customers look naturally low-risk.
- Acquisition channels may encode marketing mix changes rather than customer quality.
- NPS and support data can be sparse or delayed.
- Models should be monitored for drift after pricing, product, or onboarding changes.
