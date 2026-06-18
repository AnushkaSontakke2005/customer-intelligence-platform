"""Data ingestion utilities.

The primary dataset for this project is IBM's Telco Customer Churn dataset. A
synthetic generator is kept as a fallback for smoke tests and demos when the
public CSV is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from customer_intelligence.config import RAW_DATA_DIR

TELCO_DATA_PATH = RAW_DATA_DIR / "ibm_telco_customer_churn.csv"

TELCO_SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _yes_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("yes").astype(int)


def transform_telco_to_customer_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """Map the IBM Telco schema to the platform's canonical customer schema.

    The Telco dataset is subscription-level, not transaction-event-level. For
    RFM, engagement, and support features we create transparent telecom proxies
    from tenure, charges, service adoption, contract type, and support add-ons.
    These proxies avoid using the target label, so they remain valid for model
    training and inference.
    """

    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"] * out["tenure"])
    tenure = out["tenure"].clip(lower=0)
    service_count = sum(_yes_flag(out[column]) for column in TELCO_SERVICE_COLUMNS)
    contract = out["Contract"].astype(str)
    electronic_check = out["PaymentMethod"].astype(str).str.contains("Electronic check", case=False, regex=False)

    canonical = pd.DataFrame(
        {
            "customer_id": out["customerID"],
            "signup_date": pd.Timestamp.today().normalize() - pd.to_timedelta(tenure * 30, unit="D"),
            "region": "IBM_Telco",
            "acquisition_channel": out["PaymentMethod"],
            "plan_type": out["Contract"],
            "customer_age_days": (tenure * 30).astype(int),
            "monthly_spend": out["MonthlyCharges"].astype(float),
            "purchases_12m": tenure.clip(lower=1, upper=12).astype(int),
            "sessions_30d": (service_count * 3 + _yes_flag(out["PaperlessBilling"]) * 2 + 1).astype(int),
            "avg_session_minutes": (out["MonthlyCharges"] / 3 + service_count * 4).round(2),
            "support_tickets_90d": (
                (out["TechSupport"].astype(str).str.lower().eq("no") & out["InternetService"].ne("No")).astype(int)
                + (out["OnlineSecurity"].astype(str).str.lower().eq("no") & out["InternetService"].ne("No")).astype(int)
                + contract.eq("Month-to-month").astype(int)
                + electronic_check.astype(int)
            ),
            "days_since_last_purchase": np.select(
                [contract.eq("Two year"), contract.eq("One year"), contract.eq("Month-to-month")],
                [15, 30, 60],
                default=45,
            ),
            "discount_usage_rate": np.select(
                [contract.eq("Two year"), contract.eq("One year"), contract.eq("Month-to-month")],
                [0.05, 0.15, 0.35],
                default=0.2,
            ),
            "nps": np.clip(
                25
                + _yes_flag(out["TechSupport"]) * 18
                + _yes_flag(out["OnlineSecurity"]) * 12
                + _yes_flag(out["OnlineBackup"]) * 6
                - contract.eq("Month-to-month").astype(int) * 10
                - electronic_check.astype(int) * 8
                - out["InternetService"].eq("Fiber optic").astype(int) * 5,
                -100,
                100,
            ),
            "total_revenue_12m": (out["MonthlyCharges"] * tenure.clip(lower=1, upper=12)).round(2),
            "churned": out["Churn"].map({"Yes": 1, "No": 0}).astype(int),
            "tenure_months": tenure.astype(int),
            "senior_citizen": out["SeniorCitizen"].astype(int),
            "service_count": service_count.astype(int),
            "has_fiber": out["InternetService"].eq("Fiber optic").astype(int),
            "month_to_month_contract": contract.eq("Month-to-month").astype(int),
            "electronic_check": electronic_check.astype(int),
            "automatic_payment": out["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int),
            "paperless_billing_flag": _yes_flag(out["PaperlessBilling"]),
            "tech_support_flag": _yes_flag(out["TechSupport"]),
            "online_security_flag": _yes_flag(out["OnlineSecurity"]),
            "gender": out["gender"],
            "partner": out["Partner"],
            "dependents": out["Dependents"],
            "phone_service": out["PhoneService"],
            "multiple_lines": out["MultipleLines"],
            "internet_service": out["InternetService"],
            "online_security": out["OnlineSecurity"],
            "online_backup": out["OnlineBackup"],
            "device_protection": out["DeviceProtection"],
            "tech_support": out["TechSupport"],
            "streaming_tv": out["StreamingTV"],
            "streaming_movies": out["StreamingMovies"],
            "contract": out["Contract"],
            "paperless_billing": out["PaperlessBilling"],
            "payment_method": out["PaymentMethod"],
            "source_dataset": "IBM Telco Customer Churn",
        }
    )
    return canonical


def load_telco_customer_data(path: str | Path = TELCO_DATA_PATH) -> pd.DataFrame:
    """Load and transform the IBM Telco Customer Churn dataset."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"IBM Telco dataset not found at {path}. Download it to data/raw/ibm_telco_customer_churn.csv."
        )
    return transform_telco_to_customer_intelligence(pd.read_csv(path))


def generate_synthetic_customer_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create realistic customer, engagement, transaction, and churn fields."""

    rng = np.random.default_rng(seed)
    acquisition_channels = np.array(["organic", "paid_search", "social", "referral", "partner"])
    plans = np.array(["free", "basic", "premium", "family", "enterprise"])
    regions = np.array(["NA", "EU", "APAC", "LATAM"])

    customer_age_days = rng.integers(15, 1800, n_customers)
    monthly_spend = rng.gamma(shape=2.4, scale=18, size=n_customers).round(2)
    monthly_spend *= rng.choice([0.0, 0.7, 1.0, 1.6, 2.5], n_customers, p=[0.18, 0.22, 0.34, 0.2, 0.06])
    sessions_30d = rng.poisson(lam=np.clip(monthly_spend / 8 + 4, 1, 40), size=n_customers)
    support_tickets_90d = rng.poisson(lam=np.clip(2.1 - sessions_30d / 20, 0.05, 3), size=n_customers)
    days_since_last_purchase = rng.integers(0, 180, n_customers)
    purchases_12m = rng.poisson(lam=np.clip(monthly_spend / 20 + 1.5, 0.1, 12), size=n_customers)
    discount_usage_rate = rng.beta(2, 6, n_customers).round(3)
    nps = np.clip(rng.normal(42 + sessions_30d * 0.8 - support_tickets_90d * 7, 18), -100, 100).round(0)

    churn_logit = (
        -1.4
        + 0.015 * days_since_last_purchase
        + 0.45 * support_tickets_90d
        - 0.035 * sessions_30d
        - 0.012 * nps
        + 0.7 * (monthly_spend == 0)
        + 0.45 * discount_usage_rate
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churned = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_customers)],
            "signup_date": pd.Timestamp.today().normalize()
            - pd.to_timedelta(customer_age_days, unit="D"),
            "region": rng.choice(regions, n_customers, p=[0.42, 0.28, 0.22, 0.08]),
            "acquisition_channel": rng.choice(acquisition_channels, n_customers),
            "plan_type": rng.choice(plans, n_customers, p=[0.18, 0.25, 0.34, 0.18, 0.05]),
            "customer_age_days": customer_age_days,
            "monthly_spend": monthly_spend,
            "purchases_12m": purchases_12m,
            "sessions_30d": sessions_30d,
            "avg_session_minutes": rng.gamma(4, 6, n_customers).round(2),
            "support_tickets_90d": support_tickets_90d,
            "days_since_last_purchase": days_since_last_purchase,
            "discount_usage_rate": discount_usage_rate,
            "nps": nps,
            "churned": churned,
        }
    )
    df["total_revenue_12m"] = (df["monthly_spend"] * rng.uniform(8, 12, n_customers)).round(2)
    return df


def ingest_customers(path: str | Path | None = None, n_customers: int = 5000) -> pd.DataFrame:
    """Load customer data from CSV, preferring IBM Telco when available."""

    if path:
        raw = pd.read_csv(path)
        if "customerID" in raw.columns and "Churn" in raw.columns:
            return transform_telco_to_customer_intelligence(raw)
        raw["signup_date"] = pd.to_datetime(raw["signup_date"])
        return raw
    if TELCO_DATA_PATH.exists():
        return load_telco_customer_data(TELCO_DATA_PATH)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_customer_data(n_customers=n_customers)
    df.to_csv(RAW_DATA_DIR / "customers.csv", index=False)
    return df


if __name__ == "__main__":
    frame = ingest_customers()
    print(f"Ingested {len(frame):,} customers into {RAW_DATA_DIR / 'customers.csv'}")
