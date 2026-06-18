"""Automated data quality checks for customer intelligence data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "signup_date",
    "region",
    "acquisition_channel",
    "plan_type",
    "customer_age_days",
    "monthly_spend",
    "purchases_12m",
    "sessions_30d",
    "avg_session_minutes",
    "support_tickets_90d",
    "days_since_last_purchase",
    "discount_usage_rate",
    "nps",
    "total_revenue_12m",
    "churned",
}


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


def validate_customer_data(df: pd.DataFrame) -> ValidationResult:
    """Run schema, completeness, uniqueness, range, and label checks."""

    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    if "customer_id" in df and df["customer_id"].duplicated().any():
        errors.append("customer_id must be unique.")

    numeric_ranges = {
        "customer_age_days": (0, 3650),
        "monthly_spend": (0, 10000),
        "purchases_12m": (0, 1000),
        "sessions_30d": (0, 10000),
        "support_tickets_90d": (0, 500),
        "days_since_last_purchase": (0, 3650),
        "discount_usage_rate": (0, 1),
        "nps": (-100, 100),
        "total_revenue_12m": (0, 1_000_000),
    }
    for column, (low, high) in numeric_ranges.items():
        if column in df:
            invalid = ~df[column].between(low, high)
            if invalid.any():
                errors.append(f"{column} has {int(invalid.sum())} values outside [{low}, {high}].")

    null_rates = df.isna().mean()
    high_nulls = null_rates[null_rates > 0.05]
    for column, rate in high_nulls.items():
        warnings.append(f"{column} null rate is {rate:.1%}; investigate source reliability.")

    if "churned" in df and not set(df["churned"].dropna().unique()).issubset({0, 1}):
        errors.append("churned must be binary 0/1.")

    return ValidationResult(passed=not errors, errors=errors, warnings=warnings)
