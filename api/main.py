"""REST API for customer intelligence inference and business simulation."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from customer_intelligence.business.revenue_simulator import simulate_retention_impact
from customer_intelligence.data.validation import validate_customer_data
from customer_intelligence.inference import score_customers
from customer_intelligence.pipeline import run_training_pipeline


app = FastAPI(
    title="Customer Intelligence Platform API",
    version="0.1.0",
    description="Production-style APIs for churn, LTV, segmentation, retention, and revenue impact.",
)


class CustomerRecord(BaseModel):
    customer_id: str
    signup_date: str
    region: Literal["NA", "EU", "APAC", "LATAM"]
    acquisition_channel: str
    plan_type: str
    customer_age_days: int = Field(ge=0)
    monthly_spend: float = Field(ge=0)
    purchases_12m: int = Field(ge=0)
    sessions_30d: int = Field(ge=0)
    avg_session_minutes: float = Field(ge=0)
    support_tickets_90d: int = Field(ge=0)
    days_since_last_purchase: int = Field(ge=0)
    discount_usage_rate: float = Field(ge=0, le=1)
    nps: float = Field(ge=-100, le=100)
    total_revenue_12m: float = Field(ge=0)
    churned: int = Field(default=0, ge=0, le=1)


class ScoreRequest(BaseModel):
    customers: list[CustomerRecord]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train() -> dict[str, object]:
    return run_training_pipeline()


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, object]:
    df = pd.DataFrame([record.model_dump() for record in request.customers])
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    validation = validate_customer_data(df)
    if not validation.passed:
        raise HTTPException(status_code=422, detail=validation.errors)
    scored = score_customers(df)
    columns = [
        "customer_id",
        "churn_probability",
        "risk_tier",
        "predicted_ltv",
        "retention_action",
        "expected_saved_revenue",
        "retention_roi",
    ]
    return {"predictions": scored[columns].to_dict(orient="records"), "warnings": validation.warnings}


@app.post("/simulate")
def simulate(request: ScoreRequest, target_top_pct: float = 0.2, success_rate: float = 0.18) -> dict[str, float]:
    df = pd.DataFrame([record.model_dump() for record in request.customers])
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    scored = score_customers(df)
    return simulate_retention_impact(scored, target_top_pct=target_top_pct, success_rate=success_rate)
