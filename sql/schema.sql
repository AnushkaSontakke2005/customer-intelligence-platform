-- Customer Intelligence Platform warehouse schema.
-- Designed for PostgreSQL/Snowflake-style analytics and ML feature generation.

CREATE TABLE dim_customer (
    customer_id VARCHAR(64) PRIMARY KEY,
    signup_date DATE NOT NULL,
    region VARCHAR(32) NOT NULL,
    acquisition_channel VARCHAR(64) NOT NULL,
    plan_type VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fact_customer_activity_daily (
    activity_date DATE NOT NULL,
    customer_id VARCHAR(64) NOT NULL REFERENCES dim_customer(customer_id),
    sessions INTEGER NOT NULL CHECK (sessions >= 0),
    listening_or_usage_minutes NUMERIC(12, 2) NOT NULL CHECK (listening_or_usage_minutes >= 0),
    support_tickets INTEGER NOT NULL CHECK (support_tickets >= 0),
    PRIMARY KEY (activity_date, customer_id)
);

CREATE TABLE fact_transactions (
    transaction_id VARCHAR(128) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES dim_customer(customer_id),
    transaction_date DATE NOT NULL,
    gross_revenue NUMERIC(12, 2) NOT NULL CHECK (gross_revenue >= 0),
    discount_amount NUMERIC(12, 2) DEFAULT 0 CHECK (discount_amount >= 0)
);

CREATE TABLE model_customer_scores (
    score_date DATE NOT NULL,
    customer_id VARCHAR(64) NOT NULL REFERENCES dim_customer(customer_id),
    churn_probability NUMERIC(6, 5) NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    predicted_ltv NUMERIC(12, 2) NOT NULL,
    kmeans_segment INTEGER,
    risk_tier VARCHAR(16) NOT NULL,
    retention_action VARCHAR(128) NOT NULL,
    expected_saved_revenue NUMERIC(12, 2),
    PRIMARY KEY (score_date, customer_id)
);

CREATE INDEX idx_scores_risk_value
ON model_customer_scores (score_date, risk_tier, predicted_ltv DESC);
