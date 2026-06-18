import pandas as pd

from customer_intelligence.data.ingestion import generate_synthetic_customer_data
from customer_intelligence.data.ingestion import transform_telco_to_customer_intelligence
from customer_intelligence.data.validation import validate_customer_data


def test_generated_data_passes_validation():
    df = generate_synthetic_customer_data(n_customers=100, seed=7)
    result = validate_customer_data(df)
    assert result.passed, result.errors


def test_validation_catches_negative_spend():
    df = generate_synthetic_customer_data(n_customers=20, seed=7)
    df.loc[0, "monthly_spend"] = -1
    result = validate_customer_data(df)
    assert not result.passed
    assert any("monthly_spend" in error for error in result.errors)


def test_telco_schema_transforms_to_canonical_customer_schema():
    raw = pd.DataFrame(
        [
            {
                "customerID": "0001-TEST",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.9,
                "TotalCharges": "1048.8",
                "Churn": "Yes",
            }
        ]
    )
    transformed = transform_telco_to_customer_intelligence(raw)
    result = validate_customer_data(transformed)
    assert result.passed, result.errors
    assert transformed.loc[0, "churned"] == 1
    assert transformed.loc[0, "source_dataset"] == "IBM Telco Customer Churn"
