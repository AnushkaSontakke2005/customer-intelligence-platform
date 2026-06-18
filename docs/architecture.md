# Architecture Notes

## Platform Flow

```mermaid
sequenceDiagram
    participant Source as "Data Sources"
    participant Pipeline as "Training Pipeline"
    participant Registry as "MLflow and Artifacts"
    participant API as "FastAPI"
    participant Dash as "Streamlit Dashboard"
    participant Ops as "Marketing and Success Teams"

    Source->>Pipeline: "Customer, transaction, support, engagement records"
    Pipeline->>Pipeline: "Validate data quality"
    Pipeline->>Pipeline: "Engineer RFM and behavioral features"
    Pipeline->>Pipeline: "Train segmentation, churn, and LTV models"
    Pipeline->>Registry: "Log metrics, artifacts, and model outputs"
    Registry->>API: "Load production model artifacts"
    API->>Ops: "Serve churn, LTV, recommendation, and ROI scores"
    Registry->>Dash: "Load processed customer intelligence outputs"
    Dash->>Ops: "Executive KPIs and retention simulator"
```

## Monitoring

```mermaid
flowchart TD
    A["Production scores"] --> B["Prediction distribution checks"]
    C["Matured churn labels"] --> D["ROC-AUC and calibration checks"]
    E["Current features"] --> F["Population Stability Index"]
    B --> G["Alert when score mix shifts"]
    D --> H["Retrain when model quality degrades"]
    F --> H
```

## Data Quality Contract

- Customer IDs must be unique.
- Required columns must be present before feature generation.
- Numeric fields are range checked.
- Churn labels must be binary.
- High null-rate fields are flagged as warnings so the pipeline can distinguish data failures from source-quality debt.
