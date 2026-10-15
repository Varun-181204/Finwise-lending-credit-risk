# FinWise Lending — Backend Service (FastAPI)

## 1. Overview & Service Scope

The `backend/` directory will host the high-performance REST API microservice responsible for serving real-time credit risk predictions, risk-band classifications, and SHAP-based model explanations to the loan officer frontend.

> [!NOTE]
> **Implementation Status:**
> In **Week 1**, this service is in the architectural planning stage. The actual FastAPI service implementation, dependency additions, and endpoint controllers will be authored in **Week 9**.

---

## 2. Planned Microservice Architecture

* **Framework:** FastAPI (Python 3.14 compatible asynchronous web framework).
* **Server:** Uvicorn (ASGI server worker).
* **Model Serialization:** JSON-based XGBoost model format loaded in-memory on application startup (`lifespan` handler).
* **Concurrency Target:** Architected and load-tested to handle **50 concurrent API requests** simultaneously.

---

## 3. Planned Endpoints & Interfaces

### 1. `GET /health`
* Liveness and readiness probe for container orchestration.
* Returns service uptime, loaded model version, and memory status.

### 2. `POST /api/v1/predict`
* Ingests applicant financial profiles via strongly typed Pydantic models.
* Executes feature transformations matching the training pipeline.
* Outputs:
  - Estimated Probability of Default ($PD \in [0, 1]$).
  - Calibrated Risk Band (`Low`, `Medium`, `High`).
  - Evaluation timestamp and model version metadata.
  - Decision support context for loan officers (no automated decline decisions).

### 3. `POST /api/v1/explain`
* Evaluates applicant features using the pre-computed `TreeSHAP` explainer.
* Generates instance-level feature attributions powering an interactive **SHAP waterfall chart per prediction** to support underwriter evaluations.
* Governance note: Regulatory compliance requirements depend on operating jurisdiction and require client/compliance team confirmation. SHAP alone does not guarantee legal compliance.

---

## 4. Planned Directory Structure (Week 9 Delivery)

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application factory and routing
│   ├── config.py            # Environment configurations (pydantic-settings)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # /predict, /explain, /health controllers
│   │   └── dependencies.py  # Model loader and authentication dependencies
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── applicant.py     # Pydantic input/output schemas
│   └── services/
│       ├── __init__.py
│       ├── inference.py     # XGBoost prediction engine
│       └── explainer.py     # SHAP attribution service
├── tests/
│   ├── __init__.py
│   └── test_api.py          # API route integration tests via TestClient
└── README.md
```

---

## 5. Security & Governance Standards

- **Input Validation:** Strict range and type validation preventing malformed or out-of-distribution payloads from reaching model memory.
- **PII Scrubbing:** Zero PII logged to standard out or telemetry databases.
- **CORS & Rate Limiting:** Configured to restrict access exclusively to authorized frontend hosts.
