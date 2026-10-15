# FinWise Lending — Frontend Application (React Dashboard)

## 1. Overview & User Experience Scope

The `frontend/` directory is designated for the interactive web dashboard designed for loan underwriters, credit risk managers, and compliance officers.

> [!NOTE]
> **Implementation Status:**
> In **Week 1**, this component is in the architectural planning stage. The actual React dashboard setup, component development, and API integration will be implemented in **Week 12**.

---

## 2. Planned Core Features (Week 12 Delivery)

1. **Underwriter Decision Portal:**
   - Single-applicant risk assessment form with instant feedback.
   - Batch loan application upload via CSV for portfolio screening.
2. **Visual Risk Indicators:**
   - Probability of Default (PD) percentage and color-coded Risk Tier badges (`Low`, `Medium`, `High`).
3. **Interactive Explainability & Waterfall Charts:**
   - Interactive **SHAP waterfall chart per prediction** illustrating exact feature attributions driving each applicant's score.
   - Transparent risk factor breakdown for underwriter decision support.
4. **"What-If" Underwriting Scenario Builder:**
   - Sliders enabling underwriters to simulate applicant adjustments (e.g., modified loan amount or term) to evaluate whether an applicant can transition into an acceptable risk band.
5. **Portfolio Monitoring View:**
   - Distribution of applicant risk bands (`Low`, `Medium`, `High`) across current submission pipelines.
   - Demographic fairness monitoring indicators (disparate impact ratio tracking across gender and age).

---

## 3. Planned Technology Stack

- **Framework:** React 18+ with TypeScript.
- **Styling:** Tailwind CSS for responsive, modern UI components.
- **Data Visualization:** Recharts / Chart.js for distribution curves and risk gauges.
- **HTTP Client:** Axios / TanStack Query (React Query) for resilient API data fetching and state caching.

---

## 4. Planned Directory Layout

```
frontend/
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── RiskGauge.tsx          # Credit score gauge visualizer
│   │   ├── ShapWaterfall.tsx      # Feature attribution breakdown
│   │   ├── ApplicantForm.tsx      # Loan application input fields
│   │   ├── WhatIfSimulator.tsx    # Scenario modeling sliders
│   │   └── AdverseActionCard.tsx  # Compliance reason code card
│   ├── services/
│   │   └── api.ts                 # Axios integration with FastAPI endpoints
│   ├── types/
│   │   └── applicant.ts           # TypeScript interfaces matching Pydantic schemas
│   ├── App.tsx
│   └── index.tsx
├── package.json
└── README.md
```
