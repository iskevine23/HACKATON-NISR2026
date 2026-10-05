# Roadmap

## Before submission (deadline 30 October 2026, 11:59 PM)

| By | Task |
|---|---|
| 6 Oct | Request EICV7 and FinScope 2024 microdata on microdata.statistics.gov.rw |
| 10 Oct | Register the team (official deadline 20 Oct, 11:59 PM). Confirm at least one Rwandan citizen. |
| 12 Oct | Verify all district values against the reports (docs/DATA_SOURCES.md checklist) |
| 15 Oct | Push to a public GitHub repo and deploy the app |
| 20 Oct | Native-speaker review of the Kinyarwanda and French text; review the assistant's answer templates; test with a SACCO or district planner |
| 25 Oct | Phase 2 model if microdata arrived (below) |
| 28 Oct | Final README, AI disclosure, demo video, submission |

## Phase 2: household-level model (if EICV7 microdata is granted)

1. Train a poverty classifier (logistic regression and gradient boosting) on EICV7 households using variables NISR itself uses to model poverty: household head education, household size, housing materials, electricity, assets, savings account, mobile money account, district.
2. Report cross-validated accuracy and calibration by district, with survey weights.
3. Add a "what-if" panel: estimated change in district poverty risk if, for example, savings account ownership among poor households rose to the non-poor level (53% to 74% of households in EICV7). Label results as associations, not causal effects.
4. If FinScope microdata is granted, replace chart-read district values with weighted estimates and confidence intervals.

## After the hackathon

- Sector-level targeting using census microdata and small-area estimation
- Overlay of SACCO and agent locations from the National Bank
- Updates when EICV8 (2026/27) and FinScope 2028 are released
