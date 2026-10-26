# Home Credit — PD Model Validation

> 🇪🇸 [Versión en español](README_ES.md) · Notebooks and the validation report are written in Spanish; code, dashboard (EN/ES) and docs are bilingual where it matters.

Development and **validation** of a Probability of Default (PD) model for consumer credit.
The goal is not only to train a model, but to answer the questions a Model Risk Management
team would ask:
- Does it rank risk correctly?
- Are its probabilities reliable PDs?
- Where does it fail?
- What happens under stress?
- Does it treat groups differently?
- Is it fit for use?

Reference framework: **SR 26-2** (Fed/OCC/FDIC, April 2026), which supersedes SR 11-7 and
keeps its validation structure.

> **Scope.** This is a portfolio project on a public Kaggle dataset. Regulatory references
> (SR 26-2, ECOA, EU AI Act, IFRS 9) guide what is analysed; they do not imply compliance
> certification. The Model Card states what is and is not covered.

---

## What this project adds to the usual Home Credit project

Most public work on this dataset is built for the Kaggle leaderboard. The pattern is to join
all seven tables, engineer thousands of features, tune LightGBM ensembles and report AUC.
Top solutions reach roughly 0.79–0.80 AUC. This project **does not compete on that axis**:
it only uses `application_train` plus the installment history, and gets 0.76 AUC
(0.53 Gini with the best challenger). What it adds is the **validation layer**, the part a
bank actually needs before using a model.

| Typical Home Credit project | This project |
|---|---|
| Reports AUC on a test set, often also used for early stopping or model selection | Leakage-free fit / validation / calibration / test split. The test set is used only once. |
| Outputs a score treated as a probability | Shows the raw score is **not** a PD (mean 38.8% vs 8.1% actual), then calibrates it and measures calibration (ECE, calibration-in-the-large) |
| — | **Master-scale backtest** by rating grade, using the Jeffreys test with a traffic-light rule to control multiple-testing false alarms |
| — | Finds that the riskiest grade is **underestimated**, and remediates it with a **margin of conservatism** estimated out-of-sample. Tested before and after. |
| Threshold 0.5 or F1-optimal | **Economic cutoff** `PD* = r/(r+LGD)` per market, checked against the empirical profit curve |
| Fairness, if any, as a single disparate-impact number | DPD, DIR and equalized odds under the real decision rule, plus a **proxy analysis**: age bias flows through bureau scores (corr. 0.60), shown three ways. Also documents the **group-calibration trade-off** (gender). |
| — | **Stress test in expected-loss terms**, with shocks on raw variables (not on standardized features) |
| — | **Monotonicity test** (asking for more money lowered PD for ~35% of applicants) and a **monotone-constrained challenger** that fixes it |
| Model comparison by point AUC | Champion vs challengers with **paired bootstrap CIs**. Simple payment-history aggregates match an LSTM with far less complexity. |
| — | Governance artifacts generated from the outputs: **validation report** with severity-ranked findings, **Model Card**, ADRs, and a three-level status (approved / with conditions / not approved) |
| Results presented as final | **Iterative and self-critical:** v1.0 failed validation and was remediated in v1.1. The bugs found in the first version are documented below. |

---

## Validation outcome: **APPROVED WITH CONDITIONS**

Validation was iterative:
- **v1.0 — not approved:** the individual PD underestimated the riskiest grade.
- **v1.1 — remediated:** a grade-level PD with a margin of conservatism (MoC) was added, and
  every quantitative check now passes.
- **Open conditions:**
  - age fairness, transmitted through the bureau scores;
  - unequal calibration by gender;
  - non-monotonicity in the loan amount.

  The last one is solved by the **recommended challenger (M3)**, which also improves the Gini.

📄 **[Validation report (PDF, 2 pages, Spanish)](reports/validation_report.pdf)** ·
📋 **[Model Card](reports/model_cards/model_card.html)** ·
📊 **Dashboard (EN/ES):** `streamlit run streamlit-app/app.py`

| Check | Result | Threshold | Status |
|---|---:|---|:---:|
| Gini (95% bootstrap CI) | 0.518 (0.503–0.532) | ≥ 0.20 | ✅ |
| KS | 0.383 | ≥ 0.15 | ✅ |
| Lift over logistic benchmark | +0.021 | > 0 | ✅ |
| ECE (calibrated PD) | 0.0045 | ≤ 0.02 | ✅ |
| Calibration-in-the-large | −1.6% | ±10% | ✅ |
| Grade backtest (Jeffreys), grade PD with MoC | all grades ok (without MoC: G red) | no grade with p < 1% | ✅ |
| Train/test PSI | 0.0002 | ≤ 0.25 | ✅ * |
| Fairness (age) | DPD 0.151, TPR gap 0.319 | DPD ≤ 0.10, gaps ≤ 0.10 | ❌ |

\* Random split: PSI only confirms train/test homogeneity, not stability over time.

---

## Key findings

### 1. The XGBoost score is not a PD until it is calibrated

`scale_pos_weight` handles the 1:11 class imbalance but inflates the probability scale.
Platt scaling, fitted on a split the model never saw, fixes the scale without changing the
ranking.

| | Mean PD | Actual default | ECE | Brier |
|---|---:|---:|---:|---:|
| Raw score | 38.8% | 8.07% | 0.308 | 0.182 |
| Calibrated PD | 7.9% | 8.07% | 0.005 | 0.068 |

Using the raw score as a PD has a concrete cost. Under the US cutoff it approves only 23% of
applicants and **cuts profit to a third** (notebook 02).

### 2. Grade-level backtest: the high-risk tail is underestimated → margin of conservatism

![Master scale](reports/figures/master_scale.png)

| Grade | PD range | % portfolio | Assigned PD | Actual default | Jeffreys p | Result |
|---|---|---:|---:|---:|---:|---|
| A | 0–2% | 14.0% | 1.3% | 1.4% | 0.21 | ok |
| B | 2–4% | 23.5% | 3.0% | 2.7% | 0.95 | ok |
| C | 4–7% | 23.4% | 5.4% | 5.3% | 0.58 | ok |
| D | 7–11% | 16.4% | 8.8% | 8.8% | 0.50 | ok |
| E | 11–17% | 11.9% | 13.6% | 13.8% | 0.33 | ok |
| F | 17–27% | 7.5% | 21.1% | 22.1% | 0.047 | 🟠 amber |
| G | ≥ 27% | 3.4% | 34.6% | 37.0% | 0.0098 | 🔴 underestimates |

With the individual PD, grade G is red. Isotonic calibration does not fix it, and every
challenger shows the same issue, so it is a property of the tail given this data.
Underestimating the highest risk is the serious error, because it leads to
under-provisioning.

**Remediation (v1.1, [ADR 0007](docs/decisions/0007-pd-de-grado-con-moc.md)).** The two uses
of the PD are separated, as in IRB schemes:
- the **individual PD** ranks applicants and drives the decision;
- for expected loss and provisioning, each grade uses a **grade PD** = max(mean PD, 75th
  Jeffreys percentile of the default rate observed on the validation split).

| Grade | Mean PD | Grade PD (with MoC) | Actual default (test) | Jeffreys p (before → after) |
|---|---:|---:|---:|---|
| E | 13.6% | 14.6% | 13.8% | 0.33 → 0.98 |
| F | 21.1% | 21.7% | 22.1% | 0.047 🟠 → 0.26 ✅ |
| G | 34.6% | 37.5% | 37.0% | 0.0098 🔴 → 0.68 ✅ |

Expected loss rises by 5.2%; the Gini is unchanged. The traffic-light rule (red at p < 1%,
amber at p < 5%) avoids false alarms: with 7 grades tested at 5%, the probability of at least
one false alarm is about 30% ([ADR 0006](docs/decisions/0006-semaforo-master-scale.md)).

### 3. Ranking and decision

- **Deciles:** the default rate goes from **1.3% in D1 to 27.5% in D10**. The riskiest 10%
  captures 34% of all defaults, and the riskiest 30% captures 65%.
- **Decision rule:** reject if `PD ≥ r/(r+LGD)`, the one-period break-even. Under US
  assumptions (r = 11.5%, LGD = 45%) the cutoff is **20.35%**:
  - approval rate: 92.6%;
  - default rate among approved: 6.3% (vs 30.7% among rejected).
- **Benchmarks:**

  | Model | Test Gini |
  |---|---:|
  | **XGBoost + Platt** | **0.518** |
  | Logistic regression (same features) | 0.496 |
  | WoE/IV scorecard, 14 variables, PDO-20 points | 0.482 |

  XGBoost beats a traditional scorecard by ~3.6 Gini points. That number is the input for the
  interpretability-vs-performance discussion.

### 4. Stress test: very sensitive to bureau scores, barely to income

Shocks are applied to the raw variables, ratios are recomputed and the calibrated PD is
re-estimated. Expected loss = grade PD × LGD × loan amount.

| Scenario | Mean PD | Δ expected loss | Δ approval |
|---|---:|---:|---:|
| Baseline | 7.9% | — | — |
| Real income −25% | 8.1% | +1.1% | −0.3 pp |
| Real income −40% | 8.2% | +2.2% | −0.5 pp |
| Bureau scores −20% | 11.1% | **+40.5%** | −5.2 pp |
| Combined recession | 10.7% | +34.2% | −4.8 pp |

The model barely reacts to income shocks; what moves the PD is the external bureau scores.
This is consistent with the EDA, but it is a blind spot when a shock hits income before it
reaches the bureau.

### 5. Fairness: the channel is the bureau score, not the age variable

Under the real decision rule:
- **Gender** (not a model feature): approval F 94.1% vs M 89.7%, DIR 0.954. ✅
- **Age** (a model feature):
  - approval for under-25s: 83.4%; for over-55s: 98.6% (DIR 0.847);
  - TPR gap between groups: 0.32. ❌

The EDA shows that `EXT_SOURCE_1` has a **0.60 correlation with age**. SHAP on wrongful
rejections shows they are driven by the EXT_SOURCE scores, not by `age_years`. Dropping age
from the model would change little: it is a classic proxy case. Whether that is acceptable
is a business and legal decision ([ADR 0004](docs/decisions/0004-atributos-protegidos.md)).

### 6. Robustness (notebook 04)

- **±5% noise on inputs:** the decision flips for 2% of applicants, almost all within 3 pp of
  the cutoff. A ±3 pp manual-review band would cover 5% of applications.
- **Non-monotonicity:** for **~35% of applicants, asking for a larger amount lowers the PD**.
  The model learned a selection effect, so it could be gamed. Fixed by challengers M1 and M3
  (finding 10).
- **Actionable counterfactual:** 58% of rejected applicants would be approved at up to half
  the requested amount.

### 7. Weak segments

| Segment | Gini | vs overall (0.518) |
|---|---:|---:|
| Age > 55 | 0.422 | −0.10 |
| Pensioners | 0.445 | −0.07 |
| Age < 25 | 0.464 | −0.05 |

### 8. Deep learning only helps where it sees new information (notebook 06)

| Model | Test Gini | Decision |
|---|---:|---|
| Tabular neural network (MLP) | 0.508 | Discarded: does not beat XGBoost; overfits after epoch 8 |
| LSTM on payment history | 0.187 | Not useful alone (2 variables per installment) |
| XGBoost + LSTM stack | 0.530 | Proves the signal is incremental (+0.013, 95% CI [+0.008, +0.016]); superseded by M3, which is simpler |
| Autoencoder (outlier detector) | — | Discarded: flagged "outliers" default only 1.08× more than the rest |

No network beats XGBoost on the same features. The sequence of payments on previous loans,
however, adds information that `application_train` does not have. This matches the error
analysis: the model lacks behavioural data.

### 9. Advanced validation (notebook 07)

- **Calibration by gender:** with gender excluded, the PD **underestimates men (−9.6%) and
  overestimates women (+4.4%)**. This is the trade-off between *fairness through unawareness*
  and group calibration; it is documented as a policy decision.
- **Isotonic vs Platt:** isotonic calibration does not fix grade G either. The problem is not
  the shape of the calibration map, hence the margin of conservatism on F–G.
- **MC Dropout:** for the 5% most uncertain applicants, XGBoost's Gini drops to 0.32 (vs 0.52
  for the rest). A useful manual-review criterion.
- **Cross-validation:** Gini 0.513 ± 0.002 over 5 folds.
- **Partial dependence:** holding bureau scores fixed, age does *not* penalize young
  applicants (inverted-U shape peaking at 39). It is the third piece of evidence that bureau
  scores carry the age bias.
- **Sector concentration:** HHI 0.137. Doubling the PD of the largest sector (28% of exposure)
  raises expected loss by 31%.

### 10. Challengers: monotonic + payment history ([ADR 0008](docs/decisions/0008-challengers.md))

![Challengers](reports/figures/challengers.png)

| Model | Test Gini | Δ vs champion (95% paired CI) | Monotonicity* | Red grade (without / with MoC) |
|---|---:|---|---:|---|
| Champion (XGBoost + Platt) | 0.518 | — | 35% | G / — |
| M1 monotonic | 0.509 | −0.009 [−0.012, −0.005] | **0%** | G / — |
| M2 + payment features | **0.534** | +0.016 [+0.011, +0.021] | 33% | G / — |
| **M3 monotonic + payments** | **0.528** | **+0.011 [+0.005, +0.016]** | **0%** | G / — |
| *XGBoost + LSTM stack (notebook 06)* | *0.530* | *+0.013* | — | — |

\* Share of applicants whose PD drops when they ask for more (scaling amount, annuity and goods price together).

- **Monotonicity is cheap:** 0.5–0.9 Gini points, and the model can no longer be gamed through
  the requested amount.
- **Simple aggregated payment-history features** (late payments, missed payments, partial
  payments, last 12 months) **match or beat the LSTM** without a recurrent network.
- **M3 is the recommended challenger:**
  - significant improvement over the champion;
  - monotonic and calibrated (ECE 0.004);
  - passes the backtest with MoC;
  - fairness similar to the champion.

  Before promotion it still needs a full validation, and the payment features must be added
  to the API.

---

## What changed from the first version

The project re-validated itself. These errors were found and fixed:

| Issue | Impact | Fix |
|---|---|---|
| Early stopping and champion selection on the **test set** | Optimistic metrics | Fit / validation / calibration split ([ADR 0001](docs/decisions/0001-esquema-de-splits.md)) |
| Stress test multiplied **standardized** features (z-score × 0.75 ≠ income −25%) | Meaningless result (Δ = 0.0000) | Shocks on raw variables, measured in PD and expected loss |
| "SR 11-7 PASS" ignored the failed calibration | Wrong approval | Calibration and grade backtest are part of the criteria |
| "Optimal" threshold 0.725 in every market | It was the edge of the search grid | Analytical break-even cutoff on calibrated PD |
| "Temporal backtest" sorted by `DAYS_BIRTH` (age) and applied to an unsorted frame | Random windows | Cross-validation; documented that Home Credit does not allow out-of-time validation |
| API `/predict` always returned HTTP 500 (~23 missing columns) | Unusable API | Full schema + integration tests |
| Dashboard with synthetic data and hardcoded "PASS" labels | Results were not real | A single app that only reads pipeline outputs |
| GNN with random edges | Measured no real contagion | Replaced by a sector-concentration analysis on real data |
| `.gitignore` pattern `models/` also excluded `src/models/` | The model code would not have been published | Root-anchored patterns |
| Grade G underestimated (v1.0) | Under-provisioning in the tail | Grade PD with MoC (v1.1) |
| PD non-monotonic in amount | Model could be gamed | Monotonic challenger M3 |

---

## Project structure

```
homecredit-model-validation/
├── configs/credit_policy.json       # Business assumptions: rates, LGD, master scale, thresholds
├── src/
│   ├── data/ingest.py               # Ingestion, data quality, derived features, split
│   ├── features/build.py            # Imputation + missing indicators + one-hot
│   ├── models/
│   │   ├── train.py                 # Logistic benchmark vs XGBoost + Platt calibration + MoC
│   │   ├── calibration.py           # Platt scaling on the logit
│   │   ├── pd_model.py              # PDScorer: raw data → calibrated PD / grade PD
│   │   ├── moc.py                   # Grade PD with margin of conservatism
│   │   ├── challengers.py           # M1 monotonic, M2 payments, M3 both
│   │   └── scorecard.py             # WoE/IV scorecard with points (PDO)
│   ├── validation/
│   │   ├── model_validation.py      # Discrimination, calibration, PSI, stress, sensitivity, segments
│   │   ├── master_scale.py          # Rating grades + binomial / Jeffreys tests
│   │   └── explain.py               # SHAP
│   ├── governance/
│   │   ├── fairness.py              # DPD, DIR, equalized odds, AUC by group
│   │   ├── model_card.py            # HTML/JSON Model Card generated from the reports
│   │   ├── status.py                # Status: approved / with conditions / not approved
│   │   └── validation_report.py     # PDF validation report
│   ├── monitoring/baseline.py       # Drift baseline and check (PSI)
│   └── api/main.py                  # FastAPI: PD, grade, grade PD, decision, expected loss, SHAP
├── notebooks/                       # 01 EDA → 07 advanced validation (Spanish)
├── streamlit-app/app.py             # EN/ES dashboard
├── docs/decisions/                  # ADRs: design decisions
├── docs/validation_findings.json    # Notebook findings consumed by the report
├── tests/                           # 27 tests (metrics, master scale, MoC, scorecard, API)
├── docker/                          # API + dashboard + MLflow
└── run_all.py
```

## Notebooks (in Spanish)

| # | Notebook | Content |
|---|---|---|
| 01 | [EDA](notebooks/01_eda.ipynb) | Imbalance, informative missing values, age as a proxy inside bureau scores |
| 02 | [Decision cutoff](notebooks/02_threshold_analysis.ipynb) | Break-even by market (AR, US, BR, CO) and the cost of fairness |
| 03 | [Error analysis](notebooks/03_error_analysis.ipynb) | High-confidence false negatives/positives, SHAP, segments |
| 04 | [Robustness](notebooks/04_robustez_counterfactuals.ipynb) | Input noise, amount monotonicity, maximum approvable amount |
| 05 | [Lifecycle](notebooks/05_lifecycle_simulation.ipynb) | Synthetic drift simulation and monitoring/retirement criteria |
| 06 | [Deep learning](notebooks/06_deep_learning.ipynb) | MLP, payment-history LSTM, stacking, autoencoder |
| 07 | [Advanced validation](notebooks/07_advanced_validation.ipynb) | Segment calibration, isotonic vs Platt, MC Dropout, CV, PDPs, concentration |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # notebooks: requirements-notebooks.txt

# Kaggle data in data/raw/:
#   application_train.csv            (required)
#   installments_payments.csv        (challengers M2/M3 and notebook 06)
# https://www.kaggle.com/c/home-credit-default-risk/data

python run_all.py                    # full pipeline (~15 min on CPU, fully reproducible)
pytest tests/ -v
uvicorn src.api.main:app --port 8000 # API → http://localhost:8000/docs
streamlit run streamlit-app/app.py   # dashboard
```

## Limitations

- **No time dimension:** Home Credit has no origination date, so there is no out-of-time
  validation and no vintage analysis.
- **Target:** it flags payment difficulties. It is not a regulatory default definition and
  has no 12-month horizon, so it is not directly usable for IFRS 9.
- **No independent validation:** the same person developed and validated the model.
- **Data:** only `application_train` plus `installments_payments`. `bureau` and
  `previous_application`, the strongest extra sources in Kaggle solutions, are the natural
  next step.
- **Business assumptions:** market rates and LGDs are illustrative
  (`configs/credit_policy.json`) and do not net out funding cost or inflation.

## Stack

Python · XGBoost · scikit-learn · PyTorch · SHAP · MLflow · FastAPI · Streamlit · pytest · Docker · GitHub Actions
