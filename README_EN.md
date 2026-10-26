# Home Credit — PD Model Validation

> 🇪🇸 [Versión en español](README.md) (notebooks and reports are in Spanish)

Development and **validation** of a Probability of Default (PD) model for consumer credit.
Training a model is only the first step. The project then answers what a Model Risk team
would ask:
- Does it rank risk well?
- Are its probabilities reliable PDs?
- Where does it fail?
- What happens under stress?
- Does it treat groups differently?
- Is it fit for use?

Reference framework: **SR 26-2** (Fed/OCC/FDIC, April 2026), which supersedes SR 11-7 and
keeps its validation structure.

> **Scope.** This is a portfolio project on a public Kaggle dataset. Regulatory references
> (SR 26-2, ECOA, EU AI Act, IFRS 9) guide the analysis and do not imply compliance
> certification. The Model Card states what is and is not covered.

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
📊 **Dashboard (ES/EN):** `streamlit run streamlit-app/app.py`

| Check | Result | Threshold | Status |
|---|---:|---|:---:|
| Gini (95% bootstrap CI) | 0.518 (0.503–0.532) | ≥ 0.20 | ✅ |
| KS | 0.383 | ≥ 0.15 | ✅ |
| Lift vs logistic benchmark | +0.021 | > 0 | ✅ |
| ECE (calibrated PD) | 0.0045 | ≤ 0.02 | ✅ |
| Calibration-in-the-large | −1.6% | ±10% | ✅ |
| Grade-level backtest (Jeffreys), grade PD with MoC | all grades ok (without MoC: G red) | no grade with p < 1% | ✅ |
| Train/test PSI | 0.0002 | ≤ 0.25 | ✅ * |
| Fairness (age) | DPD 0.151, TPR gap 0.319 | DPD ≤ 0.10, gaps ≤ 0.10 | ❌ |

\* Random split: PSI only confirms train/test homogeneity, not stability over time.

---

## Key findings

**1. The XGBoost score is not a PD until it is calibrated.**
- `scale_pos_weight` handles the 1:11 imbalance but inflates the score: the mean raw score
  is 38.8% against an 8.07% actual default rate.
- Platt scaling, fitted on a held-out split, brings the mean PD to 7.9%. ECE drops from 0.308
  to 0.005, and the Gini is unchanged.
- Treating the raw score as a PD would approve only 23% of applicants and cut profit to a
  third (notebook 02).

**2. The grade-level backtest finds underestimation in the riskiest tail.**

![Master scale](reports/figures/master_scale.png)

- **Grade G** (PD ≥ 27%, 3.4% of the portfolio) is assigned a 34.6% PD but defaults at 37.0%
  (Jeffreys p = 0.0098, red). **Grade F** is amber (p = 0.047). Grades A–E pass.
- Platt is linear in the logit and misses the tail.
- A traffic-light rule (red at p < 1%, amber at p < 5%) avoids false alarms: with 7 grades
  at 5%, a false alarm has ~30% probability
  ([ADR 0006](docs/decisions/0006-semaforo-master-scale.md)).
- **Remediation (v1.1):** as in IRB master scales, the individual PD ranks and decides, while
  expected loss uses a **grade PD** = max(mean PD, 75th Jeffreys percentile of the observed
  default rate on the validation split). Isotonic calibration did not fix the tail.
- **Result:** grade G goes from p = 0.0098 to 0.68, F from amber to ok, and expected loss
  rises by 5.2%. The Gini is unchanged
  ([ADR 0007](docs/decisions/0007-pd-de-grado-con-moc.md)).

**3. Ranking and decision.**
- Default rate ranges from 1.3% (D1) to 27.5% (D10). The top decile captures 34% of
  defaults, and the top 30% captures 65%.
- **Decision rule:** reject if `PD ≥ r/(r+LGD)`, the one-period break-even. With US
  assumptions (r = 11.5%, LGD = 45%), the cutoff is 20.35%. That gives 92.6% approval, a 6.3%
  default rate among approved and 30.7% among rejected.
- **Benchmarks:**

  | Model | Test Gini |
  |---|---:|
  | **XGBoost + Platt** | **0.518** |
  | Logistic regression | 0.496 |
  | WoE/IV scorecard, 14 variables, PDO 20 points | 0.482 |

**4. Stress test: very sensitive to bureau scores, barely to income.**

Shocks are applied to raw variables, ratios are recomputed and PD is re-estimated.

| Scenario | Change in expected loss |
|---|---:|
| Real income −25% | +1.1% |
| Bureau scores −20% | +40.5% |
| Combined recession | +34.2% |

**5. Fairness: the channel is the bureau score, not the age variable.**
- **Gender** is not a model feature: DIR 0.954. ✅
- **Age** is a feature: approval is 83.4% for under-25s and 98.6% for over-55s (DIR 0.847),
  with a TPR gap of 0.32. ❌
- `EXT_SOURCE_1` has a **0.60 correlation with age**. SHAP shows that wrongful rejections
  are driven by the bureau scores, so dropping `age_years` would change little. It is a
  classic proxy case, and whether it is acceptable is a business and legal decision
  ([ADR 0004](docs/decisions/0004-atributos-protegidos.md)).

**6. Robustness (notebook 04).**
- With ±5% input noise, the decision flips for 2% of applicants, almost all within 3 pp of
  the cutoff.
- **For 36% of applicants, asking for a larger amount lowers the PD.** This is a
  non-monotonic relationship learned from selection effects. It is fixed by challengers M1
  and M3 (finding 9).
- 58% of rejected applicants would be approved at up to half the requested amount.

**7. Deep learning only helps where it sees new information (notebook 06).**

| Model | Test Gini |
|---|---:|
| Tabular MLP (overfits after epoch 8) | 0.508 |
| Payment-history LSTM alone | 0.187 |
| **XGBoost + LSTM stack** | **0.530** (+0.013, 95% CI [+0.008, +0.016]) |

The stack is the **recommended challenger**. An autoencoder outlier filter was discarded:
flagged applicants default only 1.08× more than the rest.

**8. Advanced validation (notebook 07).**
- **Calibration by gender:** with gender excluded, PD underestimates men (−9.6%) and
  overestimates women (+4.4%). This is the trade-off between fairness-through-unawareness
  and group calibration.
- **Isotonic calibration** does not fix grade G either, so the recommendation is a margin of
  conservatism.
- **MC Dropout:** the 5% most uncertain applicants have a Gini of 0.32 (vs 0.52).
- **5-fold CV:** Gini 0.513 ± 0.002.
- **Partial dependence:** holding the bureau scores fixed, age does not penalize young
  applicants.
- **Sector concentration:** HHI 0.137. Doubling PD in the largest sector raises expected loss
  by 31%.

**9. Challengers ([ADR 0008](docs/decisions/0008-challengers.md)).**

| Model | Test Gini | Δ vs champion (95% paired CI) | Monotonicity violation |
|---|---:|---|---:|
| Champion | 0.518 | — | 35% |
| M1 monotonic | 0.509 | −0.009 [−0.012, −0.005] | 0% |
| M2 + payment features | 0.534 | +0.016 [+0.011, +0.021] | 33% |
| **M3 monotonic + payments** | **0.528** | **+0.011 [+0.005, +0.016]** | **0%** |

- Simple aggregated payment-history features match the LSTM stack (0.530) with far less
  complexity.
- **M3 is the recommended challenger.** It is monotonic, calibrated and passes the backtest
  with MoC.

## What changed from the first version

The project re-validated itself and fixed several issues:
- Early stopping and champion selection used the **test set**. Fixed with a
  fit / validation / calibration split.
- The stress test multiplied **standardized** features: a z-score × 0.75 is not "income −25%".
- An "SR 11-7 PASS" label ignored the failed calibration.
- The "optimal threshold" was the edge of the search grid.
- The "temporal backtest" sorted by age and sliced an unsorted frame.
- The API always returned HTTP 500.
- The dashboard showed synthetic data with hardcoded PASS labels.
- A GNN used random edges.

The full table is in the [Spanish README](README.md#qué-cambió-respecto-de-la-primera-versión).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put application_train.csv (and installments_payments.csv for notebook 06) in data/raw/
python run_all.py
pytest tests/ -v
uvicorn src.api.main:app --port 8000
streamlit run streamlit-app/app.py
```

## Limitations

- **No time dimension:** there is no origination date, so no out-of-time validation is possible.
- **Target:** it is not a regulatory default definition and has no 12-month horizon.
- **No independent validation:** the same author developed and validated the model.
- **Data:** only `application_train` is used; `bureau` and `previous_application` are future work.
- **Business assumptions:** market rates and LGDs are illustrative and do not net out
  funding or inflation.

## Stack

Python · XGBoost · scikit-learn · PyTorch · SHAP · MLflow · FastAPI · Streamlit · pytest · Docker · GitHub Actions
