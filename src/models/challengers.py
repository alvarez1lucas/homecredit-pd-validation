"""
Challengers del modelo de PD
============================
Responden a dos observaciones abiertas de la validación del champion:

  M1 — Monotónico: la PD no era monótona respecto del monto solicitado (notebook 04).
       XGBoost con monotone_constraints (+1) en monto, cuota y ratios de carga, y sin
       AMT_GOODS_PRICE en nivel (queda credit_goods_ratio). Así, escalar el pedido
       (monto, cuota y bien juntos) nunca puede bajar la PD.
  M2 — Pagos: el historial de cuotas aportaba información (stack con LSTM, notebook 06).
       Variables agregadas de installments_payments dentro de XGBoost: la alternativa
       simple al LSTM.
  M3 — M1 + M2.

Todos usan el mismo esquema que el champion: entrenan en fit, early stopping en
validación, Platt en calibración y evaluación en test.
"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import roc_auc_score

from src.config import (DATA_PROCESSED, DATA_RAW, FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, ROOT,
                        load_policy, reference_cutoff)
from src.governance.fairness import demographic_parity_difference, disparate_impact_ratio
from src.models.calibration import PlattCalibrator
from src.models.moc import estimate_grade_pd, grade_pd
from src.models.pd_model import PDScorer
from src.validation.master_scale import master_scale_summary, master_scale_table
from src.validation.model_validation import (age_groups, expected_calibration_error,
                                             gini, gini_bootstrap_ci)

CHALLENGERS_DIR = ROOT / "models" / "challengers"

MONOTONE_UP = ["num__AMT_CREDIT", "num__AMT_ANNUITY",
               "num__credit_income_ratio", "num__annuity_income_ratio"]
DROP_FOR_MONOTONE = ["num__AMT_GOODS_PRICE", "num__missingindicator_AMT_GOODS_PRICE"]
AMOUNT_COLS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]


# ── Variables agregadas de pagos ──────────────────────────────────────────────
def build_payment_features() -> pd.DataFrame | None:
    """
    Agrega installments_payments (cuotas de créditos ANTERIORES; todas las fechas son
    previas a la solicitud) a nivel cliente. Una cuota sin pago registrado cuenta como impaga.
    """
    path = DATA_RAW / "installments_payments.csv"
    if not path.exists():
        logger.warning("installments_payments.csv no encontrado: se saltean M2 y M3")
        return None
    inst = pd.read_csv(path, usecols=["SK_ID_PREV", "SK_ID_CURR", "DAYS_INSTALMENT",
                                      "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"])
    inst["unpaid"]    = inst["AMT_PAYMENT"].isna()
    inst["days_late"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0).fillna(30)
    inst["late"]      = inst["days_late"] > 0
    inst["late_30"]   = inst["days_late"] > 30
    inst["underpaid"] = inst["AMT_PAYMENT"].fillna(0) < 0.99 * inst["AMT_INSTALMENT"]
    inst["recent"]    = inst["DAYS_INSTALMENT"] >= -365

    g = inst.groupby("SK_ID_CURR")
    feats = pd.DataFrame({
        "pay_n_installments":   g.size(),
        "pay_n_prev_loans":     g["SK_ID_PREV"].nunique(),
        "pay_late_share":       g["late"].mean(),
        "pay_late30_share":     g["late_30"].mean(),
        "pay_days_late_mean":   g["days_late"].mean(),
        "pay_days_late_max":    g["days_late"].max(),
        "pay_underpaid_share":  g["underpaid"].mean(),
        "pay_unpaid_count":     g["unpaid"].sum(),
    })
    rec = inst[inst["recent"]].groupby("SK_ID_CURR")
    feats["pay_recent_late_share"]     = rec["late"].mean()
    feats["pay_recent_days_late_mean"] = rec["days_late"].mean()
    feats["pay_recent_n"]              = rec.size()
    feats["pay_recent_n"] = feats["pay_recent_n"].fillna(0)
    logger.info(f"  Variables de pagos: {feats.shape[1]} para {len(feats):,} clientes")
    return feats


class ChallengerScorer:
    """Mismo contrato que PDScorer (raw → PD), con columnas removidas o agregadas."""

    def __init__(self, base: PDScorer, model, calibrator, drop_cols, pay_feats):
        self.base, self.model, self.calibrator = base, model, calibrator
        self.drop_cols, self.pay_feats = drop_cols, pay_feats

    def features(self, X_base: pd.DataFrame, sk_ids) -> pd.DataFrame:
        X = X_base.drop(columns=[c for c in self.drop_cols if c in X_base.columns])
        if self.pay_feats is not None:
            P = self.pay_feats.reindex(np.asarray(sk_ids)).reset_index(drop=True)
            X = pd.concat([X.reset_index(drop=True), P], axis=1)
        return X

    def pd_from_raw(self, df_raw: pd.DataFrame) -> np.ndarray:
        X = self.features(self.base.transform_raw(df_raw), df_raw["SK_ID_CURR"].values)
        return self.calibrator.transform(self.model.predict_proba(X)[:, 1])


def _train_xgb(X_fit, y_fit, X_val, y_val, monotone: dict | None):
    params = PDScorer().model.get_params()
    params.update(early_stopping_rounds=50, n_estimators=1000)
    if monotone:
        params["monotone_constraints"] = monotone
    params["scale_pos_weight"] = float((y_fit == 0).sum() / (y_fit == 1).sum())
    m = xgb.XGBClassifier(**params)
    m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return m


def monotonicity_share(pd_fn, raw_sample: pd.DataFrame) -> dict:
    """
    Escala monto, cuota y valor del bien de 0,5x a 2x para cada cliente y mide:
      - trend: % de clientes cuya PD tiene pendiente NEGATIVA (pedir más baja la PD; métrica de la notebook 04)
      - any_step: % con algún escalón descendente, aunque sea mínimo (los árboles son escalonados)
    """
    factors = np.round(np.arange(0.5, 2.01, 0.1), 2)
    curves = []
    for f in factors:
        d = raw_sample.copy()
        for c in AMOUNT_COLS:
            d[c] = d[c] * f
        curves.append(pd_fn(d))
    curves = np.vstack(curves).T
    slope = np.array([np.polyfit(factors, c, 1)[0] for c in curves])
    return {"trend": round(float((slope < 0).mean()), 4),
            "any_step": round(float((np.diff(curves, axis=1) < -1e-6).any(axis=1).mean()), 4)}


def paired_delta_ci(y, pd_new, pd_ref, n_boot=200, seed=RANDOM_STATE) -> list:
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append((2 * roc_auc_score(y[i], pd_new[i]) - 1) - (2 * roc_auc_score(y[i], pd_ref[i]) - 1))
    return [round(float(q), 4) for q in np.percentile(d, [2.5, 97.5])]


def evaluate(name, pd_test, y_test, raw_test, mono, policy, cutoff, n_features, best_iter,
             grade_table, pd_ref):
    ms = policy["master_scale"]
    lim, grs = ms["limites_superiores"], ms["grados"]
    t = master_scale_table(y_test, pd_test, lim, grs)
    s = master_scale_summary(t)
    t_moc = master_scale_table(y_test, grade_pd(pd_test, grade_table, lim, grs), lim, grs, pd_rank=pd_test)
    s_moc = master_scale_summary(t_moc)
    approved = (pd_test < cutoff).astype(int)
    age = age_groups(raw_test["age_years"]).values
    gm = raw_test["CODE_GENDER"].isin(["M", "F"]).values
    return {
        "model": name,
        "n_features": n_features,
        "best_iteration": best_iter,
        "gini": gini(y_test, pd_test),
        "gini_ci95": gini_bootstrap_ci(y_test, pd_test, n_boot=100),
        "ece": expected_calibration_error(y_test, pd_test),
        "mean_pd": round(float(pd_test.mean()), 4),
        "delta_gini_ci95": paired_delta_ci(y_test, pd_test, pd_ref),
        "grades_red": s["grades_underestimating"],
        "grades_amber": s["grades_amber"],
        "grades_red_with_moc": s_moc["grades_underestimating"],
        "grades_amber_with_moc": s_moc["grades_amber"],
        "moc_factors": {g: r["moc_factor"] for g, r in grade_table.items()},
        "monotonic_trend_violation": mono["trend"],
        "monotonic_any_step_violation": mono["any_step"],
        "approval_rate": round(float(approved.mean()), 4),
        "default_rate_approved": round(float(y_test[approved == 1].mean()), 4),
        "age_dpd": demographic_parity_difference(approved, age),
        "age_dir": disparate_impact_ratio(approved, age),
        "gender_dir": disparate_impact_ratio(approved[gm], raw_test["CODE_GENDER"].values[gm]),
    }


def run_challengers():
    logger.info("=" * 50)
    logger.info("STAGE — Challengers")
    logger.info("=" * 50)

    policy, cutoff = load_policy(), reference_cutoff()
    base = PDScorer()
    X_train  = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_test   = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_train  = pd.read_parquet(DATA_PROCESSED / "y_train.parquet").iloc[:, 0]
    y_test   = pd.read_parquet(DATA_PROCESSED / "y_test.parquet").iloc[:, 0].values
    raw_tr   = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    raw_test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    sp       = np.load(DATA_PROCESSED / "train_splits.npz")
    sample   = raw_test.sample(1000, random_state=RANDOM_STATE).reset_index(drop=True)

    pay = build_payment_features()
    specs = {"M1_monotonico": (True, False)}
    if pay is not None:
        specs.update({"M2_pagos": (False, True), "M3_monotonico_pagos": (True, True)})

    results = []
    # Champion como referencia, con las mismas métricas
    ms = policy["master_scale"]
    pd_champ = base.pd_from_features(X_test)
    results.append(evaluate("Champion (XGBoost + Platt)", pd_champ, y_test, raw_test,
                            monotonicity_share(base.pd_from_raw, sample), policy, cutoff,
                            X_test.shape[1], int(base.model.best_iteration),
                            base.grade_table, pd_champ))

    CHALLENGERS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (mono, use_pay) in specs.items():
        logger.info(f"Entrenando {name}...")
        drop = DROP_FOR_MONOTONE if mono else []
        cs = ChallengerScorer(base, None, None, drop, pay if use_pay else None)
        Xtr = cs.features(X_train, raw_tr["SK_ID_CURR"].values)
        Xte = cs.features(X_test, raw_test["SK_ID_CURR"].values)
        constraints = {c: 1 for c in MONOTONE_UP} if mono else None
        model = _train_xgb(Xtr.iloc[sp["fit"]], y_train.iloc[sp["fit"]],
                           Xtr.iloc[sp["val"]], y_train.iloc[sp["val"]], constraints)
        cal = PlattCalibrator().fit(model.predict_proba(Xtr.iloc[sp["cal"]])[:, 1], y_train.iloc[sp["cal"]])
        cs.model, cs.calibrator = model, cal
        pd_te = cal.transform(model.predict_proba(Xte)[:, 1])
        pd_va = cal.transform(model.predict_proba(Xtr.iloc[sp["val"]])[:, 1])
        grade_table = estimate_grade_pd(y_train.values[sp["val"]], pd_va,
                                        ms["limites_superiores"], ms["grados"])
        r = evaluate(name, pd_te, y_test, raw_test, monotonicity_share(cs.pd_from_raw, sample),
                     policy, cutoff, Xte.shape[1], int(model.best_iteration), grade_table, pd_champ)
        results.append(r)
        logger.info(f"  {name}: Gini {r['gini']} (Δ IC95% {r['delta_gini_ci95']}) | ECE {r['ece']} | "
                    f"monotonía (tendencia) {r['monotonic_trend_violation']:.1%} | rojo sin/con MoC "
                    f"{r['grades_red'] or '-'} / {r['grades_red_with_moc'] or '-'}")
        with open(CHALLENGERS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump({"model": model, "calibrator": cal, "drop_cols": drop,
                         "uses_payments": use_pay, "grade_pd": grade_table}, f)
        if name.startswith("M2") or name.startswith("M3"):
            imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
            r["top_features_gain"] = {k.replace("num__", "").replace("cat__", ""): round(float(v), 2)
                                      for k, v in imp.sort_values(ascending=False).head(10).items()}

    champ_gini = results[0]["gini"]
    for r in results:
        r["delta_gini_vs_champion"] = round(r["gini"] - champ_gini, 4)

    dl_path = ROOT / "models" / "dl" / "dl_metadata.json"
    dl = json.load(open(dl_path)) if dl_path.exists() else {}
    report = {
        "cutoff_pd": cutoff,
        "models": results,
        "lstm_stack_reference": dl.get("stack_xgb_lstm"),
        "notes": {
            "monotonic_trend_violation": "% de clientes con pendiente negativa de la PD al escalar monto, "
                                         "cuota y valor del bien de 0,5x a 2x (1.000 clientes de test)",
            "monotonic_any_step_violation": "% con algún escalón descendente, aunque sea mínimo",
            "grades_red_with_moc": "backtest por grado con PD de grado (MoC estimado en validación)",
            "payments": "Variables agregadas de installments_payments (cuotas de créditos anteriores)",
        },
    }
    with open(REPORTS_DIR / "challengers.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _plot(results, report["lstm_stack_reference"])
    logger.success("Challengers OK — reports/challengers.json")
    return report


def _plot(results, lstm_ref):
    names = [r["model"] for r in results]
    ginis = [r["gini"] for r in results]
    err = np.array([[r["gini"] - r["gini_ci95"][0], r["gini_ci95"][1] - r["gini"]] for r in results]).T
    mono = [100 * r["monotonic_trend_violation"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = ["#888780"] + ["#534AB7"] * (len(results) - 1)
    axes[0].barh(names, ginis, xerr=err, color=colors, capsize=4)
    if lstm_ref:
        axes[0].axvline(lstm_ref["gini"], color="#D85A30", ls="--", label=f"stack XGB+LSTM ({lstm_ref['gini']:.3f})")
        axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_xlim(min(ginis) - 0.04, max(ginis) + 0.03)
    axes[0].set_title("Gini test (IC 95% bootstrap)")
    axes[1].barh(names, mono, color=["#D85A30" if m > 0 else "#1D9E75" for m in mono])
    for i, m in enumerate(mono):
        axes[1].text(m, i, f" {m:.1f}%", va="center", fontsize=9)
    axes[1].set_title("% de clientes cuya PD baja al pedir más monto")
    for ax in axes:
        ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "challengers.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_challengers()
