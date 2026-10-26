"""
Validación del modelo de PD — Home Credit Default Risk
======================================================
Marco de referencia: SR 26-2 (Fed/OCC/FDIC, abril 2026), que reemplaza a
SR 11-7 (2011) y mantiene su estructura de validación. Tests:

  1. Poder discriminatorio: Gini (con IC 95% bootstrap), AUC, KS, lift vs benchmark
  2. Calibración: Hosmer-Lemeshow, ECE, calibration-in-the-large, Brier
     (sobre el score crudo y sobre la PD calibrada)
  3. Master scale: test binomial y de Jeffreys por grado
  4. Estabilidad: PSI train vs test
  5. Stress test: shocks sobre variables crudas → impacto en PD y pérdida esperada
  6. Sensibilidad: permutation importance sobre las features más importantes
  7. Performance por segmento
"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.config import (DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, RANDOM_STATE,
                        REPORTS_DIR, load_policy, reference_cutoff)
from src.models.pd_model import PDScorer
from src.validation.master_scale import master_scale_summary, master_scale_table


# ── 1. Discriminación ─────────────────────────────────────────────────────────
def gini(y_true, y_score) -> float:
    return round(2 * roc_auc_score(y_true, y_score) - 1, 4)


def gini_bootstrap_ci(y_true, y_score, n_boot=200, seed=RANDOM_STATE) -> list:
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    ginis = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ginis.append(2 * roc_auc_score(y_true[i], y_score[i]) - 1)
    return [round(float(q), 4) for q in np.percentile(ginis, [2.5, 97.5])]


def compute_ks(y_true, y_score) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    ks_stat, ks_p = stats.ks_2samp(y_score[y_true == 0], y_score[y_true == 1])
    return {"ks_statistic": round(float(ks_stat), 4), "ks_pvalue": round(float(ks_p), 4)}


# ── 2. Calibración ────────────────────────────────────────────────────────────
def hosmer_lemeshow(y_true, y_score, n_bins=10) -> dict:
    """
    Con muestras grandes (60k filas) el HL rechaza ante desvíos mínimos:
    se informa, pero el criterio de aprobación usa ECE y calibration-in-the-large.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["decile"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        obs_events=("y", "sum"), total=("y", "count"), mean_pred=("p", "mean"),
    )
    g["exp_events"]    = g["total"] * g["mean_pred"]
    g["exp_nonevents"] = g["total"] * (1 - g["mean_pred"])
    g["obs_nonevents"] = g["total"] - g["obs_events"]
    hl = (
        ((g["obs_events"]    - g["exp_events"])    ** 2 / g["exp_events"].clip(1e-6)).sum()
        + ((g["obs_nonevents"] - g["exp_nonevents"]) ** 2 / g["exp_nonevents"].clip(1e-6)).sum()
    )
    p = 1 - stats.chi2.cdf(hl, len(g) - 2)
    return {
        "hl_statistic":    round(float(hl), 2),
        "hl_pvalue":       round(float(p), 4),
        "well_calibrated": bool(p > 0.05),
    }


def expected_calibration_error(y_true, y_score, n_bins=10) -> float:
    """ECE con bins de igual frecuencia: Σ peso_bin · |PD media − default observado|."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return round(float((g["n"] / len(df) * (g["p"] - g["y"]).abs()).sum()), 4)


def calibration_report(y_true, y_score) -> dict:
    mean_pd, dr = float(np.mean(y_score)), float(np.mean(y_true))
    return {
        **hosmer_lemeshow(y_true, y_score),
        "ece":            expected_calibration_error(y_true, y_score),
        "mean_pd":        round(mean_pd, 4),
        "default_rate":   round(dr, 4),
        "citl_rel_diff":  round((mean_pd - dr) / dr, 4),
        "brier":          round(float(brier_score_loss(y_true, y_score)), 4),
    }


# ── 4. Estabilidad ────────────────────────────────────────────────────────────
def compute_psi(expected, actual, n_bins=10) -> float:
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6; bins[-1] += 1e-6
    def bucket(d):
        c, _ = np.histogram(d, bins=bins)
        p = c / len(d)
        return np.where(p == 0, 1e-4, p)
    e, a = bucket(expected), bucket(actual)
    return round(float(np.sum((a - e) * np.log(a / e))), 4)


# ── 5. Stress test ────────────────────────────────────────────────────────────
STRESS_SCENARIOS = {
    "income_shock_moderate": {
        "description": "Ingreso real −25% (desempleo / inflación no compensada)",
        "shocks": {"AMT_INCOME_TOTAL": 0.75},
    },
    "income_shock_severe": {
        "description": "Ingreso real −40%",
        "shocks": {"AMT_INCOME_TOTAL": 0.60},
    },
    "bureau_deterioration": {
        "description": "Scores de bureau (EXT_SOURCE 1-3) −20%",
        "shocks": {"EXT_SOURCE_1": 0.80, "EXT_SOURCE_2": 0.80, "EXT_SOURCE_3": 0.80},
    },
    "combined_recession": {
        "description": "Recesión: ingreso −25% + bureau −15% + cuotas +15% (tasas)",
        "shocks": {"AMT_INCOME_TOTAL": 0.75, "AMT_ANNUITY": 1.15,
                   "EXT_SOURCE_1": 0.85, "EXT_SOURCE_2": 0.85, "EXT_SOURCE_3": 0.85},
    },
}


def stress_test(scorer: PDScorer, raw_test: pd.DataFrame, y_test, lgd: float, cutoff: float) -> dict:
    """
    Shocks aplicados sobre las variables CRUDAS; los ratios derivados se recalculan
    y la PD se vuelve a estimar con el modelo calibrado.
    Mide la sensibilidad del modelo a los shocks (no es un modelo macro satélite).
    EL = PD × LGD × EAD, con EAD = AMT_CREDIT y LGD del mercado de referencia.
    """
    ead = raw_test["AMT_CREDIT"].values

    def summarize(pd_hat):
        pd_el = scorer.grade_pd(pd_hat)          # pérdida esperada con PD de grado (con MoC)
        return {
            "mean_pd":       round(float(pd_hat.mean()), 4),
            "expected_loss": round(float((pd_el * lgd * ead).sum()), 0),
            "el_rate":       round(float((pd_el * lgd * ead).sum() / ead.sum()), 4),
            "approval_rate": round(float((pd_hat < cutoff).mean()), 4),
            "auc":           round(float(roc_auc_score(y_test, pd_hat)), 4),
        }

    base = summarize(scorer.pd_from_raw(raw_test))
    results = {"lgd": lgd, "cutoff_pd": cutoff, "baseline": base, "scenarios": {}}
    for name, sc in STRESS_SCENARIOS.items():
        df_s = raw_test.copy()
        for col, mult in sc["shocks"].items():
            df_s[col] = df_s[col] * mult
        s = summarize(scorer.pd_from_raw(df_s))
        s["description"]     = sc["description"]
        s["pd_change_rel"]   = round(s["mean_pd"] / base["mean_pd"] - 1, 4)
        s["el_change_rel"]   = round(s["expected_loss"] / base["expected_loss"] - 1, 4)
        s["approval_change"] = round(s["approval_rate"] - base["approval_rate"], 4)
        results["scenarios"][name] = s
        logger.info(
            f"  Stress [{name}]: PD media {s['mean_pd']:.2%} ({s['pd_change_rel']:+.1%}) | "
            f"EL {s['el_change_rel']:+.1%} | aprobación {s['approval_change']:+.1%}"
        )
    return results


# ── 6. Sensibilidad ───────────────────────────────────────────────────────────
def sensitivity_analysis(model, X_test, y_test, top_n=20, seed=RANDOM_STATE) -> dict:
    """Permutation importance (caída de AUC) sobre las top-N features por gain."""
    rng = np.random.default_rng(seed)
    gain = model.get_booster().get_score(importance_type="gain")
    top = sorted(gain, key=gain.get, reverse=True)[:top_n]
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    imp = {}
    for feat in top:
        X_p = X_test.copy()
        X_p[feat] = rng.permutation(X_p[feat].values)
        imp[feat] = round(float(base_auc - roc_auc_score(y_test, model.predict_proba(X_p)[:, 1])), 5)
    return dict(sorted(imp.items(), key=lambda x: -x[1]))


# ── 7. Segmentos ──────────────────────────────────────────────────────────────
def age_groups(age_years) -> pd.Series:
    return pd.cut(age_years, bins=[0, 25, 35, 45, 55, 100],
                  labels=["<25", "25-35", "35-45", "45-55", ">55"]).astype(str)


def segment_performance(raw_test, y_test, pd_hat) -> dict:
    df = pd.DataFrame({"y": np.asarray(y_test), "pd": pd_hat,
                       "age_group": age_groups(raw_test["age_years"]).values,
                       "income_type": raw_test["NAME_INCOME_TYPE"].values,
                       "gender": raw_test["CODE_GENDER"].values})
    out = {}
    for seg in ["age_group", "income_type", "gender"]:
        rows = {}
        for g, sub in df.groupby(seg):
            if len(sub) < 500 or sub["y"].nunique() < 2:
                continue
            rows[str(g)] = {"n": int(len(sub)), "gini": gini(sub["y"], sub["pd"]),
                            "mean_pd": round(float(sub["pd"].mean()), 4),
                            "default_rate": round(float(sub["y"].mean()), 4)}
        out[seg] = rows
    return out


# ── Gráficos ──────────────────────────────────────────────────────────────────
def plot_roc_ks(y_true, y_score, baseline_score):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, color="#534AB7", linewidth=2,
                 label=f"XGBoost (Gini={gini(y_true, y_score):.3f})")
    fb, tb, _ = roc_curve(y_true, baseline_score)
    axes[0].plot(fb, tb, color="#888780", linewidth=1.5, linestyle="--",
                 label=f"Benchmark logístico (Gini={gini(y_true, baseline_score):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.5)
    axes[0].set_title("Curva ROC — test set"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")

    thresholds = np.quantile(y_score, np.linspace(0, 1, 200))
    cum_pos = np.array([(y_score[y_true == 1] <= t).mean() for t in thresholds])
    cum_neg = np.array([(y_score[y_true == 0] <= t).mean() for t in thresholds])
    ks_vals = np.abs(cum_pos - cum_neg)
    k = np.argmax(ks_vals)
    axes[1].plot(thresholds, cum_pos, color="#D85A30", linewidth=2, label="Defaults")
    axes[1].plot(thresholds, cum_neg, color="#1D9E75", linewidth=2, label="No defaults")
    axes[1].axvline(thresholds[k], color="#534AB7", linestyle="--", linewidth=1.5,
                    label=f"KS={ks_vals[k]:.3f}")
    axes[1].set_title("KS — distribución acumulada de la PD"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("PD calibrada"); axes[1].set_ylabel("CDF")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_ks.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, raw_score, pd_cal):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, lim in [(axes[0], 1.0), (axes[1], 0.35)]:
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Calibración perfecta")
        for s, lab, c in [(raw_score, "Score crudo XGBoost", "#D85A30"),
                          (pd_cal, "PD calibrada (Platt)", "#534AB7")]:
            pt, pp = calibration_curve(y_true, s, n_bins=10, strategy="quantile")
            ax.plot(pp, pt, "o-", color=c, linewidth=2, label=lab)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("PD predicha (media del decil)"); ax.set_ylabel("Default observado")
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_title("Reliability diagram — antes y después de calibrar")
    axes[1].set_title("Zoom: rango 0–35%")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_master_scale(table: pd.DataFrame, pre: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(table))
    ax.bar(x, table["share"], color="#E8E6F7", label="% de la cartera")
    ax2 = ax.twinx()
    ci = np.array(table["dr_ci95"].tolist())
    dr = table["default_rate"].values
    ax2.errorbar(x, dr, yerr=[dr - ci[:, 0], ci[:, 1] - dr], fmt="o", color="#D85A30",
                 capsize=4, label="Default observado (IC 95% Jeffreys)")
    if pre is not None and not pre["pd_avg"].equals(table["pd_avg"]):
        ax2.plot(x, pre["pd_avg"], "^:", color="#888780", label="PD individual media (sin MoC)")
        ax2.plot(x, table["pd_avg"], "s--", color="#534AB7", label="PD de grado (con MoC)")
    else:
        ax2.plot(x, table["pd_avg"], "s--", color="#534AB7", label="PD promedio asignada")
    ax.set_xticks(x); ax.set_xticklabels(table["grade"])
    ax.set_ylabel("% de la cartera"); ax2.set_ylabel("Tasa de default")
    ax.set_title("Master scale — PD asignada vs default observado por grado (test)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "master_scale.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_stress(stress: dict):
    names = ["baseline"] + list(stress["scenarios"])
    el = [stress["baseline"]["el_rate"]] + [s["el_rate"] for s in stress["scenarios"].values()]
    pdm = [stress["baseline"]["mean_pd"]] + [s["mean_pd"] for s in stress["scenarios"].values()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["#888780"] + ["#D85A30"] * (len(names) - 1)
    axes[0].barh(names, pdm, color=colors); axes[0].set_title("PD media por escenario")
    axes[1].barh(names, el, color=colors); axes[1].set_title("Pérdida esperada / exposición")
    for ax, vals in [(axes[0], pdm), (axes[1], el)]:
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2%}", va="center", fontsize=9)
        ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stress_test.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Orquestación ──────────────────────────────────────────────────────────────
def validate():
    logger.info("=" * 50)
    logger.info("STAGE 4 — Validación del modelo (SR 26-2 / SR 11-7)")
    logger.info("=" * 50)

    policy = load_policy()
    th     = policy["umbrales_validacion"]
    cutoff = reference_cutoff(policy)
    lgd    = policy["mercados"][policy["mercado_referencia"]]["lgd"]
    ms     = policy["master_scale"]

    scorer = PDScorer()
    with open(MODELS_DIR / "baseline.pkl", "rb") as f:
        baseline = pickle.load(f)

    X_train  = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_test   = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_test   = pd.read_parquet(DATA_PROCESSED / "y_test.parquet").iloc[:, 0].values
    raw_test = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    raw_score  = scorer.raw_score(X_test)
    pd_test    = scorer.calibrator.transform(raw_score)
    pd_train   = scorer.pd_from_features(X_train)
    base_score = baseline.predict_proba(X_test)[:, 1]

    # 1. Discriminación
    g, g_base = gini(y_test, pd_test), gini(y_test, base_score)
    disc = {
        "gini": g, "gini_ci95": gini_bootstrap_ci(y_test, pd_test),
        "auc_roc": round(float(roc_auc_score(y_test, pd_test)), 4),
        "gini_benchmark": g_base, "gini_lift_benchmark": round(g - g_base, 4),
        **compute_ks(y_test, pd_test),
    }
    # 2. Calibración
    cal = {"raw_score": calibration_report(y_test, raw_score),
           "calibrated_pd": calibration_report(y_test, pd_test)}
    # 3. Master scale: se testea la PD de grado con MoC (la que va a provisiones);
    #    el resultado con la PD individual se conserva como "pre_moc"
    ms_pre     = master_scale_table(y_test, pd_test, ms["limites_superiores"], ms["grados"])
    ms_pre_sum = master_scale_summary(ms_pre)
    if scorer.grade_table is not None:
        ms_table = master_scale_table(y_test, scorer.grade_pd(pd_test), ms["limites_superiores"],
                                      ms["grados"], pd_rank=pd_test)
    else:
        ms_table = ms_pre
    ms_sum = master_scale_summary(ms_table)
    ead = raw_test["AMT_CREDIT"].values
    moc_el_impact = float((scorer.grade_pd(pd_test) * ead).sum() / (pd_test * ead).sum() - 1)
    # 4. Estabilidad
    psi = compute_psi(pd_train, pd_test)
    # 5-7
    stress = stress_test(scorer, raw_test, y_test, lgd, cutoff)
    sens   = sensitivity_analysis(scorer.model, X_test, y_test)
    segs   = segment_performance(raw_test, y_test, pd_test)

    c = cal["calibrated_pd"]
    checks = {
        "gini":           {"value": g, "threshold": f">= {th['gini_min']}", "pass": g >= th["gini_min"]},
        "ks":             {"value": disc["ks_statistic"], "threshold": f">= {th['ks_min']}",
                           "pass": disc["ks_statistic"] >= th["ks_min"]},
        "lift_benchmark": {"value": disc["gini_lift_benchmark"], "threshold": "> 0",
                           "pass": disc["gini_lift_benchmark"] > 0},
        "ece":            {"value": c["ece"], "threshold": f"<= {th['ece_max']}", "pass": c["ece"] <= th["ece_max"]},
        "citl":           {"value": c["citl_rel_diff"], "threshold": f"|x| <= {th['citl_max_rel']}",
                           "pass": abs(c["citl_rel_diff"]) <= th["citl_max_rel"]},
        "master_scale":   {"value": ms_sum["grades_underestimating"] or "ninguno",
                           "threshold": "sin grados en rojo (Jeffreys p < 1%)",
                           "pass": not ms_sum["grades_underestimating"]},
        "psi":            {"value": psi, "threshold": f"<= {th['psi_max']}", "pass": psi <= th["psi_max"]},
    }
    overall = all(v["pass"] for v in checks.values())

    for k, v in checks.items():
        logger.info(f"  {'✔' if v['pass'] else '✘'} {k:<15} {v['value']}  ({v['threshold']})")

    report = {
        "framework": "SR 26-2 (abr 2026, reemplaza a SR 11-7) — estructura de validación de SR 11-7",
        "overall_pass": overall,
        "checks": checks,
        "discriminatory_power": disc,
        "calibration": cal,
        "master_scale": {
            "pd_tested": "PD de grado con MoC" if scorer.grade_table is not None else "PD individual",
            "table": ms_table.to_dict(orient="records"), "summary": ms_sum,
            "pre_moc": {"table": ms_pre.to_dict(orient="records"), "summary": ms_pre_sum},
            "moc": scorer.grade_table,
            "moc_el_impact": round(moc_el_impact, 4),
        },
        "stability": {"psi_train_test": psi,
                      "note": "Split aleatorio: el PSI mide homogeneidad train/test, no estabilidad temporal."},
        "stress_testing": stress,
        "sensitivity_top": sens,
        "segment_performance": segs,
        "cutoff_policy": {"market": policy["mercado_referencia"], "cutoff_pd": cutoff, "lgd": lgd},
    }
    with open(REPORTS_DIR / "model_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(REPORTS_DIR / "validation_metrics.json", "w") as f:
        json.dump({"gini": g, "auc_roc": disc["auc_roc"], "ks": disc["ks_statistic"],
                   "ece": c["ece"], "psi": psi, "validation_pass": int(overall)}, f, indent=2)

    plot_roc_ks(y_test, pd_test, base_score)
    plot_calibration(y_test, raw_score, pd_test)
    plot_master_scale(ms_table, ms_pre)
    plot_stress(stress)

    logger.success(f"Validación: {'PASS' if overall else 'FAIL — ver checks'}")
    return report


if __name__ == "__main__":
    validate()
