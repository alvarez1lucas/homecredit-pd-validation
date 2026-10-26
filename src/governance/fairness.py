"""
Fairness & Bias Analysis — Home Credit Default Risk
====================================================
Atributos protegidos analizados:
  - CODE_GENDER (no es feature del modelo)
  - Edad en buckets (age_years SÍ es feature del modelo)

La decisión se evalúa con la PD calibrada y el cutoff de la política de crédito
(configs/credit_policy.json), que es la regla que se usaría en producción.

Métricas: Demographic Parity Difference (máx − mín tasa de aprobación),
Disparate Impact Ratio (mín / máx), Equalized Odds (gaps de TPR/FPR), AUC por grupo.
Referencias: ECOA / Reg B (regla de los 4/5 como heurística), EU AI Act Art. 10.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.config import DATA_PROCESSED, FIGURES_DIR, REPORTS_DIR, load_policy, reference_cutoff
from src.models.pd_model import PDScorer
from src.validation.model_validation import age_groups

MIN_GROUP_SIZE = 30


def _rates(y_approved, group) -> dict:
    return {str(g): float(y_approved[group == g].mean()) for g in np.unique(group)}


def demographic_parity_difference(y_approved, group) -> float:
    """Diferencia máxima de tasa de aprobación entre grupos (funciona con 2+ grupos)."""
    rates = list(_rates(y_approved, group).values())
    if len(rates) < 2:
        return float("nan")
    return round(max(rates) - min(rates), 4)


def disparate_impact_ratio(y_approved, group) -> float:
    rates = list(_rates(y_approved, group).values())
    if len(rates) < 2 or max(rates) == 0:
        return float("nan")
    return round(min(rates) / max(rates), 4)


def equalized_odds(y_true, y_pred, group) -> dict:
    """
    TPR/FPR por grupo, con y_pred = 1 si el modelo rechaza (predice default).
    tpr = defaults correctamente rechazados; fpr = buenos pagadores rechazados.
    """
    result = {}
    for g in np.unique(group):
        m = group == g
        if m.sum() < MIN_GROUP_SIZE:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel()
        result[str(g)] = {
            "tpr": round(float(tp / (tp + fn + 1e-9)), 4),
            "fpr": round(float(fp / (fp + tn + 1e-9)), 4),
        }
    if len(result) >= 2:
        tprs = [v["tpr"] for v in result.values()]
        fprs = [v["fpr"] for v in result.values()]
        result["tpr_gap"] = round(max(tprs) - min(tprs), 4)
        result["fpr_gap"] = round(max(fprs) - min(fprs), 4)
    return result


def group_auc(y_true, y_score, group) -> dict:
    result = {}
    for g in np.unique(group):
        m = group == g
        if m.sum() >= MIN_GROUP_SIZE and len(np.unique(y_true[m])) == 2:
            result[str(g)] = round(float(roc_auc_score(y_true[m], y_score[m])), 4)
        else:
            result[str(g)] = None
    return result


def analyze_attribute(y_true, pd_hat, group, attr_name, cutoff, th) -> dict:
    y_true     = np.asarray(y_true)
    y_pred     = (pd_hat >= cutoff).astype(int)      # 1 = rechazado
    y_approved = 1 - y_pred

    dpd    = demographic_parity_difference(y_approved, group)
    dir_r  = disparate_impact_ratio(y_approved, group)
    eq_odd = equalized_odds(y_true, y_pred, group)

    flags = []
    if not np.isnan(dpd) and dpd > th["fairness_dpd_max"]:
        flags.append(f"DPD {dpd:.3f} supera {th['fairness_dpd_max']}")
    if not np.isnan(dir_r) and dir_r < th["fairness_dir_min"]:
        flags.append(f"DIR {dir_r:.3f} por debajo de {th['fairness_dir_min']} (regla 4/5)")
    if eq_odd.get("tpr_gap", 0) > th["fairness_eo_gap_max"]:
        flags.append(f"TPR gap {eq_odd['tpr_gap']:.3f} supera {th['fairness_eo_gap_max']}")
    if eq_odd.get("fpr_gap", 0) > th["fairness_eo_gap_max"]:
        flags.append(f"FPR gap {eq_odd['fpr_gap']:.3f} supera {th['fairness_eo_gap_max']}")

    for flag in flags:
        logger.warning(f"  [FAIRNESS] {attr_name}: {flag}")
    if not flags:
        logger.info(f"  {attr_name}: OK — todos los checks de fairness pasan")

    return {
        "approval_rates":                {k: round(v, 4) for k, v in _rates(y_approved, group).items()},
        "default_rates":                 {str(g): round(float(y_true[group == g].mean()), 4)
                                          for g in np.unique(group)},
        "mean_pd":                       {str(g): round(float(pd_hat[group == g].mean()), 4)
                                          for g in np.unique(group)},
        "demographic_parity_difference": dpd,
        "disparate_impact_ratio":        dir_r,
        "equalized_odds":                eq_odd,
        "auc_by_group":                  group_auc(y_true, pd_hat, group),
        "regulatory_flags":              flags,
        "passed":                        len(flags) == 0,
    }


def plot_fairness(fairness_results: dict, cutoff: float):
    n = len(fairness_results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.8 * n))
    axes = np.atleast_2d(axes)

    for idx, (attr, res) in enumerate(fairness_results.items()):
        ax_bar, ax_eqo = axes[idx]
        groups = list(res["approval_rates"].keys())
        x = np.arange(len(groups))
        ax_bar.bar(x - 0.2, [res["approval_rates"][g] for g in groups], 0.4,
                   color="#534AB7", label="Tasa de aprobación")
        ax_bar.bar(x + 0.2, [res["default_rates"][g] for g in groups], 0.4,
                   color="#D85A30", label="Tasa de default real")
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(groups)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_title(f"{attr}: aprobación vs default real (cutoff PD {cutoff:.1%})")
        ax_bar.legend(fontsize=9)
        for xi, g in zip(x, groups):
            ax_bar.text(xi - 0.2, res["approval_rates"][g] + 0.01,
                        f"{res['approval_rates'][g]:.0%}", ha="center", fontsize=8)

        eq = res.get("equalized_odds", {})
        gkeys = [k for k in eq if k not in ("tpr_gap", "fpr_gap")]
        xe = np.arange(len(gkeys))
        ax_eqo.bar(xe - 0.18, [eq[k]["tpr"] for k in gkeys], 0.35,
                   label="TPR (defaults detectados)", color="#1D9E75")
        ax_eqo.bar(xe + 0.18, [eq[k]["fpr"] for k in gkeys], 0.35,
                   label="FPR (buenos pagadores rechazados)", color="#D85A30")
        ax_eqo.set_xticks(xe); ax_eqo.set_xticklabels(gkeys)
        ax_eqo.set_ylim(0, 1)
        ax_eqo.set_title(f"{attr}: Equalized Odds")
        ax_eqo.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def compute_fairness():
    logger.info("=" * 50)
    logger.info("STAGE 5 — Fairness Analysis")
    logger.info("=" * 50)

    policy = load_policy()
    th     = policy["umbrales_validacion"]
    cutoff = reference_cutoff(policy)

    scorer   = PDScorer()
    X_test   = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_test   = pd.read_parquet(DATA_PROCESSED / "y_test.parquet").iloc[:, 0].values
    raw_test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    pd_test  = scorer.pd_from_features(X_test)

    results = {}

    gender = raw_test["CODE_GENDER"].fillna("Unknown").values
    mask   = np.isin(gender, ["M", "F"])            # excluye XNA (4 casos en todo el dataset)
    results["gender"] = analyze_attribute(
        y_test[mask], pd_test[mask], gender[mask], "gender", cutoff, th
    )

    age_group = age_groups(raw_test["age_years"]).values
    results["age_group"] = analyze_attribute(
        y_test, pd_test, age_group, "age_group", cutoff, th
    )

    overall_pass = all(v["passed"] for v in results.values())
    report = {
        "overall_fairness_passed":       overall_pass,
        "decision_rule":                 f"rechazar si PD calibrada >= {cutoff} "
                                         f"(break-even {policy['mercado_referencia']})",
        "cutoff_pd":                     cutoff,
        "protected_attributes_analyzed": list(results.keys()),
        "attribute_in_model":            {"gender": False, "age_group": True},
        "results":                       results,
        "thresholds": {
            "demographic_parity_max": th["fairness_dpd_max"],
            "disparate_impact_min":   th["fairness_dir_min"],
            "equalized_odds_max_gap": th["fairness_eo_gap_max"],
        },
        "references": ["ECOA / Regulation B", "EU AI Act Art. 10"],
    }
    with open(REPORTS_DIR / "fairness_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(REPORTS_DIR / "fairness_metrics.json", "w") as f:
        json.dump({
            "fairness_pass": int(overall_pass),
            "gender_dpd": results["gender"]["demographic_parity_difference"],
            "gender_dir": results["gender"]["disparate_impact_ratio"],
            "age_dpd":    results["age_group"]["demographic_parity_difference"],
            "age_dir":    results["age_group"]["disparate_impact_ratio"],
        }, f, indent=2)

    plot_fairness(results, cutoff)
    logger.success(f"Fairness: {'PASSED' if overall_pass else 'FAILED — revisar flags'}")
    return report


if __name__ == "__main__":
    compute_fairness()
