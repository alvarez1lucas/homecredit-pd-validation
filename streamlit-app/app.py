"""
Home Credit — PD Model Validation Dashboard
===========================================
Lee únicamente resultados reales del pipeline:
  - reports/*.json
  - reports/dashboard/test_predictions.parquet  (PD calibrada + default real del test set)
El simulador individual se habilita sólo si los artefactos del modelo están presentes.

    streamlit run streamlit-app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT    = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="PD Model Validation — Home Credit", page_icon="📊", layout="wide")

PURPLE, GREEN, CORAL, GRAY = "#534AB7", "#1D9E75", "#D85A30", "#888780"

# ─────────────────────────────────────────────────────────────────────────────
# Textos
# ─────────────────────────────────────────────────────────────────────────────
T = {
    "es": {
        "pages": ["Resumen", "Discriminación y calibración", "Stress test", "Fairness",
                  "Política de crédito", "Scorecard", "Challengers", "Simulador", "Model Card"],
        "title": "Validación de un modelo de PD — Home Credit",
        "subtitle": "Proyecto de portfolio. Todos los números salen del pipeline sobre el test set "
                    "(61.503 solicitudes no usadas para entrenar). Marco: SR 26-2 (abr 2026), que "
                    "reemplaza a SR 11-7 y mantiene su estructura de validación.",
        "checks": "Checks de validación", "check": "Check", "value": "Valor",
        "threshold": "Umbral", "status": "Estado",
        "findings": "Hallazgos principales",
        "f_rank": "El modelo ordena bien el riesgo: Gini {g:.3f} (IC 95% {lo:.3f}–{hi:.3f}), "
                  "{lift:+.3f} sobre el benchmark logístico.",
        "f_cal": "El score crudo NO es una PD (media {raw:.1%} vs default real {dr:.1%}). "
                 "Tras calibrar con Platt: PD media {cal:.1%}, ECE {ece:.3f}.",
        "f_fair": "Fairness: {flags}",
        "f_fair_ok": "Fairness: sin flags con la regla de decisión actual.",
        "f_seg": "Segmentos con menor discriminación: {segs}.",
        "dec_table": "Default real por decil de PD",
        "decile": "Decil", "dr": "Default real", "mean_pd": "PD media", "share_def": "% de defaults capturados",
        "reliability": "Calibración — PD predicha vs default observado (deciles)",
        "raw_vs_cal": "Score crudo vs PD calibrada",
        "ms": "Master scale — backtesting por grado (test de Jeffreys, semáforo)",
        "grade": "Grado",
        "stress_note": "Shocks sobre variables crudas → PD recalculada con el modelo calibrado → "
                       "pérdida esperada (PD × LGD × monto). Mide la sensibilidad del modelo, no es un modelo macro.",
        "scenario": "Escenario", "el": "Pérdida esperada / exposición",
        "fair_rule": "Regla de decisión",
        "fair_slider": "Explorar otro cutoff de PD",
        "approval": "Tasa de aprobación", "attribute": "Atributo",
        "in_model": "feature del modelo", "not_in_model": "no es feature del modelo",
        "policy_intro": "Con PD calibrada, el cutoff que hace cero la ganancia esperada de un préstamo "
                        "a un período es **PD\\* = r / (r + LGD)**. La curva muestra la ganancia empírica "
                        "en el test set (por unidad prestada) para cada cutoff.",
        "market": "Mercado", "rate": "Tasa anual", "lgd": "LGD",
        "breakeven": "PD* analítico", "emp_opt": "Óptimo empírico", "approval_at": "Aprobación en PD*",
        "profit": "Ganancia por unidad prestada", "cutoff": "Cutoff de PD",
        "policy_caveat": "Supuesto simplificado: no descuenta fondeo, costos operativos ni inflación. "
                         "Con la TEA nominal de Argentina el cutoff real sería bastante más bajo.",
        "sc_intro": "Scorecard tradicional: binning con WoE monótono, selección por Information Value, "
                    "regresión logística y escala de puntos ({base} puntos = odds {odds}, PDO {pdo}).",
        "sc_gini": "Gini scorecard", "sc_vars": "Variables", "sc_gap": "Gap vs XGBoost",
        "variable": "Variable",
        "sim_off": "El simulador necesita los artefactos del modelo (correr `python run_all.py`). "
                   "En la versión online sólo se muestran los resultados de validación.",
        "sim_btn": "Calcular PD", "pd": "PD calibrada", "decision": "Decisión",
        "approved": "Aprobado", "rejected": "Rechazado", "factors": "Factores principales (SHAP)",
        "card_off": "Model Card no encontrado. Correr `python run_all.py`.",
        "missing": "No se encontraron los reportes del pipeline. Correr `python run_all.py` primero.",
        "pass": "✅ PASS", "fail": "❌ FAIL",
        "status_title": "Estado de la validación", "findings_table": "Hallazgos y recomendaciones",
        "sev": "Severidad", "finding": "Hallazgo", "rec": "Recomendación", "ref": "Referencia",
        "moc_title": "PD de grado con margen de conservadurismo (MoC)",
        "moc_note": "El modelo ordena y decide con la PD individual; para pérdida esperada y provisiones cada grado "
                    "usa una **PD de grado** = máx(PD media, percentil 75 de Jeffreys del default observado en el set "
                    "de validación). Sin MoC, el backtest con la PD individual daba rojo en {red} y ámbar en {amber}. "
                    "Impacto en la pérdida esperada: {el:+.1%}.",
        "pd_val": "PD media (validación)", "dr_val": "Default observado (validación)",
        "pd_grade": "PD de grado", "factor": "Factor MoC",
        "ch_intro": "Challengers que responden a las observaciones abiertas, entrenados con el mismo esquema de "
                    "datos que el champion:\n\n"
                    "- **M1 — monotónico:** `monotone_constraints` en monto, cuota y ratios de carga (pedir más plata "
                    "nunca baja la PD).\n"
                    "- **M2 — pagos:** variables agregadas del historial de cuotas de créditos anteriores.\n"
                    "- **M3 — monotónico + pagos.**",
        "ch_reco": "Challenger recomendado", "ch_none": "Ningún challenger cumple todos los criterios.",
        "ch_rule": "Criterios: Δ Gini significativo (IC 95% pareado > 0), monótono en el monto y sin grados en rojo con MoC.",
        "ch_gini": "Gini test (IC 95% bootstrap)", "ch_mono": "% de clientes cuya PD baja al pedir más monto",
        "ch_lstm": "stack XGB + LSTM (notebook 06)", "ch_feats": "Variables con más ganancia en {m}",
        "model": "Modelo", "delta": "Δ Gini vs champion", "mono": "Monotonía (violación)",
        "red_pre": "Rojo sin MoC", "red_post": "Rojo con MoC",
        "el_sim": "Pérdida esperada",
    },
    "en": {
        "pages": ["Overview", "Discrimination & calibration", "Stress test", "Fairness",
                  "Credit policy", "Scorecard", "Challengers", "Simulator", "Model Card"],
        "title": "PD Model Validation — Home Credit",
        "subtitle": "Portfolio project. Every number comes from the pipeline on the test set "
                    "(61,503 applications never used for training). Framework: SR 26-2 (Apr 2026), "
                    "which supersedes SR 11-7 and keeps its validation structure.",
        "checks": "Validation checks", "check": "Check", "value": "Value",
        "threshold": "Threshold", "status": "Status",
        "findings": "Key findings",
        "f_rank": "The model ranks risk well: Gini {g:.3f} (95% CI {lo:.3f}–{hi:.3f}), "
                  "{lift:+.3f} over the logistic benchmark.",
        "f_cal": "The raw score is NOT a PD (mean {raw:.1%} vs actual default {dr:.1%}). "
                 "After Platt calibration: mean PD {cal:.1%}, ECE {ece:.3f}.",
        "f_fair": "Fairness: {flags}",
        "f_fair_ok": "Fairness: no flags under the current decision rule.",
        "f_seg": "Segments with weaker discrimination: {segs}.",
        "dec_table": "Actual default rate by PD decile",
        "decile": "Decile", "dr": "Default rate", "mean_pd": "Mean PD", "share_def": "% of defaults captured",
        "reliability": "Calibration — predicted PD vs observed default (deciles)",
        "raw_vs_cal": "Raw score vs calibrated PD",
        "ms": "Master scale — backtesting by grade (Jeffreys test, traffic light)",
        "grade": "Grade",
        "stress_note": "Shocks on raw variables → PD recomputed with the calibrated model → "
                       "expected loss (PD × LGD × amount). Measures model sensitivity; it is not a macro model.",
        "scenario": "Scenario", "el": "Expected loss / exposure",
        "fair_rule": "Decision rule",
        "fair_slider": "Explore another PD cutoff",
        "approval": "Approval rate", "attribute": "Attribute",
        "in_model": "model feature", "not_in_model": "not a model feature",
        "policy_intro": "With a calibrated PD, the cutoff that sets one-period expected profit to zero is "
                        "**PD\\* = r / (r + LGD)**. The curve shows the empirical profit on the test set "
                        "(per unit lent) for each cutoff.",
        "market": "Market", "rate": "Annual rate", "lgd": "LGD",
        "breakeven": "Analytical PD*", "emp_opt": "Empirical optimum", "approval_at": "Approval at PD*",
        "profit": "Profit per unit lent", "cutoff": "PD cutoff",
        "policy_caveat": "Simplified assumption: ignores funding cost, operating cost and inflation. "
                         "With Argentina's nominal rate the real cutoff would be much lower.",
        "sc_intro": "Traditional scorecard: monotonic WoE binning, Information Value selection, "
                    "logistic regression and a points scale ({base} points = odds {odds}, PDO {pdo}).",
        "sc_gini": "Scorecard Gini", "sc_vars": "Variables", "sc_gap": "Gap vs XGBoost",
        "variable": "Variable",
        "sim_off": "The simulator needs the model artifacts (run `python run_all.py`). "
                   "The online version only shows validation results.",
        "sim_btn": "Compute PD", "pd": "Calibrated PD", "decision": "Decision",
        "approved": "Approved", "rejected": "Rejected", "factors": "Top factors (SHAP)",
        "card_off": "Model Card not found. Run `python run_all.py`.",
        "missing": "Pipeline reports not found. Run `python run_all.py` first.",
        "pass": "✅ PASS", "fail": "❌ FAIL",
        "status_title": "Validation status", "findings_table": "Findings and recommendations (in Spanish)",
        "sev": "Severity", "finding": "Finding", "rec": "Recommendation", "ref": "Reference",
        "moc_title": "Grade PD with margin of conservatism (MoC)",
        "moc_note": "The model ranks and decides with the individual PD; for expected loss and provisioning each "
                    "grade uses a **grade PD** = max(mean PD, 75th Jeffreys percentile of the observed default rate "
                    "on the validation split). Without MoC, the backtest on individual PD was red in {red} and amber "
                    "in {amber}. Impact on expected loss: {el:+.1%}.",
        "pd_val": "Mean PD (validation)", "dr_val": "Observed default (validation)",
        "pd_grade": "Grade PD", "factor": "MoC factor",
        "ch_intro": "Challengers addressing the open findings, trained with the champion's data scheme:\n\n"
                    "- **M1 — monotonic:** `monotone_constraints` on amount, annuity and burden ratios (asking for "
                    "more never lowers PD).\n"
                    "- **M2 — payments:** aggregated features from previous loans' installment history.\n"
                    "- **M3 — monotonic + payments.**",
        "ch_reco": "Recommended challenger", "ch_none": "No challenger meets all criteria.",
        "ch_rule": "Criteria: significant Δ Gini (95% paired CI > 0), monotonic in amount and no red grades with MoC.",
        "ch_gini": "Test Gini (95% bootstrap CI)", "ch_mono": "% of applicants whose PD drops when asking for more",
        "ch_lstm": "XGB + LSTM stack (notebook 06)", "ch_feats": "Top features by gain in {m}",
        "model": "Model", "delta": "Δ Gini vs champion", "mono": "Monotonicity (violation)",
        "red_pre": "Red without MoC", "red_post": "Red with MoC",
        "el_sim": "Expected loss",
    },
}

STATUS_EN = {"NO APROBADO — requiere remediación": "NOT APPROVED — remediation required",
             "APROBADO CON CONDICIONES": "APPROVED WITH CONDITIONS", "APROBADO": "APPROVED"}


# ─────────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_reports():
    def j(p):
        p = REPORTS / p
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return {
        "val":   j("model_validation.json"),
        "fair":  j("fairness_report.json"),
        "sc":    j("scorecard.json"),
        "meta":  j("dashboard/model_metadata.json"),
        "dl":    j("dashboard/dl_metadata.json"),
        "status": j("validation_status.json"),
        "ch":    j("challengers.json"),
        "policy": json.loads((ROOT / "configs" / "credit_policy.json").read_text(encoding="utf-8")),
    }


@st.cache_data
def load_predictions():
    p = REPORTS / "dashboard" / "test_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def load_scorer():
    try:
        import pickle
        from src.config import MODELS_DIR
        from src.models.pd_model import PDScorer
        scorer = PDScorer()
        with open(MODELS_DIR / "shap_explainer.pkl", "rb") as f:
            explainer = pickle.load(f)
        return scorer, explainer
    except Exception:
        return None, None


lang = st.sidebar.radio("Idioma / Language", ["es", "en"], horizontal=True,
                        format_func=lambda x: "Español" if x == "es" else "English")
t = T[lang]
page_idx = st.sidebar.radio("Sección / Section", range(len(t["pages"])),
                            format_func=lambda i: t["pages"][i], label_visibility="collapsed")

R = load_reports()
P = load_predictions()
if R["val"] is None or P is None:
    st.error(t["missing"])
    st.stop()

val, fair, policy = R["val"], R["fair"], R["policy"]
disc, cal = val["discriminatory_power"], val["calibration"]
cutoff_ref = val["cutoff_policy"]["cutoff_pd"]

# ─────────────────────────────────────────────────────────────────────────────
# 0. Resumen
# ─────────────────────────────────────────────────────────────────────────────
if page_idx == 0:
    st.title(t["title"])
    st.caption(t["subtitle"])

    status = R["status"]
    if status:
        label = status["label"] if lang == "es" else STATUS_EN.get(status["label"], status["label"])
        n = status["n_findings"]
        msg = f"**{t['status_title']}: {label}** · {n['Alta']} / {n['Media']} / {n['Baja']} (Alta / Media / Baja)"
        {"pass": st.success, "warn": st.warning, "fail": st.error}[status["level"]](msg)

    c = st.columns(5)
    c[0].metric("Gini", f"{disc['gini']:.3f}", f"{disc['gini_lift_benchmark']:+.3f} vs benchmark")
    c[1].metric("AUC", f"{disc['auc_roc']:.3f}")
    c[2].metric("KS", f"{disc['ks_statistic']:.3f}")
    c[3].metric("ECE (PD calibrada)", f"{cal['calibrated_pd']['ece']:.3f}",
                f"{cal['raw_score']['ece']:.3f} sin calibrar", delta_color="off")
    c[4].metric(t["approval"], f"{(P['pd'] < cutoff_ref).mean():.1%}", f"cutoff PD {cutoff_ref:.1%}",
                delta_color="off")

    st.subheader(t["checks"])
    rows = [{t["check"]: k, t["value"]: str(v["value"]), t["threshold"]: v["threshold"],
             t["status"]: t["pass"] if v["pass"] else t["fail"]} for k, v in val["checks"].items()]
    rows.append({t["check"]: "fairness", t["value"]: "—", t["threshold"]: "DPD ≤ 0.10, DIR ≥ 0.80, EO gap ≤ 0.10",
                 t["status"]: t["pass"] if fair["overall_fairness_passed"] else t["fail"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader(t["findings"])
    st.markdown("- " + t["f_rank"].format(g=disc["gini"], lo=disc["gini_ci95"][0],
                                          hi=disc["gini_ci95"][1], lift=disc["gini_lift_benchmark"]))
    st.markdown("- " + t["f_cal"].format(raw=cal["raw_score"]["mean_pd"], dr=cal["raw_score"]["default_rate"],
                                         cal=cal["calibrated_pd"]["mean_pd"], ece=cal["calibrated_pd"]["ece"]))
    flags = [f"{a}: {'; '.join(r['regulatory_flags'])}" for a, r in fair["results"].items() if r["regulatory_flags"]]
    st.markdown("- " + (t["f_fair"].format(flags=" | ".join(flags)) if flags else t["f_fair_ok"]))
    weak = [f"{n} ({r['gini']:.3f})" for segs in val["segment_performance"].values()
            for n, r in segs.items() if r["gini"] < disc["gini"] - 0.05]
    if weak:
        st.markdown("- " + t["f_seg"].format(segs=", ".join(weak)))

    if status:
        st.subheader(t["findings_table"])
        icon = {"Alta": "🔴", "Media": "🟠", "Baja": "⚪"}
        st.dataframe(pd.DataFrame([{t["sev"]: f"{icon[f['severidad']]} {f['severidad']}", t["finding"]: f["titulo"],
                                    t["rec"]: f["recomendacion"], t["ref"]: f["referencia"]}
                                   for f in status["findings"]]), hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Discriminación y calibración
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 1:
    st.title(t["pages"][1])
    df = P.copy()
    df["decile"] = pd.qcut(df["pd"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    g = df.groupby("decile", observed=True).agg(mean_pd=("pd", "mean"), dr=("default", "mean"),
                                                defaults=("default", "sum"))
    g["capt"] = g["defaults"][::-1].cumsum()[::-1] / g["defaults"].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["dec_table"])
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g["dr"], marker_color=PURPLE,
                               text=[f"{v:.1%}" for v in g["dr"]], textposition="outside"))
        fig.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(t=20))
        st.plotly_chart(fig)
    with col2:
        st.subheader(t["reliability"])
        fig = go.Figure()
        lim = float(max(g["mean_pd"].max(), g["dr"].max())) * 1.1
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(dash="dash", color=GRAY),
                                 name="perfect"))
        fig.add_trace(go.Scatter(x=g["mean_pd"], y=g["dr"], mode="markers+lines", marker_color=PURPLE,
                                 name=t["pd"]))
        fig.update_layout(xaxis_title=t["mean_pd"], yaxis_title=t["dr"], height=380, margin=dict(t=20),
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig)

    tbl = g.reset_index()[["decile", "mean_pd", "dr", "capt"]]
    tbl.columns = [t["decile"], t["mean_pd"], t["dr"], t["share_def"]]
    st.dataframe(tbl.style.format({tbl.columns[1]: "{:.2%}", tbl.columns[2]: "{:.2%}", tbl.columns[3]: "{:.0%}"}),
                 hide_index=True)

    st.subheader(t["raw_vs_cal"])
    cmp_rows = []
    for k, lab in [("raw_score", "XGBoost raw"), ("calibrated_pd", "Platt")]:
        r = cal[k]
        cmp_rows.append({"": lab, t["mean_pd"]: f"{r['mean_pd']:.2%}", t["dr"]: f"{r['default_rate']:.2%}",
                         "ECE": r["ece"], "Brier": r["brier"], "HL p": r["hl_pvalue"]})
    st.dataframe(pd.DataFrame(cmp_rows), hide_index=True)

    st.subheader(t["ms"])
    msv = val["master_scale"]
    ms = pd.DataFrame(msv["table"])
    ms_pre = pd.DataFrame(msv["pre_moc"]["table"]) if msv.get("pre_moc") else None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ms["grade"], y=ms["share"], name="% portfolio", marker_color="#E8E6F7", yaxis="y2"))
    err = np.array(ms["dr_ci95"].tolist())
    fig.add_trace(go.Scatter(x=ms["grade"], y=ms["default_rate"], mode="markers", marker=dict(color=CORAL, size=10),
                             error_y=dict(type="data", symmetric=False, array=err[:, 1] - ms["default_rate"],
                                          arrayminus=ms["default_rate"] - err[:, 0]), name=t["dr"]))
    if ms_pre is not None and msv.get("moc"):
        fig.add_trace(go.Scatter(x=ms_pre["grade"], y=ms_pre["pd_avg"], mode="markers+lines",
                                 marker=dict(color=GRAY, symbol="triangle-up"), line=dict(dash="dot"),
                                 name=t["mean_pd"] + " (sin MoC)"))
    fig.add_trace(go.Scatter(x=ms["grade"], y=ms["pd_avg"], mode="markers+lines", marker=dict(color=PURPLE, symbol="square"),
                             line=dict(dash="dash"), name=t["pd_grade"] if msv.get("moc") else t["mean_pd"]))
    fig.update_layout(yaxis=dict(tickformat=".0%", title=t["dr"]),
                      yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False),
                      height=420, margin=dict(t=20), legend=dict(orientation="h"))
    st.plotly_chart(fig)
    st.dataframe(ms[["grade", "pd_range", "n", "share", "pd_avg", "default_rate", "p_jeffreys", "result"]],
                 hide_index=True)

    if msv.get("moc"):
        st.subheader(t["moc_title"])
        pre_sum = msv["pre_moc"]["summary"]
        st.markdown(t["moc_note"].format(red=", ".join(pre_sum["grades_underestimating"]) or "—",
                                         amber=", ".join(pre_sum.get("grades_amber", [])) or "—",
                                         el=msv.get("moc_el_impact", 0)))
        moc = pd.DataFrame(msv["moc"]).T.reset_index(names=t["grade"])
        moc = moc[[t["grade"], "pd_avg", "default_rate", "pd_grade", "moc_factor"]]
        moc.columns = [t["grade"], t["pd_val"], t["dr_val"], t["pd_grade"], t["factor"]]
        st.dataframe(moc.style.format({t["pd_val"]: "{:.2%}", t["dr_val"]: "{:.2%}", t["pd_grade"]: "{:.2%}",
                                       t["factor"]: "x{:.3f}"}), hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stress test
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 2:
    st.title(t["pages"][2])
    st.info(t["stress_note"])
    s = val["stress_testing"]
    rows = [{t["scenario"]: "baseline", "": "—", t["mean_pd"]: s["baseline"]["mean_pd"],
             "Δ PD": 0.0, "Δ EL": 0.0, t["el"]: s["baseline"]["el_rate"], "Δ " + t["approval"]: 0.0}]
    for name, sc in s["scenarios"].items():
        rows.append({t["scenario"]: name, "": sc["description"], t["mean_pd"]: sc["mean_pd"],
                     "Δ PD": sc["pd_change_rel"], "Δ EL": sc["el_change_rel"], t["el"]: sc["el_rate"],
                     "Δ " + t["approval"]: sc["approval_change"]})
    df = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=df[t["el"]], y=df[t["scenario"]], orientation="h",
                           marker_color=[GRAY] + [CORAL] * (len(df) - 1),
                           text=[f"{v:.2%}" for v in df[t["el"]]], textposition="outside"))
    fig.update_layout(xaxis_tickformat=".1%", yaxis_autorange="reversed", height=360, margin=dict(t=20),
                      xaxis_title=f"{t['el']} (LGD {s['lgd']:.0%})")
    st.plotly_chart(fig)
    st.dataframe(df.style.format({t["mean_pd"]: "{:.2%}", "Δ PD": "{:+.1%}", "Δ EL": "{:+.1%}",
                                  t["el"]: "{:.2%}", "Δ " + t["approval"]: "{:+.1%}"}),
                 hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fairness
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 3:
    st.title(t["pages"][3])
    st.markdown(f"**{t['fair_rule']}:** {fair['decision_rule']}")
    cut = st.slider(t["fair_slider"], 0.02, 0.60, float(cutoff_ref), 0.01, format="%.2f")
    cols = st.columns(2)
    for col, (attr, key) in zip(cols, [("gender", "gender"), ("age_group", "age_group")]):
        sub = P[P["gender"].isin(["M", "F"])] if key == "gender" else P
        grp = sub.assign(approved=sub["pd"] < cut).groupby(key)
        rates, drs = grp["approved"].mean(), grp["default"].mean()
        if key == "age_group":
            order = ["<25", "25-35", "35-45", "45-55", ">55"]
            rates, drs = rates.reindex(order), drs.reindex(order)
        dir_ = rates.min() / rates.max()
        tag = t["in_model"] if fair["attribute_in_model"][attr] else t["not_in_model"]
        with col:
            st.subheader(f"{attr} ({tag})")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=rates.index, y=rates, name=t["approval"], marker_color=PURPLE,
                                 text=[f"{v:.0%}" for v in rates], textposition="outside"))
            fig.add_trace(go.Bar(x=drs.index, y=drs, name=t["dr"], marker_color=CORAL))
            fig.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(t=20), legend=dict(orientation="h"))
            st.plotly_chart(fig)
            c1, c2 = st.columns(2)
            c1.metric("DIR (min/max)", f"{dir_:.3f}", "≥ 0.80" if dir_ >= 0.8 else "< 0.80",
                      delta_color="normal" if dir_ >= 0.8 else "inverse")
            c2.metric("DPD (max−min)", f"{rates.max() - rates.min():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Política de crédito
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 4:
    st.title(t["pages"][4])
    st.markdown(t["policy_intro"])
    markets = policy["mercados"]
    m = st.selectbox(t["market"], list(markets), index=list(markets).index(policy["mercado_referencia"]))
    c1, c2 = st.columns(2)
    r   = c1.slider(t["rate"], 0.02, 1.20, float(markets[m]["tasa_interes_anual"]), 0.005, format="%.3f")
    lgd = c2.slider(t["lgd"], 0.10, 0.90, float(markets[m]["lgd"]), 0.01)
    st.caption(f"{markets[m]['fuente_tasa']} · {markets[m]['fuente_lgd']}")

    pstar = r / (r + lgd)
    cuts = np.linspace(0.01, 0.80, 160)
    pd_arr, y = P["pd"].values, P["default"].values
    profit = np.array([((pd_arr < c) * ((1 - y) * r - y * lgd)).mean() for c in cuts])
    c_emp = cuts[profit.argmax()]

    k = st.columns(3)
    k[0].metric(t["breakeven"], f"{pstar:.1%}")
    k[1].metric(t["emp_opt"], f"{c_emp:.1%}")
    k[2].metric(t["approval_at"], f"{(pd_arr < pstar).mean():.1%}")

    fig = go.Figure(go.Scatter(x=cuts, y=profit, mode="lines", line=dict(color=PURPLE, width=3)))
    fig.add_vline(x=pstar, line_dash="dash", line_color=CORAL, annotation_text="PD*")
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.update_layout(xaxis_title=t["cutoff"], yaxis_title=t["profit"], xaxis_tickformat=".0%",
                      height=420, margin=dict(t=20))
    st.plotly_chart(fig)
    st.warning(t["policy_caveat"])

# ─────────────────────────────────────────────────────────────────────────────
# 5. Scorecard
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 5:
    st.title(t["pages"][5])
    sc = R["sc"]
    if sc is None:
        st.error(t["missing"])
        st.stop()
    st.markdown(t["sc_intro"].format(base=sc["escala"]["base_score"], odds=sc["escala"]["base_odds"],
                                     pdo=sc["escala"]["pdo"]))
    k = st.columns(3)
    k[0].metric(t["sc_gini"], f"{sc['gini_test']:.3f}")
    k[1].metric(t["sc_vars"], sc["n_variables"])
    k[2].metric(t["sc_gap"], f"{-sc['gap_vs_champion']:+.3f}")
    iv = pd.DataFrame(sc["iv"]).head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=iv["iv"], y=iv["variable"], orientation="h",
                           marker_color=[PURPLE if s else "#C9C7BF" for s in iv["seleccionada"]]))
    for x in (0.02, 0.10, 0.30):
        fig.add_vline(x=x, line_dash="dot", line_color=GRAY)
    fig.update_layout(xaxis_title="Information Value", height=520, margin=dict(t=20))
    st.plotly_chart(fig)
    pts_path = REPORTS / "scorecard_points.csv"
    if pts_path.exists():
        pts = pd.read_csv(pts_path)
        var = st.selectbox(t["variable"], sc["variables"])
        st.dataframe(pts[pts["variable"] == var], hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Challengers
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 6:
    st.title(t["pages"][6])
    ch = R["ch"]
    if ch is None:
        st.error(t["missing"])
        st.stop()
    st.markdown(t["ch_intro"])
    models = ch["models"]
    ok = [m for m in models[1:] if m["monotonic_trend_violation"] == 0
          and not m["grades_red_with_moc"] and m["delta_gini_ci95"][0] > 0]
    best = max(ok, key=lambda m: m["gini"]) if ok else None
    if best:
        k = st.columns(4)
        k[0].metric(t["ch_reco"], best["model"])
        k[1].metric("Gini", f"{best['gini']:.3f}", f"{best['delta_gini_vs_champion']:+.3f} vs champion")
        k[2].metric(t["mono"], f"{best['monotonic_trend_violation']:.0%}",
                    f"{models[0]['monotonic_trend_violation']:.0%} champion", delta_color="off")
        k[3].metric("ECE", f"{best['ece']:.4f}")
    else:
        st.info(t["ch_none"])
    st.caption(t["ch_rule"])

    names = [m["model"] for m in models]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=[m["gini"] for m in models], y=names, orientation="h",
            marker_color=[GRAY] + [PURPLE] * (len(models) - 1),
            error_x=dict(type="data", symmetric=False,
                         array=[m["gini_ci95"][1] - m["gini"] for m in models],
                         arrayminus=[m["gini"] - m["gini_ci95"][0] for m in models])))
        if ch.get("lstm_stack_reference"):
            fig.add_vline(x=ch["lstm_stack_reference"]["gini"], line_dash="dash", line_color=CORAL,
                          annotation_text=t["ch_lstm"], annotation_position="top left")
        fig.update_layout(title=t["ch_gini"],
                          xaxis_range=[min(m["gini_ci95"][0] for m in models) - 0.01,
                                       max(m["gini_ci95"][1] for m in models) + 0.01],
                          yaxis_autorange="reversed", height=380, margin=dict(t=50))
        st.plotly_chart(fig)
    with col2:
        mono = [m["monotonic_trend_violation"] for m in models]
        fig = go.Figure(go.Bar(x=mono, y=names, orientation="h",
                               marker_color=[CORAL if v > 0 else GREEN for v in mono],
                               text=[f"{v:.0%}" for v in mono], textposition="outside"))
        fig.update_layout(title=t["ch_mono"], xaxis_tickformat=".0%", xaxis_range=[0, max(mono + [0.1]) * 1.25],
                          yaxis_autorange="reversed", height=380, margin=dict(t=50))
        st.plotly_chart(fig)

    tbl = pd.DataFrame([{
        t["model"]: m["model"], "Gini": m["gini"],
        t["delta"]: (f"{m['delta_gini_vs_champion']:+.4f} [{m['delta_gini_ci95'][0]:+.4f}; {m['delta_gini_ci95'][1]:+.4f}]"
                     if i else "—"),
        "ECE": m["ece"], t["mono"]: f"{m['monotonic_trend_violation']:.1%}",
        t["red_pre"]: ", ".join(m["grades_red"]) or "—", t["red_post"]: ", ".join(m["grades_red_with_moc"]) or "—",
        t["approval"]: f"{m['approval_rate']:.1%}", "DIR edad / age": m["age_dir"],
        "DIR género / gender": m["gender_dir"],
    } for i, m in enumerate(models)])
    st.dataframe(tbl, hide_index=True)

    with_feats = [m for m in models if m.get("top_features_gain")]
    if with_feats:
        m = best if best and best.get("top_features_gain") else with_feats[-1]
        feats = pd.Series(m["top_features_gain"]).sort_values()
        fig = go.Figure(go.Bar(x=feats.values, y=feats.index, orientation="h",
                               marker_color=[CORAL if i.startswith("pay_") else PURPLE for i in feats.index]))
        fig.update_layout(title=t["ch_feats"].format(m=m["model"]), height=380, margin=dict(t=50),
                          xaxis_title="gain")
        st.plotly_chart(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Simulador
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 7:
    st.title(t["pages"][7])
    scorer, explainer = load_scorer()
    if scorer is None:
        st.info(t["sim_off"])
        st.stop()
    from src.validation.explain import top_factors
    from src.validation.master_scale import assign_grades
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        credit  = c1.number_input("AMT_CREDIT", 10_000, 4_000_000, 500_000, 10_000)
        annuity = c1.number_input("AMT_ANNUITY", 1_000, 300_000, 25_000, 500)
        income  = c1.number_input("AMT_INCOME_TOTAL", 10_000, 5_000_000, 150_000, 5_000)
        age     = c2.slider("Edad / Age", 20, 70, 38)
        years_e = c2.slider("Antigüedad laboral / Years employed", 0, 40, 5)
        inc_t   = c2.selectbox("NAME_INCOME_TYPE", ["Working", "Commercial associate", "Pensioner", "State servant"])
        e1 = c3.slider("EXT_SOURCE_1", 0.0, 1.0, 0.50)
        e2 = c3.slider("EXT_SOURCE_2", 0.0, 1.0, 0.55)
        e3 = c3.slider("EXT_SOURCE_3", 0.0, 1.0, 0.50)
        go_btn = st.form_submit_button(t["sim_btn"])
    if go_btn:
        app = pd.DataFrame([{
            "AMT_CREDIT": credit, "AMT_ANNUITY": annuity, "AMT_INCOME_TOTAL": income,
            "AMT_GOODS_PRICE": credit * 0.9, "DAYS_BIRTH": -age * 365, "DAYS_EMPLOYED": -years_e * 365,
            "NAME_INCOME_TYPE": inc_t, "EXT_SOURCE_1": e1, "EXT_SOURCE_2": e2, "EXT_SOURCE_3": e3,
        }])
        X = scorer.transform_raw(app)
        p = float(scorer.pd_from_features(X)[0])
        ms = policy["master_scale"]
        grade = assign_grades([p], ms["limites_superiores"], ms["grados"])[0]
        p_grade = float(scorer.grade_pd([p])[0])
        lgd_ref = policy["mercados"][policy["mercado_referencia"]]["lgd"]
        k = st.columns(5)
        k[0].metric(t["pd"], f"{p:.2%}")
        k[1].metric(t["grade"], grade)
        k[2].metric(t["pd_grade"], f"{p_grade:.2%}")
        k[3].metric(t["decision"], t["rejected"] if p >= cutoff_ref else t["approved"],
                    f"cutoff {cutoff_ref:.1%}", delta_color="off")
        k[4].metric(t["el_sim"], f"{p_grade * lgd_ref * credit:,.0f}", f"LGD {lgd_ref:.0%}", delta_color="off")
        f = top_factors(scorer.feature_names, explainer.shap_values(X).flatten(), k=8)
        fdf = pd.DataFrame(f)
        fig = go.Figure(go.Bar(x=fdf["shap_value"], y=fdf["feature"], orientation="h",
                               marker_color=[CORAL if v > 0 else GREEN for v in fdf["shap_value"]]))
        fig.update_layout(yaxis_autorange="reversed", height=380, margin=dict(t=30), title=t["factors"])
        st.plotly_chart(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Model Card
# ─────────────────────────────────────────────────────────────────────────────
elif page_idx == 8:
    card = REPORTS / "model_cards" / "model_card.html"
    if card.exists():
        st.iframe(card, height=2600)
    else:
        st.error(t["card_off"])
