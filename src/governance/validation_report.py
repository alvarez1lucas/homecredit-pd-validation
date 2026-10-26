"""
Informe de validación (PDF, 2 páginas)
======================================
Página 1: alcance, resultado, checks y hallazgos ordenados por severidad.
Página 2: evidencia gráfica y limitaciones.

Fuentes:
  - reports/model_validation.json, fairness_report.json, scorecard.json → hallazgos automáticos
  - docs/validation_findings.json → hallazgos documentados en las notebooks
"""
import json
import re
import textwrap
from datetime import date

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ROOT
from src.governance.status import manual_findings, validation_status

PURPLE, RED, AMBER, GREEN, GRAY = "#3C3489", "#A32D2D", "#854F0B", "#0F6E56", "#666666"
SEV_COLOR = {"Alta": RED, "Media": AMBER, "Baja": GRAY}
SEV_ORDER = {"Alta": 0, "Media": 1, "Baja": 2}
A4 = (8.27, 11.69)


def auto_findings(val: dict, fair: dict, sc: dict | None) -> list:
    out = []
    ms = val["master_scale"]
    tbl = {r["grade"]: r for r in ms["table"]}
    pre = ms.get("pre_moc")
    if pre and pre["summary"]["grades_underestimating"] and not ms["summary"]["grades_underestimating"]:
        pre_tbl = {r["grade"]: r for r in pre["table"]}
        detail = "; ".join(
            f"grado {g}: Jeffreys p {pre_tbl[g]['p_jeffreys']:.3f} → {tbl[g]['p_jeffreys']:.3f}"
            for g in pre["summary"]["grades_underestimating"] + pre["summary"].get("grades_amber", []))
        out.append({
            "id": "VAL-MOC", "severidad": "Baja",
            "titulo": "Remediación aplicada: PD de grado con margen de conservadurismo",
            "detalle": f"Con la PD individual el backtest fallaba en la cola ({detail}). La PD de grado "
                       "= máx(PD media, percentil 75 de Jeffreys del default observado en validación) "
                       f"corrige la subestimación; la pérdida esperada sube {ms.get('moc_el_impact', 0):+.1%}.",
            "recomendacion": "Re-estimar el MoC en cada recalibración y backtestearlo semestralmente; "
                             "la PD individual sigue usándose para ordenar y decidir.",
            "referencia": "models/champion/grade_pd.json; src/models/moc.py",
        })
    for g in ms["summary"]["grades_underestimating"]:
        r = tbl[g]
        out.append({
            "id": f"VAL-MS-{g}", "severidad": "Alta",
            "titulo": f"Subestimación de PD en el grado {g} (test de Jeffreys p={r['p_jeffreys']:.3f})",
            "detalle": f"PD asignada {r['pd_avg']:.1%} vs default observado {r['default_rate']:.1%} "
                       f"(IC95% {r['dr_ci95'][0]:.1%}–{r['dr_ci95'][1]:.1%}); {r['share']:.1%} de la cartera. "
                       "La calibración Platt es lineal en el logit y no captura la cola de alto riesgo.",
            "recomendacion": "La calibración isotónica tampoco lo corrige (notebook 07): aplicar un margen "
                             "de conservadurismo en los grados altos y re-testear por grado.",
            "referencia": "reports/model_validation.json → master_scale; notebook 07 §1",
        })
    for g in ms["summary"].get("grades_amber", []):
        r = tbl[g]
        out.append({
            "id": f"VAL-MS-{g}", "severidad": "Media",
            "titulo": f"Grado {g} en ámbar (Jeffreys p={r['p_jeffreys']:.3f})",
            "detalle": f"PD asignada {r['pd_avg']:.1%} vs default observado {r['default_rate']:.1%}.",
            "recomendacion": "Monitorear en el próximo backtest semestral.",
            "referencia": "reports/model_validation.json → master_scale",
        })
    for attr, res in fair["results"].items():
        if res["regulatory_flags"]:
            out.append({
                "id": f"FAIR-{attr}", "severidad": "Media",
                "titulo": f"Flags de fairness en {attr}",
                "detalle": "; ".join(res["regulatory_flags"]) +
                           f". DIR {res['disparate_impact_ratio']:.3f} (regla 4/5 cumplida)." if
                           res["disparate_impact_ratio"] >= 0.8 else "; ".join(res["regulatory_flags"]),
                "recomendacion": "Decisión de negocio/legal documentada sobre el uso de la edad; "
                                 "ver hallazgo NB03-1 sobre el canal de los scores de bureau.",
                "referencia": "reports/fairness_report.json",
            })
    g = val["discriminatory_power"]["gini"]
    weak = [(n, r["gini"]) for segs in val["segment_performance"].values()
            for n, r in segs.items() if r["gini"] < g - 0.05]
    if weak:
        out.append({
            "id": "VAL-SEG", "severidad": "Media",
            "titulo": "Discriminación menor en segmentos",
            "detalle": ", ".join(f"{n} (Gini {v:.3f})" for n, v in weak) + f" vs {g:.3f} global.",
            "recomendacion": "Monitorear Gini por segmento; revisión manual o modelo específico si se usa para pricing.",
            "referencia": "reports/model_validation.json → segment_performance; notebook 03 §3.4",
        })
    lift = val["discriminatory_power"]["gini_lift_benchmark"]
    if sc:
        out.append({
            "id": "VAL-BENCH", "severidad": "Baja",
            "titulo": "Ganancia moderada sobre los benchmarks",
            "detalle": f"Gini XGBoost {g:.3f} vs logística {val['discriminatory_power']['gini_benchmark']:.3f} "
                       f"(+{lift:.3f}) y scorecard WoE {sc['gini_test']:.3f} (+{sc['gap_vs_champion']:.3f}).",
            "recomendacion": "Si la interpretabilidad pesa más que ~3–4 puntos de Gini, el scorecard es una alternativa defendible.",
            "referencia": "reports/scorecard.json",
        })
    ch_path = REPORTS_DIR / "challengers.json"
    if ch_path.exists():
        ch = json.load(open(ch_path, encoding="utf-8"))["models"]
        ok = [m for m in ch[1:] if m["monotonic_trend_violation"] == 0
              and not m["grades_red_with_moc"] and m["delta_gini_ci95"][0] > 0]
        if ok:
            best = max(ok, key=lambda m: m["gini"])
            out.append({
                "id": "CH-1", "severidad": "Baja",
                "titulo": f"Challenger recomendado: {best['model']}",
                "detalle": f"Gini {best['gini']:.3f} (Δ {best['delta_gini_vs_champion']:+.3f} vs champion, IC95% pareado "
                           f"[{best['delta_gini_ci95'][0]:+.3f}; {best['delta_gini_ci95'][1]:+.3f}]), monótono en el monto, "
                           f"ECE {best['ece']:.3f} y backtest por grado sin rojos con MoC.",
                "recomendacion": "Promoverlo a candidato a champion en el próximo ciclo, con validación completa "
                                 "(fairness, estabilidad y análisis de errores).",
                "referencia": "reports/challengers.json; src/models/challengers.py",
            })
    return out


def _wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def es(text: str) -> str:
    """Coma decimal en los números (0.518 → 0,518) sin tocar el resto del texto."""
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


def page_one(pdf, val, fair, meta, findings):
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    y = 0.955
    ax.text(0.07, y, "Informe de validación — Modelo de PD de crédito al consumo",
            fontsize=15, weight="bold", color=PURPLE)
    y -= 0.025
    ax.text(0.07, y, f"Home Credit Default Risk · {date.today():%d/%m/%Y} · Marco: SR 26-2 "
                     "(estructura de validación de SR 11-7) · Proyecto de portfolio",
            fontsize=8.5, color=GRAY)
    y -= 0.04
    status = validation_status(val, fair, findings)
    ax.text(0.07, y, "Resultado:", fontsize=11, weight="bold")
    ax.text(0.20, y, status["label"], fontsize=11, weight="bold",
            color={"pass": GREEN, "warn": AMBER, "fail": RED}[status["level"]])
    y -= 0.022
    d = val["discriminatory_power"]; c = val["calibration"]["calibrated_pd"]
    n = lambda x: f"{x:.3f}".replace(".", ",")
    n_test = f"{meta['rows']['test']:,}".replace(",", ".")
    moc = val["master_scale"].get("pd_tested") == "PD de grado con MoC"
    summary = (f"Modelo XGBoost + calibración Platt{' + PD de grado con margen de conservadurismo' if moc else ''}, "
               f"{meta['feature_count']} features, test de {n_test} solicitudes no usadas para entrenar. "
               f"Gini {n(d['gini'])} (IC95% {n(d['gini_ci95'][0])}–{n(d['gini_ci95'][1])}), "
               f"KS {n(d['ks_statistic'])}, ECE {n(c['ece'])}. El modelo ordena bien el riesgo y está "
               "calibrado en promedio, pero "
               + ("subestima la PD en el tramo de mayor riesgo y " if not val["checks"]["master_scale"]["pass"] else "")
               + "presenta observaciones de fairness y de monotonía que condicionan su uso: "
               "ver hallazgos y challengers propuestos.")
    ax.text(0.07, y, _wrap(summary, 118), fontsize=8.5, va="top", linespacing=1.4)
    y -= 0.085

    ax.text(0.07, y, "Checks cuantitativos", fontsize=11, weight="bold", color=PURPLE)
    y -= 0.018
    rows = [(k, str(v["value"]), v["threshold"], "PASS" if v["pass"] else "FAIL")
            for k, v in val["checks"].items()]
    rows.append(("fairness", "ver hallazgos", "DPD ≤ 0,10 · DIR ≥ 0,80 · gap EO ≤ 0,10",
                 "PASS" if fair["overall_fairness_passed"] else "FAIL"))
    for col, x in zip(["Check", "Valor", "Umbral", "Estado"], [0.07, 0.25, 0.45, 0.85]):
        ax.text(x, y, col, fontsize=8, weight="bold")
    y -= 0.004
    ax.plot([0.07, 0.93], [y, y], color="#ccc", lw=0.6)
    for r in rows:
        y -= 0.016
        for val_, x in zip(r, [0.07, 0.25, 0.45, 0.85]):
            color = (GREEN if val_ == "PASS" else RED) if x == 0.85 else "black"
            ax.text(x, y, es(val_[:40]), fontsize=8, color=color, weight="bold" if x == 0.85 else "normal")
    y -= 0.035

    ax.text(0.07, y, "Hallazgos y recomendaciones", fontsize=11, weight="bold", color=PURPLE)
    y -= 0.02
    findings = sorted(findings, key=lambda f: SEV_ORDER[f["severidad"]])
    y, remaining = draw_findings(ax, findings, y, y_min=0.05)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)    # el separador (ax.plot) no debe re-escalar el lienzo
    pdf.savefig(fig); plt.close(fig)
    return remaining


def finding_height(f) -> float:
    body = _wrap(f["detalle"], 112) + "\n" + _wrap(f["recomendacion"], 110)
    return 0.012 + 0.0115 * (body.count("\n") + 1) + 0.004 + 0.016


def draw_findings(ax, findings, y, y_min):
    """Dibuja hallazgos mientras entren en la página; devuelve los que quedaron afuera."""
    for i, f in enumerate(findings):
        if y - finding_height(f) < y_min:
            return y, findings[i:]
        ax.text(0.07, y, f["severidad"].upper(), fontsize=7.5, weight="bold", color=SEV_COLOR[f["severidad"]])
        ax.text(0.15, y, es(f"{f['titulo']}  [{f['id']}]"), fontsize=8.3, weight="bold")
        y -= 0.012
        body = es(_wrap(f["detalle"], 112) + "\n→ " + _wrap(f["recomendacion"], 110))
        ax.text(0.15, y, body, fontsize=7.4, va="top", linespacing=1.3)
        y -= 0.0115 * (body.count("\n") + 1) + 0.004
        ax.text(0.15, y, f["referencia"], fontsize=6.6, color=GRAY, style="italic")
        y -= 0.016
    return y, []


def page_two(pdf, val, remaining):
    fig = plt.figure(figsize=A4)
    top = 0.955
    if remaining:
        ax0 = fig.add_axes([0, 0, 1, 1]); ax0.axis("off")
        ax0.text(0.07, top, "Hallazgos (continuación)", fontsize=11, weight="bold", color=PURPLE)
        top, _ = draw_findings(ax0, remaining, top - 0.02, y_min=0.0)
        ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
        top -= 0.01
    fig.text(0.07, top, "Evidencia", fontsize=13, weight="bold", color=PURPLE)
    imgs = [("master_scale.png", "Backtest por grado (master scale)"),
            ("calibration.png", "Calibración: score crudo vs PD calibrada"),
            ("stress_test.png", "Stress test: PD y pérdida esperada"),
            ("fairness_dashboard.png", "Fairness con la regla de decisión")]
    # grilla 2x2 entre el título "Evidencia" y el bloque de limitaciones (y = 0.30)
    h = (top - 0.04 - 0.30) / 2
    pos = [[0.05, 0.30 + h, 0.43, h - 0.02], [0.52, 0.30 + h, 0.43, h - 0.02],
           [0.05, 0.30, 0.43, h - 0.02], [0.52, 0.30, 0.43, h - 0.02]]
    for (fn, title), p in zip(imgs, pos):
        path = FIGURES_DIR / fn
        if not path.exists():
            continue
        ax = fig.add_axes(p); ax.imshow(mpimg.imread(path)); ax.axis("off")
        ax.set_title(title, fontsize=9, weight="bold")

    s = val["stress_testing"]["scenarios"]
    worst = max(s.items(), key=lambda kv: kv[1]["el_change_rel"])
    lims = [
        f"Stress test: el escenario más severo ({worst[0]}) sube la pérdida esperada "
        f"{worst[1]['el_change_rel']:+.0%}; el modelo es poco sensible a shocks de ingreso "
        f"({s['income_shock_moderate']['el_change_rel']:+.1%} con ingreso −25%) y muy sensible a los scores de bureau.",
        "Sin dimensión temporal: no hay validación out-of-time; el PSI train/test sólo mide homogeneidad del split.",
        "El TARGET no equivale a una definición regulatoria de default ni tiene horizonte de 12 meses (no apto para IFRS 9 sin ajustes).",
        "Validación hecha por el mismo autor del modelo: no hay validación independiente (effective challenge).",
        "Sólo application_train: bureau y previous_application quedan como próximo paso.",
    ]
    ax = fig.add_axes([0, 0, 1, 0.3]); ax.axis("off")
    ax.text(0.07, 0.88, "Limitaciones", fontsize=11, weight="bold", color=PURPLE)
    yy = 0.80
    for l in lims:
        t = es(_wrap("• " + l, 115))
        ax.text(0.07, yy, t, fontsize=8, va="top", linespacing=1.35)
        yy -= 0.055 * (t.count("\n") + 1) + 0.02
    pdf.savefig(fig); plt.close(fig)


def generate_validation_report():
    with open(REPORTS_DIR / "model_validation.json", encoding="utf-8") as f: val = json.load(f)
    with open(REPORTS_DIR / "fairness_report.json") as f: fair = json.load(f)
    with open(MODELS_DIR / "model_metadata.json") as f: meta = json.load(f)
    sc_path = REPORTS_DIR / "scorecard.json"
    sc = json.load(open(sc_path, encoding="utf-8")) if sc_path.exists() else None

    findings = auto_findings(val, fair, sc)
    findings += manual_findings()

    # Estado + hallazgos en JSON, para el dashboard
    status = validation_status(val, fair, findings)
    with open(REPORTS_DIR / "validation_status.json", "w", encoding="utf-8") as f:
        json.dump(status | {"findings": sorted(findings, key=lambda x: SEV_ORDER[x["severidad"]])},
                  f, indent=2, ensure_ascii=False)

    out = REPORTS_DIR / "validation_report.pdf"
    with PdfPages(out) as pdf:
        remaining = page_one(pdf, val, fair, meta, findings)
        page_two(pdf, val, remaining)
    logger.success(f"Informe de validación: {out} ({len(findings)} hallazgos)")


if __name__ == "__main__":
    generate_validation_report()
