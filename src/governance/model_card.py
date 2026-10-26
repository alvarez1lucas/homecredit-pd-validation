"""
Model Card Generator — Home Credit Default Risk
===============================================
Genera el Model Card (HTML + JSON) a partir de los reportes del pipeline.
Ningún resultado está escrito a mano: todo sale de reports/*.json.
"""
import json
from datetime import datetime

from jinja2 import Environment
from loguru import logger

from src.config import MODELS_DIR, REPORTS_DIR
from src.governance.status import load_status

TEMPLATE = """<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><title>Model Card — PD Home Credit</title>
<style>
body{font-family:Arial,sans-serif;font-size:13px;color:#222;max-width:960px;margin:0 auto;padding:40px}
h1{font-size:22px;border-bottom:3px solid #534AB7;padding-bottom:8px;color:#26215C}
h2{font-size:16px;color:#534AB7;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:14px;color:#3C3489;margin-top:16px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:bold;margin:2px}
.pass{background:#E1F5EE;color:#085041}.fail{background:#FCEBEB;color:#791F1F}.warn{background:#FAEEDA;color:#633806}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.metric{background:#f8f7f5;border:1px solid #ddd;border-radius:8px;padding:12px;text-align:center}
.metric .val{font-size:22px;font-weight:bold;color:#534AB7}
.metric .lbl{font-size:11px;color:#666;margin-top:4px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:12px}
th{background:#EEEDFE;color:#26215C;padding:7px;text-align:left}
td{padding:6px 7px;border-bottom:1px solid #eee}
.ok{color:#0F6E56}.flag{color:#A32D2D;font-weight:bold}.warnc{color:#854F0B;font-weight:bold}.note{color:#666;font-size:11px}
.footer{margin-top:40px;padding-top:16px;border-top:1px solid #ddd;font-size:11px;color:#888}
</style></head><body>
<h1>Model Card — Modelo de PD, crédito al consumo (Home Credit)</h1>
<p><strong>Generado:</strong> {{ generated_at }} &nbsp;|&nbsp; <strong>Modelo:</strong> {{ meta.champion }}
 + calibración Platt &nbsp;|&nbsp; <strong>Marco:</strong> SR 26-2 (estructura de validación de SR 11-7)</p>
<span class="badge {{ status.level }}">Validación: {{ status.label }}</span>
<span class="badge {{ 'pass' if fair.overall_fairness_passed else 'fail' }}">Fairness: {{ 'PASS' if fair.overall_fairness_passed else 'FAIL — ver sección 6' }}</span>
<span class="badge warn">Proyecto de portfolio — no es un modelo en producción</span>

<h2>1. Detalles del modelo</h2>
<table>
<tr><th>Campo</th><th>Valor</th></tr>
<tr><td>Tarea</td><td>Probabilidad de default (PD) de solicitudes de crédito al consumo</td></tr>
<tr><td>Algoritmo</td><td>XGBoost (best iteration {{ cm.best_iteration }}) + Platt scaling sobre el logit del score</td></tr>
<tr><td>Benchmark</td><td>Regresión logística (Gini test {{ disc.gini_benchmark }})</td></tr>
<tr><td>Datos</td><td>Home Credit Default Risk (Kaggle) — application_train.csv</td></tr>
<tr><td>Filas</td><td>fit {{ fmt(meta.rows.fit) }} · validación {{ fmt(meta.rows.validation) }} · calibración {{ fmt(meta.rows.calibration) }} · test {{ fmt(meta.rows.test) }}</td></tr>
<tr><td>Features</td><td>{{ meta.feature_count }} (tras imputación con indicadores de missing y one-hot)</td></tr>
<tr><td>Atributos protegidos</td><td>CODE_GENDER: <strong>no</strong> es feature (sólo auditoría) · Edad: <strong>sí</strong> es feature (auditada)</td></tr>
<tr><td>Regla de decisión</td><td>{{ fair.decision_rule }}</td></tr>
</table>

<h2>2. Uso previsto</h2>
<ul>
<li><strong>Uso:</strong> estimar PD para originación de crédito al consumo y ordenar solicitudes por riesgo.</li>
<li><strong>Fuera de alcance:</strong> crédito corporativo, hipotecario, PD lifetime para IFRS 9 (el target no tiene horizonte definido).</li>
<li><strong>Supervisión humana:</strong> los segmentos listados en la sección 7 deberían ir a revisión manual en lugar de rechazo automático.</li>
</ul>

<h2>3. Poder discriminatorio (test, n={{ fmt(meta.rows.test) }})</h2>
<div class="metric-grid">
<div class="metric"><div class="val">{{ disc.gini }}</div><div class="lbl">Gini (IC95% {{ disc.gini_ci95[0] }}–{{ disc.gini_ci95[1] }})</div></div>
<div class="metric"><div class="val">{{ disc.auc_roc }}</div><div class="lbl">AUC-ROC</div></div>
<div class="metric"><div class="val">{{ disc.ks_statistic }}</div><div class="lbl">KS</div></div>
<div class="metric"><div class="val">{{ '%+.3f'|format(disc.gini_lift_benchmark) }}</div><div class="lbl">Lift de Gini vs benchmark</div></div>
</div>
{% if sc %}<p>Scorecard WoE/IV ({{ sc.n_variables }} variables, puntos con PDO {{ sc.escala.pdo }}): Gini test {{ sc.gini_test }}
 — {{ '%.3f'|format(sc.gap_vs_champion) }} por debajo de XGBoost.</p>{% endif %}

<h2>4. Calibración</h2>
<table>
<tr><th></th><th>PD media</th><th>Default real</th><th>ECE</th><th>Brier</th><th>Hosmer-Lemeshow p</th></tr>
<tr><td>Score crudo XGBoost</td><td>{{ pct(cal.raw_score.mean_pd) }}</td><td>{{ pct(cal.raw_score.default_rate) }}</td>
<td class="flag">{{ cal.raw_score.ece }}</td><td>{{ cal.raw_score.brier }}</td><td>{{ cal.raw_score.hl_pvalue }}</td></tr>
<tr><td>PD calibrada</td><td>{{ pct(cal.calibrated_pd.mean_pd) }}</td><td>{{ pct(cal.calibrated_pd.default_rate) }}</td>
<td class="{{ 'ok' if checks.ece.pass else 'flag' }}">{{ cal.calibrated_pd.ece }}</td><td>{{ cal.calibrated_pd.brier }}</td><td>{{ cal.calibrated_pd.hl_pvalue }}</td></tr>
</table>
<p class="note">Con 61k observaciones el test de Hosmer-Lemeshow rechaza ante desvíos mínimos; el criterio de aprobación usa ECE, calibration-in-the-large y el test de Jeffreys por grado.</p>

<h3>Master scale (test) — PD testeada: {{ ms.pd_tested }}</h3>
{% if ms.moc %}<p class="note">La PD individual ordena y decide; para pérdida esperada y provisiones cada grado usa una
<strong>PD de grado</strong> = máx(PD media, percentil 75 de Jeffreys del default observado en el set de validación).
Sin el MoC, el backtest con la PD individual daba: rojo {{ ms.pre_moc.summary.grades_underestimating | join(', ') or 'ninguno' }},
ámbar {{ ms.pre_moc.summary.grades_amber | join(', ') or 'ninguno' }}. Impacto del MoC en la pérdida esperada: {{ '%+.1f%%'|format(100*ms.moc_el_impact) }}.</p>
<table>
<tr><th>Grado</th><th>PD media (validación)</th><th>Default observado (validación)</th><th>PD de grado</th><th>Factor MoC</th></tr>
{% for g, r in ms.moc.items() %}<tr><td>{{ g }}</td><td>{{ pct(r.pd_avg) }}</td><td>{{ pct(r.default_rate) }}</td>
<td><strong>{{ pct(r.pd_grade) }}</strong></td><td>x{{ r.moc_factor }}</td></tr>{% endfor %}
</table>{% endif %}
<table>
<tr><th>Grado</th><th>Rango PD</th><th>% cartera</th><th>PD asignada</th><th>Default real</th><th>IC 95%</th><th>p Jeffreys</th><th>Resultado</th></tr>
{% for r in ms.table %}<tr><td>{{ r.grade }}</td><td>{{ r.pd_range }}</td><td>{{ pct(r.share) }}</td><td>{{ pct(r.pd_avg) }}</td>
<td>{{ pct(r.default_rate) }}</td><td>{{ pct(r.dr_ci95[0]) }}–{{ pct(r.dr_ci95[1]) }}</td><td>{{ r.p_jeffreys }}</td>
<td class="{{ 'flag' if r.result == 'subestima' else ('warnc' if r.result == 'alerta' else 'ok') }}">{{ r.result }}</td></tr>{% endfor %}
</table>
<p class="note">Semáforo: subestima = p &lt; 1% (falla) · alerta = p &lt; 5% (monitorear). Con 7 grados al 5%, una falsa alarma tiene ~30% de probabilidad.</p>

<h2>5. Stress test (variables crudas → PD → pérdida esperada)</h2>
<p class="note">LGD {{ pct(stress.lgd) }}, EAD = monto del crédito. Mide la sensibilidad del modelo a shocks; no es un modelo macroeconómico.</p>
<table>
<tr><th>Escenario</th><th>Descripción</th><th>PD media</th><th>Δ PD</th><th>Δ pérdida esperada</th><th>Δ aprobación</th></tr>
<tr><td>baseline</td><td>Condiciones actuales</td><td>{{ pct(stress.baseline.mean_pd) }}</td><td>—</td><td>—</td><td>—</td></tr>
{% for name, s in stress.scenarios.items() %}<tr><td>{{ name }}</td><td>{{ s.description }}</td><td>{{ pct(s.mean_pd) }}</td>
<td>{{ '%+.1f%%'|format(100*s.pd_change_rel) }}</td><td>{{ '%+.1f%%'|format(100*s.el_change_rel) }}</td>
<td>{{ '%+.1f pp'|format(100*s.approval_change) }}</td></tr>{% endfor %}
</table>

<h2>6. Fairness (regla de decisión: {{ fair.decision_rule }})</h2>
{% for attr, res in fair.results.items() %}
<h3>{{ attr }} {{ '(feature del modelo)' if fair.attribute_in_model[attr] else '(no es feature del modelo)' }}</h3>
<table>
<tr><th>Grupo</th><th>Aprobación</th><th>Default real</th><th>PD media</th><th>AUC</th></tr>
{% for g, r in res.approval_rates.items() %}<tr><td>{{ g }}</td><td>{{ pct(r) }}</td><td>{{ pct(res.default_rates[g]) }}</td>
<td>{{ pct(res.mean_pd[g]) }}</td><td>{{ res.auc_by_group[g] }}</td></tr>{% endfor %}
</table>
<p>DPD {{ res.demographic_parity_difference }} (máx {{ fair.thresholds.demographic_parity_max }}) ·
DIR {{ res.disparate_impact_ratio }} (mín {{ fair.thresholds.disparate_impact_min }}) ·
gap TPR {{ res.equalized_odds.tpr_gap }} · gap FPR {{ res.equalized_odds.fpr_gap }}</p>
{% if res.regulatory_flags %}<p class="flag">Flags: {{ res.regulatory_flags | join('; ') }}</p>{% endif %}
{% endfor %}

<h2>7. Limitaciones conocidas</h2>
<ul>
{% for l in limitations %}<li>{{ l }}</li>{% endfor %}
</ul>

<h2>8. Cobertura de referencias regulatorias</h2>
<table>
<tr><th>Referencia</th><th>Elemento</th><th>Estado en este proyecto</th></tr>
<tr><td>SR 26-2 / SR 11-7</td><td>Solidez conceptual, análisis de resultados, monitoreo</td><td>Cubierto (secciones 3–5, drift baseline)</td></tr>
<tr><td>SR 26-2 / SR 11-7</td><td>Validación independiente (effective challenge)</td><td class="flag">No cubierto — desarrollo y validación hechos por la misma persona</td></tr>
<tr><td>ECOA / Reg B</td><td>Análisis de impacto diferencial</td><td>Cubierto (sección 6)</td></tr>
<tr><td>EU AI Act (Anexo III: credit scoring = alto riesgo)</td><td>Transparencia / explicación por decisión</td><td>Parcial — SHAP por predicción en la API</td></tr>
<tr><td>IFRS 9</td><td>PD point-in-time, 12 meses / lifetime</td><td class="flag">No cubierto — el target no tiene horizonte temporal definido</td></tr>
</table>

{% if ch %}
<h2>9. Challengers</h2>
<p class="note">Mismo esquema de datos que el champion. Monotonía = % de clientes cuya PD baja al pedir más monto
(escalando monto, cuota y valor del bien de 0,5x a 2x). Backtest con PD de grado y MoC estimado en validación.</p>
<table>
<tr><th>Modelo</th><th>Gini test</th><th>Δ vs champion (IC95% pareado)</th><th>ECE</th><th>Monotonía</th><th>Rojos sin / con MoC</th><th>DIR edad</th></tr>
{% for m in ch.models %}<tr><td>{{ m.model }}</td><td>{{ m.gini }}</td>
<td>{{ '%+.3f'|format(m.delta_gini_vs_champion) }}{% if loop.index > 1 %} [{{ '%+.3f'|format(m.delta_gini_ci95[0]) }}; {{ '%+.3f'|format(m.delta_gini_ci95[1]) }}]{% endif %}</td>
<td>{{ m.ece }}</td><td class="{{ 'ok' if m.monotonic_trend_violation == 0 else 'flag' }}">{{ pct(m.monotonic_trend_violation) }}</td>
<td>{{ m.grades_red | join(', ') or '—' }} / {{ m.grades_red_with_moc | join(', ') or '—' }}</td><td>{{ m.age_dir }}</td></tr>{% endfor %}
</table>
{% endif %}

<div class="footer"><p>Generado automáticamente por homecredit-model-validation a partir de reports/*.json — {{ generated_at }}</p></div>
</body></html>"""


def build_limitations(val: dict, fair: dict) -> list:
    g = val["discriminatory_power"]["gini"]
    lims = []
    weak = []
    for seg_type, segs in val["segment_performance"].items():
        for name, r in segs.items():
            if r["gini"] < g - 0.05:
                weak.append(f"{name} (Gini {r['gini']:.3f})")
    tbl = {r["grade"]: r for r in val["master_scale"]["table"]}
    for gr in val["master_scale"]["summary"]["grades_underestimating"]:
        r = tbl[gr]
        lims.append(f"Subestimación de PD en el grado {gr} (PD {r['pd_avg']:.1%} vs default observado "
                    f"{r['default_rate']:.1%}, Jeffreys p={r['p_jeffreys']:.3f}): la calibración no "
                    "captura la cola de alto riesgo. Requiere recalibración o margen de conservadurismo.")
    if weak:
        lims.append(f"Discriminación más baja en: {', '.join(weak)}, contra {g:.3f} global.")
    for attr, res in fair["results"].items():
        if res["regulatory_flags"]:
            lims.append(f"Fairness — {attr}: {'; '.join(res['regulatory_flags'])}.")
    lims += [
        "Sin dimensión temporal: application_train no trae fecha de originación, así que no hay "
        "validación out-of-time. El PSI train/test sólo mide la homogeneidad del split aleatorio.",
        "El TARGET marca dificultades de pago; no equivale a una definición regulatoria de default "
        "(p. ej. 90+ días de mora) ni tiene un horizonte de 12 meses.",
        "La calibración se ajustó sobre un 10% del train; debería re-estimarse con datos recientes antes de usarse.",
        "Sólo usa application_train: bureau y previous_application quedan sin explotar.",
    ]
    return lims


def generate_model_card():
    logger.info("Generando Model Card...")

    with open(MODELS_DIR / "model_metadata.json") as f: meta = json.load(f)
    with open(REPORTS_DIR / "model_validation.json", encoding="utf-8") as f: val = json.load(f)
    with open(REPORTS_DIR / "fairness_report.json") as f: fair = json.load(f)
    sc_path = REPORTS_DIR / "scorecard.json"
    sc = json.load(open(sc_path, encoding="utf-8")) if sc_path.exists() else None
    ch_path = REPORTS_DIR / "challengers.json"
    ch = json.load(open(ch_path, encoding="utf-8")) if ch_path.exists() else None

    env = Environment()
    html = env.from_string(TEMPLATE).render(
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M"),
        meta = meta, cm = meta["champion_metrics"], val = val, fair = fair, sc = sc,
        disc = val["discriminatory_power"], cal = val["calibration"], checks = val["checks"],
        ms = val["master_scale"], stress = val["stress_testing"],
        limitations = build_limitations(val, fair),
        status = load_status(), ch = ch,
        fmt = lambda n: f"{n:,}".replace(",", "."),
        pct = lambda x: f"{100 * x:.1f}%",
    )

    out = REPORTS_DIR / "model_cards" / "model_card.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    json_card = {
        "schema_version":  "2.0",
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
        "model":           meta["champion"],
        "validation_pass": val["overall_pass"],
        "validation_status": load_status()["label"],
        "fairness_pass":   fair["overall_fairness_passed"],
        "gini":            val["discriminatory_power"]["gini"],
        "auc":             val["discriminatory_power"]["auc_roc"],
        "ece_calibrated":  val["calibration"]["calibrated_pd"]["ece"],
        "limitations":     build_limitations(val, fair),
    }
    with open(REPORTS_DIR / "model_cards" / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(json_card, f, indent=2, ensure_ascii=False)

    logger.success(f"Model Card generado: {out}")


if __name__ == "__main__":
    generate_model_card()
