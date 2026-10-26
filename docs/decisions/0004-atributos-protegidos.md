# 0004 — Género fuera del modelo, edad dentro y auditada

**Contexto.** El EDA muestra diferencias reales de default por género (M 10,1% vs F 7,0%)
y por edad (12,3% en <25 vs 5,2% en >55).

**Decisión.**
- `CODE_GENDER` no es feature: evita trato diferencial directo. Se conserva sólo para auditar fairness.
- La edad entra una sola vez, como `age_years` (antes estaba duplicada con `DAYS_BIRTH`).
  Tiene poder predictivo con sustento de negocio, y su impacto se mide en `fairness_report.json`.

**Consecuencias / pendiente.** El análisis de fairness por edad da flags (DIR < 0,80).
Mantener la edad es una decisión de negocio y legal: ECOA/Reg B permite usarla en sistemas
"empirically derived" siempre que no perjudique a los mayores de 62. Las alternativas a
evaluar son:
- sacar la edad y medir la pérdida de Gini;
- usar cutoffs diferenciados;
- mandar a revisión manual a los segmentos afectados.
