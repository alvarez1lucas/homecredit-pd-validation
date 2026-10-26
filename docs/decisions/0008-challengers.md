# 0008 — Challengers monotónico y con historial de pagos

**Contexto.** La validación dejó dos observaciones abiertas:
1. **La PD no es monótona respecto del monto:** en ~35% de los clientes, pedir más plata
   baja la PD (notebook 04).
2. **El historial de pagos anteriores aporta información:** el stack XGBoost + LSTM gana
   +0,013 de Gini (notebook 06), pero un LSTM es caro de mantener y validar.

**Decisión.** Se entrenan tres challengers, con el mismo esquema de datos que el champion:

| | Cambio |
|---|---|
| M1 | `monotone_constraints` (+1) en AMT_CREDIT, AMT_ANNUITY y los ratios crédito/ingreso y cuota/ingreso. AMT_GOODS_PRICE sale en nivel y queda `credit_goods_ratio`, así escalar el pedido completo nunca baja la PD. |
| M2 | 11 variables agregadas de `installments_payments` (atrasos, impagos, pagos parciales, últimos 12 meses). |
| M3 | M1 + M2. |

**Criterios de recomendación:**
- Δ Gini significativo (IC 95% de bootstrap pareado > 0);
- 0% de violación de monotonía;
- sin grados en rojo con MoC.

**Resultado** (`reports/challengers.json`):

| Modelo | Δ Gini (IC 95%) | Monótono |
|---|---|:---:|
| M1 | −0,009 | sí |
| M2 | +0,016 | no |
| **M3** | **+0,011 [+0,005; +0,016]** | **sí** |

- **M3 es el challenger recomendado.**
- Las variables agregadas igualan o superan al stack con LSTM, con mucha menos complejidad.
- La monotonía cuesta ~0,5 puntos de Gini.

**Pendiente antes de promoverlo:** validación completa de M3 (fairness, análisis de errores,
estabilidad), integrar las variables de pagos a la API y monitorear el segmento sin
historial (~5%).
