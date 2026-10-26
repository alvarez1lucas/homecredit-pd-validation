# 0007 — PD de grado con margen de conservadurismo (MoC)

**Contexto.** El backtest por grado (ADR 0006) mostró que la PD individual calibrada
subestima el tramo de mayor riesgo:
- grado G (PD ≥ 27%): 34,6% asignado vs 37,0% observado, Jeffreys p = 0,0098 (rojo);
- grado F: ámbar.

La calibración isotónica no lo corrige (notebook 07). Todos los challengers tienen el mismo
problema, así que es una característica de la cola con estos datos, no de un modelo puntual.

**Decisión.** Se separan dos usos de la PD, como en los esquemas de master scale IRB:

| Uso | PD |
|---|---|
| Ordenar y decidir (cutoff) | PD individual calibrada |
| Pérdida esperada y provisiones | **PD de grado** = máx(PD media del grado, percentil 75 de Jeffreys del default observado) |

La PD de grado se fuerza a ser no decreciente entre grados.

El MoC se estima en el **split de validación**:
- no en el de calibración, porque Platt se ajustó ahí y se ve bien calibrado por construcción;
- no en el test, que queda reservado para el backtest.

**Consecuencias.**
- El backtest por grado pasa en todos los grados (G: p 0,0098 → 0,68).
- La pérdida esperada sube ~5% (ver `model_validation.json → master_scale.moc_el_impact`).
- El Gini no cambia, porque la decisión sigue usando la PD individual.
- El MoC hay que re-estimarlo en cada recalibración.

Implementación: `src/models/moc.py`, `models/champion/grade_pd.json`.
