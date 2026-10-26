# 0001 — Separar fit / validación / calibración dentro del train

**Contexto.** La primera versión usaba el test set como `eval_set` del early stopping de
XGBoost y elegía el champion por su Gini en test. El test participaba en decisiones de
entrenamiento, así que las métricas reportadas eran optimistas.

**Decisión.** Split estratificado 80/20 train/test. Dentro del train:

| Subconjunto | % del train | Uso |
|---|---|---|
| fit | 80% | entrenar los modelos |
| validación | 10% | early stopping y selección del champion |
| calibración | 10% | ajustar Platt scaling |

El test se usa sólo para la evaluación final.

**Consecuencias.** Las métricas de test son out-of-sample de verdad. El costo es entrenar
con ~20% menos de filas.

**Limitación que no resuelve.** Home Credit no trae fecha de originación, así que no hay
validación out-of-time.
