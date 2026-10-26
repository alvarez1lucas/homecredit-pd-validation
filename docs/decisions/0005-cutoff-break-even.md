# 0005 — Cutoff de decisión por break-even de PD

**Contexto.** La versión anterior usaba un corte fijo de 0,50 sobre un score sin calibrar.
Además buscaba el "óptimo" en una grilla que terminaba en 0,725, y el óptimo caía justo en
el borde.

**Decisión.** Con PD calibrada, un préstamo a un período tiene ganancia esperada
`(1 − PD) · r − PD · LGD`. Esa ganancia se anula en

    PD* = r / (r + LGD)

Se rechaza si `PD ≥ PD*`. Los parámetros por mercado están en `configs/credit_policy.json`.
El mercado de referencia para el pipeline es USA (r = 11,5%, LGD = 45% → PD* ≈ 20%).

**Limitaciones del supuesto.**
- No descuenta costo de fondeo, costos operativos ni inflación (relevante para la TEA nominal de Argentina).
- Considera un solo período.
- El cutoff real sería más bajo.
