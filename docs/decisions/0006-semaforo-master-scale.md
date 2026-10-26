# 0006 — Semáforo en el backtesting de la master scale

**Contexto.** Cada grado de rating se testea con el test de Jeffreys: ¿el default
observado es significativamente mayor que la PD asignada? Con 7 grados testeados al 5%,
la probabilidad de al menos una falsa alarma ronda el 30% aunque el modelo esté bien
calibrado. Lo verificamos por simulación en `tests/test_pipeline.py`.

**Decisión.** Semáforo, al estilo del backtesting de Basilea:

| Color | Resultado | Condición | Efecto |
|---|---|---|---|
| Rojo | `subestima` | p < 1% | falla la validación |
| Ámbar | `alerta` | 1% ≤ p < 5% | se informa y se monitorea |

**Consecuencias.** Menos falsos positivos, sin dejar de detectar subestimaciones claras.
