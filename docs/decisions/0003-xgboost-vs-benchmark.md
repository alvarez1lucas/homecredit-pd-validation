# 0003 — XGBoost como champion, logística y scorecard como benchmarks

**Contexto.** Un modelo complejo tiene que justificar su complejidad frente a alternativas
más simples y más interpretables.

**Decisión.** Hay tres benchmarks:
- regresión logística sobre las mismas features;
- un scorecard WoE/IV con puntos (PDO 20), el estándar de la industria;
- en la notebook 06, MLP, LSTM y ensemble.

XGBoost queda como champion si supera al benchmark en el set de validación.

**Consecuencias.** La mejora de XGBoost sobre los benchmarks se reporta explícitamente; ver
`reports/model_validation.json` y `reports/scorecard.json`. Si la ganancia fuese marginal,
el scorecard sería una alternativa razonable por interpretabilidad.
