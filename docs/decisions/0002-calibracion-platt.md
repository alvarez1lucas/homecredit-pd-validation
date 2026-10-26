# 0002 — Calibrar el score con Platt scaling

**Contexto.** `scale_pos_weight` compensa el desbalance (1:11) pero infla la escala del
score: el score medio queda cerca de 0,39 contra un default real de 0,08. El score ordena
bien, pero no es una PD. Sin calibrar no se puede usar para pérdida esperada, pricing ni
un cutoff económico.

**Alternativas.**
- Entrenar sin pesos: escala correcta de entrada, a costo de un aprendizaje más pobre de la clase minoritaria.
- Isotonic regression: más flexible, pero escalonada y propensa a sobreajustar en grados con pocos datos.
- Platt sobre el logit del score.

**Decisión.** Platt scaling sobre el logit del score, ajustado en el set de calibración
(ADR 0001). Es monótona, así que el Gini no cambia. Tiene 2 parámetros y es fácil de auditar.

**Consecuencias.** La PD calibrada alimenta el cutoff, la master scale, el stress test y la
API. La calibración debe re-estimarse cuando cambie la tasa de default de la población.
