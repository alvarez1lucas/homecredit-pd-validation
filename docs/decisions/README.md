# Architectural Decision Records

Decisiones de diseño del modelo y de la validación, con su contexto y sus consecuencias.

| # | Decisión | Estado |
|---|---|---|
| [0001](0001-esquema-de-splits.md) | Separar fit / validación / calibración dentro del train | Aceptada |
| [0002](0002-calibracion-platt.md) | Calibrar el score con Platt scaling | Aceptada |
| [0003](0003-xgboost-vs-benchmark.md) | XGBoost como champion, logística y scorecard como benchmarks | Aceptada |
| [0004](0004-atributos-protegidos.md) | Género fuera del modelo, edad dentro y auditada | Aceptada, a revisar |
| [0005](0005-cutoff-break-even.md) | Cutoff de decisión por break-even de PD | Aceptada |
| [0006](0006-semaforo-master-scale.md) | Semáforo en el backtesting de la master scale | Aceptada |
| [0007](0007-pd-de-grado-con-moc.md) | PD de grado con margen de conservadurismo | Aceptada |
| [0008](0008-challengers.md) | Challengers monotónico y con historial de pagos | M3 recomendado |
