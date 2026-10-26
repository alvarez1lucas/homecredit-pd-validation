# Dashboard de validación — Streamlit

Dashboard bilingüe (ES/EN) con los resultados **reales** del pipeline. No tiene modo demo
ni datos sintéticos.

## Qué lee

| Fuente | Contenido |
|---|---|
| `reports/*.json` | validación, fairness, scorecard, challengers, estado y hallazgos |
| `reports/dashboard/test_predictions.parquet` | PD calibrada y default real del test set (~1 MB) |
| `reports/model_cards/model_card.html` | Model Card |
| `models/` + `data/processed/` *(opcional)* | habilitan el simulador individual |

Como `reports/` se versiona, la app se puede deployar sin los datos crudos ni el modelo.
En ese caso el simulador muestra un aviso y el resto funciona igual.

## Secciones

1. **Resumen:** estado de la validación (aprobado / con condiciones / no aprobado), KPIs, checks y tabla de hallazgos con recomendaciones.
2. **Discriminación y calibración:** deciles, reliability diagram, master scale con test de Jeffreys, PD de grado con MoC.
3. **Stress test:** PD y pérdida esperada por escenario.
4. **Fairness:** aprobación vs default real por género y edad, con un slider de cutoff.
5. **Política de crédito:** cutoff por break-even `PD* = r/(r+LGD)` por mercado, con tasa y LGD ajustables.
6. **Scorecard:** Information Value y tabla de puntos por variable.
7. **Challengers:** monotónico, con historial de pagos y la combinación de ambos, contra el champion (Gini con IC, monotonía, backtest con MoC, fairness).
8. **Simulador:** PD individual, grado, PD de grado, decisión, pérdida esperada y factores SHAP (requiere el modelo).
9. **Model Card.**

## Correr localmente

```bash
# desde la raíz del proyecto, después de `python run_all.py`
pip install -r streamlit-app/requirements.txt
streamlit run streamlit-app/app.py
```

## Deploy en Streamlit Community Cloud

1. Subir el repo a GitHub, incluyendo `reports/`.
2. En https://share.streamlit.io, elegir el repo y poner `streamlit-app/app.py` como main file.
3. En *Advanced settings*, apuntar a `streamlit-app/requirements.txt`.
