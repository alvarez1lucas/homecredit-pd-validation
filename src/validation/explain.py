"""
Explainability — SHAP
=====================
Genera explicaciones globales (summary) y locales (por predicción).
Guarda el explainer para uso en la API y el dashboard.

Los SHAP values están en escala log-odds del score crudo de XGBoost.
La calibración (Platt) es monótona, así que el signo y el ranking de
los factores son los mismos para la PD calibrada.
"""
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap
from loguru import logger

from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, RANDOM_STATE


def clean_name(f: str) -> str:
    return f.replace("num__", "").replace("cat__", "")


def top_factors(feature_names, shap_row, k=5) -> list:
    top = sorted(zip(feature_names, shap_row), key=lambda x: -abs(x[1]))[:k]
    return [
        {"feature": clean_name(f), "shap_value": round(float(v), 4),
         "direction": "aumenta_riesgo" if v > 0 else "reduce_riesgo"}
        for f, v in top
    ]


def explain():
    logger.info("=" * 50)
    logger.info("STAGE 6 — Explainability (SHAP)")
    logger.info("=" * 50)

    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    feature_names = [clean_name(c) for c in X_test.columns]

    background = X_train.sample(1000, random_state=RANDOM_STATE)
    explainer  = shap.TreeExplainer(model, background)

    X_sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
    logger.info(f"  Calculando SHAP para {len(X_sample)} muestras...")
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, color="#534AB7", max_display=20)
    plt.title("Importancia global de features (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Beeswarm — dirección y magnitud")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(MODELS_DIR / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    sv_row = explainer.shap_values(X_test.iloc[[0]]).flatten()
    sample_exp = {
        "base_value_logodds": round(float(explainer.expected_value), 4),
        "top_factors":        top_factors(X_test.columns, sv_row),
    }
    with open(MODELS_DIR / "sample_explanation.json", "w") as f:
        json.dump(sample_exp, f, indent=2)

    logger.success("SHAP OK — explainer y figuras guardados")


if __name__ == "__main__":
    explain()
