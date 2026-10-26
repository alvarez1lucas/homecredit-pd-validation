"""
Drift Monitoring — Home Credit Default Risk
===========================================
create_baseline(): guarda la distribución de referencia (PD calibrada y
features crudas del set de entrenamiento).

check_drift(path): compara un lote nuevo de solicitudes (mismo formato que
application_train.csv) contra la referencia. Sin un lote real, corre una
simulación explícitamente marcada como tal.
"""
import json

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from src.data.ingest import DERIVED_FEATURES, NUMERIC_FEATURES, add_derived_features
from src.models.pd_model import PDScorer
from src.validation.model_validation import compute_psi

PSI_ALERT, PSI_RETRAIN = 0.10, 0.25
MONITORED_FEATURES = NUMERIC_FEATURES + DERIVED_FEATURES


def create_baseline():
    logger.info("=" * 50)
    logger.info("STAGE 7 — Drift Baseline")
    logger.info("=" * 50)

    scorer = PDScorer()
    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    raw     = pd.read_parquet(DATA_PROCESSED / "train.parquet")

    ref = raw[MONITORED_FEATURES].copy()
    ref["pd"] = scorer.pd_from_features(X_train)
    ref.to_parquet(MODELS_DIR / "reference_data.parquet", index=False)

    config = {
        "psi_alert_threshold":    PSI_ALERT,
        "psi_retrain_threshold":  PSI_RETRAIN,
        "reference_mean_pd":      float(ref["pd"].mean()),
        "reference_default_rate": float(raw["default"].mean()),
        "n_reference_rows":       int(len(ref)),
        "monitored_features":     MONITORED_FEATURES,
    }
    with open(MODELS_DIR / "drift_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.success(f"Baseline OK — {len(ref):,} filas | PD media: {ref['pd'].mean():.4f}")


def _feature_psi(ref: pd.Series, new: pd.Series) -> float | None:
    r, n = ref.dropna().values, new.dropna().values
    if len(r) < 100 or len(n) < 100 or np.unique(r).size < 5:
        return None
    return compute_psi(r, n)


def check_drift(new_data_path: str | None = None) -> dict:
    """Compara un lote nuevo contra la referencia. Retorna y guarda el reporte de drift."""
    with open(MODELS_DIR / "drift_config.json") as f:
        config = json.load(f)
    reference = pd.read_parquet(MODELS_DIR / "reference_data.parquet")
    scorer = PDScorer()

    if new_data_path:
        new_raw, simulated = pd.read_csv(new_data_path), False
    else:
        # SIMULACIÓN para demo: test set con inflación nominal del 20% en montos
        # y deterioro del 10% en scores de bureau. No es data real de producción.
        new_raw = pd.read_parquet(DATA_PROCESSED / "test.parquet")
        for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            new_raw[c] = new_raw[c] * 1.20
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            new_raw[c] = new_raw[c] * 0.90
        simulated = True

    new_pd = scorer.pd_from_raw(new_raw)
    if "DAYS_BIRTH" in new_raw.columns:
        new_raw = add_derived_features(new_raw)

    score_psi = compute_psi(reference["pd"].values, new_pd)
    feature_psi = {}
    for feat in config["monitored_features"]:
        if feat in new_raw.columns:
            v = _feature_psi(reference[feat], new_raw[feat])
            if v is not None:
                feature_psi[feat] = v

    alert = ("ok" if score_psi <= PSI_ALERT
             else "monitor" if score_psi <= PSI_RETRAIN
             else "retrain_required")

    report = {
        "simulated":   simulated,
        "score_psi":   score_psi,
        "mean_pd_new": round(float(new_pd.mean()), 4),
        "mean_pd_ref": round(config["reference_mean_pd"], 4),
        "alert_level": alert,
        "thresholds":  {"alert": PSI_ALERT, "retrain": PSI_RETRAIN},
        "top_drifting_features": dict(sorted(feature_psi.items(), key=lambda x: -x[1])[:5]),
        "recommendation": {
            "ok":               "Sin acción requerida",
            "monitor":          "Aumentar frecuencia de monitoreo — investigar features que derivan",
            "retrain_required": "Escalar al model owner: evaluar recalibración o reentrenamiento",
        }[alert],
    }
    with open(REPORTS_DIR / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Drift check{' (SIMULADO)' if simulated else ''}: PSI={score_psi:.4f} | {alert.upper()}")
    return report


if __name__ == "__main__":
    create_baseline()
