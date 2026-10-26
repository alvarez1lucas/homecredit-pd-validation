"""
Exporta las predicciones del test set para el dashboard de Streamlit.

El dashboard lee sólo reports/*.json y este parquet (~1 MB), así puede
deployarse sin los datos crudos ni el modelo y mostrar siempre resultados reales.
"""
import shutil

import pandas as pd
from loguru import logger

from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, ROOT, load_policy
from src.models.pd_model import PDScorer
from src.validation.master_scale import assign_grades
from src.validation.model_validation import age_groups


def export_dashboard_data():
    scorer = PDScorer()
    X_test = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    raw    = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    ms     = load_policy()["master_scale"]

    pd_hat = scorer.pd_from_features(X_test)
    out = pd.DataFrame({
        "pd":          pd_hat.astype("float32"),
        "default":     raw["default"].astype("int8"),
        "grade":       assign_grades(pd_hat, ms["limites_superiores"], ms["grados"]),
        "gender":      raw["CODE_GENDER"],
        "age_group":   age_groups(raw["age_years"]).values,
        "income_type": raw["NAME_INCOME_TYPE"],
        "amt_credit":  raw["AMT_CREDIT"].astype("float32"),
    })
    path = REPORTS_DIR / "dashboard" / "test_predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    # Metadata del modelo y de DL, para que el dashboard no dependa de models/
    for src_path in [MODELS_DIR / "model_metadata.json", ROOT / "models" / "dl" / "dl_metadata.json"]:
        if src_path.exists():
            shutil.copy(src_path, path.parent / src_path.name)
    logger.success(f"Dashboard data exportada: {path} ({len(out):,} filas)")


if __name__ == "__main__":
    export_dashboard_data()
