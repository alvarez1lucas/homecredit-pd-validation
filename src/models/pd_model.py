"""
PDScorer — punto único para pasar de datos crudos a PD calibrada.

Lo usan la API, el stress test, las notebooks y el dashboard, para que
todos calculen la PD exactamente de la misma forma.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR
from src.data.ingest import add_derived_features


class PDScorer:
    def __init__(self, models_dir: Path = MODELS_DIR, data_dir: Path = DATA_PROCESSED):
        with open(models_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(models_dir / "calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)
        with open(data_dir / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(data_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)
        # PD de grado con margen de conservadurismo (src/models/moc.py), si existe
        grade_path = models_dir / "grade_pd.json"
        self.grade_table = json.load(open(grade_path))["grades"] if grade_path.exists() else None
        cols_by_kind = {
            name: list(cols) for name, _, cols in self.preprocessor.transformers_
            if name in ("num", "cat")
        }
        self.numeric_columns     = cols_by_kind["num"]
        self.categorical_columns = cols_by_kind["cat"]
        self.raw_columns = self.numeric_columns + self.categorical_columns

    # ── desde features ya transformadas ───────────────────────────────────────
    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def pd_from_features(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.raw_score(X))

    # ── desde datos crudos (columnas de application_train) ───────────────────
    def transform_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        # Campos opcionales que llegan como None quedan con dtype object → numérico
        for c in df.columns:
            if c not in self.categorical_columns and c != "CODE_GENDER":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "DAYS_BIRTH" in df.columns:
            df = add_derived_features(df)
        # Columnas no informadas → missing (el preprocessor las imputa)
        for c in self.numeric_columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        for c in self.categorical_columns:
            if c not in df.columns:
                df[c] = pd.Series([None] * len(df), index=df.index, dtype=object)
            else:
                df[c] = df[c].astype(object)
        return pd.DataFrame(self.preprocessor.transform(df), columns=self.feature_names)

    def pd_from_raw(self, df_raw: pd.DataFrame) -> np.ndarray:
        return self.pd_from_features(self.transform_raw(df_raw))

    # ── PD de grado (para pérdida esperada / provisiones) ─────────────────────
    def grade_pd(self, pd_individual) -> np.ndarray:
        """PD de grado con MoC. Sin tabla de grados, devuelve la PD individual."""
        if self.grade_table is None:
            return np.asarray(pd_individual)
        from src.config import load_policy
        from src.models.moc import grade_pd
        ms = load_policy()["master_scale"]
        return grade_pd(pd_individual, self.grade_table, ms["limites_superiores"], ms["grados"])
