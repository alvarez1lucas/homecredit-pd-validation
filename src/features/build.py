"""
Feature engineering — Home Credit Default Risk
===============================================
Lee train.parquet y test.parquet generados por ingest.py.
Aplica el preprocessor (imputación + scaler + OHE) sin data leakage.
Guarda X_train, X_test, y_train, y_test y el preprocessor.
"""
import json
import pickle
import warnings

import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED

warnings.filterwarnings("ignore")


def load_metadata() -> dict:
    with open(DATA_PROCESSED / "metadata.json") as f:
        return json.load(f)


def model_feature_lists(meta: dict) -> tuple[list, list]:
    """Columnas crudas que entran al preprocessor (numéricas + derivadas, categóricas)."""
    numeric = list(dict.fromkeys(meta["numeric_features"] + meta["derived_features"]))
    return numeric, list(meta["categorical_features"])


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Pipeline de transformación:
      - Numéricos: mediana + indicador de missing + StandardScaler.
        El indicador conserva la información de los missing (el EDA mostró que
        son MNAR: p.ej. OWN_CAR_AGE vacío ≈ no tiene auto).
      - Categóricos: imputar con 'missing' + OneHotEncoder.
    Se fitea SOLO sobre train para evitar data leakage.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      numeric_features),
            ("cat", categorical_pipeline,  categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build():
    logger.info("=" * 50)
    logger.info("STAGE 2 — Features")
    logger.info("=" * 50)

    meta = load_metadata()

    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    test_df  = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    numeric_features, categorical_features = model_feature_lists(meta)
    logger.info(f"Features numéricas:    {len(numeric_features)}")
    logger.info(f"Features categóricas:  {len(categorical_features)}")

    target_col = meta["target"]
    y_train = train_df[target_col]
    y_test  = test_df[target_col]

    # Fit preprocessor SOLO sobre train
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_arr  = preprocessor.fit_transform(train_df)
    X_test_arr   = preprocessor.transform(test_df)

    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)

    X_train.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    X_test.to_parquet(DATA_PROCESSED  / "X_test.parquet",  index=False)
    y_train.to_frame().to_parquet(DATA_PROCESSED / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(DATA_PROCESSED  / "y_test.parquet",  index=False)

    with open(DATA_PROCESSED / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(DATA_PROCESSED / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    logger.success(
        f"Features OK — {X_train.shape[1]} features | "
        f"train {X_train.shape[0]:,} | test {X_test.shape[0]:,}"
    )


if __name__ == "__main__":
    build()
