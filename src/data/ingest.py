"""
Data ingestion — Home Credit Default Risk
=========================================
Lee application_train.csv, corre data quality checks,
y genera train/test split estratificado.

Dataset: https://www.kaggle.com/c/home-credit-default-risk/data
Archivo requerido: data/raw/application_train.csv
"""
import json
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, DATA_RAW, RANDOM_STATE, REPORTS_DIR

warnings.filterwarnings("ignore")

# ── Features numéricas crudas ─────────────────────────────────────────────────
# Decisiones del EDA (notebook 01):
#   - REGION_RATING_CLIENT se descarta: casi colineal con REGION_RATING_CLIENT_W_CITY
#   - DAYS_BIRTH no entra directo: la edad entra una sola vez como age_years
NUMERIC_FEATURES = [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "HOUR_APPR_PROCESS_START",
    "REGION_RATING_CLIENT_W_CITY",
    "OWN_CAR_AGE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# ── Features categóricas ──────────────────────────────────────────────────────
# FONDKAPREMONT_MODE se descarta (68% missing, regla del EDA: >65%)
CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

# ── Features derivadas (calculadas por add_derived_features) ──────────────────
DERIVED_FEATURES = [
    "age_years",
    "credit_income_ratio",
    "annuity_income_ratio",
    "credit_goods_ratio",
]

# ── Atributos protegidos ──────────────────────────────────────────────────────
# CODE_GENDER se guarda sólo para el fairness audit y NO entra al modelo.
# La edad SÍ entra al modelo (age_years): es una variable de riesgo con sustento
# en el EDA, y su impacto se audita en src/governance/fairness.py.
PROTECTED_ATTRIBUTES = ["CODE_GENDER", "age_years"]

RAW_ONLY_COLUMNS = ["CODE_GENDER", "DAYS_BIRTH"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y features derivadas a partir de columnas crudas.
    La usan el ingest, el stress test y la API, para que las tres
    calculen exactamente lo mismo.
    """
    df = df.copy()

    # Edad en años (positiva)
    df["age_years"] = (-df["DAYS_BIRTH"] / 365).round(1)

    # DAYS_EMPLOYED = 365243 es un código de "sin empleo formal / jubilado" → NaN
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    # Endeudamiento: monto del crédito sobre ingreso anual
    df["credit_income_ratio"]  = (df["AMT_CREDIT"]  / income).round(4)
    # Carga financiera: cuota sobre ingreso
    df["annuity_income_ratio"] = (df["AMT_ANNUITY"] / income).round(4)
    # Financiamiento por encima del valor del bien (EDA: AMT_CREDIT y AMT_GOODS_PRICE son colineales)
    df["credit_goods_ratio"]   = (
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    ).round(4)
    return df


def load_home_credit() -> pd.DataFrame:
    """Carga y prepara application_train.csv."""
    raw_path = DATA_RAW / "application_train.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"\nNo se encontró: {raw_path}\n"
            "Descargá application_train.csv de Kaggle:\n"
            "  https://www.kaggle.com/c/home-credit-default-risk/data\n"
            "y colocalo en data/raw/"
        )

    logger.info(f"Cargando {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape[0]:,} filas × {df.shape[1]} cols")

    # Renombrar target para que el pipeline sea agnóstico al dataset
    df = df.rename(columns={"TARGET": "default"})

    cols = list(dict.fromkeys(
        ["SK_ID_CURR", "default"]
        + NUMERIC_FEATURES + CATEGORICAL_FEATURES + RAW_ONLY_COLUMNS
    ))
    df = add_derived_features(df[cols])

    logger.info(
        f"Shape post-selección: {df.shape[0]:,} × {df.shape[1]} | "
        f"default rate: {df['default'].mean():.2%}"
    )
    return df


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    """Checks de calidad adaptados a Home Credit."""
    default_rate  = df["default"].mean()
    missing_pct   = df.isnull().mean().round(4).to_dict()
    high_missing  = {c: v for c, v in missing_pct.items() if v > 0.30}

    report = {
        "dataset":             "Home Credit Default Risk",
        "n_rows":              int(len(df)),
        "n_cols":              int(len(df.columns)),
        "default_rate":        round(float(default_rate), 4),
        "class_ratio":         f"1:{int((1 - default_rate) / default_rate)}",
        "missing_pct":         missing_pct,
        "high_missing_cols":   high_missing,
        "target_distribution": df["default"].value_counts(normalize=True).round(4).to_dict(),
        "duplicated_ids":      int(df["SK_ID_CURR"].duplicated().sum()),
        "passed":              True,
        "warnings":            [],
    }

    if len(df) < 1000:
        report["warnings"].append("Menos de 1000 filas")
        report["passed"] = False

    if report["duplicated_ids"] > 0:
        report["warnings"].append(f"{report['duplicated_ids']} SK_ID_CURR duplicados")
        report["passed"] = False

    if not (0.05 <= default_rate <= 0.15):
        report["warnings"].append(
            f"Default rate {default_rate:.2%} fuera del rango esperado 5-15%"
        )

    logger.info(
        f"DQ: {report['n_rows']:,} filas | "
        f"default {default_rate:.2%} | ratio {report['class_ratio']}"
    )
    if high_missing:
        logger.warning(f"  {len(high_missing)} cols con >30% missing (normal en Home Credit)")
    for w in report["warnings"]:
        logger.warning(f"  [DQ WARNING] {w}")

    return report


def ingest():
    logger.info("=" * 50)
    logger.info("STAGE 1 — Ingest (Home Credit)")
    logger.info("=" * 50)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_home_credit()
    quality_report = run_data_quality_checks(df)

    with open(REPORTS_DIR / "data_quality.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    # Split estratificado — misma proporción de defaults en train y test.
    # Home Credit no trae fecha de originación, así que no es posible un split temporal.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["default"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_df.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROCESSED  / "test.parquet",  index=False)

    metadata = {
        "dataset":              "Home Credit Default Risk",
        "n_train":              int(len(train_df)),
        "n_test":               int(len(test_df)),
        "numeric_features":     NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "derived_features":     DERIVED_FEATURES,
        "target":               "default",
        "protected_attributes": PROTECTED_ATTRIBUTES,
        "default_rate_train":   float(train_df["default"].mean()),
        "default_rate_test":    float(test_df["default"].mean()),
    }
    with open(DATA_PROCESSED / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success(
        f"Ingest OK — train: {len(train_df):,} | test: {len(test_df):,} | "
        f"default rate: {train_df['default'].mean():.2%}"
    )


if __name__ == "__main__":
    ingest()
