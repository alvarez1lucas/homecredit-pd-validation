"""
Training pipeline — Home Credit Default Risk
=============================================
Entrena XGBoost (challenger) vs Regresión Logística (benchmark).
Registra experimentos en MLflow.

Esquema de datos (el test set NO se usa para ninguna decisión de entrenamiento):
    train (80%) ─┬─ fit          (80% de train) → entrenar modelos
                 ├─ validation   (10% de train) → early stopping + selección de champion
                 └─ calibration  (10% de train) → ajuste de Platt scaling
    test  (20%)  → sólo evaluación final
"""
import json
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
                        load_policy, reference_cutoff)
from src.models.calibration import PlattCalibrator
from src.models.moc import MOC_QUANTILE, estimate_grade_pd


def gini(y_true, y_score) -> float:
    return round(2 * roc_auc_score(y_true, y_score) - 1, 4)


def ranking_metrics(y, score, name: str) -> dict:
    """Métricas que no dependen de un threshold (válidas aunque el score no esté calibrado)."""
    return {
        "model":         name,
        "gini":          gini(y, score),
        "auc_roc":       round(float(roc_auc_score(y, score)), 4),
        "avg_precision": round(float(average_precision_score(y, score)), 4),
    }


def decision_metrics(y, pd_hat, cutoff: float) -> dict:
    """Métricas de la decisión aprobar/rechazar con PD calibrada y el cutoff de la política."""
    y = np.asarray(y)
    rejected = pd_hat >= cutoff
    tp = int((rejected & (y == 1)).sum())
    fp = int((rejected & (y == 0)).sum())
    fn = int((~rejected & (y == 1)).sum())
    tn = int((~rejected & (y == 0)).sum())
    return {
        "cutoff_pd":              cutoff,
        "approval_rate":          round(float((~rejected).mean()), 4),
        "default_rate_approved":  round(float(y[~rejected].mean()), 4),
        "default_rate_rejected":  round(float(y[rejected].mean()), 4) if rejected.any() else None,
        "precision_reject":       round(tp / (tp + fp), 4) if tp + fp else None,
        "recall_defaults":        round(tp / (tp + fn), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def split_train(y_train: pd.Series) -> dict:
    idx = np.arange(len(y_train))
    idx_fit, idx_hold = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )
    idx_val, idx_cal = train_test_split(
        idx_hold, test_size=0.50, random_state=RANDOM_STATE, stratify=y_train.iloc[idx_hold]
    )
    return {"fit": np.sort(idx_fit), "val": np.sort(idx_val), "cal": np.sort(idx_cal)}


def train_baseline(X, y):
    """Regresión logística — benchmark: el modelo complejo tiene que superarla."""
    model = LogisticRegression(
        C=0.1, max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="saga",
    )
    model.fit(X, y)
    return model


def train_xgboost(X_fit, y_fit, X_val, y_val):
    """
    XGBoost con early stopping sobre el set de validación.
    scale_pos_weight compensa el desbalance (~8% default); la escala de
    probabilidades resultante se corrige después con Platt scaling.
    """
    ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
    logger.info(f"  scale_pos_weight = {ratio:.2f} (ratio no-default/default)")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        scale_pos_weight=ratio,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(f"  Best iteration: {model.best_iteration}")
    return model


def train():
    logger.info("=" * 50)
    logger.info("STAGE 3 — Training")
    logger.info("=" * 50)

    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet").iloc[:, 0]
    y_test  = pd.read_parquet(DATA_PROCESSED / "y_test.parquet").iloc[:, 0]

    splits = split_train(y_train)
    np.savez(DATA_PROCESSED / "train_splits.npz", **splits)
    X_fit, y_fit = X_train.iloc[splits["fit"]], y_train.iloc[splits["fit"]]
    X_val, y_val = X_train.iloc[splits["val"]], y_train.iloc[splits["val"]]
    X_cal, y_cal = X_train.iloc[splits["cal"]], y_train.iloc[splits["cal"]]
    logger.info(f"  fit {len(X_fit):,} | validation {len(X_val):,} | calibration {len(X_cal):,}")

    mlflow.set_experiment("homecredit-model-validation")
    candidates = {}

    with mlflow.start_run(run_name="logistic_baseline"):
        logger.info("Entrenando benchmark (Regresión Logística)...")
        lr = train_baseline(X_fit, y_fit)
        m  = ranking_metrics(y_val, lr.predict_proba(X_val)[:, 1], "logistic_baseline")
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(lr, "model")
        candidates["logistic_baseline"] = (lr, m)
        logger.info(f"  Benchmark (val) → Gini: {m['gini']}")

    with mlflow.start_run(run_name="xgboost"):
        logger.info("Entrenando XGBoost...")
        xgb_model = train_xgboost(X_fit, y_fit, X_val, y_val)
        m = ranking_metrics(y_val, xgb_model.predict_proba(X_val)[:, 1], "xgboost")
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if isinstance(v, float)})
        mlflow.log_param("best_iteration", xgb_model.best_iteration)
        mlflow.sklearn.log_model(xgb_model, "model")
        candidates["xgboost"] = (xgb_model, m)
        logger.info(f"  XGBoost (val)   → Gini: {m['gini']}")

    # ── Champion: se elige en VALIDATION, no en test ──────────────────────────
    champion_name = max(candidates, key=lambda k: candidates[k][1]["gini"])
    champion      = candidates[champion_name][0]

    # ── Calibración sobre el set de calibración (no visto por el modelo) ─────
    calibrator = PlattCalibrator().fit(champion.predict_proba(X_cal)[:, 1], y_cal)
    logger.info(
        f"  Platt: PD = sigmoid({calibrator.slope_:.3f}·logit(score) "
        f"{calibrator.intercept_:+.3f})"
    )

    # ── PD de grado con margen de conservadurismo (estimada en VALIDATION) ───
    ms = load_policy()["master_scale"]
    grade_table = estimate_grade_pd(
        y_val, calibrator.transform(champion.predict_proba(X_val)[:, 1]),
        ms["limites_superiores"], ms["grados"],
    )
    for g, r in grade_table.items():
        if r["moc_factor"] > 1:
            logger.info(f"  MoC grado {g}: PD {r['pd_avg']:.2%} → {r['pd_grade']:.2%} (x{r['moc_factor']})")

    # ── Evaluación final en test ──────────────────────────────────────────────
    test_results = []
    for name, (model, val_m) in candidates.items():
        score = model.predict_proba(X_test)[:, 1]
        tm = ranking_metrics(y_test, score, name)
        tm["val_gini"] = val_m["gini"]
        test_results.append(tm)

    raw_test = champion.predict_proba(X_test)[:, 1]
    pd_test  = calibrator.transform(raw_test)
    cutoff   = reference_cutoff()

    champion_metrics = next(r for r in test_results if r["model"] == champion_name).copy()
    baseline_gini    = next(r for r in test_results if r["model"] == "logistic_baseline")["gini"]
    champion_metrics.update({
        "gini_lift_vs_baseline": round(champion_metrics["gini"] - baseline_gini, 4),
        "best_iteration":        int(getattr(champion, "best_iteration", -1)),
        "mean_raw_score":        round(float(raw_test.mean()), 4),
        "mean_pd_calibrated":    round(float(pd_test.mean()), 4),
        "observed_default_rate": round(float(y_test.mean()), 4),
        "brier_raw":             round(float(brier_score_loss(y_test, raw_test)), 4),
        "brier_calibrated":      round(float(brier_score_loss(y_test, pd_test)), 4),
        "decision":              decision_metrics(y_test, pd_test, cutoff),
    })

    logger.success(
        f"Champion: {champion_name} | Gini test: {champion_metrics['gini']} | "
        f"Lift vs benchmark: {champion_metrics['gini_lift_vs_baseline']:+}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(champion, f)
    with open(MODELS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(MODELS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    with open(MODELS_DIR / "grade_pd.json", "w") as f:
        json.dump({"method": f"max(PD media, Jeffreys q{int(MOC_QUANTILE * 100)} del default "
                             "observado) en el set de validación",
                   "grades": grade_table}, f, indent=2)

    metadata = {
        "champion":         champion_name,
        "selection":        "Gini en set de validación (test no usado para entrenar ni elegir)",
        "champion_metrics": champion_metrics,
        "all_models_test":  test_results,
        "calibration":      {"method": "platt_on_logit", "slope": calibrator.slope_,
                             "intercept": calibrator.intercept_, "n_calibration": int(len(X_cal))},
        "feature_count":    int(X_train.shape[1]),
        "rows":             {"fit": int(len(X_fit)), "validation": int(len(X_val)),
                             "calibration": int(len(X_cal)), "test": int(len(X_test))},
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(REPORTS_DIR / "train_metrics.json", "w") as f:
        json.dump(champion_metrics, f, indent=2)

    logger.success("Training OK — artefactos guardados en models/champion/")


if __name__ == "__main__":
    train()
