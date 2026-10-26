"""
FastAPI — Credit Risk Scoring API
===================================
POST /predict       → PD calibrada (decisión), grado y PD de grado con MoC (pérdida esperada) + SHAP
GET  /model/health  → health check
GET  /model/info    → metadata del modelo y política de decisión
"""
import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, load_policy, reference_cutoff
from src.models.pd_model import PDScorer
from src.validation.explain import top_factors
from src.validation.master_scale import assign_grades

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelo...")
    _state["scorer"] = PDScorer()
    with open(MODELS_DIR / "shap_explainer.pkl", "rb") as f:
        _state["explainer"] = pickle.load(f)
    with open(MODELS_DIR / "model_metadata.json") as f:
        _state["metadata"] = json.load(f)
    policy = load_policy()
    _state["policy"] = policy
    _state["cutoff"] = reference_cutoff(policy)
    _state["lgd"]    = policy["mercados"][policy["mercado_referencia"]]["lgd"]
    _state["request_count"] = 0
    _state["start_time"]    = time.time()
    logger.success("API lista")
    yield
    _state.clear()


app = FastAPI(
    title="Home Credit — PD Scoring API",
    description="PD calibrada + explicación SHAP por solicitud (proyecto de portfolio).",
    version="2.0.0",
    lifespan=lifespan,
)


class CreditApplication(BaseModel):
    """Campos de application_train.csv. Los opcionales que no se envíen se imputan."""
    # Obligatorios
    AMT_CREDIT:        float = Field(..., gt=0, examples=[270000])
    AMT_ANNUITY:       float = Field(..., gt=0, examples=[13500])
    AMT_INCOME_TOTAL:  float = Field(..., gt=0, examples=[135000])
    DAYS_BIRTH:        int   = Field(..., lt=0, examples=[-12000], description="Días desde el nacimiento (negativo)")
    # Numéricos opcionales
    AMT_GOODS_PRICE:              Optional[float] = Field(None, examples=[270000])
    DAYS_EMPLOYED:                Optional[int]   = Field(None, examples=[-3000])
    DAYS_REGISTRATION:            Optional[float] = Field(None, examples=[-4000])
    DAYS_ID_PUBLISH:              Optional[int]   = Field(None, examples=[-3000])
    EXT_SOURCE_1:                 Optional[float] = Field(None, ge=0, le=1, examples=[0.50])
    EXT_SOURCE_2:                 Optional[float] = Field(None, ge=0, le=1, examples=[0.60])
    EXT_SOURCE_3:                 Optional[float] = Field(None, ge=0, le=1, examples=[0.55])
    CNT_CHILDREN:                 Optional[int]   = Field(None, ge=0, examples=[0])
    CNT_FAM_MEMBERS:              Optional[float] = Field(None, ge=0, examples=[2])
    REGION_POPULATION_RELATIVE:   Optional[float] = Field(None, examples=[0.02])
    HOUR_APPR_PROCESS_START:      Optional[int]   = Field(None, ge=0, le=23, examples=[11])
    REGION_RATING_CLIENT_W_CITY:  Optional[int]   = Field(None, ge=1, le=3, examples=[2])
    OWN_CAR_AGE:                  Optional[float] = Field(None, ge=0, examples=[None])
    DEF_30_CNT_SOCIAL_CIRCLE:     Optional[float] = Field(None, ge=0, examples=[0])
    DEF_60_CNT_SOCIAL_CIRCLE:     Optional[float] = Field(None, ge=0, examples=[0])
    DAYS_LAST_PHONE_CHANGE:       Optional[float] = Field(None, examples=[-1000])
    AMT_REQ_CREDIT_BUREAU_YEAR:   Optional[float] = Field(None, ge=0, examples=[1])
    # Categóricos opcionales
    NAME_CONTRACT_TYPE:           Optional[str] = Field(None, examples=["Cash loans"])
    NAME_INCOME_TYPE:             Optional[str] = Field(None, examples=["Working"])
    NAME_EDUCATION_TYPE:          Optional[str] = Field(None, examples=["Secondary / secondary special"])
    NAME_FAMILY_STATUS:           Optional[str] = Field(None, examples=["Married"])
    NAME_HOUSING_TYPE:            Optional[str] = Field(None, examples=["House / apartment"])
    OCCUPATION_TYPE:              Optional[str] = Field(None, examples=["Laborers"])
    ORGANIZATION_TYPE:            Optional[str] = Field(None, examples=["Business Entity Type 3"])
    WEEKDAY_APPR_PROCESS_START:   Optional[str] = Field(None, examples=["TUESDAY"])
    HOUSETYPE_MODE:               Optional[str] = Field(None, examples=["block of flats"])
    WALLSMATERIAL_MODE:           Optional[str] = Field(None, examples=["Panel"])
    EMERGENCYSTATE_MODE:          Optional[str] = Field(None, examples=["No"])


class PredictionResponse(BaseModel):
    probability_of_default: float
    grade:                  str
    grade_pd:               float
    decision:               str
    cutoff_pd:              float
    expected_loss:          float
    explanation:            dict
    model_version:          str
    latency_ms:             float


@app.get("/model/health")
async def health():
    return {
        "status":          "healthy",
        "uptime_seconds":  round(time.time() - _state.get("start_time", time.time()), 1),
        "requests_served": _state.get("request_count", 0),
    }


@app.get("/model/info")
async def model_info():
    meta = _state.get("metadata", {})
    return {
        "champion":       meta.get("champion"),
        "test_metrics":   meta.get("champion_metrics"),
        "calibration":    meta.get("calibration"),
        "decision_rule":  {"market": _state["policy"]["mercado_referencia"],
                           "reject_if_pd_gte": _state["cutoff"], "lgd": _state["lgd"]},
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    t0 = time.time()
    _state["request_count"] += 1
    try:
        scorer = _state["scorer"]
        X      = scorer.transform_raw(pd.DataFrame([application.model_dump()]))
        pd_hat = float(scorer.pd_from_features(X)[0])
        ms     = _state["policy"]["master_scale"]
        grade  = str(assign_grades([pd_hat], ms["limites_superiores"], ms["grados"])[0])
        # PD de grado con margen de conservadurismo: se usa para pérdida esperada
        pd_grade = float(scorer.grade_pd([pd_hat])[0])

        try:
            sv = _state["explainer"].shap_values(X).flatten()
            explanation = {"top_factors": top_factors(scorer.feature_names, sv),
                           "scale": "log-odds del score (positivo = aumenta riesgo)"}
        except Exception:
            explanation = {"top_factors": [], "note": "explanation_unavailable"}

        return PredictionResponse(
            probability_of_default = round(pd_hat, 4),
            grade                  = grade,
            grade_pd               = round(pd_grade, 4),
            decision               = "rejected" if pd_hat >= _state["cutoff"] else "approved",
            cutoff_pd              = _state["cutoff"],
            expected_loss          = round(pd_grade * _state["lgd"] * application.AMT_CREDIT, 2),
            explanation            = explanation,
            model_version          = _state["metadata"].get("champion", "unknown"),
            latency_ms             = round((time.time() - t0) * 1000, 2),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="prediction_failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
