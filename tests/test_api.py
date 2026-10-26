"""
Test de integración de la API. Requiere los artefactos del pipeline
(python run_all.py); si no existen, se saltea.
"""
import pytest

from src.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "calibrator.pkl").exists(),
    reason="Artefactos del modelo no disponibles (correr run_all.py)",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/model/health").json()["status"] == "healthy"


def test_predict_minimal_payload(client):
    payload = {"AMT_CREDIT": 270000, "AMT_ANNUITY": 13500,
               "AMT_INCOME_TOTAL": 135000, "DAYS_BIRTH": -12000}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    assert body["decision"] in {"approved", "rejected"}
    assert body["grade_pd"] >= body["probability_of_default"] * 0.8   # PD de grado del mismo orden
    assert len(body["explanation"]["top_factors"]) == 5


def test_predict_risk_ordering(client):
    base = {"AMT_CREDIT": 270000, "AMT_ANNUITY": 13500, "AMT_INCOME_TOTAL": 135000,
            "DAYS_BIRTH": -12000, "NAME_INCOME_TYPE": "Working"}
    good = client.post("/predict", json={**base, "EXT_SOURCE_2": 0.8, "EXT_SOURCE_3": 0.8}).json()
    bad  = client.post("/predict", json={**base, "EXT_SOURCE_2": 0.1, "EXT_SOURCE_3": 0.1}).json()
    assert bad["probability_of_default"] > good["probability_of_default"]


def test_predict_rejects_invalid(client):
    r = client.post("/predict", json={"AMT_CREDIT": -5, "AMT_ANNUITY": 1,
                                      "AMT_INCOME_TOTAL": 1, "DAYS_BIRTH": -1})
    assert r.status_code == 422
