"""
Configuración compartida: rutas, semilla y política de crédito.
"""
import json
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models" / "champion"
REPORTS_DIR    = ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"
CONFIGS_DIR    = ROOT / "configs"

RANDOM_STATE = 42


def load_policy() -> dict:
    with open(CONFIGS_DIR / "credit_policy.json", encoding="utf-8") as f:
        return json.load(f)


def breakeven_pd(tasa: float, lgd: float) -> float:
    """
    PD de corte que hace cero la ganancia esperada de un préstamo a un período:
        E[ganancia] = (1 - PD) * tasa - PD * LGD = 0  ->  PD* = tasa / (tasa + LGD)
    Sólo tiene sentido aplicado sobre PDs calibradas.
    """
    return tasa / (tasa + lgd)


def reference_cutoff(policy: dict | None = None) -> float:
    policy = policy or load_policy()
    m = policy["mercados"][policy["mercado_referencia"]]
    return round(breakeven_pd(m["tasa_interes_anual"], m["lgd"]), 4)
