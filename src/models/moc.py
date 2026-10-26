"""
PD de grado con margen de conservadurismo (MoC)
===============================================
Esquema de master scale al estilo IRB: el modelo ORDENA y asigna un grado; cada grado
tiene una PD única que se usa para pérdida esperada y provisiones.

    PD_grado = max(PD media del grado, percentil q del default observado del grado)

con el percentil q tomado de la posterior de Jeffreys Beta(D + 0.5, N − D + 0.5).
Se estima sobre el split de VALIDACIÓN: el de calibración no sirve, porque Platt se
ajustó ahí y se ve calibrado por construcción, y el test queda reservado para el backtest.

La PD individual calibrada se sigue usando para ordenar y para la decisión de aprobación.
"""
import numpy as np
from scipy import stats

from src.validation.master_scale import assign_grades

MOC_QUANTILE = 0.75


def estimate_grade_pd(y, pd_hat, limits: list, grades: list, q: float = MOC_QUANTILE) -> dict:
    y, pd_hat = np.asarray(y), np.asarray(pd_hat)
    g = assign_grades(pd_hat, limits, grades)
    out = {}
    prev = 0.0
    for grade in grades:
        m = g == grade
        n, d = int(m.sum()), int(y[m].sum())
        pd_avg = float(pd_hat[m].mean())
        dr_q = float(stats.beta.ppf(q, d + 0.5, n - d + 0.5))
        pd_grade = max(pd_avg, dr_q, prev)          # PD de grado no decreciente
        out[grade] = {
            "n": n, "defaults": d, "pd_avg": round(pd_avg, 4),
            "default_rate": round(d / n, 4), f"dr_jeffreys_q{int(q * 100)}": round(dr_q, 4),
            "pd_grade": round(pd_grade, 4), "moc_factor": round(pd_grade / pd_avg, 3),
        }
        prev = pd_grade
    return out


def grade_pd(pd_hat, grade_table: dict, limits: list, grades: list) -> np.ndarray:
    """PD de grado (con MoC) para cada cliente."""
    g = assign_grades(pd_hat, limits, grades)
    lookup = {k: v["pd_grade"] for k, v in grade_table.items()}
    return np.array([lookup[x] for x in g])
