"""
Master scale — grados de rating y backtesting de calibración por grado
======================================================================
Cada cliente se asigna a un grado según su PD calibrada. Para cada grado
se compara la PD promedio asignada con la tasa de default observada:

  - Test binomial (una cola): H0 = la PD del grado NO subestima el riesgo.
        p = P(X >= D | N, PD)
  - Test de Jeffreys (usado en las guías de validación del BCE):
        posterior Beta(D + 0.5, N - D + 0.5);  p = F_Beta(PD)
    p < 0.05 → el default observado es significativamente mayor que la PD → subestimación.

La subestimación es el error grave (provisiones y capital insuficientes);
la sobreestimación se informa pero es conservadora.

Semáforo (como en el backtesting de Basilea): con 7 grados testeados al 5%, la
probabilidad de al menos una falsa alarma ronda el 30%, así que:
    rojo  ("subestima") → p < 1%   (falla la validación)
    ámbar ("alerta")    → 1% ≤ p < 5% (se informa y se monitorea)
"""
import numpy as np
import pandas as pd
from scipy import stats

ALPHA_RED, ALPHA_AMBER = 0.01, 0.05


def assign_grades(pd_hat, limits: list, grades: list) -> np.ndarray:
    idx = np.searchsorted(np.asarray(limits), np.asarray(pd_hat), side="right")
    return np.asarray(grades)[idx]


def jeffreys_pvalue(d: int, n: int, pd_avg: float) -> float:
    return float(stats.beta.cdf(pd_avg, d + 0.5, n - d + 0.5))


def binomial_pvalue(d: int, n: int, pd_avg: float) -> float:
    return float(stats.binom.sf(d - 1, n, pd_avg))


def master_scale_table(y, pd_hat, limits: list, grades: list, pd_rank=None) -> pd.DataFrame:
    """
    pd_hat:  PD que se testea (individual, o PD de grado con MoC).
    pd_rank: PD usada para asignar el grado; por defecto, la misma pd_hat.
    """
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_hat)})
    rank = np.asarray(pd_hat if pd_rank is None else pd_rank)
    df["grade"] = assign_grades(rank, limits, grades)
    lower = [0.0] + list(limits)
    upper = list(limits) + [1.0]

    rows = []
    for i, g in enumerate(grades):
        sub = df[df["grade"] == g]
        n, d = len(sub), int(sub["y"].sum())
        if n == 0:
            continue
        pd_avg = float(sub["pd"].mean())
        p_jef  = jeffreys_pvalue(d, n, pd_avg)
        p_bin  = binomial_pvalue(d, n, pd_avg)
        if p_jef < ALPHA_RED:
            result = "subestima"
        elif p_jef < ALPHA_AMBER:
            result = "alerta"
        elif 1 - p_jef < ALPHA_RED:
            result = "sobreestima"
        else:
            result = "ok"
        lo, hi = stats.beta.ppf([0.025, 0.975], d + 0.5, n - d + 0.5)
        rows.append({
            "grade":            g,
            "pd_range":         f"[{lower[i]:.0%}, {upper[i]:.0%})",
            "n":                n,
            "share":            round(n / len(df), 4),
            "defaults":         d,
            "pd_avg":           round(pd_avg, 4),
            "default_rate":     round(d / n, 4),
            "dr_ci95":          [round(float(lo), 4), round(float(hi), 4)],
            "p_binomial":       round(p_bin, 4),
            "p_jeffreys":       round(p_jef, 4),
            "result":           result,
        })
    return pd.DataFrame(rows)


def master_scale_summary(table: pd.DataFrame) -> dict:
    dr = table["default_rate"].values
    shares = table["share"].values
    return {
        "monotonic_default_rate": bool(np.all(np.diff(dr) > 0)),
        "grades_underestimating": table.loc[table["result"] == "subestima", "grade"].tolist(),
        "grades_amber":           table.loc[table["result"] == "alerta", "grade"].tolist(),
        "grades_overestimating":  table.loc[table["result"] == "sobreestima", "grade"].tolist(),
        "max_grade_share":        round(float(shares.max()), 4),
        "hhi":                    round(float((shares ** 2).sum()), 4),
    }
