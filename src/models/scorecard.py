"""
Scorecard tradicional — WoE / IV + Regresión Logística + escala de puntos
=========================================================================
Es el estándar histórico de la industria para scoring de originación:
cada variable se discretiza en tramos, cada tramo suma o resta puntos,
y el puntaje total se traduce a odds con una regla fija (PDO).

Convenciones:
  - WoE = ln(%buenos / %malos)  → WoE alto = menor riesgo
  - Puntos: 600 puntos = odds 50:1 (buenos:malos), +20 puntos duplica las odds (PDO = 20)

Pasos:
  1. Binning: cuantiles (numéricas) / agrupación de categorías raras, con missing como tramo propio
  2. Monotonía: se fusionan tramos adyacentes hasta que el WoE sea monótono
  3. Selección: IV ≥ 0.02, sin pares con |corr(WoE)| > 0.70, coeficientes con el signo esperado
  4. Logística sobre WoE → puntos por tramo
"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import DATA_PROCESSED, FIGURES_DIR, REPORTS_DIR, ROOT
from src.data.ingest import CATEGORICAL_FEATURES, DERIVED_FEATURES, NUMERIC_FEATURES

SCORECARD_DIR = ROOT / "models" / "scorecard"

PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
MIN_IV, MAX_CORR, MAX_FEATURES = 0.02, 0.70, 15
MISSING = "MISSING"


def iv_strength(iv: float) -> str:
    """Escala de referencia habitual (Siddiqi)."""
    if iv < 0.02: return "no predictiva"
    if iv < 0.10: return "débil"
    if iv < 0.30: return "media"
    return "fuerte"


def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(bins, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["good", "bad"]
    # suavizado +0.5 para evitar log(0) en tramos sin malos
    pg = (t["good"] + 0.5) / (t["good"].sum() + 0.5)
    pb = (t["bad"]  + 0.5) / (t["bad"].sum()  + 0.5)
    t["n"]        = t["good"] + t["bad"]
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"]      = np.log(pg / pb)
    t["iv"]       = (pg - pb) * t["woe"]
    return t


class NumericBinner:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, x: pd.Series, y: pd.Series):
        xs = x.dropna()
        edges = np.unique(np.quantile(xs, np.linspace(0, 1, self.n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        self.edges_ = list(edges)
        # fusionar tramos adyacentes hasta que el bad rate sea monótono
        while len(self.edges_) > 3:
            t = _woe_table(self._cut(x[x.notna()]), y[x.notna()])
            br = t["bad_rate"].values
            r = np.corrcoef(np.arange(len(br)), br)[0, 1]
            trend = np.sign(r) if np.isfinite(r) and r != 0 else 1.0
            diffs = np.diff(br) * trend
            if (diffs >= 0).all():
                break
            j = int(np.argmin(diffs))      # par que más viola la tendencia
            del self.edges_[j + 1]
        self.table_ = _woe_table(self.transform_bins(x), y)
        return self

    def _cut(self, x):
        return pd.cut(x, self.edges_, right=False, labels=False).astype("Int64")

    def transform_bins(self, x: pd.Series) -> pd.Series:
        b = self._cut(x).astype(object)
        return b.where(x.notna(), MISSING)

    def labels(self) -> dict:
        lab = {}
        for i in range(len(self.edges_) - 1):
            lo, hi = self.edges_[i], self.edges_[i + 1]
            lab[i] = (f"< {hi:.4g}" if lo == -np.inf else
                      f">= {lo:.4g}" if hi == np.inf else f"[{lo:.4g}, {hi:.4g})")
        lab[MISSING] = "missing"
        return lab


class CategoricalBinner:
    def __init__(self, min_share=0.01):
        self.min_share = min_share

    def fit(self, x: pd.Series, y: pd.Series):
        share = x.fillna(MISSING).value_counts(normalize=True)
        self.keep_ = set(share[share >= self.min_share].index)
        self.table_ = _woe_table(self.transform_bins(x), y)
        return self

    def transform_bins(self, x: pd.Series) -> pd.Series:
        x = x.fillna(MISSING)
        return x.where(x.isin(self.keep_), "OTHER")

    def labels(self) -> dict:
        return {k: str(k) for k in self.table_.index}


class Scorecard:
    def fit(self, df: pd.DataFrame, y: pd.Series, numeric: list, categorical: list):
        y = y.reset_index(drop=True)
        df = df.reset_index(drop=True)
        self.binners_ = {}
        for c in numeric:
            self.binners_[c] = NumericBinner().fit(df[c], y)
        for c in categorical:
            self.binners_[c] = CategoricalBinner().fit(df[c], y)

        self.iv_ = pd.Series(
            {c: float(b.table_["iv"].sum()) for c, b in self.binners_.items()}
        ).sort_values(ascending=False)

        W = self.woe(df)
        selected = []
        for c in self.iv_[self.iv_ >= MIN_IV].index:
            if all(abs(np.corrcoef(W[c], W[s])[0, 1]) <= MAX_CORR for s in selected):
                selected.append(c)
            if len(selected) == MAX_FEATURES:
                break

        # con WoE = ln(buenos/malos), todos los coeficientes deben ser negativos
        while True:
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(W[selected], y)
            coefs = pd.Series(lr.coef_[0], index=selected)
            wrong = coefs[coefs >= 0]
            if wrong.empty:
                break
            selected.remove(wrong.idxmax())

        self.features_, self.lr_ = selected, lr
        self.factor_ = PDO / np.log(2)
        self.offset_ = BASE_SCORE - self.factor_ * np.log(BASE_ODDS)
        self._build_points()
        return self

    def woe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, b in self.binners_.items():
            woe_map = b.table_["woe"].to_dict()
            out[c] = b.transform_bins(df[c].reset_index(drop=True)).map(woe_map).fillna(0.0).astype(float)
        return pd.DataFrame(out)

    def _build_points(self):
        n = len(self.features_)
        b0 = self.lr_.intercept_[0]
        rows = []
        for c, beta in zip(self.features_, self.lr_.coef_[0]):
            b = self.binners_[c]
            labels = b.labels()
            for bin_id, r in b.table_.iterrows():
                pts = -(beta * r["woe"] + b0 / n) * self.factor_ + self.offset_ / n
                rows.append({
                    "variable": c, "tramo": labels.get(bin_id, str(bin_id)),
                    "n": int(r["n"]), "bad_rate": round(float(r["bad_rate"]), 4),
                    "woe": round(float(r["woe"]), 4), "puntos": int(round(pts)),
                })
        self.points_ = pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Puntaje total. Mayor puntaje = menor riesgo."""
        log_odds_bad = self.lr_.decision_function(self.woe(df)[self.features_])
        return self.offset_ - self.factor_ * log_odds_bad

    def pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.woe(df)[self.features_])[:, 1]


def build_scorecard():
    logger.info("=" * 50)
    logger.info("STAGE — Scorecard WoE/IV")
    logger.info("=" * 50)

    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    test  = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    splits = np.load(DATA_PROCESSED / "train_splits.npz")
    fit_df = train.iloc[splits["fit"]]

    numeric = NUMERIC_FEATURES + DERIVED_FEATURES
    sc = Scorecard().fit(fit_df, fit_df["default"], numeric, CATEGORICAL_FEATURES)

    y_test   = test["default"].values
    pd_test  = sc.pd(test)
    pts_test = sc.score(test)
    gini_sc  = round(2 * roc_auc_score(y_test, pd_test) - 1, 4)

    with open(REPORTS_DIR / "validation_metrics.json") as f:
        gini_champion = json.load(f)["gini"]

    iv_table = [
        {"variable": c, "iv": round(v, 4), "fuerza": iv_strength(v),
         "seleccionada": c in sc.features_}
        for c, v in sc.iv_.items()
    ]
    report = {
        "metodo": "WoE/IV + regresión logística + escala de puntos",
        "escala": {"pdo": PDO, "base_score": BASE_SCORE, "base_odds": f"{BASE_ODDS}:1"},
        "n_variables": len(sc.features_),
        "variables": sc.features_,
        "gini_test": gini_sc,
        "gini_champion_xgboost": gini_champion,
        "gap_vs_champion": round(gini_champion - gini_sc, 4),
        "puntaje_test": {
            "p05": int(np.percentile(pts_test, 5)), "mediana": int(np.median(pts_test)),
            "p95": int(np.percentile(pts_test, 95)),
        },
        "iv": iv_table,
    }
    with open(REPORTS_DIR / "scorecard.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sc.points_.to_csv(REPORTS_DIR / "scorecard_points.csv", index=False)

    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORECARD_DIR / "scorecard.pkl", "wb") as f:
        pickle.dump(sc, f)

    _plot(sc, pts_test, y_test)
    logger.success(
        f"Scorecard OK — {len(sc.features_)} variables | Gini test {gini_sc} "
        f"(XGBoost {gini_champion})"
    )
    return report


def _plot(sc: Scorecard, pts, y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    iv = sc.iv_.head(20)[::-1]
    colors = ["#534AB7" if c in sc.features_ else "#B4B2A9" for c in iv.index]
    axes[0].barh(iv.index, iv.values, color=colors)
    for x, lab in [(0.02, "0.02"), (0.10, "0.10"), (0.30, "0.30")]:
        axes[0].axvline(x, color="#888780", linestyle="--", linewidth=0.8)
    axes[0].set_title("Information Value (violeta = variable seleccionada)")
    axes[0].set_xlabel("IV")

    bins = np.linspace(np.percentile(pts, 0.5), np.percentile(pts, 99.5), 50)
    axes[1].hist(pts[y == 0], bins=bins, alpha=0.6, density=True, label="No default", color="#1D9E75")
    axes[1].hist(pts[y == 1], bins=bins, alpha=0.6, density=True, label="Default", color="#D85A30")
    axes[1].set_title("Distribución del puntaje (test)")
    axes[1].set_xlabel(f"Puntos ({BASE_SCORE} = odds {BASE_ODDS}:1, PDO {PDO})")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scorecard.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    build_scorecard()
