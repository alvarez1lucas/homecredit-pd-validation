"""
Calibración post-hoc del score → PD.

XGBoost se entrena con scale_pos_weight para compensar el desbalance (1:11).
Eso mejora el aprendizaje de la clase minoritaria pero infla la escala de las
probabilidades (score medio ≈ 0.39 vs default real ≈ 0.08). Platt scaling sobre
el logit del score corrige la escala sin cambiar el orden (Gini idéntico).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """PD = sigmoid(a · logit(score) + b), ajustado en una muestra NO usada para entrenar."""

    def fit(self, raw_score, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000)
        self.lr_.fit(_logit(raw_score), np.asarray(y))
        self.slope_     = float(self.lr_.coef_[0, 0])
        self.intercept_ = float(self.lr_.intercept_[0])
        return self

    def transform(self, raw_score):
        return self.lr_.predict_proba(_logit(raw_score))[:, 1]
