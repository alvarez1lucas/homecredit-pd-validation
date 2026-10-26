"""
Tests del pipeline — Home Credit Default Risk
=============================================
Correr con: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from src.config import breakeven_pd
from src.data.ingest import add_derived_features
from src.governance.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds,
)
from src.models.calibration import PlattCalibrator
from src.models.scorecard import NumericBinner, Scorecard
from src.validation.master_scale import (
    assign_grades,
    jeffreys_pvalue,
    master_scale_table,
)
from src.validation.model_validation import (
    compute_ks,
    compute_psi,
    expected_calibration_error,
    gini,
    hosmer_lemeshow,
)


@pytest.fixture
def good_predictions():
    rng = np.random.default_rng(42)
    y_true  = np.array([0] * 200 + [1] * 200)
    y_score = np.concatenate([rng.beta(2, 5, 200), rng.beta(5, 2, 200)])
    return y_true, y_score


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, 500), rng.uniform(0, 1, 500)


@pytest.fixture
def calibrated_sample():
    """PDs verdaderas conocidas y defaults simulados a partir de ellas."""
    rng = np.random.default_rng(0)
    pd_true = rng.beta(1.2, 12, 50_000)
    y = rng.binomial(1, pd_true)
    return y, pd_true


class TestDiscriminatoryPower:
    def test_gini_good_model(self, good_predictions):
        y, s = good_predictions
        assert gini(y, s) > 0.40

    def test_gini_random_model(self, random_predictions):
        y, s = random_predictions
        assert abs(gini(y, s)) < 0.15

    def test_ks_range(self, good_predictions):
        y, s = good_predictions
        ks = compute_ks(y, s)
        assert 0 <= ks["ks_statistic"] <= 1
        assert 0 <= ks["ks_pvalue"] <= 1


class TestCalibration:
    def test_hl_output_keys(self, good_predictions):
        hl = hosmer_lemeshow(*good_predictions)
        assert {"hl_statistic", "hl_pvalue", "well_calibrated"} <= hl.keys()

    def test_ece_low_for_calibrated(self, calibrated_sample):
        y, p = calibrated_sample
        assert expected_calibration_error(y, p) < 0.01

    def test_ece_high_for_inflated(self, calibrated_sample):
        y, p = calibrated_sample
        assert expected_calibration_error(y, np.clip(p * 4, 0, 1)) > 0.10

    def test_platt_fixes_scale_and_keeps_ranking(self, calibrated_sample):
        y, p = calibrated_sample
        inflated = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + 1.5)))   # sesgo tipo scale_pos_weight
        cal = PlattCalibrator().fit(inflated[:25_000], y[:25_000])
        out = cal.transform(inflated[25_000:])
        assert abs(out.mean() - y[25_000:].mean()) < 0.01
        assert gini(y[25_000:], out) == gini(y[25_000:], inflated[25_000:])


class TestMasterScale:
    def test_assign_grades_boundaries(self):
        grades = assign_grades([0.01, 0.02, 0.05, 0.5], [0.02, 0.04], ["A", "B", "C"])
        assert list(grades) == ["A", "B", "C", "C"]

    def test_jeffreys_detects_underestimation(self):
        # 100 defaults en 1000 con PD asignada del 5% → subestimación clara
        assert jeffreys_pvalue(100, 1000, 0.05) < 0.01
        # 50 defaults en 1000 con PD 5% → consistente
        assert jeffreys_pvalue(50, 1000, 0.05) > 0.05

    def test_false_red_rate_when_calibrated(self):
        # Con PDs bien calibradas, el rojo (p < 1%) debería aparecer ~1% de las veces por grado
        reds, total = 0, 0
        for seed in range(40):
            rng = np.random.default_rng(seed)
            p = rng.beta(1.2, 12, 20_000)
            t = master_scale_table(rng.binomial(1, p), p, [0.02, 0.05, 0.10, 0.20], list("ABCDE"))
            reds += (t["result"] == "subestima").sum()
            total += len(t)
        assert reds / total < 0.04

    def test_table_flags_underestimation(self, calibrated_sample):
        y, p = calibrated_sample
        t = master_scale_table(y, p * 0.6, [0.02, 0.05, 0.10, 0.20], list("ABCDE"))
        assert (t["result"] == "subestima").sum() >= 3
        assert np.all(np.diff(t["default_rate"]) > 0)


class TestStability:
    def test_psi_identical(self):
        d = np.random.default_rng(1).normal(0, 1, 1000)
        assert compute_psi(d, d) < 0.01

    def test_psi_different(self):
        rng = np.random.default_rng(1)
        assert compute_psi(rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)) > 0.25


class TestFairness:
    def test_dpd_equal_groups(self):
        y_app = np.array([1] * 100 + [0] * 100 + [1] * 100 + [0] * 100)
        group = np.array(["M"] * 200 + ["F"] * 200)
        assert abs(demographic_parity_difference(y_app, group)) < 0.05

    def test_dpd_multigroup_is_max_minus_min(self):
        y_app = np.array([1] * 90 + [0] * 10 + [1] * 50 + [0] * 50 + [1] * 70 + [0] * 30)
        group = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
        assert demographic_parity_difference(y_app, group) == pytest.approx(0.40)

    def test_dir_unequal_groups(self):
        y_app = np.array([1] * 160 + [0] * 40 + [1] * 40 + [0] * 160)
        group = np.array(["M"] * 200 + ["F"] * 200)
        assert disparate_impact_ratio(y_app, group) < 0.80

    def test_equalized_odds_keys(self):
        rng = np.random.default_rng(42)
        eo = equalized_odds(rng.integers(0, 2, 300), rng.integers(0, 2, 300),
                            np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100))
        assert {"A", "B", "C", "tpr_gap", "fpr_gap"} <= eo.keys()


class TestPolicyAndFeatures:
    def test_breakeven_pd(self):
        # ganancia esperada nula en el cutoff: (1 - PD) * r = PD * LGD
        r, lgd = 0.115, 0.45
        p = breakeven_pd(r, lgd)
        assert (1 - p) * r == pytest.approx(p * lgd)

    def test_derived_features(self):
        df = pd.DataFrame({"DAYS_BIRTH": [-36500], "DAYS_EMPLOYED": [365243],
                           "AMT_INCOME_TOTAL": [100.0], "AMT_CREDIT": [300.0],
                           "AMT_ANNUITY": [20.0], "AMT_GOODS_PRICE": [250.0]})
        out = add_derived_features(df)
        assert out["age_years"].iloc[0] == 100.0
        assert np.isnan(out["DAYS_EMPLOYED"].iloc[0])
        assert out["credit_income_ratio"].iloc[0] == 3.0
        assert out["credit_goods_ratio"].iloc[0] == 1.2


class TestScorecard:
    @pytest.fixture
    def synthetic(self):
        rng = np.random.default_rng(7)
        n = 20_000
        x1 = rng.normal(0, 1, n)
        x2 = rng.choice(["a", "b", "c"], n)
        logit = -2.5 + 1.0 * x1 + np.select([x2 == "a", x2 == "b"], [0.8, 0.0], -0.8)
        y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))))
        x1[rng.random(n) < 0.05] = np.nan
        return pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.normal(0, 1, n)}), y

    def test_binner_monotonic(self, synthetic):
        df, y = synthetic
        b = NumericBinner().fit(df["x1"], y)
        br = b.table_.drop(index="MISSING")["bad_rate"].values
        assert np.all(np.diff(br) >= 0) or np.all(np.diff(br) <= 0)

    def test_scorecard_selects_signal_and_scores(self, synthetic):
        df, y = synthetic
        sc = Scorecard().fit(df, y, numeric=["x1", "noise"], categorical=["x2"])
        assert "x1" in sc.features_ and "x2" in sc.features_
        assert "noise" not in sc.features_
        pts = sc.score(df)
        # mayor puntaje = menor riesgo
        assert gini(y, -pts) > 0.4


class TestMarginOfConservatism:
    def test_grade_pd_is_conservative_and_monotone(self, calibrated_sample):
        from src.models.moc import estimate_grade_pd
        y, p = calibrated_sample
        t = estimate_grade_pd(y, p * 0.8, [0.02, 0.05, 0.10, 0.20], list("ABCDE"))
        pds = [r["pd_grade"] for r in t.values()]
        assert all(r["pd_grade"] >= r["pd_avg"] for r in t.values())
        assert all(np.diff(pds) >= 0)

    def test_moc_fixes_underestimation_out_of_sample(self):
        from src.models.moc import estimate_grade_pd, grade_pd
        rng = np.random.default_rng(3)
        limits, grades = [0.02, 0.05, 0.10, 0.20], list("ABCDE")
        def sample(n):
            p_true = rng.beta(1.2, 12, n)
            return rng.binomial(1, p_true), p_true * 0.8          # modelo que subestima 20%
        y_est, p_est = sample(40_000)
        y_new, p_new = sample(40_000)
        table = estimate_grade_pd(y_est, p_est, limits, grades)
        before = master_scale_table(y_new, p_new, limits, grades)
        after = master_scale_table(y_new, grade_pd(p_new, table, limits, grades), limits, grades, pd_rank=p_new)
        assert (before["result"] == "subestima").sum() >= 3
        assert (after["result"] == "subestima").sum() == 0
