"""不确定性校准（split conformal）与结构描述符的测试。

覆盖：
- conformal 分位数的有限样本修正与单调性；
- K 折交叉校准的覆盖率是否接近名义值（可交换数据上应满足保证）；
- 可靠性曲线单调、可信度分级规则；
- ``predictor`` 在有/无校准文件时的区间列行为；
- 结构描述符的维数、立方晶格特例、以及"同种元素单原子"回归（曾出现 inf bug）。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import conformal  # noqa: E402
import predictor as P  # noqa: E402
from structure_features import DESCRIPTOR_NAMES, structure_descriptors  # noqa: E402


class TestConformalQuantile:
    def test_matches_order_statistic(self):
        scores = np.arange(1, 101, dtype=float)          # 1..100
        # ceil((n+1)(1-a)) = ceil(101*0.9) = 91 -> 第 91 个（值 91）
        assert conformal.conformal_quantile(scores, 0.10) == 91.0

    def test_monotonic_in_alpha(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=500)
        q_small = conformal.conformal_quantile(s, 0.01)
        q_mid = conformal.conformal_quantile(s, 0.10)
        q_big = conformal.conformal_quantile(s, 0.50)
        assert q_small >= q_mid >= q_big

    def test_alpha_near_zero_returns_max(self):
        s = np.array([1.0, 2.0, 3.0])
        assert conformal.conformal_quantile(s, 0.0) == pytest.approx(3.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            conformal.conformal_quantile(np.array([]), 0.1)


class TestCoverage:
    def test_coverage_near_nominal_on_exchangeable_data(self):
        rng = np.random.default_rng(42)
        scores = np.abs(rng.normal(0, 1, size=4000))     # 可交换的绝对残差
        for nominal in (0.80, 0.90, 0.95):
            cov = conformal.cross_validated_coverage(scores, 1 - nominal)
            assert abs(cov - nominal) < 0.03, (nominal, cov)

    def test_reliability_curve_is_monotonic(self):
        rng = np.random.default_rng(1)
        scores = np.abs(rng.normal(0, 1, size=2000))
        curve = conformal.reliability_curve(scores, [0.20, 0.10, 0.05, 0.01])
        nominals = [c[0] for c in curve]
        empirical = [c[1] for c in curve]
        assert nominals == sorted(nominals)
        assert empirical == sorted(empirical)


class TestConfidenceTiers:
    def test_tiers(self):
        assert P._confidence(spread=0.01, q=1.0) == "High"
        assert P._confidence(spread=0.5, q=1.0) == "Medium"
        assert P._confidence(spread=0.95, q=1.0) == "Low"

    def test_unknown_without_calibration(self):
        assert P._confidence(spread=0.1, q=0.0) == "Unknown"


class TestPredictorIntervals:
    def _bundle(self):
        class _M:
            def predict(self, X):
                arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, float)
                return arr[:, 0]

        names = ["Z_mean", "EN_mean", "molar_mass"]
        return {"models": {"RF": _M(), "HGB": _M()},
                "feature_names": names, "element_list": ["Li", "Co", "O"]}

    def test_intervals_present_when_calibration_exists(self):
        conf_path = ROOT / "results" / "conformal_voltage.json"
        if not conf_path.exists():
            pytest.skip("未生成 conformal 校准文件")
        out = P.predict_single("LiCoO2", {"voltage": self._bundle()})
        assert out["Voltage_pi90_high"] > out["Voltage_pi90_low"]
        assert out["Voltage_confidence"] in ("High", "Medium", "Low")

    def test_no_interval_without_calibration(self):
        P._load_conformal.cache_clear()
        saved = P.CONFORMAL_FILES["voltage"]
        try:
            P.CONFORMAL_FILES["voltage"] = "__missing__.json"
            P._load_conformal.cache_clear()
            out = P.predict_single("LiCoO2", {"voltage": self._bundle()})
            assert "Voltage_pi90_low" not in out
        finally:
            P.CONFORMAL_FILES["voltage"] = saved
            P._load_conformal.cache_clear()


def _cubic_cell(a: float = 4.0):
    return {
        "lattice": {"matrix": [[a, 0, 0], [0, a, 0], [0, 0, a]]},
        "sites": [
            {"species": [{"element": "Sr", "occu": 1}], "abc": [0.0, 0.0, 0.0]},
            {"species": [{"element": "Ti", "occu": 1}], "abc": [0.5, 0.5, 0.5]},
            {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.5, 0.0]},
            {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.0, 0.5]},
            {"species": [{"element": "O", "occu": 1}], "abc": [0.0, 0.5, 0.5]},
        ],
    }


class TestStructureDescriptors:
    def test_dimension_and_keys(self):
        feats = structure_descriptors(_cubic_cell())
        assert feats is not None
        assert list(feats.keys()) == DESCRIPTOR_NAMES
        assert len(feats) == 21

    def test_cubic_lattice_geometry(self):
        feats = structure_descriptors(_cubic_cell(4.0))
        assert feats["lat_a"] == pytest.approx(4.0)
        assert feats["lat_volume"] == pytest.approx(64.0)
        assert feats["vol_per_atom"] == pytest.approx(64.0 / 5)
        for ang in ("lat_alpha", "lat_beta", "lat_gamma"):
            assert feats[ang] == pytest.approx(90.0)
        assert feats["nsites"] == 5.0

    def test_shortest_distance_is_half_lattice(self):
        feats = structure_descriptors(_cubic_cell(4.0))
        # Ti(0.5,0.5,0.5) 与 O(0.5,0.5,0) 的最小镜像距离 = a/2 = 2.0（最短）
        assert feats["pair_dist_min"] == pytest.approx(2.0, rel=1e-3)
        assert feats["nn_dist_min"] == pytest.approx(2.0, rel=1e-3)
        # 最远的最近邻是 Sr（最近邻为面心 O，距离 a/√2 ≈ 2.828）
        assert feats["nn_dist_max"] == pytest.approx(4.0 / np.sqrt(2), rel=1e-3)

    def test_single_element_cell_has_finite_elem_nn(self):
        """回归：全同元素胞里"其他元素集合"为空，曾算出 inf。"""
        st = {
            "lattice": {"matrix": [[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]]},
            "sites": [
                {"species": [{"element": "Cu", "occu": 1}], "abc": [0.0, 0.0, 0.0]},
                {"species": [{"element": "Cu", "occu": 1}], "abc": [0.5, 0.5, 0.5]},
            ],
        }
        feats = structure_descriptors(st)
        assert np.isfinite(feats["elem_nn_mean"])
        assert np.isfinite(feats["elem_nn_std"])

    def test_invalid_structure_returns_none(self):
        assert structure_descriptors({"sites": []}) is None
        assert structure_descriptors({"lattice": {"matrix": [[1, 0, 0]]}, "sites": []}) is None