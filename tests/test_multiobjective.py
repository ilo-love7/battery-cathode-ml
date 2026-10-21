"""多目标（比容量）与 MP 外部验证的测试（全部离线，不需要 API Key）。

覆盖：
- ``train_capacity`` 是否给底层训练器注入了正确的默认参数；
- ``predictor`` 在模型束包含容量模型时是否输出容量列；
- ``prepare_mp_summary`` 的清洗/去重规则；
- ``validate_mp_external._metrics`` 的「去偏移 MAE / Spearman」计算。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import predictor as P  # noqa: E402
import prepare_mp_summary as pms  # noqa: E402
import train_capacity as tc  # noqa: E402
import validate_mp_external as vme  # noqa: E402
from features import featurize_one  # noqa: E402


class _EchoColumnModel:
    def __init__(self, idx: int = 0):
        self.idx = idx

    def predict(self, X):
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        return arr[:, self.idx]


ELEMENTS = ["Li", "Co", "O", "Fe", "P"]
FEATURE_NAMES = ["Z_mean", "EN_mean", "molar_mass"] + [f"frac_{e}" for e in ELEMENTS]


def _bundle(names, els):
    return {"models": {"RF": _EchoColumnModel(), "HGB": _EchoColumnModel()},
            "feature_names": list(names), "element_list": list(els)}


class TestTrainCapacityDefaults:
    def test_injects_target_and_tag(self, monkeypatch):
        captured = {}

        def fake_main(argv):
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(tc, "_train", fake_main)
        assert tc.main([]) == 0
        argv = captured["argv"]
        assert "--target" in argv and argv[argv.index("--target") + 1] == "capacity_grav"
        assert "--tag" in argv and argv[argv.index("--tag") + 1] == "capacity"
        assert "--data" in argv

    def test_user_args_win(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(tc, "_train",
                            lambda argv: captured.setdefault("argv", list(argv)) or 0)
        tc.main(["--target", "band_gap", "--tag", "custom"])
        argv = captured["argv"]
        assert argv[argv.index("--target") + 1] == "band_gap"
        assert argv[argv.index("--tag") + 1] == "custom"


class TestPredictorCapacity:
    def test_capacity_columns_present(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        df = P.predict_material_properties(
            ["LiCoO2"], models={"capacity": bundle}, verbose=False)
        assert "Capacity_ensemble_mAh_g" in df.columns
        assert "Capacity_spread" in df.columns

    def test_capacity_value_matches_first_feature(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        out = P.predict_single("LiCoO2", {"capacity": bundle})
        expected = featurize_one("LiCoO2", element_list=ELEMENTS)["Z_mean"]
        assert out["Capacity_ensemble_mAh_g"] == pytest.approx(round(expected, 1))

    def test_no_capacity_model_no_column(self):
        df = P.predict_material_properties(["LiCoO2"], models={}, verbose=False)
        assert "Capacity_ensemble_mAh_g" not in df.columns


class TestPrepareMpSummary:
    def test_dedup_keeps_most_stable_phase(self, tmp_path):
        raw = tmp_path / "summary.csv"
        pd.DataFrame([
            dict(material_id="mp-1", formula_pretty="LiCoO2",
                 formation_energy_per_atom=-2.0, band_gap=0.5, energy_above_hull=0.30,
                 nsites=4, is_stable=False, theoretical=True),
            dict(material_id="mp-2", formula_pretty="LiCoO2",
                 formation_energy_per_atom=-2.5, band_gap=0.7, energy_above_hull=0.01,
                 nsites=4, is_stable=True, theoretical=True),
            dict(material_id="mp-3", formula_pretty="Li??O2",
                 formation_energy_per_atom=-1.0, band_gap=1.0, energy_above_hull=0.05,
                 nsites=4, is_stable=False, theoretical=True),
        ]).to_csv(raw, index=False, encoding="utf-8-sig")

        out = tmp_path / "out.csv.gz"
        assert pms.main(["--raw", str(raw), "--out", str(out)]) == 0
        df = pd.read_csv(out)
        assert len(df) == 1                                  # 无法解析的被剔除 + 去重
        assert df.iloc[0]["material_id"] == "mp-2"           # 保留 energy_above_hull 最小
        assert df.iloc[0]["atoms_per_fu"] == pytest.approx(4.0)

    def test_missing_raw_returns_code_2(self, tmp_path):
        code = pms.main(["--raw", str(tmp_path / "nope.csv"),
                         "--out", str(tmp_path / "o.csv.gz")])
        assert code == 2


class TestExternalValidationMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = vme._metrics(y, y)
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert m["R2"] == pytest.approx(1.0)
        assert m["spearman"] == pytest.approx(1.0)
        assert m["MAE_offset_corrected"] == 0.0

    def test_constant_offset_is_removed(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        p = y + 5.0                     # 纯系统性偏移
        m = vme._metrics(y, p)
        assert m["MAE"] == pytest.approx(5.0)
        assert m["bias_median"] == pytest.approx(5.0)
        assert m["MAE_offset_corrected"] == pytest.approx(0.0)
        assert m["spearman"] == pytest.approx(1.0)   # 趋势仍然完美

    def test_anticorrelated_has_negative_rho(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = vme._metrics(y, -y)
        assert m["spearman"] == pytest.approx(-1.0)