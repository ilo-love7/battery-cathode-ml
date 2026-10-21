"""预测管线回归测试。

重点守护一个曾在 1.0.0 出现的静默 bug：训练集特征列顺序是
[统计特征..., 元素分数...]，而旧的 predictor.py 用 [元素分数..., 统计特征...]
拼矩阵后又 ``to_numpy()`` 绕过列名校验，导致**任何化学式都预测出同一个值**。
这里的桩模型只回传"第一列特征值"，能把列顺序错误直接暴露出来。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import predictor as P  # noqa: E402
from features import featurize_one  # noqa: E402


class _EchoColumnModel:
    """返回第 idx 列特征的桩模型（模拟 sklearn 的 predict 接口）。"""

    def __init__(self, idx: int = 0):
        self.idx = idx

    def predict(self, X):
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        return arr[:, self.idx]


def _bundle(feature_names, element_list):
    return {
        "models": {"RF": _EchoColumnModel(0), "HGB": _EchoColumnModel(0)},
        "feature_names": list(feature_names),
        "element_list": list(element_list),
    }


ELEMENTS = ["Li", "Co", "O", "Fe", "P"]
FEATURE_NAMES = ["Z_mean", "EN_mean", "molar_mass"] + [f"frac_{e}" for e in ELEMENTS]


class TestFeatureFrameAlignment:
    def test_columns_follow_bundle_order(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        X = P._feature_frame("LiCoO2", bundle)
        assert list(X.columns) == FEATURE_NAMES

    def test_values_match_featurize_one(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        X = P._feature_frame("LiCoO2", bundle)
        feats = featurize_one("LiCoO2", element_list=ELEMENTS)
        for name in FEATURE_NAMES:
            assert X.iloc[0][name] == pytest.approx(feats[name])

    def test_missing_feature_names_raises(self):
        bundle = {"models": {"RF": _EchoColumnModel(), "HGB": _EchoColumnModel()},
                  "element_list": ELEMENTS}
        with pytest.raises(ValueError, match="feature_names"):
            P._feature_frame("LiCoO2", bundle)

    def test_unknown_formula_returns_none(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        assert P._feature_frame("Xx2O3", bundle) is None


class TestPredictSingle:
    def test_prediction_uses_first_feature_in_order(self):
        """桩模型回传第一列；预测值必须等于 Z_mean，而不是 frac_*。"""
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        out = P.predict_single("LiCoO2", {"gap": bundle})
        expected = featurize_one("LiCoO2", element_list=ELEMENTS)["Z_mean"]
        assert out["Egap_ensemble_eV"] == pytest.approx(round(expected, 4))

    def test_different_formulas_give_different_predictions(self):
        """旧 bug 的症状：不同化学式得到完全相同的结果。"""
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        a = P.predict_single("LiCoO2", {"gap": bundle})["Egap_ensemble_eV"]
        b = P.predict_single("LiFePO4", {"gap": bundle})["Egap_ensemble_eV"]
        assert a != b

    def test_parse_error_reported(self):
        out = P.predict_single("not-a-formula", {})
        assert "parse_error" in out

    def test_molar_mass_and_atom_count(self):
        out = P.predict_single("LiCoO2", {})
        assert out["n_elements"] == 3
        assert out["total_atoms"] == 4.0
        assert out["molar_mass_g_mol"] == pytest.approx(97.87, abs=0.2)


class TestBatchPredict:
    def test_dataframe_shape(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        df = P.predict_material_properties(
            ["LiCoO2", "LiFePO4"], models={"gap": bundle}, verbose=False)
        assert len(df) == 2
        assert "Egap_ensemble_eV" in df.columns
        assert df["Egap_ensemble_eV"].nunique() == 2

    def test_accepts_single_string(self):
        bundle = _bundle(FEATURE_NAMES, ELEMENTS)
        df = P.predict_material_properties("LiCoO2", models={"gap": bundle}, verbose=False)
        assert len(df) == 1

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            P.predict_material_properties([], models={}, verbose=False)
