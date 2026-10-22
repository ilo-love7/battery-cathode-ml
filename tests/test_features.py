"""Tests for feature engineering functionality."""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import featurize_one, build_universe


class TestFeatureExtraction:
    """Test feature extraction from chemical formulas."""

    def test_basic_feature_extraction(self, sample_formulas):
        """Test basic feature extraction."""
        formula = "LiFePO4"
        features = featurize_one(formula)
        
        assert features is not None
        assert "n_elements" in features
        assert "atoms_per_fu" in features
        assert "molar_mass" in features
        
        # Check specific values
        assert features["n_elements"] == 4
        assert features["atoms_per_fu"] == 7

    def test_element_fraction_features(self, element_list_fixture):
        """Test element fraction feature extraction."""
        formula = "LiFePO4"
        features = featurize_one(formula, element_list=element_list_fixture)
        
        assert features is not None
        assert "frac_Li" in features
        assert "frac_Fe" in features
        assert "frac_P" in features
        assert "frac_O" in features
        
        # Fractions should sum to 1
        frac_keys = [k for k in features.keys() if k.startswith("frac_")]
        total_frac = sum(features[k] for k in frac_keys)
        assert abs(total_frac - 1.0) < 1e-6

    def test_statistical_features(self):
        """Test statistical feature computation."""
        formula = "LiCoO2"
        features = featurize_one(formula)
        
        # Check that all statistical features are present
        required_features = [
            "Z_mean", "Z_std", "Z_max", "Z_min", "Z_range",
            "mass_mean", "mass_std", "mass_max", "mass_min", "mass_range",
            "EN_mean", "EN_std", "EN_max", "EN_min", "EN_range",
            "radius_mean", "radius_std", "radius_max", "radius_min", "radius_range",
            "group_mean", "group_std", "group_max", "group_min", "group_range",
            "period_mean", "period_std", "period_max", "period_min", "period_range",
        ]
        
        for feat in required_features:
            assert feat in features
            assert not np.isnan(features[feat])

    def test_invalid_formula_returns_none(self):
        """Test that invalid formula returns None."""
        result = featurize_one("invalid_formula_xyz")
        assert result is None

    def test_unknown_element_returns_none(self):
        """Test that formula with unknown element returns None."""
        # Use an element that doesn't exist in our database
        result = featurize_one("LiXxO2")
        assert result is None

    def test_complex_formula(self):
        """Test complex formula with multiple brackets and decimals."""
        formula = "LiNi0.5Mn0.3Co0.2O2"
        features = featurize_one(formula)
        
        assert features is not None
        assert features["n_elements"] == 5
        assert features["atoms_per_fu"] == 4.0

    def test_build_universe(self, sample_formulas):
        """Test building element universe from formulas."""
        universe = build_universe(sample_formulas)
        
        assert isinstance(universe, list)
        assert len(universe) > 0
        assert "Li" in universe
        assert "O" in universe

    def test_missing_property_renormalizes_weights(self):
        """缺失属性（稀有气体电负性为 None）不能稀释加权均值。

        回归：旧实现用 nansum(w*vals)，LiHeO2 的 EN_mean 被拉低到 1.97，
        正确值（只对 Li/O 重新归一化权重）是 2.62。
        """
        features = featurize_one("LiHeO2")
        # w: Li=0.25, O=0.5 (He 权重 0.25 被剔除) -> (0.25*0.98+0.5*3.44)/0.75
        assert features["EN_mean"] == pytest.approx(2.62, abs=0.01)
        # max/min/range 不受缺失值影响
        assert features["EN_max"] == pytest.approx(3.44)
        assert features["EN_min"] == pytest.approx(0.98)
        assert features["EN_range"] == pytest.approx(3.44 - 0.98)

    def test_no_nan_weights_leak_into_stats(self):
        """含 NaN 属性的成分，其统计量不应是 NaN（除非该属性全缺）。"""
        features = featurize_one("Ba5(AsO4)3(OH,F)")
        for name in ("EN_mean", "EN_std", "radius_mean", "Z_mean"):
            assert features[name] == features[name]  # not NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
