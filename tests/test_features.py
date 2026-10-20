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

    def test_element_fraction_features(self, test_element_list):
        """Test element fraction feature extraction."""
        formula = "LiFePO4"
        features = featurize_one(formula, element_list=test_element_list)
        
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
