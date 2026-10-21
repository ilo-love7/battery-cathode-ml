"""Tests for formula parsing functionality."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from formula import parse_formula, canonical_formula


class TestFormulaParsing:
    """Test chemical formula parsing."""

    def test_simple_formula(self):
        """Test parsing of simple chemical formulas."""
        result = parse_formula("LiFePO4")
        assert result == {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0}

    def test_decimal_formula(self):
        """Test parsing of formulas with decimal subscripts."""
        result = parse_formula("LiNi0.33Co0.33Mn0.34O2")
        assert abs(result["Li"] - 1.0) < 1e-6
        assert abs(result["Ni"] - 0.33) < 1e-6
        assert abs(result["Co"] - 0.33) < 1e-6
        assert abs(result["Mn"] - 0.34) < 1e-6
        assert result["O"] == 2.0

    def test_nested_brackets(self):
        """Test parsing of formulas with nested brackets."""
        result = parse_formula("Li3V2(PO4)3")
        assert result == {"Li": 3.0, "V": 2.0, "P": 3.0, "O": 12.0}

    def test_alternative_notation(self):
        """Test parsing with alternative bracket notations."""
        # Test square brackets
        result = parse_formula("Li[Mn2]O4")
        assert result == {"Li": 1.0, "Mn": 2.0, "O": 4.0}

        # Test curly braces
        result = parse_formula("Li{Co}O2")
        assert result == {"Li": 1.0, "Co": 1.0, "O": 2.0}

    def test_empty_formula_raises_error(self):
        """Test that empty formula raises ValueError."""
        with pytest.raises(ValueError):
            parse_formula("")

    def test_invalid_formula_raises_error(self):
        """Test that invalid formula raises ValueError."""
        with pytest.raises(ValueError):
            parse_formula("invalid")

    def test_canonical_formulation(self):
        """Test conversion to canonical formula string."""
        counts = {"Li": 1.0, "Co": 1.0, "O": 2.0}
        result = canonical_formula(counts)
        assert result == "LiCoO2" or result == "CoLiO2"  # Order may vary

    def test_whitespace_handling(self):
        """Test handling of whitespace in formulas."""
        result = parse_formula(" LiFePO4 ")
        assert result == {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0}

    def test_charge_notation(self):
        """电荷后缀是离子电荷，不能被当成化学计量数。"""
        # Fe³⁺ -> 1 个 Fe（旧实现会误判为 Fe3）
        assert parse_formula("Fe3+") == {"Fe": 1.0}
        # O²⁻ -> 1 个 O（旧实现会误判为 O2）
        assert parse_formula("O2-") == {"O": 1.0}
        # 带 ^ 记法
        assert parse_formula("SO4^2-") == {"S": 1.0, "O": 4.0}


class TestCanonicalRoundtrip:
    """canonical_formula 的输出必须能被 parse_formula 原样解析回来。"""

    @pytest.mark.parametrize("formula", [
        "LiFePO4",
        "Li3V2(PO4)3",
        "LiNi0.8Co0.1Mn0.1O2",
        "Ba5(AsO4)3(OH,F)",
        "Li2FeSiO4",
        "LiNi0.33Co0.33Mn0.34O2",
    ])
    def test_roundtrip_preserves_composition(self, formula):
        counts = parse_formula(formula)
        again = parse_formula(canonical_formula(counts))
        assert set(again) == set(counts)
        for element, amount in counts.items():
            assert again[element] == pytest.approx(amount, abs=1e-6)
