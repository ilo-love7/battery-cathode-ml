"""电压 / 容量计算测试（voltage.py）。

数值基准来自教科书常见正极材料：
- LiFePO4 理论比容量 ≈ 170 mAh/g
- LiCoO2  理论比容量 ≈ 274 mAh/g
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from formula import parse_formula  # noqa: E402
from voltage import (avg_voltage_from_formation_energies,  # noqa: E402
                     per_atom_to_per_fu,
                     theoretical_capacity_mAh_g)


class TestTheoreticalCapacity:
    @pytest.mark.parametrize("formula,n_li,expected", [
        ("LiFePO4", 1.0, 169.9),
        ("LiCoO2", 1.0, 273.8),
        ("LiMn2O4", 1.0, 148.3),
    ])
    def test_reference_values(self, formula, n_li, expected):
        q = theoretical_capacity_mAh_g(parse_formula(formula), n_li)
        assert q == pytest.approx(expected, abs=0.6)

    def test_scales_with_n_li(self):
        counts = parse_formula("Li3V2(PO4)3")
        q1 = theoretical_capacity_mAh_g(counts, 1)
        q2 = theoretical_capacity_mAh_g(counts, 2)
        assert q2 == pytest.approx(2 * q1)

    def test_zero_n_li_is_zero_capacity(self):
        assert theoretical_capacity_mAh_g(parse_formula("LiFePO4"), 0) == 0.0


class TestAverageVoltage:
    def test_voltage_is_positive_for_typical_pair(self):
        # 脱锂相形成能较正、嵌锂相较负 -> 电压为正
        v = avg_voltage_from_formation_energies(
            e_form_delith_per_fu=-3.30, e_form_lith_per_fu=-6.40, n_li=1.0)
        assert v == pytest.approx(3.10, abs=1e-6)

    def test_voltage_divides_by_n_li(self):
        v1 = avg_voltage_from_formation_energies(-3.0, -5.0, 1.0)
        v2 = avg_voltage_from_formation_energies(-3.0, -5.0, 2.0)
        assert v2 == pytest.approx(v1 / 2)

    def test_non_positive_n_li_raises(self):
        with pytest.raises(ValueError):
            avg_voltage_from_formation_energies(-3.0, -5.0, 0)


class TestUnitConversion:
    def test_per_atom_to_per_fu(self):
        assert per_atom_to_per_fu(0.5, 5) == pytest.approx(2.5)

    def test_non_positive_sites_raises(self):
        with pytest.raises(ValueError):
            per_atom_to_per_fu(0.5, 0)
