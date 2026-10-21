"""元素性质表的完整性测试。

element_data.ELEMENT_ROWS 是全部特征工程的唯一数据源，一旦有重复/缺失/错位，
所有模型都会静默退化，所以这里做硬性约束检查。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from element_data import ELEMENT_ROWS  # noqa: E402
from features import ELEMENTS  # noqa: E402


class TestElementTable:
    def test_covers_all_118_elements(self):
        assert len(ELEMENT_ROWS) == 118

    def test_atomic_numbers_are_complete_and_unique(self):
        zs = [row[0] for row in ELEMENT_ROWS]
        assert zs == list(range(1, 119))

    def test_symbols_unique(self):
        syms = [row[1] for row in ELEMENT_ROWS]
        assert len(set(syms)) == 118

    @pytest.mark.parametrize("symbol,mass,en", [
        ("H", 1.008, 2.20),
        ("Li", 6.94, 0.98),
        ("O", 15.999, 3.44),
        ("Fe", 55.845, 1.83),
        ("Co", 58.933, 1.88),
        ("Ni", 58.693, 1.91),
    ])
    def test_known_reference_values(self, symbol, mass, en):
        assert ELEMENTS[symbol]["mass"] == pytest.approx(mass, abs=1e-3)
        assert ELEMENTS[symbol]["EN"] == pytest.approx(en, abs=1e-2)

    def test_noble_gases_have_no_electronegativity(self):
        for sym in ("He", "Ne", "Ar"):
            assert ELEMENTS[sym]["EN"] != ELEMENTS[sym]["EN"]  # NaN

    def test_ranges_are_physical(self):
        for _z, sym, mass, en, radius, group, period in ELEMENT_ROWS:
            assert 1.0 <= mass <= 300.0, sym
            assert radius > 0, sym
            assert 0 <= group <= 18, sym
            assert 1 <= period <= 7, sym
            if en is not None:
                assert 0.7 <= en <= 4.0, sym


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
