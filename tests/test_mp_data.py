"""Materials Project 数据链路测试（不需要网络/API Key）。

覆盖：
- ``prepare_mp_data`` 的清洗规则（电压范围、化学式可解析、未知元素剔除）；
- 容量交叉校验列是否按预期落盘；
- ``fetch_mp_battery.resolve_key`` 的 key 读取优先级。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import prepare_mp_data as prep  # noqa: E402
from fetch_mp_battery import resolve_key  # noqa: E402


def _raw_fixture(tmp_path: Path) -> Path:
    """构造一份含各类脏数据的原始电极表。"""
    rows = [
        # 合法行：Li2FePO4F，电压 4.0 V
        dict(average_voltage=4.0, formula_discharge="Li2FePO4F",
             capacity_grav=291.8, framework_formula="FePO4F", battery_type="insertion",
             battery_formula="Li0-2FePO4F"),
        # 负电压：应被剔除
        dict(average_voltage=-1.5, formula_discharge="LiCoO2",
             capacity_grav=274.0, framework_formula="CoO2", battery_type="insertion",
             battery_formula="Li0-1CoO2"),
        # 超范围电压：应被剔除
        dict(average_voltage=9.9, formula_discharge="LiMn2O4",
             capacity_grav=148.0, framework_formula="Mn2O4", battery_type="insertion",
             battery_formula="Li0-1Mn2O4"),
        # 化学式无法解析：应被剔除
        dict(average_voltage=3.5, formula_discharge="Li??O2",
             capacity_grav=200.0, framework_formula="?O2", battery_type="insertion",
             battery_formula="Li0-1?O2"),
        # 元素不在内置表：应被剔除
        dict(average_voltage=3.5, formula_discharge="LiZzO2",
             capacity_grav=200.0, framework_formula="ZzO2", battery_type="insertion",
             battery_formula="Li0-1ZzO2"),
    ]
    path = tmp_path / "raw_electrodes.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


class TestPrepareMpData:
    def test_filters_dirty_rows(self, tmp_path):
        raw = _raw_fixture(tmp_path)
        out = tmp_path / "mp_voltage.csv"
        assert prep.main(["--raw", str(raw), "--out", str(out)]) == 0
        df = pd.read_csv(out, encoding="utf-8-sig")
        assert list(df["formula"]) == ["Li2FePO4F"]

    def test_expected_columns_and_values(self, tmp_path):
        raw = _raw_fixture(tmp_path)
        out = tmp_path / "mp_voltage.csv"
        prep.main(["--raw", str(raw), "--out", str(out)])
        df = pd.read_csv(out, encoding="utf-8-sig")
        for col in ("formula", "formula_canon", "average_voltage_V", "capacity_grav",
                    "capacity_calc", "n_li", "molar_mass", "atoms_per_fu",
                    "framework_formula", "battery_type"):
            assert col in df.columns
        row = df.iloc[0]
        assert row["n_li"] == pytest.approx(2.0)
        assert row["atoms_per_fu"] == pytest.approx(9.0)   # Li2 Fe P O4 F
        # Q = n·F/3.6/M，应与 MP 的 capacity_grav 同量级
        assert row["capacity_calc"] == pytest.approx(row["capacity_grav"], rel=0.05)

    def test_missing_raw_file_returns_code_2(self, tmp_path):
        code = prep.main(["--raw", str(tmp_path / "nope.csv"),
                          "--out", str(tmp_path / "out.csv")])
        assert code == 2


class TestResolveKey:
    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "from-env")
        assert resolve_key("from-arg") == "from-arg"

    def test_env_var_used(self, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "from-env")
        assert resolve_key(None) == "from-env"

    def test_missing_everywhere_exits(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() 指向空目录
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(SystemExit):
            resolve_key(None)