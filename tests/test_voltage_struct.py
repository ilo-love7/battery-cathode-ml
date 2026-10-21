"""电压侧结构消歧的数据准备与 CLI 测试（离线、无 torch）。

覆盖：
- ``prepare_voltage_struct`` 能把（电压记录 + 结构库）拼成结构特征表，
  正确处理"只有单相结构"的记录，并算出 delta_vol / delta_natom；
- 缺文件时给出可执行的错误码与提示；
- 新增的 argparse CLI 都能被发现（由 test_cgcnn 的自动发现测试覆盖 --help）。
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import prepare_voltage_struct as pvs  # noqa: E402


def _struct(a: float, species_positions):
    return {
        "lattice": {"matrix": [[a, 0, 0], [0, a, 0], [0, 0, a]]},
        "sites": [{"species": [{"element": el, "occu": 1.0}], "abc": list(pos)}
                  for el, pos in species_positions],
    }


def _fixtures(tmp_path: Path):
    """构造 2 条电压记录：一条两相齐全、一条只有嵌锂相。"""
    lith = _struct(4.0, [("Li", (0, 0, 0)), ("Co", (0.5, 0.5, 0.5)), ("O", (0.5, 0.5, 0))])
    frame = _struct(3.8, [("Co", (0, 0, 0)), ("O", (0.5, 0.5, 0.5))])
    volt = pd.DataFrame([
        {"formula": "LiCoO2", "formula_canon": "CoLiO2", "average_voltage_V": 3.9,
         "capacity_grav": 274.0, "n_li": 1.0, "molar_mass": 97.87, "atoms_per_fu": 4.0,
         "framework_formula": "CoO2", "id_charge": "mp-frame", "id_discharge": "mp-lith",
         "max_delta_volume": 0.05},
        {"formula": "LiFePO4", "formula_canon": "FeLiO4P", "average_voltage_V": 3.45,
         "capacity_grav": 170.0, "n_li": 1.0, "molar_mass": 157.8, "atoms_per_fu": 7.0,
         "framework_formula": "FePO4", "id_charge": "mp-missing", "id_discharge": "mp-lith",
         "max_delta_volume": 0.03},
    ])
    vpath = tmp_path / "mp_voltage.csv"
    volt.to_csv(vpath, index=False, encoding="utf-8-sig")

    structures = [
        {"material_id": "mp-lith", "structure": lith},
        {"material_id": "mp-frame", "structure": frame},
    ]
    spath = tmp_path / "structures.json.gz"
    with gzip.open(spath, "wt", encoding="utf-8") as f:
        json.dump(structures, f)
    return vpath, spath


class TestPrepareVoltageStruct:
    def test_builds_struct_features(self, tmp_path):
        vpath, spath = _fixtures(tmp_path)
        out = tmp_path / "out.csv.gz"
        assert pvs.main(["--voltage", str(vpath), "--structures", str(spath),
                         "--out", str(out)]) == 0
        df = pd.read_csv(out)

        assert len(df) == 2
        for col in ("lith_lat_volume", "frame_lat_volume", "delta_vol", "delta_natom",
                    "has_both_structures", "average_voltage_V", "formula_canon"):
            assert col in df.columns
        assert "lith_" + "nn_dist_mean" in df.columns

        both = df[df["has_both_structures"]]
        assert len(both) == 1
        row = both.iloc[0]
        # 4.0^3 = 64，3.8^3 = 54.872 -> delta_vol ≈ -0.1426
        assert row["lith_lat_volume"] == pytest.approx(64.0)
        assert row["frame_lat_volume"] == pytest.approx(3.8 ** 3)
        assert row["delta_vol"] == pytest.approx((3.8 ** 3) / 64.0 - 1.0, abs=1e-6)
        assert row["delta_natom"] == pytest.approx(3.0 - 2.0)

        single = df[~df["has_both_structures"]].iloc[0]
        assert single["has_both_structures"] in (False, np.False_)
        assert np.isnan(single["delta_vol"])          # 单相无法算体积变化
        assert not np.isnan(single["lith_lat_volume"])

    def test_missing_structures_file_returns_2(self, tmp_path):
        vpath, _ = _fixtures(tmp_path)
        code = pvs.main(["--voltage", str(vpath),
                         "--structures", str(tmp_path / "nope.json.gz"),
                         "--out", str(tmp_path / "o.csv.gz")])
        assert code == 2

    def test_missing_voltage_file_returns_2(self, tmp_path):
        _, spath = _fixtures(tmp_path)
        code = pvs.main(["--voltage", str(tmp_path / "nope.csv"),
                         "--structures", str(spath),
                         "--out", str(tmp_path / "o.csv.gz")])
        assert code == 2


class TestVoltageStructTrainingScript:
    def test_cli_help_available(self):
        import subprocess

        proc = subprocess.run([sys.executable, str(ROOT / "src" / "train_voltage_struct.py"),
                               "--help"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60)
        assert proc.returncode == 0
        assert "--subset" in proc.stdout and "--skip-group" in proc.stdout

    def test_committed_struct_table_is_usable(self):
        """随仓库分发的 mp_voltage_struct.csv.gz 应可直接训练（列齐全）。"""
        path = ROOT / "data" / "processed" / "mp_voltage_struct.csv.gz"
        if not path.exists():
            pytest.skip("未生成音量结构表")
        df = pd.read_csv(path)
        assert len(df) > 1000
        assert {"formula", "formula_canon", "average_voltage_V", "delta_vol",
                "lith_lat_volume", "frame_lat_volume"} <= set(df.columns)
        assert df["has_both_structures"].sum() > 0.9 * len(df)