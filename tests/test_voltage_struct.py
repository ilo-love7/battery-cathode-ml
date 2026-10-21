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

class TestDemoVoltagePairs:
    """双塔电压 CLI 用的 demo 电极对文件：schema + 划分标注（离线可检查）。"""

    PAIRS = ROOT / "data" / "raw" / "mp_voltage_pairs_demo.json.gz"

    def test_file_exists_with_required_keys(self):
        if not self.PAIRS.exists():
            pytest.skip("未随仓库提供 demo 电极对文件")
        rows = json.load(gzip.open(self.PAIRS, "rt", encoding="utf-8"))
        assert len(rows) >= 5
        required = {"query_formula", "formula_charge", "formula_discharge",
                    "average_voltage_true_V", "id_charge", "id_discharge",
                    "structure_charge", "structure_discharge", "dataset_split"}
        for r in rows:
            assert required <= set(r), sorted(required - set(r))
            assert r["structure_charge"].get("sites") and r["structure_discharge"].get("sites")

    def test_split_labels_are_honest(self):
        """至少要有一条 held-out（test/unseen）样本，否则 demo 会拿训练样本冒充精度。"""
        if not self.PAIRS.exists():
            pytest.skip("未随仓库提供 demo 电极对文件")
        rows = json.load(gzip.open(self.PAIRS, "rt", encoding="utf-8"))
        labels = {r["dataset_split"] for r in rows}
        assert labels <= {"train", "val", "test", "unseen", "unknown"}
        assert labels & {"test", "unseen"}, f"没有 held-out 样本: {labels}"


class TestStructDrivenScreening:
    """结构驱动筛选的产物契约（离线检查，不需要 torch）。"""

    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"
    FALLBACK = ROOT / "results" / "cathode_ranking_struct_fallback.csv"

    def test_metrics_exist_and_account_for_all_candidates(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["n_candidates"] == 40
        assert m["n_ranked_with_structure"] + m["n_fallback_without_structure"] == 40
        assert m["weights"]["stability"] == 0.40
        # 尺度无关的一致性指标必须存在（两套口径不可比绝对值）
        assert "stability_rank_correlation_struct_vs_comp" in m

    def test_main_ranking_only_contains_structured_candidates(self):
        if not (self.RANKED.exists() and self.FALLBACK.exists()):
            pytest.skip("未生成结构筛选排序表")
        ranked = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        fallback = pd.read_csv(self.FALLBACK, encoding="utf-8-sig")
        assert len(ranked) and len(fallback)
        # 主排名里必须有 CGCNN 结构稳定性；回退表里必须没有
        assert ranked["cgcnn_ef_eV_atom"].notna().all()
        assert fallback["cgcnn_ef_eV_atom"].isna().all()
        # 两个表不能有重复候选
        assert not (set(ranked["formula"]) & set(fallback["formula"]))
        # 主排名按分数降序
        assert ranked["score"].is_monotonic_decreasing


class TestStructGapModel:
    """结构版带隙模型（CGCNN on MP band_gap）的产物契约与口径决策（离线）。"""

    METRICS = ROOT / "results" / "metrics_cgcnn_gap_8k.json"
    SCREEN = ROOT / "results" / "metrics_screening_struct.json"

    def test_gap_metrics_have_specialised_fields(self):
        if not self.METRICS.exists():
            pytest.skip("未训练结构带隙模型")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["target"] == "band_gap"
        assert m["transform"] == "log1p"          # 带隙右偏，必须用 log1p
        for split in ("train", "val", "test"):
            assert "MAE_eV_atom" in m[split]
        # 带隙专用指标：半导体子集 MAE + 金属/非金属判对率
        assert "MAE_semiconductor_eV" in m["test"]
        assert 0.0 <= m["test"]["metal_class_accuracy"] <= 1.0
        assert m["composition_baseline"]["MAE"] > 0

    def test_screening_records_gap_scoring_source(self):
        if not self.SCREEN.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.SCREEN, encoding="utf-8"))
        assert "gap_scoring_source" in m
        assert "struct_gap_coverage_among_ranked" in m
        # PBE 低估导致结构/实验口径排序负相关，这是选择评分口径的依据
        rho = m.get("gap_rank_correlation_struct_vs_comp")
        assert rho is None or rho < 0.5, f"若结构带隙与实验口径强相关，应重新评估口径决策: {rho}"
