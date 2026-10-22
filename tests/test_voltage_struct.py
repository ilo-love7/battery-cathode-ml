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
    """结构驱动筛选的产物契约（离线检查，不需要 torch）。

    v1.11.0 起候选分三个置信层级：A=MP 弛豫结构（主排名）、B=原型结构（组内可比）、
    C=无结构（成分回退）。三者互斥且并集=全部候选。
    """

    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"
    PROTOTYPE = ROOT / "results" / "cathode_ranking_struct_prototype.csv"
    FALLBACK = ROOT / "results" / "cathode_ranking_struct_fallback.csv"

    def test_metrics_account_for_every_candidate_in_three_tiers(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["n_candidates"] == 40
        tiers = m["structure_tiers"]
        assert sum(tiers.values()) == 40
        assert tiers["A_MP_relaxed"] == m["n_ranked_with_structure"]
        assert tiers["B_prototype_CHGNet"] == m["n_prototype_only"]
        assert tiers["C_no_structure"] == m["n_fallback_without_structure"]
        assert m["weights"]["stability"] == 0.40
        # 尺度无关的一致性指标必须存在（两套口径不可比绝对值）
        assert "stability_rank_correlation_struct_vs_comp" in m

    def test_tier_tables_are_disjoint_and_schema_consistent(self):
        if not self.RANKED.exists():
            pytest.skip("未生成结构筛选排序表")
        ranked = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        assert len(ranked)
        # A 层：必须都有 MP 结构，且分数降序
        assert ranked["cgcnn_ef_eV_atom"].notna().all()
        assert (ranked["structure_origin"] == "MP弛豫").all()
        assert ranked["score"].is_monotonic_decreasing

        sets = [set(ranked["formula"])]
        if self.PROTOTYPE.exists():
            proto = pd.read_csv(self.PROTOTYPE, encoding="utf-8-sig")
            if len(proto):
                assert proto["cgcnn_ef_eV_atom"].notna().all()
                assert proto["structure_from_prototype"].all()
                assert proto["score"].is_monotonic_decreasing
                sets.append(set(proto["formula"]))
        if self.FALLBACK.exists():
            fb = pd.read_csv(self.FALLBACK, encoding="utf-8-sig")
            if len(fb):
                assert fb["cgcnn_ef_eV_atom"].isna().all()
                sets.append(set(fb["formula"]))
        # 层级之间不得重复
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert not (sets[i] & sets[j]), "不同置信层级出现重复候选"

    def test_prototype_tier_reports_extra_uncertainty(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        if m["n_prototype_only"] == 0:
            pytest.skip("本次没有原型层候选")
        unc = m["prototype_extra_uncertainty"]
        assert unc["ef_bias_eV_atom"] is not None
        assert abs(unc["ef_bias_eV_atom"]) > 0.5, (
            "若原型与 MP 结构预测几乎无偏差，应重新评估分层必要性")


class TestPrototypeStructures:
    """原型结构生成器（需要 pymatgen；CI 未安装则跳过）。"""

    def test_prototype_is_graph_ready_and_composition_close(self):
        pytest.importorskip("pymatgen")
        sys.path.insert(0, str(ROOT / "src"))
        import generate_prototype_structures as gps
        from cgcnn_data import build_graph

        st, info = gps.rocksalt_prototype("LiNi0.5Co0.2Mn0.3O2", a=4.1, supercell=2)
        assert info["decorated"] is True
        assert info["composition_deviation"] < 0.05      # 整数占位带来的微小计量偏差
        g = build_graph(st)
        assert g is not None
        # 有序整数占位 -> 每位点 one-hot，行和为 1
        assert np.allclose(g["node_feat"].sum(axis=1), 1.0)
        assert g["n_atoms"] == info["n_sites"]

    def test_disordered_mode_uses_partial_occupancy(self):
        pytest.importorskip("pymatgen")
        sys.path.insert(0, str(ROOT / "src"))
        import generate_prototype_structures as gps
        from cgcnn_data import build_graph

        st, info = gps.rocksalt_prototype("Li1.2Mn0.4Ti0.4O2", a=4.1, supercell=1,
                                         decorated=False)
        assert info["decorated"] is False
        assert info["composition_deviation"] == 0.0
        g = build_graph(st)
        assert g is not None
        # 部分占位 -> 加权 one-hot，行和仍为 1，但不再是 0/1 二值
        sums = g["node_feat"].sum(axis=1)
        assert np.allclose(sums, 1.0)
        assert not np.allclose(g["node_feat"], np.round(g["node_feat"]))

    def test_sensitivity_metrics_schema(self):
        path = ROOT / "results" / "metrics_prototype_sensitivity.json"
        if not path.exists():
            pytest.skip("未运行原型敏感性分析")
        m = json.load(open(path, encoding="utf-8"))
        assert m["n_compared"] >= 5
        for key in ("bias_eV_atom", "std_eV_atom", "MAE_proto_vs_mp_eV_atom",
                    "spearman"):
            assert key in m["ef"], key
        # 原型 vs MP 差异应显著（多晶型能量差），否则分层无意义
        assert abs(m["ef"]["bias_eV_atom"]) > 0.5


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
