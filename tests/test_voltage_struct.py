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
        assert tiers["A_comparable"] == m["n_ranked_with_structure"]
        assert tiers["B_prototype_rocksalt"] == m["n_prototype_only"]
        assert tiers["C_no_structure"] == m["n_fallback_without_structure"]
        # A 层 = MP 结构 + 同构替换相 + 无序系综代表结构（三者之和必须等于 A 层规模）
        assert (m["n_from_mp_structure"] + m["n_from_polymorph_substitution"]
                + m.get("n_from_ensemble", 0) == tiers["A_comparable"])
        assert m["weights"]["stability"] == 0.40
        # 尺度无关的一致性指标必须存在（两套口径不可比绝对值）
        assert "stability_rank_correlation_struct_vs_comp" in m

    def test_tier_tables_are_disjoint_and_schema_consistent(self):
        if not self.RANKED.exists():
            pytest.skip("未生成结构筛选排序表")
        ranked = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        assert len(ranked)
        # A 层：必须都有结构（MP 或同构替换相），且分数降序
        assert ranked["cgcnn_ef_eV_atom"].notna().all()
        assert set(ranked["structure_origin"]) <= {"MP弛豫", "同构替换(CHGNet)",
                                                   "无序系综(CHGNet)"}
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


class TestTemplateSubstitution:
    """同构替换（多晶型枚举的第一步）：替换阳离子、保留阴离子骨架（需 pymatgen）。"""

    @staticmethod
    def _template():
        pytest.importorskip("pymatgen")
        from pymatgen.core import Lattice, Structure

        # 简易岩盐型 LiCoO2 模板并扩成 2x2x2（16 个阳离子位点）：
        # 位点太少时整数占位取整会产生大偏差，放大后更能反映真实行为
        st = Structure(Lattice.cubic(4.0),
                       ["Li", "Co", "O", "O"],
                       [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0, 0, 0.5]])
        st.make_supercell([2, 2, 2])
        return st.as_dict()

    def test_substitution_preserves_anion_sublattice(self):
        pytest.importorskip("pymatgen")
        import template_substitution as ts
        from pymatgen.core import Composition, Structure

        st, meta = ts.substitution_candidates("LiCoO2", self._template(),
                                              "LiNi0.5Co0.2Mn0.3O2")
        assert meta["template"] == "LiCoO2"
        assert meta["substitution_deviation"] < 0.15
        new = Structure.from_dict(st)
        # 阴离子位点不动
        anions = [i for i, s in enumerate(new) if s.species_string in ts.ANION_ELEMENTS]
        assert len(anions) == 16
        # 阳离子位点全部为目标元素（Li/Ni/Co/Mn）
        cations = {new[i].species_string for i in range(len(new)) if i not in anions}
        assert cations <= {"Li", "Ni", "Co", "Mn"}
        # 还原出来的成分与目标成分的**阳离子位点比例**接近
        # （注意：要按阳离子归一化后比，直接比原子分数会把 O 也算进分母）
        counts = {el: sum(1 for s in new if s.species_string == el) for el in cations}
        got_total = sum(counts.values())
        want = Composition("LiNi0.5Co0.2Mn0.3O2")
        want_cat = {el: want[el] for el in ("Li", "Ni", "Co", "Mn")}
        want_total = sum(want_cat.values())
        for el in ("Li", "Ni", "Co", "Mn"):
            assert abs(counts[el] / got_total - want_cat[el] / want_total) < 0.05

    def test_incompatible_anion_set_returns_none(self):
        pytest.importorskip("pymatgen")
        import template_substitution as ts

        # 氧化物模板 -> 氟化物目标：阴离子骨架不同，必须拒绝
        got = ts.substitution_candidates("LiCoO2", self._template(), "LiFePO4F")
        assert got is None

    def test_cation_anion_ratio_mismatch_returns_none(self):
        pytest.importorskip("pymatgen")
        import template_substitution as ts

        # LiCoO2（阳离子:阴离子 = 1:1）-> Li2FeSiO4（3:4）：比例不符，必须拒绝
        got = ts.substitution_candidates("LiCoO2", self._template(), "Li2FeSiO4")
        assert got is None


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


class TestPolymorphComparability:
    """同构替换相能否并入主排名，取决于它与 MP 真实结构的实测残差（报告附录 L）。"""

    METRICS = ROOT / "results" / "metrics_polymorph_sensitivity.json"

    def test_residual_is_small_enough_to_merge(self):
        if not self.METRICS.exists():
            pytest.skip("未跑同构替换敏感性分析")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["n_compared"] >= 10
        # 并入 A 层的依据：bias 远小于岩盐原型（~1.25），且排序高度一致
        assert abs(m["ef"]["bias_eV_atom"]) < 0.3, m["ef"]
        assert m["ef"]["spearman"] > 0.8, m["ef"]
        assert m["ef"]["MAE_proto_vs_mp_eV_atom"] < 0.3

    def test_rocksalt_prototype_residual_is_much_larger(self):
        """对照：岩盐原型残差必须显著更大，否则分层逻辑需要重新评估。"""
        rs = ROOT / "results" / "metrics_prototype_sensitivity.json"
        if not (self.METRICS.exists() and rs.exists()):
            pytest.skip("缺少对比文件")
        poly = json.load(open(self.METRICS, encoding="utf-8"))["ef"]
        rock = json.load(open(rs, encoding="utf-8"))["ef"]
        assert abs(rock["bias_eV_atom"]) > 5 * abs(poly["bias_eV_atom"])


class TestDrxEnsemble:
    """无序系综（DRX/富锂）：把「装饰噪声」量化成误差条（v1.13.0）。"""

    METRICS = ROOT / "results" / "metrics_drx_ensemble.json"
    STRUCTS = ROOT / "data" / "raw" / "mp_structures_candidates_ensemble.json.gz"

    def test_metrics_schema(self):
        if not self.METRICS.exists():
            pytest.skip("未运行无序系综")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["n_targets"] >= 5
        assert m["n_decorations_per_target"] >= 2
        for key in ("ef_std_mean_eV_atom", "ef_std_max_eV_atom",
                    "gap_std_mean_eV", "gap_std_max_eV"):
            assert key in m
            assert m[key] >= 0
        for row in m["per_target"]:
            assert {"target", "template", "n", "ef_mean", "ef_std", "gap_mean", "gap_std"} <= set(row)
            assert row["ef_std"] >= 0 and row["gap_std"] >= 0

    def test_ensemble_structures_keep_representative_and_stats(self):
        if not self.STRUCTS.exists():
            pytest.skip("未生成系综结构文件")
        rows = json.load(gzip.open(self.STRUCTS, "rt", encoding="utf-8"))
        assert len(rows) >= 5
        for r in rows:
            assert r["match_type"] == "polymorph_substitution_ensemble"
            assert r["structure"].get("sites")
            ens = r["ensemble"]
            assert ens["n_decorations"] >= 2
            assert ens["selection"] == "lowest_CHGNet_energy"
            assert "cgcnn_ef_std_eV_atom" in ens and "cgcnn_gap_std_eV" in ens

    def test_decoration_noise_is_small_vs_model_error(self):
        """一致性检查：装饰噪声应远小于模型自身误差，否则单装饰结论不可用。"""
        if not self.METRICS.exists():
            pytest.skip("未运行无序系综")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["ef_std_mean_eV_atom"] < 0.1, (
            "装饰噪声过大时，必须改用系综均值而不是单个装饰")


class TestEnsembleErrorBars:
    """系综误差条必须带进筛选产物（带隙噪声不可忽略，稳定性可忽略）。"""

    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"

    def test_decoration_noise_is_reported(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        noise = m.get("decoration_noise") or {}
        assert noise.get("ef_std_mean_eV_atom") is not None
        # 稳定性噪声应远小于模型误差；带隙噪声则不可忽略（>0.05 eV）
        assert noise["ef_std_mean_eV_atom"] < 0.1
        assert noise["gap_std_mean_eV"] > 0.05

    def test_error_columns_present_for_ensemble_candidates(self):
        if not self.RANKED.exists():
            pytest.skip("未生成排序表")
        d = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        for col in ("ef_ens_std_eV_atom", "gap_ens_std_eV"):
            assert col in d.columns
        ens_rows = d[d["structure_origin"] == "无序系综(CHGNet)"]
        if len(ens_rows):
            assert ens_rows["ef_ens_std_eV_atom"].notna().all()
            assert ens_rows["gap_ens_std_eV"].notna().all()


class TestConservativeScoring:
    """1σ 保守评分：系综候选用 mean+σ，无误差条者退化为点预测（v1.14.0）。"""

    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"

    def test_scoring_mode_is_declared(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["scoring_mode"] == "conservative_1sigma"
        assert m["decoration_noise"]["n_ensemble_candidates"] >= 1

    def test_conservative_equals_point_plus_sigma(self):
        if not self.RANKED.exists():
            pytest.skip("未生成排序表")
        d = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        for col in ("stability_conservative_eV_atom", "gap_conservative_eV",
                    "cgcnn_ef_best_member_eV_atom"):
            assert col in d.columns
        # 无误差条（MP 结构）的候选：保守值 == 点预测
        plain = d[d["ef_ens_std_eV_atom"].isna()]
        assert (plain["stability_conservative_eV_atom"]
                - plain["stability_eV_atom"]).abs().max() < 1e-9
        # 系综候选：保守值 == 系综均值 + σ，且点预测已换成系综均值
        ens = d[d["ef_ens_std_eV_atom"].notna()]
        if len(ens):
            delta = ens["stability_conservative_eV_atom"] - ens["stability_eV_atom"]
            assert (delta - ens["ef_ens_std_eV_atom"]).abs().max() < 1e-6
            assert (ens["cgcnn_ef_eV_atom"]
                    - ens["cgcnn_ef_best_member_eV_atom"]).abs().max() > 0, (
                "系综候选的点预测应区别于单个 best member")
