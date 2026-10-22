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
        # v1.15.0 起为四维权重（稳定性/导电性/容量/电压），总和为 1
        assert abs(sum(m["weights"].values()) - 1.0) < 1e-9
        assert m["weights_3dim_legacy"]["stability"] == 0.40
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


class TestVoltageDimension:
    """电压维度进评分 + 局部自适应区间（v1.15.0）。"""

    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"
    LOCAL = ROOT / "results" / "metrics_local_conformal.json"

    def test_weights_include_voltage_and_sum_to_one(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        w = m["weights"]
        assert "voltage" in w
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert "weights_3dim_legacy" in m          # 旧口径保留供对照

    def test_score_is_weighted_sum_of_four_dimensions(self):
        if not self.RANKED.exists():
            pytest.skip("未生成排序表")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        d = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        for col in ("pct_stability", "pct_conduction", "pct_capacity", "pct_voltage",
                    "voltage_conservative_V", "score", "score_3dim"):
            assert col in d.columns
        w = m["weights"]
        expect = (w["stability"] * d["pct_stability"] + w["conduction"] * d["pct_conduction"]
                  + w["capacity"] * d["pct_capacity"] + w["voltage"] * d["pct_voltage"])
        assert (d["score"] - expect).abs().max() < 1e-6
        # 保守电压 = 点预测 − 所用半宽
        delta = d["voltage_V"] - d["voltage_conservative_V"] - d["voltage_halfwidth_used_V"]
        assert delta.abs().max() < 1e-6
        # 新旧口径应确实不同（否则说明电压维度没起作用）
        assert (d["score"] - d["score_3dim"]).abs().max() > 1.0

    def test_local_conformal_is_calibrated_and_differentiated(self):
        if not self.LOCAL.exists():
            pytest.skip("未运行局部自适应 conformal")
        m = json.load(open(self.LOCAL, encoding="utf-8"))
        v = m["validation"]
        # 原始局部覆盖率不足（局部可交换假设不成立），校准后才接近名义值
        assert v["empirical_coverage"] < 0.85
        assert v.get("empirical_coverage_calibrated", 0) > 0.85
        assert m["local_halfwidth_V"]["max_over_min"] > 2, "缺少区分度则不值得用局部区间"


class TestWeightCalibration:
    """权重校准：用文献实测比能量给权重找依据（v1.16.0）。"""

    CAL = ROOT / "results" / "metrics_weights_calibration.json"
    REF = ROOT / "data" / "reference_cathodes.csv"
    SCREEN = ROOT / "results" / "metrics_screening_struct.json"

    def test_reference_table_is_plausible(self):
        ref = pd.read_csv(self.REF, encoding="utf-8-sig")
        assert len(ref) >= 20
        assert {"formula", "voltage_practical_V", "capacity_practical_mAh_g"} <= set(ref.columns)
        assert ref["voltage_practical_V"].between(1.5, 5.5).all()
        assert ref["capacity_practical_mAh_g"].between(50, 400).all()
        assert ref["formula"].is_unique

    def test_calibration_beats_prior(self):
        if not self.CAL.exists():
            pytest.skip("未运行权重校准")
        m = json.load(open(self.CAL, encoding="utf-8"))
        assert m["n_aligned"] >= 20
        w = m["best_weights"]
        assert abs(sum(w.values()) - 1.0) < 1e-9
        c = m["constraints"]
        assert w["stability"] >= c["min_stability"] - 1e-9
        assert w["conduction"] >= c["min_conduction"] - 1e-9
        assert w["voltage"] >= c["min_voltage"] - 1e-9
        assert w["capacity"] <= c["max_capacity"] + 1e-9
        # 校准的核心目的：排序一致性优于先验（本项目实测先验为负相关）
        assert m["best_rho"] > m["prior_rho"]

    def test_screening_records_weights_source(self):
        if not self.SCREEN.exists():
            pytest.skip("未生成筛选指标")
        m = json.load(open(self.SCREEN, encoding="utf-8"))
        assert m["weights_source"] in {"calibrated", "prior"}
        assert abs(sum(m["weights"].values()) - 1.0) < 1e-9
        assert "energy_density" in m
        assert m["energy_density"]["median_Wh_kg"] > 100


class TestConditionalCoverage:
    """条件覆盖率：只报边际覆盖率会掩盖分层失效（v1.16.0）。"""

    LOCAL = ROOT / "results" / "metrics_local_conformal.json"

    def test_family_level_coverage_reported(self):
        if not self.LOCAL.exists():
            pytest.skip("未运行局部自适应 conformal")
        m = json.load(open(self.LOCAL, encoding="utf-8"))
        fam = m.get("coverage_by_family") or {}
        assert len(fam) >= 2, "至少要有两个体系族的条件覆盖率"
        for name, v in fam.items():
            assert v["n_test"] > 0
            for key in ("global_q_coverage", "local_raw_coverage",
                        "local_calibrated_coverage", "mean_local_halfwidth_V"):
                assert key in v, (name, key)
                assert 0.0 <= v[key] <= 1.0 or key == "mean_local_halfwidth_V"

    def test_calibration_fixes_worst_family(self):
        """校准应把最差族拉回名义值附近（这是局部区间可用的关键证据）。"""
        if not self.LOCAL.exists():
            pytest.skip("未运行局部自适应 conformal")
        fam = json.load(open(self.LOCAL, encoding="utf-8")).get("coverage_by_family") or {}
        if not fam:
            pytest.skip("无条件覆盖率数据")
        worst_global = min(v["global_q_coverage"] for v in fam.values())
        worst_cal = min(v["local_calibrated_coverage"] for v in fam.values())
        assert worst_cal > worst_global, (
            "若校准后最差族反而不如全局 q，则没有理由使用局部区间")


class TestMultiObjectiveCalibration:
    """多元外部标签校准 + 权重不确定度（v1.17.0）。"""

    CAL = ROOT / "results" / "metrics_weights_calibration.json"
    REF = ROOT / "data" / "reference_cathodes.csv"

    def test_reference_table_has_multiple_targets(self):
        ref = pd.read_csv(self.REF, encoding="utf-8-sig")
        for col in ("specific_energy_Wh_kg", "cycle_life_cycles", "rate_retention_pct",
                    "cost_index", "thermal_onset_C"):
            if col == "specific_energy_Wh_kg":
                continue
            assert col in ref.columns, col
            assert ref[col].notna().all()
        assert ref["cycle_life_cycles"].between(50, 10000).all()
        assert ref["cost_index"].between(0, 100).all()

    def test_multi_objective_block_present(self):
        if not self.CAL.exists():
            pytest.skip("未运行权重校准")
        m = json.load(open(self.CAL, encoding="utf-8"))
        assert len(m["external_targets"]) >= 4
        mo = m["multi_objective"]
        assert "best_weights_multi" in mo and "best_weights_maximin" in mo
        assert abs(sum(mo["best_weights_multi"].values()) - 1.0) < 1e-9

    def test_single_dimension_matrix_covers_all_targets(self):
        if not self.CAL.exists():
            pytest.skip("未运行权重校准")
        m = json.load(open(self.CAL, encoding="utf-8"))
        mat = m["single_dimension_rho_matrix"]
        assert set(mat) == {"pct_stability", "pct_conduction", "pct_capacity", "pct_voltage"}
        for dim, row in mat.items():
            assert set(row) == set(m["external_targets"]), dim

    def test_bootstrap_uncertainty_reported(self):
        if not self.CAL.exists():
            pytest.skip("未运行权重校准")
        m = json.load(open(self.CAL, encoding="utf-8"))
        bu = m["weight_uncertainty_bootstrap"]
        assert bu["n_resamples"] >= 10
        for k, v in bu["per_weight"].items():
            assert v["p05"] <= v["median"] <= v["p95"], (k, v)


class TestVoltageRiskItems:
    """电压不做线性排序，改为显式风险项（v1.17.0）。"""

    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"

    def test_risk_thresholds_and_counts(self):
        if not self.METRICS.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        vr = m["voltage_risk"]
        assert vr["thresholds_V"] == {"high": 4.5, "very_high": 5.0, "low": 2.5}
        d = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        assert (d["voltage_V"] > 4.5).sum() == vr["n_high_voltage_gt4_5"]
        assert (d["voltage_V"] > 5.0).sum() == vr["n_beyond_electrolyte_window"]
        assert (d["voltage_V"] < 2.5).sum() == vr["n_low_voltage_lt2_5"]

    def test_risk_level_column(self):
        if not self.RANKED.exists():
            pytest.skip("未生成排序表")
        d = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        assert "risk_level" in d.columns
        assert set(d["risk_level"]) <= {"低", "中", "高"}
        assert "voltage_risk" in d.columns
        # 超窗口的候选必须被标为高/中风险且带电压风险文本
        risky = d[d["voltage_V"] > 5.0]
        if len(risky):
            assert (risky["voltage_risk"] != "").all()


class TestProvenanceAndPareto:
    """参考表溯源 + Pareto 前沿（v1.18.0）。"""

    REF = ROOT / "data" / "reference_cathodes.csv"
    CAL = ROOT / "results" / "metrics_weights_calibration.json"
    METRICS = ROOT / "results" / "metrics_screening_struct.json"
    RANKED = ROOT / "results" / "cathode_ranking_struct.csv"

    def test_reference_has_provenance_and_per_row_uncertainty(self):
        ref = pd.read_csv(self.REF, encoding="utf-8-sig")
        for col in ("source_type", "uncertainty_pct", "source_note"):
            assert col in ref.columns, col
            assert ref[col].notna().all(), col
        assert ref["uncertainty_pct"].between(5, 30).all()
        assert ref["source_type"].nunique() >= 2
        # 商用材料的标注不确定度应不高于探索类
        grouped = ref.groupby("source_type")["uncertainty_pct"].mean()
        assert grouped.max() > grouped.min()

    def test_bootstrap_uses_per_row_uncertainty(self):
        if not self.CAL.exists():
            pytest.skip("未运行权重校准")
        m = json.load(open(self.CAL, encoding="utf-8"))
        bu = m["weight_uncertainty_bootstrap"]
        assert bu.get("mean_uncertainty_pct") is not None
        assert bu.get("provenance_counts")
        assert 5 <= bu["mean_uncertainty_pct"] <= 30

    def test_pareto_front_is_actually_non_dominated(self):
        if not self.RANKED.exists():
            pytest.skip("未生成排序表")
        d = pd.read_csv(self.RANKED, encoding="utf-8-sig")
        for col in ("pareto_rank", "is_pareto"):
            assert col in d.columns
        front = d[d["is_pareto"]]
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert len(front) >= 2
        # 必须与 screening_struct 的 6 目标一致：4 维投影下"被支配"不代表整体被支配
        assert len(m["pareto"]["objectives"]) == 6
        objs = np.column_stack([
            -front["stability_conservative_eV_atom"], -front["gap_conservative_eV"],
            front["Q_theoretical_mAh_g"], front["voltage_conservative_V"],
            -front["resource_risk_score"], -front["toxicity_class"]])
        # 前沿内部任意两点不得互相支配
        for i in range(len(objs)):
            for j in range(len(objs)):
                if i == j:
                    continue
                dominated = (np.all(objs[j] >= objs[i]) and np.any(objs[j] > objs[i]))
                assert not dominated, f"前沿内第 {i} 条被第 {j} 条支配"
        # 层级计数必须覆盖全部候选
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert sum(m["pareto"]["rank_counts"].values()) == len(d)
        assert m["pareto"]["n_front"] == len(front)


class TestPerRowCitations:
    """逐条溯源：引用 + DOI + 是否核验（v1.19.0）。"""

    REF = ROOT / "data" / "reference_cathodes.csv"

    def test_columns_present_and_consistent(self):
        d = pd.read_csv(self.REF, encoding="utf-8-sig")
        for col in ("citation", "doi", "citation_verified"):
            assert col in d.columns, col
            assert d[col].notna().all(), col
        # 语义：citation_verified=True 表示"该文献直接覆盖该材料"；
        # False 表示只引用了家族/综述文献（仍带 DOI，但不宣称已逐条核验）。
        ver = d[d["citation_verified"]]
        assert len(ver) >= 10, "已核验条目过少"
        assert (ver["doi"].astype(str).str.len() > 0).all()
        # 必须存在明确标注为"未核验"的条目（诚实性检查：不允许全部标成已核验）
        assert int((~d["citation_verified"]).sum()) > 0
        assert d["citation"].str.len().gt(0).all()

    def test_dois_are_well_formed_and_unique(self):
        d = pd.read_csv(self.REF, encoding="utf-8-sig")
        dois = [x for x in d["doi"].astype(str) if x]
        assert len(set(dois)) >= 5
        for doi in set(dois):
            assert doi.startswith("10."), doi
            assert "/" in doi, doi


class TestParetoInteractive:
    """偏好驱动的最优子集（v1.19.0）。"""

    METRICS = ROOT / "results" / "metrics_pareto_profiles.json"
    CSV = ROOT / "results" / "pareto_profiles.csv"

    def test_profiles_and_scores_present(self):
        if not (self.METRICS.exists() and self.CSV.exists()):
            pytest.skip("未运行偏好分析")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        assert m["n_profiles"] >= 4
        assert m["objectives"] == ["stability", "conduction", "capacity", "voltage"]
        d = pd.read_csv(self.CSV, encoding="utf-8-sig")
        for name in m["profiles"]:
            assert f"score_{name}" in d.columns and f"rank_{name}" in d.columns
            w = m["profiles"][name]["weights"]
            assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_preference_actually_changes_the_answer(self):
        """若所有偏好给出同一个第一名，则"偏好交互"毫无意义。"""
        if not self.METRICS.exists():
            pytest.skip("未运行偏好分析")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        firsts = {name: v["top"][0]["formula"] for name, v in m["profiles"].items()}
        assert len(set(firsts.values())) >= 2, f"所有偏好同解: {firsts}"

    def test_robust_picks_definition(self):
        if not self.METRICS.exists():
            pytest.skip("未运行偏好分析")
        m = json.load(open(self.METRICS, encoding="utf-8"))
        half = max(2, m["n_profiles"] // 2)
        for r in m["robust_picks"]:
            assert r["profile_votes"] >= half


class TestResourceRisk:
    """资源与法规维度（序数打分，v1.20.0）。"""

    def test_scores_are_ordinal_and_monotonic_in_cobalt(self):
        import resource_risk as rr
        lco = rr.score_formula("LiCoO2")
        lfp = rr.score_formula("LiFePO4")
        assert 0 <= lfp["resource_risk_score"] < lco["resource_risk_score"] <= 100
        # LiCoO2 每 4 个原子只有 1 个 Co -> 原子分数加权后风险分约 37（"中"）；
        # 但 Co 本身必须被识别为高风险关键元素
        assert lco["resource_risk_level"] in ("中", "高")
        assert lfp["resource_risk_level"] == "低"
        assert "Co" in lco["critical_elements"] and "Co供应链" in lco["flags"]
        # Li 的供应风险分恰为 55（同样列入关键矿产），故两者都被标出
        assert set(lco["high_risk_elements"]) == {"Co", "Li"}
        assert lco["toxicity_class"] >= lfp["toxicity_class"] >= 1

    def test_vanadium_toxicity_is_flagged(self):
        import resource_risk as rr
        v = rr.score_formula("LiVPO4F")
        assert v["toxicity_class"] >= 3
        assert "毒性≥3" in v["flags"]

    def test_screening_reports_resource_columns(self):
        path = ROOT / "results" / "cathode_ranking_struct.csv"
        if not path.exists():
            pytest.skip("未生成排序表")
        d = pd.read_csv(path, encoding="utf-8-sig")
        for col in ("resource_risk_score", "resource_risk_level", "toxicity_class",
                    "critical_elements", "resource_flags"):
            assert col in d.columns, col
        m = json.load(open(ROOT / "results" / "metrics_screening_struct.json",
                           encoding="utf-8"))
        rr = m["resource_regulatory"]
        assert rr["n_high_risk"] == int((d["resource_risk_score"] >= 55).sum())
        assert rr["n_toxicity_ge3"] == int((d["toxicity_class"] >= 3).sum())

    def test_adding_objectives_can_only_grow_the_frontier(self):
        """性质检查：目标越多，非支配集合只会变大（不能变小）。"""
        path = ROOT / "results" / "metrics_screening_struct.json"
        if not path.exists():
            pytest.skip("未生成结构筛选指标")
        m = json.load(open(path, encoding="utf-8"))
        assert len(m["pareto"]["objectives"]) == 6
        assert m["pareto"]["n_front"] >= 32, "六目标前沿不应少于四目标的 32 条"
        assert m["pareto"]["n_front"] <= 40


class TestSubsetTagGuard:
    """v1.4.2 的教训：带 --limit/--subset 的脚本绝不能覆盖全量产物。

    这里用源码级断言固化该策略（比"跑一次看看"更早失效）。
    """

    # 脚本 -> 其"部分运行不覆盖全量产物"的守卫标记
    GUARDS = {
        "drx_ensemble.py": "_smoke",
        "voltage_from_cgcnn.py": "_smoke",
        "pareto_interactive.py": "部分运行",
        "train_cgcnn.py": "_smoke",
        "train_cgcnn_mp.py": "_smoke",
        "train_cgcnn_voltage.py": "_smoke",
        "train_ef_struct.py": "_smoke",
        "train_voltage_struct.py": "_smoke",
    }

    def test_every_partial_run_renames_its_output(self):
        for name, marker in self.GUARDS.items():
            src = open(ROOT / "src" / name, encoding="utf-8").read()
            assert marker in src, f"{name} 缺少部分运行改名防护（期望标记 {marker!r}）"
