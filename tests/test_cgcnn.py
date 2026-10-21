"""CGCNN 晶体图构建测试（纯 NumPy，CI 无需安装 torch）。

模型与训练脚本需要 torch（可选依赖），但**图构建是训练输入的唯一来源**，
所以把它做成无依赖模块并在这里完整覆盖：形状、双向边、one-hot、截断半径、可复现性。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cgcnn_data import (ATOM_FEAT_DIM, EDGE_FEAT_DIM, build_graph,  # noqa: E402
                        graph_signature)


def _cubic_cell(a: float = 3.9):
    """简化钙钛矿胞：Ca1 Ti1 O3（5 原子，立方）。"""
    return {
        "lattice": {"matrix": [[a, 0, 0], [0, a, 0], [0, 0, a]]},
        "sites": [
            {"species": [{"element": "Ca", "occu": 1}], "abc": [0.0, 0.0, 0.0]},
            {"species": [{"element": "Ti", "occu": 1}], "abc": [0.5, 0.5, 0.5]},
            {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.5, 0.0]},
            {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.0, 0.5]},
            {"species": [{"element": "O", "occu": 1}], "abc": [0.0, 0.5, 0.5]},
        ],
    }


class TestBuildGraph:
    def test_shapes(self):
        g = build_graph(_cubic_cell())
        assert g is not None
        assert g["node_feat"].shape == (5, ATOM_FEAT_DIM)
        assert g["edge_feat"].shape[1] == EDGE_FEAT_DIM
        assert len(g["edge_src"]) == len(g["edge_dst"]) == len(g["edge_feat"])
        assert int(g["edge_src"].max()) < 5 and int(g["edge_dst"].max()) < 5

    def test_edges_are_bidirectional(self):
        g = build_graph(_cubic_cell())
        edges = {(int(s), int(d)) for s, d in zip(g["edge_src"], g["edge_dst"])}
        for s, d in edges:
            assert (d, s) in edges, f"缺少反向边 ({d},{s})"
        assert len(g["edge_src"]) % 2 == 0

    def test_no_self_loops(self):
        g = build_graph(_cubic_cell())
        assert not np.any(g["edge_src"] == g["edge_dst"])

    def test_node_features_are_one_hot(self):
        g = build_graph(_cubic_cell())
        assert np.allclose(g["node_feat"].sum(axis=1), 1.0)
        # Ca=20, Ti=22, O=8 -> 第 Z-1 列应为 1
        assert g["node_feat"][0, 19] == 1.0
        assert g["node_feat"][1, 21] == 1.0
        assert g["node_feat"][2, 7] == 1.0
        assert g["node_feat"][3, 7] == 1.0

    def test_edge_features_are_gaussian_and_positive(self):
        g = build_graph(_cubic_cell())
        assert np.all(g["edge_feat"] >= 0)
        # 每个高斯基向量至少有一个显著分量
        assert np.all(g["edge_feat"].sum(axis=1) > 0.5)

    def test_cutoff_drops_far_neighbors(self):
        """把 cutoff 设得很小，边数应显著减少（但至少保留最近邻）。"""
        g_full = build_graph(_cubic_cell(3.9), cutoff=8.0)
        g_tight = build_graph(_cubic_cell(3.9), cutoff=2.2)
        assert len(g_tight["edge_src"]) < len(g_full["edge_src"])
        assert len(g_tight["edge_src"]) > 0

    def test_max_neighbors_is_enforced(self):
        for k in (4, 12):
            g = build_graph(_cubic_cell(4.5), max_neighbors=k)
            out_deg = np.bincount(g["edge_src"], minlength=int(g["n_atoms"]))
            assert out_deg.max() <= k

    def test_deterministic(self):
        g1 = build_graph(_cubic_cell())
        g2 = build_graph(_cubic_cell())
        for key in ("node_feat", "edge_src", "edge_dst", "edge_feat"):
            assert np.array_equal(g1[key], g2[key])

    def test_signature(self):
        g = build_graph(_cubic_cell())
        n_atoms, n_edges, atom_dim, edge_dim = graph_signature(g)
        assert n_atoms == 5
        assert atom_dim == ATOM_FEAT_DIM
        assert edge_dim == EDGE_FEAT_DIM
        assert n_edges == len(g["edge_src"])

    @pytest.mark.parametrize("bad", [
        None,
        {},
        {"sites": []},
        {"sites": [], "lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
        {"lattice": {"matrix": [[1, 0, 0]]},
         "sites": [{"species": [{"element": "Li", "occu": 1}], "abc": [0, 0, 0]}]},
    ])
    def test_invalid_returns_none(self, bad):
        assert build_graph(bad) is None

    def test_unknown_element_symbol_returns_none(self):
        st = {
            "lattice": {"matrix": [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]},
            "sites": [
                {"species": [{"element": "Xx", "occu": 1}], "abc": [0, 0, 0]},
                {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.5, 0.5]},
            ],
        }
        assert build_graph(st) is None

    def test_occupancy_picks_dominant_species(self):
        st = {
            "lattice": {"matrix": [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]]},
            "sites": [
                {"species": [{"element": "Li", "occu": 0.4},
                             {"element": "Na", "occu": 0.6}], "abc": [0.0, 0.0, 0.0]},
                {"species": [{"element": "O", "occu": 1.0}], "abc": [0.5, 0.5, 0.5]},
            ],
        }
        g = build_graph(st)
        assert g is not None
        assert g["node_feat"][0, 10] == 1.0     # Na (Z=11) 占主导
        assert g["node_feat"][0, 2] == 0.0      # Li 未入选

class TestScriptInterface:
    """训练脚本的 CLI 接口应在没有 torch 的环境里也能被检查（--help 早于 torch 导入）。"""

    def test_help_works_without_torch(self):
        import subprocess

        import os

        # Windows 控制台代码页无法编码中文/特殊字符：子进程强制 UTF-8，
        # 父进程解码用 errors="replace"，让测试与终端语言环境解耦。
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, str(ROOT / "src" / "train_cgcnn.py"), "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=60)
        assert proc.returncode == 0
        assert "CGCNN" in proc.stdout
        assert "--epochs" in proc.stdout and "--device" in proc.stdout


class TestConsoleEncoding:
    """回归：Windows 代码页（GBK/cp1252）下打印中文/特殊字符不能把脚本打挂。

    这正是 GitHub Actions Windows Runner 上 v1.5.0 的失败原因：
    argparse 打印中文 help 时抛 UnicodeEncodeError。
    """

    def test_console_guard_runs_under_gbk(self):
        import os
        import subprocess

        env = {**os.environ, "PYTHONIOENCODING": "gbk"}
        proc = subprocess.run(
            [sys.executable, str(ROOT / "src" / "console.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_cli_help_survives_gbk(self):
        import os
        import subprocess

        env = {**os.environ, "PYTHONIOENCODING": "gbk"}
        for script in ("train_cgcnn.py", "predictor.py", "conformal.py",
                       "train_voltage.py", "screening.py"):
            proc = subprocess.run(
                [sys.executable, str(ROOT / "src" / script), "--help"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, timeout=90)
            assert proc.returncode == 0, (script, proc.stderr[-500:])
