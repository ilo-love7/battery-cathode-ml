# -*- coding: utf-8 -*-
"""晶体图的构建（纯 NumPy，不依赖 torch）。

CGCNN（Xie & Grossman, PRL 2018）把晶体表示为图：原子为节点、近邻为边。
本模块只负责**把结构变成索引数组**，这样：

- 图构建逻辑可以在没有 torch 的环境里被单元测试覆盖（CI 不需要装 torch）；
- 训练脚本只需消费 ``(node_feat, edge_src, edge_dst, edge_feat)`` 四个数组。

约定
----
- 节点特征：**按占位度加权的原子序数 one-hot**，维度 ``ATOM_FEAT_DIM = 95``（Z=1..94 + "未知"）。
  有序结构（占位度=1）退化为普通 one-hot，与早期训练完全一致；
  部分占位的原型结构（如无序岩盐 DRX）则给出混合向量，无需人为凑整化学计量比。
- 边特征：距离的高斯展开（中心 0..8 Å，步长 0.2 Å → 41 维）；
- 近邻：周期边界最小镜像距离 ≤ ``cutoff``，每个原子最多取 ``max_neighbors`` 个最近邻，
  每条键按 CGCNN 惯例**双向各存一条**。

用法::

    from cgcnn_data import build_graph, ATOM_FEAT_DIM, EDGE_FEAT_DIM
    g = build_graph(structure_dict)         # dict of numpy arrays
    g["node_feat"]   # (n_atoms, 95)
    g["edge_src"]    # (n_edges,)
    g["edge_dst"]    # (n_edges,)
    g["edge_feat"]   # (n_edges, 41)
"""

from __future__ import annotations

import numpy as np

ATOM_FEAT_DIM = 95          # 原子序数 one-hot（1..94 有效，其余归到索引 94）
EDGE_FEAT_DIM = 41          # 高斯距离展开维数
CUTOFF = 8.0                # Å
MAX_NEIGHBORS = 12
GAUSS_STEP = 0.2
GAUSS_SIGMA = 0.2

_GAUSS_CENTERS = np.arange(0.0, CUTOFF + 1e-9, GAUSS_STEP)     # 0,0.2,...,8.0


def _gauss_expand(distances: np.ndarray) -> np.ndarray:
    """距离 -> 高斯展开 (n, 41)。"""
    d = np.asarray(distances, dtype=np.float32)[:, None]
    c = _GAUSS_CENTERS[None, :].astype(np.float32)
    return np.exp(-((d - c) ** 2) / (2.0 * GAUSS_SIGMA ** 2)).astype(np.float32)


def _min_image_distances(frac: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """全部原子对的周期最小镜像距离矩阵（Å），对角为 +inf。"""
    d = frac[:, None, :] - frac[None, :, :]
    d -= np.round(d)
    cart = d @ matrix
    dist = np.linalg.norm(cart, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return dist


def build_graph(st: dict, cutoff: float = CUTOFF,
                max_neighbors: int = MAX_NEIGHBORS) -> dict | None:
    """pymatgen Structure dict -> 晶体图数组；结构非法返回 None。"""
    if not st or "sites" not in st or "lattice" not in st:
        return None
    try:
        matrix = np.asarray(st["lattice"]["matrix"], dtype=np.float64)
        sites = st["sites"]
        frac = np.asarray([s["abc"] for s in sites], dtype=np.float64)
        zis = []          # 每个位点的 [(Z, occu), ...]，支持部分占位
        for s in sites:
            pairs = []
            for sp in s["species"]:
                el = sp.get("element")
                if el is None:
                    return None
                z = int(el) if isinstance(el, (int, np.integer)) else _Z.get(str(el), 0)
                if z <= 0:
                    return None
                pairs.append((z, float(sp.get("occu", 1.0))))
            if not pairs:
                return None
            zis.append(pairs)
    except (KeyError, TypeError, ValueError):
        return None

    n = len(frac)
    if n == 0 or matrix.shape != (3, 3):
        return None

    node_feat = np.zeros((n, ATOM_FEAT_DIM), dtype=np.float32)
    for i, zi in enumerate(zis):
        total_occ = float(sum(o for _, o in zi)) or 1.0
        for z, occ in zi:
            node_feat[i, min(z, ATOM_FEAT_DIM) - 1] += occ / total_occ

    dist = _min_image_distances(frac, matrix)

    src, dst, ef = [], [], []
    for i in range(n):
        order = np.argsort(dist[i])[:max_neighbors]
        for j in order:
            dij = dist[i, j]
            if not np.isfinite(dij) or dij > cutoff:
                continue
            src.append(i)
            dst.append(int(j))
            ef.append(dij)
    if not src:
        return None

    return {
        "node_feat": node_feat,
        "edge_src": np.asarray(src, dtype=np.int64),
        "edge_dst": np.asarray(dst, dtype=np.int64),
        "edge_feat": _gauss_expand(np.asarray(ef, dtype=np.float32)),
        "n_atoms": np.int64(n),
    }


# --- 元素符号 -> 原子序数（只建一次，覆盖数据集常见元素 + 全部 1..94） ---
def _build_symbol_table() -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from element_data import ELEMENT_ROWS

    return {sym: z for z, sym, *_ in ELEMENT_ROWS}


_Z = _build_symbol_table()


def graph_signature(g: dict) -> tuple:
    """图的形状签名（供测试与日志使用）。"""
    return (int(g["n_atoms"]), int(len(g["edge_src"])),
            int(g["node_feat"].shape[1]), int(g["edge_feat"].shape[1]))


if __name__ == "__main__":
    import gzip
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with gzip.open(root / "data/raw/matbench_perovskites.json.gz", "rt", encoding="utf-8") as f:
        d = json.load(f)
    st = d["data"][0][d["columns"].index("structure")]
    g = build_graph(st)
    print("signature (n_atoms, n_edges, atom_dim, edge_dim):", graph_signature(g))
    print("node_feat sum per atom:", g["node_feat"].sum(axis=1))
    print("edge_feat 非零个数:", int((g["edge_feat"] > 1e-6).sum()))