# -*- coding: utf-8 -*-
"""结构描述符：从晶体结构里抽出**成分之外的**几何信息。

为什么需要它
------------
``matbench_perovskites`` 的 18,928 条样本只对应 9,646 个唯一成分：同一化学式
因**阳离子占位方式/畸变**不同而对应不同形成能（同成分目标 std 中位数 0.42 eV/胞）。
纯成分特征原理上无法区分这些样本，MAE 卡在 0.379 eV/胞 的"成分不可约下限"。
本模块提供一组**轻量、纯 CPU、无需 GPU** 的结构描述符，用来验证
"引入结构信息能否突破该上限"——这也是通往 CGCNN 的中间台阶。

描述符（固定 21 维，全部对周期性边界取最小镜像距离）
----------------------------------------------------
- 晶格：a, b, c（Å）、α, β, γ（°）、体积、每原子体积
- 位点：原子数
- 近邻距离（每个原子到最近邻）：mean / std / min / max / range
- 全体原子对距离（含 PBC）：mean / std / min / max / range
- 按元素分组的平均近邻距离：mean / std / min / max / range（元素身份 × 几何耦合）

用法::

    from structure_features import structure_descriptors, descriptor_names
    st = {...}          # pymatgen Structure dict（matbench 里的 'structure' 列）
    feats = structure_descriptors(st)
"""

from __future__ import annotations

import numpy as np

DESCRIPTOR_NAMES = [
    "lat_a", "lat_b", "lat_c", "lat_alpha", "lat_beta", "lat_gamma",
    "lat_volume", "vol_per_atom", "nsites",
    "nn_dist_mean", "nn_dist_std", "nn_dist_min", "nn_dist_max", "nn_dist_range",
    "pair_dist_mean", "pair_dist_std", "pair_dist_min", "pair_dist_max", "pair_dist_range",
    "elem_nn_mean", "elem_nn_std",
]


def _min_image_distances(frac: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """全部原子对的周期性最小镜像距离矩阵（Å）。"""
    n = len(frac)
    d = frac[:, None, :] - frac[None, :, :]
    d -= np.round(d)                       # 最小镜像
    cart = d @ matrix
    dist = np.linalg.norm(cart, axis=-1)
    np.fill_diagonal(dist, np.inf)         # 排除自身
    return dist


def structure_descriptors(st: dict) -> dict | None:
    """pymatgen Structure dict -> 21 维结构描述符；解析失败返回 None。"""
    try:
        matrix = np.asarray(st["lattice"]["matrix"], dtype=float)
        sites = st["sites"]
        frac = np.asarray([s["abc"] for s in sites], dtype=float)
        elements = [max(s["species"], key=lambda sp: sp.get("occu", 1.0))["element"]
                    for s in sites]
    except (KeyError, TypeError, ValueError):
        return None

    n = len(frac)
    if n < 2 or matrix.shape != (3, 3):
        return None

    # 晶格参数
    a, b, c = (np.linalg.norm(matrix[i]) for i in range(3))

    def _angle(u, v):
        cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    alpha = _angle(matrix[1], matrix[2])
    beta = _angle(matrix[0], matrix[2])
    gamma = _angle(matrix[0], matrix[1])
    volume = abs(float(np.linalg.det(matrix)))

    dist = _min_image_distances(frac, matrix)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return None

    nn = dist.min(axis=1)                                  # 每个原子的最近邻距离
    pair = dist[np.triu_indices(n, k=1)]                    # 无序对

    # 按元素分组：每种元素的原子到「其他元素」原子的最近距离
    # （同种元素单原子时 self 距离是 inf，必须排除自身所在集合）
    elem_nn = []
    for el in set(elements):
        idx = np.array([i for i, e in enumerate(elements) if e == el])
        other = np.array([i for i, e in enumerate(elements) if e != el])
        if other.size:
            vals = dist[np.ix_(idx, other)].min(axis=1)
        else:                                   # 全同元素胞（如单质）
            vals = dist[np.ix_(idx, idx)]
            vals = vals[np.isfinite(vals)]
        if vals.size:
            elem_nn.append(float(np.mean(vals)))
    elem_nn = np.asarray(elem_nn, dtype=float) if elem_nn else np.asarray([np.nan])

    feats = {
        "lat_a": float(a), "lat_b": float(b), "lat_c": float(c),
        "lat_alpha": alpha, "lat_beta": beta, "lat_gamma": gamma,
        "lat_volume": volume, "vol_per_atom": volume / n, "nsites": float(n),
        "nn_dist_mean": float(np.mean(nn)), "nn_dist_std": float(np.std(nn)),
        "nn_dist_min": float(np.min(nn)), "nn_dist_max": float(np.max(nn)),
        "nn_dist_range": float(np.max(nn) - np.min(nn)),
        "pair_dist_mean": float(np.mean(pair)), "pair_dist_std": float(np.std(pair)),
        "pair_dist_min": float(np.min(pair)), "pair_dist_max": float(np.max(pair)),
        "pair_dist_range": float(np.max(pair) - np.min(pair)),
        "elem_nn_mean": float(np.nanmean(elem_nn)),
        "elem_nn_std": float(np.nanstd(elem_nn)),
    }
    return feats


def descriptor_names() -> list:
    """返回固定列顺序（供训练脚本对齐）。"""
    return list(DESCRIPTOR_NAMES)


if __name__ == "__main__":
    import gzip
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with gzip.open(root / "data/raw/matbench_perovskites.json.gz", "rt", encoding="utf-8") as f:
        d = json.load(f)
    st = d["data"][0][d["columns"].index("structure")]
    feats = structure_descriptors(st)
    print("维数:", len(feats))
    for k, v in feats.items():
        print(f"  {k:16s} {v:.4f}")