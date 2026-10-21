# -*- coding: utf-8 -*-
"""成分特征 = 33 个统计特征 + 可选的元素分数（element fraction）特征。

统计特征（33 个）:
  6 种元素属性 × 5 种统计量（原子分数加权 mean/std/max/min/range）
  + 3 个式级特征（n_elements / atoms_per_fu / molar_mass）

元素分数特征（可选）:
  对给定元素列表 element_list，每元素一列 = 该元素原子数 / 总原子数；
  样本中不含该元素时记 0（即训练集元素宇宙上的 one-hot 分数向量）。
  这等价于 matminer / Magpie 的 element fraction 特征，让树模型可以
  直接学习"元素身份"效应（如卤素种类对钙钛矿形成能的强影响），
  这是纯统计特征表达能力的关键补充。

电负性缺失（稀有气体等）记 NaN：HGB 原生支持，RF 流水线用中位数插补。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from element_data import ELEMENT_ROWS  # noqa: E402

# 元素属性查询表: symbol -> dict
ELEMENTS = {
    sym: {"Z": z, "mass": mass, "EN": (float(en) if en is not None else np.nan),
          "radius": radius, "group": group, "period": period}
    for z, sym, mass, en, radius, group, period in ELEMENT_ROWS
}

PROPS = ["Z", "mass", "EN", "radius", "group", "period"]
STATS = ["mean", "std", "max", "min", "range"]

FEATURE_NAMES = [f"{p}_{s}" for p in PROPS for s in STATS] + [
    "n_elements", "atoms_per_fu", "molar_mass",
]


def build_universe(formulas) -> list:
    """一组化学式 -> 排序后的元素宇宙列表（解析失败的跳过）。"""
    from formula import parse_formula
    universe = set()
    for f in formulas:
        try:
            universe |= set(parse_formula(str(f)).keys())
        except ValueError:
            pass
    return sorted(universe)


def element_fraction_feats(counts: dict, element_list) -> dict:
    total = float(sum(counts.values()))
    if total <= 0:
        return {f"frac_{el}": 0.0 for el in element_list}
    return {f"frac_{el}": counts.get(el, 0.0) / total for el in element_list}


def featurize_one(composition: str, element_list=None) -> dict | None:
    """单条化学式 -> 特征字典（统计 + 可选元素分数）；失败返回 None。"""
    from formula import parse_formula
    try:
        counts = parse_formula(str(composition))
    except ValueError:
        return None
    if any(el not in ELEMENTS for el in counts):
        return None

    total = float(sum(counts.values()))
    xs = {el: c / total for el, c in counts.items()}

    feats = {}
    for p in PROPS:
        vals = np.array([ELEMENTS[el][p] for el in counts], dtype=float)
        w = np.array([xs[el] for el in counts], dtype=float)
        # 只在有值的元素上重新归一化权重：否则含缺失属性（如稀有气体电负性
        # 记 NaN）的成分会被 nansum 稀释，均值系统性偏低（LiHeO2 曾偏差 25%）。
        ok = ~np.isnan(vals)
        if ok.any() and w[ok].sum() > 0:
            v_ok = vals[ok]
            w_ok = w[ok] / w[ok].sum()
            mean = float(np.sum(w_ok * v_ok))
            var = float(np.sum(w_ok * (v_ok - mean) ** 2))
            vmax, vmin = float(v_ok.max()), float(v_ok.min())
            std = math.sqrt(max(var, 0.0))
        else:
            mean = std = vmax = vmin = float("nan")
        feats[f"{p}_mean"] = mean
        feats[f"{p}_std"] = std
        feats[f"{p}_max"] = vmax
        feats[f"{p}_min"] = vmin
        feats[f"{p}_range"] = vmax - vmin

    feats["n_elements"] = float(len(counts))
    feats["atoms_per_fu"] = total
    feats["molar_mass"] = float(sum(counts[el] * ELEMENTS[el]["mass"] for el in counts))

    if element_list is not None:
        feats.update(element_fraction_feats(counts, element_list))
    return feats


def featurize_dataframe(compositions, element_list=None, verbose=True):
    """批量特征化。返回 (特征 DataFrame, 失败样本下标列表)。"""
    import pandas as pd
    names = list(FEATURE_NAMES)
    if element_list is not None:
        names += [f"frac_{el}" for el in element_list]
    rows, failed = [], []
    for i, comp in enumerate(compositions):
        f = featurize_one(comp, element_list)
        if f is None:
            failed.append(i)
            rows.append({name: np.nan for name in names})
        else:
            rows.append(f)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  已特征化 {i + 1} 条")
    df = pd.DataFrame(rows, columns=names)
    return df, failed


if __name__ == "__main__":
    demo = ["LiCoO2", "LiNi0.8Co0.1Mn0.1O2", "LiFePO4", "Ba5(AsO4)3(OH,F)", "CsPbI3"]
    els = build_universe(demo)
    print("元素宇宙:", els)
    df, failed = featurize_dataframe(demo, element_list=els, verbose=False)
    print(df[["Z_mean", "EN_mean", "molar_mass", "n_elements",
              "frac_Li", "frac_Cs", "frac_Pb", "frac_I"]])
    assert df.notna().all().all() or failed, "不应有解析失败"
    print("统计特征数:", len(FEATURE_NAMES), " 总列数:", len(df.columns))
