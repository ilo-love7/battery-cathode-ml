# -*- coding: utf-8 -*-
"""评分权重校准：用**文献实测性能**给权重找依据（v1.16.0）。

### 问题

v1.15.0 的四维权重（稳定性 0.35 / 导电性 0.20 / 容量 0.25 / 电压 0.20）是**先验设定**。
本脚本用一份**文献典型性能表**（`data/reference_cathodes.csv`，实测工作电压与实用比容量）
构造外部目标，用网格搜索选出与实测性能排序最一致的权重。

### 方法

1. 外部目标（**多元**）：
   - 比能量 `Wh/kg = V_practical × Q_practical`（Wh/kg）
   - 循环寿命 `cycle_life_cycles`
   - 倍率保持率 `rate_retention_pct`
   - **成本指数**（越小越好，取负号参与）
   - 热稳定性 `thermal_onset_C`；
2. 候选侧：用 `results/cathode_ranking_struct.csv` 的四维百分位
   （稳定性 / 导电性 / 容量 / 电压，均已在 A 层内归一化）；
3. 在权重单纯形（步长 0.05，和为 1）上网格搜索，最大化**加权评分与实测比能量的 Spearman ρ**；
4. 约束：稳定性权重 ≥ 0.20（不能把"结构稳定性"当作可牺牲项）、电压 ≥ 0.10；
5. 输出推荐权重、各维度单独 ρ、先验权重 ρ、以及"能量密度口径"的 ρ 对照。

⚠️ 参考表是**文献典型值**（同一材料不同报道会有 ±10% 差异），因此校准结论应读作
"权重取值的合理区间"，而不是精确到小数点的定论。

用法::

    python src/calibrate_weights.py

输出::

    results/metrics_weights_calibration.json
    results/weights_calibration.csv      （网格搜索结果，按 ρ 排序）
    results/weights_calibration.png      （ρ 热力图/曲线）
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference_cathodes.csv"
RANKING = ROOT / "results" / "cathode_ranking_struct.csv"
OUT_JSON = ROOT / "results" / "metrics_weights_calibration.json"
OUT_CSV = ROOT / "results" / "weights_calibration.csv"
OUT_PNG = ROOT / "results" / "weights_calibration.png"

PRIOR = {"stability": 0.35, "conduction": 0.20, "capacity": 0.25, "voltage": 0.20}
DIMS = ["pct_stability", "pct_conduction", "pct_capacity", "pct_voltage"]
# 非可牺牲项（工程约束）：稳定性/导电性/电压都必须保留最低权重，容量不得独大
MIN_STAB, MIN_COND, MIN_VOLT, MAX_CAP = 0.20, 0.10, 0.10, 0.50


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="用文献实测性能校准评分权重")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--ranking", default=str(RANKING))
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=60,
                    help="自助法重采样次数（权重不确定度）")
    args = ap.parse_args(argv)

    for p in (Path(args.reference), Path(args.ranking)):
        if not p.exists():
            print(f"[ERROR] 找不到 {p}")
            return 2

    ref = pd.read_csv(args.reference, encoding="utf-8-sig")
    rank = pd.read_csv(args.ranking, encoding="utf-8-sig")
    ref["specific_energy_Wh_kg"] = (ref["voltage_practical_V"]
                                    * ref["capacity_practical_mAh_g"])
    ext_cols = ["specific_energy_Wh_kg", "cycle_life_cycles",
                "rate_retention_pct", "thermal_onset_C"]
    for c in ext_cols:
        if c not in ref.columns:
            print(f"[ERROR] 参考表缺列 {c}")
            return 2
    ref["cost_score"] = -ref["cost_index"]        # 成本越低越好 -> 取负
    ext_cols.append("cost_score")
    # 溯源列必须一并带过来，否则逐行不确定度会静默退回 ±10%
    prov_cols = [c for c in ("uncertainty_pct", "source_type") if c in ref.columns]
    merged = rank.merge(ref[["formula"] + ext_cols + prov_cols], on="formula", how="inner")
    print(f"参考表 {len(ref)} 条 | 排序表 {len(rank)} 条 | 可对齐 {len(merged)} 条 | "
          f"外部目标 {len(ext_cols)} 个")
    if len(merged) < 12:
        print("[ERROR] 可对齐样本过少（<12），无法可靠校准")
        return 1

    for c in DIMS:
        if c not in merged.columns:
            print(f"[ERROR] 排序表缺列 {c}（先跑 screening_struct.py）")
            return 2

    # 多元目标：各目标 z 标准化后等权合成（避免某一量纲独大）
    Z = {}
    for c in ext_cols:
        v = merged[c].to_numpy(dtype=float)
        Z[c] = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
    multi = np.mean([Z[c] for c in ext_cols], axis=0)
    y = merged["specific_energy_Wh_kg"].to_numpy(dtype=float)
    X = merged[DIMS].to_numpy(dtype=float)

    def rho_against(w, target):
        return float(pd.Series(X @ np.asarray(w)).corr(pd.Series(target),
                                                       method="spearman"))

    def rho(w):
        return rho_against(w, y)

    def rho_multi(w):
        return rho_against(w, multi)

    # 单维度基线
    single = {}
    single_matrix = {}
    for dim in DIMS:
        w = np.zeros(len(DIMS))
        w[DIMS.index(dim)] = 1.0
        single[dim] = round(rho(w), 4)
        single_matrix[dim] = {c: round(rho_against(
            w, Z[c] * (merged[c].std() if merged[c].std() > 0 else 1.0)
            + merged[c].mean()), 4) for c in ext_cols}
    prior_vec = np.array([PRIOR["stability"], PRIOR["conduction"],
                           PRIOR["capacity"], PRIOR["voltage"]])
    prior_rho = round(rho(prior_vec), 4)

    # 网格搜索（步长 step，四维和为 1，满足约束）
    step = args.step
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    rows = []
    for a, b, c in itertools.product(grid, grid, grid):
        d = round(1.0 - a - b - c, 4)
        if d < 0 or d > 1:
            continue
        if a < MIN_STAB or b < MIN_COND or d < MIN_VOLT or c > MAX_CAP:
            continue
        w = (a, b, c, d)
        per = {c: rho_against(w, Z[c] * (merged[c].std() if merged[c].std() > 0 else 1.0)
                              + merged[c].mean()) for c in ext_cols}
        rows.append({"stability": a, "conduction": b, "capacity": c, "voltage": d,
                     "spearman_rho": round(rho(w), 4),
                     "rho_multi": round(rho_multi(w), 4),
                     "rho_min_across_targets": round(min(per.values()), 4)})
    res = pd.DataFrame(rows).sort_values("rho_multi", ascending=False).reset_index(drop=True)
    best = res.iloc[0]                      # 多目标（等权 z 合成）最优
    best_maximin = res.sort_values("rho_min_across_targets", ascending=False).iloc[0]
    print(f"\n网格点数 {len(res)} | 先验权重 ρ={prior_rho} | "
          f"最优 ρ={best['spearman_rho']}")
    print("最优权重: 稳定性 {stability:.2f} / 导电性 {conduction:.2f} / "
          "容量 {capacity:.2f} / 电压 {voltage:.2f}".format(**best.to_dict()))

    # 能量密度联合口径（全部用**模型侧**量，目标仍是实测比能量，避免循环论证）：
    #   评分 = 0.35×稳定性 + 0.20×导电性 + 0.45×(电压下界 × 理论容量) 的百分位
    merged["energy_density_model_Wh_kg"] = (merged["voltage_conservative_V"]
                                            * merged["Q_theoretical_mAh_g"])
    merged["pct_energy_model"] = merged["energy_density_model_Wh_kg"].rank(pct=True) * 100
    y_ser = pd.Series(y)
    score_energy = (merged["pct_stability"] * PRIOR["stability"]
                    + merged["pct_conduction"] * PRIOR["conduction"]
                    + merged["pct_energy_model"] * (PRIOR["capacity"] + PRIOR["voltage"]))
    rho_energy = round(float(score_energy.corr(y_ser, method="spearman")), 4)
    rho_fourdim = round(float((merged["stability_conservative_eV_atom"].rank(pct=True) * 0
                               + merged["pct_stability"] * PRIOR["stability"]
                               + merged["pct_conduction"] * PRIOR["conduction"]
                               + merged["pct_capacity"] * PRIOR["capacity"]
                               + merged["pct_voltage"] * PRIOR["voltage"]).corr(
                                   y_ser, method="spearman")), 4)

    # ---- 自助法：权重不确定度（重采样 + ±10% 观测噪声） ----
    rng = np.random.default_rng(42)
    unc = None
    if "uncertainty_pct" in merged.columns:
        unc = merged["uncertainty_pct"].to_numpy(dtype=float)
        print(f"逐行不确定度: 均值 {unc.mean():.1f}% （来自溯源分级）")
    boot_best = []
    n = len(merged)
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        # 观测噪声用**逐行不确定度**（来自 annotate_reference.py 的溯源分级）
        sig = unc[idx] / 100.0 if unc is not None else 0.10
        yb = multi[idx] * (1 + rng.normal(0, 1.0, size=n) * sig)
        best_w, best_r = None, -2.0
        for a, b, c in itertools.product(grid, grid, grid):
            d = round(1.0 - a - b - c, 4)
            if d < 0 or d > 1 or a < MIN_STAB or b < MIN_COND or d < MIN_VOLT or c > MAX_CAP:
                continue
            r = float(pd.Series(Xb @ np.array([a, b, c, d])).corr(
                pd.Series(yb), method="spearman"))
            if r > best_r:
                best_r, best_w = r, (a, b, c, d)
        if best_w:
            boot_best.append({"stability": best_w[0], "conduction": best_w[1],
                              "capacity": best_w[2], "voltage": best_w[3],
                              "rho_multi": round(best_r, 4)})
    boot = pd.DataFrame(boot_best)
    weight_uncertainty = {}
    if len(boot):
        for k in ("stability", "conduction", "capacity", "voltage"):
            s = boot[k]
            weight_uncertainty[k] = {
                "median": round(float(s.median()), 3),
                "p05": round(float(s.quantile(0.05)), 3),
                "p95": round(float(s.quantile(0.95)), 3),
                "share_equal_to_best": round(float((s == best[k]).mean()), 3),
            }

    payload = {
        "n_reference": int(len(ref)),
        "n_aligned": int(len(merged)),
        "target": "specific_energy_Wh_kg = V_practical × Q_practical",
        "single_dimension_rho": single,
        "single_dimension_rho_matrix": single_matrix,
        "external_targets": ext_cols,
        "multi_objective": {
            "definition": "各外部目标 z 标准化后等权平均，再与评分做 Spearman ρ",
            "best_rho_multi": float(best["rho_multi"]),
            "best_weights_multi": {k: float(best[k]) for k in
                                   ("stability", "conduction", "capacity", "voltage")},
            "best_maximin_rho_min": float(best_maximin["rho_min_across_targets"]),
            "best_weights_maximin": {k: float(best_maximin[k]) for k in
                                     ("stability", "conduction", "capacity", "voltage")},
        },
        "weight_uncertainty_bootstrap": {
            "n_resamples": int(args.bootstrap),
            "noise": ("per_row_uncertainty_pct" if "uncertainty_pct" in merged.columns
                      else 0.10),
            "mean_uncertainty_pct": (None if "uncertainty_pct" not in merged.columns
                                     else round(float(merged["uncertainty_pct"].mean()), 2)),
            "provenance_counts": (None if "source_type" not in merged.columns
                                  else merged["source_type"].value_counts().to_dict()),
            "per_weight": weight_uncertainty,
            "note": ("自助法（重采样 + ±10% 观测噪声）给出的权重区间；"
                     "区间宽说明参考表不足以唯一确定权重，应读作合理范围。"),
        },
        "prior_weights": PRIOR,
        "prior_rho": prior_rho,
        "best_weights": {k: float(best[k]) for k in
                         ("stability", "conduction", "capacity", "voltage")},
        "best_rho": float(best["spearman_rho"]),
        "improvement_over_prior": round(float(best["spearman_rho"]) - prior_rho, 4),
        "constraints": {"min_stability": MIN_STAB, "min_conduction": MIN_COND,
                        "min_voltage": MIN_VOLT, "max_capacity": MAX_CAP},
        "energy_density_variant_rho": rho_energy,
        "four_dimension_rho_recomputed": rho_fourdim,
        "top10_weight_sets": res.head(10).to_dict("records"),
        "caveats": [
            "参考表是文献典型值，同一材料不同报道可差 ±10%，权重应读作合理区间",
            "实测比能量只覆盖 36 条候选中的一部分（对齐 "
            f"{len(merged)} 条），且未含循环寿命/倍率/成本",
            "ρ 是排序一致性，不等于绝对性能预测精度",
        ],
        "note": ("校准只用于把权重从「先验」变成「有依据」；"
                 "screening_struct.py 会读取本文件的 best_weights 作为评分权重"
                 "（可用 --prior-weights 强制使用先验）。"),
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    # 供 screening_struct.py 直接消费的推荐权重文件
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].hist(res["spearman_rho"], bins=25, color="#4393c3")
    axes[0].axvline(prior_rho, color="#b2182b", ls="--", lw=1.4, label=f"先验 ρ={prior_rho}")
    axes[0].axvline(best["spearman_rho"], color="#1b7837", ls="-", lw=1.4,
                    label=f"最优 ρ={best['spearman_rho']}")
    axes[0].set_xlabel("Spearman ρ（评分 vs 实测比能量）")
    axes[0].set_ylabel("网格点数")
    axes[0].set_title("权重网格搜索分布")
    axes[0].legend(fontsize=8)
    dims_cn = {"pct_stability": "稳定性", "pct_conduction": "导电性",
               "pct_capacity": "容量", "pct_voltage": "电压"}
    axes[1].bar([dims_cn[d] for d in DIMS], [single[d] for d in DIMS], color="#4393c3")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("单维度 Spearman ρ")
    axes[1].set_title("各维度单独与实测比能量的一致性")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print("\n各维度单独 ρ:", {dims_cn[k]: v for k, v in single.items()})
    print("多目标（等权 z 合成）: 最优 ρ_multi =", round(best["rho_multi"], 4),
          "权重", {k: float(best[k]) for k in ("stability", "conduction", "capacity", "voltage")})
    print("maximin 稳健权重:", {k: float(best_maximin[k]) for k in
                                ("stability", "conduction", "capacity", "voltage")},
          "| 最差目标 ρ =", round(best_maximin["rho_min_across_targets"], 4))
    if weight_uncertainty:
        print("权重不确定度（自助法 90% 区间）:")
        for k, v in weight_uncertainty.items():
            print(f"  {dims_cn['pct_' + k]:6s} 中位数 {v['median']:.2f} "
                  f"[{v['p05']:.2f}, {v['p95']:.2f}]")
    print(f"能量密度联合口径 ρ（用实测能量密度作对照）= {rho_energy}")
    print(f"[SAVE] results/{OUT_JSON.name} / {OUT_CSV.name} / {OUT_PNG.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())