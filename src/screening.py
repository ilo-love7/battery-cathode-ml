# -*- coding: utf-8 -*-
"""任务 C：候选正极成分筛选（集成预测 + 不确定性 + 域检查 + 理论容量）。

流程:
1) 读入 data/cathode_candidates.csv（42 个候选正极成分，含家族/可脱锂数/文献电压）；
2) 用 models_gap.pkl 与 models_ef.pkl 预测带隙与形成能（RF+HGB 集成，离散度作不确定性）；
3) 域检查: 候选成分的每种元素是否出现在训练集元素宇宙中（出现与否 -> 域内/域外标记）；
4) 理论比容量 Q = n_Li * F/3.6 / M (mAh/g)；
5) 综合评分 = 0.40*稳定性百分位(-E_form) + 0.30*导电性百分位(exp(-E_gap))
             + 0.30*理论容量百分位，输出排序表与排名图。

评分仅为成分级"优先级排序"，不替代 DFT/实验验证；不确定性高或域外的
候选应降权解读（表中单独标记）。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import featurize_one  # noqa: E402
from formula import parse_formula  # noqa: E402
from models import load_pickle, set_plot_style  # noqa: E402
from voltage import theoretical_capacity_mAh_g  # noqa: E402

OUT = ROOT / "results"
W_STAB, W_COND, W_CAP = 0.40, 0.30, 0.30


def training_element_universe():
    """两个训练集元素宇宙的并集。"""
    universe = set()
    for csv in ["data/processed/expt_gap.csv", "data/processed/perovskites_ef.csv"]:
        df = pd.read_csv(ROOT / csv)
        for f in df["formula_canon"]:
            try:
                universe |= set(parse_formula(f).keys())
            except ValueError:
                pass
    return universe


def predict_with_bundle(bundle, comps):
    """用保存的模型束预测 (ensemble, spread)。特征列与训练时一致。"""
    els = bundle.get("element_list")
    X = pd.DataFrame([featurize_one(c, element_list=els) for c in comps],
                     columns=bundle["feature_names"])
    rf = bundle["models"]["RF"].predict(X)
    hgb = bundle["models"]["HGB"].predict(X)
    return 0.5 * (rf + hgb), np.abs(rf - hgb) / 2.0


def main():
    OUT.mkdir(exist_ok=True)
    cand = pd.read_csv(ROOT / "data/cathode_candidates.csv", encoding="utf-8-sig")
    print(f"候选成分: {len(cand)} 条")

    universe = training_element_universe()
    print(f"训练集元素宇宙: {len(universe)} 种元素")

    bundle_gap = load_pickle(OUT / "models_gap.pkl")
    bundle_ef = load_pickle(OUT / "models_ef.pkl")

    comps = cand["formula"].tolist()
    eg, eg_sd = predict_with_bundle(bundle_gap, comps)
    ef, ef_sd = predict_with_bundle(bundle_ef, comps)

    rows = []
    for i, r in cand.iterrows():
        try:
            counts = parse_formula(r["formula"])
            q = theoretical_capacity_mAh_g(counts, float(r["nLi_extract"]))
            from features import ELEMENTS
            molar = sum(c * ELEMENTS[el]["mass"] for el, c in counts.items())
            atoms = float(sum(counts.values()))
            unseen = sorted(set(counts) - universe)
        except ValueError:
            q, molar, atoms, unseen = np.nan, np.nan, np.nan, ["PARSE_FAIL"]
        rows.append({
            "name": r["name"], "formula": r["formula"], "family": r["family"],
            "Egap_pred_eV": eg[i], "Egap_spread": eg_sd[i],
            "Eform_pred_eV_per_fu": ef[i], "Eform_spread": ef_sd[i],
            "atoms_per_fu": atoms,
            "Q_theoretical_mAh_g": q, "molar_mass": molar,
            "lit_voltage_V": r["avg_voltage_V_lit"],
            "unseen_elements": ",".join(unseen) if unseen else "",
        })

    df = pd.DataFrame(rows)

    # ---- 评分（百分位秩，越接近 100 越好）----
    # 稳定性按每原子形成能排名: 训练目标是 eV/5原子单胞(广延量),
    # 跨不同大小的化学式比较必须归一到每原子。
    df["Eform_pred_eV_per_atom"] = df["Eform_pred_eV_per_fu"] / df["atoms_per_fu"]
    df["pct_stability"] = df["Eform_pred_eV_per_atom"].rank(pct=True, ascending=False) * 100
    df["pct_conduction"] = np.exp(-df["Egap_pred_eV"].clip(lower=0)).rank(pct=True) * 100
    df["pct_capacity"] = df["Q_theoretical_mAh_g"].rank(pct=True) * 100
    df["score"] = (W_STAB * df["pct_stability"] + W_COND * df["pct_conduction"]
                   + W_CAP * df["pct_capacity"])

    # ---- 域外 / 高不确定性标记 ----
    spread_hi = max(df["Eform_spread"].quantile(0.75), 1e-6)
    df["flag"] = ""
    df.loc[df["unseen_elements"] != "", "flag"] += "域外元素;"
    df.loc[df["Eform_spread"] > spread_hi, "flag"] += "模型分歧高;"
    df.loc[df["Egap_pred_eV"] <= 0, "flag"] += "带隙外推(≤0);"
    df.loc[df["flag"] == "", "flag"] = "域内&稳定"

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    out_csv = OUT / "cathode_ranking.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"排序表 -> {out_csv}")
    print(df[["rank", "name", "formula", "Egap_pred_eV", "Eform_pred_eV_per_atom",
              "Q_theoretical_mAh_g", "score", "flag"]].head(12).to_string(index=False))

    # ---- 排名图（Top 20，按家族着色）----
    set_plot_style()
    top = df.head(20).iloc[::-1]
    fam_colors = {fam: c for fam, c in zip(
        top["family"].unique(),
        ["#4393c3", "#d6604d", "#5aae61", "#762a83", "#e08214", "#1b7837", "#8073ac"])}
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.barh([f"{n} ({f})" for n, f in zip(top["name"], top["formula"])],
            top["score"],
            color=[fam_colors[f] for f in top["family"]])
    ax.set_xlabel("综合评分 (0-100)")
    ax.set_title(f"候选正极成分筛选 Top 20\n"
                 f"评分 = {W_STAB:.2f}×稳定性 + {W_COND:.2f}×导电性 + {W_CAP:.2f}×理论容量")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.9) for c in fam_colors.values()]
    ax.legend(handles, fam_colors.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cathode_ranking.png")
    plt.close(fig)
    print("排名图 -> results/cathode_ranking.png")


if __name__ == "__main__":
    main()
