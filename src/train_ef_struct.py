# -*- coding: utf-8 -*-
"""结构感知的成分模型：验证「引入结构信息能否突破成分不可约上限」。

实验设计（同一份 18,928 条钙钛矿数据、同一组超参数、同一套评估协议）：

======== ==========================================================
特征集    内容
======== ==========================================================
comp     33 统计特征 + 元素分数（与 train_ef.py 完全一致）
struct   21 维结构描述符（晶格/体积/近邻与原子对距离统计，见 structure_features.py）
comp+struct 两者拼接
======== ==========================================================

两种划分协议：

- **random**：5 折随机划分，与 matbench 榜单口径一致，但同一成分的不同占位会跨折
  → 对"记住成分均值"的模型有利；
- **group**：按 ``formula_canon`` 分组的 GroupKFold，衡量"对**没见过的成分**"的泛化。

预期：comp 在 random 下严重受限（同成分目标 std 中位数 0.42 eV/胞），
而 struct / comp+struct 能靠几何描述符区分同成分的不同占位与畸变，
从而把 MAE 从 0.48 压到更低——这正是通往 CGCNN（GPU）的中间验证。

用法::

    python src/prepare_struct_features.py    # 首次：生成结构描述符表
    python src/train_ef_struct.py

输出::

    results/metrics_ef_struct.json     三种特征集 × 两种协议的完整指标
    results/oof_ef_struct.csv          comp+struct 的 OOF 预测
    results/parity_ef_struct.png       平差图
    results/importance_ef_struct.png   特征重要性 Top15
    results/models_ef_struct.pkl       comp+struct 全量拟合模型束
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features import build_universe, featurize_dataframe  # noqa: E402
from models import build_models, set_plot_style  # noqa: E402

RNG = 42
OUT = ROOT / "results"
STRUCT_PATH = ROOT / "data" / "processed" / "perovskites_struct.csv.gz"
TARGET = "e_form_eV_cell"
REFERENCE_COMP_MAE = 0.4837        # train_ef.py 的成分基线（random 5 折，v1.1.0 起）


def _oof(model, X, y, protocol: str, groups=None):
    if protocol == "random":
        return np.asarray(cross_val_predict(model, X, y,
                                            cv=KFold(5, shuffle=True, random_state=RNG),
                                            n_jobs=1), dtype=float)
    return np.asarray(cross_val_predict(model, X, y, cv=GroupKFold(n_splits=5),
                                        groups=groups, n_jobs=1), dtype=float)


def _score(y, p) -> dict:
    err = np.asarray(p) - np.asarray(y)
    ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": round(float(np.mean(np.abs(err))), 4),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "R2": round(float(1 - np.sum(err ** 2) / ss), 4),
    }


def evaluate_features(X, y, groups, protocol: str, label: str) -> dict:
    """对一组特征跑 RF/HGB/集成，返回三者的指标。"""
    models = build_models()
    out, preds = {}, {}
    t0 = time.time()
    for name, model in models.items():
        preds[name] = _oof(model, X, y, protocol, groups)
        out[name] = _score(y, preds[name])
    ens = 0.5 * (preds["RF"] + preds["HGB"])
    out["Ensemble"] = _score(y, ens)
    m = out["Ensemble"]
    print(f"  {label:12s} [{protocol:6s}] MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
          f"R2={m['R2']:.4f}   ({time.time() - t0:.0f}s)")
    return {"metrics": out, "oof": preds, "ens": ens}


def plot_parity(y, pred, tag: str, title_extra: str = ""):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.05 * (hi - lo)
    lims = [lo - pad, hi + pad]
    ax.scatter(y, pred, s=9, alpha=0.30, edgecolors="none", color="#2166ac")
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("真实形成能 (eV/5原子胞)")
    ax.set_ylabel("预测形成能 (eV/5原子胞)")
    m = _score(y, pred)
    ax.set_title(f"结构感知模型 平差图（5 折随机 OOF, n={len(y)}）\n"
                 f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}"
                 + (f"\n{title_extra}" if title_extra else ""), fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"parity_{tag}.png")
    plt.close(fig)


def plot_importance(model, names, tag: str):
    imp = model.named_steps["model"].feature_importances_
    n = min(15, len(imp))
    order = np.argsort(imp)[::-1][:n]
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color="#b2182b", alpha=0.9)
    ax.set_xlabel("RandomForest 特征重要性")
    ax.set_title(f"结构感知模型：特征重要性 Top {n}")
    fig.tight_layout()
    fig.savefig(OUT / f"importance_{tag}.png")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="结构感知形成能模型（CPU）")
    ap.add_argument("--data", default=str(STRUCT_PATH))
    ap.add_argument("--tag", default="ef_struct")
    ap.add_argument("--skip-group", action="store_true", help="只跑 random 协议（更快）")
    args = ap.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        print(f"[ERROR] 找不到 {path}")
        print("        请先运行: python src/prepare_struct_features.py")
        return 2

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"样本: {len(df)}  唯一成分: {df['formula_canon'].nunique()}")

    struct_cols = [c for c in df.columns if c not in
                   ("index", "formula_canon", TARGET, "n_sites")]
    struct_cols.append("n_sites")
    print(f"结构描述符: {len(struct_cols)} 维")

    # ---- 成分特征 ----
    formulas = df["formula_canon"].astype(str).tolist()
    element_list = build_universe(formulas)
    Xc, failed = featurize_dataframe(formulas, element_list=element_list, verbose=False)
    keep = [i for i in range(len(df)) if i not in set(failed)]
    if failed:
        print(f"成分特征化失败 {len(failed)} 条（丢弃）")
    Xc = Xc.iloc[keep].reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=float)[keep]
    groups = df["formula_canon"].astype(str).to_numpy()[keep]
    Xs = df[struct_cols].iloc[keep].reset_index(drop=True)
    print(f"成分特征 {Xc.shape[1]} 维 | 总样本 {len(y)}")

    feature_sets = {
        "comp": Xc,
        "struct": Xs,
        "comp+struct": pd.concat([Xc, Xs], axis=1),
    }

    OUT.mkdir(exist_ok=True)
    print("\n=== random 5 折（matbench 同口径）===")
    results = {}
    for label, X in feature_sets.items():
        results.setdefault(label, {})["random"] = evaluate_features(X, y, groups, "random", label)

    if not args.skip_group:
        print("\n=== GroupKFold（按成分分组，衡量对未见成分的泛化）===")
        for label in ("comp", "comp+struct"):
            results[label]["group"] = evaluate_features(feature_sets[label], y, groups,
                                                        "group", label)

    # ---- 保存 comp+struct 的产物 ----
    best = results["comp+struct"]["random"]
    X_best = feature_sets["comp+struct"]
    names = X_best.columns.tolist()

    oof = pd.DataFrame({
        "y_true": y,
        "pred_RF": best["oof"]["RF"],
        "pred_HGB": best["oof"]["HGB"],
        "pred_ens": best["ens"],
        "spread": np.abs(best["oof"]["RF"] - best["oof"]["HGB"]) / 2.0,
    })
    oof.to_csv(OUT / f"oof_{args.tag}.csv", index=False, encoding="utf-8-sig")

    plot_parity(y, best["ens"], args.tag,
                title_extra=f"成分基线 MAE={REFERENCE_COMP_MAE:.3f} → 结构感知 {best['metrics']['Ensemble']['MAE']:.3f}")
    fitted = {name: m.fit(X_best, y) for name, m in build_models().items()}
    plot_importance(fitted["RF"], names, args.tag)

    import pickle
    with open(OUT / f"models_{args.tag}.pkl", "wb") as f:
        pickle.dump({"models": fitted, "feature_names": names,
                     "element_list": element_list, "struct_cols": struct_cols,
                     "uses_structure": True, "target": TARGET}, f)

    payload = {
        "target": TARGET,
        "n_samples": int(len(y)),
        "n_unique_compositions": int(len(set(groups))),
        "reference_comp_only_MAE": REFERENCE_COMP_MAE,
        "summary": {
            label: {proto: v["metrics"]["Ensemble"] for proto, v in res.items()}
            for label, res in results.items()
        },
        "full_metrics": {label: {proto: v["metrics"] for proto, v in res.items()}
                         for label, res in results.items()},
    }
    with open(OUT / f"metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 汇总表 ----
    print("\n" + "=" * 74)
    print("  形成能 MAE 对比（eV/5原子胞，越小越好）")
    print("=" * 74)
    print(f"  {'特征集':12s} {'random':>10s} {'group':>10s}   R²(random)")
    for label in ("comp", "struct", "comp+struct"):
        r = results[label]["random"]["metrics"]["Ensemble"]
        g = results.get(label, {}).get("group", {}).get("metrics", {}).get("Ensemble", {})
        print(f"  {label:12s} {r['MAE']:>10.4f} {g.get('MAE', float('nan')):>10.4f}   {r['R2']:>8.4f}")
    print("=" * 74)
    print(f"[SAVE] results/metrics_{args.tag}.json / oof_ / parity_ / importance_ / models_")
    return 0


if __name__ == "__main__":
    sys.exit(main())