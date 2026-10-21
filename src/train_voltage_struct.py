# -*- coding: utf-8 -*-
"""电压模型的**结构消歧**对照实验（对应报告附录 G）。

背景：真实电压数据集（MP 插入电极，2,672 条）里同一化学式可能对应多个框架，
同成分电压 std 中位数 **0.218 V**——这是当前最大的误差来源：

| 协议 | 集成 MAE |
|---|---|
| 随机 5 折（成分特征） | 0.4187 V |
| GroupKFold（按框架分组） | 0.4806 V |

本脚本给成分特征补上**结构信息**（嵌锂相/脱锂相各 21 维几何描述符 + 脱锂收缩率 + 原子数差，
见 ``prepare_voltage_struct.py``），看误差能否降下来。

三组对照（同数据、同超参数、同协议）：``comp`` / ``struct`` / ``comp+struct``。

用法::

    python src/prepare_voltage_struct.py     # 首次：生成结构特征表
    python src/train_voltage_struct.py

输出::

    results/metrics_voltage_struct.json
    results/voltage_struct_best.pt? -> 实际为 models_voltage_struct.pkl（sklearn 模型束）
    results/oof_voltage_struct.csv, parity_voltage_struct.png, importance_voltage_struct.png
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
DATA = ROOT / "data" / "processed" / "mp_voltage_struct.csv.gz"
TARGET = "average_voltage_V"
REFERENCE = {"comp_random": 0.4187, "comp_group": 0.4806}


def _oof(model, X, y, protocol, groups=None):
    if protocol == "random":
        return np.asarray(cross_val_predict(
            model, X, y, cv=KFold(5, shuffle=True, random_state=RNG), n_jobs=1), dtype=float)
    return np.asarray(cross_val_predict(
        model, X, y, cv=GroupKFold(n_splits=5), groups=groups, n_jobs=1), dtype=float)


def _score(y, p) -> dict:
    err = np.asarray(p) - np.asarray(y)
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": round(float(np.mean(np.abs(err))), 4),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "R2": round(float(1 - np.sum(err ** 2) / ss), 4)}


def evaluate(X, y, groups, protocol, label):
    models = build_models()
    preds, out = {}, {}
    t0 = time.time()
    for name, m in models.items():
        preds[name] = _oof(m, X, y, protocol, groups)
        out[name] = _score(y, preds[name])
    ens = 0.5 * (preds["RF"] + preds["HGB"])
    out["Ensemble"] = _score(y, ens)
    m = out["Ensemble"]
    print(f"  {label:12s} [{protocol:6s}] MAE={m['MAE']:.4f} V  RMSE={m['RMSE']:.4f}  "
          f"R2={m['R2']:.4f}   ({time.time()-t0:.0f}s)")
    return {"metrics": out, "oof": preds, "ens": ens}


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="电压模型结构消歧对照实验")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--tag", default="voltage_struct")
    ap.add_argument("--skip-group", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="抽样冒烟（0=全量）")
    args = ap.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        print(f"[ERROR] 找不到 {path}（先跑 python src/prepare_voltage_struct.py）")
        return 2

    df = pd.read_csv(path, encoding="utf-8-sig")
    if args.subset and args.subset < len(df):
        df = df.sample(n=args.subset, random_state=RNG).reset_index(drop=True)
        print(f"[冒烟模式] 抽样 {args.subset} 条")
    if args.subset and args.tag == "voltage_struct":
        args.tag = "voltage_struct_smoke"
        print(f"[冒烟模式] 输出 tag 改为 {args.tag}")

    struct_cols = [c for c in df.columns if c.startswith(("lith_", "frame_", "delta_"))]
    print(f"样本 {len(df)} | 唯一成分 {df['formula_canon'].nunique()} | 结构特征 {len(struct_cols)} 维")

    formulas = df["formula"].astype(str).tolist()
    els = build_universe(formulas)
    Xc, failed = featurize_dataframe(formulas, element_list=els, verbose=False)
    keep = [i for i in range(len(df)) if i not in set(failed)]
    Xc = Xc.iloc[keep].reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=float)[keep]
    groups = df["formula_canon"].astype(str).to_numpy()[keep]
    Xs = df[struct_cols].iloc[keep].reset_index(drop=True)
    print(f"成分特征 {Xc.shape[1]} 维 | 建模样本 {len(y)}")

    feature_sets = {"comp": Xc, "struct": Xs, "comp+struct": pd.concat([Xc, Xs], axis=1)}

    results = {}
    print("\n=== random 5 折 ===")
    for label, X in feature_sets.items():
        results.setdefault(label, {})["random"] = evaluate(X, y, groups, "random", label)
    if not args.skip_group:
        print("\n=== GroupKFold（按成分分组）===")
        for label in ("comp", "comp+struct"):
            results[label]["group"] = evaluate(feature_sets[label], y, groups, "group", label)

    best_label = "comp+struct" if "comp+struct" in results else "comp"
    best = results[best_label]["random"]
    X_best = feature_sets[best_label]
    names = X_best.columns.tolist()

    pd.DataFrame({
        "y_true": y, "pred_RF": best["oof"]["RF"], "pred_HGB": best["oof"]["HGB"],
        "pred_ens": best["ens"],
        "spread": np.abs(best["oof"]["RF"] - best["oof"]["HGB"]) / 2.0,
    }).to_csv(OUT / f"oof_{args.tag}.csv", index=False, encoding="utf-8-sig")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo, hi = float(y.min()) - 0.3, float(y.max()) + 0.3
    ax.scatter(y, best["ens"], s=10, alpha=0.35, edgecolors="none", color="#4393c3")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    met = best["metrics"]["Ensemble"]
    ax.set_xlabel("真实平均电压 (V)")
    ax.set_ylabel("预测平均电压 (V)")
    ax.set_title(f"电压结构消歧（{best_label}, 5 折随机 OOF, n={len(y)}）\n"
                 f"MAE={met['MAE']:.3f} V（成分基线 {REFERENCE['comp_random']:.3f} V）"
                 f"  R²={met['R2']:.3f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"parity_{args.tag}.png", dpi=150)
    plt.close(fig)

    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RNG))])
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=RNG)
    fitted = {"RF": rf.fit(X_best, y), "HGB": hgb.fit(X_best, y)}
    imp = fitted["RF"].named_steps["m"].feature_importances_
    n = min(15, len(imp))
    order = np.argsort(imp)[::-1][:n]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color="#4393c3")
    ax.set_xlabel("RandomForest 特征重要性")
    ax.set_title(f"电压模型 Top {n} 特征（{best_label}）")
    fig.tight_layout()
    fig.savefig(OUT / f"importance_{args.tag}.png", dpi=150)
    plt.close(fig)

    import pickle
    with open(OUT / f"models_{args.tag}.pkl", "wb") as f:
        pickle.dump({"models": fitted, "feature_names": names, "element_list": els,
                     "struct_cols": struct_cols, "feature_set": best_label,
                     "target": TARGET}, f)

    payload = {
        "mode": "subset-smoke" if args.subset else "full",
        "n_samples": int(len(y)),
        "n_unique_compositions": int(len(set(groups))),
        "struct_features": len(struct_cols),
        "reference_comp_only": REFERENCE,
        "summary": {label: {proto: v["metrics"]["Ensemble"] for proto, v in res.items()}
                    for label, res in results.items()},
        "full_metrics": {label: {proto: v["metrics"] for proto, v in res.items()}
                         for label, res in results.items()},
        "best": best_label,
    }
    with open(OUT / f"metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print("  电压 MAE 对比（V，越小越好）")
    print("=" * 74)
    print(f"  {'特征集':12s} {'random':>10s} {'group':>10s}   R²(random)")
    for label in ("comp", "struct", "comp+struct"):
        r = results[label]["random"]["metrics"]["Ensemble"]
        g = results.get(label, {}).get("group", {}).get("metrics", {}).get("Ensemble", {})
        print(f"  {label:12s} {r['MAE']:>10.4f} {g.get('MAE', float('nan')):>10.4f}   {r['R2']:>8.4f}")
    print("=" * 74)
    print("[SAVE] results/metrics_%s.json / oof_ / parity_ / importance_ / models_" % args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())