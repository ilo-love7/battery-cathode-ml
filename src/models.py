# -*- coding: utf-8 -*-
"""共享建模工具：模型构建、5 折交叉验证评估、图表样式。

模型组合（纯 CPU，符合项目约束）：
- RF : 中位数插补 + RandomForest(300 棵, min_samples_leaf=2)
- HGB: HistGradientBoosting(400 轮, lr=0.06)，原生支持 NaN 特征
- 集成: 两模型 OOF 预测的算术平均
不确定性代理: 两模型 OOF 预测的标准差（模型间离散度，越小越可信）。
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

RNG = 42
FIG_DIR = None  # 由各脚本设置


def set_plot_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def build_models():
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            n_jobs=-1, random_state=RNG)),
    ])
    hgb = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RNG)
    return {"RF": rf, "HGB": hgb}


def _metrics(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": rmse,
            "R2": float(r2_score(y, p))}


def evaluate_dataset(X, y, tag, outdir, feature_names):
    """对 (X, y) 做两模型 + 集成的 5 折 OOF 评估，落盘结果/图/模型。

    返回 dict: metrics, oof(DataFrame), models(全量拟合的 {name: model})
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    models = build_models()
    preds, metrics = {}, {}
    t0 = time.time()
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=kf, n_jobs=1)
        preds[name] = np.asarray(p, dtype=float)
        metrics[name] = _metrics(y, preds[name])
        print(f"  [{tag}] {name:3s} OOF: MAE={metrics[name]['MAE']:.4f} "
              f"RMSE={metrics[name]['RMSE']:.4f} R2={metrics[name]['R2']:.4f} "
              f"({time.time() - t0:.0f}s)")
    ens = 0.5 * preds["RF"] + 0.5 * preds["HGB"]
    metrics["Ensemble"] = _metrics(y, ens)
    spread = np.abs(preds["RF"] - preds["HGB"]) / 2.0   # 两模型离散度
    print(f"  [{tag}] ENS OOF: MAE={metrics['Ensemble']['MAE']:.4f} "
          f"RMSE={metrics['Ensemble']['RMSE']:.4f} "
          f"R2={metrics['Ensemble']['R2']:.4f} "
          f"(总耗时 {time.time() - t0:.0f}s)")

    # ---- 全量重拟合，供筛选阶段使用 ----
    fitted = {name: m.fit(X, y) for name, m in build_models().items()}

    # ---- 预测表 ----
    import pandas as pd
    oof = pd.DataFrame({"y_true": np.asarray(y, dtype=float),
                        "pred_RF": preds["RF"],
                        "pred_HGB": preds["HGB"],
                        "pred_ens": ens,
                        "spread": spread})

    # ---- 图 1: 平差图 ----
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lims = [float(np.min(y)), float(np.max(y))]
    pad = 0.05 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    ax.scatter(y, ens, s=9, alpha=0.30, edgecolors="none", color="#2166ac")
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("真实值"); ax.set_ylabel("集成 OOF 预测")
    m = metrics["Ensemble"]
    ax.set_title(f"{tag} 平差图（5 折 OOF, n={len(y)}）\n"
                 f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/parity_{tag}.png")
    plt.close(fig)

    # ---- 图 2: RF 特征重要性 Top15 ----
    imp = fitted["RF"].named_steps["model"].feature_importances_
    order = np.argsort(imp)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([feature_names[i] for i in order], imp[order], color="#b2182b", alpha=0.85)
    ax.set_xlabel("RandomForest 特征重要性")
    ax.set_title(f"{tag}: 特征重要性 Top 15")
    fig.tight_layout()
    fig.savefig(f"{outdir}/importance_{tag}.png")
    plt.close(fig)

    # ---- 指标 JSON ----
    with open(f"{outdir}/metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {"metrics": metrics, "oof": oof, "models": fitted}


def save_pickle(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
