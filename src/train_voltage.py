# -*- coding: utf-8 -*-
"""训练平均电压预测模型（支持合成演示数据与 Materials Project 真实数据）。

数据来源两种，用 ``--data`` 切换：

- ``data/battery_voltage.csv``：合成演示数据（默认）。**指标只能验证流程可跑通，
  不含真实物理信息**，不要对外引用为结论。
- ``data/processed/mp_voltage.csv``：Materials Project 插入电极真实电压标签
  （``python src/fetch_mp_battery.py --what electrodes`` + ``python src/prepare_mp_data.py``）。
  这才是可用于研究的真实数据。

流程：成分特征（33 统计 + 元素分数）→ RF + HGB 双模型 → 5 折 OOF 评估 →
平差图 / 特征重要性 / OOF 预测 / 模型束。若数据带分组列（如
``framework_formula``），额外跑一次 GroupKFold，用来暴露「同框架不同锂含量」
造成的成分级重复，给出更诚实的泛化估计。

用法::

    # 真实数据（推荐）
    python src/train_voltage.py --data data/processed/mp_voltage.csv --tag voltage

    # 合成数据（保留作对照）
    python src/train_voltage.py --data data/battery_voltage.csv --tag voltage_synthetic

输出（results/）::

    metrics_{tag}.json, parity_{tag}.png, importance_{tag}.png,
    oof_{tag}.csv, models_{tag}.pkl
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features import build_universe, featurize_dataframe  # noqa: E402
from models import set_plot_style  # noqa: E402

RNG = 42
OUTPUT_DIR = ROOT / "results"
SYNTHETIC_DATA = ROOT / "data" / "battery_voltage.csv"


def load_data(path: Path) -> pd.DataFrame:
    """读取数据；默认合成数据不存在时按需生成。"""
    if path.exists():
        print(f"读取数据: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")
    if path == SYNTHETIC_DATA:
        print("未找到电压数据，生成演示用合成数据 ...")
        from generate_voltage_data import generate_voltage_data
        df = generate_voltage_data()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df
    raise SystemExit(f"[ERROR] 数据不存在: {path}")


def build_voltage_models() -> dict:
    """RF + HGB（与 models.build_models 同一组超参数）。"""
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                        n_jobs=-1, random_state=RNG)),
    ])
    hgb = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RNG)
    return {"RF": rf, "HGB": hgb}


def build_features(df: pd.DataFrame, target: str = "average_voltage_V",
                   group_col: str = "framework_formula"):
    """化学式 -> (X, y, groups, element_list)。特征化失败样本与标签同步剔除。"""
    formulas = df["formula"].astype(str).tolist()
    element_list = build_universe(formulas)
    X, failed = featurize_dataframe(formulas, element_list=element_list, verbose=False)
    keep = [i for i in range(len(df)) if i not in set(failed)]
    if failed:
        print(f"特征化失败 {len(failed)} 条（已与标签同步丢弃）")
    X = X.iloc[keep].reset_index(drop=True)
    y = df[target].to_numpy(dtype=float)[keep]
    groups = (df[group_col].fillna("unknown").astype(str).to_numpy()[keep]
              if group_col and group_col in df.columns else None)
    return X, y, groups, element_list


def _metrics(y, p) -> dict:
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def evaluate_voltage(X, y):
    """5 折 OOF 评估，返回 (metrics, preds, ens_pred, fitted_models)。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    models = build_voltage_models()
    preds, metrics = {}, {}
    t0 = time.time()

    for name, model in models.items():
        oof = cross_val_predict(model, X, y, cv=kf, n_jobs=1)
        preds[name] = np.asarray(oof, dtype=float)
        metrics[name] = _metrics(y, preds[name])
        print(f"  [{name}] MAE={metrics[name]['MAE']:.4f} V, "
              f"RMSE={metrics[name]['RMSE']:.4f} V, R2={metrics[name]['R2']:.4f}")

    ens_pred = 0.5 * preds["RF"] + 0.5 * preds["HGB"]
    metrics["Ensemble"] = _metrics(y, ens_pred)
    print(f"\n[Ensemble] MAE={metrics['Ensemble']['MAE']:.4f} V, "
          f"RMSE={metrics['Ensemble']['RMSE']:.4f} V, "
          f"R2={metrics['Ensemble']['R2']:.4f}  ({time.time() - t0:.0f}s)")

    fitted = {name: m.fit(X, y) for name, m in build_voltage_models().items()}
    return metrics, preds, ens_pred, fitted


def evaluate_grouped(X, y, groups) -> dict:
    """按分组列做 GroupKFold（同框架不同锂含量不跨折泄漏），返回三种预测器指标。"""
    if groups is None or len(np.unique(groups)) < 5:
        return {}
    print(f"\n[GroupKFold] 按 {len(np.unique(groups))} 个分组评估（更诚实的泛化估计）...")
    cv = GroupKFold(n_splits=5)
    out, preds = {}, {}
    for name, model in build_voltage_models().items():
        p = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=1)
        preds[name] = np.asarray(p, dtype=float)
        out[name] = _metrics(y, preds[name])
        print(f"  [{name}] MAE={out[name]['MAE']:.4f} V, R2={out[name]['R2']:.4f}")
    ens = 0.5 * preds["RF"] + 0.5 * preds["HGB"]
    out["Ensemble"] = _metrics(y, ens)
    print(f"  [Ensemble] MAE={out['Ensemble']['MAE']:.4f} V, "
          f"R2={out['Ensemble']['R2']:.4f}")
    return out


def plot_results(y_true, y_pred_ens, tag: str):
    """平差图。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_plot_style()

    fig, ax = plt.subplots(figsize=(6, 5.5))
    lo = max(0.0, float(np.min(y_true)) - 0.5)
    hi = float(np.max(y_true)) + 0.5
    ax.scatter(y_true, y_pred_ens, s=22, alpha=0.45, edgecolors="none", color="#219150")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Average Voltage (V)", fontsize=11)
    ax.set_ylabel("Predicted Voltage (V)", fontsize=11)
    ax.set_title(f"Voltage Prediction ({tag}, 5-fold OOF, n={len(y_true)})\n"
                 f"MAE={mean_absolute_error(y_true, y_pred_ens):.3f}V  "
                 f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred_ens)):.3f}V  "
                 f"R2={r2_score(y_true, y_pred_ens):.3f}", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"parity_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Parity plot -> results/parity_{tag}.png")


def plot_importance(fitted_models, feature_names, tag: str):
    """RF 特征重要性 Top 15（标签直接取训练列名）。"""
    imp = fitted_models["RF"].named_steps["model"].feature_importances_
    top_n = min(15, len(imp))
    order = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(top_n), imp[order][::-1], color="#b2182b")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]], fontsize=9)
    ax.set_xlabel("RandomForest Feature Importance", fontsize=11)
    ax.set_title(f"Voltage Prediction ({tag}) - Top 15 Features", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"importance_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Feature importance -> results/importance_{tag}.png")


def save_results(metrics, preds, ens_pred, fitted_models, feature_names,
                 element_list, y, tag: str):
    """保存指标、模型束与 OOF 预测。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metrics -> results/metrics_{tag}.json")

    import pickle
    bundle = {
        "models": fitted_models,
        "feature_names": list(feature_names),
        "element_list": element_list,
        "target": "average_voltage_V",
        "tag": tag,
    }
    with open(OUTPUT_DIR / f"models_{tag}.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"[OK] Models -> results/models_{tag}.pkl")

    pd.DataFrame({
        "y_true": y,
        "pred_RF": preds["RF"],
        "pred_HGB": preds["HGB"],
        "pred_ens": ens_pred,
        "spread": np.abs(preds["RF"] - preds["HGB"]) / 2.0,
    }).to_csv(OUTPUT_DIR / f"oof_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] OOF -> results/oof_{tag}.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练平均电压预测模型")
    ap.add_argument("--data", default=str(SYNTHETIC_DATA), help="训练数据 CSV")
    ap.add_argument("--tag", default="voltage",
                    help="输出文件名后缀，如 voltage / voltage_synthetic")
    ap.add_argument("--target", default="average_voltage_V")
    ap.add_argument("--group-col", default="framework_formula",
                    help="GroupKFold 分组列（数据里没有则跳过分组评估）")
    args = ap.parse_args(argv)

    print("=" * 60)
    print("  Battery Cathode Average Voltage Prediction Training")
    print("=" * 60)

    df = load_data(Path(args.data))
    if args.target not in df.columns:
        print(f"[ERROR] 数据缺少目标列 {args.target}，现有列: {list(df.columns)[:10]}")
        return 2
    print(f"\nDataset: {len(df)} samples | unique formulas: {df['formula'].nunique()}")
    t = df[args.target]
    print(f"Target: mean {t.mean():.3f} +/- {t.std():.3f} V | "
          f"MAD {float((t - t.median()).abs().median()):.3f} V（中位数基线 MAE）")

    print("\nExtracting compositional features ...")
    X, y, groups, element_list = build_features(df, args.target, args.group_col)
    feature_names = X.columns.tolist()
    print(f"Element universe: {len(element_list)} | features: {len(feature_names)}")

    print("\n5-fold cross-validation (random split) ...")
    metrics, preds, ens_pred, fitted = evaluate_voltage(X, y)

    group_metrics = evaluate_grouped(X, y, groups)
    if group_metrics:
        metrics["group_cv"] = group_metrics

    print("\nGenerating visualizations ...")
    plot_results(y, ens_pred, args.tag)
    plot_importance(fitted, feature_names, args.tag)

    save_results(metrics, preds, ens_pred, fitted, feature_names, element_list, y, args.tag)

    print("\n" + "=" * 60)
    print(f"  Done. tag={args.tag}  MAE={metrics['Ensemble']['MAE']:.4f} V  "
          f"R2={metrics['Ensemble']['R2']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
