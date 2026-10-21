# -*- coding: utf-8 -*-
"""训练电池正极材料平均电压预测模型。

与带隙 / 形成能模型共用同一套特征工程与评估协议：
1. 读取（或按需生成）电压数据；
2. 成分特征 = 33 个统计特征 + 元素分数特征（与 train_gap / train_ef 完全一致）；
3. RF（中位数插补）+ HGB 双模型，5 折交叉验证 OOF 评估；
4. 保存 metrics / 平差图 / 特征重要性 / OOF 预测 / 模型束。

⚠️ 数据说明：默认的 ``data/battery_voltage.csv`` 是由
``generate_voltage_data.py`` 按材料族电压先验 + 噪声合成的 **演示数据**，
其指标只能验证流程可跑通，不能作为科学结论。接入真实数据（Materials
Project 电池数据或文献电压表）后指标才有物理意义。

用法::

    python src/train_voltage.py
    python src/train_voltage.py --data data/battery_voltage.csv
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
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features import build_universe, featurize_dataframe  # noqa: E402
from models import set_plot_style  # noqa: E402

RNG = 42
OUTPUT_DIR = ROOT / "results"
DEFAULT_DATA = ROOT / "data" / "battery_voltage.csv"


def load_or_generate_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    """读取电压数据；不存在时用合成数据生成器补一份。"""
    if path.exists():
        print(f"读取电压数据: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")
    print("未找到电压数据，生成演示用合成数据 ...")
    from generate_voltage_data import generate_voltage_data
    df = generate_voltage_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def build_voltage_models() -> dict:
    """RF + HGB（与 models.build_models 保持同一组超参数）。"""
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                        n_jobs=-1, random_state=RNG)),
    ])
    hgb = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RNG)
    return {"RF": rf, "HGB": hgb}


def build_features(df: pd.DataFrame):
    """化学式 -> (X, y, element_list)。特征化失败的样本与 y 同步剔除。"""
    formulas = df["formula"].astype(str).tolist()
    element_list = build_universe(formulas)
    X, failed = featurize_dataframe(formulas, element_list=element_list, verbose=False)
    keep = [i for i in range(len(df)) if i not in set(failed)]
    if failed:
        print(f"特征化失败 {len(failed)} 条（已与标签同步丢弃）")
    X = X.iloc[keep].reset_index(drop=True)
    y = df["average_voltage_V"].to_numpy(dtype=float)[keep]
    return X, y, element_list


def evaluate_voltage(X, y):
    """5 折 OOF 评估，返回 (metrics, preds, ens_pred, fitted_models)。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    models = build_voltage_models()
    preds, metrics = {}, {}
    t0 = time.time()

    for name, model in models.items():
        oof = cross_val_predict(model, X, y, cv=kf, n_jobs=1)
        preds[name] = np.asarray(oof, dtype=float)
        metrics[name] = {
            "MAE": float(mean_absolute_error(y, preds[name])),
            "RMSE": float(np.sqrt(mean_squared_error(y, preds[name]))),
            "R2": float(r2_score(y, preds[name])),
        }
        print(f"  [{name}] MAE={metrics[name]['MAE']:.4f} V, "
              f"RMSE={metrics[name]['RMSE']:.4f} V, R2={metrics[name]['R2']:.4f}")

    ens_pred = 0.5 * preds["RF"] + 0.5 * preds["HGB"]
    metrics["Ensemble"] = {
        "MAE": float(mean_absolute_error(y, ens_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, ens_pred))),
        "R2": float(r2_score(y, ens_pred)),
    }
    print(f"\n[Ensemble] MAE={metrics['Ensemble']['MAE']:.4f} V, "
          f"RMSE={metrics['Ensemble']['RMSE']:.4f} V, "
          f"R2={metrics['Ensemble']['R2']:.4f}  ({time.time() - t0:.0f}s)")

    fitted = {name: m.fit(X, y) for name, m in build_voltage_models().items()}
    return metrics, preds, ens_pred, fitted


def plot_results(y_true, y_pred_ens, feature_names):
    """平差图 + RF 特征重要性 Top 15。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_plot_style()

    fig, ax = plt.subplots(figsize=(6, 5.5))
    lo = max(0.0, float(np.min(y_true)) - 0.5)
    hi = float(np.max(y_true)) + 0.5
    ax.scatter(y_true, y_pred_ens, s=30, alpha=0.5, edgecolors="none", color="#219150")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Average Voltage (V)", fontsize=11)
    ax.set_ylabel("Predicted Voltage (V)", fontsize=11)
    ax.set_title(f"Voltage Prediction (5-fold OOF)\n"
                 f"MAE={mean_absolute_error(y_true, y_pred_ens):.3f}V  "
                 f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred_ens)):.3f}V  "
                 f"R2={r2_score(y_true, y_pred_ens):.3f}", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parity_voltage.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Parity plot saved: {OUTPUT_DIR / 'parity_voltage.png'}")


def plot_importance(fitted_models, feature_names):
    """RF 特征重要性 Top 15（标签直接取训练列名，避免手写映射错位）。"""
    imp = fitted_models["RF"].named_steps["model"].feature_importances_
    top_n = min(15, len(imp))
    order = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(top_n), imp[order][::-1], color="#b2182b")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]], fontsize=9)
    ax.set_xlabel("RandomForest Feature Importance", fontsize=11)
    ax.set_title("Voltage Prediction - Top 15 Features", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "importance_voltage.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Feature importance saved: {OUTPUT_DIR / 'importance_voltage.png'}")


def save_results(metrics, preds, ens_pred, fitted_models, feature_names, element_list, y):
    """保存指标、模型束与 OOF 预测。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "metrics_voltage.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metrics saved: {OUTPUT_DIR / 'metrics_voltage.json'}")

    import pickle
    bundle = {
        "models": fitted_models,
        "feature_names": list(feature_names),   # 真实训练列顺序（不再误写成元素列表）
        "element_list": element_list,
    }
    with open(OUTPUT_DIR / "models_voltage.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"[OK] Models saved: {OUTPUT_DIR / 'models_voltage.pkl'}")

    pd.DataFrame({
        "y_true": y,
        "pred_RF": preds["RF"],
        "pred_HGB": preds["HGB"],
        "pred_ens": ens_pred,
        "spread": np.abs(preds["RF"] - preds["HGB"]) / 2.0,
    }).to_csv(OUTPUT_DIR / "oof_voltage.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] OOF predictions saved: {OUTPUT_DIR / 'oof_voltage.csv'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练平均电压预测模型")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="电压数据 CSV")
    args = ap.parse_args(argv)

    print("=" * 60)
    print("  Battery Cathode Average Voltage Prediction Training")
    print("=" * 60)

    df = load_or_generate_data(Path(args.data))
    print(f"\nDataset size: {len(df)} samples")
    print(f"Unique formulas: {df['formula'].nunique()}")
    print(f"Voltage stats: {df['average_voltage_V'].mean():.3f} "
          f"+/- {df['average_voltage_V'].std():.3f} V")

    print("\nExtracting compositional features ...")
    X, y, element_list = build_features(df)
    feature_names = X.columns.tolist()
    print(f"Element universe: {len(element_list)} elements")
    print(f"Feature dimension: {len(feature_names)}")

    print("\nStarting 5-fold cross-validation ...")
    metrics, preds, ens_pred, fitted = evaluate_voltage(X, y)

    print("\nGenerating visualizations ...")
    plot_results(y, ens_pred, feature_names)
    plot_importance(fitted, feature_names)

    print("\nSaving models and results ...")
    save_results(metrics, preds, ens_pred, fitted, feature_names, element_list, y)

    print("\n" + "=" * 60)
    print("  Voltage model training complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
