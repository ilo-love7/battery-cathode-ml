# -*- coding: utf-8 -*-
"""训练电池正极材料平均电压预测模型。

使用与带隙/形成能模型相同的特征工程框架，
但目标是预测平均电压 (V)。

流程:
1. 读取或生成电池电压数据
2. 提取成分特征
3. 训练 RandomForest 和 HistGradientBoosting 模型
4. 5 折交叉验证评估
5. 保存模型和指标
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

# 添加 src 到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import featurize_one
from models import set_plot_style

RNG = 42
OUTPUT_DIR = ROOT / "results"
DATA_PATH = ROOT / "data" / "battery_voltage.csv"

# Global variables for plotting and saving
metrics = None
preds = None
element_list = None
fitted_models = None
y = None


def load_or_generate_data():
    """Load voltage data, generate mock data if not exists."""
    if DATA_PATH.exists():
        print(f"Reading voltage data: {DATA_PATH}")
        return pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    else:
        print("Voltage data file not found, generating mock data...")
        from generate_voltage_data import generate_voltage_data
        df = generate_voltage_data()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')
        return df


def build_voltage_models():
    """构建电压预测模型。"""
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            n_jobs=-1, random_state=RNG))
    ])
    
    hgb = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RNG
    )
    
    return {"RF": rf, "HGB": hgb}


def evaluate_voltage(X, y):
    """对电压数据进行 5 折交叉验证评估。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    models = build_voltage_models()
    
    preds = {}
    metrics = {}
    t0 = time.time()
    
    for name, model in models.items():
        oof_pred = cross_val_predict(model, X, y, cv=kf, n_jobs=1)
        preds[name] = np.asarray(oof_pred, dtype=float)
        
        mae = float(mean_absolute_error(y, preds[name]))
        rmse = float(np.sqrt(mean_squared_error(y, preds[name])))
        r2 = float(r2_score(y, preds[name]))
        
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        
        print(f"  [{name}] MAE={mae:.4f} V, RMSE={rmse:.4f} V, R-sq={r2:.4f}")
    
    # Ensemble prediction
    ens_pred = 0.5 * preds["RF"] + 0.5 * preds["HGB"]
    ens_mae = float(mean_absolute_error(y, ens_pred))
    ens_rmse = float(np.sqrt(mean_squared_error(y, ens_pred)))
    ens_r2 = float(r2_score(y, ens_pred))
    metrics["Ensemble"] = {"MAE": ens_mae, "RMSE": ens_rmse, "R2": ens_r2}
    
    print(f"\n[Ensemble] MAE={ens_mae:.4f} V, RMSE={ens_rmse:.4f} V, R-sq={ens_r2:.4f}")
    print(f"Total time: {time.time() - t0:.0f} seconds")
    
    # Fit full models
    fitted_models = {name: model.fit(X, y) for name, model in build_voltage_models().items()}
    
    return metrics, preds, ens_pred, fitted_models


def plot_results(y_true, y_pred_ens, preds, feature_importance=None):
    """绘制评估结果图。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_plot_style()
    
    # Parity plot
    fig, ax = plt.subplots(figsize=(6, 5.5))
    lims = [max(0, y_true.min() - 0.5), y_true.max() + 0.5]
    ax.scatter(y_true, y_pred_ens, s=30, alpha=0.5, edgecolors='none', color='#219150')
    ax.plot(lims, lims, 'k--', lw=1.5, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Average Voltage (V)', fontsize=11)
    ax.set_ylabel('Predicted Voltage (V)', fontsize=11)
    
    mae_val = mean_absolute_error(y_true, y_pred_ens)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred_ens))
    r2_val = r2_score(y_true, y_pred_ens)
    title = f"Voltage Prediction (5-fold OOF)\nMAE={mae_val:.3f}V  RMSE={rmse_val:.3f}V  R-sq={r2_val:.3f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parity_voltage.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Parity plot saved: {OUTPUT_DIR / 'parity_voltage.png'}")
    
    # Feature importance figure
    if feature_importance is None:
        rf_model = fitted_models["RF"]
        feature_importance = rf_model.named_steps["model"].feature_importances_
    
    # Get element names for labeling (first 12 are elemental fractions)
    n_frac_features = len(element_list) if element_list else 0
    
    top_n = min(15, len(feature_importance))
    indices = np.argsort(feature_importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#b2182b' if i < 6 else '#8dd3c7' for i in range(top_n)]
    ax.barh(range(top_n), feature_importance[indices][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    
    # Create feature labels
    labels = []
    for i in indices[::-1]:
        if i < n_frac_features and element_list:
            label = f"frac_{element_list[i]}"
        elif i < 45:  # statistical features
            stat_idx = i - n_frac_features
            props = ["Z", "mass", "EN", "radius", "group", "period"]
            stats = ["mean", "std", "max", "min", "range"]
            prop_idx = stat_idx // 5
            stat_idx2 = stat_idx % 5
            if prop_idx < len(props) and stat_idx2 < len(stats):
                label = f"{props[prop_idx]}_{stats[stat_idx2]}"
            else:
                label = f"feat_{i}"
        else:
            label = f"feat_{i}"
        
        if len(label) > 20:
            label = label[:17] + "..."
        labels.append(label)
    
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('RandomForest Feature Importance', fontsize=11)
    ax.set_title('Voltage Prediction - Top 15 Features', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "importance_voltage.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Feature importance saved: {OUTPUT_DIR / 'importance_voltage.png'}")


def save_results(metrics, ens_pred, fitted_models):
    """保存结果文件。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save metrics JSON
    with open(OUTPUT_DIR / "metrics_voltage.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metrics saved: {OUTPUT_DIR / 'metrics_voltage.json'}")
    
    # Save model bundle
    import pickle
    bundle = {
        "models": fitted_models,
        "feature_names": element_list,
        "element_list": element_list
    }
    with open(OUTPUT_DIR / "models_voltage.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"[OK] Models saved: {OUTPUT_DIR / 'models_voltage.pkl'}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        "y_true": y,
        "pred_RF": preds["RF"],
        "pred_HGB": preds["HGB"],
        "pred_ens": ens_pred,
        "spread": np.abs(preds["RF"] - preds["HGB"]) / 2.0
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_voltage.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] OOF predictions saved: {OUTPUT_DIR / 'oof_voltage.csv'}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Battery Cathode Average Voltage Prediction Training")
    print("=" * 60)
    
    # 1. Load data
    df = load_or_generate_data()
    print(f"\nDataset size: {len(df)} samples")
    print(f"Unique formulas: {df['formula'].nunique()}")
    print(f"Voltage stats: {df['average_voltage_V'].mean():.3f} +/- {df['average_voltage_V'].std():.3f} V")
    
    # 2. Feature extraction
    print("\nExtracting compositional features...")
    
    # Get all elements
    from formula import parse_formula
    element_set = set()
    for formula in df['formula']:
        try:
            element_set.update(parse_formula(formula).keys())
        except:
            pass
    
    element_list = sorted(element_set)
    print(f"Element universe: {len(element_list)} elements")
    
    # Build feature matrix
    X_rows = []
    for formula in df['formula']:
        feats = featurize_one(formula, element_list=element_list)
        if feats is not None:
            X_rows.append(feats)
    
    stats_features = []
    for _, row in df.iterrows():
        feats = featurize_one(row['formula'], element_list=None)
        if feats is not None:
            stats_features.append(feats)
    
    # Combine: first the fraction features, then statistical features
    frac_cols = [f"frac_{el}" for el in element_list]
    stat_cols = [k for k in stats_features[0].keys() if k.startswith(('Z_', 'mass_', 'EN_', 'radius_', 'group_', 'period_', 'n_elements', 'atoms_per_fu', 'molar_mass'))]
    
    X_frac = pd.DataFrame(X_rows, columns=frac_cols)
    X_stat = pd.DataFrame(stats_features, columns=stat_cols)
    X = pd.concat([X_frac, X_stat], axis=1)
    
    y = df['average_voltage_V'].values
    
    print(f"Feature dimension: {X.shape[1]}")
    
    # 3. Evaluation
    print("\nStarting 5-fold cross-validation...")
    metrics, preds, ens_pred, fitted_models = evaluate_voltage(X, y)
    
    # 4. Plotting and saving
    print("\nGenerating visualizations...")
    rf_model = fitted_models["RF"]
    feature_importance = rf_model.named_steps["model"].feature_importances_
    plot_results(y, ens_pred, preds, feature_importance)
    
    print("\nSaving models and results...")
    save_results(metrics, ens_pred, fitted_models)
    
    print("\n" + "=" * 60)
    print("  Voltage model training complete!")
    print("=" * 60)
