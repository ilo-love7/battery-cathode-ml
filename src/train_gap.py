# -*- coding: utf-8 -*-
"""任务 A：实验带隙预测模型（成分特征 → E_gap）。

数据: matbench_expt_gap (实验测量带隙, eV)
特征: 33 统计特征 + 训练集元素宇宙上的元素分数特征
协议: 5 折交叉验证 OOF 评估; 模型 RF / HGB / 集成
输出: results/metrics_gap.json, results/parity_gap.png,
      results/importance_gap.png, results/oof_gap.csv, results/models_gap.pkl
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import build_universe, featurize_dataframe  # noqa: E402
from models import evaluate_dataset, save_pickle  # noqa: E402

OUT = ROOT / "results"


def main() -> int:
    data_path = ROOT / "data/processed/expt_gap.csv"
    if not data_path.exists():
        print("[ERROR] 缺少 data/processed/expt_gap.csv")
        print("        请先运行: python src/download_data.py && python src/prepare_data.py")
        return 2

    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"实验带隙样本数: {len(df)}")

    element_list = build_universe(df["formula_canon"].tolist())
    print(f"元素宇宙: {len(element_list)} 种")

    X, failed = featurize_dataframe(df["formula_canon"].tolist(),
                                    element_list=element_list, verbose=False)
    keep = np.array([i for i in range(len(df)) if i not in set(failed)])
    X = X.iloc[keep].reset_index(drop=True)
    y = df["gap_eV"].iloc[keep].reset_index(drop=True)
    if failed:
        print(f"特征化失败 {len(failed)} 条（丢弃）")
    feat_names = X.columns.tolist()
    print(f"建模样本: {len(y)}, 特征数: {len(feat_names)}")

    res = evaluate_dataset(X, y, "gap", str(OUT), feat_names)

    res["oof"].to_csv(OUT / "oof_gap.csv", index=False)
    save_pickle({"models": res["models"], "feature_names": feat_names,
                 "element_list": element_list}, OUT / "models_gap.pkl")
    print("带隙模型完成 -> results/models_gap.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
