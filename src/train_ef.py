# -*- coding: utf-8 -*-
"""任务 B：形成能预测模型（成分特征 → E_form）。

数据: matbench_perovskites (MP DFT 钙钛矿形成能, eV/5原子单胞, RPBE 泛函)
特征: 33 统计特征 + 训练集元素宇宙上的元素分数特征
协议: 5 折交叉验证 OOF 评估; 模型 RF / HGB / 集成
输出: results/metrics_ef.json, results/parity_ef.png,
      results/importance_ef.png, results/oof_ef.csv, results/models_ef.pkl

说明: 该模型仅用成分特征预测形成能（未用结构特征）。数据集 18,928 条
      样本仅对应 9,646 个唯一成分: 同一成分因阳离子占位方式不同可对应
      不同形成能（重复成分内部目标 std 中位数 0.42 eV/胞, 成分不可约
      MAE 上限 0.379 eV/胞）, 成分特征模型 MAE≈0.48 eV/胞已接近该上限;
      结构感知模型 (coGN) 可达 0.0269 eV/胞 (matbench 榜单), 这正是
      CGCNN 扩展路线的动机。详见报告 §5。
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
    data_path = ROOT / "data/processed/perovskites_ef.csv"
    if not data_path.exists():
        print("[ERROR] 缺少 data/processed/perovskites_ef.csv")
        print("        请先运行: python src/download_data.py && python src/prepare_data.py")
        return 2

    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"钙钛矿形成能样本数: {len(df)}")

    element_list = build_universe(df["formula_canon"].tolist())
    print(f"元素宇宙: {len(element_list)} 种 -> {element_list}")

    X, failed = featurize_dataframe(df["formula_canon"].tolist(),
                                    element_list=element_list, verbose=False)
    keep = np.array([i for i in range(len(df)) if i not in set(failed)])
    X = X.iloc[keep].reset_index(drop=True)
    y = df["e_form_eV_cell"].iloc[keep].reset_index(drop=True)
    if failed:
        print(f"特征化失败 {len(failed)} 条（丢弃）")
    feat_names = X.columns.tolist()
    print(f"建模样本: {len(y)}, 特征数: {len(feat_names)}")

    res = evaluate_dataset(X, y, "ef", str(OUT), feat_names)

    res["oof"].to_csv(OUT / "oof_ef.csv", index=False)
    save_pickle({"models": res["models"], "feature_names": feat_names,
                 "element_list": element_list}, OUT / "models_ef.pkl")
    print("形成能模型完成 -> results/models_ef.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
