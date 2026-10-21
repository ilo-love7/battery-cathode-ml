#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""电池正极材料机器学习预测 —— 完整流程运行脚本。

运行顺序::

    1. 数据准备    prepare_data.py      （需要 data/raw/*.json.gz）
    2. 模型训练    train_gap / train_ef / train_voltage
    3. 候选筛选    screening.py
    4. 预测演示    predictor.py

用法::

    python run_pipeline.py                 # 全流程
    python run_pipeline.py --fast          # 跳过形成能模型（最快）
    python run_pipeline.py --skip-data     # 复用已有 data/processed
    python run_pipeline.py --check         # 只做前置条件检查

前置数据缺失时脚本会给出明确提示而不是抛 traceback。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

RAW_FILES = ["data/raw/matbench_expt_gap.json.gz",
             "data/raw/matbench_perovskites.json.gz"]
PROCESSED_FILES = ["data/processed/expt_gap.csv",
                   "data/processed/perovskites_ef.csv"]


def _missing(paths):
    return [p for p in paths if not (ROOT / p).exists()]


def check_prerequisites(skip_data: bool) -> bool:
    """检查数据前置条件；不满足时打印可执行的修复建议。"""
    if skip_data:
        missing = _missing(PROCESSED_FILES)
        if missing:
            print("[ERROR] --skip-data 但缺少处理后数据：")
            for m in missing:
                print(f"        - {m}")
            print("        请先运行: python src/download_data.py && python src/prepare_data.py")
            print("        或去掉 --skip-data 让脚本自动处理。")
            return False
        return True

    missing = _missing(RAW_FILES)
    if missing:
        print("[ERROR] 缺少原始数据：")
        for m in missing:
            print(f"        - {m}")
        print("        请先下载（无需 API Key）: python src/download_data.py")
        print("        离线环境：手动下载同名文件放入 data/raw/ 后重试。")
        return False
    return True


def run_script(script_name: str, description: str) -> bool:
    """运行 src/ 下的一个脚本，返回是否成功。"""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"  运行：src/{script_name}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / script_name)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"[FAIL] {script_name} 执行失败 (exit {result.returncode})")
        return False
    print(f"\n[OK] {description} 完成")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="运行电池正极材料 ML 预测完整流程")
    ap.add_argument("--skip-data", action="store_true", help="跳过数据准备（复用 data/processed）")
    ap.add_argument("--skip-training", action="store_true", help="跳过模型训练")
    ap.add_argument("--skip-screening", action="store_true", help="跳过候选筛选")
    ap.add_argument("--fast", action="store_true", help="快速模式：跳过形成能模型训练")
    ap.add_argument("--check", action="store_true", help="只检查前置条件后退出")
    args = ap.parse_args(argv)

    print("=" * 70)
    print("  电池正极材料机器学习预测 - 完整流程")
    print("=" * 70)
    print(f"\n工作目录：{ROOT}")
    print(f"Python  ：{sys.executable}")

    if not check_prerequisites(args.skip_data):
        return 2
    if args.check:
        print("\n[OK] 前置条件检查通过")
        return 0

    ok = True

    # 步骤 1：数据准备
    if args.skip_data:
        print("\n[SKIP] 数据准备")
    elif not run_script("prepare_data.py", "步骤 1: 数据清洗与预处理"):
        print("\n数据准备失败，后续训练依赖处理后数据，终止。")
        return 1

    # 步骤 2：模型训练
    if args.skip_training:
        print("\n[SKIP] 模型训练")
    else:
        print(f"\n{'=' * 70}\n  步骤 2: 模型训练\n{'=' * 70}")
        ok &= run_script("train_gap.py", "2.1 带隙预测模型")
        if not args.fast:
            ok &= run_script("train_ef.py", "2.2 形成能预测模型")
        else:
            print("\n[SKIP] 形成能模型（--fast）")
        ok &= run_script("train_voltage.py", "2.3 平均电压预测模型")

    # 步骤 3：候选筛选
    if args.skip_screening:
        print("\n[SKIP] 候选筛选")
    else:
        ok &= run_script("screening.py", "步骤 3: 候选正极材料筛选")

    # 步骤 4：预测演示
    ok &= run_script("predictor.py", "步骤 4: 新材料性质预测演示")

    print("\n" + "=" * 70)
    print("  [OK] 所有步骤执行成功" if ok else "  [WARN] 部分步骤失败，请查看上方日志")
    print("=" * 70)
    print("\n输出文件：")
    print(f"  - 指标与图表：{ROOT / 'results'}")
    print(f"  - 处理后数据：{ROOT / 'data' / 'processed'}")
    print(f"  - 项目报告  ：{ROOT / 'docs'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
