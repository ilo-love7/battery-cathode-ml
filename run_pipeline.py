#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""电池正极材料机器学习预测 - 完整流程运行脚本。

运行顺序:
1. 数据准备 (prepare_data.py)
2. 模型训练 (train_gap.py, train_ef.py, train_voltage.py)
3. 候选筛选 (screening.py)
4. 预测示例 (predictor.py)

使用方式:
    python run_pipeline.py [--skip-data] [--skip-training] [--skip-screening]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]


def run_script(script_name, description):
    """运行一个 Python 脚本。"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  运行：{script_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / script_name)],
        cwd=str(ROOT),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"❌ {script_name} 执行失败!")
        return False
    
    print(f"\n✅ {description} 完成!")
    return True


def main():
    parser = argparse.ArgumentParser(description="运行电池正极材料 ML 预测完整流程")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据准备步骤")
    parser.add_argument("--skip-training", action="store_true", help="跳过模型训练步骤")
    parser.add_argument("--skip-screening", action="store_true", help="跳过候选筛选步骤")
    parser.add_argument("--fast", action="store_true", help="快速模式：仅训练带隙和电压模型")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  电池正极材料机器学习预测 - 完整流程")
    print("=" * 70)
    print(f"\n工作目录：{ROOT}")
    print(f"Python: {sys.executable}")
    
    success = True
    
    # 步骤 1: 数据准备
    if not args.skip_data:
        if not run_script("prepare_data.py", "步骤 1: 数据清洗与预处理"):
            print("\n数据准备失败，终止流程（后续步骤依赖处理后的数据）。")
            return 1
    else:
        print("\n⊘ 跳过数据准备步骤")
    
    # 步骤 2: 模型训练
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("  步骤 2: 模型训练")
        print("=" * 70)
        
        # 带隙模型
        if not run_script("train_gap.py", "子任务：带隙预测模型训练"):
            success = False
        
        # 形成能模型（除非是快速模式）
        if not args.fast:
            if not run_script("train_ef.py", "子任务：形成能预测模型训练"):
                success = False
        
        # 电压模型
        if not run_script("train_voltage.py", "子任务：平均电压预测模型训练"):
            success = False
    else:
        print("\n⊘ 跳过模型训练步骤")
    
    # 步骤 3: 候选筛选
    if not args.skip_screening:
        if not run_script("screening.py", "步骤 3: 候选正极材料筛选"):
            success = False
    else:
        print("\n⊘ 跳过候选筛选步骤")
    
    # 步骤 4: 预测演示
    print("\n" + "=" * 70)
    print("  步骤 4: 新材料预测演示")
    print("=" * 70)
    if not run_script("predictor.py", "新材料性质预测"):
        success = False
    
    # 总结
    print("\n" + "=" * 70)
    if success:
        print("  ✅ 所有步骤执行成功!")
    else:
        print("  ⚠️ 部分步骤执行失败，请查看错误信息")
    print("=" * 70)
    
    print("\n输出文件位于:")
    print(f"  - 结果图表：{ROOT / 'results'}")
    print(f"  - 处理数据：{ROOT / 'data' / 'processed'}")
    print(f"  - 项目文档：{ROOT / 'docs'}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
