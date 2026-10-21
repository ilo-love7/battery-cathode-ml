# -*- coding: utf-8 -*-
"""训练比容量预测模型（多目标任务）。

真实标签来自 MP 插入电极的 ``capacity_grav``（mAh/g），与电压模型共用同一套成分
特征与评估协议（RF + HGB 双模型、5 折 OOF、按框架分组诊断），只是目标不同，
因此直接复用 :mod:`train_voltage` 的实现，这里只补默认参数。

用途：对**任意化学式**给出容量估计（不需要事先知道可脱锂数 n_Li），
从而让候选筛选在缺少 ``nLi_extract`` 时也有容量维度。

用法::

    python src/train_capacity.py
    python src/train_capacity.py --tag capacity --data data/processed/mp_voltage.csv

输出（results/）::

    metrics_capacity.json, parity_capacity.png, importance_capacity.png,
    oof_capacity.csv, models_capacity.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train_voltage import main as _train  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed" / "mp_voltage.csv"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--data" not in args:
        args += ["--data", str(DEFAULT_DATA)]
    if "--target" not in args:
        args += ["--target", "capacity_grav"]
    if "--tag" not in args:
        args += ["--tag", "capacity"]
    return _train(args)


if __name__ == "__main__":
    sys.exit(main())