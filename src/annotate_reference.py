# -*- coding: utf-8 -*-
"""给参考性能表补**溯源**与**逐行不确定度**（v1.18.0）。

### 为什么需要

`data/reference_cathodes.csv` 里的数值是"文献典型值"，此前统一按 ±10% 噪声做自助法。
但不同条目的可信度并不相同：商用材料的规格范围窄、研究阶段材料的报道差异可达 ±25%。
**不确定度本身也要有来源**，否则权重区间会失真。

### 做法（规则透明、可复现）

按材料类别分配来源类型与不确定度：

| 类别 | 判定规则 | 来源类型 | 不确定度 |
|---|---|---|---|
| 商用成熟 | 命中商用清单（LFP/NMC111/523/622/811/NCA/LCO/LMO…） | 厂商数据手册典型值 | ±8% |
| 研究热点 | 富锂/DRX/高压尖晶石/氟磷酸盐… | 同行综述典型值 | ±15% |
| 早期探索 | 硅酸盐/硫化物/磷酸镍等 | 本文估算（文献区间中值） | ±25% |

输出即覆盖原文件（新增 `source_type` / `uncertainty_pct` / `source_note` 三列），
并打印分布，便于在报告里如实说明。

用法::

    python src/annotate_reference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference_cathodes.csv"

COMMERCIAL = {
    "LiFePO4", "LiCoO2", "LiMn2O4", "LiNi0.333Co0.333Mn0.334O2",
    "LiNi0.5Co0.2Mn0.3O2", "LiNi0.6Co0.2Mn0.2O2", "LiNi0.8Co0.1Mn0.1O2",
    "LiNi0.8Co0.15Al0.05O2", "LiNi0.4Co0.2Mn0.4O2", "LiNi0.4Co0.3Mn0.3O2",
    "LiMn0.6Fe0.4PO4", "LiMn0.8Fe0.2PO4", "LiNi0.9Co0.05Mn0.05O2",
}
RESEARCH = {
    "Li1.2Ni0.13Co0.13Mn0.54O2", "Li1.2Ni0.2Mn0.6O2", "LiNi0.5Mn1.5O4",
    "LiCoMnO4", "LiVPO4F", "LiFeSO4F", "Li2FePO4F", "Li3V2(PO4)3",
    "Li1.2Mn0.4Ti0.4O2", "Li1.2Mn0.4Zr0.4O2", "Li1.3Mn0.4Nb0.3O2",
    "Li1.2Mn0.4Nb0.4O2", "LiMnPO4", "LiV3O8",
}


def classify(formula: str) -> tuple:
    if formula in COMMERCIAL:
        return "厂商数据手册典型值", 8, "商用材料，规格区间较窄"
    if formula in RESEARCH:
        return "同行综述典型值", 15, "研究热点，不同报道差异中等"
    return "本文估算（文献区间中值）", 25, "早期探索阶段，报道差异大"


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="给参考性能表补溯源与逐行不确定度")
    ap.add_argument("--reference", default=str(REFERENCE))
    args = ap.parse_args(argv)

    p = Path(args.reference)
    if not p.exists():
        print(f"[ERROR] 找不到 {p}")
        return 2
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "uncertainty_pct" not in df.columns:
        info = df["formula"].astype(str).map(classify)
        df["source_type"] = [x[0] for x in info]
        df["uncertainty_pct"] = [x[1] for x in info]
        df["source_note"] = [x[2] for x in info]
    df.to_csv(p, index=False, encoding="utf-8-sig")

    print(f"[OK ] {p.name}：{len(df)} 条已标注溯源")
    print(df["source_type"].value_counts().to_string())
    print("不确定度分布:", df["uncertainty_pct"].value_counts().to_dict())
    print("\n注意：这些来源类型是**规则分配**（按商用/研究/探索三档），")
    print("      逐条 DOI 溯源仍是本项目可信度的天花板，见报告附录 R。")
    return 0


if __name__ == "__main__":
    sys.exit(main())