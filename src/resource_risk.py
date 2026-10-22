# -*- coding: utf-8 -*-
"""资源与法规维度：供应风险 / 毒性（v1.20.0）。

### 为什么

前 19 个版本只回答"性能好不好"，没回答"能不能大规模用"。而正极的成本与合规风险几乎
完全由**元素**决定：Co 的供应链集中度、Ni 的产地政治风险、V 的毒性、S 的加工安全。

### 数据来源与诚实声明

本模块的数值是**按关键矿产清单的定性分级换算的序数分数**（0–100，越高越紧），
依据公开的"关键矿产"共识（IEA《Global Critical Minerals Outlook》、USGS 关键矿产清单、
欧盟 CRM Act 的候选清单）**人工整理**：

| 元素 | 供应风险 | 毒性等级 | 依据要点 |
|---|---|---|---|
| Co | 85 | 3 | 刚果金占产量 ~70%，供应链高度集中 |
| Ni | 65 | 2 | 印尼占比高、地缘与环保压力 |
| Li | 55 | 1 | 产量集中（澳/智利/中国）、需求激增 |
| V | 60 | 3 | 钒化合物毒性（V₂O₅ 吸入危害） |
| Mn | 35 | 1 | 资源丰富但冶炼能耗高 |
| Mo / Nb / Zr / Ti | 30 / 45 / 35 / 25 | 1 | 伴生矿为主，供应随主矿波动 |
| Fe / P / Si / S / O / Al / F / C | 5–15 | 1 | 储量丰富、供应稳定 |

⚠️ 这些是**序数打分**，不是权威统计量：用来做"相对提示"，不能当作法规结论。
真正的合规评估必须查具体法规文本与实时供应数据。

用法（一般由 screening_struct.py 自动调用）::

    python src/resource_risk.py --formulas "LiNi0.8Co0.1Mn0.1O2,LiFePO4"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from formula import parse_formula  # noqa: E402

SUPPLY_RISK = {
    "Co": 85, "Ni": 65, "V": 60, "Li": 55, "Nb": 45, "Mn": 35, "Zr": 35,
    "Mo": 30, "Ti": 25, "P": 15, "Si": 12, "S": 12, "Al": 10, "Fe": 8,
    "F": 8, "O": 5, "C": 5,
}
TOXICITY = {
    "Co": 3, "Ni": 2, "V": 3, "Mn": 1, "Mo": 1, "Nb": 1, "Zr": 1, "Ti": 1,
    "Li": 1, "P": 1, "Si": 1, "S": 1, "Al": 1, "Fe": 1, "F": 2, "O": 1, "C": 1,
}
CRITICAL_ELEMENTS = {"Co", "Ni", "V", "Li"}


def score_formula(formula: str) -> dict:
    """按原子分数加权得到供应风险/毒性；同时给出关键元素清单。"""
    counts = parse_formula(formula)
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError(f"{formula}: 空化学式")
    supply = sum(SUPPLY_RISK.get(el, 20) * n for el, n in counts.items()) / total
    tox = max((TOXICITY.get(el, 1) for el in counts), default=1)
    critical = sorted([el for el in counts if el in CRITICAL_ELEMENTS])
    flagged = [el for el in critical if SUPPLY_RISK.get(el, 0) >= 55]
    level = "高" if supply >= 55 else ("中" if supply >= 30 else "低")
    return {
        "formula": formula,
        "resource_risk_score": round(float(supply), 1),
        "resource_risk_level": level,
        "toxicity_class": int(tox),
        "critical_elements": critical,
        "high_risk_elements": flagged,
        "flags": ("|".join(f"{el}供应链" for el in flagged)
                  + ("|毒性≥3" if tox >= 3 else "")).strip("|"),
    }


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="资源与法规维度（供应风险/毒性序数打分）")
    ap.add_argument("--formulas", default="", help="逗号分隔化学式")
    ap.add_argument("--out", default=None, help="输出 JSON（可选）")
    args = ap.parse_args(argv)

    fml = [f.strip() for f in args.formulas.split(",") if f.strip()]
    if not fml:
        print("[ERROR] --formulas 为空")
        return 2
    rows = []
    for f in fml:
        try:
            r = score_formula(f)
        except ValueError as exc:
            print(f"  [SKIP] {f}: {exc}")
            continue
        rows.append(r)
        print(f"  {f:34s} 风险分 {r['resource_risk_score']:5.1f}（{r['resource_risk_level']}）"
              f" 毒性 {r['toxicity_class']}  关键元素 {','.join(r['critical_elements']) or '-'}"
              f"  {r['flags']}")
    if args.out and rows:
        p = Path(args.out).resolve()
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] {p}")
    print("\n注意：分值为按关键矿产清单人工整理的**序数**指标，仅作相对提示，不是法规结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())