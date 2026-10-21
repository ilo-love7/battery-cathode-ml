# -*- coding: utf-8 -*-
"""把 MP 材料摘要清洗成外部验证集（带隙 / 形成能）。

输入::

    data/raw/mp_summary_Li.csv      (fetch_mp_battery.py --what summary 产出)

输出::

    data/processed/mp_summary_Li.csv.gz

清洗规则
--------
1. 化学式必须可解析、元素都在内置表中；
2. 按 ``formula_canon`` 去重，保留 ``energy_above_hull`` 最小的记录
   （同一成分多个相时取最稳定相，避免外部验证集里自相矛盾）；
3. 保留字段：material_id / formula / formation_energy_per_atom / band_gap /
   energy_above_hull / nsites / is_stable / theoretical，另加 atoms_per_fu。

⚠️ 单位与参考态提醒（做外部验证时必须注意）:
- 本项目形成能模型的目标是 matbench_perovskites 的 ``e_form``（**5 原子单胞的广延量**，
  RPBE 泛函、O 参考态取水蒸气化学势），而 MP 给的是 ``formation_energy_per_atom``
  （GGA/GGA+U，元素参考态）——**口径不同，绝对值不可直接比较**，只能看趋势/相关性，
  或减去常数偏移后再比。
- 本项目带隙模型训练于**实验**带隙，MP 的 ``band_gap`` 是 DFT-PBE 值（金属记 0），
  存在系统性泛函偏差。

用法::

    python src/prepare_mp_summary.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features import ELEMENTS  # noqa: E402
from formula import canonical_formula, parse_formula  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "mp_summary_Li.csv"
OUT_PATH = ROOT / "data" / "processed" / "mp_summary_Li.csv.gz"

KEEP = ["material_id", "formula", "formation_energy_per_atom", "band_gap",
        "energy_above_hull", "nsites", "is_stable", "theoretical"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="清洗 MP 摘要为外部验证集")
    ap.add_argument("--raw", default=str(RAW_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    if not raw.exists():
        print(f"[ERROR] 找不到 {raw}")
        print("        请先运行: python src/fetch_mp_battery.py --what summary")
        return 2

    df = pd.read_csv(raw, encoding="utf-8-sig")
    print(f"原始记录: {len(df)}")
    if "formula_pretty" in df.columns and "formula" not in df.columns:
        df = df.rename(columns={"formula_pretty": "formula"})

    df = df.dropna(subset=["formula", "formation_energy_per_atom", "band_gap"]).copy()
    rows, bad = [], 0
    for _, r in df.iterrows():
        try:
            counts = parse_formula(str(r["formula"]))
        except ValueError:
            bad += 1
            continue
        if any(el not in ELEMENTS for el in counts):
            bad += 1
            continue
        row = {k: r.get(k) for k in KEEP}
        row["formula_canon"] = canonical_formula(counts)
        row["atoms_per_fu"] = float(sum(counts.values()))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print("[ERROR] 清洗后没有可用样本")
        return 1

    before = len(out)
    out = (out.sort_values("energy_above_hull", na_position="last")
              .drop_duplicates(subset=["formula_canon"], keep="first")
              .reset_index(drop=True))

    print(f"  化学式无法解析/含未知元素: -{bad}")
    print(f"  按成分去重（保留最稳定相）: {before} -> {len(out)}")
    if "is_stable" in out:
        print(f"  其中 is_stable=True: {int(out['is_stable'].fillna(False).sum())}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig", compression="gzip")
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"[SAVE] {shown}")
    print("\n下一步: python src/validate_mp_external.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())