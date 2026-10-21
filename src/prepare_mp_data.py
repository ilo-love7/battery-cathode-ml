# -*- coding: utf-8 -*-
"""把 MP 插入电极原始数据清洗成可训练表格。

输入::

    data/raw/mp_insertion_electrodes_Li.csv     (fetch_mp_battery.py 产出)

清洗规则
--------
1. 电压取 0 ≤ V ≤ 6 V（MP 里存在 -7.75 V / 33 V 这类非物理值，来自极端
   计算/参考态错误，直接剔除）；
2. 用**嵌锂相** ``formula_discharge`` 的成分做特征（例如 Li2FePO4F）；
3. 化学式必须可解析且元素都在内置元素表中；
4. 额外算出 n_Li / 摩尔质量 / 每式原子数，并用 ``Q = n_Li·F/3.6/M``
   与 MP 自带的 ``capacity_grav`` 交叉校验（一致性差 >5% 的记录会被标记）。

输出::

    data/processed/mp_voltage.csv
      formula, formula_canon, average_voltage_V, capacity_grav, capacity_calc,
      capacity_rel_err, n_li, molar_mass, atoms_per_fu, framework_formula,
      battery_type, battery_formula

注意：同一个 ``formula_canon`` 可能对应多个电压（不同框架/不同脱锂区间），
这是本数据集的内禀噪声（同名化学式电压 std 中位数约 0.22 V），
训练脚本会同时报告随机划分与按框架分组的交叉验证结果。

用法::

    python src/prepare_mp_data.py
    python src/prepare_mp_data.py --max-voltage 5.0
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
from voltage import theoretical_capacity_mAh_g  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "mp_insertion_electrodes_Li.csv"
OUT_PATH = ROOT / "data" / "processed" / "mp_voltage.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="清洗 MP 插入电极数据为训练表")
    ap.add_argument("--raw", default=str(RAW_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--min-voltage", type=float, default=0.0)
    ap.add_argument("--max-voltage", type=float, default=6.0)
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    if not raw.exists():
        print(f"[ERROR] 找不到 {raw}")
        print("        请先运行: python src/fetch_mp_battery.py --what electrodes")
        return 2

    df = pd.read_csv(raw, encoding="utf-8-sig")
    print(f"原始记录: {len(df)}")

    n0 = len(df)
    df = df.dropna(subset=["average_voltage", "formula_discharge"]).copy()
    print(f"  去掉缺失电压/化学式: -{n0 - len(df)}")

    n1 = len(df)
    df = df[df["average_voltage"].between(args.min_voltage, args.max_voltage)]
    print(f"  去掉 {args.min_voltage}-{args.max_voltage} V 之外: -{n1 - len(df)}")

    rows, bad = [], 0
    for _, r in df.iterrows():
        formula = str(r["formula_discharge"])
        try:
            counts = parse_formula(formula)
        except ValueError:
            bad += 1
            continue
        if any(el not in ELEMENTS for el in counts):
            bad += 1
            continue

        n_li = float(counts.get("Li", 0.0))
        atoms = float(sum(counts.values()))
        molar = float(sum(counts[el] * ELEMENTS[el]["mass"] for el in counts))
        cap_calc = theoretical_capacity_mAh_g(counts, n_li) if n_li > 0 else float("nan")
        cap_mp = r.get("capacity_grav")
        rel_err = (abs(cap_calc - float(cap_mp)) / float(cap_mp)
                   if pd.notna(cap_mp) and float(cap_mp) > 0 and n_li > 0 else float("nan"))

        rows.append({
            "formula": formula,
            "formula_canon": canonical_formula(counts),
            "average_voltage_V": float(r["average_voltage"]),
            "capacity_grav": cap_mp,
            "capacity_calc": round(cap_calc, 3) if n_li > 0 else None,
            "capacity_rel_err": round(rel_err, 4) if pd.notna(rel_err) else None,
            "n_li": n_li,
            "molar_mass": round(molar, 3),
            "atoms_per_fu": atoms,
            "framework_formula": r.get("framework_formula"),
            "battery_type": r.get("battery_type"),
            "battery_formula": r.get("battery_formula"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[ERROR] 清洗后没有可用样本")
        return 1

    print(f"  化学式无法解析/含未知元素: -{bad}")
    print(f"保留: {len(out)} 行 / {out['formula_canon'].nunique()} 个唯一成分")

    # 容量交叉校验（MP capacity_grav 与 Q=n·F/3.6/M 的差异）
    valid = out["capacity_rel_err"].dropna()
    if len(valid):
        print(f"  理论容量 vs MP capacity_grav 相对误差: 中位数 {valid.median():.3f}, "
              f">5% 的记录 {int((valid > 0.05).sum())} 条")

    dup = out["formula_canon"].value_counts()
    dup = dup[dup > 1]
    if len(dup):
        sub = out[out["formula_canon"].isin(dup.index)]
        std = sub.groupby("formula_canon")["average_voltage_V"].std()
        print(f"  同名化学式多条记录: {len(dup)} 个，内部电压 std 中位数 {std.median():.3f} V")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:      # --out 指向仓库外（例如测试的临时目录）
        shown = out_path
    print(f"[SAVE] {shown}")

    v = out["average_voltage_V"]
    print(f"\n电压分布: {v.min():.2f} ~ {v.max():.2f} V, 均值 {v.mean():.2f} V, "
          f"中位数 {v.median():.2f} V, MAD {float((v - v.median()).abs().median()):.3f} V")
    print("\n下一步: python src/train_voltage.py --data data/processed/mp_voltage.csv --tag voltage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
