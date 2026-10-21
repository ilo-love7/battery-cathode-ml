# -*- coding: utf-8 -*-
"""从 Materials Project 官方 API 拉取真实电池/材料数据。

两类数据：

1. **插入电极（真实电压标签）** —— ``materials.insertion_electrodes``
   每条是一个（宿主框架 → 嵌锂相）的电化学对，字段含
   ``average_voltage`` / ``capacity_grav`` / ``formula_discharge`` 等，
   这是本项目「化学式 → 平均电压」任务**唯一真正的标签来源**。

2. **材料摘要（真实形成能/带隙标签）** —— ``materials.summary``
   用于给现有 matbench 模型做**外部验证**（同域外推到锂电体系时的真实误差）。

API Key 读取顺序：``--key`` > 环境变量 ``MP_API_KEY`` > ``~/.mp_api_key``。
（key 只落在用户目录，不进仓库；``.gitignore`` 已包含 ``.mp_api_key`` / ``.env``）

用法::

    python src/fetch_mp_battery.py --what electrodes
    python src/fetch_mp_battery.py --what summary --max-summary 20000
    python src/fetch_mp_battery.py --what all

输出（data/raw/，均不入库）::

    mp_insertion_electrodes_Li.csv
    mp_summary_Li.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ELECTRODE_FIELDS = [
    "battery_type",
    "battery_formula",
    "formula_charge",
    "formula_discharge",
    "framework_formula",
    "framework",
    "working_ion",
    "average_voltage",
    "capacity_grav",
    "energy_grav",
    "capacity_vol",
    "energy_vol",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
    "chemsys",
    "nelements",
    "elements",
    "material_ids",
    "id_charge",
    "id_discharge",
    "num_steps",
    "max_voltage_step",
]

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "formation_energy_per_atom",
    "band_gap",
    "energy_above_hull",
    "nsites",
    "volume",
    "density",
    "is_stable",
    "is_metal",
    "theoretical",
]


def resolve_key(explicit: str | None) -> str:
    """按优先级取 API Key；都没有时给出可执行的修复提示。"""
    key = explicit or os.getenv("MP_API_KEY")
    if not key:
        key_file = Path.home() / ".mp_api_key"
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(
            "缺少 Materials Project API Key。\n"
            "  1) 免费注册: https://materialsproject.org -> Dashboard -> API Keys\n"
            "  2) 任选一种存放方式:\n"
            "     - 环境变量: setx MP_API_KEY \"你的key\"     (Windows)\n"
            "     - 文件    : 写入 %USERPROFILE%\\.mp_api_key\n"
            "     - 命令行  : python src/fetch_mp_battery.py --key 你的key"
        )
    return key


def fetch_electrodes(key: str, working_ion: str) -> "list[dict]":
    """拉取指定工作离子的插入电极数据。"""
    from mp_api.client import MPRester

    print(f"[GET] insertion_electrodes working_ion={working_ion} ...")
    with MPRester(key) as mpr:
        docs = mpr.materials.insertion_electrodes.search(
            working_ion=working_ion, fields=ELECTRODE_FIELDS)

    rows = []
    for d in docs:
        row = {}
        for f in ELECTRODE_FIELDS:
            v = getattr(d, f, None)
            if isinstance(v, (list, tuple)):
                v = ";".join(str(x) for x in v)
            elif v is not None and not isinstance(v, (str, int, float, bool)):
                v = str(v)
            row[f] = v
        rows.append(row)
    print(f"[OK ] {len(rows)} 条电极记录")
    return rows


def fetch_summary(key: str, max_rows: int, elements: list[str]) -> "list[dict]":
    """拉取含指定元素的材料摘要（用于外部验证）。max_rows=0 表示不限。

    注意：mp-api 的 ``num_chunks``/``chunk_size`` 才是分页机制；循环调用同一个
    ``search`` 只会重复拿第一页，所以这里一次性按需要的页数取。
    """
    from mp_api.client import MPRester

    print(f"[GET] summary elements={elements} (max_rows={max_rows or 'all'}) ...")
    with MPRester(key) as mpr:
        if max_rows:
            chunk_size = 1000
            num_chunks = max(1, (max_rows + chunk_size - 1) // chunk_size)
            docs = mpr.materials.summary.search(
                elements=elements, fields=SUMMARY_FIELDS,
                num_chunks=num_chunks, chunk_size=chunk_size)
        else:
            docs = mpr.materials.summary.search(elements=elements, fields=SUMMARY_FIELDS)

    rows = []
    seen = set()
    for d in docs:
        mid = getattr(d, "material_id", None)
        if mid in seen:          # 防御性去重（分页边界偶发重复）
            continue
        seen.add(mid)
        rows.append({f: getattr(d, f, None) for f in SUMMARY_FIELDS})
    print(f"[OK ] {len(rows)} 条材料摘要")
    return rows


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="从 Materials Project 拉取真实电池数据")
    ap.add_argument("--what", choices=["electrodes", "summary", "all"], default="electrodes")
    ap.add_argument("--key", default=None, help="MP API Key（省略则读环境变量/文件）")
    ap.add_argument("--working-ion", default="Li")
    ap.add_argument("--max-summary", type=int, default=20000,
                    help="摘要最多取多少条（0=全部，约 3-4 万条含 Li 材料）")
    ap.add_argument("--out-dir", default=str(RAW))
    args = ap.parse_args(argv)

    key = resolve_key(args.key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    if args.what in ("electrodes", "all"):
        rows = fetch_electrodes(key, args.working_ion)
        if rows:
            path = out_dir / f"mp_insertion_electrodes_{args.working_ion}.csv"
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            df = pd.DataFrame(rows)
            print(f"[SAVE] {path.relative_to(ROOT)}  ({len(df)} 行)")
            if "average_voltage" in df:
                v = df["average_voltage"].dropna()
                print(f"       average_voltage: {v.min():.2f} ~ {v.max():.2f} V, "
                      f"中位数 {v.median():.2f} V")

    if args.what in ("summary", "all"):
        rows = fetch_summary(key, args.max_summary, [args.working_ion])
        if rows:
            path = out_dir / f"mp_summary_{args.working_ion}.csv"
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {path.relative_to(ROOT)}  ({len(rows)} 行)")

    print("\n完成。下一步: python src/prepare_mp_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
