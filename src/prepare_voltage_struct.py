# -*- coding: utf-8 -*-
"""给电压数据集补**结构特征**：脱锂相 / 嵌锂相的几何描述符 + 体积变化。

动机（见报告附录 G）：电压数据集里同一化学式可能对应多个框架，
同成分电压 std 中位数 0.22 V，是当前最大误差来源（随机 MAE 0.419 vs 分组 0.481）。
成分特征原理上无法区分这些框架，因此补结构信息：

- ``lith_*``：嵌锂相（``id_discharge``）的 21 维结构描述符
- ``frame_*``：脱锂相（``id_charge``）的 21 维结构描述符
- ``delta_vol``：脱锂相体积 / 嵌锂相体积 − 1（脱锂收缩率）
- ``delta_natom``：两相原子数差（≈ 可脱嵌 Li 数）

输入::

    data/processed/mp_voltage.csv                  （prepare_mp_data.py 产出，含 id_charge/id_discharge）
    data/raw/mp_structures_electrodes.json.gz      （fetch_mp_structures.py --mode ids 产出）

输出::

    data/processed/mp_voltage_struct.csv.gz        （随仓库分发，供离线训练/CI）

用法::

    python src/prepare_mp_data.py                                     # 先确保 processed 里有 id 列
    python src/fetch_mp_structures.py --mode ids --ids-file data/raw/voltage_material_ids.txt \
        --out data/raw/mp_structures_electrodes.json.gz
    python src/prepare_voltage_struct.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from structure_features import DESCRIPTOR_NAMES, structure_descriptors  # noqa: E402

VOLTAGE_PATH = ROOT / "data" / "processed" / "mp_voltage.csv"
STRUCT_PATH = ROOT / "data" / "raw" / "mp_structures_electrodes.json.gz"
OUT_PATH = ROOT / "data" / "processed" / "mp_voltage_struct.csv.gz"

KEEP_VOLTAGE_COLS = ["formula", "formula_canon", "average_voltage_V", "capacity_grav",
                     "n_li", "molar_mass", "atoms_per_fu", "framework_formula",
                     "id_charge", "id_discharge", "max_delta_volume"]


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="给电压数据补结构特征")
    ap.add_argument("--voltage", default=str(VOLTAGE_PATH))
    ap.add_argument("--structures", default=str(STRUCT_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    vp, sp = Path(args.voltage), Path(args.structures)
    if not vp.exists():
        print(f"[ERROR] 找不到 {vp}（先跑 python src/prepare_mp_data.py）")
        return 2
    if not sp.exists():
        print(f"[ERROR] 找不到 {sp}")
        print("        请先运行:")
        print("          python src/fetch_mp_structures.py --mode ids "
              "--ids-file data/raw/voltage_material_ids.txt "
              "--out data/raw/mp_structures_electrodes.json.gz")
        return 2

    volt = pd.read_csv(vp, encoding="utf-8-sig")
    with gzip.open(sp, "rt", encoding="utf-8") as f:
        structs = {r["material_id"]: r["structure"] for r in json.load(f) if r.get("structure")}
    print(f"电压记录 {len(volt)} 条 | 结构库 {len(structs)} 个 material_id")

    def desc(mid):
        st = structs.get(str(mid)) if pd.notna(mid) else None
        return structure_descriptors(st) if st is not None else None

    rows, n_both, n_none = [], 0, 0
    for _, r in volt.iterrows():
        dl = desc(r.get("id_discharge"))     # 嵌锂相
        df_ = desc(r.get("id_charge"))       # 脱锂相
        if dl is None and df_ is None:
            n_none += 1
            continue
        if dl is not None and df_ is not None:
            n_both += 1
        out = {c: r.get(c) for c in KEEP_VOLTAGE_COLS if c in volt.columns}
        for name in DESCRIPTOR_NAMES:
            out[f"lith_{name}"] = (dl or {}).get(name, float("nan"))
            out[f"frame_{name}"] = (df_ or {}).get(name, float("nan"))
        vl = (dl or {}).get("lat_volume", float("nan"))
        vf = (df_ or {}).get("lat_volume", float("nan"))
        out["delta_vol"] = (vf / vl - 1.0) if (vl and vl > 0 and vf == vf) else float("nan")
        nl = (dl or {}).get("nsites", float("nan"))
        nf = (df_ or {}).get("nsites", float("nan"))
        out["delta_natom"] = nl - nf if (nl == nl and nf == nf) else float("nan")
        out["has_both_structures"] = bool(dl is not None and df_ is not None)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("[ERROR] 没有任何记录同时拿到结构")
        return 1

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(op, index=False, encoding="utf-8-sig", compression="gzip")

    struct_cols = [c for c in out_df.columns if c.startswith(("lith_", "frame_", "delta_"))]
    print(f"保留 {len(out_df)} 条（两相结构齐全 {n_both} 条，无结构丢弃 {n_none} 条）")
    print(f"结构特征 {len(struct_cols)} 维；delta_vol 中位数 "
          f"{out_df['delta_vol'].median():.3f}（负值=脱锂收缩）")
    try:
        shown = op.relative_to(ROOT)
    except ValueError:
        shown = op
    print(f"[SAVE] {shown}")
    print("\n下一步: python src/train_voltage_struct.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())