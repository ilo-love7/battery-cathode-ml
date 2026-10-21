# -*- coding: utf-8 -*-
"""把 matbench_perovskites 的 Structure 导出为 CGCNN 训练目录。

输出::

    <out-dir>/id_prop.csv      每行: <id>,<target>
    <out-dir>/<id>.poscar      pymatgen Structure 字典手工序列化的 POSCAR

⚠️ 目标单位（重要）:
    matbench 原始 ``e_form`` 是**整个 5 原子单胞的广延量**（记为 cell）。
    CGCNN / 文献报的 MAE 通常是 **eV/atom**，两者相差 5 倍。默认
    ``--scale atom`` 会除以单胞位点数，输出 eV/atom；需要原始广延量时用
    ``--scale cell``。混用会导致误差评估差一个数量级。

仅依赖标准库（自己实现 POSCAR 写出），本地 CPU 秒级完成；
随后按 docs/CGCNN_colab_extension.md 在 Colab/GPU 上训练。

用法::

    python src/export_cgcnn.py --out-dir cgcnn_data                    # eV/atom
    python src/export_cgcnn.py --out-dir cgcnn_data --scale cell       # eV/cell
    python src/export_cgcnn.py --out-dir /tmp/test --limit 5           # 试跑
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def write_poscar(st: dict, path: Path):
    """把 pymatgen Structure 字典写成 POSCAR（元素连续分组）。"""
    lat = st["lattice"]["matrix"]
    lines = ["POSCAR exported from matbench_perovskites (CGCNN pipeline)", "1.0"]
    for v in lat:
        lines.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}")

    species, coords = [], []
    for site in st["sites"]:
        # CGCNN 只支持单占位 site；取占位最大的元素（本数据集几乎全为单占位）
        sp = max(site["species"], key=lambda s: s.get("occu", 1.0))
        species.append(sp["element"])
        coords.append(site["abc"])

    groups: dict = {}
    for el, c in zip(species, coords):
        groups.setdefault(el, []).append(c)

    lines.append("  " + "  ".join(groups.keys()))
    lines.append("  " + "  ".join(str(len(v)) for v in groups.values()))
    lines.append("Direct")
    for el, pts in groups.items():
        for c in pts:
            lines.append(f"  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="导出 CGCNN 训练目录")
    ap.add_argument("--dataset", default="data/raw/matbench_perovskites.json.gz")
    ap.add_argument("--out-dir", default="cgcnn_data")
    ap.add_argument("--scale", choices=["atom", "cell"], default="atom",
                    help="目标单位：atom=eV/atom（推荐，除以单胞位点数），cell=原始 eV/胞")
    ap.add_argument("--limit", type=int, default=0, help="只导出前 N 条（0=全部，用于试跑）")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    src = root / args.dataset
    if not src.exists():
        print(f"[ERROR] 找不到数据集: {src}")
        print("        请先运行: python src/download_data.py")
        return 2

    out = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with gzip.open(src, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cols = d["columns"]
    i_st, i_ef = cols.index("structure"), cols.index("e_form")

    n = 0
    with open(out / "id_prop.csv", "w", encoding="utf-8", newline="") as f:
        f.write("id,target\n")
        for k, rec in enumerate(d["data"]):
            st, ef = rec[i_st], rec[i_ef]
            if st is None or ef is None:
                continue
            nsites = len(st.get("sites", [])) or 1
            target = float(ef) / nsites if args.scale == "atom" else float(ef)
            sid = f"perov-{k:05d}"
            write_poscar(st, out / f"{sid}.poscar")
            f.write(f"{sid},{target:.10f}\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    unit = "eV/atom" if args.scale == "atom" else "eV/5原子胞"
    print(f"导出 {n} 个样本 -> {out} (id_prop.csv + *.poscar)")
    print(f"目标单位: {unit}")
    print("打包: Compress-Archive -Path cgcnn_data -DestinationPath cgcnn_data.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
