# -*- coding: utf-8 -*-
"""把 matbench_perovskites 的 Structure 导出为 CGCNN 训练目录。

输出:
  <out-dir>/id_prop.csv        每行: <id>,<target>  (target = e_form eV/atom)
  <out-dir>/<id>.poscar        pymatgen Structure 字典手工序列化的 POSCAR

仅依赖标准库（自己实现 POSCAR 写出），在本地 CPU 秒级完成。
随后打包上传 Colab 按 docs/CGCNN_colab_extension.md 训练。
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def write_poscar(st: dict, path: Path):
    lat = st["lattice"]["matrix"]
    lines = ["POSCAR exported from matbench_perovskites (CGCNN pipeline)",
             "1.0"]
    for v in lat:
        lines.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}")
    species, coords = [], []
    for site in st["sites"]:
        # CGCNN 只支持单占位 site；取占位最大的元素（本数据集几乎全为单占位）
        sp = max(site["species"], key=lambda s: s.get("occu", 1.0))
        species.append(sp["element"])
        coords.append(site["abc"])
    # 元素连续分组
    order = {}
    for el in species:
        order.setdefault(el, len(order))
    groups = {}
    for el, c in zip(species, coords):
        groups.setdefault(el, []).append(c)
    lines.append("  " + "  ".join(groups.keys()))
    lines.append("  " + "  ".join(str(len(v)) for v in groups.values()))
    lines.append("Direct")
    for el in groups:
        for c in groups[el]:
            lines.append(f"  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/raw/matbench_perovskites.json.gz")
    ap.add_argument("--out-dir", default="cgcnn_data")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / args.dataset
    out = root / args.out_dir
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
            sid = f"perov-{k:05d}"
            write_poscar(st, out / f"{sid}.poscar")
            f.write(f"{sid},{ef}\n")
            n += 1
    print(f"导出 {n} 个样本 -> {out} (id_prop.csv + *.poscar)")
    print("打包: Compress-Archive -Path cgcnn_data -DestinationPath cgcnn_data.zip")


if __name__ == "__main__":
    main()
