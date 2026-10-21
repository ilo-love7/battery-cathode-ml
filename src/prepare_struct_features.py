# -*- coding: utf-8 -*-
"""从 matbench_perovskites 原始结构生成结构描述符表（供 CPU 结构感知模型使用）。

输入::  data/raw/matbench_perovskites.json.gz     （4 MB，download_data.py 获取）
输出::  data/processed/perovskites_struct.csv.gz   （结构描述符 + 目标 + 成分，随仓库分发）

为什么单独落盘：原始 json.gz 体积较大且要联网，而结构感知训练只需要这 21 维描述符；
把它固化后，CI 与离线环境可以直接跑 ``train_ef_struct.py``（不需要下载、不需要 GPU）。

用法::

    python src/prepare_struct_features.py
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

from formula import canonical_formula, parse_formula  # noqa: E402
from structure_features import structure_descriptors  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "matbench_perovskites.json.gz"
OUT_PATH = ROOT / "data" / "processed" / "perovskites_struct.csv.gz"


def structure_to_counts(st: dict) -> dict:
    """结构 -> {元素: 原子数}（占位加权）。"""
    counts: dict = {}
    for site in st["sites"]:
        for sp in site["species"]:
            el = sp.get("element")
            occ = float(sp.get("occu", 1.0))
            if el is None:
                raise ValueError("site 缺少 element")
            counts[el] = counts.get(el, 0.0) + occ
    return counts


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="生成结构描述符表")
    ap.add_argument("--raw", default=str(RAW_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    if not raw.exists():
        print(f"[ERROR] 找不到 {raw}")
        print("        请先运行: python src/download_data.py")
        print("        （云主机/CI 可能被 CDN 403 拦截，见 download_data.py 的提示）")
        return 2

    with gzip.open(raw, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cols = d["columns"]
    i_st, i_ef = cols.index("structure"), cols.index("e_form")

    rows, dropped = [], 0
    for k, rec in enumerate(d["data"]):
        st, ef = rec[i_st], rec[i_ef]
        if st is None or ef is None:
            dropped += 1
            continue
        feats = structure_descriptors(st)
        if feats is None:
            dropped += 1
            continue
        try:
            counts = structure_to_counts(st)
            canon = canonical_formula(counts)
            parse_formula(canon)
        except ValueError:
            dropped += 1
            continue
        rows.append({
            "index": k,
            "formula_canon": canon,
            "e_form_eV_cell": float(ef),
            "n_sites": len(st.get("sites", [])),
            **feats,
        })

    df = pd.DataFrame(rows)
    # 描述符精度到 1e-5 已远超模型所需，压缩体积（2.6 MB -> ~1.4 MB）
    desc_cols = [c for c in df.columns if c not in ("index", "formula_canon",
                                                    "e_form_eV_cell", "n_sites")]
    df[desc_cols] = df[desc_cols].round(5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig", compression="gzip")

    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"原始样本: {len(d['data'])}  丢弃: {dropped}  保留: {len(df)}")
    print(f"结构描述符: {len(df.columns) - 4} 维")
    print(f"[SAVE] {shown}")
    print("\n下一步: python src/train_ef_struct.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())