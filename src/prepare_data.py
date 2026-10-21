# -*- coding: utf-8 -*-
"""数据准备：把两个 matbench 公开数据集清洗为 processed CSV。

输入（data/raw/）:
- matbench_expt_gap.json.gz     实验带隙 4,604 条 (composition, gap expt)
- matbench_perovskites.json.gz  MP DFT 钙钛矿形成能 18,928 条 (structure, e_form)
                                注: e_form 单位为 eV/5 原子单胞 (RPBE 泛函,
                                O 参考态取水蒸气中化学势, 见 Castelli 2012)

输出（data/processed/）:
- expt_gap.csv       [formula_raw, formula_canon, gap_eV]
- perovskites_ef.csv [formula_raw, formula_canon, e_form_eV, n_sites]

处理规则：
- 化学式统一经 formula.parse_formula 解析后输出 canonical 形式；
- 解析失败 / 含未知元素的样本丢弃并计数（形成能集从 Structure 字典提取成分）。
"""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formula import parse_formula, canonical_formula  # noqa: E402

SRC_GAP = ROOT / "data/raw/matbench_expt_gap.json.gz"
SRC_PEROV = ROOT / "data/raw/matbench_perovskites.json.gz"


def clean_expt_gap():
    src = SRC_GAP
    with gzip.open(src, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cols = d["columns"]          # ['composition', 'gap expt']
    i_comp, i_gap = cols.index("composition"), cols.index("gap expt")
    rows, dropped = [], 0
    for rec in d["data"]:
        raw, gap = str(rec[i_comp]), rec[i_gap]
        try:
            counts = parse_formula(raw)
            canon = canonical_formula(counts)
        except ValueError:
            dropped += 1
            continue
        if gap is None:
            dropped += 1
            continue
        rows.append({"formula_raw": raw, "formula_canon": canon, "gap_eV": float(gap)})
    import pandas as pd
    df = pd.DataFrame(rows).drop_duplicates(subset=["formula_canon", "gap_eV"])
    out = ROOT / "data/processed/expt_gap.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"expt_gap: 保留 {len(df)} 条, 丢弃 {dropped} 条 -> {out.name}")
    return df


def structure_to_counts(st: dict) -> dict:
    """从 pymatgen Structure 字典提取 {元素: 总原子数}（占位求和）。"""
    counts = {}
    for site in st["sites"]:
        for sp in site["species"]:
            el = sp.get("element")
            occ = float(sp.get("occu", 1.0))
            if el is None:
                raise ValueError("site 缺少 element 字段")
            counts[el] = counts.get(el, 0.0) + occ
    return counts


def clean_perovskites():
    src = SRC_PEROV
    with gzip.open(src, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cols = d["columns"]          # ['structure', 'e_form']
    i_st, i_ef = cols.index("structure"), cols.index("e_form")
    rows, dropped = [], 0
    for rec in d["data"]:
        st, ef = rec[i_st], rec[i_ef]
        try:
            counts = structure_to_counts(st)
            canon = canonical_formula(counts)
        except (ValueError, KeyError, TypeError):
            dropped += 1
            continue
        if ef is None:
            dropped += 1
            continue
        nsites = len(st.get("sites", []))
        rows.append({"formula_raw": canon, "formula_canon": canon,
                     "e_form_eV_cell": float(ef), "n_sites": nsites})
    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "data/processed/perovskites_ef.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"perovskites: 保留 {len(df)} 条, 丢弃 {dropped} 条 -> {out.name}")
    print(f"  e_form 范围: [{df.e_form_eV_cell.min():.3f}, {df.e_form_eV_cell.max():.3f}] eV/5原子单胞")
    return df


def main() -> int:
    missing = [p for p in (SRC_GAP, SRC_PEROV) if not p.exists()]
    if missing:
        print("[ERROR] 缺少原始数据：")
        for p in missing:
            print(f"        - {p.relative_to(ROOT)}")
        print("        请先运行: python src/download_data.py")
        print("        离线环境：手动下载同名文件放入 data/raw/ 后重试。")
        return 2

    print("== 清洗实验带隙数据 ==")
    clean_expt_gap()
    print("== 清洗钙钛矿形成能数据 ==")
    clean_perovskites()
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
