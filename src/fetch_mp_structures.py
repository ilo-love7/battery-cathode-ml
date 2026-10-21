# -*- coding: utf-8 -*-
"""从 Materials Project 拉取**晶体结构**（供 CGCNN 训练/推理与电压侧结构消歧使用）。

三种模式：

1. ``--mode li``（默认）：拉含 Li 材料的摘要 + 结构 + 每原子形成能（CGCNN 训练集）
2. ``--mode formulas``：按化学式取每个式的**最稳定结构**（低能于凸包）——用于对任意
   化学式做 CGCNN 预测（``predict_cgcnn.py``）
3. ``--mode ids``：按 material_id 列表取结构——用于给插入电极数据补结构（电压侧消歧）

输出（data/raw/，默认不入库；demo 小样本会另存 processed）::

    mp_structures_Li.json.gz          [{material_id, formula, e_form_per_atom, ..., structure}]
    mp_structures_formulas.json.gz
    mp_structures_ids.json.gz

用法::

    python src/fetch_mp_structures.py --mode li --n 8000
    python src/fetch_mp_structures.py --mode formulas --formulas "LiCoO2,LiFePO4,LiMn2O4"
    python src/fetch_mp_structures.py --mode ids --ids mp-849273,mp-19017
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

RAW = ROOT / "data" / "raw"
SUMMARY_FIELDS = ["material_id", "formula_pretty", "formation_energy_per_atom", "band_gap",
                  "energy_above_hull", "nsites", "is_stable", "structure"]


def resolve_key(explicit: str | None) -> str:
    key = explicit or os.getenv("MP_API_KEY")
    if not key:
        f = Path.home() / ".mp_api_key"
        if f.exists():
            key = f.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("缺少 MP API Key（--key / MP_API_KEY / ~/.mp_api_key）")
    return key


def _rec(doc) -> dict:
    st = getattr(doc, "structure", None)
    return {
        "material_id": str(getattr(doc, "material_id", "")),
        "formula": getattr(doc, "formula_pretty", None),
        "e_form_per_atom": getattr(doc, "formation_energy_per_atom", None),
        "band_gap": getattr(doc, "band_gap", None),
        "energy_above_hull": getattr(doc, "energy_above_hull", None),
        "nsites": getattr(doc, "nsites", None),
        "is_stable": getattr(doc, "is_stable", None),
        "structure": st.as_dict() if st is not None else None,
    }


def save(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    mb = path.stat().st_size / 1e6
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"[SAVE] {shown}  ({len(rows)} 条, {mb:.1f} MB)")


def fetch_li(key: str, n: int) -> list:
    from mp_api.client import MPRester
    rows, chunk = [], 500
    n_chunks = max(1, (n + chunk - 1) // chunk)
    print(f"[GET] 含 Li 材料 {n} 条（{n_chunks} 块 × {chunk}）...")
    t0 = time.time()
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(elements=["Li"], fields=SUMMARY_FIELDS,
                                            num_chunks=n_chunks, chunk_size=chunk)
    seen = set()
    for d in docs:
        r = _rec(d)
        if r["structure"] is None or r["e_form_per_atom"] is None:
            continue
        if r["material_id"] in seen:
            continue
        seen.add(r["material_id"])
        rows.append(r)
        if len(rows) >= n:
            break
    print(f"[OK ] {len(rows)} 条（{time.time()-t0:.0f}s）")
    return rows


def fetch_formulas(key: str, formulas: list) -> list:
    from mp_api.client import MPRester
    rows = []
    with MPRester(key) as mpr:
        for f in formulas:
            print(f"[GET] {f} ...")
            try:
                docs = mpr.materials.summary.search(formula=f, fields=SUMMARY_FIELDS)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {f}: {exc}")
                continue
            cands = [d for d in docs if getattr(d, "structure", None) is not None
                     and getattr(d, "energy_above_hull", None) is not None]
            if not cands:
                print(f"  [SKIP] {f}: 无结构")
                continue
            best = min(cands, key=lambda d: d.energy_above_hull)
            r = _rec(best)
            r["query_formula"] = f
            rows.append(r)
            print(f"  -> {r['material_id']} {r['formula']} "
                  f"(hull={r['energy_above_hull']:.4f}, nsites={r['nsites']})")
    print(f"[OK ] {len(rows)}/{len(formulas)} 个化学式取到结构")
    return rows


def fetch_ids(key: str, ids: list) -> list:
    """按 material_id 批量取结构（每批 100）。"""
    from mp_api.client import MPRester
    rows = []
    with MPRester(key) as mpr:
        for s in range(0, len(ids), 100):
            batch = ids[s:s + 100]
            try:
                docs = mpr.materials.summary.search(material_ids=batch,
                                                    fields=["material_id", "nsites", "structure"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] batch {s//100}: {exc}")
                continue
            for d in docs:
                st = getattr(d, "structure", None)
                if st is None:
                    continue
                rows.append({"material_id": str(d.material_id), "nsites": d.nsites,
                             "structure": st.as_dict()})
            print(f"  已取 {len(rows)}/{len(ids)}")
    print(f"[OK ] {len(rows)} 条结构")
    return rows


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="从 Materials Project 拉取晶体结构")
    ap.add_argument("--mode", choices=["li", "formulas", "ids"], default="li")
    ap.add_argument("--n", type=int, default=8000, help="li 模式条数")
    ap.add_argument("--formulas", default="", help="逗号分隔化学式（formulas 模式）")
    ap.add_argument("--ids", default="", help="逗号分隔 material_id（ids 模式）")
    ap.add_argument("--ids-file", default=None,
                    help="material_id 列表文件（每行一个；ids 数量多时用这个）")
    ap.add_argument("--key", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    key = resolve_key(args.key)
    if args.mode == "li":
        rows = fetch_li(key, args.n)
        out = Path(args.out) if args.out else RAW / "mp_structures_Li.json.gz"
    elif args.mode == "formulas":
        fml = [f.strip() for f in args.formulas.split(",") if f.strip()]
        if not fml:
            print("[ERROR] --formulas 为空")
            return 2
        rows = fetch_formulas(key, fml)
        out = Path(args.out) if args.out else RAW / "mp_structures_formulas.json.gz"
    else:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        if args.ids_file:
            ids += [ln.strip() for ln in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
        ids = sorted(set(ids))
        if not ids:
            print("[ERROR] --ids / --ids-file 为空")
            return 2
        print(f"待取结构: {len(ids)} 个 material_id")
        rows = fetch_ids(key, ids)
        out = Path(args.out) if args.out else RAW / "mp_structures_ids.json.gz"

    if not rows:
        print("[ERROR] 未取到任何结构")
        return 1
    save(rows, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())