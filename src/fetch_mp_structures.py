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


def fetch_formulas(key: str, formulas: list, tolerant: bool = True,
                   tol: float = 0.02) -> list:
    """按化学式取结构。

    三级匹配（后一级在前一级无结果时启用）：

    1. 精确 ``formula=f``；
    2. pymatgen 约化式 ``reduced_formula``（如 LiNi0.333Co0.333Mn0.334O2 → Li3MnCoNiO6）；
    3. **元素集合 + 成分容差**匹配（固溶体/含小数配比的候选，MP 里往往只有
       近似有序超胞，例如 NMC811 对应 mp-1120802），取成分偏差最小、其次 hull 最低者。

    每条记录会带上 ``match_type``（exact/reduced/tolerant）与 ``comp_deviation``。
    """
    from mp_api.client import MPRester

    try:
        from pymatgen.core import Composition
    except ImportError:
        Composition = None

    rows = []
    with MPRester(key) as mpr:
        for f in formulas:
            print(f"[GET] {f} ...")
            best, match_type, dev = None, None, float("nan")

            try:
                docs = mpr.materials.summary.search(formula=f, fields=SUMMARY_FIELDS)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {f}: {exc}")
                continue
            cands = [d for d in docs if getattr(d, "structure", None) is not None
                     and getattr(d, "energy_above_hull", None) is not None]
            if cands:
                best, match_type = min(cands, key=lambda d: d.energy_above_hull), "exact"

            if best is None and tolerant and Composition is not None:
                comp = Composition(f)
                try:
                    docs = mpr.materials.summary.search(formula=comp.reduced_formula,
                                                        fields=SUMMARY_FIELDS)
                    cands = [d for d in docs if getattr(d, "structure", None) is not None
                             and getattr(d, "energy_above_hull", None) is not None]
                    if cands:
                        best, match_type = min(cands, key=lambda d: d.energy_above_hull), "reduced"
                except Exception:  # noqa: BLE001
                    pass

            if best is None and tolerant and Composition is not None:
                comp = Composition(f)
                els = sorted(str(e) for e in comp.elements)
                try:
                    docs = mpr.materials.summary.search(
                        elements=els, num_chunks=2, chunk_size=500, fields=SUMMARY_FIELDS)
                except Exception:  # noqa: BLE001
                    docs = []
                hits = []
                for d in docs:
                    if getattr(d, "structure", None) is None or d.energy_above_hull is None:
                        continue
                    try:
                        c2 = Composition(d.formula_pretty)
                    except Exception:  # noqa: BLE001
                        continue
                    if sorted(str(e) for e in c2.elements) != els:
                        continue
                    dmax = max(abs(c2.get_atomic_fraction(e) - comp.get_atomic_fraction(e))
                               for e in comp.elements)
                    if dmax <= tol:
                        hits.append((dmax, d.energy_above_hull, d))
                if hits:
                    hits.sort(key=lambda x: (round(x[0], 4), x[1]))
                    dev, _, best = hits[0]
                    match_type = "tolerant"

            if best is None:
                print(f"  [SKIP] {f}: MP 无匹配结构（含容差匹配）")
                continue
            r = _rec(best)
            r["query_formula"] = f
            r["match_type"] = match_type
            r["comp_deviation"] = None if dev != dev else round(float(dev), 4)
            rows.append(r)
            print(f"  -> {r['material_id']} {r['formula']} [{match_type}"
                  f"{'' if dev != dev else f', dev={dev:.4f}'}] "
                  f"hull={r['energy_above_hull']:.4f} nsites={r['nsites']}")
    n_exact = sum(1 for r in rows if r.get("match_type") == "exact")
    print(f"[OK ] {len(rows)}/{len(formulas)} 个化学式取到结构"
          f"（exact={n_exact}, 其余为约化式/容差匹配）")
    return rows


def fetch_pairs(key: str, formulas: list, exact: bool = True) -> list:
    """按化学式取**插入电极对**（两相结构 + 真实电压），供电压模型推理/验证。

    先用 ``insertion_electrodes.search(formula=...)`` 找候选，再按
    ``formula_discharge`` 精确匹配（``exact=True``）取第一条；随后用
    ``summary.search(material_ids=[...])`` 取两相结构。
    """
    from mp_api.client import MPRester

    elec_fields = ["battery_formula", "formula_charge", "formula_discharge",
                   "average_voltage", "capacity_grav", "id_charge", "id_discharge",
                   "max_delta_volume", "battery_type"]
    rows = []
    with MPRester(key) as mpr:
        for f in formulas:
            print(f"[GET] electrode pairs for {f} ...")
            try:
                docs = mpr.materials.insertion_electrodes.search(
                    working_ion="Li", formula=f, fields=elec_fields)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {f}: {exc}")
                continue
            if not docs:
                print(f"  [SKIP] {f}: 无电极记录")
                continue
            cands = [d for d in docs if d.id_charge and d.id_discharge]
            if exact:
                exact_c = [d for d in cands if str(d.formula_discharge) == f]
                if exact_c:
                    cands = exact_c
                else:
                    # 容差匹配：按元素分数近似（固溶体候选，如 NMC811/LMFP）
                    try:
                        from pymatgen.core import Composition as _C
                        want = _C(f)
                        near = [d for d in cands
                                if d.formula_discharge and abs(
                                    _C(str(d.formula_discharge)).num_atoms - want.num_atoms) >= 0]
                        if near:
                            cands = near
                    except Exception:  # noqa: BLE001
                        pass
            if not cands:
                print(f"  [SKIP] {f}: 无成对 id")
                continue
            best = min(cands, key=lambda d: abs(float(d.average_voltage or 0) - 3.7))
            ids = [str(best.id_charge), str(best.id_discharge)]
            try:
                st_docs = mpr.materials.summary.search(material_ids=ids,
                                                       fields=["material_id", "nsites", "structure"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {f} 结构: {exc}")
                continue
            smap = {str(d.material_id): d.structure for d in st_docs if d.structure is not None}
            st_c, st_d = smap.get(str(best.id_charge)), smap.get(str(best.id_discharge))
            if st_c is None or st_d is None:
                print(f"  [SKIP] {f}: 结构缺失（{len(smap)}/2）")
                continue
            rows.append({
                "query_formula": f,
                "battery_formula": best.battery_formula,
                "formula_charge": best.formula_charge,
                "formula_discharge": best.formula_discharge,
                "average_voltage_true_V": float(best.average_voltage),
                "capacity_grav": getattr(best, "capacity_grav", None),
                "max_delta_volume": getattr(best, "max_delta_volume", None),
                "id_charge": str(best.id_charge),
                "id_discharge": str(best.id_discharge),
                "n_charge": st_c.num_sites if hasattr(st_c, "num_sites") else len(st_c),
                "structure_charge": st_c.as_dict(),
                "structure_discharge": st_d.as_dict(),
            })
            print(f"  -> {best.formula_charge} -> {best.formula_discharge} "
                  f"(V_true={float(best.average_voltage):.3f}) ids={best.id_charge}/{best.id_discharge}")
    print(f"[OK ] {len(rows)}/{len(formulas)} 个化学式取到电极对")
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
    ap.add_argument("--mode", choices=["li", "formulas", "ids", "pairs"], default="li")
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
    elif args.mode == "pairs":
        fml = [f.strip() for f in args.formulas.split(",") if f.strip()]
        if not fml:
            print("[ERROR] --formulas 为空（pairs 模式）")
            return 2
        rows = fetch_pairs(key, fml)
        out = Path(args.out) if args.out else RAW / "mp_voltage_pairs.json.gz"
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