# -*- coding: utf-8 -*-
"""为没有 MP 电极对的候选**构造两相结构**，提升电压模型覆盖率（v1.22.0）。

### 动机

双塔 CGCNN 电压模型需要**两相结构对**（脱锂相 + 嵌锂相）。MP 的 insertion_electrodes
只覆盖 40 条候选中的 **20 条**，其余候选的电压只能退回成分模型。

### 做法

1. 取候选的**嵌锂相**结构（MP 结构 / 无序系综代表 / 同构替换相，优先级同 screening）；
2. 按候选的 `nLi_extract` 从嵌锂相中**移除相应数量的 Li**（去掉最近的 Li–Li 对以保持电荷平衡）；
3. 用 **CHGNet** 弛豫脱锂相，得到两相结构对；
4. 与 MP 真实电极对合并，输出统一格式文件；
5. **校验**：对 MP 已有电极对的 20 条候选，用同一流程构造并比较预测差异 ——
   若构造法本身偏差很大，就必须在文档里说清楚，而不是直接用。

输出::

    data/raw/mp_voltage_pairs_constructed.json.gz   （仅构造的）
    data/raw/mp_voltage_pairs_all.json.gz           （MP + 构造，供 screening 使用）
    results/metrics_voltage_pairs.json

用法::

    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/build_voltage_pairs.py
    ... --validate-only        # 只做校验（用已有 20 条 MP 对）
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

CANDIDATES = ROOT / "data" / "cathode_candidates.csv"
MP_PAIRS = ROOT / "data" / "raw" / "mp_voltage_pairs_candidates.json.gz"
STRUCT_SOURCES = [
    ROOT / "data" / "raw" / "mp_structures_candidates.json.gz",
    ROOT / "data" / "raw" / "mp_structures_candidates_ensemble.json.gz",
    ROOT / "data" / "raw" / "mp_structures_candidates_polymorph.json.gz",
]
OUT_CONSTRUCTED = ROOT / "data" / "raw" / "mp_voltage_pairs_constructed.json.gz"
OUT_ALL = ROOT / "data" / "raw" / "mp_voltage_pairs_all.json.gz"
OUT_JSON = ROOT / "results" / "metrics_voltage_pairs.json"


def _delithiate(structure_dict: dict, n_remove: int, seed: int = 42):
    """从结构里移除 n_remove 个 Li（优先移除彼此最近的 Li，减少局部电荷失衡）。"""
    from pymatgen.core import Structure

    st = Structure.from_dict(structure_dict)
    li_idx = [i for i, s in enumerate(st) if s.species_string == "Li"]
    if not li_idx or n_remove <= 0:
        return None, 0
    n_remove = min(n_remove, len(li_idx) - 1)          # 至少保留 1 个 Li
    if n_remove <= 0:
        return None, 0
    coords = st.cart_coords
    # 贪心：每次移除与其它 Li 距离之和最小的 Li（即最"拥挤"处先脱）
    remaining = set(li_idx)
    removed = []
    for _ in range(n_remove):
        best, best_d = None, float("inf")
        for i in remaining:
            d = sum(float(np.linalg.norm(coords[i] - coords[j])) for j in remaining if j != i)
            if d < best_d:
                best, best_d = i, d
        remaining.discard(best)
        removed.append(best)
    # pymatgen 的 Structure 不支持列表索引 -> 逐位点构造新结构
    keep = [i for i in range(len(st)) if i not in set(removed)]
    new_st = Structure(st.lattice, [st[i].species for i in keep],
                       [st[i].frac_coords for i in keep],
                       coords_are_cartesian=False)
    return new_st.as_dict(), n_remove


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="构造两相结构以提升电压覆盖率")
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--mp-pairs", default=str(MP_PAIRS))
    ap.add_argument("--out-constructed", default=str(OUT_CONSTRUCTED))
    ap.add_argument("--out-all", default=str(OUT_ALL))
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--relax-steps", type=int, default=300)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README）")
        return 2
    import pandas as pd
    import template_substitution as ts

    if args.limit:
        # 部分运行不得覆盖全量产物（v1.4.2 教训）
        for attr in ("out_constructed", "out_all"):
            val = getattr(args, attr)
            stem, dot, ext = val.partition(".")
            setattr(args, attr, f"{stem}_smoke{dot}{ext}" if dot else f"{val}_smoke")
        print(f"[冒烟模式] 输出改名: {Path(args.out_all).name}")

    cand = pd.read_csv(args.candidates, encoding="utf-8-sig")
    nli_map = dict(zip(cand["formula"].astype(str), cand["nLi_extract"].astype(float)))
    structs = {}
    for src in STRUCT_SOURCES:
        if not src.exists():
            continue
        with gzip.open(src, "rt", encoding="utf-8") as f:
            for r in json.load(f):
                f_ = r.get("query_formula")
                if f_ and r.get("structure") and f_ not in structs:
                    structs[f_] = r["structure"]
    mp_pairs = {}
    if Path(args.mp_pairs).exists():
        with gzip.open(args.mp_pairs, "rt", encoding="utf-8") as f:
            mp_pairs = {r["query_formula"]: r for r in json.load(f)}
    print(f"候选 {len(cand)} | 可用结构 {len(structs)} | MP 电极对 {len(mp_pairs)}")

    # 需要构造的目标：有结构但没有 MP 电极对（--validate-only 时改为对已有对的做对照）
    targets = [f for f in structs if f in nli_map and (f in mp_pairs if args.validate_only
                                                       else f not in mp_pairs)]
    if args.limit:
        targets = targets[:args.limit]
    print(f"待构造 {len(targets)} 条（{'校验模式' if args.validate_only else '补全模式'}）")

    rows, fails = [], 0
    for f in targets:
        st = structs[f]
        li_total = sum(1 for s in st["sites"]
                       if any(sp["element"] == "Li" for sp in s["species"]))
        if li_total <= 1:
            print(f"  [SKIP] {f}: Li 数不足（{li_total}）")
            fails += 1
            continue
        # 目标脱锂量：nLi_extract 相对嵌锂相 Li 总数的比例（至少 1 个、至少留 1 个 Li）
        n_remove = int(round(float(nli_map[f])))
        n_remove = max(1, min(n_remove, li_total - 1))
        charge, n_rm = _delithiate(st, n_remove)
        if charge is None:
            print(f"  [SKIP] {f}: 脱锂失败")
            fails += 1
            continue
        try:
            charge_relaxed, e = ts.relax(charge, fmax=args.fmax, steps=args.relax_steps)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] {f} 脱锂相弛豫失败: {exc}")
            fails += 1
            continue
        ref = mp_pairs.get(f) or {}
        rows.append({
            "query_formula": f,
            "formula_charge": "".join([]) or f"delithiated-{f}",
            "formula_discharge": ref.get("formula_discharge", f),
            "average_voltage_true_V": ref.get("average_voltage_true_V"),
            "id_charge": ref.get("id_charge", f"constructed-charge:{f}"),
            "id_discharge": ref.get("id_discharge", f"constructed-discharge:{f}"),
            "n_charge_atoms": len(charge_relaxed["sites"]),
            "n_removed_Li": int(n_rm),
            "relax_energy_eV_atom": round(float(e), 4),
            "source": "constructed" if not ref else "constructed(validation)",
            "structure_charge": charge_relaxed,
            "structure_discharge": st,
        })
        print(f"  [OK ] {f:26s} 移除 {n_rm} Li -> 脱锂相 {len(charge_relaxed['sites'])} 位点 "
              f"(E={e:.3f} eV/atom)")

    op = Path(args.out_constructed).resolve()
    with gzip.open(op, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"\\n[SAVE] {op.name}（构造 {len(rows)} 条，失败 {fails} 条）")

    merged = list(mp_pairs.values()) + [r for r in rows if r["source"] == "constructed"]
    op2 = Path(args.out_all).resolve()
    with gzip.open(op2, "wt", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"[SAVE] {op2.name}（MP {len(mp_pairs)} + 构造 "
          f"{sum(1 for r in rows if r['source'] == 'constructed')} = {len(merged)} 条）")

    # ---- 校验：构造法本身的可信度 ----
    validation = None
    val_rows = [r for r in rows if r["source"].startswith("constructed(validation)")]
    if val_rows:
        from predict_voltage_cgcnn import predict as predict_pairs
        model = ROOT / "results" / "cgcnn_voltage.pt"
        if model.exists():
            df_c = predict_pairs(model, val_rows)
            mp_sub = [mp_pairs[r["query_formula"]] for r in val_rows
                      if r["query_formula"] in mp_pairs]
            df_m = predict_pairs(model, mp_sub)
            m = df_c.merge(df_m, on="query_formula", suffixes=("_constructed", "_mp"))
            diff = (m["voltage_cgcnn_V_constructed"]
                    - m["voltage_cgcnn_V_mp"]).abs()
            validation = {
                "n": int(len(m)),
                "MAE_constructed_vs_mp_V": round(float(diff.mean()), 4),
                "max_abs_diff_V": round(float(diff.max()), 4),
                "note": ("用同一流程对**已有 MP 电极对**的候选重新构造两相，比较双塔模型的预测差异；"
                         "该差异即「构造法引入的额外不确定度」，应写进文档。"),
            }
            print(f"\\n[校验] 构造法 vs MP 真电极对：n={validation['n']} "
                  f"MAE={validation['MAE_constructed_vs_mp_V']} V "
                  f"max={validation['max_abs_diff_V']} V")

    payload = {
        "n_candidates_with_mp_pairs": len(mp_pairs),
        "n_constructed": sum(1 for r in rows if r["source"] == "constructed"),
        "n_total_pairs": len(merged),
        "coverage_after": round(len(merged) / max(len(cand), 1) * 100, 1),
        "n_failed": fails,
        "validation": validation,
        "note": ("用「移除 Li + CHGNet 弛豫」构造脱锂相；构造法在已有 MP 电极对的候选上做了对照，"
                 "残差见 validation —— 覆盖率的提升以该残差为代价。"),
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {OUT_JSON.name}（覆盖率 {payload['coverage_after']}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())