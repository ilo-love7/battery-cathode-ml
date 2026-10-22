# -*- coding: utf-8 -*-
"""多晶型枚举 + 同构替换：让「补出来的结构」接近真实基态相（v1.12.0）。

### 为什么需要它（承接 v1.11.0 的结论）

v1.11.0 用**岩盐型原型**补上了 9 条 MP 无结构的候选，但实测发现：
原型与 MP 真实结构的 CGCNN 预测差 **+1.25 eV/atom**，且与是否 MLIP 弛豫几乎无关、
与晶胞体积也无关 —— 根因是**多晶型不同**（岩盐是亚稳相，MP 收录的是基态相）。

本模块换成**同构替换（isostructural substitution）**这条标准路线：

1. 收集 MP 里**已弛豫的模板结构**（层状 R-3m / 尖晶石 / 橄榄石 / 磷酸盐…）；
2. 对目标化学式，挑出**阴离子骨架相容**的模板，把模板的阳离子位点按目标
   化学计量比替换（整数占位，晶格与氧骨架沿用模板的弛豫几何）；
3. 每个候选结构都用 **CHGNet** 弛豫，然后**按能量/原子选最低者** —— 即
   "用 MLIP 做多晶型筛选"，与 MP 收录结构的生成逻辑一致（枚举 + 总能排序）。

这样得到的结构在**物相上**就是该成分的合理基态候选，CGCNN 预测自然更可比。

用法::

    # 模板库来自仓库内已有的 MP 结构（候选 + demo 正极）
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/template_substitution.py
    # 对全部 40 条候选做（用于敏感性分析对照）
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/template_substitution.py --all \
        --out data/raw/mp_structures_candidates_polymorph_all.json.gz

输出（schema 与 fetch_mp_structures / 原型文件一致，可直接喂给 prototype_sensitivity.py）::

    data/raw/mp_structures_candidates_polymorph.json.gz
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
TEMPLATE_SOURCES = [
    ROOT / "data" / "raw" / "mp_structures_candidates.json.gz",     # 31 条候选的 MP 结构
    ROOT / "data" / "raw" / "mp_structures_cathodes.json.gz",      # 10 条常见正极 demo
    ROOT / "data" / "raw" / "mp_structures_Li.json.gz",            # 8000 条 MP 含 Li 结构（本地）
]
PROTO_ALL = ROOT / "data" / "raw" / "mp_structures_candidates_prototype_all.json.gz"
OUT = ROOT / "data" / "raw" / "mp_structures_candidates_polymorph.json.gz"

ANION_ELEMENTS = {"O", "F", "S", "Cl", "N"}
# 模板族归类（按 MP 结构的空间群/化学式人工归类太脆；用"阳离子:阴离子比 + 元素"近似）
FAMILY_HINTS = [
    ("层状R-3m", {"Li", "O"}, 0.5),        # LiMO2：阳离子 2 / 阴离子 2
    ("尖晶石", {"Li", "O"}, 0.75),          # LiM2O4：阳离子 3 / 阴离子 4
    ("橄榄石", {"Li", "O", "P"}, 0.75),      # LiMPO4：阳离子 3 / 阴离子 4
]


def _anion_set(comp) -> frozenset:
    return frozenset(el.symbol for el in comp.elements if el.symbol in ANION_ELEMENTS)


def _cation_counts(comp) -> dict:
    return {el.symbol: amt for el, amt in comp.items() if el.symbol not in ANION_ELEMENTS}


def _integer_counts(frac: dict, n_sites: int) -> dict:
    """把分数占位换算成 n_sites 个位点上的整数计数（最大余数法）。"""
    raw = {el: f * n_sites for el, f in frac.items()}
    counts = {el: int(np.floor(v)) for el, v in raw.items()}
    short = n_sites - sum(counts.values())
    order = sorted(raw, key=lambda el: -(raw[el] - np.floor(raw[el])))
    for el in order[:max(short, 0)]:
        counts[el] += 1
    return counts


def substitution_candidates(template_formula: str, template_st, target_formula: str,
                            seed: int = 42, max_templates_per_family: int = 1):
    """把一个模板结构改造成目标成分（同构替换）-> (structure_dict, meta) 或 None。

    只替换**阳离子**位点，阴离子骨架与晶格保持模板的弛豫几何。
    """
    from pymatgen.core import Composition, Structure

    tgt = Composition(target_formula)
    tpl = Composition(template_formula)
    if _anion_set(tgt) != _anion_set(tpl):
        return None                                   # 阴离子骨架不同，不能同构替换
    tgt_cat = _cation_counts(tgt)
    tpl_cat = _cation_counts(tpl)
    if not tgt_cat or not tpl_cat:
        return None
    # 阳离子/阴离子比必须接近，否则位点数不匹配
    r_tgt = sum(tgt_cat.values()) / max(sum(1 for el in tgt.elements if el.symbol in ANION_ELEMENTS), 1)
    r_tpl = sum(tpl_cat.values()) / max(sum(1 for el in tpl.elements if el.symbol in ANION_ELEMENTS), 1)
    if abs(r_tgt - r_tpl) > 0.05:
        return None

    st = Structure.from_dict(template_st)
    cat_idx = [i for i, site in enumerate(st) if site.species_string not in ANION_ELEMENTS]
    if not cat_idx:
        return None
    tot = sum(tgt_cat.values())
    frac = {el: amt / tot for el, amt in tgt_cat.items()}
    counts = _integer_counts(frac, len(cat_idx))

    labels = []
    for el, c in counts.items():
        labels += [el] * c
    if len(labels) != len(cat_idx):                   # 取整后不足/超出 -> 补齐或截断
        if len(labels) < len(cat_idx):
            fill = max(counts, key=counts.get)
            labels += [fill] * (len(cat_idx) - len(labels))
        labels = labels[:len(cat_idx)]
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)

    species = list(st.species)
    for k, i in enumerate(cat_idx):
        species[i] = labels[k]
    new_st = Structure(st.lattice, species, st.frac_coords, coords_are_cartesian=False)

    realized = {el: labels.count(el) / len(cat_idx) for el in counts}
    dev = max(abs(realized.get(el, 0.0) - frac[el]) for el in frac)
    return new_st.as_dict(), {"template": template_formula, "substitution_deviation": round(float(dev), 4),
                              "n_cation_sites": len(cat_idx)}


def relax(structure_dict: dict, fmax: float = 0.05, steps: int = 300):
    """CHGNet MLIP 弛豫 -> (structure_dict, energy_eV_per_atom)。"""
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    from pymatgen.core import Structure

    global _RELAXER
    try:
        _RELAXER
    except NameError:
        _RELAXER = StructOptimizer()
    st = Structure.from_dict(structure_dict)
    res = _RELAXER.relax(st, fmax=fmax, steps=steps, verbose=False)
    relaxed = res["final_structure"]
    e_per_atom = float(res["trajectory"].energies[-1]) / len(st)
    return relaxed.as_dict(), e_per_atom


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="多晶型枚举 + 同构替换（按 MLIP 能量选相）")
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--all", action="store_true",
                    help="对全部候选生成（默认只补 MP 无结构的）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--relax-steps", type=int, default=300)
    ap.add_argument("--no-relax", action="store_true", help="跳过 MLIP 弛豫（仅做替换）")
    ap.add_argument("--max-templates", type=int, default=4,
                    help="每个目标最多弛豫多少个候选模板（按成分相近度排序取前 N）")
    ap.add_argument("--max-sites", type=int, default=40,
                    help="模板单胞原子数上限（大单胞弛豫很慢）")
    ap.add_argument("--templates-from", default="",
                    help="逗号分隔的模板来源文件名（默认候选+demo，不含 8000 条全库）")
    args = ap.parse_args(argv)

    try:
        import pymatgen  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 pymatgen")
        return 2
    import pandas as pd
    from pymatgen.core import Composition

    cand = pd.read_csv(args.candidates, encoding="utf-8-sig")
    have_mp = set()
    mp_path = ROOT / "data" / "raw" / "mp_structures_candidates.json.gz"
    if mp_path.exists() and not args.all:
        with gzip.open(mp_path, "rt", encoding="utf-8") as f:
            have_mp = {r["query_formula"] for r in json.load(f)}
        print(f"已有 MP 结构: {len(have_mp)} 条（本轮只补其余）；--all 可全量")

    # ---- 模板库 ----
    if args.templates_from:
        want = {s.strip() for s in args.templates_from.split(",") if s.strip()}
        sources = [s for s in TEMPLATE_SOURCES if s.name in want]
    else:
        # 默认只用"候选 + demo"两个小库：足够覆盖层状/尖晶石/橄榄石/磷酸盐族，
        # 也更可控（8000 条全库会让弛豫次数爆炸，需要时用 --templates-from 显式指定）
        sources = [s for s in TEMPLATE_SOURCES if s.name != "mp_structures_Li.json.gz"]
    templates = []
    for src in sources:
        if not src.exists():
            continue
        with gzip.open(src, "rt", encoding="utf-8") as f:
            for r in json.load(f):
                st, fml = r.get("structure"), r.get("query_formula") or r.get("formula")
                if not st or not fml:
                    continue
                try:
                    comp = Composition(fml)
                except Exception:  # noqa: BLE001
                    continue
                templates.append((fml, comp, st, src.name))
    n_big = sum(1 for _, _, st, _ in templates if len(st["sites"]) > args.max_sites)
    templates = [x for x in templates if len(x[2]["sites"]) <= args.max_sites]
    print(f"模板库: {len(templates)} 个结构（来自 {len(sources)} 个文件；"
          f"按 {args.max_sites} 原子上限剔除 {n_big} 个大单胞）")

    rows, summary = [], []
    for _, r in cand.iterrows():
        target = str(r["formula"])
        if target in have_mp:
            continue
        tgt = Composition(target)
        scored_tpl = []
        for fml, comp, st, src in templates:
            if fml == target:
                continue
            # 兼容性 + 相近度打分：阴离子集合、阳离子:阴离子比、元素重叠
            tcat = _cation_counts(tgt)
            pcat = _cation_counts(comp)
            if not tcat or not pcat:
                continue
            n_anion_t = sum(v for el, v in tgt.items() if el.symbol in ANION_ELEMENTS)
            n_anion_p = sum(v for el, v in comp.items() if el.symbol in ANION_ELEMENTS)
            if n_anion_t <= 0 or n_anion_p <= 0:
                continue
            ratio_t = sum(tcat.values()) / n_anion_t
            ratio_p = sum(pcat.values()) / n_anion_p
            if abs(ratio_t - ratio_p) > 0.05:
                continue
            overlap = len(set(tcat) & set(pcat)) / max(len(set(tcat) | set(pcat)), 1)
            score = -overlap - 0.2 * abs(n_anion_t - n_anion_p) / max(n_anion_t, n_anion_p)
            scored_tpl.append((score, fml, st, src))
        scored_tpl.sort(key=lambda x: x[0])
        cands = []
        for _, fml, st, src in scored_tpl[:args.max_templates]:
            got = substitution_candidates(fml, st, target, seed=args.seed)
            if got is None:
                continue
            sd, meta = got
            meta["template_source"] = src
            cands.append((sd, meta))
        if not cands:
            print(f"  [SKIP] {target}: 没有阴离子骨架相容的模板（候选 {len(scored_tpl)} 个）")
            continue

        # 同构替换后按 MLIP 能量选相
        scored = []
        for sd, meta in cands:
            if args.no_relax:
                scored.append((float("nan"), sd, meta))
                continue
            try:
                rsd, e = relax(sd, fmax=args.fmax, steps=args.relax_steps)
                meta["energy_eV_per_atom"] = round(e, 4)
                scored.append((e, rsd, meta))
            except Exception as exc:  # noqa: BLE001
                print(f"    [WARN] {target} <- {meta['template']} 弛豫失败: {exc}")
        if not scored:
            continue
        scored.sort(key=lambda x: (x[0] != x[0], x[0]))     # NaN 排最后
        best_e, best_sd, best_meta = scored[0]
        rows.append({
            "query_formula": target,
            "formula": r.get("name", target),
            "material_id": f"poly:{best_meta['template']}->{target}",
            "match_type": "polymorph_substitution",
            "e_form_per_atom": None, "band_gap": None, "energy_above_hull": None,
            "nsites": len(best_sd["sites"]),
            "structure": best_sd,
            "polymorph": {
                "selected_template": best_meta["template"],
                "selected_energy_eV_per_atom": (None if best_e != best_e else round(best_e, 4)),
                "n_families_tried": len(scored),
                "family_energies": {m["template"]: m.get("energy_eV_per_atom")
                                    for _, _, m in scored},
                "note": "同构替换 + CHGNet 弛豫，按能量/原子选最低相",
            },
        })
        summary.append((target, best_meta["template"], best_e, len(scored)))
        print(f"  [OK ] {target:26s} <- {best_meta['template']:12s} "
              f"E={best_e if best_e == best_e else float('nan'):.4f} eV/atom "
              f"（共试 {len(scored)} 个模板）")

    op = Path(args.out).resolve()
    op.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(op, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    try:
        shown = op.relative_to(ROOT)
    except ValueError:
        shown = op
    print(f"\n[SAVE] {shown}（{len(rows)} 条；relax={'否' if args.no_relax else 'CHGNet'}）")
    if summary:
        print("选中的模板族分布:", {t: sum(1 for s in summary if s[1] == t)
                                    for t in sorted({s[1] for s in summary})})
    print("下一步: python src/prototype_sensitivity.py --prototypes "
          f"{op.name}   # 对比与 MP 真实结构的偏差是否下降")
    return 0


if __name__ == "__main__":
    sys.exit(main())