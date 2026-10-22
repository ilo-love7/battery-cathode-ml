# -*- coding: utf-8 -*-
"""为 MP 里没有合适结构的候选**生成原型结构**（v1.11.0）。

背景（见报告附录 I）：40 个候选里有 9 个是固溶体/富锂/无序岩盐（NCA、NMC523/622、富锂 A·B、DRX-A~D），
MP 没有对应结构，导致它们无法进入结构模型主排名。本轮用 pymatgen 生成**岩盐型无序阳离子原型**：

- 阴离子亚晶格：面心立方 O；
- 阳离子亚晶格：按目标化学计量比**随机占位**（`Composition` 部分占位，无需凑整），
  Li 与过渡金属共占同一亚晶格 —— 这正是富锂/DRX 的物理图像（阳离子无序）；
- 晶胞常数默认 4.10 Å（典型 TM 氧化物岩盐值）；
- **默认用 CHGNet（通用机器学习势）做结构弛豫**（``--relax chgnet``）：
  未弛豫原型的 CGCNN 预测与 MP 弛豫结构相差 ~1.1 eV/atom（见 prototype_sensitivity），
  远大于模型自身误差，必须先把几何弛豫到合理构型。

节点特征用**占位度加权 one-hot**（`cgcnn_data.build_graph` 已支持），所以部分占位可直接进 CGCNN，
无需人为凑整化学计量比。

同时支持 ``--all``：给**已有 MP 结构**的候选也生成原型，用于量化
「原型 vs 弛豫结构」对 CGCNN 预测的影响（原型结构的额外不确定度）。

用法::

    python src/generate_prototype_structures.py                # 只补缺失的 9 条
    python src/generate_prototype_structures.py --all          # 全部 40 条（做敏感性分析）
    python src/generate_prototype_structures.py --a 4.20       # 改晶胞常数
    python src/generate_prototype_structures.py --relax none    # 关闭弛豫（对照用）

输出::

    data/raw/mp_structures_candidates_prototype.json.gz
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
MP_STRUCTS = ROOT / "data" / "raw" / "mp_structures_candidates.json.gz"
OUT = ROOT / "data" / "raw" / "mp_structures_candidates_prototype.json.gz"

CATION_ABC = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
ANION_ABC = [(0.5, 0.5, 0.5), (0.0, 0.0, 0.5), (0.0, 0.5, 0.0), (0.5, 0.0, 0.0)]


def rocksalt_prototype(formula: str, a: float = 4.10, supercell: int = 2, seed: int = 42,
                       decorated: bool = True):
    """生成岩盐型原型结构。

    ``decorated=True``（默认）：**整数占位的有序超胞** —— 按目标阳离子比例用
    "最大余数法"取整后随机分配到阳离子位点（固定种子）。
    这样 ASE/CHGNet 才能弛豫（ASE 不支持部分占位），代价是化学计量比有微小偏差，
    偏差量会随结果返回（``composition_deviation``）。

    ``decorated=False``：**部分占位**（`Composition` 精确化学计量比，但无法弛豫）。
    """
    from pymatgen.core import Composition, Lattice, Structure

    comp = Composition(formula)
    anions, cations = {}, {}
    for el, amt in comp.items():
        (anions if el.symbol in ("O", "F", "S", "Cl", "N") else cations)[el.symbol] = amt
    if not cations:
        raise ValueError(f"{formula}: 没有阳离子")
    tot_cat = sum(cations.values())
    cat_frac = {el: amt / tot_cat for el, amt in cations.items()}

    n_anion_per_cell = max(1, int(round(sum(anions.values()) / tot_cat * 4))) if anions else 4
    an_frac = {}
    if anions:
        tot_an = sum(anions.values())
        an_frac = {el: amt / tot_an for el, amt in anions.items()}

    n_cells = supercell ** 3 if supercell > 1 else 1

    if not decorated:
        species = [Composition(cat_frac) for _ in CATION_ABC]
        coords = list(CATION_ABC)
        for i in range(n_anion_per_cell):
            species.append(Composition(an_frac) if an_frac else "O")
            coords.append(ANION_ABC[i % len(ANION_ABC)])
        st = Structure(Lattice.cubic(a), species, coords)
        if supercell > 1:
            st.make_supercell([supercell] * 3)
        return st.as_dict(), {
            "cell_a_A": a, "supercell": supercell, "decorated": False,
            "cation_occupancy": {k: round(v, 4) for k, v in cat_frac.items()},
            "n_sites": len(st), "composition_deviation": 0.0,
        }

    # 有序整数占位：阳离子位点总数 = 4 * n_cells
    n_cat_sites = 4 * n_cells
    raw = {el: frac * n_cat_sites for el, frac in cat_frac.items()}
    counts = {el: int(np.floor(v)) for el, v in raw.items()}
    remainder = n_cat_sites - sum(counts.values())
    for el, _ in sorted(raw.items(), key=lambda kv: -(kv[1] - np.floor(kv[1])))[:remainder]:
        counts[el] += 1
    labels = []
    for el, c in counts.items():
        labels += [el] * c
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)

    species, coords = [], []
    for i in range(n_cells):
        for ci, abc in enumerate(CATION_ABC):
            species.append(labels[i * 4 + ci])
            coords.append((abc[0] + 0, abc[1], abc[2]))      # 晶格已按超胞放大
        for ai in range(n_anion_per_cell):
            el = max(an_frac, key=an_frac.get) if an_frac else "O"
            species.append(el)
            coords.append(ANION_ABC[ai % len(ANION_ABC)])

    lattice = Lattice.cubic(a * supercell)
    # 坐标用分数坐标，超胞内平移
    frac = []
    s = supercell
    for i in range(n_cells):
        ix, iy, iz = i % s, (i // s) % s, i // (s * s)
        for abc in CATION_ABC:
            frac.append(((abc[0] + ix) / s, (abc[1] + iy) / s, (abc[2] + iz) / s))
        for ai in range(n_anion_per_cell):
            abc = ANION_ABC[ai % len(ANION_ABC)]
            frac.append(((abc[0] + ix) / s, (abc[1] + iy) / s, (abc[2] + iz) / s))
    st = Structure(lattice, species, frac)

    realized = {el: counts[el] / n_cat_sites for el in counts}
    dev = max(abs(realized[el] - cat_frac[el]) for el in cat_frac)
    return st.as_dict(), {
        "cell_a_A": a, "supercell": supercell, "decorated": True,
        "cation_occupancy": {k: round(v, 4) for k, v in cat_frac.items()},
        "realized_occupancy": {k: round(v, 4) for k, v in realized.items()},
        "composition_deviation": round(float(dev), 4),
        "n_sites": len(st),
    }


def relax_structure(structure, fmax: float = 0.05, steps: int = 300):
    """用 CHGNet（通用机器学习势）弛豫结构 -> (relaxed_structure, info)。"""
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer

    global _RELAXER
    try:
        _RELAXER
    except NameError:
        _RELAXER = StructOptimizer()          # 权重随包加载，无需联网
    res = _RELAXER.relax(structure, fmax=fmax, steps=steps, verbose=False)
    relaxed = res["final_structure"]
    return relaxed, {"relaxer": "CHGNet", "fmax": fmax, "steps": steps,
                     "energy_eV_per_atom": round(float(res["trajectory"].energies[-1])
                                                 / len(structure), 4)}


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="为无 MP 结构的候选生成原型结构")
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--mp-structures", default=str(MP_STRUCTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--all", action="store_true", help="给全部候选生成（敏感性分析用）")
    ap.add_argument("--a", type=float, default=4.10, help="岩盐晶胞常数 (Å)")
    ap.add_argument("--supercell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decor", choices=["ordered", "disordered"], default="ordered",
                    help="ordered=整数占位超胞（可弛豫，成分有微小偏差）；"
                         "disordered=部分占位（精确计量比，无法弛豫）")
    ap.add_argument("--relax", choices=["chgnet", "none"], default="chgnet",
                    help="结构弛豫器（默认 CHGNet；none=保留未弛豫原型）")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--relax-steps", type=int, default=300)
    args = ap.parse_args(argv)

    try:
        import pymatgen  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 pymatgen（pip install pymatgen）")
        return 2

    import pandas as pd

    cand = pd.read_csv(args.candidates, encoding="utf-8-sig")
    have_mp = set()
    mp_path = Path(args.mp_structures)
    if mp_path.exists() and not args.all:
        with gzip.open(mp_path, "rt", encoding="utf-8") as f:
            have_mp = {r["query_formula"] for r in json.load(f)}
        print(f"已有 MP 结构的候选: {len(have_mp)} 个（本轮只补其余）")

    rows = []
    for _, r in cand.iterrows():
        f = str(r["formula"])
        if f in have_mp:
            continue
        try:
            st, info = rocksalt_prototype(f, a=args.a, supercell=args.supercell,
                                          seed=args.seed, decorated=(args.decor == "ordered"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {f}: {exc}")
            continue
        proto_meta = {"note": "岩盐型无序阳离子原型"}
        if args.relax == "chgnet":
            try:
                from pymatgen.core import Structure as _S
                relaxed, rinfo = relax_structure(_S.from_dict(st), fmax=args.fmax,
                                                 steps=args.relax_steps)
                st = relaxed.as_dict()
                proto_meta.update(rinfo)
                proto_meta["note"] += "，已用 CHGNet 弛豫"
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] {f} 弛豫失败（保留未弛豫原型）: {exc}")
                proto_meta["note"] += "，弛豫失败（未弛豫）"
        else:
            proto_meta["note"] += "，未弛豫"
        rows.append({
            "query_formula": f,
            "formula": r.get("name", f),
            "material_id": f"proto:rocksalt:{f}",
            "match_type": "prototype",
            "e_form_per_atom": None, "band_gap": None, "energy_above_hull": None,
            "nsites": info["n_sites"],
            "structure": st,
            "prototype": proto_meta,
        })
        occ = ", ".join(f"{k}:{v:.2f}" for k, v in info["cation_occupancy"].items())
        dev = info.get("composition_deviation", 0.0)
        print(f"  [OK ] {f:26s} {info['n_sites']:4d} 位点 | 阳离子 {occ} | 成分偏差 {dev:.4f}")

    op = Path(args.out).resolve()
    op.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(op, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    try:
        shown = op.relative_to(ROOT)
    except ValueError:
        shown = op
    print(f"[SAVE] {shown}（{len(rows)} 条原型结构，relax={args.relax}）")
    print("\n额外不确定度请用 src/prototype_sensitivity.py 量化（原型 vs MP 弛豫结构）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())