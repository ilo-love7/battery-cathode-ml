# -*- coding: utf-8 -*-
"""DRX / 富锂候选的**无序系综**：多个随机装饰 + MLIP 弛豫 + CGCNN 系综平均（v1.13.0）。

### 问题（承接报告附录 L.6）

无序岩盐（DRX）与富锂材料的**本质是无序**：阳离子在同一个亚晶格上随机分布。
v1.12.0 用"同构替换 + MLIP 选相"补结构时，每个目标只生成了**一个**随机装饰，
于是把"某个偶然的装饰"当成了结论 —— 预测值会带一层**装饰噪声**，且无法估计。

### 做法

对每个目标：固定其**最优模板族**（来自 `template_substitution.py` 的选相结果），
生成 N 个**不同种子**的装饰（`--n-decorations`，默认 8）：

1. 每个装饰都用 **CHGNet** 弛豫 → 得到能量 E_i；
2. 对全部 N 个装饰用 CGCNN 预测（每原子形成能 / 带隙）→ 得系综均值与标准差；
3. 代表性结构取**系综内能量最低**者（与"按能量选相"一致），
   同时把系综均值/标准差作为该候选的**不确定度来源之一**记录下来。

输出::

    data/raw/mp_structures_candidates_ensemble.json.gz   （代表结构 + 系综统计）
    results/metrics_drx_ensemble.json                    （每个目标的系综统计总表）

用法::

    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/drx_ensemble.py
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/drx_ensemble.py --n-decorations 4 --limit 3
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

POLYMORPH = ROOT / "data" / "raw" / "mp_structures_candidates_polymorph.json.gz"
MP_STRUCTS = ROOT / "data" / "raw" / "mp_structures_candidates.json.gz"
OUT_JSON = ROOT / "data" / "raw" / "mp_structures_candidates_ensemble.json.gz"
METRICS = ROOT / "results" / "metrics_drx_ensemble.json"
CGCNN_EF = ROOT / "results" / "cgcnn_mp_8k.pt"
CGCNN_GAP = ROOT / "results" / "cgcnn_gap_8k.pt"


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="DRX/富锂候选的无序系综（随机装饰 + MLIP + CGCNN 系综）")
    ap.add_argument("--polymorph", default=str(POLYMORPH),
                    help="v1.12.0 的选相结果（提供每个目标的最优模板）")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--metrics", default=str(METRICS))
    ap.add_argument("--n-decorations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--relax-steps", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个目标（冒烟）")
    ap.add_argument("--no-relax", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README）")
        return 2
    try:
        import pymatgen  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 pymatgen")
        return 2
    import template_substitution as ts
    from prototype_sensitivity import predict

    if not Path(args.polymorph).exists():
        print(f"[ERROR] 找不到 {args.polymorph}（先跑 src/template_substitution.py）")
        return 2
    if args.limit:
        # v1.4.2 的教训：子集运行绝不能覆盖全量产物
        for attr in ("out", "metrics"):
            val = getattr(args, attr)
            if "_smoke" not in val:
                stem, dot, ext = val.partition(".")
                setattr(args, attr, f"{stem}_smoke{dot}{ext}" if dot else f"{val}_smoke")
        print(f"[冒烟模式] 输出改名: {Path(args.out).name} / {Path(args.metrics).name}")

    with gzip.open(args.polymorph, "rt", encoding="utf-8") as f:
        poly_rows = json.load(f)
    if args.limit:
        poly_rows = poly_rows[:args.limit]

    # CGCNN 预测要在 GPU 上跑 N 次（每个装饰一次）-> 统一收集后批量预测
    rows, metrics = [], []
    for r in poly_rows:
        target = r["query_formula"]
        template = (r.get("polymorph") or {}).get("selected_template")
        if not template:
            print(f"  [SKIP] {target}: 选相结果里没有模板")
            continue
        print(f"\n=== {target}（模板 {template}，{args.n_decorations} 个随机装饰）===")
        members, energies = [], []
        for k in range(args.n_decorations):
            got = ts.substitution_candidates(template, _template_struct(template),
                                             target, seed=args.seed + k)
            if got is None:
                print(f"  [WARN] 装饰 {k} 失败（模板不兼容）")
                break
            sd, meta = got
            if args.no_relax:
                members.append(sd)
                energies.append(float("nan"))
                continue
            try:
                rsd, e = ts.relax(sd, fmax=args.fmax, steps=args.relax_steps)
                members.append(rsd)
                energies.append(e)
                print(f"  装饰 {k}: E={e:.4f} eV/atom")
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] 装饰 {k} 弛豫失败: {exc}")
        if not members:
            continue

        best_i = int(np.nanargmin(energies)) if not args.no_relax else 0
        ef_pred = predict(Path(CGCNN_EF), {str(i): m for i, m in enumerate(members)},
                          log1p=False)
        gap_pred = predict(Path(CGCNN_GAP), {str(i): m for i, m in enumerate(members)},
                           log1p=True)
        ef_vals = np.array([ef_pred[str(i)] for i in range(len(members))], dtype=float)
        gap_vals = np.array([gap_pred[str(i)] for i in range(len(members))], dtype=float)
        e_arr = np.array(energies, dtype=float)

        rows.append({
            "query_formula": target,
            "formula": r.get("formula", target),
            "material_id": f"poly-ens:{template}->{target}",
            "match_type": "polymorph_substitution_ensemble",
            "e_form_per_atom": None, "band_gap": None, "energy_above_hull": None,
            "nsites": len(members[best_i]["sites"]),
            "structure": members[best_i],
            "ensemble": {
                "template": template,
                "n_decorations": len(members),
                "selection": "lowest_CHGNet_energy",
                "energy_min_eV_atom": (None if np.isnan(e_arr).all() else round(float(np.nanmin(e_arr)), 4)),
                "energy_std_eV_atom": (None if np.isnan(e_arr).all() else round(float(np.nanstd(e_arr)), 4)),
                "cgcnn_ef_mean_eV_atom": round(float(ef_vals.mean()), 4),
                "cgcnn_ef_std_eV_atom": round(float(ef_vals.std()), 4),
                "cgcnn_gap_mean_eV": round(float(gap_vals.mean()), 4),
                "cgcnn_gap_std_eV": round(float(gap_vals.std()), 4),
                "best_member_ef_eV_atom": round(float(ef_vals[best_i]), 4),
                "note": ("无序岩盐/富锂的系综近似：N 个随机装饰的 CGCNN 预测均值±标准差，"
                         "代表结构取系综内能量最低者"),
            },
        })
        metrics.append({
            "target": target, "template": template, "n": len(members),
            "ef_mean": round(float(ef_vals.mean()), 4), "ef_std": round(float(ef_vals.std()), 4),
            "ef_best": round(float(ef_vals[best_i]), 4),
            "gap_mean": round(float(gap_vals.mean()), 4), "gap_std": round(float(gap_vals.std()), 4),
            "energy_std": (None if np.isnan(e_arr).all() else round(float(np.nanstd(e_arr)), 4)),
        })
        print(f"  → 系综: e_form={ef_vals.mean():.4f} ± {ef_vals.std():.4f} eV/atom | "
              f"gap={gap_vals.mean():.4f} ± {gap_vals.std():.4f} eV | "
              f"最低能量装饰 ef={ef_vals[best_i]:.4f}")

    if not rows:
        print("[ERROR] 没有生成任何系综")
        return 1
    op = Path(args.out).resolve()
    op.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(op, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    ef_stds = [m["ef_std"] for m in metrics]
    gap_stds = [m["gap_std"] for m in metrics]
    payload = {
        "n_targets": len(metrics),
        "n_decorations_per_target": args.n_decorations,
        "ef_std_mean_eV_atom": round(float(np.mean(ef_stds)), 4),
        "ef_std_max_eV_atom": round(float(np.max(ef_stds)), 4),
        "gap_std_mean_eV": round(float(np.mean(gap_stds)), 4),
        "gap_std_max_eV": round(float(np.max(gap_stds)), 4),
        "per_target": metrics,
        "interpretation": ("系综标准差就是「无序装饰」带来的预测不确定度："
                           "应在引用这些候选的稳定性/带隙时把它作为误差条，"
                           "而不是把单个装饰的结果当成确定值。"),
    }
    mp = Path(args.metrics).resolve()
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print("  无序系综统计（N 个随机装饰）")
    print("=" * 74)
    print(f"  目标数 {len(metrics)}；每目标 {args.n_decorations} 个装饰")
    print(f"  CGCNN 每原子形成能：系综标准差 均值 {payload['ef_std_mean_eV_atom']:.4f}、"
          f"最大 {payload['ef_std_max_eV_atom']:.4f} eV/atom")
    print(f"  CGCNN 带隙：       系综标准差 均值 {payload['gap_std_mean_eV']:.4f}、"
          f"最大 {payload['gap_std_max_eV']:.4f} eV")
    print(f"[SAVE] {op.name} / {mp.name}")
    return 0


def _template_struct(template_formula: str):
    """从仓库内 MP 结构文件里取模板结构（候选 + demo 两个小库）。"""
    for src in ("mp_structures_candidates.json.gz", "mp_structures_cathodes.json.gz"):
        p = ROOT / "data" / "raw" / src
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for r in json.load(f):
                if (r.get("query_formula") or r.get("formula")) == template_formula:
                    return r["structure"]
    raise SystemExit(f"[ERROR] 模板库里找不到 {template_formula}")


if __name__ == "__main__":
    sys.exit(main())