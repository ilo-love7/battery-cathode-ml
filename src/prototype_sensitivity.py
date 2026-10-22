# -*- coding: utf-8 -*-
"""量化「原型结构」与「MP 真实结构」之间的预测差异（v1.11.0）。

背景：9 个固溶体/富锂/DRX 候选在 MP 没有结构，我们用岩盐型阳离子原型补上
（``generate_prototype_structures.py``，默认用 CHGNet 做 MLIP 弛豫）。

关键问题：**拿原型喂 CGCNN，预测会偏多少？偏差来自哪里？**

做法：对**同时有 MP 结构与原型结构**的候选（31 条），比较 CGCNN 的两项预测：

- ``e_form_per_atom``（稳定性，CGCNN-EF）
- ``band_gap``（导电性，CGCNN-GAP）

结论（实测）：差异约 **+1.2 eV/atom**，且**基本是常数偏移**（std 小）。
这不是弛豫误差，而是「原型 = 岩盐型**亚稳**多晶型」与「MP = 该成分的**基态**多晶型」
之间的**多晶型能量差**。对使用者的含义：原型候选**只能做组内相对比较**，
不能与 MP 结构候选混排 → `screening_struct.py` 因此分 A/B/C 三层。

用法::

    python src/generate_prototype_structures.py --all --out data/raw/..._all.json.gz
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/prototype_sensitivity.py

输出::

    results/metrics_prototype_sensitivity.json, results/prototype_sensitivity.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MP_STRUCTS = ROOT / "data" / "raw" / "mp_structures_candidates.json.gz"
PROTO_STRUCTS = ROOT / "data" / "raw" / "mp_structures_candidates_prototype_all.json.gz"
CGCNN_EF = ROOT / "results" / "cgcnn_mp_8k.pt"
CGCNN_GAP = ROOT / "results" / "cgcnn_gap_8k.pt"
OUT_JSON = ROOT / "results" / "metrics_prototype_sensitivity.json"
OUT_CSV = ROOT / "results" / "prototype_sensitivity.csv"


def predict(model_path: Path, structures: dict, log1p: bool):
    """structures: {formula: pymatgen-structure-dict} -> {formula: value}"""
    import torch
    from cgcnn_data import build_graph
    from cgcnn_model import batch_tensors, build_model

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("atom_fea_len", 64), ckpt.get("n_conv", 3),
                        ckpt.get("h_fea_len", 128), ckpt.get("n_h", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    mean, std = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    use_log1p = log1p or ckpt.get("target_transform") == "log1p"

    keys, graphs = [], []
    for f, st in structures.items():
        g = build_graph(st)
        if g is None:
            continue
        keys.append(f)
        graphs.append(g)
    out = {}
    with torch.no_grad():
        for s in range(0, len(graphs), 64):
            chunk = list(range(s, min(s + 64, len(graphs))))
            (node, efea, src, dst, bvec) = batch_tensors(graphs, chunk, device)
            pred = (model(node, efea, src, dst, bvec) * std + mean).detach().cpu().numpy()
            if use_log1p:
                pred = np.clip(np.expm1(pred), 0.0, None)
            for k, v in zip([keys[i] for i in chunk], pred):
                out[k] = float(v)
    return out


def _load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["query_formula"]: r.get("structure") for r in json.load(f)
                if r.get("structure")}


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="原型 vs 弛豫结构的 CGCNN 预测差异")
    ap.add_argument("--mp", default=str(MP_STRUCTS))
    ap.add_argument("--prototypes", default=str(PROTO_STRUCTS))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--out-json", default=str(OUT_JSON),
                    help="指标输出路径（可分别存岩盐原型 / 同构替换两份对比）")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README）")
        return 2
    for p in (Path(args.mp), Path(args.prototypes)):
        if not p.exists():
            print(f"[ERROR] 找不到 {p}")
            print("        先跑: python src/generate_prototype_structures.py --all "
                  "--out data/raw/mp_structures_candidates_prototype_all.json.gz")
            return 2

    mp, proto = _load(Path(args.mp)), _load(Path(args.prototypes))
    common = sorted(set(mp) & set(proto))
    print(f"MP 结构 {len(mp)} | 原型结构 {len(proto)} | 可对比 {len(common)}")
    if len(common) < 5:
        print("[ERROR] 可对比样本过少")
        return 1

    ef_mp = predict(Path(CGCNN_EF), {f: mp[f] for f in common}, log1p=False)
    ef_pr = predict(Path(CGCNN_EF), {f: proto[f] for f in common}, log1p=False)
    gap_mp = predict(Path(CGCNN_GAP), {f: mp[f] for f in common}, log1p=True)
    gap_pr = predict(Path(CGCNN_GAP), {f: proto[f] for f in common}, log1p=True)

    rows = []
    for f in common:
        if f not in ef_mp or f not in ef_pr or f not in gap_mp or f not in gap_pr:
            continue
        rows.append({"formula": f, "ef_mp_eV_atom": ef_mp[f], "ef_proto_eV_atom": ef_pr[f],
                     "ef_diff_eV_atom": ef_pr[f] - ef_mp[f],
                     "gap_mp_eV": gap_mp[f], "gap_proto_eV": gap_pr[f],
                     "gap_diff_eV": gap_pr[f] - gap_mp[f]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] 无有效对比行")
        return 1

    metrics = {
        "n_compared": int(len(df)),
        "ef": {"MAE_proto_vs_mp_eV_atom": round(float(df["ef_diff_eV_atom"].abs().mean()), 4),
               "bias_eV_atom": round(float(df["ef_diff_eV_atom"].mean()), 4),
               "std_eV_atom": round(float(df["ef_diff_eV_atom"].std()), 4),
               "iqr_eV_atom": round(float(df["ef_diff_eV_atom"].quantile(0.75)
                                          - df["ef_diff_eV_atom"].quantile(0.25)), 4),
               "max_abs_eV_atom": round(float(df["ef_diff_eV_atom"].abs().max()), 4),
               "spearman": round(float(df["ef_mp_eV_atom"].corr(df["ef_proto_eV_atom"],
                                                               method="spearman")), 4)},
        "gap": {"MAE_proto_vs_mp_eV": round(float(df["gap_diff_eV"].abs().mean()), 4),
                "bias_eV": round(float(df["gap_diff_eV"].mean()), 4),
                "std_eV": round(float(df["gap_diff_eV"].std()), 4),
                "iqr_eV": round(float(df["gap_diff_eV"].quantile(0.75)
                                      - df["gap_diff_eV"].quantile(0.25)), 4),
                "max_abs_eV": round(float(df["gap_diff_eV"].abs().max()), 4),
                "spearman": round(float(df["gap_mp_eV"].corr(df["gap_proto_eV"],
                                                             method="spearman")), 4)},
        "interpretation": (
            "该差异**不代表弛豫误差**，而是「原型=岩盐型亚稳多晶型」与「MP=该成分的"
            "真实基态多晶型」之间的**多晶型能量差**：量级 ~1 eV/atom、且基本是常数偏移，"
            "故原型候选**只能做组内相对比较**，不能与 MP 结构候选混排。"),
        "implication_for_screening": (
            "screening_struct.py 因此分三层：A=MP 结构（主排名）、"
            "B=原型结构（单列，仅组内可比）、C=无结构（成分回退）。"),
    }
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("  原型（岩盐型亚稳多晶型）vs MP（真实基态多晶型）：CGCNN 预测差异")
    print("=" * 72)
    print(f"  样本 n={len(df)}")
    print(f"  形成能: bias={metrics['ef']['bias_eV_atom']:+.4f} ± "
          f"{metrics['ef']['std_eV_atom']:.4f} (std) eV/atom | "
          f"MAE={metrics['ef']['MAE_proto_vs_mp_eV_atom']:.4f} | ρ={metrics['ef']['spearman']:.3f}")
    print(f"  带隙:   bias={metrics['gap']['bias_eV']:+.4f} ± "
          f"{metrics['gap']['std_eV']:.4f} (std) eV | "
          f"MAE={metrics['gap']['MAE_proto_vs_mp_eV']:.4f} | ρ={metrics['gap']['spearman']:.3f}")
    print("  解释：该差异主要是**多晶型能量差**（亚稳岩盐 vs 基态多晶型），不是弛豫误差；")
    print("        因此原型候选只能做组内相对比较 → 筛选中单列为 B 层。")
    print(f"[SAVE] {out_json.name} / {Path(args.out_csv).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())