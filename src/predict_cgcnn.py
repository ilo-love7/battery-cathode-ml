# -*- coding: utf-8 -*-
"""CGCNN 推理 CLI：给定**晶体结构**，预测每原子形成能。

为什么需要它
------------
v1.5.0 的 CGCNN 只在评估集上可用——因为 CGCNN 的输入是结构而不是化学式。
本脚本补齐最后一环：

- ``--structures``：直接读结构文件（``fetch_mp_structures.py`` 的产物，或仓库内 demo）
- ``--formulas``：给化学式，自动去 Materials Project 取**最稳定结构**再预测（需 API Key）

模型默认加载 ``results/cgcnn_mp.pt``（在 MP 含 Li 材料上训练，目标 eV/atom）。

用法::

    # 仓库内 10 个常见正极结构（离线可用）
    python src/predict_cgcnn.py --structures data/raw/mp_structures_cathodes.json.gz

    # 任意化学式（联网取结构）
    python src/predict_cgcnn.py --formulas "LiCoO2,LiFePO4,LiMn2O4"

输出::

    results/cgcnn_predictions.csv
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

from cgcnn_data import build_graph  # noqa: E402

DEFAULT_MODEL = ROOT / "results" / "cgcnn_mp.pt"
DEFAULT_STRUCTURES = ROOT / "data" / "raw" / "mp_structures_cathodes.json.gz"
OUT_CSV = ROOT / "results" / "cgcnn_predictions.csv"


def load_rows(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def fetch_rows(formulas: list) -> list:
    """联网：按化学式取最稳定结构（复用 fetch_mp_structures 的逻辑）。"""
    try:
        import fetch_mp_structures as fms
    except ImportError:
        print("[ERROR] 无法导入 fetch_mp_structures")
        return []
    key = fms.resolve_key(None)
    rows = fms.fetch_formulas(key, formulas)
    for r in rows:
        r.setdefault("material_id", "")
    return rows


def predict(model_path: Path, rows: list) -> pd.DataFrame:
    import torch
    from cgcnn_model import batch_tensors

    if not model_path.exists():
        raise SystemExit(
            f"[ERROR] 找不到模型 {model_path}\n"
            "        请先训练: python src/train_cgcnn_mp.py --epochs 150")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from cgcnn_model import build_model
    model = build_model(ckpt.get("atom_fea_len", 64), ckpt.get("n_conv", 3),
                        ckpt.get("h_fea_len", 128), ckpt.get("n_h", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    mean = float(ckpt.get("target_mean", 0.0))
    std = float(ckpt.get("target_std", 1.0))
    target = ckpt.get("target", "e_form_per_atom")
    print(f"模型: {model_path.name} | 训练于 {ckpt.get('trained_on', '?')} "
          f"({ckpt.get('n_train', '?')} 训练样本) | 目标 {target} | device={device}")

    graphs, meta = [], []
    for r in rows:
        st = r.get("structure")
        if st is None:
            continue
        g = build_graph(st)
        if g is None:
            continue
        graphs.append(g)
        meta.append({"formula": r.get("formula") or r.get("query_formula") or "",
                     "material_id": r.get("material_id", ""),
                     "nsites": int(g["n_atoms"]),
                     "energy_above_hull": r.get("energy_above_hull"),
                     "true_e_form_per_atom": r.get("e_form_per_atom")})
    if not graphs:
        raise SystemExit("[ERROR] 没有可用结构")

    preds = []
    with torch.no_grad():
        for s in range(0, len(graphs), 128):
            chunk = list(range(s, min(s + 128, len(graphs))))
            (node, efea, src, dst, bvec) = batch_tensors(graphs, chunk, device)
            out = model(node, efea, src, dst, bvec) * std + mean
            preds.append(out.detach().cpu().numpy())
    pred = np.concatenate(preds)

    df = pd.DataFrame(meta)
    df["cgcnn_e_form_eV_atom"] = pred
    df["cgcnn_e_form_eV_cell"] = pred * df["nsites"]
    if df["true_e_form_per_atom"].notna().any():
        df["abs_err_eV_atom"] = (df["cgcnn_e_form_eV_atom"]
                                 - df["true_e_form_per_atom"]).abs()
    return df


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="CGCNN 推理（结构 -> 每原子形成能）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--structures", default=None,
                    help="结构文件（json.gz，含 structure 字段）")
    ap.add_argument("--formulas", default="", help="逗号分隔化学式（联网取结构）")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] CGCNN 推理需要 PyTorch。GPU 环境见 README「GPU 结构模型（CGCNN）」。")
        return 2

    if args.formulas:
        fml = [f.strip() for f in args.formulas.split(",") if f.strip()]
        print(f"按化学式联网取结构: {fml}")
        rows = fetch_rows(fml)
    else:
        path = Path(args.structures) if args.structures else DEFAULT_STRUCTURES
        if not path.exists():
            print(f"[ERROR] 找不到结构文件 {path}")
            print("        可先用 fetch_mp_structures.py 拉取，例如：")
            print("          python src/fetch_mp_structures.py --mode formulas "
                  "--formulas \"LiCoO2,LiFePO4\"")
            return 2
        rows = load_rows(path)
        print(f"结构文件: {path}（{len(rows)} 条）")

    if not rows:
        print("[ERROR] 没有结构可预测")
        return 1

    df = predict(Path(args.model), rows)
    cols = [c for c in ("formula", "material_id", "nsites", "cgcnn_e_form_eV_atom",
                        "cgcnn_e_form_eV_cell", "true_e_form_per_atom",
                        "abs_err_eV_atom", "energy_above_hull") if c in df.columns]
    print("\n" + "=" * 78)
    print("  CGCNN 预测（eV/atom；越负越稳定）")
    print("=" * 78)
    print(df[cols].round(4).to_string(index=False))
    if "abs_err_eV_atom" in df.columns and df["abs_err_eV_atom"].notna().any():
        v = df["abs_err_eV_atom"].dropna()
        print("-" * 78)
        print(f"  与 MP 真实值对比（仅 demo 结构带标签）: n={len(v)} MAE={v.mean():.4f} eV/atom")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {Path(args.out).relative_to(ROOT) if str(args.out).startswith(str(ROOT)) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())