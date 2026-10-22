# -*- coding: utf-8 -*-
"""双塔 CGCNN 电压**推理 CLI**：化学式（或电极结构对）→ 平均电压。

补齐电压链路最后一环（与 ``predict_cgcnn.py`` 之于形成能同理）：

- ``--pairs``：读电极对文件（``fetch_mp_structures.py --mode pairs`` 的产物，
  仓库内附 10 个常见正极 demo，可离线跑）；
- ``--formulas``：给化学式，联网到 Materials Project 取**最典型电极对**（含两相结构）再预测；
- 模型默认 ``results/cgcnn_voltage.pt``（双塔 CGCNN，test MAE 0.3525 V）。

用法::

    # 离线 demo（仓库内 10 个常见正极，含 MP 真实电压可对比）
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/predict_voltage_cgcnn.py `
        --pairs data/raw/mp_voltage_pairs_demo.json.gz

    # 任意化学式（联网，需 MP Key）
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/predict_voltage_cgcnn.py `
        --formulas "LiFePO4,LiCoO2,LiMn2O4"

输出::

    results/voltage_cgcnn_predictions.csv
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

DEFAULT_MODEL = ROOT / "results" / "cgcnn_voltage.pt"
DEFAULT_PAIRS = ROOT / "data" / "raw" / "mp_voltage_pairs_demo.json.gz"
OUT_CSV = ROOT / "results" / "voltage_cgcnn_predictions.csv"


def load_pairs(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def fetch_pairs_online(formulas: list) -> list:
    try:
        import fetch_mp_structures as fms
    except ImportError:
        print("[ERROR] 无法导入 fetch_mp_structures（联网模式需要 mp-api）")
        return []
    return fms.fetch_pairs(fms.resolve_key(None), formulas)


def predict(model_path: Path, rows: list) -> pd.DataFrame:
    import torch
    from cgcnn_model import batch_tensors

    if not model_path.exists():
        raise SystemExit(
            f"[ERROR] 找不到模型 {model_path}\n"
            "        请先训练: D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe "
            "src/train_cgcnn_voltage.py --epochs 120")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 双塔模型 = 共享编码器 + 差分拼接 + MLP；这里按保存的超参重建
    from train_cgcnn_voltage import build_pair_net

    model = build_pair_net(None, int(ckpt.get("atom_fea_len", 64)), int(ckpt.get("n_conv", 3)),
                           int(ckpt.get("h_fea_len", 128)), n_scalar=3, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    print(f"模型 {model_path.name} | 目标 {ckpt.get('target')} | device={device}")

    usable = []
    for r in rows:
        gc = build_graph(r.get("structure_charge")) if r.get("structure_charge") else None
        gl = build_graph(r.get("structure_discharge")) if r.get("structure_discharge") else None
        if gc is None or gl is None:
            continue
        usable.append((r, gc, gl))
    if not usable:
        raise SystemExit("[ERROR] 没有可用电极对（检查文件格式）")

    preds = []
    with torch.no_grad():
        for s in range(0, len(usable), 64):
            chunk = usable[s:s + 64]
            ga = [c[1] for c in chunk]
            gb = [c[2] for c in chunk]
            a = batch_tensors(ga, list(range(len(ga))), device)
            b = batch_tensors(gb, list(range(len(gb))), device)
            n_f = np.array([int(c[1]["n_atoms"]) for c in chunk], dtype=np.float32)
            n_l = np.array([int(c[2]["n_atoms"]) for c in chunk], dtype=np.float32)
            sc = torch.as_tensor(np.stack([np.log1p(np.maximum(n_l - n_f, 1)),
                                           n_f / 50.0, n_l / 50.0], axis=1),
                                 device=device)
            out = model(a, b, sc) * std + mean
            preds.append(out.detach().cpu().numpy())
    pred = np.concatenate(preds)

    meta = []
    for (r, gc, gl), p in zip(usable, pred):
        n_f, n_l = int(gc["n_atoms"]), int(gl["n_atoms"])
        meta.append({
            "query_formula": r.get("query_formula") or r.get("formula_discharge"),
            "formula_charge": r.get("formula_charge"),
            "formula_discharge": r.get("formula_discharge"),
            "id_charge": r.get("id_charge"), "id_discharge": r.get("id_discharge"),
            "n_charge": n_f, "n_discharge": n_l, "n_li": max(n_l - n_f, 1),
            "voltage_true_V": r.get("average_voltage_true_V"),
            "voltage_cgcnn_V": float(p),
            "dataset_split": r.get("dataset_split", "unknown"),
        })
    df = pd.DataFrame(meta)
    if df["voltage_true_V"].notna().any():
        df["abs_err_V"] = (df["voltage_cgcnn_V"] - df["voltage_true_V"]).abs()
    return df


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="双塔 CGCNN 电压推理（两相结构 -> 平均电压）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--pairs", default=None, help="电极对文件（json.gz）")
    ap.add_argument("--formulas", default="", help="逗号分隔化学式（联网取电极对）")
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--max-voltage", type=float, default=6.0,
                    help="只保留真实电压 ≤ 该值的记录（MP 有 >6 V 的非物理条目）")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README「GPU 结构模型（CGCNN）」）")
        return 2

    if args.formulas:
        fml = [f.strip() for f in args.formulas.split(",") if f.strip()]
        print(f"按化学式联网取电极对: {fml}")
        rows = fetch_pairs_online(fml)
    else:
        path = Path(args.pairs) if args.pairs else DEFAULT_PAIRS
        if not path.exists():
            print(f"[ERROR] 找不到电极对文件 {path}")
            print("        可先取: python src/fetch_mp_structures.py --mode pairs "
                  "--formulas \"LiFePO4,LiCoO2\" --out data/raw/mp_voltage_pairs_demo.json.gz")
            return 2
        rows = load_pairs(path)
        print(f"电极对文件: {path}（{len(rows)} 条）")

    if args.max_voltage:
        before = len(rows)
        rows = [r for r in rows
                if r.get("average_voltage_true_V") is None
                or float(r["average_voltage_true_V"]) <= args.max_voltage]
        if len(rows) < before:
            print(f"按 {args.max_voltage} V 上限过滤掉 {before-len(rows)} 条非物理条目")

    df = predict(Path(args.model), rows)

    cols = [c for c in ("query_formula", "formula_charge", "formula_discharge", "n_li",
                        "voltage_true_V", "voltage_cgcnn_V", "abs_err_V",
                        "dataset_split") if c in df.columns]
    print("\n" + "=" * 84)
    print("  双塔 CGCNN 电压预测（V）")
    print("=" * 84)
    print(df[cols].round(3).to_string(index=False))
    if "abs_err_V" in df.columns and df["abs_err_V"].notna().any():
        print("-" * 84)
        print("  与 MP 真实电压对比（按数据划分分组，避免拿训练样本冒充精度）")
        for grp in ("train", "val", "test", "unseen", "unknown"):
            sub = df[df["dataset_split"] == grp].dropna(subset=["abs_err_V"])
            if len(sub):
                tag = {"train": "训练集内（in-sample，仅供参考）",
                       "val": "验证集", "test": "★ 测试集（held-out，可信）",
                       "unseen": "★ 训练数据之外（held-out，可信）",
                       "unknown": "未标注"}[grp]
                print(f"    {tag:32s} n={len(sub):3d}  MAE={sub['abs_err_V'].mean():.4f} V")
        held = df[df["dataset_split"].isin(("test", "unseen"))].dropna(subset=["abs_err_V"])
        if len(held):
            print(f"  → held-out 子集合计: n={len(held)}  MAE={held['abs_err_V'].mean():.4f} V "
                  f"（模型在完整测试划分上的 MAE 为 0.3525 V）")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[SAVE] results/{Path(args.out).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())