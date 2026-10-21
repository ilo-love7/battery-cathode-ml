# -*- coding: utf-8 -*-
r"""双塔 CGCNN：直接从**两相结构对**预测平均电压（端到端）。

为什么需要它（v1.6.1 的物理路径诊断）
------------------------------------
把逐原子形成能模型套进热力学公式 ``V = [E_f(脱锂) − E_f(嵌锂)] / n_Li`` 会**放大误差**：
每原子误差 ~0.1 eV/atom，脱锂相单胞可达 80 个原子、n_Li 又常为 1–2，
于是「误差 × 原子数 / n_Li」可达数伏。实测该路径 MAE **0.70 V**，还不如成分模型（0.42 V）。
根因是两相的能量误差**各自独立、不会抵消**。

本脚本训练一个「成对」模型，让网络直接在**同一对结构**上学习差异：

- 共享编码器（默认用 ``cgcnn_mp_8k.pt`` 的卷积权重初始化）分别编码脱锂相与嵌锂相；
- 拼接头向量 ``[h_frame, h_lith, h_frame − h_lith]`` + 标量（log n_Li、原子数、脱锂收缩率）；
- MLP 直接回归电压（目标标准化），因此**系统性偏差天然抵消**。

用法::

    D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/train_cgcnn_voltage.py --epochs 150

输出::

    results/cgcnn_voltage.pt, results/metrics_cgcnn_voltage.json, results/parity_cgcnn_voltage.png

⚠️ 需要 ``data/raw/mp_structures_electrodes.json.gz``（用 fetch_mp_structures.py --mode ids 获取）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cgcnn_data import build_graph  # noqa: E402

VOLTAGE_STRUCT = ROOT / "data" / "processed" / "mp_voltage_struct.csv.gz"
STRUCT_PATH = ROOT / "data" / "raw" / "mp_structures_electrodes.json.gz"
PRETRAINED = ROOT / "results" / "cgcnn_mp_8k.pt"
OUT = ROOT / "results"
TARGET = "average_voltage_V"
REFERENCE_COMP_OOF_MAE = 0.4157        # 同子集上的成分模型 OOF（v1.6.1 实测）
REFERENCE_PHYSICS_MAE = 0.7004        # CGCNN-ΔE 物理路径（v1.6.1 实测）


def build_pair_net(pretrained: "Path | None", atom_fea_len: int, n_conv: int,
                   h_fea_len: int, n_scalar: int, device):
    """共享编码器 + 差异拼接 + MLP 回归（工厂函数，torch 在此处才导入）。"""
    import torch
    import torch.nn as nn

    from cgcnn_model import CrystalGraphConvNet

    class PairVoltageNet(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(3 * atom_fea_len + n_scalar, h_fea_len), nn.Softplus(),
                nn.Linear(h_fea_len, h_fea_len), nn.Softplus(),
                nn.Linear(h_fea_len, 1),
            )

        def forward(self, frame, lith, scalars):
            h_f = self.encoder.encode(*frame)
            h_l = self.encoder.encode(*lith)
            x = torch.cat([h_f, h_l, h_f - h_l, scalars], dim=1)
            return self.head(x).squeeze(1)

    encoder = CrystalGraphConvNet(atom_fea_len, n_conv, atom_fea_len, 1)
    if pretrained is not None and Path(pretrained).exists():
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        enc = CrystalGraphConvNet(ckpt.get("atom_fea_len", 64), ckpt.get("n_conv", 3),
                                  ckpt.get("h_fea_len", 128), ckpt.get("n_h", 1))
        enc.load_state_dict(ckpt["state_dict"])
        encoder.embedding.load_state_dict(enc.embedding.state_dict())
        encoder.convs.load_state_dict(enc.convs.state_dict())
        print(f"编码器初始化自 {Path(pretrained).name}（训练于 {ckpt.get('trained_on')}）")
    else:
        print("[WARN] 未找到预训练权重，编码器随机初始化")
    return PairVoltageNet(encoder).to(device)


def load_pairs(voltage_csv: Path, structures_json: Path, limit: int = 0):
    """构造成对数据：每个样本 = (脱锂相图, 嵌锂相图, 标量, 目标电压)。"""
    if not voltage_csv.exists():
        raise SystemExit(f"[ERROR] 找不到 {voltage_csv}（先跑 prepare_voltage_struct.py）")
    if not structures_json.exists():
        raise SystemExit(
            f"[ERROR] 找不到 {structures_json}\n"
            "        先运行: python src/fetch_mp_structures.py --mode ids "
            "--ids-file data/raw/voltage_material_ids.txt "
            "--out data/raw/mp_structures_electrodes.json.gz")

    df = pd.read_csv(voltage_csv, encoding="utf-8-sig")
    with gzip.open(structures_json, "rt", encoding="utf-8") as f:
        structures = {r["material_id"]: r["structure"] for r in json.load(f)
                      if r.get("structure")}

    pairs, scalars, ys, formulas = [], [], [], []
    skipped = 0
    for _, r in df.iterrows():
        gc = build_graph(structures.get(str(r["id_charge"]))) if pd.notna(r["id_charge"]) else None
        gl = build_graph(structures.get(str(r["id_discharge"]))) if pd.notna(r["id_discharge"]) else None
        if gc is None or gl is None:
            skipped += 1
            continue
        n_f, n_l = int(gc["n_atoms"]), int(gl["n_atoms"])
        n_li = max(n_l - n_f, 1)
        dv = r.get("delta_vol")
        pairs.append((gc, gl))
        scalars.append([float(np.log1p(n_li)), n_f / 50.0, n_l / 50.0])
        ys.append(float(r[TARGET]))
        formulas.append(str(r["formula_canon"]))
        if limit and len(pairs) >= limit:
            break
    print(f"成对样本: {len(pairs)}（跳过 {skipped} 条缺结构）")
    if pairs:
        print(f"n_li 中位数: {np.median([np.exp(s[0]) - 1 for s in scalars]):.1f}")
    return pairs, np.asarray(scalars, dtype=np.float32), np.asarray(ys, dtype=np.float32), \
        np.asarray(formulas)


def batch_pairs(pairs, idx, device, scalars=None, target=None):
    """把一批『对』拼成两个图批次（frame / lith 各自一个 batch）。

    scalars / target 均为**全量数组**，内部按 idx 取值。
    """
    import torch
    from cgcnn_model import batch_tensors

    flat_a, flat_b = [], []
    for i in idx:
        flat_a.append(pairs[i][0])
        flat_b.append(pairs[i][1])
    a = batch_tensors(flat_a, list(range(len(flat_a))), device)
    b = batch_tensors(flat_b, list(range(len(flat_b))), device)
    out = (a, b)
    if scalars is None:
        return out
    s = torch.as_tensor(np.asarray(scalars[idx], dtype=np.float32), device=device)
    if target is None:
        return out, s
    y = torch.as_tensor(np.asarray(target[idx], dtype=np.float32), device=device)
    return out, s, y


def evaluate(model, pairs, scalars, y_raw, idx, mean, std, device, bs=64):
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            chunk = idx[s:s + bs]
            (a, b), sc = batch_pairs(pairs, chunk, device, scalars)
            out = model(a, b, sc) * std + mean
            preds.append(out.detach().cpu().numpy())
    pred = np.concatenate(preds) if preds else np.asarray([])
    true = y_raw[idx]
    err = pred - true
    return {"n": int(len(idx)), "MAE_V": float(np.mean(np.abs(err))),
            "RMSE_V": float(np.sqrt(np.mean(err ** 2))),
            "R2": float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)),
            "pred": pred, "true": true}


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="双塔 CGCNN：两相结构 -> 平均电压")
    ap.add_argument("--voltage", default=str(VOLTAGE_STRUCT))
    ap.add_argument("--structures", default=str(STRUCT_PATH))
    ap.add_argument("--pretrained", default=str(PRETRAINED),
                    help="用于初始化共享编码器的 CGCNN 权重（缺失则随机初始化）")
    ap.add_argument("--tag", default="cgcnn_voltage")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr", type=float, default=2e-4)
    ap.add_argument("--h-fea-len", type=int, default=128)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--group-split", action="store_true",
                    help="按成分分组划分（检验对未见成分的泛化）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README「GPU 结构模型（CGCNN）」）")
        return 2
    if not torch.cuda.is_available():
        print("[WARN] 未检测到 CUDA，将用 CPU（较慢）")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args.limit and args.tag == "cgcnn_voltage":
        args.tag = "cgcnn_voltage_smoke"
        print(f"[冒烟模式] 输出 tag 改为 {args.tag}")

    pairs, scalars, y, formulas = load_pairs(Path(args.voltage), Path(args.structures), args.limit)
    if len(pairs) < 200:
        print("[ERROR] 成对样本过少（<200）")
        return 2

    rng = np.random.default_rng(args.seed)
    if args.group_split:
        uniq = sorted(set(formulas.tolist()))
        rng.shuffle(uniq)
        n_tr = int(0.8 * len(uniq))
        n_va = int(0.1 * len(uniq))
        set_tr, set_va = set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va])
        train_idx = np.array([i for i, f in enumerate(formulas) if f in set_tr])
        val_idx = np.array([i for i, f in enumerate(formulas) if f in set_va])
        test_idx = np.array([i for i, f in enumerate(formulas)
                             if f not in set_tr and f not in set_va])
        print("[划分] 按成分分组（同成分不会跨折）")
    else:
        perm = rng.permutation(len(pairs))
        n_tr, n_va = int(0.8 * len(pairs)), int(0.1 * len(pairs))
        train_idx, val_idx, test_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    mean, std = float(y[train_idx].mean()), float(y[train_idx].std())
    y_norm = (y - mean) / std
    print(f"目标: 均值 {mean:.3f} ± {std:.3f} V | MAD "
          f"{np.abs(y - np.median(y)).mean():.3f} V（中位数基线）")

    torch.manual_seed(args.seed)
    model = build_pair_net(Path(args.pretrained), args.atom_fea_len, args.n_conv,
                          args.h_fea_len, n_scalar=scalars.shape[1], device=device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_par/1e3:.1f}k（其中编码器 "
          f"{sum(p.numel() for p in model.encoder.parameters())/1e3:.1f}k）")

    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.head.parameters(), "lr": args.lr},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            chunk = order[s:s + args.batch_size]
            (a, b), sc, yy = batch_pairs(pairs, chunk, device, scalars, y_norm)
            opt.zero_grad()
            loss = loss_fn(model(a, b, sc), yy)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        val = evaluate(model, pairs, scalars, y, val_idx, mean, std, device)
        sched.step(val["MAE_V"])
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1),
                        "val_MAE_V": val["MAE_V"]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss={tot/max(nb,1):.4f}  "
                  f"val_MAE={val['MAE_V']:.4f} V  ({time.time()-t0:.0f}s)")
        if val["MAE_V"] < best_val - 1e-4:
            best_val, bad = val["MAE_V"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  早停：验证集 {args.patience} 轮未改善（epoch {epoch}）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tr = evaluate(model, pairs, scalars, y, train_idx, mean, std, device)
    va = evaluate(model, pairs, scalars, y, val_idx, mean, std, device)
    te = evaluate(model, pairs, scalars, y, test_idx, mean, std, device)
    elapsed = time.time() - t0

    OUT.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hyper": vars(args),
                "target_mean": mean, "target_std": std, "target": TARGET,
                "atom_fea_len": args.atom_fea_len, "n_conv": args.n_conv,
                "h_fea_len": args.h_fea_len, "group_split": bool(args.group_split)},
               OUT / f"{args.tag}.pt")

    metrics = {
        "mode": "subset-smoke" if args.limit else "full",
        "n_pairs": len(pairs),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "group_split": bool(args.group_split),
        "hyper": {k: v for k, v in vars(args).items() if k not in ("voltage", "structures", "pretrained")},
        "n_params": int(n_par),
        "elapsed_sec": round(elapsed, 1),
        "epochs_run": len(history),
        "train": {k: round(v, 4) for k, v in tr.items() if k != "pred" and k != "true"},
        "val": {k: round(v, 4) for k, v in va.items() if k != "pred" and k != "true"},
        "test": {k: round(v, 4) for k, v in te.items() if k != "pred" and k != "true"},
        "median_baseline_MAD_V": round(float(np.abs(y - np.median(y)).mean()), 4),
        "reference": {"composition_model_OOF_MAE_V": REFERENCE_COMP_OOF_MAE,
                      "physics_dE_path_MAE_V": REFERENCE_PHYSICS_MAE},
        "history": history,
    }
    with open(OUT / f"metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    t_, p_ = te["true"], te["pred"]
    lo, hi = float(min(t_.min(), p_.min())) - 0.3, float(max(t_.max(), p_.max())) + 0.3
    ax.scatter(t_, p_, s=12, alpha=0.4, edgecolors="none", color="#d6604d")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实平均电压 (V)")
    ax.set_ylabel("双塔 CGCNN 预测 (V)")
    ax.set_title(f"双塔 CGCNN 电压预测（test n={te['n']}，"
                 f"{'成分分组' if args.group_split else '随机'}划分）\n"
                 f"MAE={te['MAE_V']:.4f} V  R²={te['R2']:.3f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"parity_{args.tag}.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 76)
    print(f"  双塔 CGCNN 电压模型（{'成分分组' if args.group_split else '随机'}划分）")
    print("=" * 76)
    for name, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s} n={m['n']:5d}  MAE={m['MAE_V']:.4f} V  "
              f"RMSE={m['RMSE_V']:.4f}  R²={m['R2']:.4f}")
    print("-" * 76)
    print(f"  参照：成分模型 OOF {REFERENCE_COMP_OOF_MAE:.4f} V | "
          f"CGCNN-ΔE 物理路径 {REFERENCE_PHYSICS_MAE:.4f} V | "
          f"中位数基线 MAD {metrics['median_baseline_MAD_V']:.4f} V")
    print(f"  用时 {elapsed:.0f}s（{len(history)} 轮）| device={device}")
    print(f"[SAVE] results/{args.tag}.pt / metrics_{args.tag}.json / parity_{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())