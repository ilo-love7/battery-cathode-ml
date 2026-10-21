# -*- coding: utf-8 -*-
"""CGCNN 结构图神经网络（PyTorch，自包含实现，无需克隆外部仓库）。

为什么需要它
------------
CPU 结构描述符基线已把形成能 MAE 从 0.484 压到 **0.1929 eV/胞**（见 train_ef_struct.py），
但几何统计描述符毕竟是"手搓特征"。CGCNN（Xie & Grossman, PRL 120, 145301, 2018）
直接在晶体图上做卷积、端到端学习，matbench 榜首结构模型（coGN）可达 **0.0269 eV/胞**。

实现要点（与原文一致的核心部分）
--------------------------------
- 节点特征：原子序数 one-hot（95 维）；边特征：距离高斯展开（0–8 Å，步长 0.2 Å → 41 维）
- 卷积层：``x_i' = x_i + Σ_j σ(BN(W_f·[x_i‖x_j‖e_ij])) ⊙ softplus(BN(W_s·[x_i‖x_j‖e_ij]))``
- 用**扁平边索引 + index_add** 聚合（不 padding），小晶胞下更省显存
- 池化：每个晶体的原子特征取平均 → 2 层全连接 → 标量回归
- 目标标准化后训练 MSE，报告时还原为 eV/胞与 eV/atom
- 划分：matbench 标准 8:1:1 随机划分，随机种子固定

用法::

    # 冒烟（几百条、几轮，验证链路）
    python src/train_cgcnn.py --limit 500 --epochs 3

    # 正式训练（RTX 4050 6 GB 足够；100 轮数分钟）
    python src/train_cgcnn.py --epochs 100

输出::

    results/cgcnn_model.pt, results/metrics_cgcnn.json, results/parity_cgcnn.png

⚠️ 需要 CUDA 版 torch（本项目在 D:\\项目\\venvs\\bcm-cgcnn 里建了 Python 3.12 + cu124 环境）。
没有 GPU 时脚本仍可用 ``--device cpu`` 跑通小规模，但全量训练会很慢。
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

from cgcnn_data import ATOM_FEAT_DIM, CUTOFF, EDGE_FEAT_DIM, MAX_NEIGHBORS, build_graph  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "matbench_perovskites.json.gz"
OUT = ROOT / "results"
N_SITES_PER_CELL = 5.0          # matbench_perovskites 固定 5 原子单胞


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def load_graphs(raw: Path, limit: int = 0, cutoff: float = CUTOFF,
                max_neighbors: int = MAX_NEIGHBORS):
    """读取 matbench 结构 -> (graphs, targets, formulas)。"""
    if not raw.exists():
        raise SystemExit(f"[ERROR] 找不到 {raw}\n        请先运行: python src/download_data.py")
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cols = d["columns"]
    i_st, i_ef = cols.index("structure"), cols.index("e_form")

    graphs, targets = [], []
    t0 = time.time()
    for k, rec in enumerate(d["data"]):
        st, ef = rec[i_st], rec[i_ef]
        if st is None or ef is None:
            continue
        g = build_graph(st, cutoff=cutoff, max_neighbors=max_neighbors)
        if g is None:
            continue
        graphs.append(g)
        targets.append(float(ef))
        if limit and len(graphs) >= limit:
            break
    print(f"构建晶体图: {len(graphs)} 个（{time.time() - t0:.0f}s，"
          f"cutoff={cutoff}Å, max_neighbors={max_neighbors}）")
    return graphs, np.asarray(targets, dtype=np.float32)


def batch_tensors(graphs, idx, device, target=None):
    """把一批图拼成 (atom_fea, edge_fea, edge_src, edge_dst, batch_idx) 张量。"""
    import torch

    node, efea, src, dst, bvec = [], [], [], [], []
    a_off = 0
    for b, i in enumerate(idx):
        g = graphs[i]
        n = int(g["n_atoms"])
        node.append(torch.as_tensor(g["node_feat"]))
        efea.append(torch.as_tensor(g["edge_feat"]))
        src.append(torch.as_tensor(g["edge_src"] + a_off))
        dst.append(torch.as_tensor(g["edge_dst"] + a_off))
        bvec.append(torch.full((n,), b, dtype=torch.long))
        a_off += n

    out = (torch.cat(node), torch.cat(efea), torch.cat(src), torch.cat(dst), torch.cat(bvec))
    out = tuple(t.to(device, non_blocking=True) for t in out)
    if target is None:
        return out
    y = torch.as_tensor(np.asarray(target, dtype=np.float32), device=device)
    return out, y


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def build_model(atom_fea_len: int, n_conv: int, h_fea_len: int, n_h: int):
    import torch
    import torch.nn as nn

    class ConvLayer(nn.Module):
        """CGCNN 门控卷积（扁平边索引 + index_add 聚合）。"""

        def __init__(self, atom_fea_len: int, nbr_fea_len: int):
            super().__init__()
            self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
            self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
            self.bn2 = nn.BatchNorm1d(atom_fea_len)
            self.softplus = nn.Softplus()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, edge_feat, src, dst):
            total = torch.cat([x[src], x[dst], edge_feat], dim=1)
            full = self.bn1(self.fc_full(total))
            gate, core = torch.chunk(full, 2, dim=1)
            msg = self.sigmoid(gate) * self.softplus(self.bn2(core))
            aggr = torch.zeros_like(x).index_add_(0, dst, msg)
            return x + aggr

    class CrystalGraphConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(ATOM_FEAT_DIM, atom_fea_len)
            self.convs = nn.ModuleList(
                [ConvLayer(atom_fea_len, EDGE_FEAT_DIM) for _ in range(n_conv)])
            self.fc_in = nn.Linear(atom_fea_len, h_fea_len)
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h)])
            self.fc_out = nn.Linear(h_fea_len, 1)
            self.softplus = nn.Softplus()

        def forward(self, atom_fea, edge_fea, edge_src, edge_dst, batch_idx):
            x = self.embedding(atom_fea)
            for conv in self.convs:
                x = conv(x, edge_fea, edge_src, edge_dst)
            n_crystals = int(batch_idx.max().item()) + 1
            pooled = torch.zeros(n_crystals, x.shape[1], device=x.device).index_add_(0, batch_idx, x)
            counts = torch.zeros(n_crystals, device=x.device).index_add_(
                0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
            pooled = pooled / counts.unsqueeze(1)
            h = self.softplus(self.fc_in(pooled))
            for fc in self.fcs:
                h = self.softplus(fc(h))
            return self.fc_out(h).squeeze(1)

    return CrystalGraphConvNet()


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def evaluate(model, graphs, idx, y_raw, mean, std, batch_size, device) -> dict:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            (node, efea, src, dst, bvec), y = batch_tensors(graphs, chunk, device,
                                                           y_raw[chunk])
            out = model(node, efea, src, dst, bvec) * std + mean
            preds.append(out.detach().cpu().numpy())
    pred = np.concatenate(preds) if preds else np.asarray([])
    true = y_raw[idx]
    err = pred - true
    return {
        "n": int(len(idx)),
        "MAE_eV_cell": float(np.mean(np.abs(err))),
        "RMSE_eV_cell": float(np.sqrt(np.mean(err ** 2))),
        "MAE_eV_atom": float(np.mean(np.abs(err)) / N_SITES_PER_CELL),
        "pred": pred,
        "true": true,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CGCNN 训练（晶体图神经网络）")
    ap.add_argument("--raw", default=str(RAW_PATH))
    ap.add_argument("--tag", default="cgcnn")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--h-fea-len", type=int, default=128)
    ap.add_argument("--n-h", type=int, default=1)
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（冒烟用）")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("[ERROR] 未安装 PyTorch。CGCNN 需要 torch（建议 CUDA 版）。")
        print("        Windows 上的现成环境：")
        print("          C:\\Users\\<you>\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -m venv D:\\venvs\\bcm-cgcnn")
        print("          D:\\venvs\\bcm-cgcnn\\Scripts\\python.exe -m pip install torch "
              "--index-url https://download.pytorch.org/whl/cu124")
        print("          D:\\venvs\\bcm-cgcnn\\Scripts\\python.exe -m pip install numpy pandas matplotlib")
        print("        纯 CPU 也可跑通（--device cpu），但全量训练很慢，建议先用 --limit 冒烟。")
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] 当前 torch 不支持 CUDA（可能是 CPU 版），改用 CPU")
        device = torch.device("cpu")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})"
                               if device.type == "cuda" else ""))

    if args.limit and args.tag == "cgcnn":
        args.tag = "cgcnn_smoke"
        print(f"[冒烟模式] --limit 生效，输出 tag 改为 {args.tag}")

    graphs, y_raw = load_graphs(Path(args.raw), args.limit, args.cutoff, args.max_neighbors)
    if len(graphs) < 50:
        print("[ERROR] 样本过少，检查数据或 --limit")
        return 2

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(graphs))
    n_train = int(0.8 * len(graphs))
    n_val = int(0.1 * len(graphs))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    mean = float(y_raw[train_idx].mean())
    std = float(y_raw[train_idx].std())
    print(f"target 标准化: mean={mean:.3f} std={std:.3f} eV/胞 "
          f"(MAD={(np.abs(y_raw[train_idx] - np.median(y_raw[train_idx])).mean()):.3f} eV/胞)")

    torch.manual_seed(args.seed)
    model = build_model(args.atom_fea_len, args.n_conv, args.h_fea_len, args.n_h).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_param/1e3:.1f}k | conv={args.n_conv} atom_fea={args.atom_fea_len} "
          f"h_fea={args.h_fea_len}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = torch.nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            chunk = order[s:s + args.batch_size]
            (node, efea, src, dst, bvec), y = batch_tensors(
                graphs, chunk, device, (y_raw[chunk] - mean) / std)
            opt.zero_grad()
            out = model(node, efea, src, dst, bvec)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1

        val = evaluate(model, graphs, val_idx, y_raw, mean, std, args.batch_size * 4, device)
        sched.step(val["MAE_eV_cell"])
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1),
                        "val_MAE_eV_cell": val["MAE_eV_cell"]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={tot/max(nb,1):.4f}  "
                  f"val_MAE={val['MAE_eV_cell']:.4f} eV/胞  "
                  f"({time.time() - t0:.0f}s)")

        if val["MAE_eV_cell"] < best_val - 1e-5:
            best_val = val["MAE_eV_cell"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  早停：验证集 {args.patience} 轮未改善（epoch {epoch}）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_m = evaluate(model, graphs, train_idx, y_raw, mean, std, args.batch_size * 4, device)
    val_m = evaluate(model, graphs, val_idx, y_raw, mean, std, args.batch_size * 4, device)
    test_m = evaluate(model, graphs, test_idx, y_raw, mean, std, args.batch_size * 4, device)
    elapsed = time.time() - t0

    OUT.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "hyper": vars(args),
                "target_mean": mean, "target_std": std,
                "atom_fea_len": args.atom_fea_len, "n_conv": args.n_conv,
                "h_fea_len": args.h_fea_len, "n_h": args.n_h,
                "atom_feat_dim": ATOM_FEAT_DIM, "edge_feat_dim": EDGE_FEAT_DIM},
               OUT / f"{args.tag}_model.pt")

    metrics = {
        "mode": "subset-smoke" if args.limit else "full",
        "n_graphs": len(graphs),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "hyper": {k: v for k, v in vars(args).items() if k != "raw"},
        "n_params": int(n_param),
        "elapsed_sec": round(elapsed, 1),
        "epochs_run": len(history),
        "target_mean_eV_cell": round(mean, 4),
        "target_std_eV_cell": round(std, 4),
        "train": {k: round(v, 4) for k, v in train_m.items() if k.startswith(("MAE", "RMSE"))},
        "val": {k: round(v, 4) for k, v in val_m.items() if k.startswith(("MAE", "RMSE"))},
        "test": {k: round(v, 4) for k, v in test_m.items() if k.startswith(("MAE", "RMSE"))},
        "history": history,
        "reference": {"comp_only_MAE_eV_cell": 0.4837,
                      "comp_plus_struct_MAE_eV_cell": 0.1929,
                      "matbench_topped_coGN_MAE_eV_cell": 0.0269},
    }
    with open(OUT / f"metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 平差图（不依赖 sklearn：只用 matplotlib 基础配置）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    t, p = test_m["true"], test_m["pred"]
    lims = [float(min(t.min(), p.min())) - 0.2, float(max(t.max(), p.max())) + 0.2]
    ax.scatter(t, p, s=12, alpha=0.45, edgecolors="none", color="#762a83")
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("真实形成能 (eV/5原子胞)")
    ax.set_ylabel("CGCNN 预测 (eV/5原子胞)")
    ax.set_title(f"CGCNN 结构图神经网络（test n={test_m['n']}）\n"
                 f"MAE={test_m['MAE_eV_cell']:.4f} eV/胞 "
                 f"({test_m['MAE_eV_atom']:.4f} eV/atom)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"parity_{args.tag}.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("  CGCNN 结果（eV/5原子胞）")
    print("=" * 70)
    for name, m in (("train", train_m), ("val", val_m), ("test", test_m)):
        print(f"  {name:5s} n={m['n']:6d}  MAE={m['MAE_eV_cell']:.4f}  "
              f"RMSE={m['RMSE_eV_cell']:.4f}  ({m['MAE_eV_atom']:.4f} eV/atom)")
    print("-" * 70)
    print(f"  参照：成分基线 0.4837 | CPU 成分+结构描述符 0.1929 | matbench 榜首 coGN 0.0269")
    print(f"  用时 {elapsed:.0f}s（{len(history)} 轮）| device={device}")
    print(f"[SAVE] results/{args.tag}_model.pt / metrics_{args.tag}.json / parity_{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())