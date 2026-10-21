# -*- coding: utf-8 -*-
"""CGCNN 在 **Materials Project 含 Li 材料**上训练（真实 DFT 形成能，eV/atom）。

与 ``train_cgcnn.py``（matbench 钙钛矿，5 原子单胞）的区别：

| | train_cgcnn.py | train_cgcnn_mp.py（本脚本） |
|---|---|---|
| 数据 | matbench_perovskites 18,928 | MP 含 Li 材料（默认仓库内 3,000 子集，本地可扩到 8,000） |
| 目标 | e_form **per 5 原子单胞** | e_form **per atom**（MP 原生口径） |
| 用途 | 与 matbench 榜单对齐 | **锂电相关体系**的结构基线 + 给预测 CLI 提供模型 |
| 基线对照 | 成分+结构描述符 0.1929 eV/胞 | 同数据上的**成分模型**（33 统计 + 元素分数，RF/HGB） |

训练完成后会输出 ``results/cgcnn_mp.pt``（含超参与目标标准化参数），
供 ``predict_cgcnn.py`` 直接加载做推理。

用法::

    # 仓库内 3,000 条子集（无需 API Key）
    python src/train_cgcnn_mp.py --epochs 120

    # 本地更全的 8,000 条（需先 fetch：python src/fetch_mp_structures.py --mode li --n 8000）
    python src/train_cgcnn_mp.py --data data/raw/mp_structures_Li.json.gz --tag cgcnn_mp_8k

输出::

    results/cgcnn_mp.pt, results/metrics_cgcnn_mp.json, results/parity_cgcnn_mp.png
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cgcnn_data import CUTOFF, MAX_NEIGHBORS, build_graph  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed" / "mp_structures_Li.json.gz"
OUT = ROOT / "results"
TARGET = "e_form_per_atom"


def load_structures(path: Path, limit: int = 0):
    """读取结构 json.gz -> (graphs, y, formulas, ids)。"""
    if not path.exists():
        raise SystemExit(
            f"[ERROR] 找不到 {path}\n"
            "        仓库内子集应随包提供；若缺失请用 "
            "python src/fetch_mp_structures.py --mode li --n 3000 重新生成")
    rows = json.load(gzip.open(path, "rt", encoding="utf-8"))
    graphs, ys, formulas, ids = [], [], [], []
    t0 = time.time()
    for r in rows:
        st, y = r.get("structure"), r.get(TARGET)
        if st is None or y is None:
            continue
        g = build_graph(st)
        if g is None:
            continue
        graphs.append(g)
        ys.append(float(y))
        formulas.append(r.get("formula") or "")
        ids.append(r.get("material_id") or "")
        if limit and len(graphs) >= limit:
            break
    print(f"构建晶体图: {len(graphs)} 个（{time.time()-t0:.0f}s）；"
          f"元素/位点多样，平均原子数 {np.mean([g['n_atoms'] for g in graphs]):.1f}")
    return graphs, np.asarray(ys, dtype=np.float32), formulas, ids


def composition_baseline(formulas, y, train_idx, test_idx, seed: int = 42):
    """同数据上的成分模型（33 统计 + 元素分数 + RF/HGB 集成），作为对照。"""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("[WARN] 未安装 scikit-learn，跳过成分基线对照")
        return None

    from features import build_universe, featurize_dataframe

    els = build_universe([f for f in formulas if f])
    X, failed = featurize_dataframe(formulas, element_list=els, verbose=False)
    keep = np.array([i for i in range(len(formulas)) if i not in set(failed)])
    X = X.iloc[keep].reset_index(drop=True)
    yk = y[keep]
    tr = np.array([i for i in train_idx if i in set(keep.tolist())])
    te = np.array([i for i in test_idx if i in set(keep.tolist())])
    if len(tr) < 50 or len(te) < 20:
        print("[WARN] 成分基线样本不足，跳过")
        return None

    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               n_jobs=-1, random_state=seed))])
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                        max_leaf_nodes=31, l2_regularization=1.0,
                                        random_state=seed)
    preds = []
    for m in (rf, hgb):
        m.fit(X.iloc[tr], yk[tr])
        preds.append(m.predict(X.iloc[te]))
    pred = 0.5 * (preds[0] + preds[1])
    err = np.abs(pred - yk[te])
    return {"n_train": int(len(tr)), "n_test": int(len(te)),
            "MAE_eV_atom": float(err.mean()),
            "note": "同数据、同划分上的成分模型（无结构信息）"}


def evaluate(model, graphs, idx, y_raw, mean, std, batch_size, device):
    import torch
    from cgcnn_model import batch_tensors

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            (node, efea, src, dst, bvec), _ = batch_tensors(graphs, chunk, device,
                                                           y_raw[chunk])
            out = model(node, efea, src, dst, bvec) * std + mean
            preds.append(out.detach().cpu().numpy())
    pred = np.concatenate(preds) if preds else np.asarray([])
    true = y_raw[idx]
    err = pred - true
    return {"n": int(len(idx)), "MAE_eV_atom": float(np.mean(np.abs(err))),
            "RMSE_eV_atom": float(np.sqrt(np.mean(err ** 2))),
            "pred": pred, "true": true}


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="CGCNN 训练（Materials Project 含 Li 材料）")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--tag", default="cgcnn_mp")
    ap.add_argument("--epochs", type=int, default=120)
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
    ap.add_argument("--no-baseline", action="store_true", help="跳过长成分基线对照")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("[ERROR] 未安装 PyTorch。GPU 环境搭建见 README「GPU 结构模型（CGCNN）」。")
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] torch 不支持 CUDA，改用 CPU")
        device = torch.device("cpu")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})"
                               if device.type == "cuda" else ""))

    if args.limit and args.tag == "cgcnn_mp":
        args.tag = "cgcnn_mp_smoke"
        print(f"[冒烟模式] 输出 tag 改为 {args.tag}")

    graphs, y, formulas, ids = load_structures(Path(args.data), args.limit)
    if len(graphs) < 100:
        print("[ERROR] 样本过少（需要 ≥100），检查 --data/--limit")
        return 2
    print(f"目标 {TARGET}: 均值 {y.mean():.3f} ± {y.std():.3f} eV/atom | "
          f"MAD {np.abs(y - np.median(y)).mean():.3f}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(graphs))
    n_train, n_val = int(0.8 * len(graphs)), int(0.1 * len(graphs))
    train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    mean, std = float(y[train_idx].mean()), float(y[train_idx].std())

    from cgcnn_model import batch_tensors, build_model, n_params

    torch.manual_seed(args.seed)
    model = build_model(args.atom_fea_len, args.n_conv, args.h_fea_len, args.n_h).to(device)
    print(f"模型参数量: {n_params(model)/1e3:.1f}k")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = torch.nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            chunk = order[s:s + args.batch_size]
            (node, efea, src, dst, bvec), yy = batch_tensors(
                graphs, chunk, device, (y[chunk] - mean) / std)
            opt.zero_grad()
            loss = loss_fn(model(node, efea, src, dst, bvec), yy)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        val = evaluate(model, graphs, val_idx, y, mean, std, args.batch_size * 4, device)
        sched.step(val["MAE_eV_atom"])
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1),
                        "val_MAE_eV_atom": val["MAE_eV_atom"]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={tot/max(nb,1):.4f}  "
                  f"val_MAE={val['MAE_eV_atom']:.4f} eV/atom  ({time.time()-t0:.0f}s)")
        if val["MAE_eV_atom"] < best_val - 1e-5:
            best_val, bad = val["MAE_eV_atom"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  早停：验证集 {args.patience} 轮未改善（epoch {epoch}）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tr = evaluate(model, graphs, train_idx, y, mean, std, args.batch_size * 4, device)
    va = evaluate(model, graphs, val_idx, y, mean, std, args.batch_size * 4, device)
    te = evaluate(model, graphs, test_idx, y, mean, std, args.batch_size * 4, device)
    elapsed = time.time() - t0

    baseline = None if args.no_baseline else composition_baseline(formulas, y, train_idx, test_idx)

    OUT.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hyper": vars(args),
                "target": TARGET, "target_mean": mean, "target_std": std,
                "atom_fea_len": args.atom_fea_len, "n_conv": args.n_conv,
                "h_fea_len": args.h_fea_len, "n_h": args.n_h,
                "cutoff": args.cutoff, "max_neighbors": args.max_neighbors,
                "trained_on": str(Path(args.data).name), "n_train": int(len(train_idx))},
               OUT / f"{args.tag}.pt")

    metrics = {
        "mode": "subset-smoke" if args.limit else "full",
        "data": str(Path(args.data).name),
        "n_graphs": len(graphs),
        "target": TARGET,
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "hyper": {k: v for k, v in vars(args).items() if k != "data"},
        "n_params": n_params(model),
        "elapsed_sec": round(elapsed, 1),
        "epochs_run": len(history),
        "train": {k: round(v, 4) for k, v in tr.items() if k.startswith(("MAE", "RMSE"))},
        "val": {k: round(v, 4) for k, v in va.items() if k.startswith(("MAE", "RMSE"))},
        "test": {k: round(v, 4) for k, v in te.items() if k.startswith(("MAE", "RMSE"))},
        "composition_baseline": baseline,
        "history": history,
        "reference": {
            "comp_plus_struct_on_perovskites_eV_cell": 0.1929,
            "cgcnn_on_perovskites_test_eV_cell": 0.0401,
            "mp_li_comp_external_offset_corrected_eV_atom": 0.7347,
        },
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
    lims = [float(min(t_.min(), p_.min())) - 0.3, float(max(t_.max(), p_.max())) + 0.3]
    ax.scatter(t_, p_, s=10, alpha=0.4, edgecolors="none", color="#1b7837")
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("真实形成能 (eV/atom)")
    ax.set_ylabel("CGCNN 预测 (eV/atom)")
    ax.set_title(f"CGCNN on MP Li 材料（test n={te['n']}）\n"
                 f"MAE={te['MAE_eV_atom']:.4f} eV/atom", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"parity_{args.tag}.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 72)
    print(f"  CGCNN（MP 含 Li 材料，{metrics['data']}）  目标 = e_form (eV/atom)")
    print("=" * 72)
    for name, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s} n={m['n']:6d}  MAE={m['MAE_eV_atom']:.4f}  "
              f"RMSE={m['RMSE_eV_atom']:.4f} eV/atom")
    if baseline:
        print("-" * 72)
        print(f"  成分基线（同数据同划分，无结构）: MAE={baseline['MAE_eV_atom']:.4f} eV/atom")
        print(f"  结构模型相对成分基线提升: {baseline['MAE_eV_atom']/te['MAE_eV_atom']:.1f}x")
    print(f"  目标分布 MAD={np.abs(y - np.median(y)).mean():.4f} eV/atom（中位数基线参照）")
    print(f"  用时 {elapsed:.0f}s（{len(history)} 轮）| device={device}")
    print(f"[SAVE] results/{args.tag}.pt / metrics_{args.tag}.json / parity_{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())