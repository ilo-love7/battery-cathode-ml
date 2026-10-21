# -*- coding: utf-8 -*-
"""不确定性校准：把「模型分歧」升级为有覆盖率保证的预测区间（split conformal）。

背景：项目原先只用 ``|RF-HGB|/2`` 当不确定性代理——那只是两个模型的**分歧度**，
没有任何覆盖率的含义。本模块用 **split conformal prediction** 给出预测区间：

1. 取 OOF 预测的绝对残差 ``|y - ŷ|`` 作为非一致性分数；
2. 在**校准集**上取 ``q = quantile(1-α)``（有限样本修正：``ceil((n+1)(1-α))/n`` 分位）；
3. 预测区间 = ``ŷ ± q``，理论上满足 ``P(y ∈ 区间) ≥ 1-α``（只需样本可交换）。

由于项目已有 5 折 OOF，这里再嵌一层 **K 折交叉校准**：在校准折上取分位数、
在留出折上测覆盖率，从而得到**诚实的覆盖率估计**（避免"自己校自己"的乐观偏差），
并据此画可靠性曲线（nominal vs empirical coverage）。

用法::

    python src/conformal.py --tag voltage           # 电压（真实 MP 数据）
    python src/conformal.py --tag capacity          # 比容量
    python src/conformal.py --tag voltage --alpha 0.05

输入:  results/oof_{tag}.csv     （train_voltage.py / train_capacity.py 产出）
输出:  results/conformal_{tag}.json, results/calibration_{tag}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from models import set_plot_style  # noqa: E402

OUT = ROOT / "results"
RNG = 42


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """split conformal 的有限样本修正分位数。

    ``q = 第 ceil((n+1)(1-alpha)) 小的分数``（越界则取最大值），
    保证覆盖率 ≥ 1-alpha（样本可交换假设下）。
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("校准集为空")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(np.max(scores))
    return float(np.sort(scores)[k - 1])


def cross_validated_coverage(scores: np.ndarray, alpha: float, n_splits: int = 5) -> float:
    """K 折交叉校准覆盖率：在校准折取 q，在留出折统计命中率。"""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n < n_splits * 2:
        return float("nan")
    idx = np.random.default_rng(RNG).permutation(n)
    folds = np.array_split(idx, n_splits)
    hit, total = 0, 0
    for i in range(n_splits):
        cal = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        test = folds[i]
        q = conformal_quantile(scores[cal], alpha)
        hit += int(np.sum(scores[test] <= q))
        total += len(test)
    return hit / total


def reliability_curve(scores: np.ndarray, alphas) -> list:
    """返回 [(nominal, empirical)] 列表，用于画可靠性曲线。"""
    return [(1.0 - a, cross_validated_coverage(scores, a)) for a in alphas]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="split conformal 不确定性校准")
    ap.add_argument("--tag", default="voltage", help="对应 results/oof_{tag}.csv")
    ap.add_argument("--alpha", type=float, default=0.10, help="显著性水平（0.10 -> 90%% 区间）")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--label", default=None, help="区间单位标签（默认按 tag 推断）")
    args = ap.parse_args(argv)

    import pandas as pd
    oof_path = OUT / f"oof_{args.tag}.csv"
    if not oof_path.exists():
        print(f"[ERROR] 找不到 {oof_path}")
        print("        请先训练对应模型（如 python src/train_voltage.py ... 或 train_capacity.py）")
        return 2

    df = pd.read_csv(oof_path, encoding="utf-8-sig")
    y = df["y_true"].to_numpy(dtype=float)
    pred = df["pred_ens"].to_numpy(dtype=float)
    scores = np.abs(y - pred)

    unit = args.label or {"voltage": "V", "capacity": "mAh/g"}.get(args.tag.split("_")[0], "")
    q = conformal_quantile(scores, args.alpha)
    cv_cov = cross_validated_coverage(scores, args.alpha, args.splits)

    alphas = [0.20, 0.10, 0.05, 0.02, 0.01]
    curve = reliability_curve(scores, alphas)

    payload = {
        "tag": args.tag,
        "n_oof": int(len(scores)),
        "alpha": args.alpha,
        "nominal_coverage": round(1.0 - args.alpha, 4),
        "q_halfwidth": round(q, 4),
        "unit": unit,
        "cv_coverage": round(cv_cov, 4),
        "mae": round(float(np.mean(scores)), 4),
        "median_halfwidth_ratio": round(float(np.median(scores) / (np.median(np.abs(y)) + 1e-12)), 4),
        "reliability_curve": [{"nominal": round(n, 4), "empirical": round(c, 4)}
                              for n, c in curve],
        "note": ("区间 = 预测值 ± q_halfwidth；q 由 OOF 绝对残差的分位数给出（有限样本修正）。"
                 "cv_coverage 为 K 折交叉校准下测得的真实覆盖率，用作诚实的覆盖估计。"),
    }
    with open(OUT / f"conformal_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 可靠性曲线 ----
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    nominals = [c["nominal"] for c in payload["reliability_curve"]]
    empirical = [c["empirical"] for c in payload["reliability_curve"]]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="理想校准")
    ax.plot(nominals, empirical, "o-", color="#2166ac", label="实测覆盖率")
    ax.set_xlabel("名义覆盖率 (1-α)")
    ax.set_ylabel("K 折交叉校准实测覆盖率")
    ax.set_title(f"Conformal 可靠性曲线（{args.tag}）\nn={len(scores)}, "
                 f"90% 区间半宽 ±{q:.3f} {unit}", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / f"calibration_{args.tag}.png", dpi=150)
    plt.close(fig)

    print(f"样本 {len(scores)} | MAE={payload['mae']:.4f} {unit}")
    print(f"{payload['nominal_coverage']*100:.0f}% 区间半宽 q = {q:.4f} {unit} "
          f"（预测值 ± q）")
    print(f"K 折交叉校准实测覆盖率 = {cv_cov*100:.1f}%  "
          f"(名义 {payload['nominal_coverage']*100:.0f}%)")
    print("可靠性曲线：")
    for c in payload["reliability_curve"]:
        print(f"  名义 {c['nominal']*100:5.1f}%  ->  实测 {c['empirical']*100:5.1f}%")
    print(f"[SAVE] results/conformal_{args.tag}.json")
    print(f"[SAVE] results/calibration_{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())