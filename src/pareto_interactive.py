# -*- coding: utf-8 -*-
"""Pareto 偏好交互：不同优先级下的最优子集（v1.19.0）。

### 为什么

v1.18.0 发现 **32/40 候选都在 Pareto 前沿上** —— 说明"谁是第一"完全取决于你优先什么。
一个固定权重只能回答一种偏好；本脚本把"偏好"变成**输入**：

    python src/pareto_interactive.py --profile capacity
    python src/pareto_interactive.py --weights 0.5,0.1,0.3,0.1
    python src/pareto_interactive.py --all-profiles         # 一次给出所有预设

四目标（与 v1.18.0 一致，均为"越大越好"的保守量）：

| 目标 | 取值 | 含义 |
|---|---|---|
| 稳定性 | −stability_conservative_eV_atom | 越稳定越好 |
| 导电性 | −gap_conservative_eV | 带隙越小越好 |
| 容量 | Q_theoretical_mAh_g | 比容量 |
| 电压 | voltage_conservative_V | 保守工作电压 |

预设偏好：`balanced`（等权）、`capacity`（高容量）、`safety`（稳定+低风险）、
`energy`（容量×电压）、`power`（导电性优先）。

输出::

    results/pareto_profiles.csv        （候选 × 各偏好下的得分与排名）
    results/metrics_pareto_profiles.json
    results/pareto_profiles.png        （Top 候选在不同偏好下的排名热图）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

RANKING = ROOT / "results" / "cathode_ranking_struct.csv"
OUT_CSV = ROOT / "results" / "pareto_profiles.csv"
OUT_JSON = ROOT / "results" / "metrics_pareto_profiles.json"
OUT_PNG = ROOT / "results" / "pareto_profiles.png"

OBJ_COLS = ["stability_conservative_eV_atom", "gap_conservative_eV",
            "Q_theoretical_mAh_g", "voltage_conservative_V"]
OBJ_KEYS = ["stability", "conduction", "capacity", "voltage"]

PROFILES = {
    "balanced": [0.25, 0.25, 0.25, 0.25],
    "capacity": [0.15, 0.05, 0.65, 0.15],
    "safety": [0.55, 0.05, 0.25, 0.15],
    "energy": [0.20, 0.05, 0.45, 0.30],
    "power": [0.20, 0.50, 0.20, 0.10],
}


def _objectives(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        -df["stability_conservative_eV_atom"].to_numpy(dtype=float),
        -df["gap_conservative_eV"].to_numpy(dtype=float),
        df["Q_theoretical_mAh_g"].to_numpy(dtype=float),
        df["voltage_conservative_V"].to_numpy(dtype=float),
    ])


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="不同偏好下的最优候选子集")
    ap.add_argument("--ranking", default=str(RANKING))
    ap.add_argument("--profile", default=None, choices=sorted(PROFILES),
                    help="单个预设偏好")
    ap.add_argument("--all-profiles", action="store_true", help="一次跑全部预设偏好")
    ap.add_argument("--weights", default=None,
                    help="自定义四目标权重，逗号分隔（稳定性,导电性,容量,电压）")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    # 单偏好/自定义权重属于"部分运行"：不得覆盖全量 5 偏好的产物（v1.4.2 教训）
    partial = bool(args.profile or args.weights)
    tag = args.profile or "custom"
    out_csv, out_json, out_png = OUT_CSV, OUT_JSON, OUT_PNG
    if partial:
        out_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}_{tag}{OUT_CSV.suffix}")
        out_json = OUT_JSON.with_name(f"{OUT_JSON.stem}_{tag}{OUT_JSON.suffix}")
        out_png = OUT_PNG.with_name(f"{OUT_PNG.stem}_{tag}{OUT_PNG.suffix}")
        print(f"[部分运行] 输出改名: {out_json.name}")

    if not Path(args.ranking).exists():
        print(f"[ERROR] 找不到 {args.ranking}（先跑 screening_struct.py）")
        return 2
    df = pd.read_csv(args.ranking, encoding="utf-8-sig")
    for c in OBJ_COLS:
        if c not in df.columns:
            print(f"[ERROR] 排序表缺列 {c}")
            return 2

    Z = _objectives(df)
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)  # 标准化

    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")], dtype=float)
        if len(w) != 4:
            print("[ERROR] --weights 需要 4 个数")
            return 2
        if w.sum() <= 0:
            print("[ERROR] 权重和为 0")
            return 2
        profiles = {"custom": (w / w.sum()).tolist()}
    elif args.profile:
        profiles = {args.profile: PROFILES[args.profile]}
    else:
        profiles = PROFILES           # 默认给全部

    out = df[["name", "formula", "family", "risk_level", "score", "pareto_rank",
              "is_pareto", "voltage_risk"]].copy()
    summary = {}
    for name, w in profiles.items():
        s = Z @ np.asarray(w, dtype=float)
        out[f"score_{name}"] = np.round(s, 4)
        out[f"rank_{name}"] = pd.Series(s).rank(ascending=False).astype(int).to_numpy()
        order = out.sort_values(f"rank_{name}")
        top = order.head(args.top)
        summary[name] = {
            "weights": {k: round(float(x), 3) for k, x in zip(OBJ_KEYS, w)},
            "top": top[["name", "formula", "risk_level", "pareto_rank", "score"]]
                     .to_dict("records"),
            "n_pareto_in_top": int(top["is_pareto"].sum()),
        }
        print(f"\n=== 偏好「{name}」（权重 " +
              " / ".join(f"{k}={x:.2f}" for k, x in zip(OBJ_KEYS, w)) + "）===")
        print(top[["name", "formula", "risk_level", "pareto_rank",
                   f"score_{name}"]].to_string(index=False))

    out_json.parent.mkdir(exist_ok=True)
    payload = {
        "n_candidates": int(len(df)),
        "n_profiles": len(profiles),
        "objectives": OBJ_KEYS,
        "profiles": summary,
        "note": ("Pareto 前沿上有 32/40 候选（v1.18.0），因此单一权重排名是任意的；"
                 "本脚本把偏好作为输入，输出该偏好下的最优子集。"
                 "各偏好共同入选的候选可视为「稳健优选」。"),
    }
    # 稳健优选：在 ≥半数偏好下都进 Top-N
    rank_cols = [f"rank_{n}" for n in profiles]
    votes = (out[rank_cols] <= args.top).sum(axis=1)
    out["profile_votes"] = votes.to_numpy()
    robust = out[out["profile_votes"] >= max(2, len(profiles) // 2)].sort_values(
        "profile_votes", ascending=False)
    payload["robust_picks"] = robust[["name", "formula", "risk_level",
                                      "profile_votes"]].to_dict("records")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print(f"  稳健优选（在 ≥{max(2, len(profiles)//2)}/{len(profiles)} 个偏好下都进 Top {args.top}）")
    print("=" * 74)
    if len(robust):
        print(robust[["name", "formula", "risk_level", "profile_votes"]].to_string(index=False))
    else:
        print("  （无候选在多数偏好下都入选 —— 说明目标冲突极强）")

    if len(profiles) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        cand = out.sort_values(f"rank_{list(profiles)[0]}").head(12)
        mat = cand[[f"rank_{n}" for n in profiles]].to_numpy()
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(profiles)), list(profiles), fontsize=9)
        ax.set_yticks(range(len(cand)), [f"{n}" for n in cand["name"]], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7)
        ax.set_title("同一批候选在不同偏好下的排名\n（越绿=排名越靠前；分歧大说明偏好决定结论）",
                     fontsize=9)
        fig.colorbar(im, ax=ax, label="排名（1=最好）")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"偏好排名图 -> results/{out_png.name}")

    print(f"[SAVE] results/{out_csv.name} / {out_json.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())