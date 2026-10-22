# -*- coding: utf-8 -*-
"""局部自适应 conformal：给每个候选中**自己的**电压区间宽度（v1.15.0）。

### 为什么要做

`src/conformal.py` 用的是 split conformal：全数据一个常数分位数 q（当前 90% 区间 ±1.00 V）。
它保证**边际覆盖率**，但对**单个候选**没有区分度 —— 区间宽的候选与窄的候选拿到同一个宽度，
于是"区间宽度"无法用于排序（这正是 v1.14.0 之前 interval 只输出、不参与评分的原因之一）。

本脚本改为**局部自适应**（locally-adaptive / Mondrian 思路）：

1. 用电压模型在训练集上的 **OOF 绝对残差**作为非一致性分数；
2. 对每个候选，在**标准化特征空间**里找 k 个最近邻（k 默认 40，距离加权）；
3. 在这 k 个邻居的残差上取有限样本修正的 90% 分位数 → **该候选自己的区间半宽**；
4. 落盘每个候选的半宽与全局 q 的对比。

输出::

    results/local_conformal_voltage.csv   （每个候选的半宽）
    results/metrics_local_conformal.json  （分布统计 + 与全局 q 的对比）

用法::

    python src/local_conformal.py
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

from conformal import conformal_quantile  # noqa: E402
from features import build_universe, featurize_dataframe  # noqa: E402

VOLTAGE_CSV = ROOT / "data" / "processed" / "mp_voltage.csv"
OOF_CSV = ROOT / "results" / "oof_voltage.csv"
CANDIDATES = ROOT / "data" / "cathode_candidates.csv"
CONFORMAL_JSON = ROOT / "results" / "conformal_voltage.json"
OUT_CSV = ROOT / "results" / "local_conformal_voltage.csv"
OUT_JSON = ROOT / "results" / "metrics_local_conformal.json"
ALPHA = 0.10


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="局部自适应 conformal（每个候选自己的区间宽度）")
    ap.add_argument("--voltage", default=str(VOLTAGE_CSV))
    ap.add_argument("--oof", default=str(OOF_CSV))
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--k", type=int, default=40, help="近邻数")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--validate-folds", type=int, default=5,
                    help="K 折验证局部覆盖率（0=跳过）：邻居只从其他折取，避免用自身残差")
    args = ap.parse_args(argv)

    for p in (Path(args.voltage), Path(args.oof)):
        if not p.exists():
            print(f"[ERROR] 找不到 {p}（先跑 train_voltage.py 生成 OOF）")
            return 2
    train = pd.read_csv(args.voltage, encoding="utf-8-sig")
    oof = pd.read_csv(args.oof, encoding="utf-8-sig")
    cand = pd.read_csv(args.candidates, encoding="utf-8-sig")
    if len(oof) != len(train):
        print(f"[ERROR] OOF 行数({len(oof)}) != 电压数据行数({len(train)})")
        return 2

    train_formulas = train["formula"].astype(str).tolist()
    cand_formulas = cand["formula"].astype(str).tolist()
    universe = build_universe(train_formulas + cand_formulas)
    Xtr, _ = featurize_dataframe(train_formulas, element_list=universe, verbose=False)
    Xca, _ = featurize_dataframe(cand_formulas, element_list=universe, verbose=False)
    print(f"特征: {Xtr.shape[1]} 维 | 训练 {len(Xtr)} 条 | 候选 {len(Xca)} 条")

    # 用训练集的中位数/四分位间距标准化（对异常值稳健）
    med = Xtr.median()
    iqr = (Xtr.quantile(0.75) - Xtr.quantile(0.25)).replace(0, 1.0)
    Ztr = ((Xtr - med) / iqr).fillna(0.0).to_numpy(dtype=float)
    Zca = ((Xca - med) / iqr).fillna(0.0).to_numpy(dtype=float)

    resid = np.abs(oof["pred_ens"].to_numpy(dtype=float)
                   - oof["y_true"].to_numpy(dtype=float))
    k = min(args.k, len(Ztr))
    rows = []
    for i, f in enumerate(cand_formulas):
        d = np.linalg.norm(Ztr - Zca[i], axis=1)
        idx = np.argpartition(d, k - 1)[:k]
        # 距离加权：权重 ~ 1/(d+eps)，避免"近邻里恰好都是远端样本"时的误判
        w = 1.0 / (d[idx] + 1e-6)
        scores_all = resid[idx]
        q_plain = conformal_quantile(scores_all, alpha=args.alpha)
        # 加权分位数：**先按分数升序**，再累积权重到 1-alpha 处插值
        # （早期版本按权重排序后插值 -> 覆盖率只有 48%，已由 K 折验证抓出）
        order = np.argsort(scores_all)
        ss, sw = scores_all[order], w[order]
        cum = np.cumsum(sw) / sw.sum()
        q_local = float(np.interp(1 - args.alpha, cum, ss))
        rows.append({
            "formula": f,
            "n_neighbors": int(k),
            "mean_neighbor_distance": round(float(d[idx].mean()), 4),
            "local_q90_plain_V": round(float(q_plain), 4),
            "local_q90_weighted_V": round(q_local, 4),
            "neighbor_resid_median_V": round(float(np.median(scores_all)), 4),
        })

    # ---- K 折验证：局部覆盖率是否仍达名义 90% ----
    validation = None
    if args.validate_folds and args.validate_folds > 1:
        n = len(Ztr)
        rng = np.random.default_rng(42)
        fold = rng.integers(0, args.validate_folds, size=n)
        covered, widths = [], []
        for f in range(args.validate_folds):
            pool = np.where(fold != f)[0]
            test = np.where(fold == f)[0]
            for i in test:
                d = np.linalg.norm(Ztr[pool] - Ztr[i], axis=1)
                kk = min(k, len(pool))
                idx = pool[np.argpartition(d, kk - 1)[:kk]]
                w = 1.0 / (np.linalg.norm(Ztr[idx] - Ztr[i], axis=1) + 1e-6)
                sc = resid[idx]
                order = np.argsort(sc)                 # 按分数升序
                cum = np.cumsum(w[order]) / w[order].sum()
                q = float(np.interp(1 - args.alpha, cum, sc[order]))
                widths.append(q)
                covered.append(resid[i] <= q)
        sweep = [{"k": k, "coverage": round(float(np.mean(covered)), 4),
                  "mean_halfwidth_V": round(float(np.mean(widths)), 4)}]
        # k 扫描：更宽的邻域降低估计方差，但会损失区分度
        for k_alt in (80, 160):
            cov_a, w_a = [], []
            kk_a = min(k_alt, n - 1)
            for f in range(args.validate_folds):
                pool = np.where(fold != f)[0]
                test = np.where(fold == f)[0]
                for i in test:
                    d = np.linalg.norm(Ztr[pool] - Ztr[i], axis=1)
                    idx = pool[np.argpartition(d, kk_a - 1)[:kk_a]]
                    w = 1.0 / (np.linalg.norm(Ztr[idx] - Ztr[i], axis=1) + 1e-6)
                    sc = resid[idx]
                    order = np.argsort(sc)
                    cum = np.cumsum(w[order]) / w[order].sum()
                    q = float(np.interp(1 - args.alpha, cum, sc[order]))
                    w_a.append(q)
                    cov_a.append(resid[i] <= q)
            sweep.append({"k": k_alt, "coverage": round(float(np.mean(cov_a)), 4),
                          "mean_halfwidth_V": round(float(np.mean(w_a)), 4)})
        # 达标 = 覆盖率不低于名义 - 2%；在达标者里取最小 k（保留最多区分度）
        target = 1 - args.alpha - 0.02
        ok = [s for s in sweep if s["coverage"] >= target]
        k_best = min(ok, key=lambda s: s["k"])["k"] if ok else max(sweep, key=lambda s: s["k"])["k"]
        validation = {
            "folds": int(args.validate_folds),
            "n_test": int(len(covered)),
            "k_sweep": sweep,
            "k_selected": int(k_best),
            "empirical_coverage": next(s["coverage"] for s in sweep if s["k"] == k),
            "nominal_coverage": round(1 - args.alpha, 4),
            "mean_halfwidth_V": round(float(np.mean(widths)), 4),
            "note": ("邻居只从其他折取（不含自身残差）。k 扫描用于在「覆盖率」与"
                     "「区分度」之间选折中：达标者里取最小 k。"),
        }
        if k_best != k:
            print(f"[验证] k 扫描选定 k={k_best}（覆盖率更接近名义值）")
        # 校准：把局部宽度整体放大到全局 q 的量级（保留相对区分度，补齐绝对校准）
        gq = None
        if Path(CONFORMAL_JSON).exists():
            with open(CONFORMAL_JSON, encoding="utf-8") as fh:
                gq = json.load(fh).get("q_halfwidth")
        if gq:
            scale0 = float(gq) / max(float(np.mean(widths)), 1e-9)
            cov_s = []
            for f in range(args.validate_folds):
                pool = np.where(fold != f)[0]
                for i in np.where(fold == f)[0]:
                    d = np.linalg.norm(Ztr[pool] - Ztr[i], axis=1)
                    kk = min(k, len(pool))
                    idx = pool[np.argpartition(d, kk - 1)[:kk]]
                    w = 1.0 / (np.linalg.norm(Ztr[idx] - Ztr[i], axis=1) + 1e-6)
                    sc = resid[idx]
                    order = np.argsort(sc)
                    cum = np.cumsum(w[order]) / w[order].sum()
                    q = float(np.interp(1 - args.alpha, cum, sc[order])) * scale0
                    cov_s.append(resid[i] <= q)
            validation["empirical_coverage_calibrated"] = round(float(np.mean(cov_s)), 4)
            validation["calibration_scale"] = round(scale0, 3)
            print(f"[验证] 校准后覆盖率 = {np.mean(cov_s)*100:.2f}%（放大 {scale0:.2f}×）")

        print(f"\n[验证] {args.validate_folds} 折局部覆盖率 = "
              f"{validation['empirical_coverage']*100:.2f}% (名义 {validation['nominal_coverage']*100:.0f}%) "
              f"| 平均半宽 {validation['mean_halfwidth_V']:.4f} V")

    global_q = None
    if Path(CONFORMAL_JSON).exists():
        with open(CONFORMAL_JSON, encoding="utf-8") as f:
            _g = json.load(f)
        global_q = _g.get("q_halfwidth", _g.get("q"))     # split conformal 的常数半宽

    # ---- 条件覆盖率：按体系族分层（只报边际覆盖率会掩盖分层失效） ----
    def family(f: str) -> str:
        if "S" in f and not f.startswith("Li2FeSi"):
            return "硫化物" if "P" not in f else "磷酸盐"
        if "P" in f:
            return "磷酸盐/聚阴离子"
        if "Si" in f:
            return "硅酸盐"
        if "F" in f:
            return "含氟"
        return "氧化物"

    fam_train = np.array([family(f) for f in train_formulas])
    coverage_by_family = {}
    if args.validate_folds and args.validate_folds > 1 and validation is not None:
        scale = validation.get("calibration_scale", 1.0)
        for fam in sorted(set(fam_train)):
            mask_f = fam_train == fam
            if mask_f.sum() < 30:
                continue
            cov_g, cov_l, cov_c, w_l = [], [], [], []
            for f in range(args.validate_folds):
                pool = np.where(fold != f)[0]
                test = np.where((fold == f) & mask_f)[0]
                for i in test:
                    d_all = np.linalg.norm(Ztr[pool] - Ztr[i], axis=1)
                    kk = min(k, len(pool))
                    idx = pool[np.argpartition(d_all, kk - 1)[:kk]]
                    w = 1.0 / (np.linalg.norm(Ztr[idx] - Ztr[i], axis=1) + 1e-6)
                    sc = resid[idx]
                    order = np.argsort(sc)
                    cum = np.cumsum(w[order]) / w[order].sum()
                    q_loc = float(np.interp(1 - args.alpha, cum, sc[order]))
                    q_glob = float(global_q) if global_q else q_loc
                    w_l.append(q_loc)
                    cov_g.append(resid[i] <= q_glob)
                    cov_l.append(resid[i] <= q_loc)
                    cov_c.append(resid[i] <= q_loc * scale)
            coverage_by_family[fam] = {
                "n_train": int(mask_f.sum()),
                "n_test": int(len(cov_g)),
                "global_q_coverage": round(float(np.mean(cov_g)), 4),
                "local_raw_coverage": round(float(np.mean(cov_l)), 4),
                "local_calibrated_coverage": round(float(np.mean(cov_c)), 4),
                "mean_local_halfwidth_V": round(float(np.mean(w_l)), 4),
            }
        if coverage_by_family:
            print("\n[条件覆盖率] 按体系族：")
            for fam, v in coverage_by_family.items():
                print(f"  {fam:14s} n={v['n_test']:4d} | 全局 {v['global_q_coverage']*100:5.1f}% "
                      f"| 局部原始 {v['local_raw_coverage']*100:5.1f}% "
                      f"| 局部校准 {v['local_calibrated_coverage']*100:5.1f}%")

    out = pd.DataFrame(rows)

    widths = out["local_q90_weighted_V"].to_numpy(dtype=float)
    metrics = {
        "alpha": args.alpha,
        "nominal_coverage": 1 - args.alpha,
        "k_neighbors": int(k),
        "n_candidates": int(len(out)),
        "global_q_V": global_q,
        "local_halfwidth_V": {
            "min": round(float(widths.min()), 4),
            "median": round(float(np.median(widths)), 4),
            "mean": round(float(widths.mean()), 4),
            "max": round(float(widths.max()), 4),
            "std": round(float(widths.std()), 4),
            "max_over_min": round(float(widths.max() / max(widths.min(), 1e-9)), 2),
        },
        "validation": validation,
        "coverage_by_family": coverage_by_family,
        "recommended_halfwidth_V": None,
        "note": ("split conformal 的宽度是全局常数（无法区分候选）；"
                 "局部自适应按特征近邻给出每个候选自己的半宽，"
                 "min/max 之比即区分度，直接决定区间能否参与排序。"),
    }
    # 覆盖率不足时给"放大到名义覆盖率"的建议宽度（保持排序信息、校准绝对宽度）
    if validation is not None and global_q:
        cov = validation["empirical_coverage"]
        if cov < (1 - args.alpha) - 0.02:
            scale = float(global_q) / max(validation["mean_halfwidth_V"], 1e-9)
            metrics["recommended_halfwidth_V"] = round(float(global_q), 4)
            metrics["recommended_scaling"] = round(scale, 3)
            metrics["recommendation_reason"] = (
                f"局部经验覆盖率 {cov:.3f} 低于名义 {1-args.alpha:.2f}，"
                f"建议把局部半宽按 {scale:.2f}× 放大到全局 q 量级，以兼顾校准与区分度")
            out["local_q90_calibrated_V"] = (out["local_q90_weighted_V"] * scale).round(4)

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print("  局部自适应 conformal（电压 90% 区间半宽，V）")
    print("=" * 74)
    print(f"  全局 split conformal q = {global_q}")
    print(f"  局部: min={metrics['local_halfwidth_V']['min']:.4f}  "
          f"中位数={metrics['local_halfwidth_V']['median']:.4f}  "
          f"max={metrics['local_halfwidth_V']['max']:.4f}  "
          f"(max/min={metrics['local_halfwidth_V']['max_over_min']:.1f}×)")
    print(f"  最窄 3 个: {', '.join(out.nsmallest(3, 'local_q90_weighted_V')['formula'])}")
    print(f"  最宽 3 个: {', '.join(out.nlargest(3, 'local_q90_weighted_V')['formula'])}")
    print(f"[SAVE] results/{Path(args.out).name} / {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())