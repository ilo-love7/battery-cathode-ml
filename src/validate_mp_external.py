# -*- coding: utf-8 -*-
"""外部验证：把带隙/形成能模型用在 Materials Project 的含 Li 材料上。

这是一个**分布外**测试集（模型训练于实验带隙与钙钛矿 DFT 形成能），
目的不是刷指标，而是回答两个问题：

1. 换到真实锂电相关材料上，模型还能不能排出正确的**相对顺序**（Spearman ρ）；
2. 绝对值偏差有多少来自**参考态/泛函口径差异**（常数偏移），多少来自模型本身。

⚠️ 口径说明（决定了怎么读结果）:
- 形成能：本项目模型目标是 matbench_perovskites 的 ``e_form``（5 原子单胞广延量，
  RPBE，O 参考态取水蒸气），MP 给的是 ``formation_energy_per_atom``（GGA/GGA+U，
  元素参考态）。所以**原始 MAE 里含一个系统性常数偏移**，因此额外报告：
  「去偏移 MAE（减去残差中位数）」与「Spearman ρ」。
- 带隙：本项目模型训练于**实验**带隙，MP 的 ``band_gap`` 是 DFT-PBE 值，
  金属被记为 0 eV，两者存在系统性差异；另外单列 ``band_gap > 0`` 子集。

用法::

    python src/prepare_mp_summary.py        # 先跑，产出 data/processed/mp_summary_Li.csv.gz
    python src/validate_mp_external.py

输出::

    results/validation_mp_external.json
    results/validation_mp_external_sample.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features import featurize_one  # noqa: E402
from models import load_pickle  # noqa: E402

SUMMARY_PATH = ROOT / "data" / "processed" / "mp_summary_Li.csv.gz"
# 形成能模型的训练口径：eV / 5 原子单胞 -> eV/atom 需除以 5
EF_CELL_ATOMS = 5.0


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    resid_median = float(np.median(err))
    mae = float(np.mean(np.abs(err)))
    corrected = float(np.mean(np.abs(err - resid_median)))
    return {
        "n": int(len(y_true)),
        "MAE": round(mae, 4),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "R2": round(float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)), 4),
        "spearman": round(float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")), 4),
        "bias_median": round(resid_median, 4),
        "MAE_offset_corrected": round(corrected, 4),
    }


def _predict_block(df: pd.DataFrame, bundle: dict, per_cell: bool = False):
    """用模型束预测；返回 (pred, unseen_element_ratio)。"""
    names = bundle.get("feature_names")
    els = bundle.get("element_list")
    if not names:
        raise SystemExit("模型束缺少 feature_names，请重新训练")

    unseen = set()
    known = set(els or [])
    feats, kept = [], []
    from formula import parse_formula
    for i, f in enumerate(df["formula"].astype(str)):
        try:
            counts = parse_formula(f)
        except ValueError:
            continue
        hidden = set(counts) - known
        if hidden:
            unseen |= hidden
        row = featurize_one(f, element_list=els)
        if row is None:
            continue
        feats.append({n: row[n] for n in names})
        kept.append(i)

    X = pd.DataFrame(feats, columns=names)
    rf = bundle["models"]["RF"].predict(X)
    hgb = bundle["models"]["HGB"].predict(X)
    pred = 0.5 * (rf + hgb)
    if per_cell:
        pred = pred / EF_CELL_ATOMS
    return pd.DataFrame({"idx": kept, "pred": pred}), unseen


def main() -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    if not SUMMARY_PATH.exists():
        print(f"[ERROR] 找不到 {SUMMARY_PATH}")
        print("        请先运行: python src/prepare_mp_summary.py")
        return 2

    df = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    print(f"外部验证集: {len(df)} 条含 Li 材料（按成分去重后）")

    missing = [p.name for p in (ROOT / "results" / "models_gap.pkl",
                                ROOT / "results" / "models_ef.pkl") if not p.exists()]
    if missing:
        print(f"[ERROR] 缺少带隙/形成能模型: {missing}")
        print("        这两个模型训练于 matbench 公开数据，请先执行：")
        print("          python src/download_data.py && python src/prepare_data.py")
        print("          python src/train_gap.py && python src/train_ef.py")
        print("        （数据约 4 MB，训练合计约 1.5 分钟；离线环境手动下载 json.gz 放 data/raw/）")
        return 2

    out = {
        "dataset": {
            "source": "Materials Project summary (elements=['Li'])",
            "n_rows": int(len(df)),
            "n_unique_formula": int(df["formula_canon"].nunique()),
        }
    }
    sample_cols = ["formula", "formula_canon"]

    # ---------- 带隙 ----------
    gap_bundle = load_pickle(ROOT / "results" / "models_gap.pkl")
    gap_pred, gap_unseen = _predict_block(df, gap_bundle)
    g = df.iloc[gap_pred["idx"].to_numpy()].reset_index(drop=True)
    g["pred_gap_eV"] = gap_pred["pred"].to_numpy()
    g["mp_gap_eV"] = g["band_gap"].astype(float)

    out["gap"] = {
        "note": "模型训练于实验带隙，MP 为 DFT-PBE；金属记 0 eV",
        "all": _metrics(g["mp_gap_eV"], g["pred_gap_eV"]),
        "semiconductor_only": _metrics(
            g.loc[g["mp_gap_eV"] > 0, "mp_gap_eV"],
            g.loc[g["mp_gap_eV"] > 0, "pred_gap_eV"]),
        "mp_mean_baseline_MAD": round(float((g["mp_gap_eV"] - g["mp_gap_eV"].mean()).abs().mean()), 4),
        "unseen_elements": sorted(gap_unseen),
    }
    print("\n[带隙] 模型（实验训练）vs MP DFT 带隙")
    for k in ("all", "semiconductor_only"):
        m = out["gap"][k]
        print(f"  {k:20s} n={m['n']:6d}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
              f"R2={m['R2']:.3f}  ρ={m['spearman']:.3f}  去偏移MAE={m['MAE_offset_corrected']:.3f}")
    print(f"  元素宇宙外的元素: {len(out['gap']['unseen_elements'])} 种 "
          f"{out['gap']['unseen_elements'][:8]}")

    # ---------- 形成能 ----------
    ef_bundle = load_pickle(ROOT / "results" / "models_ef.pkl")
    ef_pred, ef_unseen = _predict_block(df, ef_bundle, per_cell=True)
    e = df.iloc[ef_pred["idx"].to_numpy()].reset_index(drop=True)
    e["pred_ef_eV_atom"] = ef_pred["pred"].to_numpy()
    e["mp_ef_eV_atom"] = e["formation_energy_per_atom"].astype(float)

    out["ef"] = {
        "note": "模型目标为 5 原子单胞 e_form(RPBE)，已除以 5 换成 eV/atom；MP 为 GGA/GGA+U 元素参考态",
        "all": _metrics(e["mp_ef_eV_atom"], e["pred_ef_eV_atom"]),
        "mp_mean_baseline_MAD": round(float((e["mp_ef_eV_atom"] - e["mp_ef_eV_atom"].mean()).abs().mean()), 4),
        "unseen_elements": sorted(ef_unseen),
    }
    m = out["ef"]["all"]
    print("\n[形成能] 模型（钙钛矿训练）vs MP 含 Li 材料")
    print(f"  n={m['n']:6d}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}  "
          f"ρ={m['spearman']:.3f}")
    print(f"  系统性偏移(残差中位数)={m['bias_median']:+.3f} eV/atom -> "
          f"去偏移 MAE={m['MAE_offset_corrected']:.3f} eV/atom")
    print(f"  MP 均值基线 MAD={out['ef']['mp_mean_baseline_MAD']:.3f} eV/atom")

    # ---------- 落盘 ----------
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "validation_mp_external.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    merged = (df.assign(pred_gap_eV=np.nan, pred_ef_eV_atom=np.nan)
                .iloc[gap_pred["idx"].to_numpy()].reset_index(drop=True))
    merged = merged.merge(
        g[["formula", "pred_gap_eV"]].reset_index(drop=True), on="formula", how="left")
    merged = merged.merge(
        e[["formula", "pred_ef_eV_atom"]].reset_index(drop=True), on="formula", how="left")
    merged.head(500).to_csv(out_dir / "validation_mp_external_sample.csv",
                            index=False, encoding="utf-8-sig")

    print("\n[SAVE] results/validation_mp_external.json")
    print("[SAVE] results/validation_mp_external_sample.csv（前 500 条）")
    print("\n读法提示：绝对值 MAE 受参考态/泛函口径影响，"
          "要看的是「去偏移 MAE」与「Spearman ρ」是否显著优于随机。")
    return 0


if __name__ == "__main__":
    sys.exit(main())