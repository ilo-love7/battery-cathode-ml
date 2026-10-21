# -*- coding: utf-8 -*-
"""电池正极材料预测工具。

支持用训练好的模型束对新化学式进行预测：
- 带隙预测 (Egap, eV)
- 形成能预测 (Eform, eV/5 原子胞)
- 平均电压预测 (Voltage, V)

用法::

    from predictor import predict_material_properties
    df = predict_material_properties(["LiFePO4", "LiCoO2"])

命令行::

    python src/predictor.py --formulas "LiFePO4,LiCoO2,LiMn2O4"
    python src/predictor.py --top 10 --min-voltage 3.0 --max-voltage 5.0
    python src/predictor.py --formulas "LiFePO4" --json

⚠️ 特征列顺序：
    训练集 X 的列顺序由 ``bundle["feature_names"]`` 记录（带隙/形成能模型是
    「统计特征 → 元素分数」，电压模型是「元素分数 → 统计特征」）。预测时必须
    按记录的列名构造 DataFrame 再 ``predict``，让 sklearn 用列名对齐；
    绝不能 ``to_numpy()``——那样列错位不会报错，只会静默返回错误数值。
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features import ELEMENTS, featurize_one  # noqa: E402
from formula import parse_formula  # noqa: E402
from models import load_pickle  # noqa: E402

MODEL_FILES = {
    "gap": "models_gap.pkl",
    "ef": "models_ef.pkl",
    "voltage": "models_voltage.pkl",
    "capacity": "models_capacity.pkl",
}

# conformal 校准结果（conformal.py 产出）；存在时预测会附带 90% 区间
CONFORMAL_FILES = {
    "voltage": "conformal_voltage.json",
    "capacity": "conformal_capacity.json",
}


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_models(results_dir: Union[str, Path, None] = None) -> Dict[str, dict]:
    """加载 results/ 下所有已训练的模型束（缺失的自动跳过）。"""
    out_dir = Path(results_dir) if results_dir else ROOT / "results"
    models: Dict[str, dict] = {}
    for key, fname in MODEL_FILES.items():
        path = out_dir / fname
        if path.exists():
            models[key] = load_pickle(str(path))
    return models


@lru_cache(maxsize=8)
def _load_conformal(tag: str, results_dir: str | None = None) -> Optional[dict]:
    """读取 conformal 校准结果（带缓存；缺失返回 None）。"""
    path = (Path(results_dir) if results_dir else ROOT / "results") / CONFORMAL_FILES[tag]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _confidence(spread: float, q: float) -> str:
    """结合「两模型分歧」与「conformal 区间半宽」给出可信度分级。

    规则（写进文档，避免玄学）：
      spread ≤ 0.25q  -> High（局部一致且区间窄）
      spread ≤ 0.75q  -> Medium
      否则            -> Low（两模型分歧已接近整体区间宽度）
    """
    if not q or q <= 0:
        return "Unknown"
    if spread <= 0.25 * q:
        return "High"
    if spread <= 0.75 * q:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# 特征矩阵构造（唯一的预测入口，保证列顺序正确）
# --------------------------------------------------------------------------- #
def _feature_frame(formula: str, bundle: dict) -> Optional[pd.DataFrame]:
    """按 ``bundle['feature_names']`` 的列名与顺序构造单样本特征矩阵。

    返回 None 表示该化学式无法特征化（含未知元素/解析失败）。
    列名缺失时抛错——宁可失败也不静默错位。
    """
    names = bundle.get("feature_names") or []
    if not names:
        raise ValueError("模型束缺少 feature_names，无法保证特征列顺序；请用最新脚本重新训练")

    feats = featurize_one(formula, element_list=bundle.get("element_list"))
    if feats is None:
        return None

    missing = [n for n in names if n not in feats]
    if missing:
        raise ValueError(f"模型需要的特征列缺失: {missing[:5]} (共 {len(missing)} 个)")
    return pd.DataFrame([{n: feats[n] for n in names}], columns=names)


def _predict_ensemble(bundle: dict, formula: str) -> Optional[dict]:
    """返回 {rf, hgb, ens, spread}；无法特征化时返回 None。"""
    X = _feature_frame(formula, bundle)
    if X is None:
        return None
    rf = float(bundle["models"]["RF"].predict(X)[0])
    hgb = float(bundle["models"]["HGB"].predict(X)[0])
    return {"rf": rf, "hgb": hgb, "ens": 0.5 * (rf + hgb), "spread": abs(rf - hgb) / 2.0}


def _safe_predict(bundle: dict, formula: str):
    """容错版预测：返回 (pred, error)。模型束损坏时不打断批量预测。"""
    try:
        return _predict_ensemble(bundle, formula), None
    except (ValueError, KeyError, AttributeError) as exc:  # noqa: BLE001
        return None, f"模型束不可用: {exc}"


# --------------------------------------------------------------------------- #
# 单材料预测
# --------------------------------------------------------------------------- #
def predict_single(formula: str, models: Dict[str, dict]) -> Dict:
    """预测单个材料的性质，返回扁平字典。"""
    result: Dict = {"formula": formula}

    try:
        counts = parse_formula(formula)
    except Exception as exc:  # noqa: BLE001 - 用户输入不可信
        result["parse_error"] = str(exc)
        return result

    result["parsed"] = counts
    result["n_elements"] = len(counts)
    result["total_atoms"] = sum(counts.values())
    unknown = [el for el in counts if el not in ELEMENTS]
    if unknown:
        result["unknown_elements"] = unknown
        return result

    molar_mass = sum(counts[el] * ELEMENTS[el]["mass"] for el in counts)
    result["molar_mass_g_mol"] = round(molar_mass, 3)

    # ---- 带隙 ----
    if "gap" in models:
        pred, err = _safe_predict(models["gap"], formula)
        if err:
            result["gap_error"] = err
        elif pred is None:
            result["gap_error"] = "Feature extraction failed"
        else:
            result.update({
                "Egap_RF_eV": round(pred["rf"], 4),
                "Egap_HGB_eV": round(pred["hgb"], 4),
                "Egap_ensemble_eV": round(pred["ens"], 4),
                "Egap_spread": round(pred["spread"], 4),
            })
            if pred["ens"] > 3.0:
                result["conductivity"] = "Insulator (>3 eV)"
            elif pred["ens"] > 1.0:
                result["conductivity"] = "Semiconductor (1-3 eV)"
            else:
                result["conductivity"] = "Narrow-gap/Metal (<1 eV)"

    # ---- 形成能 ----
    if "ef" in models:
        pred, err = _safe_predict(models["ef"], formula)
        if err:
            result["ef_error"] = err
        elif pred is None:
            result["ef_error"] = "Feature extraction failed"
        else:
            atoms = float(sum(counts.values()))
            result.update({
                "Eform_RF_eV_fu": round(pred["rf"], 4),
                "Eform_HGB_eV_fu": round(pred["hgb"], 4),
                "Eform_ensemble_eV_fu": round(pred["ens"], 4),
                "Eform_spread": round(pred["spread"], 4),
                "Eform_per_atom_eV": round(pred["ens"] / atoms, 4) if atoms else None,
            })
            if pred["ens"] < -2.0:
                result["stability"] = "Highly stable (<-2 eV/fu)"
            elif pred["ens"] < 0:
                result["stability"] = "Stable (-2~0 eV/fu)"
            else:
                result["stability"] = "Unstable (>0 eV/fu)"

    # ---- 平均电压 ----
    if "voltage" in models:
        pred, err = _safe_predict(models["voltage"], formula)
        if err:
            result["voltage_error"] = err
        elif pred is None:
            result["voltage_error"] = "Feature extraction failed"
        else:
            result.update({
                "Voltage_RF_V": round(pred["rf"], 3),
                "Voltage_HGB_V": round(pred["hgb"], 3),
                "Voltage_ensemble_V": round(pred["ens"], 3),
                "Voltage_spread": round(pred["spread"], 3),
            })
            conf = _load_conformal("voltage")
            if conf:
                q = float(conf.get("q_halfwidth", 0.0))
                result.update({
                    "Voltage_pi90_low": round(pred["ens"] - q, 3),
                    "Voltage_pi90_high": round(pred["ens"] + q, 3),
                    "Voltage_confidence": _confidence(pred["spread"], q),
                })
            if pred["ens"] > 4.0:
                result["voltage_level"] = "High voltage (>4 V)"
            elif pred["ens"] > 3.5:
                result["voltage_level"] = "Medium voltage (3.5-4 V)"
            else:
                result["voltage_level"] = "Low voltage (<3.5 V)"

    # ---- 比容量（多目标：模型给出"可达容量"，无需事先知道 n_Li）----
    if "capacity" in models:
        pred, err = _safe_predict(models["capacity"], formula)
        if err:
            result["capacity_error"] = err
        elif pred is None:
            result["capacity_error"] = "Feature extraction failed"
        else:
            result.update({
                "Capacity_RF_mAh_g": round(pred["rf"], 1),
                "Capacity_HGB_mAh_g": round(pred["hgb"], 1),
                "Capacity_ensemble_mAh_g": round(pred["ens"], 1),
                "Capacity_spread": round(pred["spread"], 1),
            })
            conf = _load_conformal("capacity")
            if conf:
                q = float(conf.get("q_halfwidth", 0.0))
                result.update({
                    "Capacity_pi90_low": round(pred["ens"] - q, 1),
                    "Capacity_pi90_high": round(pred["ens"] + q, 1),
                    "Capacity_confidence": _confidence(pred["spread"], q),
                })

    return result


# --------------------------------------------------------------------------- #
# 批量预测
# --------------------------------------------------------------------------- #
APPROX_NOTE = (
    "[!] 成分级近似：电压模型训练于 Materials Project 插入电极真实数据"
    "（average_voltage，n≈2.7k）与比容量（capacity_grav）；带隙/形成能模型训练于"
    " matbench 公开数据（实验带隙、钙钛矿 DFT 形成能），对锂电池体系仍属外推——"
    "跨口径外部验证显示形成能去偏移 MAE≈0.74 eV/atom、带隙 ρ≈0.33，请以风险标记为准。"
)


def predict_material_properties(formulas: Union[str, List[str]],
                                include_details: bool = False,
                                models: Optional[Dict[str, dict]] = None,
                                verbose: bool = True) -> Union[pd.DataFrame, List[Dict]]:
    """批量预测材料性质。

    Args:
        formulas: 化学式或化学式列表。
        include_details: True 返回完整字典列表，False 返回精简 DataFrame。
        models: 预加载的模型束（避免重复 unpickle 数百 MB）。
        verbose: 是否打印提示信息。
    """
    if isinstance(formulas, str):
        formulas = [formulas]
    formulas = [f.strip() for f in formulas if f and f.strip()]
    if not formulas:
        raise ValueError("formulas 不能为空")

    if models is None:
        models = load_models()
    missing = [k for k in ("gap", "ef", "voltage", "capacity") if k not in models]
    if verbose:
        print(f"正在预测 {len(formulas)} 个材料 ...")
        if missing:
            print(f"警告：未找到模型 {missing}，对应性质将被跳过（先运行训练脚本）")
        if len(missing) or "voltage" in models:
            print(APPROX_NOTE)

    results = [predict_single(f, models) for f in formulas]
    if include_details:
        return results

    df = pd.DataFrame(results)
    main_cols = ["formula"]
    for col in ("Egap_ensemble_eV", "Eform_ensemble_eV_fu", "Voltage_ensemble_V",
                "Capacity_ensemble_mAh_g", "conductivity", "stability", "voltage_level"):
        if col in df.columns:
            main_cols.append(col)
    for col in ("Egap_spread", "Eform_spread", "Eform_per_atom_eV", "Voltage_spread",
                "Voltage_pi90_low", "Voltage_pi90_high", "Voltage_confidence",
                "Capacity_spread", "Capacity_pi90_low", "Capacity_pi90_high",
                "Capacity_confidence"):
        if col in df.columns:
            main_cols.append(col)
    return df[[c for c in main_cols if c in df.columns]]


# --------------------------------------------------------------------------- #
# 候选筛选
# --------------------------------------------------------------------------- #
W_STAB, W_COND, W_CAP = 0.40, 0.30, 0.30


def screen_candidates(candidates_file: Union[str, Path, None] = None,
                      n_top: int = 10,
                      min_voltage: float = 3.0,
                      max_voltage: float = 5.0,
                      models: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """对候选化学式做成分级筛选排序。

    综合评分 = 0.40 × 稳定性百分位 + 0.30 × 导电性百分位 + 0.30 × 理论容量百分位
    （与 screening.py 同一套权重；缺少 nLi_extract 列时容量项权重按比例归一）。

    注意：本函数只做「成分级优先级排序」，不替代 DFT / 实验验证。
    """
    if candidates_file is None:
        candidates_file = ROOT / "data" / "cathode_candidates.csv"
    path = Path(candidates_file)
    if not path.exists():
        raise FileNotFoundError(f"找不到候选文件：{path}")
    if models is None:
        models = load_models()

    from voltage import theoretical_capacity_mAh_g

    cand = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for _, r in cand.iterrows():
        pred = predict_single(str(r["formula"]), models)
        row = {
            "name": r.get("name", r["formula"]),
            "formula": r["formula"],
            "family": r.get("family", "Unknown"),
            "lit_voltage_V": r.get("avg_voltage_V_lit"),
        }
        row.update({k: pred.get(k) for k in
                    ("Egap_ensemble_eV", "Eform_ensemble_eV_fu", "Eform_per_atom_eV",
                     "Voltage_ensemble_V", "Voltage_confidence",
                     "Capacity_ensemble_mAh_g", "Capacity_confidence")})
        if "Voltage_pi90_high" in pred:
            row["Voltage_pi90_width"] = round(
                pred["Voltage_pi90_high"] - pred["Voltage_pi90_low"], 3)
        # 理论比容量（物理定义，n_Li 已知时优先使用）
        try:
            counts = parse_formula(str(r["formula"]))
            n_li = float(r["nLi_extract"]) if "nLi_extract" in r else 1.0
            row["Q_theoretical_mAh_g"] = round(theoretical_capacity_mAh_g(counts, n_li), 1)
        except (ValueError, KeyError):
            row["Q_theoretical_mAh_g"] = np.nan
        # 模型容量（任意化学式都可用；n_Li 缺失时作为评分依据）
        row["Q_model_mAh_g"] = pred.get("Capacity_ensemble_mAh_g")
        rows.append(row)

    df = pd.DataFrame(rows)

    if "Voltage_ensemble_V" in df.columns:
        mask = df["Voltage_ensemble_V"].between(min_voltage, max_voltage)
        df = df[mask].copy()

    if {"Eform_per_atom_eV", "Egap_ensemble_eV"} <= set(df.columns):
        df["pct_stability"] = df["Eform_per_atom_eV"].rank(pct=True, ascending=False) * 100
        df["pct_conduction"] = np.exp(-df["Egap_ensemble_eV"].clip(lower=0)).rank(pct=True) * 100
        cap_col = "Q_theoretical_mAh_g"
        if not df[cap_col].notna().any() and "Q_model_mAh_g" in df.columns \
                and df["Q_model_mAh_g"].notna().any():
            cap_col = "Q_model_mAh_g"      # 物理容量不可得时用模型容量
        df["Q_capacity_used_mAh_g"] = df[cap_col]
        if df[cap_col].notna().any():
            df["pct_capacity"] = df[cap_col].rank(pct=True) * 100
            w_stab, w_cond, w_cap = W_STAB, W_COND, W_CAP
        else:  # 无容量列时重新归一，避免总分只有 70
            df["pct_capacity"] = 0.0
            w_stab, w_cond, w_cap = W_STAB, W_COND, 0.0
            total = w_stab + w_cond
            w_stab, w_cond = w_stab / total, w_cond / total
        df["score_total"] = (w_stab * df["pct_stability"]
                             + w_cond * df["pct_conduction"]
                             + w_cap * df["pct_capacity"])
        df = df.sort_values("score_total", ascending=False).reset_index(drop=True)

    return df.head(n_top) if n_top else df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="电池正极材料性质预测 / 候选筛选")
    ap.add_argument("--formulas", default="", help="逗号分隔的化学式，如 'LiFePO4,LiCoO2'")
    ap.add_argument("--top", type=int, default=0, help="对候选库筛选并返回 Top N")
    ap.add_argument("--min-voltage", type=float, default=3.0)
    ap.add_argument("--max-voltage", type=float, default=5.0)
    ap.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本调用")
    ap.add_argument("--results-dir", default=None, help="模型目录，默认 <repo>/results")
    args = ap.parse_args(argv)

    models = load_models(args.results_dir)
    if not models:
        print("未找到任何模型，请先运行训练脚本（train_gap / train_ef / train_voltage）。")
        return 1

    formulas = [f for f in args.formulas.split(",") if f.strip()]
    if formulas:
        df = predict_material_properties(formulas, models=models, verbose=not args.json)
        print(df.to_json(orient="records", force_ascii=False, indent=2) if args.json
              else df.to_string(index=False))
        return 0

    if args.top:
        df = screen_candidates(n_top=args.top, min_voltage=args.min_voltage,
                               max_voltage=args.max_voltage, models=models)
        print(df.to_json(orient="records", force_ascii=False, indent=2) if args.json
              else df.to_string(index=False))
        return 0

    # 默认演示
    demo = ["LiFePO4", "LiCoO2", "LiMn2O4", "LiNi0.33Co0.33Mn0.34O2", "Li3V2(PO4)3"]
    print(predict_material_properties(demo, models=models).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
