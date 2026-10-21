# -*- coding: utf-8 -*-
"""结构模型驱动的候选筛选（v1.9.0）。

与原 ``screening.py``（纯成分模型）的区别：

| 维度 | screening.py（旧） | screening_struct.py（本脚本） |
|---|---|---|
| 稳定性 | 成分形成能模型（matbench 钙钛矿，跨域外推） | **CGCNN（MP 含 Li 域，每原子形成能）**，无结构时回退成分模型 |
| 电压 | 不参与评分 | **双塔 CGCNN**（有电极对时）/ 成分电压模型（回退） |
| 导电性 | 成分带隙模型 | **CGCNN 带隙（结构，v1.10.0）**，无结构时回退成分模型 |
| 风险标记 | 域外元素 / 分歧 / 带隙外推 | 追加 **结构缺失**、**结构 vs 成分分歧大**、**电压来自成分回退** |

流程：

1. 读 ``data/cathode_candidates.csv``（40 条）；
2. 结构：``data/raw/mp_structures_candidates.json.gz``（``fetch_mp_structures.py --mode formulas``）
   → CGCNN 预测每原子形成能；
3. 电极对：``data/raw/mp_voltage_pairs_candidates.json.gz``（``--mode pairs``）
   → 双塔 CGCNN 预测电压；缺失时用成分电压模型；
4. 理论比容量 Q = n_Li·F/3.6/M；
5. 评分 = 0.40×稳定性百分位 + 0.30×导电性百分位 + 0.30×容量百分位；
6. 落盘排序表、排名图与指标 JSON。

用法（推理需要 torch 环境）::

    python src/fetch_mp_structures.py --mode formulas --formulas "<40 条候选化学式>" \
        --out data/raw/mp_structures_candidates.json.gz
    python src/fetch_mp_structures.py --mode pairs --formulas "<同上>" \
        --out data/raw/mp_voltage_pairs_candidates.json.gz
    D:\\项目\\venvs\\bcm-cgcnn\\Scripts\\python.exe src/screening_struct.py

输出::

    results/cathode_ranking_struct.csv, results/cathode_ranking_struct.png,
    results/metrics_screening_struct.json
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
from features import ELEMENTS  # noqa: E402
from formula import parse_formula  # noqa: E402
from voltage import theoretical_capacity_mAh_g  # noqa: E402

CANDIDATES = ROOT / "data" / "cathode_candidates.csv"
STRUCTURES = ROOT / "data" / "raw" / "mp_structures_candidates.json.gz"
PAIRS = ROOT / "data" / "raw" / "mp_voltage_pairs_candidates.json.gz"
CGCNN_EF = ROOT / "results" / "cgcnn_mp_8k.pt"
CGCNN_GAP = ROOT / "results" / "cgcnn_gap_8k.pt"
GAP_DIFF_THRESHOLD = 1.5            # eV，结构带隙(DFT-PBE) 与 成分带隙(实验训练) 的系统差异阈值
CGCNN_V = ROOT / "results" / "cgcnn_voltage.pt"
OUT = ROOT / "results"
W_STAB, W_COND, W_CAP = 0.40, 0.30, 0.30
# 结构与成分两套形成能口径不同（MP 每原子 vs matbench 5 原子胞，正负号都不同），
# 因此**不能比绝对值**，只能用「排名偏离」这类尺度无关的量。
RANK_SHIFT_THRESHOLD = 15         # 同一候选在两个来源里的排名差超过该值则标记


def _cgcnn_e_form(model_path: Path, structures: dict) -> dict:
    """用 CGCNN 预测每原子形成能：{query_formula: (e_form_per_atom, material_id)}。"""
    import torch
    from cgcnn_model import batch_tensors, build_model

    if not model_path.exists():
        print(f"[WARN] 缺少 {model_path.name}，跳过结构稳定性预测")
        return {}
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("atom_fea_len", 64), ckpt.get("n_conv", 3),
                        ckpt.get("h_fea_len", 128), ckpt.get("n_h", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    mean, std = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))

    graphs, keys = [], []
    for r in structures:
        g = build_graph(r.get("structure"))
        if g is None:
            continue
        graphs.append(g)
        keys.append((r.get("query_formula") or r.get("formula"), r.get("material_id")))
    out = {}
    with torch.no_grad():
        for s in range(0, len(graphs), 64):
            chunk = list(range(s, min(s + 64, len(graphs))))
            (node, efea, src, dst, bvec) = batch_tensors(graphs, chunk, device)
            pred = ((model(node, efea, src, dst, bvec) * std + mean)
                    .detach().cpu().numpy())
            for (f, mid), p in zip([keys[i] for i in chunk], pred):
                out[f] = (float(p), mid)
    return out


def _cgcnn_scalar(model_path: Path, structures: dict, log1p: bool = False) -> dict:
    """通用：用 CGCNN 对结构预测单个标量（默认按 log1p 训练则反变换并裁剪到 >=0）。

    返回 {query_formula: value}。
    """
    import torch
    from cgcnn_model import batch_tensors, build_model

    if not model_path.exists():
        print(f"[WARN] 缺少 {model_path.name}，跳过该结构标量预测")
        return {}
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("atom_fea_len", 64), ckpt.get("n_conv", 3),
                        ckpt.get("h_fea_len", 128), ckpt.get("n_h", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    mean, std = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    use_log1p = log1p or ckpt.get("target_transform") == "log1p"

    graphs, keys = [], []
    for f, st in structures.items():
        g = build_graph(st)
        if g is None:
            continue
        graphs.append(g)
        keys.append(f)
    out = {}
    with torch.no_grad():
        for s in range(0, len(graphs), 64):
            chunk = list(range(s, min(s + 64, len(graphs))))
            (node, efea, src, dst, bvec) = batch_tensors(graphs, chunk, device)
            pred = (model(node, efea, src, dst, bvec) * std + mean).detach().cpu().numpy()
            if use_log1p:
                pred = np.clip(np.expm1(pred), 0.0, None)
            for k, p in zip([keys[i] for i in chunk], pred):
                out[k] = float(p)
    return out


def _cgcnn_voltage(model_path: Path, pairs: dict) -> dict:
    """用双塔 CGCNN 预测电压：{query_formula: (V_pred, n_li, V_true)}。"""
    import torch
    from train_cgcnn_voltage import build_pair_net

    if not model_path.exists():
        print(f"[WARN] 缺少 {model_path.name}，跳过结构电压预测")
        return {}
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pair_net(None, int(ckpt.get("atom_fea_len", 64)), int(ckpt.get("n_conv", 3)),
                           int(ckpt.get("h_fea_len", 128)), n_scalar=3, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))

    items = []
    for f, r in pairs.items():
        gc = build_graph(r.get("structure_charge"))
        gl = build_graph(r.get("structure_discharge"))
        if gc is None or gl is None:
            continue
        items.append((f, r, gc, gl))
    out = {}
    with torch.no_grad():
        for s in range(0, len(items), 32):
            chunk = items[s:s + 32]
            from cgcnn_model import batch_tensors
            a = batch_tensors([c[2] for c in chunk], list(range(len(chunk))), device)
            b = batch_tensors([c[3] for c in chunk], list(range(len(chunk))), device)
            n_f = np.array([int(c[2]["n_atoms"]) for c in chunk], dtype=np.float32)
            n_l = np.array([int(c[3]["n_atoms"]) for c in chunk], dtype=np.float32)
            sc = torch.as_tensor(np.stack([np.log1p(np.maximum(n_l - n_f, 1)),
                                           n_f / 50.0, n_l / 50.0], axis=1), device=device)
            pred = (model(a, b, sc) * std + mean).detach().cpu().numpy()
            for (f, r, gc, gl), p in zip(chunk, pred):
                out[f] = (float(p), int(max(gl["n_atoms"] - gc["n_atoms"], 1)),
                          r.get("average_voltage_true_V"))
    return out


def _composition_predictions(formulas: list) -> pd.DataFrame:
    """成分模型的 gap / ef-per-atom / voltage（用于回退与分歧对照）。"""
    import pickle
    from predictor import _feature_frame

    res = pd.DataFrame({"formula": formulas})
    bundles = {}
    for key, name in (("gap", "models_gap.pkl"), ("ef", "models_ef.pkl"),
                      ("voltage", "models_voltage.pkl")):
        p = OUT / name
        if p.exists():
            with open(p, "rb") as fh:
                bundles[key] = pickle.load(fh)

    gap, ef_atom, volt = [], [], []
    for f_ in formulas:
        g = e = v = np.nan
        for key, sink in (("gap", "g"), ("ef", "e"), ("voltage", "v")):
            if key not in bundles:
                continue
            try:
                X = _feature_frame(f_, bundles[key])
                pred = 0.5 * (float(bundles[key]["models"]["RF"].predict(X)[0])
                              + float(bundles[key]["models"]["HGB"].predict(X)[0]))
            except Exception:  # noqa: BLE001
                continue
            if sink == "g":
                g = pred
            elif sink == "e":
                try:
                    counts = parse_formula(f_)
                    atoms = float(sum(counts.values()))
                    e = pred / atoms        # eV/5原子胞 -> eV/atom（模型目标为 5 原子胞）
                except ValueError:
                    e = np.nan
            else:
                v = pred
        gap.append(g)
        ef_atom.append(e)
        volt.append(v)
    res["comp_gap_eV"] = gap
    res["comp_ef_eV_atom"] = ef_atom
    res["comp_voltage_V"] = volt
    return res


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="结构模型驱动的候选筛选")
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--structures", default=str(STRUCTURES))
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--tag", default="struct")
    ap.add_argument("--no-cgcnn", action="store_true", help="只用成分模型（对照）")
    ap.add_argument("--prefer-struct-gap", action="store_true",
                    help="用 CGCNN 结构带隙参与评分（默认用实验口径成分带隙，见 metrics.note）")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README「GPU 结构模型（CGCNN）」）")
        return 2

    cand = pd.read_csv(args.candidates, encoding="utf-8-sig")
    formulas = cand["formula"].astype(str).tolist()
    print(f"候选: {len(cand)} 条")

    # ---- 结构：CGCNN 形成能 ----
    cg_ef = {}
    if not args.no_cgcnn and Path(args.structures).exists():
        with gzip.open(args.structures, "rt", encoding="utf-8") as f:
            struct_rows = json.load(f)
        cg_ef = _cgcnn_e_form(Path(CGCNN_EF), struct_rows)
        print(f"结构命中（CGCNN 稳定性）: {len(cg_ef)}/{len(formulas)}")
    else:
        print("[INFO] 无结构文件或已禁用 CGCNN → 全部回退成分模型")

    # ---- 结构带隙：CGCNN ----
    cg_gap = {}
    if not args.no_cgcnn and Path(args.structures).exists():
        struct_map = {r["query_formula"]: r.get("structure") for r in struct_rows
                      if r.get("structure")}
        cg_gap = _cgcnn_scalar(Path(CGCNN_GAP), struct_map, log1p=True)
        print(f"结构带隙命中（CGCNN）: {len(cg_gap)}/{len(formulas)}")

    # ---- 电极对：双塔 CGCNN 电压 ----
    cg_v = {}
    if not args.no_cgcnn and Path(args.pairs).exists():
        with gzip.open(args.pairs, "rt", encoding="utf-8") as f:
            pair_rows = json.load(f)
        cg_v = _cgcnn_voltage(Path(CGCNN_V), {r["query_formula"]: r for r in pair_rows})
        print(f"电极对命中（结构电压）: {len(cg_v)}/{len(formulas)}")
    else:
        print("[INFO] 无电极对文件或已禁用双塔模型 → 电压全部来自成分模型")

    comp = _composition_predictions(formulas)
    df = cand.merge(comp, on="formula", how="left")

    # ---- 组装结构列 ----
    df["cgcnn_ef_eV_atom"] = [cg_ef.get(f, (np.nan, None))[0] for f in formulas]
    df["struct_material_id"] = [cg_ef.get(f, (np.nan, None))[1] for f in formulas]
    df["cgcnn_voltage_V"] = [cg_v.get(f, (np.nan, np.nan, np.nan))[0] for f in formulas]
    df["struct_voltage_n_li"] = [cg_v.get(f, (np.nan, np.nan, np.nan))[1] for f in formulas]
    df["struct_voltage_true_V"] = [cg_v.get(f, (np.nan, np.nan, np.nan))[2] for f in formulas]

    df["cgcnn_gap_eV"] = [cg_gap.get(f, np.nan) for f in formulas]
    df["has_struct_gap"] = df["cgcnn_gap_eV"].notna()
    # 评分口径选择（见 metrics.note / 报告附录 J）：
    # CGCNN 带隙来自 MP 的 PBE 值，对 Li 体系的排序与实验口径**负相关**（ρ≈-0.57，PBE 系统性低估），
    # 因此默认仍用「实验带隙训练的成分模型」参与评分，结构带隙作为**交叉检查列**保留；
    # 需要纯结构口径时用 --prefer-struct-gap 切换。
    if args.prefer_struct_gap:
        df["gap_eV"] = df["cgcnn_gap_eV"].where(df["has_struct_gap"], df["comp_gap_eV"])
        df["gap_source"] = np.where(df["has_struct_gap"], "CGCNN(结构,PBE)", "成分模型")
    else:
        df["gap_eV"] = df["comp_gap_eV"]
        df["gap_source"] = "成分模型(实验口径) + CGCNN 交叉检查"
    df["gap_diff_eV"] = (df["cgcnn_gap_eV"] - df["comp_gap_eV"]).abs()
    df["has_structure"] = df["cgcnn_ef_eV_atom"].notna()
    df["has_struct_voltage"] = df["cgcnn_voltage_V"].notna()
    df["stability_eV_atom"] = df["cgcnn_ef_eV_atom"].where(df["has_structure"],
                                                           df["comp_ef_eV_atom"])
    df["voltage_V"] = df["cgcnn_voltage_V"].where(df["has_struct_voltage"], df["comp_voltage_V"])
    df["stability_source"] = np.where(df["has_structure"], "CGCNN", "成分模型")
    df["voltage_source"] = np.where(df["has_struct_voltage"], "双塔CGCNN", "成分模型")
    # 排名偏离（仅对有结构的候选可比：在该子集内分别按两个来源排名）
    sub = df[df["has_structure"]].copy()
    if len(sub) > 3:
        r_struct = sub["cgcnn_ef_eV_atom"].rank(ascending=True)      # 越负越稳定 -> 名次越小
        r_comp = sub["comp_ef_eV_atom"].rank(ascending=True)
        df.loc[sub.index, "rank_struct"] = r_struct
        df.loc[sub.index, "rank_comp"] = r_comp
        df.loc[sub.index, "rank_shift"] = (r_struct - r_comp).abs()
    else:
        df["rank_struct"] = np.nan
        df["rank_comp"] = np.nan
        df["rank_shift"] = np.nan

    # ---- 理论容量 ----
    q = []
    for f, n_li in zip(df["formula"], df["nLi_extract"]):
        try:
            q.append(theoretical_capacity_mAh_g(parse_formula(str(f)), float(n_li)))
        except ValueError:
            q.append(np.nan)
    df["Q_theoretical_mAh_g"] = q

    # ---- 评分 ----
    df["pct_stability"] = df["stability_eV_atom"].rank(pct=True, ascending=False) * 100
    df["pct_conduction"] = np.exp(-df["gap_eV"].clip(lower=0)).rank(pct=True) * 100
    df["pct_capacity"] = df["Q_theoretical_mAh_g"].rank(pct=True) * 100
    df["score"] = (W_STAB * df["pct_stability"] + W_COND * df["pct_conduction"]
                   + W_CAP * df["pct_capacity"])

    # ---- 风险标记 ----
    df["flag"] = ""
    df.loc[~df["has_structure"], "flag"] += "结构缺失(仅成分回退);"
    df.loc[df["rank_shift"] > RANK_SHIFT_THRESHOLD, "flag"] += "结构/成分排名偏离大;"
    df.loc[df["gap_eV"] <= 0, "flag"] += "带隙≤0(金属/外推);"
    df.loc[df["has_struct_gap"] & (df["gap_diff_eV"] > GAP_DIFF_THRESHOLD),
           "flag"] += "带隙结构/成分差异大;"
    # PBE 系统性低估带隙：结构模型判金属、成分模型判半导体时单独提示（避免导电性分虚高）
    df.loc[(df["cgcnn_gap_eV"] <= 0.05) & (df["comp_gap_eV"] > 0.5),
           "flag"] += "结构判金属/成分判半导体(PBE低估);"
    df.loc[df["has_structure"] & ~df["has_struct_voltage"], "flag"] += "电压成分回退;"
    df.loc[df["flag"] == "", "flag"] = "结构&成分一致"

    # ---- 关键方法学处理：两套稳定性口径不可混排 ----
    # 主排名只收录**有结构**的候选（全部来自 CGCNN 口径，尺度一致）；
    # 无结构的候选单列，不参与主排名（否则会因口径差异被系统性压低/抬高）。
    covered = df[df["has_structure"]].sort_values("score", ascending=False).reset_index(drop=True)
    covered.insert(0, "rank", np.arange(1, len(covered) + 1))
    fallback = df[~df["has_structure"]].sort_values("score", ascending=False).reset_index(drop=True)

    rho = np.nan
    if len(covered) > 3:
        rho = float(covered["cgcnn_ef_eV_atom"].corr(covered["comp_ef_eV_atom"],
                                                    method="spearman"))
    df = covered

    OUT.mkdir(exist_ok=True)
    keep = ["rank", "name", "formula", "family", "cgcnn_ef_eV_atom", "rank_struct",
            "rank_comp", "rank_shift", "gap_source", "gap_eV", "cgcnn_gap_eV",
            "comp_gap_eV", "gap_diff_eV", "voltage_source", "voltage_V",
            "Q_theoretical_mAh_g", "score", "flag", "struct_material_id"]
    out_csv = OUT / f"cathode_ranking_struct.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    fb_csv = OUT / f"cathode_ranking_struct_fallback.csv"
    fallback.to_csv(fb_csv, index=False, encoding="utf-8-sig")
    print(f"\n主排名（有结构，{len(df)} 条）-> results/{out_csv.name}")
    print(df[keep].head(12).round(3).to_string(index=False))
    if len(fallback):
        print(f"\n无结构回退表（{len(fallback)} 条，不参与主排名）-> results/{fb_csv.name}")
        print(fallback[["name", "formula", "comp_ef_eV_atom", "comp_gap_eV",
                        "comp_voltage_V", "Q_theoretical_mAh_g", "score"]]
              .round(3).to_string(index=False))

    # ---- 排名图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    colors = ["#4393c3" if s == "CGCNN" else "#b2182b" for s in top["stability_source"]]
    ax.barh([f"{n} ({f})" for n, f in zip(top["name"], top["formula"])], top["score"],
            color=colors)
    ax.set_xlabel("综合评分 (0-100)")
    ax.set_title(f"结构模型驱动的候选筛选 Top 20\n"
                 f"评分 = {W_STAB:.2f}×稳定性 + {W_COND:.2f}×导电性 + {W_CAP:.2f}×理论容量"
                 f"（仅收录有结构的候选；蓝=CGCNN 结构稳定性）", fontsize=9)
    from matplotlib.patches import Rectangle
    handles = [Rectangle((0, 0), 1, 1, color="#4393c3")]
    ax.legend(handles, ["CGCNN 结构稳定性"], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"cathode_ranking_{args.tag}.png", dpi=150)
    plt.close(fig)
    print(f"排名图 -> results/cathode_ranking_{args.tag}.png")

    metrics = {
        "n_candidates": int(len(cand)),
        "n_ranked_with_structure": int(len(df)),
        "n_fallback_without_structure": int(len(fallback)),
        "struct_voltage_coverage_among_ranked": int(df["has_struct_voltage"].sum()),
        "struct_gap_coverage_among_ranked": int(df["has_struct_gap"].sum()),
        "gap_metal_disagreement_count": int(
            ((df["cgcnn_gap_eV"] <= 0.05) & (df["comp_gap_eV"] > 0.5)).sum()),
        "gap_rank_correlation_struct_vs_comp": (None if len(df) < 4 else round(float(
            df["cgcnn_gap_eV"].corr(df["comp_gap_eV"], method="spearman")), 4)),
        "weights": {"stability": W_STAB, "conduction": W_COND, "capacity": W_CAP},
        "flag_counts": df["flag"].value_counts().to_dict(),
        "top10": df.head(10)[["rank", "name", "formula", "score", "flag"]].to_dict("records"),
        "stability_rank_correlation_struct_vs_comp": None if rho != rho else round(rho, 4),
        "gap_scoring_source": ("CGCNN(结构,PBE)" if args.prefer_struct_gap
                               else "成分模型(实验口径)"),
        "note": ("**主排名只收录有结构的候选**：稳定性来自 CGCNN（MP 含 Li 域，每原子形成能）。"
                 "成分模型的 e_form 是 matbench 5 原子胞口径（正负号与 MP 都不同），"
                 "两者不可混排，故无结构候选单列于 *_fallback.csv。"
                 "电压优先双塔 CGCNN（需电极对），否则回退成分电压模型；"
                 "导电性默认用实验口径的成分带隙（CGCNN 结构带隙在 Li 体系与实验口径排序负相关 ρ≈-0.57，"
                 "PBE 系统性低估，仅作交叉检查列），可用 --prefer-struct-gap 切换为结构口径。"),
        "caveats": [
            "CGCNN 稳定性为 MP PBE 口径，与实验值仍有系统偏差",
            "结构带隙为 MP DFT-PBE 口径：在 Li 体系上与实验口径排序负相关（ρ≈-0.57），"
            "金属/非金属判对率 83%，不适合单独作为导电性判据",
            "容差匹配得到的结构是 MP 的近似有序超胞，与真实固溶体/无序相有差异",
        ],
    }
    with open(OUT / f"metrics_screening_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] results/metrics_screening_{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())