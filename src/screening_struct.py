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
PROTOTYPES = ROOT / "data" / "raw" / "mp_structures_candidates_prototype.json.gz"
POLYMORPHS = ROOT / "data" / "raw" / "mp_structures_candidates_polymorph.json.gz"
ENSEMBLES = ROOT / "data" / "raw" / "mp_structures_candidates_ensemble.json.gz"
LOCAL_CONFORMAL = ROOT / "results" / "local_conformal_voltage.csv"
CGCNN_EF = ROOT / "results" / "cgcnn_mp_8k.pt"
CGCNN_GAP = ROOT / "results" / "cgcnn_gap_8k.pt"
GAP_DIFF_THRESHOLD = 1.5            # eV，结构带隙(DFT-PBE) 与 成分带隙(实验训练) 的系统差异阈值
CGCNN_V = ROOT / "results" / "cgcnn_voltage.pt"
OUT = ROOT / "results"
# v1.15.0：加入"工作电压"维度（正极最核心指标），并让局部自适应区间宽度参与：
#   稳定性 0.35 / 导电性 0.20 / 容量 0.25 / 电压 0.20
W_STAB, W_COND, W_CAP, W_VOLT = 0.35, 0.20, 0.25, 0.20
W_STAB_3D, W_COND_3D, W_CAP_3D = 0.40, 0.30, 0.30      # v1.14.0 的三维口径（保留供对照）
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
    ap.add_argument("--prototypes", default=str(PROTOTYPES),
                    help="岩盐型原型文件（最后手段；见 generate_prototype_structures.py）")
    ap.add_argument("--polymorphs", default=str(POLYMORPHS),
                    help="同构替换相文件（优先级高于岩盐原型；见 template_substitution.py）")
    ap.add_argument("--local-conformal", default=str(LOCAL_CONFORMAL),
                    help="局部自适应区间宽度（见 local_conformal.py）；缺失则退回全局 q")
    ap.add_argument("--ensemble", default=str(ENSEMBLES),
                    help="无序系综文件（优先级最高；提供装饰噪声误差条；见 drx_ensemble.py）")
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

    # ---- 结构来源：MP 弛豫结构优先，缺失时用原型（v1.11.0，CHGNet 弛豫） ----
    cg_ef, cg_gap = {}, {}
    struct_map, origin, struct_ids = {}, {}, {}
    if not args.no_cgcnn:
        if Path(args.structures).exists():
            with gzip.open(args.structures, "rt", encoding="utf-8") as f:
                for r in json.load(f):
                    if r.get("structure"):
                        f_ = r["query_formula"]
                        struct_map[f_] = r["structure"]
                        origin[f_] = "MP弛豫"
                        struct_ids[f_] = r.get("material_id", "")
        # 第二优先级：无序系综的代表结构（系综内 CHGNet 能量最低者），
        # 并把"装饰噪声"作为误差条带进筛选（v1.13.0）
        ens_stats = {}
        if Path(args.ensemble).exists():
            n_ens = 0
            with gzip.open(args.ensemble, "rt", encoding="utf-8") as f:
                for r in json.load(f):
                    f_ = r.get("query_formula")
                    if not f_ or not r.get("structure") or f_ in struct_map:
                        continue
                    struct_map[f_] = r["structure"]
                    origin[f_] = "无序系综(CHGNet)"
                    struct_ids[f_] = r.get("material_id", f"ens:{f_}")
                    e = r.get("ensemble") or {}
                    ens_stats[f_] = {
                        "ef_mean": e.get("cgcnn_ef_mean_eV_atom"),
                        "ef_std": e.get("cgcnn_ef_std_eV_atom"),
                        "ef_best": e.get("best_member_ef_eV_atom"),
                        "gap_mean": e.get("cgcnn_gap_mean_eV"),
                        "gap_std": e.get("cgcnn_gap_std_eV"),
                        "n_decorations": e.get("n_decorations"),
                    }
                    n_ens += 1
            print(f"无序系综代表结构补充: {n_ens} 条 → 结构覆盖 {len(struct_map)}/{len(formulas)}")

        # 第三优先级：同构替换 + MLIP 选相（实测与 MP 结构 bias +0.09 eV/atom、ρ 0.97，
        # 因此可与 MP 结构同层比较；见报告附录 L）
        if Path(args.polymorphs).exists():
            n_poly = 0
            with gzip.open(args.polymorphs, "rt", encoding="utf-8") as f:
                for r in json.load(f):
                    f_ = r.get("query_formula")
                    if not f_ or not r.get("structure") or f_ in struct_map:
                        continue
                    struct_map[f_] = r["structure"]
                    origin[f_] = "同构替换(CHGNet)"
                    struct_ids[f_] = r.get("material_id", f"poly:{f_}")
                    n_poly += 1
            print(f"同构替换相补充: {n_poly} 条 → 结构覆盖 {len(struct_map)}/{len(formulas)}")

        # 第四优先级：岩盐型原型（仅组内可比）
        if Path(args.prototypes).exists():
            n_proto = 0
            with gzip.open(args.prototypes, "rt", encoding="utf-8") as f:
                for r in json.load(f):
                    f_ = r.get("query_formula")
                    if not f_ or not r.get("structure") or f_ in struct_map:
                        continue                      # MP 结构优先
                    struct_map[f_] = r["structure"]
                    origin[f_] = "原型(CHGNet弛豫)"
                    struct_ids[f_] = r.get("material_id", f"proto:{f_}")
                    n_proto += 1
            print(f"原型结构补充: {n_proto} 条 → 结构覆盖 {len(struct_map)}/{len(formulas)}")
        else:
            print("[INFO] 未找到原型结构文件（可跑 generate_prototype_structures.py 补缺失候选）")

        if struct_map:
            rows_for_ef = [{"query_formula": f, "material_id": struct_ids.get(f, ""),
                            "structure": st} for f, st in struct_map.items()]
            cg_ef = _cgcnn_e_form(Path(CGCNN_EF), rows_for_ef)
            print(f"结构命中（CGCNN 稳定性）: {len(cg_ef)}/{len(formulas)}")
            cg_gap = _cgcnn_scalar(Path(CGCNN_GAP), struct_map, log1p=True)
            print(f"结构带隙命中（CGCNN）: {len(cg_gap)}/{len(formulas)}")
        else:
            print("[INFO] 无结构文件或已禁用 CGCNN → 全部回退成分模型")

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
    df["structure_origin"] = [origin.get(f, "无结构") for f in formulas]
    df["structure_from_prototype"] = df["structure_origin"].str.startswith("原型")
    # 置信层级（跨层不可混排，见下方评分与产物）：
    #   A_MP=MP 弛豫结构 / B_prototype=原型结构 / C_none=无结构
    # A_comparable：MP 弛豫结构 或 同构替换相（两者实测可比，见 metrics 里的残差）
    # B_prototype ：只有岩盐型原型（仅组内可比）
    df["structure_tier"] = np.where(
        df["structure_origin"].isin(["MP弛豫", "同构替换(CHGNet)", "无序系综(CHGNet)"]),
        "A_comparable",
        np.where(df["structure_from_prototype"], "B_prototype", "C_none"))
    df["has_struct_voltage"] = df["cgcnn_voltage_V"].notna()
    # ---- 局部自适应区间宽度（每候选自己的 90% 半宽） ----
    loc_widths, loc_info = {}, {}
    lc_path = Path(args.local_conformal)
    if lc_path.exists():
        lc = pd.read_csv(lc_path, encoding="utf-8-sig")
        col = ("local_q90_calibrated_V" if "local_q90_calibrated_V" in lc.columns
               else "local_q90_weighted_V")
        loc_widths = dict(zip(lc["formula"].astype(str), lc[col].astype(float)))
        try:
            with open(OUT / "metrics_local_conformal.json", encoding="utf-8") as fh:
                loc_info = json.load(fh)
        except Exception:  # noqa: BLE001
            loc_info = {}
        print(f"局部自适应区间: {len(loc_widths)} 个候选（列 {col}）")
    df["voltage_pi90_halfwidth_V"] = [loc_widths.get(f, np.nan) for f in formulas]
    # 全局 q 兜底（无局部宽度时用常数）
    glob_q = ((loc_info.get("global_q_V")) if loc_info else None)
    df["voltage_halfwidth_used_V"] = df["voltage_pi90_halfwidth_V"].fillna(
        float(glob_q) if glob_q else 0.0)
    # 装饰噪声误差条（仅系综来源的候选有值）
    def _ens(f, key):
        return (ens_stats.get(f) or {}).get(key, np.nan)

    df["ef_ens_std_eV_atom"] = [_ens(f, "ef_std") for f in formulas]
    df["gap_ens_std_eV"] = [_ens(f, "gap_std") for f in formulas]
    df["ef_ens_mean_eV_atom"] = [_ens(f, "ef_mean") for f in formulas]
    df["gap_ens_mean_eV"] = [_ens(f, "gap_mean") for f in formulas]
    df["n_decorations"] = [_ens(f, "n_decorations") for f in formulas]

    # ---- 系综来源的候选：**评分改用系综均值**（单装饰名次不稳，见报告附录 M） ----
    df["cgcnn_ef_best_member_eV_atom"] = df["cgcnn_ef_eV_atom"]
    df["cgcnn_gap_best_member_eV"] = df["cgcnn_gap_eV"]
    ens_mask = df["ef_ens_mean_eV_atom"].notna()
    df.loc[ens_mask, "cgcnn_ef_eV_atom"] = df.loc[ens_mask, "ef_ens_mean_eV_atom"]
    df.loc[ens_mask & df["gap_ens_mean_eV"].notna(), "cgcnn_gap_eV"] = \
        df.loc[ens_mask & df["gap_ens_mean_eV"].notna(), "gap_ens_mean_eV"]
    df["stability_eV_atom"] = df["cgcnn_ef_eV_atom"].where(df["has_structure"],
                                                           df["comp_ef_eV_atom"])
    df["voltage_V"] = df["cgcnn_voltage_V"].where(df["has_struct_voltage"], df["comp_voltage_V"])
    df["stability_source"] = np.where(df["has_structure"], "CGCNN", "成分模型")
    df["voltage_source"] = np.where(df["has_struct_voltage"], "双塔CGCNN", "成分模型")
    # 电压保守值：区间下界（电压作为"越高越好"的指标，保守取减法）
    df["voltage_conservative_V"] = df["voltage_V"] - df["voltage_halfwidth_used_V"]
    # 排名偏离：只在 **A 层（MP 结构）** 内部比 —— B 层原型的预测带多晶型系统偏移，
    # 拿它和成分模型比排名会引入假信号。
    df["rank_struct"] = np.nan
    df["rank_comp"] = np.nan
    df["rank_shift"] = np.nan
    sub = df[df["structure_tier"] == "A_comparable"].copy()
    if len(sub) > 3:
        r_struct = sub["cgcnn_ef_eV_atom"].rank(ascending=True)      # 越负越稳定 -> 名次越小
        r_comp = sub["comp_ef_eV_atom"].rank(ascending=True)
        df.loc[sub.index, "rank_struct"] = r_struct
        df.loc[sub.index, "rank_comp"] = r_comp
        df.loc[sub.index, "rank_shift"] = (r_struct - r_comp).abs()

    # ---- 理论容量 ----
    q = []
    for f, n_li in zip(df["formula"], df["nLi_extract"]):
        try:
            q.append(theoretical_capacity_mAh_g(parse_formula(str(f)), float(n_li)))
        except ValueError:
            q.append(np.nan)
    df["Q_theoretical_mAh_g"] = q

    # ---- 1σ 保守估计（无误差条时 σ=0） ----
    #  稳定性：越负越稳定 -> 保守取 ef + σ（更不稳定的一侧），σ 来自结构系综；
    #  导电性：带隙越大越差 -> 保守取 gap + σ，σ 是**结构装饰噪声**。
    #  注意：默认导电性评分用的是实验口径的成分带隙（附录 J），此时 σ 不表示
    #  成分带隙自身的误差，而是"该候选的结构数据质量较低"的保守惩罚；
    #  若用 --prefer-struct-gap，σ 则正好是结构带隙的不确定度。
    df["ef_sigma_eV_atom"] = df["ef_ens_std_eV_atom"].fillna(0.0)
    df["gap_sigma_eV"] = df["gap_ens_std_eV"].fillna(0.0)
    df["stability_conservative_eV_atom"] = df["stability_eV_atom"] + df["ef_sigma_eV_atom"]
    df["gap_conservative_eV"] = df["gap_eV"] + df["gap_sigma_eV"]

    # ---- 评分（**层内百分位**：跨层不可比，故每层各自归一化；用 1σ 保守量） ----
    for col in ("pct_stability", "pct_conduction", "pct_capacity"):
        df[col] = np.nan
    for _tier, idx in df.groupby("structure_tier").groups.items():
        sub = df.loc[idx]
        df.loc[idx, "pct_stability"] = (sub["stability_conservative_eV_atom"]
                                        .rank(pct=True, ascending=False) * 100)
        df.loc[idx, "pct_conduction"] = (np.exp(-sub["gap_conservative_eV"].clip(lower=0))
                                         .rank(pct=True) * 100)
        df.loc[idx, "pct_capacity"] = (sub["Q_theoretical_mAh_g"]
                                       .rank(pct=True) * 100)
    for _tier, idx in df.groupby("structure_tier").groups.items():
        sub = df.loc[idx]
        df.loc[idx, "pct_voltage"] = (sub["voltage_conservative_V"]
                                      .rank(pct=True) * 100)
    df["score_3dim"] = (W_STAB_3D * df["pct_stability"] + W_COND_3D * df["pct_conduction"]
                        + W_CAP_3D * df["pct_capacity"])
    df["score"] = (W_STAB * df["pct_stability"] + W_COND * df["pct_conduction"]
                   + W_CAP * df["pct_capacity"] + W_VOLT * df["pct_voltage"])

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
    # 原型结构：未由 DFT 弛豫，稳定性/带隙带额外不确定度（见 prototype_sensitivity）
    df.loc[df["structure_from_prototype"], "flag"] += "结构为岩盐原型(仅组内可比);"
    df.loc[df["structure_origin"] == "同构替换(CHGNet)",
           "flag"] += "结构为同构替换相(已验证与MP可比);"
    df.loc[df["structure_origin"] == "无序系综(CHGNet)",
           "flag"] += "评分用系综均值(已含1σ保守惩罚);"
    df.loc[df["flag"] == "", "flag"] = "结构&成分一致"

    # ---- 分层产物（tier 已在前面算好，此处只做分组与排序） ----
    covered = (df[df["structure_tier"] == "A_comparable"]
               .sort_values("score", ascending=False).reset_index(drop=True))
    covered.insert(0, "rank", np.arange(1, len(covered) + 1))
    proto = (df[df["structure_tier"] == "B_prototype"]
             .sort_values("score", ascending=False).reset_index(drop=True))
    proto.insert(0, "rank", np.arange(1, len(proto) + 1))
    fallback = (df[df["structure_tier"] == "C_none"]
                .sort_values("score", ascending=False).reset_index(drop=True))

    rho = np.nan
    if len(covered) > 3:
        rho = float(covered["cgcnn_ef_eV_atom"].corr(covered["comp_ef_eV_atom"],
                                                    method="spearman"))

    # 原型组与 MP 组的系统偏移（来自 prototype_sensitivity.py 的实测）
    proto_unc, poly_unc = {}, {}
    unc_path = OUT / "metrics_prototype_sensitivity.json"
    if unc_path.exists():
        with open(unc_path, encoding="utf-8") as fh:
            proto_unc = json.load(fh)
    poly_path = OUT / "metrics_polymorph_sensitivity.json"
    if poly_path.exists():
        with open(poly_path, encoding="utf-8") as fh:
            poly_unc = json.load(fh)
    df = covered

    OUT.mkdir(exist_ok=True)
    keep = ["rank", "name", "formula", "family", "structure_origin", "cgcnn_ef_eV_atom",
            "ef_ens_std_eV_atom", "gap_ens_std_eV", "n_decorations",
            "cgcnn_ef_best_member_eV_atom", "stability_conservative_eV_atom",
            "gap_conservative_eV",
            "rank_struct", "rank_comp", "rank_shift", "gap_source", "gap_eV",
            "cgcnn_gap_eV", "comp_gap_eV", "gap_diff_eV", "voltage_source", "voltage_V",
            "Q_theoretical_mAh_g", "voltage_conservative_V", "voltage_pi90_halfwidth_V",
            "pct_stability", "pct_conduction", "pct_capacity", "pct_voltage",
            "score", "score_3dim", "flag", "struct_material_id"]
    out_csv = OUT / f"cathode_ranking_struct.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    fb_csv = OUT / f"cathode_ranking_struct_fallback.csv"
    fallback.to_csv(fb_csv, index=False, encoding="utf-8-sig")
    proto_csv = OUT / f"cathode_ranking_struct_prototype.csv"
    proto.to_csv(proto_csv, index=False, encoding="utf-8-sig")
    print(f"\nA 层主排名（MP 弛豫结构，{len(df)} 条）-> results/{out_csv.name}")
    print(df[keep].head(12).round(3).to_string(index=False))
    if len(proto):
        print(f"\nB 层原型组（CHGNet 弛豫原型，{len(proto)} 条；**仅组内可比**，"
              f"与 A 层系统偏移约 "
              f"{proto_unc.get('ef', {}).get('bias_eV_atom', float('nan')):+.2f} eV/atom）"
              f"-> results/{proto_csv.name}")
        print(proto[["rank", "name", "formula", "cgcnn_ef_eV_atom", "cgcnn_gap_eV",
                     "comp_voltage_V", "Q_theoretical_mAh_g", "score"]]
              .round(3).to_string(index=False))
    if len(fallback):
        print(f"\nC 层无结构回退（{len(fallback)} 条）-> results/{fb_csv.name}")
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
        "n_prototype_only": int(len(proto)),
        "n_fallback_without_structure": int(len(fallback)),
        "structure_tiers": {"A_comparable": int(len(df)),
                            "B_prototype_rocksalt": int(len(proto)),
                            "C_no_structure": int(len(fallback))},
        "n_from_mp_structure": int((df["structure_origin"] == "MP弛豫").sum()),
        "n_from_polymorph_substitution": int((df["structure_origin"] == "同构替换(CHGNet)").sum()),
        "n_from_ensemble": int((df["structure_origin"] == "无序系综(CHGNet)").sum()),
        "scoring_mode": "conservative_1sigma",
        "decoration_noise": {
            "ef_std_mean_eV_atom": (None if df["ef_ens_std_eV_atom"].dropna().empty
                                    else round(float(df["ef_ens_std_eV_atom"].mean()), 4)),
            "gap_std_mean_eV": (None if df["gap_ens_std_eV"].dropna().empty
                                else round(float(df["gap_ens_std_eV"].mean()), 4)),
            "n_ensemble_candidates": int(df["n_decorations"].notna().sum()),
            "note": ("无序系综（随机装饰）带来的预测不确定度。评分为 1σ 保守口径："
                     "稳定性取 mean+σ（更不稳定一侧）；导电性在带隙上加 σ，"
                     "该 σ 是结构装饰噪声（默认评分用成分带隙时，它代表数据质量惩罚；"
                     "配合 --prefer-struct-gap 时即为结构带隙的不确定度）。"
                     "系综候选的稳定性/带隙点预测已改用**系综均值**，"
                     "并保留 best-member 值供对照。"),
        },
        "polymorph_residual_uncertainty": {
            "n_compared": poly_unc.get("n_compared"),
            "ef_bias_eV_atom": poly_unc.get("ef", {}).get("bias_eV_atom"),
            "ef_MAE_eV_atom": poly_unc.get("ef", {}).get("MAE_proto_vs_mp_eV_atom"),
            "ef_spearman": poly_unc.get("ef", {}).get("spearman"),
            "gap_bias_eV": poly_unc.get("gap", {}).get("bias_eV"),
            "note": "同构替换相 vs MP 真实结构的残差；足够小 -> 可与 MP 结构同层比较",
        },
        "prototype_extra_uncertainty": {
            "ef_bias_eV_atom": proto_unc.get("ef", {}).get("bias_eV_atom"),
            "ef_MAE_eV_atom": proto_unc.get("ef", {}).get("MAE_proto_vs_mp_eV_atom"),
            "gap_bias_eV": proto_unc.get("gap", {}).get("bias_eV"),
            "gap_MAE_eV": proto_unc.get("gap", {}).get("MAE_proto_vs_mp_eV"),
            "note": "原型（岩盐型）与 MP 真实多晶型的差异；B 层候选仅组内可比",
        },
        "struct_voltage_coverage_among_ranked": int(df["has_struct_voltage"].sum()),
        "struct_gap_coverage_among_ranked": int(df["has_struct_gap"].sum()),
        "gap_metal_disagreement_count": int(
            ((df["cgcnn_gap_eV"] <= 0.05) & (df["comp_gap_eV"] > 0.5)).sum()),
        "gap_rank_correlation_struct_vs_comp": (None if len(df) < 4 else round(float(
            df["cgcnn_gap_eV"].corr(df["comp_gap_eV"], method="spearman")), 4)),
        "weights": {"stability": W_STAB, "conduction": W_COND, "capacity": W_CAP,
                    "voltage": W_VOLT},
        "weights_3dim_legacy": {"stability": W_STAB_3D, "conduction": W_COND_3D,
                                "capacity": W_CAP_3D},
        "voltage_dimension": {
            "source": "局部自适应 conformal（已校准）",
            "n_candidates_with_local_width": int(df["voltage_pi90_halfwidth_V"].notna().sum()),
            "global_q_V": glob_q,
            "local_coverage_raw": (loc_info.get("validation", {}) or {}).get("empirical_coverage"),
            "local_coverage_calibrated": (loc_info.get("validation", {}) or {}).get(
                "empirical_coverage_calibrated"),
            "differentiation_max_over_min": (loc_info.get("local_halfwidth_V", {}) or {}).get(
                "max_over_min"),
            "note": ("局部自适应区间按特征近邻给出每个候选自己的半宽；实测原始覆盖率 77% "
                     "(局部可交换假设不成立)，按 1.47× 校准后 88.7% ≈ 名义 90%，"
                     "同时保留 9.4× 的相对区分度。评分的电压维度用区间下界（保守）。"),
        },
        "flag_counts": df["flag"].value_counts().to_dict(),
        "top10": df.head(10)[["rank", "name", "formula", "score", "flag"]].to_dict("records"),
        "stability_rank_correlation_struct_vs_comp": None if rho != rho else round(rho, 4),
        "gap_scoring_source": ("CGCNN(结构,PBE)" if args.prefer_struct_gap
                               else "成分模型(实验口径)"),
        "note": ("**主排名（A 层）只收录有 MP 弛豫结构的候选**：稳定性来自 CGCNN（MP 含 Li 域，每原子形成能）。"
                 "成分模型的 e_form 是 matbench 5 原子胞口径（正负号与 MP 都不同），"
                 "两者不可混排，故无结构候选单列于 *_fallback.csv（C 层）。"
                 "A 层除 MP 结构外，还包含**同构替换 + CHGNet 选相**补出的结构："
                 "实测其与 MP 真实结构的 CGCNN 预测 bias +0.09 eV/atom、ρ 0.97，"
                 "与模型自身误差同量级，故可同层比较。另有 B 层（岩盐型原型）单列，"
                 "因其与真实多晶型差 +1.25 eV/atom，**仅组内可比**。"
                 "电压优先双塔 CGCNN（需电极对），否则回退成分电压模型；"
                 "导电性默认用实验口径的成分带隙（CGCNN 结构带隙在 Li 体系与实验口径排序负相关 ρ≈-0.57，"
                 "PBE 系统性低估，仅作交叉检查列），可用 --prefer-struct-gap 切换为结构口径。"),
        "caveats": [
            "CGCNN 稳定性为 MP PBE 口径，与实验值仍有系统偏差",
            "结构带隙为 MP DFT-PBE 口径：在 Li 体系上与实验口径排序负相关（ρ≈-0.57），"
            "金属/非金属判对率 83%，不适合单独作为导电性判据",
            "容差匹配得到的结构是 MP 的近似有序超胞，与真实固溶体/无序相有差异",
            "B 层原型结构为岩盐型亚稳多晶型（CHGNet 弛豫），与真实多晶型有系统能量差，"
            "只能做组内相对比较；要并入主排名须做 DFT/MLIP 多晶型搜索",
        ],
    }
    with open(OUT / f"metrics_screening_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] results/metrics_screening_{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())