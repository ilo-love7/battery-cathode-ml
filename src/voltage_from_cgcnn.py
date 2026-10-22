# -*- coding: utf-8 -*-
r"""用 CGCNN 从**两相结构**计算平均电压（物理化路径，无需额外训练）。

原理（与 ``src/voltage.py`` 同一套约定）：

    V = -ΔG / (n_Li · e) ≈ [ E_form(脱锂相) − E_form(嵌锂相) ] / n_Li

其中 Li 金属形成能为 0。这里 E_form 由 ``cgcnn_mp*.pt``（每原子形成能，eV/atom）
给出，乘以各自单胞原子数得到每胞值，n_Li 取两相原子数差（≈ 可脱嵌 Li 数）。

输入::

    data/processed/mp_voltage_struct.csv.gz     （含 id_charge / id_discharge / n_li / 真值电压）
    data/raw/mp_structures_electrodes.json.gz   （5,337 个结构，按 material_id 索引）

输出::

    results/voltage_from_cgcnn.csv
    results/metrics_voltage_from_cgcnn.json

用法::

    # 结构文件缺失时先取（约 3 分钟）
    python src/fetch_mp_structures.py --mode ids --ids-file data/raw/voltage_material_ids.txt \
        --out data/raw/mp_structures_electrodes.json.gz

    D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/voltage_from_cgcnn.py
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

VOLTAGE_STRUCT = ROOT / "data" / "processed" / "mp_voltage_struct.csv.gz"
STRUCT_PATH = ROOT / "data" / "raw" / "mp_structures_electrodes.json.gz"
OUT_CSV = ROOT / "results" / "voltage_from_cgcnn.csv"
OUT_JSON = ROOT / "results" / "metrics_voltage_from_cgcnn.json"


def load_model(model_path: Path):
    import torch
    from cgcnn_model import build_model

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("atom_fea_len", 64), ckpt.get("n_conv", 3),
                        ckpt.get("h_fea_len", 128), ckpt.get("n_h", 1))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, ckpt, device


def predict_e_per_atom(model, ckpt, device, structures: dict, ids: list):
    """对一批 material_id 预测每原子形成能（eV/atom），返回 {id: (e_atom, n_atoms)}。"""
    import torch
    from cgcnn_model import batch_tensors

    mean, std = float(ckpt.get("target_mean", 0.0)), float(ckpt.get("target_std", 1.0))
    graphs, keys, natoms = [], [], []
    for mid in ids:
        st = structures.get(str(mid))
        if st is None:
            continue
        g = build_graph(st)
        if g is None:
            continue
        graphs.append(g)
        keys.append(str(mid))
        natoms.append(int(g["n_atoms"]))

    out = {}
    with torch.no_grad():
        for s in range(0, len(graphs), 128):
            chunk = list(range(s, min(s + 128, len(graphs))))
            (node, efea, src, dst, bvec) = batch_tensors(graphs, chunk, device)
            pred = model(node, efea, src, dst, bvec) * std + mean
            pred = pred.detach().cpu().numpy()
            for k, p, n in zip([keys[i] for i in chunk], pred, [natoms[i] for i in chunk]):
                out[k] = (float(p), int(n))
    return out


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="用 CGCNN 从两相结构计算平均电压")
    ap.add_argument("--model", default=str(ROOT / "results" / "cgcnn_mp_8k.pt"))
    ap.add_argument("--voltage", default=str(VOLTAGE_STRUCT))
    ap.add_argument("--structures", default=str(STRUCT_PATH))
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] 需要 PyTorch（GPU 环境见 README「GPU 结构模型（CGCNN）」）")
        return 2

    vp, sp = Path(args.voltage), Path(args.structures)
    if not vp.exists():
        print(f"[ERROR] 找不到 {vp}（先跑 python src/prepare_voltage_struct.py）")
        return 2
    if not sp.exists():
        print(f"[ERROR] 找不到 {sp}")
        print("        请先运行: python src/fetch_mp_structures.py --mode ids "
              "--ids-file data/raw/voltage_material_ids.txt "
              "--out data/raw/mp_structures_electrodes.json.gz")
        return 2
    if not Path(args.model).exists():
        print(f"[ERROR] 找不到模型 {args.model}（先跑 python src/train_cgcnn_mp.py）")
        return 2

    df = pd.read_csv(vp, encoding="utf-8-sig")
    if args.limit:
        df = df.head(args.limit)
        # 子集运行不得覆盖全量产物（v1.4.2 教训）
        stem, dot, ext = str(args.out).partition(".")
        args.out = f"{stem}_smoke{dot}{ext}" if dot else f"{args.out}_smoke"
        print(f"[冒烟模式] 输出改名: {Path(args.out).name}")
    with gzip.open(sp, "rt", encoding="utf-8") as f:
        structures = {r["material_id"]: r["structure"] for r in json.load(f) if r.get("structure")}

    model, ckpt, device = load_model(Path(args.model))
    print(f"模型 {Path(args.model).name}（训练于 {ckpt.get('trained_on')}，"
          f"{ckpt.get('n_train')} 训练样本）| device={device}")

    ids = sorted({str(i) for i in df["id_charge"].dropna()} |
                 {str(i) for i in df["id_discharge"].dropna()})
    print(f"待预测结构: {len(ids)} 个（{len(df)} 条记录）")
    energies = predict_e_per_atom(model, ckpt, device, structures, ids)
    print(f"成功预测: {len(energies)} / {len(ids)} 个结构")

    rows = []
    for idx, r in df.iterrows():
        if not bool(r.get("has_both_structures")):
            continue
        ec = energies.get(str(r["id_charge"]))
        el = energies.get(str(r["id_discharge"]))
        if ec is None or el is None:
            continue
        e_frame, n_frame = ec          # 脱锂相（eV/atom, 原子数）
        e_lith, n_lith = el            # 嵌锂相
        n_li = n_lith - n_frame
        if n_li <= 0:
            continue
        v_pred = (e_frame * n_frame - e_lith * n_lith) / n_li
        rows.append({
            "row_idx": int(idx),
            "formula": r["formula"], "framework_formula": r.get("framework_formula"),
            "voltage_true_V": float(r["average_voltage_V"]),
            "voltage_from_cgcnn_V": float(v_pred),
            "n_frame_atoms": n_frame, "n_lith_atoms": n_lith, "n_li": n_li,
            "e_frame_eV_atom": e_frame, "e_lith_eV_atom": e_lith,
            "voltage_comp_model_V": None,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("[ERROR] 没有可用记录（检查结构覆盖与 n_Li>0）")
        return 1
    res["abs_err_V"] = (res["voltage_from_cgcnn_V"] - res["voltage_true_V"]).abs()

    # 与成分模型对比：**用 OOF 预测**（同口径——否则拿全量拟合比逐条预测不公平）
    oof_path = ROOT / "results" / "oof_voltage.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path, encoding="utf-8-sig")
        if len(oof) == len(df):
            # 注意：res 是重建索引，必须用记录的 row_idx 回查 OOF
            res["voltage_comp_model_V"] = oof["pred_ens"].to_numpy()[res["row_idx"].to_numpy()]
        else:
            print(f"[WARN] oof_voltage.csv 行数({len(oof)}) != 电压数据行数({len(df)})，跳过对比")
    else:
        print("[WARN] 未找到 results/oof_voltage.csv，跳过成分模型对比")

    mae_c = float(res["abs_err_V"].mean())
    rmse_c = float(np.sqrt((res["abs_err_V"] ** 2).mean()))
    rho_c = float(res["voltage_true_V"].corr(res["voltage_from_cgcnn_V"], method="spearman"))
    mad = float((res["voltage_true_V"] - res["voltage_true_V"].median()).abs().mean())

    comp = {}
    if res["voltage_comp_model_V"].notna().any():
        sub = res.dropna(subset=["voltage_comp_model_V"])
        comp_err = (sub["voltage_comp_model_V"] - sub["voltage_true_V"]).abs()
        comp = {"n": int(len(sub)), "MAE_V": round(float(comp_err.mean()), 4),
                "spearman": round(float(sub["voltage_true_V"].corr(
                    sub["voltage_comp_model_V"], method="spearman")), 4)}

    metrics = {
        "n_pairs": int(len(res)),
        "model": Path(args.model).name,
        "trained_on": ckpt.get("trained_on"),
        "cgcnn_voltage": {"MAE_V": round(mae_c, 4), "RMSE_V": round(rmse_c, 4),
                          "spearman": round(rho_c, 4)},
        "composition_model_reference": comp,
        "median_baseline_MAD_V": round(mad, 4),
        "note": ("V = [E_form(脱锂相) − E_form(嵌锂相)] / n_Li，"
                 "其中每胞形成能 = 每原子预测 × 单胞原子数，n_Li 取两相原子数差。"),
    }
    out_json = OUT_JSON if not args.limit else OUT_JSON.with_name(
        OUT_JSON.stem + "_smoke" + OUT_JSON.suffix)
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 76)
    print("  用 CGCNN 两相结构计算电压（物理化路径）")
    print("=" * 76)
    print(f"  配对数 n={len(res)}（覆盖 {len(res)/max(len(df),1)*100:.1f}% 记录）")
    print(f"  CGCNN-ΔE 路径: MAE={mae_c:.4f} V  RMSE={rmse_c:.4f} V  ρ={rho_c:.3f}")
    if comp:
        print(f"  成分模型（同口径参考）: MAE={comp['MAE_V']:.4f} V  ρ={comp['spearman']:.3f}  n={comp['n']}")
    print(f"  中位数基线 MAD={mad:.4f} V")
    print(f"[SAVE] {Path(args.out).name} / {out_json.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())