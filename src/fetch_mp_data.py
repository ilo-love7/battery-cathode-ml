# -*- coding: utf-8 -*-
"""Materials Project 正式 API 数据拉取工具（需免费注册的 API Key）。

两种模式:
- --mode mpapi (推荐): 使用官方 mp-api 库 (pip install mp-api)
- --mode rest : 直接 requests 调 REST 端点（不装额外依赖）

拉取内容: Li-O 化学体系（可改 --formula）的
  material_id, formula_pretty, formation_energy_per_atom, band_gap,
  energy_above_hull, nsites, volume, is_stable, elements

用法示例:
  python fetch_mp_data.py --mode mpapi --key YOUR_MP_API_KEY
  python fetch_mp_data.py --mode rest  --key YOUR_MP_API_KEY --formula Li-Fe-P-O

输出: data/processed/mp_summary.csv

注意:
- API Key 在 https://next.materialsproject.org 注册后于 Dashboard 免费获取；
- REST 端点路径/参数可能随 MP API 版本演进，若 rest 模式报 4xx，
  请优先使用 mp-api 模式（库会自动跟随官方版本）；
- 学术用途免费，但请遵守 MP 使用条款与配额限制。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_FORMULA = "Li-*"          # 通配: 含 Li 的化学体系; 也可写 'Li-Fe-P-O'
FIELDS = ["material_id", "formula_pretty", "formation_energy_per_atom",
          "band_gap", "energy_above_hull", "nsites", "volume",
          "is_stable", "elements"]


def fetch_mpapi(key: str, formula: str) -> list[dict]:
    from mp_api.client import MPRester
    rows = []
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(formula=formula)
        for d in docs:
            rows.append({
                "material_id": str(getattr(d, "material_id", "") or ""),
                "formula_pretty": getattr(d, "formula_pretty", None),
                "formation_energy_per_atom": getattr(d, "formation_energy_per_atom", None),
                "band_gap": getattr(d, "band_gap", None),
                "energy_above_hull": getattr(d, "energy_above_hull", None),
                "nsites": getattr(d, "nsites", None),
                "volume": getattr(d, "volume", None),
                "is_stable": getattr(d, "is_stable", None),
                "elements": ",".join(str(e) for e in (getattr(d, "elements", None) or [])),
            })
    return rows


def fetch_rest(key: str, formula: str, limit: int) -> list[dict]:
    import requests
    url = "https://api.materialsproject.org/materials/summary/"
    headers = {"X-API-KEY": key}
    params = {"formula": formula, "_limit": limit,
              "_fields": ",".join(FIELDS)}
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for d in data:
        rows.append({
            "material_id": d.get("material_id", ""),
            "formula_pretty": d.get("formula_pretty"),
            "formation_energy_per_atom": d.get("formation_energy_per_atom"),
            "band_gap": d.get("band_gap"),
            "energy_above_hull": d.get("energy_above_hull"),
            "nsites": d.get("nsites"),
            "volume": d.get("volume"),
            "is_stable": d.get("is_stable"),
            "elements": ",".join(e.get("element", str(e)) if isinstance(e, dict) else str(e)
                                 for e in (d.get("elements") or [])),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Materials Project 数据拉取")
    ap.add_argument("--mode", choices=["mpapi", "rest"], default="mpapi")
    ap.add_argument("--key", required=True, help="MP API Key（免费注册获取）")
    ap.add_argument("--formula", default=DEFAULT_FORMULA)
    ap.add_argument("--limit", type=int, default=1000, help="rest 模式单页上限")
    ap.add_argument("--out", default=str(ROOT / "data/processed/mp_summary.csv"))
    args = ap.parse_args()

    print(f"模式={args.mode}  体系={args.formula}")
    if args.mode == "mpapi":
        rows = fetch_mpapi(args.key, args.formula)
    else:
        rows = fetch_rest(args.key, args.formula, args.limit)

    if not rows:
        print("未取回数据（检查 Key / 体系写法 / 配额）")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"取回 {len(df)} 条 -> {out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
