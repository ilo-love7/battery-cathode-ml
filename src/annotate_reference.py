# -*- coding: utf-8 -*-
"""参考性能表的**逐条溯源**（v1.19.0）。

### 为什么

v1.18.0 的溯源只是"规则分级"（厂商/综述/估算三档），并不能回答
"这个数字到底出自哪篇文献"。本轮把每条记录绑定到**具体引用**，并用 **Crossref**
核验 DOI（`api.crossref.org`）—— 已核验的 DOI 直接写入表，未核验的明确标出。

### 已核验的锚点文献（Crossref 查询确认）

| 家族 | 引用 | DOI |
|---|---|---|
| 橄榄石 LiMPO4（LFP/LMP/LCP/LNP/LMFP） | Padhi, Nanjundaswamy, Goodenough, *J. Electrochem. Soc.* **144**, 1188 (1997) | `10.1149/1.1837571` |
| 层状 LiCoO2 | Mizushima et al., *Mater. Res. Bull.* **15**, 783 (1980) | `10.1016/0025-5408(80)90012-4` |
| 尖晶石 LiMn2O4 | Thackeray et al., *Mater. Res. Bull.* **18**, 461 (1983) | `10.1016/0025-5408(83)90138-1` |
| 层状 NMC（333 起） | Ohzuku & Makimura, *Chem. Lett.* **30**, 642 (2001) | `10.1246/cl.2001.642` |
| 无序岩盐 DRX | Lee et al., *Science* **343**, 519 (2014) | `10.1126/science.1246432` |
| 氟硫酸盐 tavorite/triplite | Barpanda et al., *Nat. Mater.* **10**, 772 (2011) | `10.1038/nmat3093` |
| 正极总览综述 | Manthiram, *Nat. Commun.* **11**, 1550 (2020) | `10.1038/s41467-020-15355-0` |
| 层状 NMC（x = 0.5 / 0.6 / 0.8 对比） | Noh, Youn, Yoon & Sun, *J. Power Sources* **233**, 121 (2013) | `10.1016/j.jpowsour.2013.01.063` |
| 层状 LiMnO2 | Armstrong & Bruce, *Nature* **381**, 499 (1996) | `10.1038/381499a0` |
| 硅酸铁锂 Li2FeSiO4 | Nytén et al., *Electrochem. Commun.* **7**, 156 (2005) | `10.1016/j.elecom.2004.11.008` |
| 磷酸钒锂 Li3V2(PO4)3 | Huang, Yin, Kerr, Taylor & Nazar, *Adv. Mater.* **14**, 1525 (2002) | `10.1002/1521-4095(20021104)14:21<1525::AID-ADMA1525>3.0.CO;2-3` |
| 二硫化钒 VS2 | Murphy, Di Salvo & Carides, *J. Solid State Chem.* **29**, 339 (1979) | `10.1016/0022-4596(79)90190-7` |
| 二硫化钼 MoS2 | Whittingham & Gamble, *Mater. Res. Bull.* (1975) | `10.1016/0025-5408(75)90006-9` |
| 氟磷酸钒锂 LiVPO4F | Barker, Saidi & Swoyer, *J. Electrochem. Soc.* **150**, A1394 (2003) | `10.1149/1.1609998` |
| 磷酸氟铁锂 Li2FePO4F | Ellis, Makahnouk, Makimura, Toghill & Nazar, *Nat. Mater.* (2007) | `10.1038/nmat2007` |
| 硅酸锰锂 Li2MnSiO4 | Dominko et al., *Electrochem. Commun.* (2006) | `10.1016/j.elecom.2005.11.010` |
| 硅酸钴锂 Li2CoSiO4 | Lyness, Delobel, Armstrong & Bruce, *Chem. Commun.* (2007) | `10.1039/b711552k` |
| 层状钒氧化物 LiV3O8 | Panero, Pasquali & Pistoia, *J. Electrochem. Soc.* (1983) | `10.1149/1.2119923` |
| 高压尖晶石 LiNi0.5Mn1.5O4 | Zhong, Bonakdarpour, Zhang, Gao & Dahn, *J. Electrochem. Soc.* **144**, 205 (1997) | `10.1149/1.1837386` |
| 高压尖晶石 LiCoMnO4 | Kawai, Nagata, Kageyama, Tukamoto & West, *Electrochim. Acta* (1999) | `10.1016/S0013-4686(99)00213-3` |

`citation_verified` 的含义是**"该文献是否直接覆盖该材料"**（不是"是否有引用"）：

- `True`：引用即该材料的奠基/权威文献（如 LFP → Padhi 1997）；
- `False`：只引用了**家族或综述**文献（如 NMC811 → Ohzuku 2001 的 NMC 家族 + Manthiram 2020 综述），
  DOI 仍在，但**不宣称已逐条核验该材料的实测数值**。

逐条核验每一篇原始文献（含具体表格页码）仍是本项目可信度的天花板。

用法::

    python src/annotate_reference.py            # 写入 citation / doi / citation_verified
    python src/annotate_reference.py --check    # 只报告溯源覆盖情况
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference_cathodes.csv"

PADHI = ("Padhi, Nanjundaswamy & Goodenough, J. Electrochem. Soc. 144, 1188 (1997)",
         "10.1149/1.1837571")
MIZUSHIMA = ("Mizushima, Jones, Wiseman & Goodenough, Mater. Res. Bull. 15, 783 (1980)",
             "10.1016/0025-5408(80)90012-4")
THACKERAY = ("Thackeray, David, Bruce & Goodenough, Mater. Res. Bull. 18, 461 (1983)",
             "10.1016/0025-5408(83)90138-1")
OHZUKU = ("Ohzuku & Makimura, Chem. Lett. 30, 642 (2001)",
          "10.1246/cl.2001.642")
LEE = ("Lee et al., Science 343, 519 (2014)", "10.1126/science.1246432")
BARPANDA = ("Barpanda et al., Nat. Mater. 10, 772 (2011)", "10.1038/nmat3093")
MANTHIRAM = ("Manthiram, Nat. Commun. 11, 1550 (2020)", "10.1038/s41467-020-15355-0")
WHITTINGHAM = ("Whittingham, Science 192, 1126 (1976)",
               "10.1126/science.192.4244.1126")
THACKERAY2007 = ("Thackeray et al., J. Mater. Chem. 17, 3112 (2007)",
                 "10.1039/b702425h")
OHZUKU1991 = ("Ohzuku et al., J. Electrochem. Soc. 138, 2207 (1991)"
              "（Crossref 收录为 ChemInform 摘要）", "10.1002/chin.199345021")
REVIEW_ONLY = ("综述汇总（Manthiram 2020 等，未逐条核验原始文献）", "")
# v1.25.0 新增：Crossref 实测核验的锚点（title / journal / year 逐项对得上）
NOH2013 = ("Noh, Youn, Yoon & Sun, J. Power Sources 233, 121 (2013)"
           "（NMC x = 0.5 / 0.6 / 0.8 对比）",
           "10.1016/j.jpowsour.2013.01.063")
ARMSTRONG1996 = ("Armstrong & Bruce, Nature 381, 499 (1996)",
                 "10.1038/381499a0")
NYTEN2005 = ("Nytén, Abouimrane, Armand, Gustafsson & Thomas,"
             " Electrochem. Commun. 7, 156 (2005)",
             "10.1016/j.elecom.2004.11.008")
HUANG2002 = ("Huang, Yin, Kerr, Taylor & Nazar, Adv. Mater. 14, 1525 (2002)"
             "（Li3V2(PO4)3 原创正极工作）",
             "10.1002/1521-4095(20021104)14:21<1525::AID-ADMA1525>3.0.CO;2-3")
MURPHY1979 = ("Murphy, Di Salvo & Carides, J. Solid State Chem. 29, 339 (1979)"
              "（VS2 锂嵌入）", "10.1016/0022-4596(79)90190-7")
WHITTINGHAM1975 = ("Whittingham & Gamble, Mater. Res. Bull. (1975)"
                   "（过渡金属二硫属化物锂嵌入，含 MoS2）",
                   "10.1016/0025-5408(75)90006-9")
BARKER2003 = ("Barker, Saidi & Swoyer, J. Electrochem. Soc. 150, A1394 (2003)",
              "10.1149/1.1609998")
ELLIS2007 = ("Ellis, Makahnouk, Makimura, Toghill & Nazar, Nat. Mater. (2007)",
             "10.1038/nmat2007")
DOMINKO2006 = ("Dominko et al., Electrochem. Commun. (2006)"
               "（Li2MnSiO4 / Li2FeSiO4 结构与电化学）",
               "10.1016/j.elecom.2005.11.010")
LYNESS2007 = ("Lyness, Delobel, Armstrong & Bruce, Chem. Commun. (2007)",
              "10.1039/b711552k")
PANERO1983 = ("Panero, Pasquali & Pistoia, J. Electrochem. Soc. (1983)"
              "（Li/Li1+xV3O8 电池）", "10.1149/1.2119923")
ZHONG1997 = ("Zhong, Bonakdarpour, Zhang, Gao & Dahn,"
             " J. Electrochem. Soc. 144, 205 (1997)", "10.1149/1.1837386")
KAWAI1999 = ("Kawai, Nagata, Kageyama, Tukamoto & West,"
             " Electrochim. Acta (1999)（Li2Co1+xMn3-xO8 固溶体含 LiCoMnO4）",
             "10.1016/S0013-4686(99)00213-3")

# 材料 -> (引用, DOI, 是否核验)
CITATIONS = {
    # 橄榄石家族：Padhi 1997（该文即 LiMPO4 家族的奠基工作）
    "LiFePO4": (PADHI, True), "LiMnPO4": (PADHI, True), "LiCoPO4": (PADHI, True),
    "LiNiPO4": (PADHI, True), "LiMn0.6Fe0.4PO4": (PADHI, True),
    "LiMn0.8Fe0.2PO4": (PADHI, True),
    "LiCoO2": (MIZUSHIMA, True),
    "LiMn2O4": (THACKERAY, True),
    # NMC 家族：Ohzuku 2001 首次报道 LiNi1/3Mn1/3Co1/3O2
    "LiNi0.333Co0.333Mn0.334O2": (OHZUKU, True),
    "LiNi0.5Co0.2Mn0.3O2": (NOH2013, True),
    "LiNi0.6Co0.2Mn0.2O2": (NOH2013, True),
    "LiNi0.8Co0.1Mn0.1O2": (NOH2013, True),
    "LiNi0.9Co0.05Mn0.05O2": (OHZUKU, False),
    "LiNi0.4Co0.2Mn0.4O2": (OHZUKU, False),
    "LiNi0.4Co0.3Mn0.3O2": (OHZUKU, False),
    "LiNi0.8Co0.15Al0.05O2": (MANTHIRAM, False),
    "LiMnO2": (ARMSTRONG1996, True),
    "LiCrO2": (MANTHIRAM, False),
    "LiNi0.5Mn1.5O4": (ZHONG1997, True),
    "LiCoMnO4": (KAWAI1999, True),
    "LiV3O8": (PANERO1983, True),
    "LiFeSO4F": (BARPANDA, True),
    "LiVPO4F": (BARKER2003, True),
    "Li2FePO4F": (ELLIS2007, True),
    "LiFeP2O7": (MANTHIRAM, False),
    "Li3V2(PO4)3": (HUANG2002, True),
    "Li2FeSiO4": (NYTEN2005, True),
    "Li2MnSiO4": (DOMINKO2006, True),
    "Li2CoSiO4": (LYNESS2007, True),
    "Li1.2Mn0.4Ti0.4O2": (LEE, True),
    "Li1.2Mn0.4Zr0.4O2": (LEE, True),
    "Li1.3Mn0.4Nb0.3O2": (LEE, True),
    "Li1.2Mn0.4Nb0.4O2": (LEE, True),
    # 层状硫化物：Whittingham 1976 Science（插层化学的奠基工作，TiS2 为主）
    "TiS2": (WHITTINGHAM, True), "VS2": (MURPHY1979, True),
    "MoS2": (WHITTINGHAM1975, True),
    # 富锂锰基：Thackeray 2007 J. Mater. Chem.
    "Li1.2Ni0.13Co0.13Mn0.54O2": (THACKERAY2007, True),
    "Li1.2Ni0.2Mn0.6O2": (THACKERAY2007, True),
    "LiNiO2": (OHZUKU1991, True),
}

VERIFIED_DOIS = {PADHI[1], MIZUSHIMA[1], THACKERAY[1], OHZUKU[1], LEE[1],
                 BARPANDA[1], MANTHIRAM[1], WHITTINGHAM[1], THACKERAY2007[1],
                 OHZUKU1991[1], NOH2013[1], ARMSTRONG1996[1], NYTEN2005[1],
                 HUANG2002[1], MURPHY1979[1], WHITTINGHAM1975[1], BARKER2003[1],
                 ELLIS2007[1], DOMINKO2006[1], LYNESS2007[1], PANERO1983[1],
                 ZHONG1997[1], KAWAI1999[1]}


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="参考表逐条溯源")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--check", action="store_true", help="只报告覆盖情况，不写文件")
    args = ap.parse_args(argv)

    p = Path(args.reference)
    if not p.exists():
        print(f"[ERROR] 找不到 {p}")
        return 2
    df = pd.read_csv(p, encoding="utf-8-sig")

    cites, dois, verified = [], [], []
    for f in df["formula"].astype(str):
        (ref, doi), ok = CITATIONS.get(f, (REVIEW_ONLY, False))
        cites.append(ref)
        dois.append(doi)
        verified.append(bool(ok))
    df["citation"] = cites
    df["doi"] = dois
    df["citation_verified"] = verified
    # 不确定度：已核验的家族给 ±10%，未核验维持 ±20%
    if "uncertainty_pct" in df.columns:
        df["uncertainty_pct"] = [10 if v else 20 for v in verified]
        if df["uncertainty_pct"].notna().all():
            df.loc[df["source_type"] == "本文估算（文献区间中值）", "uncertainty_pct"] = 25

    n_ver = int(df["citation_verified"].sum())
    n_doi = int((df["doi"].astype(str) != "").sum())
    print(f"参考表 {len(df)} 条 | 已核验 {n_ver} 条 | 带 DOI {n_doi} 条 "
          f"| 覆盖率 {n_ver/len(df)*100:.0f}%")
    print("\n引用分布:")
    print(df["citation"].value_counts().to_string())
    print("\n未核验条目（应诚实标注）:")
    miss = df.loc[~df["citation_verified"], "formula"].tolist()
    print("  " + ", ".join(miss[:12]) + (" ..." if len(miss) > 12 else ""))

    if not args.check:
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())