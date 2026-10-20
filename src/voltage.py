# -*- coding: utf-8 -*-
"""电压与容量计算工具（供筛选与 MP 数据扩展流程复用）。

原理:
- 理论比容量: 每脱嵌 1 mol Li+ 转移 1 mol 电子 = F 库仑 = F/3.6 mAh，
  故 Q = n_Li * F / 3.6 / M(formula)  (mAh/g)，M 为化学式摩尔质量。
- 平均电压: 嵌锂反应 nLi + Host -> Li_nHost 的吉布斯自由能变 ΔG，
  V_avg = -ΔG / (n_Li * e) = ΔG[eV, per formula unit] / n_Li  (V)。
  使用 Materials Project 形成能近似 ΔG 时（忽略熵项与电压曲线形状），
  ΔG ≈ E_form(脱锂相, per f.u.) + n_Li * E_form(Li 金属, =0) - E_form(嵌锂相, per f.u.)。
  Li 金属为元素参考态，形成能恒为 0，因此公式中无需额外项。

注意: MP 形成能是 0 K DFT 生成焓，由此得到的电压为"形成能近似平均电压"，
      系统性偏差通常在 ±0.1–0.3 V（与 GGA/GGA+U 误差相关），仅用于横向比较。
"""

import sys
from pathlib import Path

F_FARADAY = 96485.33212  # C/mol


def theoretical_capacity_mAh_g(counts: dict, n_li: float) -> float:
    """{元素:数量} + 可脱嵌 Li 数 -> 理论比容量 (mAh/g)。

    n_li 为每化学式单位可脱嵌的 Li 摩尔数（如 LiFePO4 取 1，
    Li3V2(PO4)3 在 1.5–4.3 V 窗口取约 2）。
    """
    from features import ELEMENTS
    molar = sum(c * ELEMENTS[el]["mass"] for el, c in counts.items())
    if molar <= 0:
        raise ValueError("摩尔质量非正")
    return n_li * F_FARADAY / 3.6 / molar


def avg_voltage_from_formation_energies(e_form_delith_per_fu: float,
                                        e_form_lith_per_fu: float,
                                        n_li: float) -> float:
    """由形成能（每化学式单位, eV）估算平均电压 (V)。

    ΔG ≈ E_delith - E_lith （Li 金属参考态形成能为 0）
    V_avg = ΔG / n_li
    """
    if n_li <= 0:
        raise ValueError("n_li 必须为正")
    return (e_form_delith_per_fu - e_form_lith_per_fu) / n_li


def per_atom_to_per_fu(e_form_per_atom: float, n_sites: int) -> float:
    """把 eV/atom 的形成能换算为 eV/化学式单位。"""
    if n_sites <= 0:
        raise ValueError("n_sites 必须为正")
    return e_form_per_atom * n_sites


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from formula import parse_formula

    # 常见正极理论容量自检
    for f, n in [("LiFePO4", 1), ("LiCoO2", 1), ("Li3V2(PO4)3", 2), ("Li2FeSiO4", 1)]:
        print(f"{f:15s} nLi={n}  Q={theoretical_capacity_mAh_g(parse_formula(f), n):7.1f} mAh/g")

    # 形成能近似电压示例（数值为演示用假想值）
    v = avg_voltage_from_formation_energies(
        e_form_delith_per_fu=-24.10, e_form_lith_per_fu=-26.05, n_li=1)
    print(f"示例平均电压: {v:.2f} V  （FePO4/FePO4Li 体系, 演示值）")
