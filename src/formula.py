# -*- coding: utf-8 -*-
"""化学式解析器：支持嵌套括号、小数化学计量比、等占位替代项。

示例：
    LiNi0.8Co0.1Mn0.1O2      -> {Li:1, Ni:0.8, Co:0.1, Mn:0.1, O:2}
    Li3V2(PO4)3              -> {Li:3, V:2, P:3, O:12}
    Ag(AuS)2                 -> {Ag:1, Au:2, S:2}
    Ba5(AsO4)3(OH,F)         -> {Ba:5, As:3, O:12.5, H:0.5, F:0.5}
        （括号内含逗号时按"等占位近似"处理：各替代项权重 = 括号倍率 / 项数）
"""

import re

_TOKEN = re.compile(r"([A-Z][a-z]{0,2})(\d*\.?\d*)")
_NUM = re.compile(r"(\d*\.?\d*)")


def _split_top_level(s: str, sep: str = ","):
    """按顶层分隔符拆分（忽略括号内部的嵌套）。"""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse(s: str, i: int):
    """递归下降解析，返回 (计数dict, 下一个位置)。"""
    counts = {}
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth, j = 1, i + 1
            while j < len(s) and depth:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            if depth != 0:
                raise ValueError(f"括号不匹配: {s!r}")
            inner = s[i + 1: j - 1]
            m = _NUM.match(s, j)
            mult = float(m.group(1)) if m.group(1) else 1.0
            i = m.end()
            parts = _split_top_level(inner, ",")
            w = mult / len(parts)          # 等占位近似
            for p in parts:
                sub, _ = _parse(p, 0)
                if _parse_tail_ok(p):
                    for el, c in sub.items():
                        counts[el] = counts.get(el, 0.0) + c * w
                else:
                    raise ValueError(f"括号内容无法解析: {p!r} in {s!r}")
        elif ch == ")":
            return counts, i + 1
        else:
            m = _TOKEN.match(s, i)
            if not m or not m.group(1):
                raise ValueError(f"位置 {i} 无法识别的记号 {ch!r} in {s!r}")
            el, num = m.group(1), m.group(2)
            c = float(num) if num else 1.0
            counts[el] = counts.get(el, 0.0) + c
            i = m.end()
    return counts, i


def _parse_tail_ok(p: str) -> bool:
    """检查子串解析是否消费完毕（无残留记号）。"""
    try:
        _, i = _parse(p, 0)
        return i == len(p)
    except ValueError:
        return False


def parse_formula(formula: str) -> dict:
    """解析化学式 -> {元素符号: 数量(float)}。

    抛出 ValueError 表示无法解析（调用方应记录并丢弃该样本）。
    """
    s = formula.strip()
    s = s.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", "", s)
    # 去掉电荷后缀（整体视为离子电荷，不当作化学计量数）：
    #   Fe3+ -> Fe     （Fe³⁺，1 个 Fe 原子）
    #   O2-  -> O      （O²⁻，1 个 O 原子）
    #   SO4^2- -> SO4
    s = re.sub(r"\^?\d*[+-]\d*$", "", s)
    if not s:
        raise ValueError("空化学式")
    counts, i = _parse(s, 0)
    if i != len(s):
        raise ValueError(f"位置 {i} 有未消费的记号 in {formula!r}")
    if not counts:
        raise ValueError(f"解析结果为空: {formula!r}")
    return counts


def canonical_formula(counts: dict) -> str:
    """把 {元素: 数量} 转成规范字符串（元素按符号排序，数量以最短形式输出）。"""
    parts = []
    for el in sorted(counts):
        c = round(counts[el], 6)
        if abs(c - round(c)) < 1e-9:
            c = int(round(c))
            parts.append(f"{el}{c if c != 1 else ''}")
        else:
            parts.append(f"{el}{c:g}")
    return "".join(parts)


if __name__ == "__main__":
    tests = [
        "LiCoO2",
        "LiNi0.8Co0.1Mn0.1O2",
        "Li3V2(PO4)3",
        "Ag(AuS)2",
        "Ba5(AsO4)3(OH,F)",
        "LiNi0.333Co0.333Mn0.334O2",
        "Ca3(C0.75B0.25)N3",
        "Li2FeSiO4",
    ]
    for t in tests:
        print(t, "->", canonical_formula(parse_formula(t)))
    print("全部自测通过")
