"""徽章一致性测试：README 封面上的数字必须等于 `results/*.json` 的实测值。

为什么值得一个测试：徽章是仓库的"封面数字"。产物重跑之后如果忘了同步徽章，
封面就在说谎，而 Markdown 不会报错——这是文档腐烂里最贵的一种，因为它出现在
任何人第一眼看到的地方。

两个已知陷阱（本项目都踩过）：

1. 徽章值本身含括号，例如 ``0.352%20V%20(two--tower%20CGCNN)``。用 ``[^)]*?``
   匹配会在第一个 ``)`` 处提前截断；正确写法是锚定 label 前缀，再贪婪到颜色词：
   ``img\\.shields\\.io/badge/<label>-.*?-(brightgreen|green|blue|...)\\)``。
2. ``pytest --collect-only -q`` 只按文件打印计数（``tests/test_x.py: 12``），
   必须求和才是总数（或读 ``N tests collected`` 汇总行）。

边界——这个测试**不能**证明什么：

- ``tests`` 徽章只校验**总数**。括号里的 ``199 pass + 1 skip`` 拆分取决于运行时
  产物是否齐全，只有真正跑一遍才知道，所以这里只校验 ``pass + skip == total``
  的算术自洽，不校验拆分本身。
- ``license`` 徽章不是从 ``results/*.json`` 派生的，只由
  ``test_all_badges_are_registered_and_parseable`` 保证它存在且可解析。
- 只覆盖 ``test`` 这个 CI 作业里读到的**已提交**产物；``smoke`` 作业会重训一批
  模型并覆盖 ``results/metrics_voltage.json`` 等文件，但它们不在本测试的校验
  范围内。
"""

import json
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results"

# shields.io 的颜色词，用作"值到哪里结束"的右锚点
COLORS = (
    "brightgreen",
    "green",
    "yellowgreen",
    "yellow",
    "orange",
    "red",
    "blue",
    "blueviolet",
    "lightgrey",
    "success",
    "important",
    "informational",
    "critical",
    "inactive",
)

# 已登记的徽章 label（README 里写成 URL 编码形式）→ 便于阅读的名字。
# 新增徽章时必须登记，否则 test_all_badges_are_registered_and_parseable 会红。
BADGE_LABELS = {
    "voltage%20MAE": "voltage MAE",
    "formation%20energy": "formation energy",
    "structure%20coverage": "structure coverage",
    "tests": "tests",
    "license": "license",
}

BADGE_RE = re.compile(r"img\.shields\.io/badge/")
TESTS_VALUE_RE = re.compile(
    r"(\d+)\s*\((\d+)\s+pass\s*\+\s*(\d+)\s+(?:conditional\s+)?skip\)"
)


def readme_text():
    return README.read_text(encoding="utf-8")


def badge_value(label, text=None):
    """返回 shields.io 徽章的值（URL 解码，并把 ``--`` 还原成字面量 ``-``）。

    正则刻意锚定 label 前缀再非贪婪到颜色词，避免徽章值里的括号把匹配截断。
    """
    text = readme_text() if text is None else text
    pattern = (
        r"img\.shields\.io/badge/"
        + re.escape(label)
        + r"-(.*?)-("
        + "|".join(COLORS)
        + r")\)"
    )
    match = re.search(pattern, text)
    assert match, f"README 里找不到 label 为 {label!r} 的 shields.io 徽章"
    # shields.io 用 `--` 转义字面量连字符（two--tower → two-tower）
    return urllib.parse.unquote(match.group(1)).replace("--", "-")


def load_json(name):
    path = RESULTS / name
    assert path.exists(), (
        f"缺少产物 results/{name}；本测试依赖随仓库提交的 results/*.json，"
        "不要在干净副本上删掉它们"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def collected_test_total():
    """求和当前仓库收集到的测试总数。

    ``--collect-only -q`` 在 pytest >= 7 只按文件打印计数（陷阱 2），因此先找
    ``N tests collected`` 汇总行，找不到再退回逐文件求和。子进程环境里剔除
    pytest-cov / coverage 注入的变量，避免把外层作业的覆盖率状态带进来。
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COV_CORE", "COVERAGE_", "PYTEST_"))
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"收集测试失败（exit {proc.returncode}）：\n{proc.stdout}\n{proc.stderr}"
    )
    out = proc.stdout + proc.stderr
    summary = re.search(r"(\d+)\s+tests?\s+collected", out)
    if summary:
        return int(summary.group(1))
    per_file = [int(n) for n in re.findall(r"^\S+\.py:\s+(\d+)\s*$", out, re.M)]
    assert per_file, f"无法从 pytest --collect-only 输出解析测试数：\n{out}"
    return sum(per_file)


def test_all_badges_are_registered_and_parseable():
    """徽章数量与登记表一致，且每个徽章的值都能解析出来。"""
    text = readme_text()
    found = len(BADGE_RE.findall(text))
    assert found == len(BADGE_LABELS), (
        f"README 里有 {found} 个 shields.io 徽章，登记表里有 {len(BADGE_LABELS)} 个。"
        "新增徽章必须登记到 BADGE_LABELS，否则它不受本测试保护"
    )
    for label in BADGE_LABELS:
        value = badge_value(label, text)
        assert value, f"徽章 {label} 解析出的值为空"


def test_voltage_badge_matches_results_json():
    """two-tower CGCNN 电压 MAE ↔ results/metrics_cgcnn_voltage.json。"""
    mae = load_json("metrics_cgcnn_voltage.json")["test"]["MAE_V"]
    expected = f"{mae:.3f} V (two-tower CGCNN)"
    actual = badge_value("voltage%20MAE")
    assert actual == expected, (
        f"电压徽章写着 {actual!r}，产物实测为 {mae} V（应为 {expected!r}）"
    )


def test_formation_energy_badge_matches_results_json():
    """CGCNN 形成能 MAE ↔ results/metrics_cgcnn.json。"""
    mae = load_json("metrics_cgcnn.json")["test"]["MAE_eV_cell"]
    expected = f"{mae:.4f} eV/cell"
    actual = badge_value("formation%20energy")
    assert actual == expected, (
        f"形成能徽章写着 {actual!r}，产物实测为 {mae} eV/cell（应为 {expected!r}）"
    )


def test_structure_coverage_badge_matches_results_json():
    """结构覆盖率 ↔ results/metrics_screening_struct.json。"""
    metrics = load_json("metrics_screening_struct.json")
    expected = f"{metrics['n_ranked_with_structure']}/{metrics['n_candidates']}"
    actual = badge_value("structure%20coverage")
    assert actual == expected, (
        f"结构覆盖徽章写着 {actual!r}，产物实测为 {expected!r}"
    )


def test_tests_badge_matches_collected_total():
    """tests 徽章的总数 == 实际收集到的测试数，且括号内拆分自洽。"""
    value = badge_value("tests")
    match = TESTS_VALUE_RE.fullmatch(value)
    assert match, f"tests 徽章格式应为 `N (P pass + S skip)`，实际为 {value!r}"
    badge_total, passed, skipped = (int(g) for g in match.groups())
    assert passed + skipped == badge_total, (
        f"tests 徽章内部不自洽：{passed} pass + {skipped} skip != {badge_total}"
    )
    total = collected_test_total()
    assert badge_total == total, (
        f"README 徽章写着 {badge_total} 个测试，实际收集到 {total} 个。"
        "新增或删除测试后必须同步 README 徽章与正文里的测试数"
    )