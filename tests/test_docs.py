"""文档一致性测试：Markdown 相对链接与 README 中提到的脚本都必须真实存在。

文档腐烂（链接指向已删除文件、命令引用不存在的脚本）不会让代码报错，
但会直接误导使用者，所以在 CI 里固化成测试。
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MD_FILES = sorted(
    list(ROOT.glob("*.md"))
    + list((ROOT / "docs").glob("*.md"))
    + list((ROOT / ".github").glob("*.md"))
    + list((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md"))
)

LINK_RE = re.compile(r"\]\(([^)#]+?)(?:#[^)]*)?\)")
SCRIPT_RE = re.compile(r"python\s+(src/[\w./-]+\.py)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def test_markdown_files_found():
    assert len(MD_FILES) >= 5


@pytest.mark.parametrize("md", MD_FILES, ids=[p.name for p in MD_FILES])
def test_relative_links_resolve(md):
    text = md.read_text(encoding="utf-8")
    broken = []
    for match in LINK_RE.finditer(text):
        target = match.group(1).strip()
        if target.startswith(SKIP_PREFIXES):
            continue
        candidate = (md.parent / target).resolve()
        if not candidate.exists():
            broken.append(target)
    assert not broken, f"{md.name} 存在失效链接: {broken}"


@pytest.mark.parametrize("md", MD_FILES, ids=[p.name for p in MD_FILES])
def test_referenced_scripts_exist(md):
    text = md.read_text(encoding="utf-8")
    missing = sorted({m.group(1) for m in SCRIPT_RE.finditer(text)
                      if not (ROOT / m.group(1)).exists()})
    assert not missing, f"{md.name} 引用了不存在的脚本: {missing}"
