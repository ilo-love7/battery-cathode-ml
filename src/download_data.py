# -*- coding: utf-8 -*-
"""下载 matbench 公开数据集到 data/raw/（无需 Materials Project API Key）。

这两个文件是 train_gap.py / train_ef.py 的输入，官方托管在
``ml.materialsproject.org``，体积分别约 36 KB 与 4 MB（gzip）：

- ``matbench_expt_gap.json.gz``     实验带隙 4,604 条
- ``matbench_perovskites.json.gz``  MP DFT 钙钛矿形成能 18,928 条

用法::

    python src/download_data.py                 # 缺什么下什么
    python src/download_data.py --force         # 强制重下
    python src/download_data.py --timeout 120
    python src/download_data.py --base-url https://mirror.example/projects

若网络受限（例如本机 DNS 被阻断），可手动下载同名文件放进 data/raw/ 再运行
``python src/prepare_data.py``，效果等价。
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DEFAULT_BASE = "https://ml.materialsproject.org/projects"
FILES = ["matbench_expt_gap.json.gz", "matbench_perovskites.json.gz"]
# 低于该字节数视为下载不完整
MIN_BYTES = {
    "matbench_expt_gap.json.gz": 20_000,
    "matbench_perovskites.json.gz": 500_000,
}


def _looks_like_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            fh.read(1)
        return True
    except (OSError, EOFError):
        return False


def download_one(name: str, base_url: str, timeout: int, force: bool = False) -> bool:
    """下载单个文件；返回是否可用。"""
    dest = RAW_DIR / name
    if dest.exists() and not force:
        size = dest.stat().st_size
        if size >= MIN_BYTES.get(name, 1) and _looks_like_gzip(dest):
            print(f"[SKIP] {name} 已存在（{size / 1024:.0f} KB）")
            return True
        print(f"[WARN] {name} 不完整或损坏，重新下载 ...")

    url = f"{base_url.rstrip('/')}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[GET ] {url}")
    # 带上浏览器式 UA：部分 CDN 会对默认的 "Python-urllib/x.y" 返回 403
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; battery-cathode-ml/1.3; "
                      "+https://github.com/ilo-love7/battery-cathode-ml)",
        "Accept": "*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r        {done / 1048576:6.2f}/{total / 1048576:6.2f} MB", end="")
        print()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"[FAIL] {name}: {exc}")
        print("       可手动下载后放入 data/raw/：")
        print(f"       {url}")
        if "403" in str(exc) or "Forbidden" in str(exc):
            print("       提示：403 常见于云主机/CI 出口 IP 被 CDN 拦截；")
            print("             本仓库已随包提供清洗后的 data/processed/*.csv，")
            print("             可直接跳到 python src/train_gap.py && python src/train_ef.py。")
        return False

    size = tmp.stat().st_size
    if size < MIN_BYTES.get(name, 1) or not _looks_like_gzip(tmp):
        tmp.unlink(missing_ok=True)
        print(f"[FAIL] {name} 下载结果校验失败（{size} 字节）")
        return False

    tmp.replace(dest)
    print(f"[OK  ] {name} -> {dest}（{size / 1024:.0f} KB）")
    return True


def main(argv=None) -> int:
    from console import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="下载 matbench 公开数据集")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="数据托管前缀")
    ap.add_argument("--timeout", type=int, default=60, help="单文件超时（秒）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in FILES:
        ok &= download_one(name, args.base_url, args.timeout, args.force)

    if not ok:
        print("\n部分文件缺失。离线环境请手动下载并放入 data/raw/ 后重试。")
        return 1
    print("\n全部就绪。下一步: python src/prepare_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
