# -*- coding: utf-8 -*-
"""控制台编码兼容层。

问题：Windows 控制台默认是本地代码页（中文系统 cp936，英文系统 cp1252）。
项目输出含中文与少量特殊字符（Å、≤、×、→ 等），在 cp1252 下会直接抛
``UnicodeEncodeError: 'charmap' codec can't encode characters``，把 CLI 打挂
（在 GitHub Actions 的 Windows Runner 上就是如此）。

做法：把 stdout/stderr 的**错误处理**改成 ``replace``（必要时顺带切到 UTF-8），
让无法编码的字符退化为 ``?`` 而不是崩溃。对终端显示的影响仅限于少数符号，
但换来的是任何语言环境下脚本都不会因为"打印一个箭头"而失败。

用法（放在各 CLI 的 ``main()`` 开头）::

    from console import enable_utf8_stdout
    enable_utf8_stdout()
"""

from __future__ import annotations

import sys


def enable_utf8_stdout(force_utf8: bool = False) -> None:
    """让标准输出/错误在任意代码页下都不会因编码失败而崩溃。

    Args:
        force_utf8: True 时强制切到 UTF-8（终端若仍是 GBK，会显示乱码但不会崩）；
                    默认只把错误处理设为 ``replace``，尽量保持终端原有编码。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            if force_utf8:
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except (ValueError, OSError):     # 已被重定向或不可重配置：忽略
            pass


if __name__ == "__main__":
    enable_utf8_stdout()
    print("中文与特殊符号测试: Å ≤ × → ± 90°")
    print("console guard OK")