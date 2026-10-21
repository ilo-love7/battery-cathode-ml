# Changelog

本文件记录项目的所有重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.1.0] - 2026-10-21

### 修复（重要）

- **预测结果静默错误（严重）**：`predictor.py` 用「元素分数 → 统计特征」顺序拼特征矩阵，
  而带隙/形成能模型训练时是「统计特征 → 元素分数」，且调用 `to_numpy()` 绕过了 sklearn
  的列名校验，导致**任意化学式都返回同一组预测值**（README 旧示例中 LiFePO4 与 LiCoO2
  的 Egap 都是 1.752）。现改为严格按 `bundle["feature_names"]` 构造 DataFrame 再预测，
  列名缺失时直接报错。新增回归测试锁死该行为。
- **特征工程缺失值偏置**：含 NaN 属性（如稀有气体电负性）的成分，加权均值被 `nansum`
  稀释（LiHeO2 的 `EN_mean` 偏差约 25%）。现按非缺失权重重新归一化。
- **`run_pipeline.py` 语法错误**：`break` 写在 `if` 内导致整个脚本 `SyntaxError`。
- **Python 3.9 兼容**：`features.py` 的 `dict | None` 注解在 3.9 运行时报错，补
  `from __future__ import annotations`。
- **`train_voltage.py` 标签错位风险**：特征化失败样本会被丢弃，而 `y` 仍取全部行，
  一旦出现失败样本就会静默错位；现与特征矩阵同步剔除。
- **`models_voltage.pkl` 元数据错误**：`feature_names` 被误存为元素列表，现保存真实训练列。
- **`setup.py` 空包**：`find_packages(where="src")` 在扁平布局下返回空列表，
  `pip install -e .` 装不出任何模块；改为 `py_modules` + `package_dir`，并把 console
  script 指向真实存在的 `main()`。
- **`predictor.screen_candidates` 性能**：每个候选都重新 unpickle 数百 MB 模型（40 次），
  改为只加载一次；同时补上文档承诺的「理论容量 30%」评分项（旧实现总分只有 0.7）。
- **Notebook 导入错误**：`from screening import screen_candidates`（该函数在 `predictor.py`）；
  路径改为从项目根自动推导，不再依赖 cwd。
- **控制台编码崩溃**：Windows GBK 终端下 emoji / `²` 触发 `UnicodeEncodeError`，改为 ASCII 标记。

### 新增

- `src/download_data.py`：从 `ml.materialsproject.org` 下载两个 matbench 数据集，带
  完整性校验、断点文件清理、离线提示——解决「clone 下来跑不了训练」的问题。
- `tests/test_predictor.py`（10 个用例）：用桩模型锁死特征列顺序、批量预测、错误处理。
- `tests/test_voltage.py`（8 个用例）：理论容量基准值（LiFePO4/LiCoO2/LiMn2O4）、电压换算。
- `run_pipeline.py --check`：只做前置条件检查；缺数据时给出可执行提示而不是 traceback。
- `screening.py` 元素宇宙缓存（`results/training_universe.json`），省去每次重解析 2 万条化学式。
- 各训练/筛选脚本在缺数据或缺模型时返回明确错误码与修复命令。
- CI 新增「CLI 预测 smoke test」：断言不同化学式得到不同电压，防止特征错位类 bug 回归。

### 文档

- README 重写：补充数据获取与复现路径、真实的 API 输出示例、**已知限制**章节、
  深化方向表；删除不存在的链接（API 文档 / CODE_STYLE.md / FAQ.md）与 PyPI 安装说明。
- 明确标注电压指标来自**合成演示数据**，不构成科学结论。
- 项目报告 §8 复现指南补充下载与电压链路步骤。

### 变更

- 电压模型特征列顺序统一为「统计特征 → 元素分数」，与带隙/形成能模型一致
  （树模型对列顺序不敏感，指标基本不变：MAE 0.1926 → 0.1922 V）。
- 测试数量 16 → 39。

## [1.0.0] - 2026-10-20

### 新增

- 首个公开版本：带隙 / 形成能 / 平均电压三个成分级预测模型（RF + HGB 集成，5 折 OOF）。
- 33 维统计特征 + 元素分数特征；化学式解析器（支持括号、小数、等占位）。
- 候选正极筛选：稳定性 / 导电性 / 理论容量加权评分 + 域外与高分歧标记。
- 完整工程化：MIT License、Issue/PR 模板、GitHub Actions CI、Jupyter 教程、单元测试。
