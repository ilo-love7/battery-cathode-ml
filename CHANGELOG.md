# Changelog

本文件记录项目的所有重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.3.0] - 2026-10-21

**核心变化：多目标（比容量）+ 跨口径外部验证。**

### 新增

- `src/train_capacity.py`：比容量模型（复用 `train_voltage.py`，`--target capacity_grav --tag capacity`）
  - 随机 5 折 OOF **MAE 25.2 mAh/g / R² 0.663**；GroupKFold **MAE 28.8 / R² 0.611**；
    中位数基线 MAD 47.8 mAh/g（相对降低约 47%）
  - 抽检：LiCoO₂ 预测 273.9（理论 273.8）、LiFePO₄ 165.0（理论 169.9）
- `src/prepare_mp_summary.py`：MP 摘要清洗为外部验证集
  （20,000 → 按成分去重保留最稳定相 → **8,942 条**，落盘 `data/processed/mp_summary_Li.csv.gz` 并入库）
- `src/validate_mp_external.py`：**跨口径**外部验证
  - 形成能：原始 MAE 2.205 eV/atom；系统偏移 **+2.410**；**去偏移 MAE 0.735 eV/atom**（基线 MAD 0.805）；**ρ 0.475**
  - 带隙：MAE 1.119 eV、ρ 0.332（模型训练于实验带隙，MP 为 DFT-PBE）
  - 明确记录为**域外推能力有限的负面结果**，作为引入结构模型（CGCNN）的直接证据
- `tests/test_multiobjective.py`（10 例）：容量训练参数注入、predictor 容量列、摘要清洗去重规则、外部验证指标（去偏移 MAE / Spearman）

### 变更

- `predictor` 支持容量模型：输出 `Capacity_ensemble_mAh_g` / `Capacity_spread`；
  候选筛选在缺 `nLi_extract` 时用模型容量参与评分（理论容量可用时仍优先）
- `train_voltage.py` 单位/标签按目标切换（V / mAh/g），绘图标题与打印不再写死「电压」
- `fetch_mp_battery.py` 修复摘要分页 bug（原实现循环调用同一 `search` 会重复取第一页），并加入 material_id 去重
- CI smoke 增加：真实数据训练容量模型 → MP 跨口径外部验证 → 产物校验
- README / 项目报告新增附录 D / 项目总结 同步；测试 86 → **96**；版本 1.2.0 → 1.3.0

## [1.2.0] - 2026-10-21

**核心变化：电压模型从合成演示数据切换到 Materials Project 真实数据。**

### 新增

- `src/fetch_mp_battery.py`：从 MP 官方 API 拉取两类真实数据
  - `materials.insertion_electrodes`（`working_ion=Li`，2,774 条）→ **真实 `average_voltage` / `capacity_grav`**
  - `materials.summary`（含 Li 材料）→ 供外部验证
  - API Key 读取优先级：`--key` > `MP_API_KEY` > `~/.mp_api_key`
- `src/prepare_mp_data.py`：清洗规则（0–6 V、化学式可解析、元素在表内、容量交叉校验），
  2,774 → **2,672 行 / 1,794 个唯一成分**
- `data/processed/mp_voltage.csv` **随仓库分发** → 没有 API Key 也能复现全部真实指标
- `src/validate_known_cathodes.py`：12 个文献常见商用正极的外部验证
  （**MAE 0.160 V、RMSE 0.212 V、Spearman ρ 0.842**；NMC111 误差 0.006 V、LFP 0.061 V）
- `tests/test_mp_data.py`：离线 fixture 覆盖清洗规则与 Key 读取优先级（6 例，不需要网络）

### 变更

- `train_voltage.py` 参数化：`--data` / `--tag` / `--target` / `--group-col`，
  并按分组列额外跑 GroupKFold，暴露「同框架不同锂含量」的重复效应
- 电压模型指标（真实数据）：随机 5 折 OOF **MAE 0.419 V / R² 0.682**；
  按 `framework_formula` 分组的 GroupKFold **MAE 0.481 V / R² 0.627**；
  中位数基线 MAD 0.679 V（模型相对基线降低约 38%）
- 合成数据结果保留为对照：`python src/train_voltage.py --data data/battery_voltage.csv --tag voltage_synthetic`
- `predictor` 新增 `Voltage_spread` 输出；数据来源提示改为真实数据
- README：指标表加入真实电压行与外部验证、新增 MP Key 配置与 CC BY 4.0 引用、
  已知限制与深化方向重写；`docs/项目报告.md` 新增附录 C（真实数据扩展）
- 测试 86 个；版本 1.1.1 → 1.2.0

### 未采用（已评估）

- 本机 D 盘的 26 套公开电池数据集（NASA/CALCE/MIT/Oxford/… 共 80 GB）均为**循环老化时序**，
  与「成分 → 材料性质」任务不同源，只能服务于寿命预测项目；不作为本项目补充数据。

## [1.1.1] - 2026-10-21

### 修复

- `screening.py` 文档字符串写「42 个候选」，实际数据为 40 个。
- 元素宇宙缓存无失效条件：重新生成 `data/processed/*.csv` 后仍会沿用旧宇宙，
  导致域内外标记错误。现缓存带数据指纹（文件大小），数据变化后自动重算。
- `predictor.py`：模型束损坏（如旧版 pkl 缺 `feature_names`）时整批预测直接抛异常，
  现改为记录 `*_error` 字段并继续处理其余材料。
- `run_pipeline.py --fast`：跳过形成能模型训练后，候选筛选仍会被执行并因缺模型失败，
  让整个流程报「部分失败」。现在检测模型是否就绪，未就绪则跳过筛选并说明原因。

### 新增

- `tests/test_element_data.py`：元素表硬约束（118 个元素、Z 连续、符号唯一、参考质量/电负性、取值范围）。
- `tests/test_docs.py`：Markdown 相对链接与文档中引用的 `src/*.py` 必须真实存在（防文档腐烂）。
- `tests/test_formula.py`：`canonical_formula` → `parse_formula` 往返一致性。
- `predictor` 测试：模型束损坏时批量预测不中断。

### 变更

- 测试数量 39 → 80。
- 版本号 1.1.0 → 1.1.1。

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
