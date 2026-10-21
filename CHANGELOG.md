# Changelog

本文件记录项目的所有重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.5.2] - 2026-10-21

### 修复

- **v1.5.1 的回归测试本身有缺陷，导致 CI 全红**：
  - `screening.py` / `validate_known_cathodes.py` 没有 argparse，传 `--help` 会**真的执行流程**，
    在无模型的干净仓库里必然失败。现改为**自动发现**所有含 `argparse.ArgumentParser` 的 CLI
    （并要求至少 10 个，防漏测），只对这些脚本跑 `--help`。
  - 测试里误用未导入的 `io.open`，改为 `open()`。
- 教训记录：新增测试必须在**干净副本**上验证（CI 环境没有 `results/*.pkl`），
  本地"能过"不代表 CI 能过。

### 变更

- 版本 1.5.1 → 1.5.2（测试数仍为 130）

## [1.5.1] - 2026-10-21

### 修复

- **Windows 控制台编码崩溃（CI 上暴露）**：GitHub Actions 的 Windows Runner
  （以及启用英文区域设置的 Windows 用户）代码页无法编码中文与 `Å ≤ × →` 等字符，
  CLI 一打印就抛 `UnicodeEncodeError: 'charmap' codec can't encode characters`。
  - 新增 `src/console.py`：把 stdout/stderr 的错误处理设为 `replace`（必要时可切 UTF-8），
    无法编码的字符退化为 `?` 而不是让脚本崩掉；已在全部 19 个 CLI 的 `main()` 开头接入。
  - `tests/test_cgcnn.py` 增加回归测试：在 `PYTHONIOENCODING=gbk` 下跑 `console.py` 与
    5 个 CLI 的 `--help`，断言退出码为 0（此前 Windows CI 正是死在这里）。
  - CLI 接口测试改为给子进程注入 `PYTHONIOENCODING=utf-8` + 父进程 `errors="replace"`，
    使测试与终端语言环境解耦。

### 变更

- 测试 128 → **130**；版本 1.5.0 → 1.5.1

## [1.5.0] - 2026-10-21

**核心变化：自包含 CGCNN 结构图神经网络落地，形成能进入 0.04 eV/胞 量级。**

### 新增

- `src/cgcnn_data.py`：晶体图构建（纯 NumPy，可在无 torch 环境测试）
  - 节点特征：原子序数 one-hot（95 维）；边特征：距离高斯展开（0–8 Å / 0.2 Å → 41 维）
  - 周期最小镜像近邻（≤8 Å、每原子 ≤12 个），边双向存储
- `src/train_cgcnn.py`：**自包含 CGCNN 实现**（无需克隆外部仓库）
  - 门控卷积 `x_i' = x_i + Σ_j σ(BN(W_f·[x_i‖x_j‖e_ij])) ⊙ softplus(BN(W_s·…))`，
    扁平边索引 + `index_add_` 聚合，晶体内平均池化，目标标准化，ReduceLROnPlateau + 早停
  - 结果（RTX 4050 Laptop 6 GB，120 轮 / **210 s** / 97.5k 参数，8:1:1 划分）

    | 模型 | test MAE (eV/胞) | MAE (eV/atom) |
    |---|---|---|
    | 成分基线 | 0.4837 | 0.0967 |
    | 成分 + 21 维结构描述符（CPU） | 0.1929 | 0.0386 |
    | **CGCNN（本实现）** | **0.0401** | **0.0080** |
    | matbench 榜首 coGN（参考） | 0.0269 | 0.0054 |

  - train/val/test = 0.0269 / 0.0384 / 0.0401 eV/胞，三者一致，无过拟合迹象
- `tests/test_cgcnn.py`（17 例）：图形状、双向边、无自环、one-hot 正确性、截断半径、
  max_neighbors 上限、可复现性、非法结构/未知元素返回 None、占位取主元素、CLI 接口可达

### 变更

- `setup.py` 新增可选依赖 `[gpu] = torch>=2.0`；模块表加入 `cgcnn_data` / `train_cgcnn`
- README 新增「GPU 结构模型（CGCNN）」小节（环境搭建、训练命令、结果与局限）
- 项目报告新增附录 F（CGCNN 方法、配置、结果、复现与下一步）
- 测试 111 → **128**；版本 1.4.2 → 1.5.0

### 已知局限

- CGCNN 的输入是**晶体结构**而非化学式：对任意候选式预测需先生成/检索结构，**尚未接入预测 CLI**；
- 单次随机划分（与 matbench 口径一致），未做 5 折交叉验证；
- GPU 训练不进 CI（CI 只覆盖图构建与 CLI 接口）。

## [1.4.2] - 2026-10-21

### 修复

- **冒烟抽样会覆盖全量产物**：`train_ef_struct.py --subset` 默认复用 tag `ef_struct`，
  导致 `results/oof_ef_struct.csv`（18,928 行）与 `metrics_ef_struct.json` 被 4,000 条抽样结果覆盖。
  现在 `--subset` 未显式指定 `--tag` 时自动改用 `ef_struct_smoke`，全量产物不会被污染；
  CI 也相应改为校验/上传 `*_smoke` 制品。
- 已用全量运行恢复 `results/metrics_ef_struct.json`、`oof_ef_struct.csv`、
  `parity_ef_struct.png`、`importance_ef_struct.png`（数字与附录 E 一致）。

## [1.4.1] - 2026-10-21

### 变更

- CI 的结构感知实验改用冒烟抽样（`--subset 4000 --skip-group`，本机 62 s，全量约 4 分钟），
  缩短 smoke job 时长；新参数只影响 CI 冒烟，报告与 README 中的数字仍来自全量运行。
- `train_ef_struct.py` 新增 `--subset`，并在 metrics JSON 中标记 `mode: subset-smoke|full`，
  避免抽样指标被误当成正式结果。

## [1.4.0] - 2026-10-21

**核心变化：引入结构信息突破成分上限 + 把「分歧度」升级为有覆盖率保证的预测区间。**

### 新增

- `src/structure_features.py`：21 维结构描述符（晶格参数/体积、近邻距离统计、原子对距离统计、
  按元素分组的近邻距离统计），全部对周期边界取最小镜像距离，纯 CPU
- `src/prepare_struct_features.py`：把 matbench_perovskites 的 18,928 个结构固化为
  `data/processed/perovskites_struct.csv.gz`（1.2 MB，随仓库分发 → CI/离线可复现）
- `src/train_ef_struct.py`：**特征集对照实验**（同一数据、同一超参数、同一协议）

  | 特征集 | random 5 折 MAE | GroupKFold MAE | R² (random) |
  |---|---|---|---|
  | comp（33 统计 + 元素分数） | 0.4837 | 0.4048 | 0.2516 |
  | struct（21 维结构描述符） | 0.4161 | — | 0.4171 |
  | **comp+struct** | **0.1929** | **0.1930** | **0.8571** |

  → **MAE 降低 60%，R² 0.25 → 0.86，且首次低于「成分不可约下限」0.379 eV/胞**；
  按成分分组的严格协议下几乎无衰减（0.1930），说明不是靠记成分。
  距 matbench 榜首结构 GNN（coGN 0.0269 eV/胞）仍有差距，属通往 CGCNN 的 CPU 台阶。
- `src/conformal.py`：**split conformal 不确定性校准**（有限样本修正分位数 + K 折交叉校准）
  - 电压：90% 区间半宽 **±1.0001 V**，交叉校准实测覆盖率 **90.0%**（可靠性曲线几乎落在对角线上）
  - 容量：**±75.7751 mAh/g**，覆盖率 **90.1%**
  - 关键诊断：两模型分歧 `|RF-HGB|/2` 只有真实 MAE 的 **17%** → 用它当不确定性会严重低估
- `predictor` 输出 conformal 区间：`Voltage_pi90_low/high`、`Voltage_confidence`（同样用于容量），
  候选表增加 `Voltage_pi90_width`
- `tests/test_uncertainty.py`：conformal 分位数/覆盖率保证/可靠性曲线单调、可信度分级、
  结构描述符几何正确性（含「单元素胞曾算出 inf」的回归）

### 变更

- CI smoke 增加：conformal 校准（电压+容量）→ 结构感知形成能模型 → 产物校验
- README / 项目报告（附录 E）/ 项目总结 / CHANGELOG 同步；测试 96 → **111**；版本 1.3.1 → 1.4.0
- 顺带同步了一处**产物与代码不一致**：已提交的 gap 模型是旧特征代码训练的，
  用当前代码重训后 MAE 0.4405 → 0.4415（R² 0.687），筛选 Top3 随之更新为
  LiV₃O₈ 84.8 / Li₂MnSiO₄ 72.2 / Li₂FeSiO₄ 70.5，文档与图表已全部对齐

## [1.3.1] - 2026-10-21

### 修复

- **CI smoke 失败**：GitHub Actions 出口 IP 访问 `ml.materialsproject.org` 被 CDN 返回
  **403 Forbidden**（本机 / 家宽正常）。现在把清洗后的 `data/processed/expt_gap.csv`、
  `perovskites_ef.csv` 一并入库，CI 与离线环境**无需下载**即可训练带隙/形成能模型；
  下载步骤在 CI 中保留为**非阻断探针**（继续监测官方 URL 可用性）。
- `download_data.py` 改用浏览器式 `User-Agent`（缓解 UA 类 403），并在 403 时打印
  可执行的替代路径提示。

### 说明

- 主干流程现在完全可离线复现：电压/容量用 `mp_voltage.csv`，带隙/形成能用两个 processed CSV，
  外部验证用 `mp_summary_Li.csv.gz`；只有"从零重放官方下载"才需要外网。

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
