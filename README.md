# Battery Cathode ML - 电池正极材料机器学习预测 🔋

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![CI/CD](https://github.com/ilo-love7/battery-cathode-ml/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/ilo-love7/battery-cathode-ml/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ilo-love7/battery-cathode-ml/issues)

**🔋 Materials Informatics | Battery Research | Reproducible ML**

</div>

基于公开的 Matbench / Materials Project 数据生态，用**成分特征 + 树模型集成**预测正极材料的
**带隙**、**形成能**与**平均电压**，并对候选正极成分做综合筛选排序。

> **一句话定位**：一个可复现、可扩展的「成分级材料性质预测」教学/原型工具链，
> 而不是经过实验验证的材料发现结论。完整的方法、结果与误差分析见
> [docs/项目报告.md](docs/项目报告.md)。

---

## 📑 目录

- [快速开始](#快速开始)
- [项目特色](#项目特色)
- [性能指标](#性能指标)
- [数据获取](#数据获取)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [已知限制](#已知限制)
- [项目结构](#项目结构)
- [深化方向](#-深化方向)
- [开发与测试](#开发与测试)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 🚀 快速开始

```bash
git clone https://github.com/ilo-love7/battery-cathode-ml.git
cd battery-cathode-ml
pip install -r requirements.txt

# ① 只体验电压模型（自带数据，无需下载，约 2 分钟）
python src/train_voltage.py
python src/predictor.py --formulas "LiFePO4,LiCoO2,LiMn2O4"

# ② 跑完整流程（需先下载 Matbench 公开数据，约 5 分钟）
python src/download_data.py      # 下载 data/raw/*.json.gz（无 API Key）
python run_pipeline.py
```

交互式教程：[`examples/01_getting_started.ipynb`](examples/01_getting_started.ipynb)

---

## ✨ 项目特色

### 🔬 **多目标预测**
同一套特征工程覆盖三个目标：
- ⚡ **带隙 (Eg)** — 电子导电性的粗筛指标
- 🔋 **形成能 (Ef)** — 热力学稳定性的粗筛指标
- ⚖️ **平均电压 (V)** — 能量密度的粗筛指标

### 🤖 **双模型集成 + 不确定性代理**
- RandomForest（中位数插补）+ HistGradientBoosting（原生 NaN）
- 5 折交叉验证 OOF 评估，集成 = 两模型等权平均
- 以两模型预测差 `|RF-HGB|/2` 作为不确定性代理，用于筛选阶段降权

### 📊 **成分特征工程**
- 33 个统计特征：6 类元素属性（Z/质量/电负性/半径/族/周期）× 5 种统计量 + 3 个式级量
- 元素分数（element fraction）特征：在训练集元素宇宙上展开
- 缺失属性（如稀有气体电负性）按权重重新归一化，避免均值被稀释

### 🎯 **候选筛选**
- 综合评分：稳定性 40% + 导电性 30% + 理论容量 30%（百分位秩）
- 风险标记：域外元素、模型分歧高、带隙外推
- 输出排序表 + 家族着色排名图

### 🧪 **工程化**
- 193 个单元测试（192 通过 + 1 个条件跳过），含特征列顺序回归、元素表完整性、文档一致性、MP 数据清洗、多目标、结构描述符、conformal 覆盖率、晶体图构建、电压结构特征、torch 延迟导入约束、demo 电极对 schema、结构筛选三层产物契约、结构带隙模型契约与原型结构生成/占位语义
- GitHub Actions：Linux/Windows × Python 3.9/3.11/3.12，含语法门禁与端到端 smoke test
- Issue / PR 模板、pre-commit、MIT License

---

## 📈 性能指标

| 预测任务 | 数据集 | 样本数 | 模型 | MAE | RMSE | R² |
|---------|--------|--------|------|-----|------|-----|
| **带隙** | matbench_expt_gap（实验值） | 4,604 | RF+HGB 集成 | **0.4415 eV** | 0.808 | 0.687 |
| **形成能** | matbench_perovskites（DFT） | 18,928 | RF+HGB 集成 | **0.484 eV/5原子胞** | 0.642 | 0.252 |
| **形成能（结构感知）** | matbench_perovskites（21 维结构描述符 + 成分） | 18,928 | RF+HGB 集成 | **0.193 eV/5原子胞** | 0.281 | 0.857 |
| **形成能（结构 GNN / CGCNN）** | matbench_perovskites（晶体图端到端） | 18,928 | CGCNN（97.5k 参数，GPU） | **0.0401 eV/5原子胞**（0.0080 eV/atom） | 0.0665 | — |
| **形成能（CGCNN · MP 含 Li）** | MP 含 Li 材料结构（真实 DFT） | 3,000 / 8,000 | CGCNN（GPU） | **0.1921 / 0.0962 eV/atom** | — | — |
| **带隙（CGCNN · MP，阴性结果）** | 同上（MP DFT 带隙，33% 金属） | 3,000 / 8,000 | CGCNN（log1p，GPU） | **0.7304 / 0.5275 eV**（成分基线 0.6479 / 0.5605） | 0.928 | — |
| **平均电压（双塔 CGCNN）** | MP 插入电极两相结构（端到端） | 2,532 | 双塔 CGCNN（GPU） | **0.352 V**（随机）/ 0.401（分组） | 0.521 | 0.742 |
| 平均电压（成分 + 结构描述符） | 同上 + 44 维描述符 | 2,667 | RF+HGB 集成 | 0.408 V（随机）/ 0.445（分组） | 0.597 | 0.705 |
| **比容量（多目标）** | 同上（真实 `capacity_grav`） | 2,672 | RF+HGB 集成 | **25.16 mAh/g** | 46.9 | 0.663 |
| 平均电压（合成对照） | `battery_voltage.csv`（合成演示数据） | 500 | RF+HGB 集成 | 0.192 V | 0.252 | 0.548 |

**外部验证 I｜文献商用正极（12 个，不参与训练）**：
**MAE 0.160 V、RMSE 0.212 V、Spearman ρ 0.842** —— NMC111 误差 0.006 V、NMC523 0.008 V、LiFePO4 0.061 V。
复现：`python src/validate_known_cathodes.py`。

**外部验证 II｜Materials Project 含 Li 材料（8,942 条，跨口径）**：
形成能**去偏移 MAE 0.735 eV/atom**（MP 均值基线 MAD 0.805）、Spearman ρ 0.475；
带隙 MAE 1.12 eV、ρ 0.332（模型训练于**实验**带隙，MP 为 DFT-PBE，属跨口径比较）。
复现：`python src/validate_mp_external.py`。这组数字说明**跨口径外推能力有限**，
也是后续引入结构模型（CGCNN，见 §4.1/4.2）与 MP 域重训的直接证据。

> **关于电压两行的读法**：真实数据那一行才是可用结论。同一化学式在该数据集里可能对应
> 多个电压（不同框架/脱锂区间），因此额外按 `framework_formula` 做 GroupKFold，
> MAE 为 **0.481 V**、R² 0.627 —— 这是更严格的泛化估计。
> 中位数基线 MAE 为 0.679 V，模型相对基线降低约 38%；容量模型相对基线（MAD 47.8 mAh/g）降低约 47%。
>
> 合成数据那一行只说明**流程可跑通**，不含真实物理信息，保留作对照，请勿引用为结论。
>
> 形成能的关键背景：18,928 条样本只对应 9,646 个唯一成分，同成分内部目标 std 中位数
> 0.42 eV/胞，纯成分模型的理论 MAE 下限约 0.379 eV/胞（0.484 已逼近该上限）。
> **引入 21 维结构描述符后 MAE 降到 0.193（随机划分）/ 0.193（按成分分组），R² 从 0.252 升到 0.857**
> ——这正是「必须看结构」的量化证据；**已进一步换成 CGCNN：0.0401 eV/胞**
> （见 §4.1；matbench 榜首 coGN 为 0.0269）。
> 对照组：结构描述符单独用 0.416，成分单独用 0.484。
>
> **不确定性**：`conformal.py` 用 split conformal 给出有覆盖率保证的区间——电压 90% 区间半宽
> ±1.00 V（K 折交叉校准实测覆盖率 90.0%），容量 ±75.8 mAh/g（90.1%）。
> 注意：两模型分歧只有真实 MAE 的 17%，**以 conformal 区间为准，不要用 spread 当置信度**。
>
> **结构侧消歧（v1.6.0）**：给电压模型补 44 维结构特征（嵌锂相/脱锂相几何描述符、脱锂收缩率、
> 原子数差）后，随机 5 折 MAE 0.4187 → **0.4082 V**，按框架分组 0.4733 → **0.4449 V**；
> 结构特征贡献 32% 特征重要性（Top：`lith_nn_dist_max`、`frame_elem_nn_mean`、`delta_vol`）。
> 只用结构特征反而更差（0.5731 V）——电压主要由成分/氧化还原决定，结构是改良项。

---

## 📥 数据获取

| 数据 | 获取方式 | 是否随仓库分发 |
|---|---|---|
| `matbench_expt_gap` / `matbench_perovskites` | `python src/download_data.py`（官方 `ml.materialsproject.org`，无需 Key） | ❌（体积原因，`.gitignore` 排除） |
| **MP 插入电极真实电压** | `python src/fetch_mp_battery.py --what electrodes` → `python src/prepare_mp_data.py`（需免费 MP API Key） | ✅ 清洗后的 `data/processed/mp_voltage.csv` 已随仓库提供，**没有 Key 也能复现真实指标** |
| `data/battery_voltage.csv` | 已随仓库提供，或 `python src/generate_voltage_data.py` 重新生成 | ✅ |
| **MP 含 Li 材料摘要（外部验证集）** | `python src/fetch_mp_battery.py --what summary` → `python src/prepare_mp_summary.py` | ✅ 清洗后的 `data/processed/mp_summary_Li.csv.gz`（8,942 条）已随仓库提供 |
| `data/cathode_candidates.csv` | 已随仓库提供（40 个候选正极成分） | ✅ |

**配置 Materials Project API Key**（免费，约 2 分钟）：

1. 注册 https://materialsproject.org → Dashboard → **API Keys** → Create New Key；
2. 任选一种方式存放（key 不进仓库，`.gitignore` 已排除 `.mp_api_key`）：
   - `setx MP_API_KEY "你的key"`（Windows，新开终端生效）
   - 写入 `%USERPROFILE%\.mp_api_key`
   - 或运行时传参 `--key 你的key`

> 数据来源与许可：Materials Project 数据按 **CC BY 4.0** 使用，引用见文末「引用」一节。
> `data/raw/mp_*.csv`（原始 API 输出）不入库，用上面的命令可随时重放。

---

## 📦 安装指南

```bash
# 从源码安装（扁平 src 布局，安装后模块为顶层名 predictor/features/...）
git clone https://github.com/ilo-love7/battery-cathode-ml.git
cd battery-cathode-ml
pip install -e .

# 开发依赖
pip install -r requirements-dev.txt
pre-commit install
```

> 本项目**未发布到 PyPI**，请勿执行 `pip install battery-cathode-ml`。
> 安装后可用的命令行入口：`battery-predict`、`battery-train-voltage`。

可选依赖（连接 Materials Project API 时需要）：

```bash
pip install -e ".[mp]"     # pymatgen + mp-api
```

---

## 💻 使用方法

### 1. 命令行

```bash
# ① Matbench 数据（无需 Key）→ 带隙 / 形成能模型
python src/download_data.py
python src/prepare_data.py
python src/train_gap.py
python src/train_ef.py

# ② Materials Project 真实电压数据（需免费 API Key）→ 真·电压模型
python src/fetch_mp_battery.py --what electrodes      # 拉 2.7k 条插入电极
python src/prepare_mp_data.py                          # 清洗 -> data/processed/mp_voltage.csv
python src/train_voltage.py --data data/processed/mp_voltage.csv --tag voltage
python src/train_capacity.py                           # 多目标：比容量模型
python src/validate_known_cathodes.py                  # 与 12 个商用正极对照
python src/validate_mp_external.py                     # 与 8,942 条 MP 含 Li 材料对照（跨口径）
python src/conformal.py --tag voltage                  # 90% 预测区间（conformal 校准）
python src/conformal.py --tag capacity

# 结构感知（CPU，无需 GPU）：先固化结构描述符，再训练
python src/prepare_struct_features.py
python src/train_ef_struct.py

# 合成数据对照（可选）
python src/train_voltage.py --data data/battery_voltage.csv --tag voltage_synthetic

# 预测（支持任意化学式，含小数/括号）
python src/predictor.py --formulas "LiFePO4,LiNi0.8Co0.1Mn0.1O2,Li3V2(PO4)3"
python src/predictor.py --formulas "LiFePO4" --json          # 机器可读输出

# GPU 侧：CGCNN 训练与推理（需 torch 环境，见「GPU 结构模型」）
python src/fetch_mp_structures.py --mode li --n 3000          # 拉结构（需 MP Key）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/train_cgcnn_mp.py --epochs 150
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_cgcnn.py --structures data/raw/mp_structures_cathodes.json.gz

# 电压结构消歧（CPU）
python src/prepare_voltage_struct.py
python src/train_voltage_struct.py

# 候选筛选（Top 10，电压窗口 3.0-5.0 V）
python src/predictor.py --top 10 --min-voltage 3.0 --max-voltage 5.0

# 候选排序表 + 排名图（写入 results/）
python src/screening.py

# 全流程
python run_pipeline.py [--fast] [--skip-data] [--skip-training] [--skip-screening]
```

> 只有 `data/processed/mp_voltage.csv` 是**已经随仓库提供**的，所以即使没有 API Key，
> 也能直接跳到 `train_voltage.py --data data/processed/mp_voltage.csv` 复现真实指标。

### 2. Python API

```python
from predictor import predict_material_properties, screen_candidates

df = predict_material_properties(["LiFePO4", "LiCoO2"])
print(df.to_string(index=False))
#     formula  Egap_ensemble_eV  Eform_ensemble_eV_fu  Voltage_ensemble_V ...
#     LiFePO4           -0.1373                1.3027               3.389 ...
#      LiCoO2            0.5155                1.1506               3.693 ...

top = screen_candidates(n_top=5, min_voltage=3.0, max_voltage=5.0)
```

> 负的预测带隙说明该成分落在训练分布之外（实验带隙数据里几乎没有 Li 正极），
> 属于外推产物——请以 `Egap_spread` 与风险标记判断可信度，不要直接采信数值。

### 4. GPU 结构模型（CGCNN）

CPU 结构描述符（0.193 eV/胞）之上，项目内置了一个**自包含的 CGCNN 实现**
（`src/cgcnn_data.py` 建图 + `src/train_cgcnn.py` 训练，无需克隆外部仓库）。

| 模型 | test MAE (eV/5原子胞) | MAE (eV/atom) | 训练耗时 |
|---|---|---|---|
| 成分基线（33 统计 + 元素分数） | 0.4837 | 0.0967 | 25 s（CPU） |
| 成分 + 21 维结构描述符 | 0.1929 | 0.0386 | 4 min（CPU） |
| **CGCNN（本项目实现）** | **0.0401** | **0.0080** | **210 s / 120 轮（RTX 4050）** |
| matbench 榜首 coGN（参考） | 0.0269 | 0.0054 | GPU 数小时 |

训练/验证/测试 = 0.0269 / 0.0384 / 0.0401 eV/胞（8:1:1 随机划分，无早停触发）。

> **环境**（GPU 版 torch 需要独立 Python 3.12 环境）：
> ```bash
> python3.12 -m venv D:\项目\venvs\bcm-cgcnn
> D:\项目\venvs\bcm-cgcnn\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu124
> D:\项目\venvs\bcm-cgcnn\Scripts\pip.exe install numpy pandas matplotlib
> ```
> **训练**：
> ```bash
> D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/train_cgcnn.py --epochs 120
> D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/train_cgcnn.py --limit 500 --epochs 3   # 冒烟
> ```
>
> ⚠️ CGCNN 的输入是**晶体结构**而非化学式。结构获取已完整落地：
> `fetch_mp_structures.py`（MP 检索，精确/约化式/容差）、`template_substitution.py`（同构替换 + CHGNet 选相）、
> `generate_prototype_structures.py`（岩盐原型，最后手段），
> 推理入口见 `predict_cgcnn.py` 与 `predict_voltage_cgcnn.py`。



### 6. 结构模型驱动的候选筛选（v1.12.0，需 GPU 环境）

`screening_struct.py` 把稳定性换成 **CGCNN（MP 含 Li 域，每原子形成能）**、电压优先用**双塔 CGCNN**：

```powershell
python src/fetch_mp_structures.py --mode formulas --formulas "<40 候选>" --out data/raw/mp_structures_candidates.json.gz
python src/fetch_mp_structures.py --mode pairs --formulas "<40 候选>" --out data/raw/mp_voltage_pairs_candidates.json.gz
# 补 MP 无结构的候选：同构替换 + CHGNet 选相（v1.12.0，推荐）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/template_substitution.py
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/template_substitution.py --all --out data/raw/mp_structures_candidates_polymorph_all.json.gz
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/prototype_sensitivity.py --prototypes data/raw/mp_structures_candidates_polymorph_all.json.gz --out-json results/metrics_polymorph_sensitivity.json
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/screening_struct.py
```

**候选结构覆盖率 40/40，且是单层主排名**（结构来源三级优先：MP → 同构替换 → 岩盐原型）：

| 层级 | 数量 | 结构来源 | 产物 | 可比性 |
|---|---|---|---|---|
| **A_comparable** | **40** | MP 弛豫结构（31）+ **无序系综代表结构（9）** | `cathode_ranking_struct.csv` | 实测可比（见下表） |
| B_prototype_rocksalt | 0 | 岩盐型原型 | `cathode_ranking_struct_prototype.csv` | 仅组内可比 |
| C_no_structure | 0 | 无结构 | `cathode_ranking_struct_fallback.csv` | 成分回退 |

**为什么同构替换相可以并入主排名**（`prototype_sensitivity.py`，同成分对比）：

| 补结构方式 | 形成能 bias | MAE | ρ | 带隙 bias | 带隙 MAE |
|---|---|---|---|---|---|
| 岩盐型原型 | +1.2490 eV/atom | 1.2594 | -0.053 | −1.1172 eV | 1.1818 |
| **同构替换 + CHGNet 选相** | **+0.0903 eV/atom** | **0.1470** | **0.972** | **-0.0094 eV** | 0.2095 |

偏差缩小 **14×**、排序相关性 **0.97**，残差与 CGCNN 自身误差（0.0962 eV/atom）同量级
→ 结构替代物几乎不引入额外误差。**做法要点**：模板取自 MP 已弛豫结构，只替换阳离子位点、
晶格与阴离子骨架沿用模板几何，再用 **CHGNet** 弛豫并按能量/原子选最低相（即用 MLIP 做多晶型筛选）。

电极对覆盖 20/40；结构 vs 成分稳定性的 Spearman ρ = **0.7462**（A 层内，尺度无关）。

**无序装饰噪声（v1.13.0）**：DRX/富锂候选改为 **N=8 个随机装饰的系综**（`drx_ensemble.py`，
每个装饰经 CHGNet 弛豫）。实测系综标准差：每原子形成能 **0.0209 eV/atom**（可忽略），
带隙 **0.1043 eV**（最大 0.2608，**必须作为误差条引用**）。
导出列：`ef_ens_std_eV_atom`、`gap_ens_std_eV`、`n_decorations`、`ef_ens_mean_eV_atom`。

**评分（v1.16.0：权重经文献实测校准）** = 0.20×稳定性 + 0.10×导电性 + 0.50×容量 + 0.20×电压
（`calibrate_weights.py` 用 39 条文献实测比能量做外部目标；**先验权重 ρ=−0.14，校准后 ρ=+0.54**；可用 `--prior-weights` 回到先验）

（旧三维口径保留为 `score_3dim`）。其中：
- 稳定性/导电性用 **1σ 保守估计**（见下）；
- **电压维度用区间下界**（`voltage_V − 局部半宽`），区间宽的候选被降权；
- 区间宽度来自 `local_conformal.py`：**局部自适应**（特征近邻加权分位数）。

  实测：全局 split conformal q=1.0001 V（**无区分度**）；局部自适应原始覆盖率仅 77.1%（局部可交换假设不成立），×1.47 校准后 **88.70%** ≈ 名义 90%，同时保留 **9.4×** 区分度。

**1σ 口径说明（v1.14.0）** —— 系综候选的稳定性/带隙点预测改用**系综均值**
（保留 `cgcnn_ef_best_member_eV_atom` 对照），评分用 `mean + σ`（稳定性取更不稳定一侧、
带隙取更绝缘一侧）；无误差条的候选 σ=0，口径统一。指标里的 `scoring_mode = conservative_1sigma` 即此。
语义说明：σ 是**结构装饰噪声**；默认导电性评分用实验口径成分带隙时，它是数据质量惩罚，配合 `--prefer-struct-gap` 时即结构带隙的不确定度。

Top 5（A 层 40 条）：LiMn₀.₈Fe₀.₂PO₄(72.8) > Li₂MnSiO₄(72.3) > LiMnPO₄(72.0) > LiMn₀.₆Fe₀.₄PO₄(71.0) > LiV₃O₈(69.8)

### 4.1 MP 域 CGCNN 与推理 CLI（v1.6.0）

上面的 CGCNN 训练于 matbench 钙钛矿（5 原子单胞）。v1.6.0 把它扩展到**锂电池相关体系**：

| 训练数据 | test MAE (eV/atom) | 同数据成分基线 | 提升 | 用时 |
|---|---|---|---|---|
| MP 含 Li 材料 3,000 条（**随仓库分发**） | **0.1921** | 0.2750 | 1.4× | 113 s |
| MP 含 Li 材料 8,000 条（本地拉取） | **0.0962** | 0.1813 | 1.9× | 286 s |

**推理 CLI**（补上「结构 → 性质」最后一环）：

```powershell
# 仓库内 10 个常见正极结构（离线可用）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_cgcnn.py --structures data/raw/mp_structures_cathodes.json.gz
# 任意化学式（联网取最稳定结构，需 MP Key）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_cgcnn.py --formulas "LiCoO2,LiFePO4,LiMn2O4"
```

10 个常见正极的预测与 MP 真值对比：整体 MAE 0.058 eV/atom；其中 **5 条结构不在训练集内**，
干净外推 MAE **0.0896 eV/atom**（LiNiO₂ 误差 0.23，属训练池外的高镍体系——域外推仍是主要风险）。


### 4.2 端到端电压模型（双塔 CGCNN，v1.7.0）

先试「物理化路径」：把每原子形成能模型套进 `V = [E(脱锂) − E(嵌锂)] / n_Li`，
结果 **MAE 0.700 V**，还不如成分模型（0.416 V）——每原子误差被
「单胞原子数（可达 80）/ n_Li（常 1–2）」放大，且两相误差各自独立、不会抵消。

**双塔模型**改为直接学「差」：共享编码器（由 `cgcnn_mp_8k.pt` 初始化）分别编码两相，
拼接 `[h_frame, h_lith, h_frame − h_lith]` + 标量（log n_Li、两相原子数）→ MLP 回归电压。

| 模型 / 划分 | test MAE (V) | R² |
|---|---|---|
| 成分模型（RF+HGB，OOF） | 0.4187（随机）/ 0.4733（分组） | 0.682 |
| CGCNN-ΔE 物理路径 | 0.7004 | — |
| **双塔 CGCNN（随机 8:1:1）** | **0.3525** | 0.742 |
| **双塔 CGCNN（成分分组，更严格）** | **0.4007** | 0.747 |
| 中位数基线（MAD） | 0.8493 | — |

→ 相对成分模型提升 **约 15%**，比中位数基线好 2.4 倍；训练 94 轮 / 155 s（RTX 4050，122.7k 参数）。

```powershell
# 需要结构文件（MP Key，约 3 分钟）
python src/fetch_mp_structures.py --mode ids --ids-file data/raw/voltage_material_ids.txt --out data/raw/mp_structures_electrodes.json.gz
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/train_cgcnn_voltage.py --epochs 120
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/train_cgcnn_voltage.py --group-split --tag cgcnn_voltage_group
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/voltage_from_cgcnn.py   # 物理路径对照
```

**推理 CLI**（化学式进去、电压出来）：

```powershell
# 离线 demo（仓库内 10 个常见正极电极对，含 MP 真实电压）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_voltage_cgcnn.py --pairs data/raw/mp_voltage_pairs_demo.json.gz
# 任意化学式（联网取电极对，需 MP Key）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_voltage_cgcnn.py --formulas "LiFePO4,LiCoO2,LiMn2O4"
```

CLI **按数据划分分组报告精度**（避免拿训练样本冒充精度）：
训练集内 n=7 MAE 0.1435 V（仅供参考）｜**测试划分 n=2 MAE 0.3855 V**（与完整测试集 0.3525 V 一致）。
### 4.3 结构版带隙模型：一个诚实的阴性结果（v1.10.0）

用同一套 CGCNN 训练 **MP DFT 带隙**（33% 为金属，需 log1p 变换）：

| 数据 | test MAE | 成分基线 | 提升 | 半导体子集 MAE | 金属/非金属判对率 |
|---|---|---|---|---|---|
| 3,000 条 | 0.7304 eV | 0.6479 eV | **0.89×（更差）** | 0.9334 eV | 78.4% |
| 8,000 条 | 0.5275 eV | 0.5605 eV | 1.06× | 0.6423 eV | 84.3% |

更关键的是：结构带隙与实验口径成分带隙在 31 条候选上的排序 **Spearman ρ = -0.5589（负相关）**，
并有 4 条出现「结构判金属、成分判半导体」——根因是 **PBE 系统性低估带隙**。

**工程决策**：导电性评分仍用**实验口径的成分带隙模型**（与「实际导电性」的标签域一致），
结构带隙作为**交叉检查列**保留并加标记；需要纯结构口径时用 `--prefer-struct-gap` 切换。
对比形成能（结构带来 12× 提升）可见：**并非所有目标都能从结构信息同等获益**，
带隙的瓶颈在交换关联泛函而非几何。

### 5. Jupyter

```bash
jupyter lab examples/
# 打开 examples/01_getting_started.ipynb
```

---

## ⚠️ 已知限制

1. ~~**CGCNN 尚未接入预测链路**~~ ✅ 已完成：`predict_cgcnn.py`（结构 → 每原子形成能）、
   `predict_voltage_cgcnn.py`（化学式 → 电压）、`screening_struct.py`（结构驱动筛选，候选结构覆盖 **40/40**）
   均已上线；结构来源支持 MP 检索（精确/约化式/容差）、同构替换 + CHGNet 选相、岩盐原型三级。
   **仍存的风险**：结构替代物有残差（同构替换相与 MP 真结构差 +0.09 eV/atom、ρ 0.97，
   已并入主排名；岩盐原型差 +1.25 eV/atom，只能组内比较），DRX 的无序本质仍用单个装饰近似。
2. **域偏移**：带隙/形成能模型训练于实验带隙与**钙钛矿**数据，用于锂电正极属外推，负带隙即外推信号；
   MP 域 CGCNN 已缓解，但带隙的结构模型收益很小（PBE 低估，见报告附录 J）。
3. **电压模型域很宽**：训练集含大量 MP 计算的非商用体系（氢化物、氟化物等），虽然覆盖了 LCO/LFP/NMC 等真实正极，
   但精度仍受框架歧义限制（双塔 CGCNN 已把 MAE 降到 0.353 V，分组 0.401 V；电极对覆盖仅 20/40）。
4. **筛选是成分级优先级**：不含 Li 含量变化、电压曲线、动力学、电解液兼容性、成本与毒性。
5. **区间很宽，且 spread 会误导**：conformal 90% 区间半宽 ±1.00 V（电压）/ ±75.8 mAh/g（容量），说明精度有限；而 `|RF-HGB|/2` 只有真实 MAE 的 17%，**不能当置信度用**。

---

## 🗂️ 项目结构

```
battery-cathode-ml/
├── data/
│   ├── raw/                     # Matbench json.gz + MP API 原始 CSV（下载，不入库）
│   ├── processed/               # 清洗后 CSV
│   │   └── mp_voltage.csv       # ★ 真实 MP 电压数据集（随仓库提供，无需 Key）
│   ├── battery_voltage.csv      # 合成演示数据
│   └── cathode_candidates.csv   # 40 个候选正极成分
├── src/
│   ├── element_data.py          # 118 元素性质表
│   ├── formula.py               # 化学式解析（括号/小数/等占位）
│   ├── features.py              # 33 统计特征 + 元素分数特征
│   ├── models.py                # 模型构建 / 5 折 OOF 评估 / 绘图
│   ├── voltage.py               # 理论容量与形成能近似电压
│   ├── download_data.py         # Matbench 数据下载
│   ├── prepare_data.py          # 数据清洗
│   ├── fetch_mp_battery.py      # ★ MP 插入电极/摘要拉取（需 Key）
│   ├── prepare_mp_data.py       # ★ MP 电极数据清洗 -> mp_voltage.csv
│   ├── prepare_mp_summary.py    # ★ MP 摘要清洗 -> mp_summary_Li.csv.gz（外部验证集）
│   ├── train_gap.py             # 带隙模型
│   ├── train_ef.py              # 形成能模型
│   ├── train_voltage.py         # 平均电压 / 比容量模型（--data/--tag/--target 切换）
│   ├── train_capacity.py        # ★ 比容量模型（多目标，复用同一训练框架）
│   ├── validate_known_cathodes.py  # ★ 文献商用正极外部验证
│   ├── validate_mp_external.py  # ★ MP 含 Li 材料外部验证（跨口径）
│   ├── structure_features.py    # ★ 21 维结构描述符（晶格/近邻距离统计）
│   ├── prepare_struct_features.py  # ★ 结构描述符固化 -> perovskites_struct.csv.gz
│   ├── train_ef_struct.py       # ★ 结构感知形成能模型（成分 vs 结构 vs 两者）
│   ├── conformal.py             # ★ split conformal 不确定性校准
│   ├── cgcnn_data.py            # ★ 晶体图构建（原子/边特征，纯 NumPy）
│   ├── train_cgcnn.py           # ★ CGCNN（matbench 钙钛矿，eV/胞）
│   ├── cgcnn_model.py           # ★ CGCNN 模型定义（训练/MP 域/推理共用）
│   ├── train_cgcnn_mp.py        # ★ CGCNN（MP 含 Li 材料，eV/atom）
│   ├── predict_cgcnn.py         # ★ CGCNN 推理 CLI（结构 -> 每原子形成能）
│   ├── fetch_mp_structures.py   # ★ 从 MP 拉晶体结构（li/formulas/ids）
│   ├── prepare_voltage_struct.py  # ★ 电压数据补结构特征（两相描述符+收缩率）
│   ├── train_voltage_struct.py  # ★ 电压结构消歧对照实验
│   ├── train_cgcnn_voltage.py   # ★ 双塔 CGCNN：两相结构 -> 电压（端到端）
│   ├── voltage_from_cgcnn.py    # ★ CGCNN-ΔE 物理路径（负面结果对照）
│   ├── predict_voltage_cgcnn.py # ★ 双塔 CGCNN 电压推理 CLI（两相结构 -> 电压）
│   ├── screening.py             # 候选筛选排序（成分模型）
│   ├── screening_struct.py      # ★ 结构模型驱动的候选筛选（CGCNN + 双塔电压，A/B/C 三层）
│   ├── generate_prototype_structures.py  # ★ 缺失候选的岩盐型原型 + CHGNet MLIP 弛豫
│   ├── prototype_sensitivity.py # ★ 原型/同构替换相 vs MP 结构预测差异
│   ├── template_substitution.py # ★ 多晶型枚举 + 同构替换（按 MLIP 能量选相）
│   ├── drx_ensemble.py          # ★ 无序系综（随机装饰 + 误差条）
│   ├── local_conformal.py       # ★ 局部自适应区间 + 条件覆盖率（按体系族）
│   ├── calibrate_weights.py     # ★ 多元外部标签校准评分权重（含自助法区间）
│   ├── annotate_reference.py    # ★ 参考表逐条引用（DOI 经 Crossref 核验）
│   ├── pareto_interactive.py    # ★ 偏好驱动的 Pareto 最优子集
│   ├── resource_risk.py         # ★ 资源供应风险与毒性（序数打分）
│   ├── predictor.py             # 预测 / 筛选 CLI
│   ├── generate_voltage_data.py # 合成电压数据
│   ├── fetch_mp_data.py         # MP API 拉取（通用摘要版）
│   └── export_cgcnn.py          # 导出 CGCNN 训练目录
├── examples/01_getting_started.ipynb
├── tests/                       # 193 个单元测试
├── results/                     # 指标 / 图 / 模型（pkl 不入库）
├── docs/项目报告.md              # 方法、结果与误差分析
└── .github/                     # Issue/PR 模板 + CI
```

---

## 🔬 深化方向

按「投入产出比」排序，详见 [docs/项目报告.md](docs/项目报告.md) 的 §7 扩展路线与 §9 局限性。

| # | 方向 | 做什么 | 预期收益 |
|---|---|---|---|
| ~~1~~ | ~~接真实电池数据~~ ✅ **已完成** | `fetch_mp_battery.py` + `prepare_mp_data.py` 拉取并清洗 2,772 条 MP 插入电极，真实电压模型 MAE 0.419 V | 电压指标首次具备物理意义 |
| ~~2~~ | ~~容量等多目标~~ ✅ **已完成** | `train_capacity.py`：真实 `capacity_grav`，MAE 25.16 mAh/g（基线 47.8），任意化学式可用 | 筛选缺 `nLi_extract` 时也有容量维度 |
| ~~2~~ | ~~结构特征 → 图神经网络~~ ✅ **已完成** | matbench 域 CGCNN 0.0401 eV/胞；**MP 含 Li 域 test 0.0962 eV/atom（8k）/ 0.1921（3k）**，比同数据成分基线好 1.9×/1.4×；已配 `predict_cgcnn.py` 推理 CLI | 结构模型可用，形成能进入 0.1 eV/atom 量级 |
| ~~3~~ | ~~不确定性校准~~ ✅ **已完成** | `conformal.py`：90% 区间 ±1.00 V / ±75.8 mAh/g，交叉校准覆盖率 90.0%/90.1%；并指出模型分歧只有真实 MAE 的 17% | 预测从「一个数」变成「一个数 + 有覆盖率保证的区间」 |
| 4 | **多目标优化** | 用 Pareto 前沿（电压 vs 容量 vs 稳定性）替代线性加权评分 | 避免权重拍脑袋，给出取舍曲线 |
| ~~5~~ | ~~框架级消歧~~ ✅ **已完成** | 44 维结构特征：随机 0.419→**0.408 V**，分组 0.473→**0.445 V**（结构占重要性 32%）；v1.7.0 进一步改为双塔 CGCNN 端到端 **0.3525 V** | 电压误差已压缩到 0.35 V 量级 |
| ~~6~~ | ~~缺失候选的结构生成~~ ✅ **已完成** | 同构替换 + CHGNet 选相（MP 无结构的 9 条）：与 MP 真实结构残差 bias +0.0903 eV/atom、ρ 0.972（岩盐原型为 +1.25、ρ -0.05）→ 候选结构覆盖率 **40/40 单层主排名** | 结构模型覆盖全部候选 |
| 6 | **在线演示** | Streamlit / Gradio 页面：输入化学式即时出三维性质 | 项目可展示、可交互 |

---

## 🧪 开发与测试

```bash
python -m compileall -q src tests run_pipeline.py setup.py   # 语法门禁
pytest tests/ -v                                             # 193 个测试
pytest tests/ --cov=src --cov-report=term-missing            # 覆盖率
black src tests && flake8 src tests                          # 风格

# 真实数据链路（需 MP API Key；没有 Key 也可直接用仓库里的 processed CSV 训练）
python src/fetch_mp_battery.py --what electrodes
python src/prepare_mp_data.py
python src/train_voltage.py --data data/processed/mp_voltage.csv --tag voltage
python src/train_capacity.py
python src/validate_known_cathodes.py
python src/validate_mp_external.py
python src/prepare_struct_features.py && python src/train_ef_struct.py
python src/conformal.py --tag voltage
```

CI（`.github/workflows/ci.yml`）包含：语法门禁 → lint/format（非阻断）→ 单元测试（含覆盖率）
→ 端到端 smoke（生成数据 → 训练电压模型 → CLI 预测 → 校验产物）→ 安全扫描（非阻断）。
MP 拉取因需要 Key 不在 CI 跑，但**清洗规则有单元测试覆盖**（`tests/test_mp_data.py`，用离线 fixture）。

---

## 🤝 贡献指南

- 📝 [贡献指南](CONTRIBUTING.md) — 分支规范、提交信息、PR 流程
- 🐛 [报告 Bug](https://github.com/ilo-love7/battery-cathode-ml/issues/new?template=bug_report.md)
- ✨ [请求功能](https://github.com/ilo-love7/battery-cathode-ml/issues/new?template=feature_request.md)
- 📄 [行为准则](CODE_OF_CONDUCT.md)

---

## 🎓 引用

```bibtex
@software{battery_cathode_ml_2026,
  author = {Battery Cathode ML Contributors},
  title = {Battery Cathode ML: Composition-based Property Prediction for Battery Cathodes},
  year = {2026},
  url = {https://github.com/ilo-love7/battery-cathode-ml},
  version = {1.21.0}
}
```

本项目使用的 Materials Project 数据按 **CC BY 4.0** 提供，请同时引用：

```bibtex
@article{jain2013materials,
  title   = {Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  author  = {Jain, Anubhav and others},
  journal = {APL Materials},
  volume  = {1},
  number  = {1},
  pages   = {011002},
  year    = {2013}
}
```

---

## 📄 许可证

[MIT License](LICENSE)。

## 🙏 致谢

- [Materials Project](https://materialsproject.org) / [Matbench](https://matbench.materialsproject.org) — 公开数据与基准
- [Scikit-learn](https://scikit-learn.org) — 建模基础设施

---

<div align="center">

**Made with ❤️ by the Battery Cathode ML Community**

[Back to Top](#battery-cathode-ml---电池正极材料机器学习预测)

</div>
