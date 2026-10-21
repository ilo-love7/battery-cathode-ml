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
- 39 个单元测试，含「特征列顺序」回归测试（防静默错位）
- GitHub Actions：Linux/Windows × Python 3.9/3.11/3.12，含语法门禁与端到端 smoke test
- Issue / PR 模板、pre-commit、MIT License

---

## 📈 性能指标

| 预测任务 | 数据集 | 样本数 | 模型 | MAE | RMSE | R² |
|---------|--------|--------|------|-----|------|-----|
| **带隙** | matbench_expt_gap（实验值） | 4,604 | RF+HGB 集成 | **0.440 eV** | 0.808 | 0.687 |
| **形成能** | matbench_perovskites（DFT） | 18,928 | RF+HGB 集成 | **0.484 eV/5原子胞** | 0.642 | 0.252 |
| **平均电压** | `battery_voltage.csv`（**合成演示数据**） | 500 | RF+HGB 集成 | **0.192 V** | 0.252 | 0.548 |

> ⚠️ **关于第三行**：电压数据由 `src/generate_voltage_data.py` 按「材料族先验 + 噪声」合成，
> 同一化学式可能被赋予不同电压。该行指标只能说明**流程可跑通**，不含真实物理信息，
> 不应引用为科学结论。接入真实数据（Materials Project 电池数据 / 文献电压表）后才有意义。
>
> 形成能那一行的关键背景：该数据集 18,928 条样本只对应 9,646 个唯一成分，
> 同成分内部目标标准差中位数 0.42 eV/胞，纯成分模型的理论 MAE 下限约 0.379 eV/胞，
> 因此 0.484 eV/胞 已接近成分特征的上限——进一步提升必须引入结构信息（见[深化方向](#-深化方向)）。

---

## 📥 数据获取

| 数据 | 获取方式 | 是否随仓库分发 |
|---|---|---|
| `matbench_expt_gap` / `matbench_perovskites` | `python src/download_data.py`（官方 `ml.materialsproject.org`，无需 Key） | ❌（体积原因，`.gitignore` 排除） |
| `data/battery_voltage.csv` | 已随仓库提供，或 `python src/generate_voltage_data.py` 重新生成 | ✅ |
| `data/cathode_candidates.csv` | 已随仓库提供（40 个候选正极成分） | ✅ |

离线环境：手动下载上述两个 `.json.gz` 放入 `data/raw/`，再执行 `python src/prepare_data.py`。

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
# 数据（首次）
python src/download_data.py
python src/prepare_data.py

# 训练三个模型
python src/train_gap.py
python src/train_ef.py
python src/train_voltage.py

# 预测（支持任意化学式，含小数/括号）
python src/predictor.py --formulas "LiFePO4,LiNi0.8Co0.1Mn0.1O2,Li3V2(PO4)3"
python src/predictor.py --formulas "LiFePO4" --json          # 机器可读输出

# 候选筛选（Top 10，电压窗口 3.0-5.0 V）
python src/predictor.py --top 10 --min-voltage 3.0 --max-voltage 5.0

# 候选排序表 + 排名图（写入 results/）
python src/screening.py

# 全流程
python run_pipeline.py [--fast] [--skip-data] [--skip-training] [--skip-screening]
```

### 2. Python API

```python
from predictor import predict_material_properties, screen_candidates

df = predict_material_properties(["LiFePO4", "LiCoO2"])
print(df.to_string(index=False))
#     formula  Egap_ensemble_eV  Eform_ensemble_eV_fu  Voltage_ensemble_V ...
#     LiFePO4           -0.1923                1.3027               3.548 ...
#      LiCoO2            0.3517                1.1506               3.802 ...

top = screen_candidates(n_top=5, min_voltage=3.0, max_voltage=5.0)
```

> 负的预测带隙说明该成分落在训练分布之外（实验带隙数据里几乎没有 Li 正极），
> 属于外推产物——请以 `Egap_spread` 与风险标记判断可信度，不要直接采信数值。

### 3. Jupyter

```bash
jupyter lab examples/
# 打开 examples/01_getting_started.ipynb
```

---

## ⚠️ 已知限制

1. **纯成分、无结构**：无法区分同成分的不同阳离子占位/相，形成能 MAE 存在 0.379 eV/胞 的成分不可约下限。
2. **域偏移**：带隙/形成能模型训练于实验带隙与**钙钛矿**数据，用于锂电正极属外推，负带隙即外推信号。
3. **电压模型是演示**：目标值为合成数据，R² 主要来自「材料族均值」，无物理意义。
4. **筛选是成分级优先级**：不含 Li 含量变化、电压曲线、动力学、电解液兼容性、成本与毒性。
5. **不确定性代理非校准**：`|RF-HGB|/2` 只是模型间分歧，不是置信区间。

---

## 🗂️ 项目结构

```
battery-cathode-ml/
├── data/
│   ├── raw/                     # Matbench 原始 json.gz（下载，不入库）
│   ├── processed/               # 清洗后 CSV（不入库）
│   ├── battery_voltage.csv      # 电压演示数据
│   └── cathode_candidates.csv   # 40 个候选正极成分
├── src/
│   ├── element_data.py          # 118 元素性质表
│   ├── formula.py               # 化学式解析（括号/小数/等占位）
│   ├── features.py              # 33 统计特征 + 元素分数特征
│   ├── models.py                # 模型构建 / 5 折 OOF 评估 / 绘图
│   ├── voltage.py               # 理论容量与形成能近似电压
│   ├── download_data.py         # Matbench 数据下载
│   ├── prepare_data.py          # 数据清洗
│   ├── train_gap.py             # 带隙模型
│   ├── train_ef.py              # 形成能模型
│   ├── train_voltage.py         # 平均电压模型
│   ├── screening.py             # 候选筛选排序
│   ├── predictor.py             # 预测 / 筛选 CLI
│   ├── generate_voltage_data.py # 合成电压数据
│   ├── fetch_mp_data.py         # MP API 拉取（可选）
│   └── export_cgcnn.py          # 导出 CGCNN 训练目录
├── examples/01_getting_started.ipynb
├── tests/                       # 39 个单元测试
├── results/                     # 指标 / 图 / 模型（pkl 不入库）
├── docs/项目报告.md              # 方法、结果与误差分析
└── .github/                     # Issue/PR 模板 + CI
```

---

## 🔬 深化方向

按「投入产出比」排序，详见 [docs/项目报告.md](docs/项目报告.md) 的 §7 扩展路线与 §9 局限性。

| # | 方向 | 做什么 | 预期收益 |
|---|---|---|---|
| 1 | **接真实电池数据** | MP API `mp-api` 拉 Li 体系 + 已知正极电压表，替换合成电压数据 | 电压指标第一次有物理意义 |
| 2 | **结构特征 / 图神经网络** | 用 `export_cgcnn.py` 导出结构，训练 CGCNN（参照 matbench coGN 0.0269 eV/胞） | 形成能 MAE 从 0.48 → 0.03-0.10，突破成分不可约上限 |
| 3 | **不确定性校准** | 用 OOF 残差做分位数回归 / conformal prediction，输出 90% 预测区间 | 从「模型分歧」升级为可解释的置信区间 |
| 4 | **多目标优化** | 用 Pareto 前沿（电压 vs 容量 vs 稳定性）替代线性加权评分 | 避免权重拍脑袋，给出取舍曲线 |
| 5 | **主动学习闭环** | 模型不确定 → 挑候选 → DFT 计算 → 回训 | 小样本下效率倍增 |
| 6 | **在线演示** | Streamlit / Gradio 页面：输入化学式即时出三维性质 | 项目可展示、可交互 |
| 7 | **可复现性** | 固定 `requirements` 版本 + `matbench` 官方评测接口对齐 | 结果可外部复验 |

---

## 🧪 开发与测试

```bash
python -m compileall -q src tests run_pipeline.py setup.py   # 语法门禁
pytest tests/ -v                                             # 39 个测试
pytest tests/ --cov=src --cov-report=term-missing            # 覆盖率
black src tests && flake8 src tests                          # 风格
```

CI（`.github/workflows/ci.yml`）包含：语法门禁 → lint/format（非阻断）→ 单元测试（含覆盖率）
→ 端到端 smoke（生成数据 → 训练电压模型 → CLI 预测 → 校验产物）→ 安全扫描（非阻断）。

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
  version = {1.1.0}
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
