# Battery Cathode ML - 电池正极材料机器学习预测 🔋

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![CI/CD](https://github.com/ilo-love7/battery-cathode-ml/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/ilo-love7/battery-cathode-ml/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ilo-love7/battery-cathode-ml/issues)

**🌟 AI for Science | Materials Informatics | Battery Research**

</div>

基于 Materials Project 开放数据生态，使用机器学习方法预测电池正极材料的**形成能**、**带隙**和**平均电压**，并对候选正极材料进行综合筛选排序。本项目整合了多个 AI 开发的优秀代码，提供完整的端到端解决方案。

---

## 📑 目录

- [快速开始](#快速开始)
- [项目特色](#项目特色)
- [性能指标](#性能指标)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [文档](#文档)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [引用](#引用)

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/ilo-love7/battery-cathode-ml.git
cd battery-cathode-ml

# 安装依赖
pip install -r requirements.txt

# 一键运行全流程
python run_pipeline.py

# 或使用 Jupyter Notebook
jupyter lab examples/
```

详细教程请查看 [`examples/01_getting_started.ipynb`](examples/01_getting_started.ipynb)。

---

## ✨ 项目特色

### 🔬 **多目标预测**
同时预测三个关键物理量：
- ⚡ **带隙 (Eg)** - 决定电子导电性
- 🔋 **形成能 (Ef)** - 判断结构稳定性  
- ⚖️ **平均电压 (V)** - 影响能量密度

### 🤖 **先进的集成学习**
- **双模型集成**: RandomForest + HistGradientBoosting
- **不确定性量化**: 通过模型离散度评估预测可靠性
- **5 折交叉验证**: 严格的模型评估协议

### 📊 **完整的特征工程**
- 33 维统计特征（原子序数、电负性、原子半径等）
- 元素分数特征（element fraction）
- 自适应元素宇宙构建

### 🎯 **智能材料筛选**
- 综合评分系统（稳定性 40% + 导电性 30% + 容量 30%）
- 风险标记机制（域外元素、高分歧、外推检测）
- Top-N 推荐与可视化

### 🐍 **Pythonic 设计**
- 模块化架构，易于扩展
- 支持命令行和 API 两种使用方式
- 完善的单元测试覆盖

---

## 📈 性能指标

| 预测任务 | 数据集 | 样本数 | 模型 | MAE | RMSE | R² |
|---------|--------|--------|------|-----|------|-----|
| **带隙预测** | matbench_expt_gap | 4,604 | RF+HGB 集成 | **0.440 eV** | 0.808 | 0.687 |
| **形成能预测** | matbench_perovskites | 18,928 | RF+HGB 集成 | **0.484 eV/胞** | 0.642 | 0.252 |
| **电压预测** | Custom Li-battery | 500 | RF+HGB 集成 | **0.193 V** | 0.253 | 0.547 |

> 💡 **提示**: 电压预测精度可通过使用真实 MP API 数据进一步提升！

---

## 📦 安装指南

### 基本安装

```bash
# 从 PyPI（发布后）
pip install battery-cathode-ml

# 或从源码
git clone https://github.com/ilo-love7/battery-cathode-ml.git
cd battery-cathode-ml
pip install -e .
```

### 开发环境

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 设置 pre-commit hooks
pre-commit install
```

### 可选功能

```bash
# 连接 Materials Project API
pip install pymatgen mp-api

# 可视化增强
pip install plotly seaborn
```

---

## 💻 使用方法

### 1. 命令行界面

```bash
# 生成电压数据
python src/generate_voltage_data.py

# 训练模型
python src/train_voltage.py

# 筛选候选材料
python src/screening.py

# 预测新材料
python src/predictor.py --formulas "LiFePO4,LiCoO2,LiMn2O4"

# 运行完整流程
python run_pipeline.py
```

### 2. Python API

```python
from battery_cathode_ml.predictor import predict_material_properties

# 预测单个材料
results = predict_material_properties(["LiFePO4", "LiCoO2"])
print(results)

# 输出:
#        formula  Egap_ensemble_eV  Eform_ensemble_eV_fu  Voltage_ensemble_V
# 0    LiFePO4            1.752                 3.055                3.548
# 1     LiCoO2            1.752                 3.055                3.802
```

### 3. Jupyter Notebook

```bash
# 启动 Jupyter
jupyter lab examples/

# 打开教程
open examples/01_getting_started.ipynb
```

---

## 📚 文档

- **[完整教程](examples/)** - Jupyter Notebook 示例
- **[API 文档](https://ilo-love7.github.io/battery-cathode-ml/)** - 自动化生成的 API 文档
- **[贡献指南](CONTRIBUTING.md)** - 如何参与项目开发
- **[代码规范](.github/CODE_STYLE.md)** - 开发风格指南
- **[常见问题](docs/FAQ.md)** - 常见问题解答

---

## 🗂️ 项目结构

```
battery-cathode-ml/
├── 📁 data/                          # 数据文件
│   ├── raw/                         # 原始数据
│   ├── processed/                   # 处理后的数据
│   └── cathode_candidates.csv       # 候选材料列表
├── 📁 src/                          # 源代码
│   ├── features.py                  # 特征工程
│   ├── models.py                    # 模型框架
│   ├── predictor.py                 # 预测工具
│   ├── screening.py                 # 材料筛选
│   └── train_*.py                   # 训练脚本
├── 📁 examples/                     # 示例代码
│   └── 01_getting_started.ipynb     # 入门教程
├── 📁 tests/                        # 测试代码
├── 📁 results/                      # 输出结果
├── 📁 docs/                         # 文档
├── 📄 README.md                     # 项目说明
├── 📄 CONTRIBUTING.md               # 贡献指南
└── 📄 LICENSE                       # 开源许可
```

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看：

- 📝 **[贡献指南](CONTRIBUTING.md)** - 提交 PR 的详细步骤
- 🐛 **[报告 Bug](https://github.com/ilo-love7/battery-cathode-ml/issues/new?template=bug_report.md)** - 发现问题？告诉我们！
- ✨ **[请求功能](https://github.com/ilo-love7/battery-cathode-ml/issues/new?template=feature_request.md)** - 有新想法？提出来！

### 快速开始贡献

```bash
# 1. Fork 本仓库
# 2. 创建特性分支
git checkout -b feature/AmazingFeature

# 3. 提交更改
git commit -m 'Add some AmazingFeature'

# 4. 推送到分支
git push origin feature/AmazingFeature

# 5. 开启 Pull Request
```

---

## 🎓 引用

如果您在研究中使用本项目，请引用：

```bibtex
@software{battery_cathode_ml_2026,
  author = {Battery Cathode ML Contributors},
  title = {Battery Cathode ML: Machine Learning Toolkit for Battery Cathode Material Property Prediction},
  year = {2026},
  url = {https://github.com/ilo-love7/battery-cathode-ml},
  version = {1.0.0}
}
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下项目和资源的支持：

- **[Materials Project](https://materialsproject.org)** - 提供高质量的材料数据
- **[Matbench](https://matbench.materialsproject.org)** - 公开的基准数据集
- **[Scikit-learn](https://scikit-learn.org)** - 强大的机器学习库
- **两个原始 AI 项目** - 提供了优秀的基础代码

---

## 📬 联系我们

- 💬 **问题反馈**: [GitHub Issues](https://github.com/ilo-love7/battery-cathode-ml/issues)
- 📧 **邮箱**: 17695324ilo@gmail\.com
- 🐦 **Twitter**: @ilo-love7

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ilo-love7/battery-cathode-ml&type=Date)](https://star-history.com/#ilo-love7/battery-cathode-ml&Date)

---

<div align="center">

**Made with ❤️ by the Battery Cathode ML Community**

[Back to Top](#battery-cathode-ml---电池正极材料机器学习预测)

</div>
