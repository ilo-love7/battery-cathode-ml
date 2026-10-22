# CGCNN 路线：现状（推荐）与 Colab/Kaggle 备用方案

> **2026 更新**：本项目现在**自带** CGCNN 实现，无需任何外部仓库、无需 Colab。
> 本页给出两条路径：
> **(A) 现状 —— 本地 GPU 训练（推荐，已验证）**；
> **(B) 备用 —— Colab/Kaggle + 官方 `txie-93/cgcnn` 仓库**（无本地 GPU 时使用）。

## 0. 现状：本项目已实现的模型（推荐路径）

### 0.1 环境（一次搭建，约 2 分钟）

```powershell
# 用笔记本自带 GPU（RTX 4050 Laptop 6 GB 足够：训练峰值显存约 1.6 GB）
python3.12 -m venv D:\项目\venvs\bcm-cgcnn
D:\项目\venvs\bcm-cgcnn\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu124
D:\项目\venvs\bcm-cgcnn\Scripts\pip.exe install numpy pandas matplotlib scikit-learn mp-api chgnet
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### 0.2 已训练并随仓库分发的模型

| 模型 | 数据 | test 指标 | 训练脚本 | 用时（RTX 4050） |
|---|---|---|---|---|
| CGCNN（matbench 钙钛矿） | 18,928 | **0.0401 eV/胞**（0.0080 eV/atom） | `src/train_cgcnn.py` | 210 s / 120 轮 |
| CGCNN（MP 含 Li 形成能） | 8,000（仓库内 3,000 子集） | **0.0962 eV/atom**（3k: 0.1921） | `src/train_cgcnn_mp.py` | 286 s / 150 轮 |
| CGCNN（MP 含 Li 带隙） | 8,000 | **0.5275 eV**，金属/非金属判对率 84.3% | `src/train_cgcnn_mp.py --target band_gap --log1p` | 355 s |
| 双塔 CGCNN（两相结构 → 电压） | 2,661 对 | **0.3525 V**（随机）/ 0.4007（分组） | `src/train_cgcnn_voltage.py` | 155 s |

**推理 CLI**（训练完即可用）：

```powershell
# 结构 -> 每原子形成能
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_cgcnn.py --structures data/raw/mp_structures_cathodes.json.gz
# 化学式 -> 电压（联网取电极对）
D:\项目\venvs\bcm-cgcnn\Scripts\python.exe src/predict_voltage_cgcnn.py --formulas "LiFePO4,LiCoO2"
```

**结构来源**：`src/fetch_mp_structures.py`（`li` / `formulas` / `ids` / `pairs` 四种模式，
含约化式与成分容差匹配）。**MLIP 弛豫**：`src/generate_prototype_structures.py --relax chgnet`
（CHGNet 已集成，用于给 MP 无结构的候选补结构；注意原型只能组内比较，见报告附录 K）。

### 0.3 与外部实现的差异

自包含实现（`src/cgcnn_model.py` + `src/cgcnn_data.py`，约 100 行模型 + 纯 NumPy 建图）：
门控卷积、扁平边索引 + `index_add_` 聚合、晶体内平均池化、目标标准化、ReduceLROnPlateau + 早停。
好处是**零外部依赖、可在 CI 里测图构建、版本可控**；代价是没做超参搜索（matbench 榜首 coGN 为 0.0269 eV/胞）。

---

## 1. 背景：CGCNN 原理

**CGCNN（Crystal Graph Convolutional Neural Networks）**
- 论文：Xie & Grossman, *Crystal graph convolutional neural networks for an accurate and
  interpretable prediction of material properties*, **Phys. Rev. Lett. 120, 145301 (2018)**.
  DOI: [10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)
- 思想：晶体 → 多图 G(V, E)；节点 = 原子，边 = 近邻对（距离高斯展开 + 近邻元素特征），
  多层门控图卷积后池化，再经全连接输出标量性质。
- 与本项目成分特征模型的区别：CGCNN **感知结构**（键长/配位/对称性）。
  实测差异见 §3：形成能提升 **12×**，但带隙几乎无提升（瓶颈在泛函，见报告附录 J）。

---

## 2. 备用路径：Colab/Kaggle + 官方仓库

仅在**没有本地 GPU** 时使用。官方实现：<https://github.com/txie-93/cgcnn>（MIT）。

### 2.1 本地导出 CGCNN 输入（CPU 即可）

```bash
python src/export_cgcnn.py --dataset data/raw/matbench_perovskites.json.gz --out-dir cgcnn_data
# 生成: cgcnn_data/id_prop.csv + cgcnn_data/*.poscar（可打包 cgcnn_data.zip，约 50 MB）
```

### 2.2 Colab 上训练

```python
# 菜单：代码执行程序 -> 更改运行时类型 -> T4 GPU
!git clone https://github.com/txie-93/cgcnn.git
%cd cgcnn
!pip install -r requirements.txt
from google.colab import files
files.upload()                       # 选择 cgcnn_data.zip
!unzip -q cgcnn_data.zip -d data
# 回归目标列 1 = e_form；导出时 --scale atom 已把 eV/胞 换算为 eV/atom
!python main.py --train-size 0.8 --val-size 0.1 --test-size 0.1 \
    --epochs 60 --task regression --disable-cuda false \
    --data-path data/cgcnn_data --train-ratio None --val-ratio None --test-ratio None
```

### 2.3 预测候选正极

用 pymatgen 构造结构 → 导出 POSCAR + `id_prop.csv` → 调 `predict.py`。
本项目里更省事的做法：直接用 `src/predict_cgcnn.py`（加载 `results/cgcnn_mp*.pt`）。

---

## 3. 收益与验证口径（含已达成的实测值）

- matbench 官方锚点（[matbench.materialsproject.org](https://matbench.materialsproject.org)）：
  - `matbench_perovskites`（18,928）：榜首 coGN MAE **0.0269 eV/胞**（标注 structure required）；
  - `matbench_expt_gap`（4,604）：榜首 MODNet v0.1.12 MAE **0.3327 eV**。
- 本项目实测（同一 80/10/10 划分）：

  | 目标 | 成分基线 | + 结构描述符 | CGCNN | 结构带来的提升 |
  |---|---|---|---|---|
  | 形成能（钙钛矿，eV/胞） | 0.4837 | 0.1929 | **0.0401** | **12×** |
  | 形成能（MP 含 Li，eV/atom） | 0.1813（8k） | — | **0.0962** | 1.9× |
  | 带隙（MP DFT，eV） | 0.5605（8k） | — | 0.5275 | **1.06×（3k 更差）** |
  | 电压（MP 插入电极，V） | 0.4187 | 0.4082 | **0.3525** | 1.15× |
- **口径警告**：`matbench_perovskites` 的目标由 RPBE 泛函计算、O 参考态为水蒸气化学势
  （Castelli 2012），与 MP 常规 GGA+U 形成能**不可直接混用**（本项目在 `train_ef_struct.py`
  与 MP 域模型里分别处理，见报告附录 D/F）。
- 结构模型也不是万灵药：带隙的瓶颈是**交换关联泛函**而非几何（报告附录 J）。

---

## 4. 常见坑（本项目踩过的）

1. **Windows 控制台编码**：中文/`Å≤×→` 在 GBK 代码页会抛 `UnicodeEncodeError` 把脚本打挂
   —— 所有 CLI 的 `main()` 都接了 `src/console.py` 的 `enable_utf8_stdout()`；
2. **torch 不能模块级导入**：CI 不装 torch，`--help` 必须可用 —— 已固化为测试
   （`TestTorchImportDiscipline`）；
3. **子集训练会污染全量产物**：`--limit` 自动把 tag 改成 `*_smoke`（v1.4.2 的教训）；
4. **结构来源要标注**：MP 结构 / 容差匹配 / 原型生成，三者可信度不同
   （原型与真实多晶型差 +1.25 eV/atom，只能组内比较）；
5. **新增测试必须在干净副本上验证**（CI 环境没有 `results/*.pkl`），本地能过 ≠ CI 能过；
6. Colab 会话 12 h 上限 → 定期保存 checkpoint；官方仓库与 torch 2.x 偶有兼容问题
   → 固定 `torch==2.1.*` 试跑；POSCAR 元素顺序须与 `id_prop.csv` 一致。

## 5. 其它可替换的结构模型

- MEGNet（TF/Keras）：<https://github.com/materialsvirtuallab/megnet>
- ALIGNN（需依赖较重）：<https://github.com/usnistgov/alignn>
- 通用 MLIP：M3GNet（matgl）/ **CHGNet**（本项目已用于结构弛豫）