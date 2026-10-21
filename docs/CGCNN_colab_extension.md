# CGCNN 扩展路线（Colab/Kaggle GPU 可复现指南）

> 对应主报告 §7「扩展路线」。本页给出从本项目当前成分特征模型升级到
> 图神经网络 CGCNN 的最小操作路径，全部步骤在免费 Colab GPU 上可完成。

## 1. 背景

**CGCNN（Crystal Graph Convolutional Neural Networks）**
- 论文: Xie & Grossman, *Crystal graph convolutional neural networks for an
  accurate and interpretable prediction of material properties*,
  **Phys. Rev. Lett. 120, 145301 (2018)**. DOI: 10.1103/PhysRevLett.120.145301
- 思想: 把晶体抽象为多图 G(V, E)。节点 = 原子（特征为元素嵌入向量），
  边 = 邻近原子对（特征为高斯展宽的距离 + 同 site 元素组合），多层图卷积
  聚合邻域信息后池化，经全连接层输出标量性质（形成能、带隙等）。
- 与本项目成分特征模型的区别: CGCNN **感知晶体结构**（键长/配位/对称性），
  在 formation energy / band gap 等性质上误差显著更低；代价是需要
  结构文件（POSCAR/CIF）与 GPU 训练。

## 2. 数据与算力

| 项 | 内容 |
|---|---|
| 训练数据 | `data/raw/matbench_perovskites.json.gz` 已含完整 pymatgen Structure |
| 导出格式 | 见 §3 脚本：每个样本 -> `id_prop.csv` 行 + POSCAR 文件 |
| 算力 | Colab 免费 T4 GPU 即可（18,928 样本, ~30-60 epoch, 约 1-3 小时）|
| 代码库 | https://github.com/txie-93/cgcnn （官方，MIT）|

## 3. 步骤

### 3.1 本地导出 CGCNN 输入（CPU 即可，几秒钟）

```bash
# 在项目根目录
python src/export_cgcnn.py --dataset data/raw/matbench_perovskites.json.gz \
    --out-dir cgcnn_data
# 生成: cgcnn_data/id_prop.csv + cgcnn_data/*.poscar (如需打包 cgcnn_data.zip)
```

（`src/export_cgcnn.py` 随本项目提供；只依赖 python 标准库。）

### 3.2 Colab 上训练

```python
# 1) 新建 Notebook, 菜单: 代码执行程序 -> 更改运行时类型 -> T4 GPU
!git clone https://github.com/txie-93/cgcnn.git
%cd cgcnn
!pip install -r requirements.txt   # torch 已预装, 其余为 pymatgen/sklearn 等

# 2) 上传 cgcnn_data.zip (约 50 MB) 并解压
from google.colab import files
files.upload()                     # 选择 cgcnn_data.zip
!unzip -q cgcnn_data.zip -d data

# 3) 训练 (回归目标列 1 = e_form; 任务类型 regression)
#    注意单位: 导出时默认 --scale atom 已把 matbench 原始 eV/胞 除以位点数,
#    因此 id_prop.csv 里的 target 是 eV/atom, 与下面报的 MAE 口径一致。
!python main.py --train-size 0.8 --val-size 0.1 --test-size 0.1 \
    --epochs 60 --task regression --disable-cuda false \
    --data-path data/cgcnn_data --train-ratio None --val-ratio None --test-ratio None
# 若脚本报 GPU 参数差异, 以仓库 README 当前版本为准 (接口偶有小改)

# 4) 输出: test 结果 MAE (eV/atom); 训练曲线在 checkpoint 目录
```

### 3.3 用 CGCNN 预测候选正极

把候选成分用 pymatgen 构造结构后导出 POSCAR + id_prop.csv，
调用仓库 `predict.py`（或加载 checkpoint 自行推理）。预测值替换
`results/cathode_ranking.csv` 的 `Eform_pred_eV_per_atom` 列即可复用评分管线。

## 4. 预期收益与验证口径

- matbench 官方榜单锚点 (matbench_v0.1, https://matbench.materialsproject.org):
  - matbench_perovskites (18,928 样本): 最好 coGN MAE **0.0269 eV/单胞**（标注 structure required）；
  - matbench_expt_gap (4,604 样本): 最好 MODNet v0.1.12 MAE **0.3327 eV**。
- 本项目成分特征基线: 形成能 MAE ≈ **0.48 eV/5原子单胞**(≈0.10 eV/atom,
  目标 MAD 0.566, scaled error 0.85)；带隙 MAE **0.44 eV**(MAD 1.143, scaled 0.39)。
- 公平对比: 与本项目相同 80/10/10 划分、相同测试集上报 MAE/RMSE/R²；
  在 `results/metrics_ef.json` 基础上增加一行 `CGCNN`。
- 注意: matbench_perovskites 的目标由 RPBE 泛函计算且 O 参考态为水蒸气中
  化学势 (Castelli 2012), 与 MP 常规 GGA+U 形成能不可直接混用。

## 5. 备选

- MEGNet (Keras/TF): https://github.com/materialsvirtuallab/megnet
- ALIGNN (PETNNO 2021, 需要 LINEQL 依赖): https://github.com/usnistgov/alignn
- M3GNet/CHGNet（通用势，可给未弛豫结构弛豫后再筛）: matgl / chgnet

## 6. 常见坑

1. Colab 会话 12 h 上限 -> 定期保存 checkpoint 到 Google Drive；
2. `cgcnn` 依赖版本与 torch 2.x 的兼容问题 -> 固定 `torch==2.1.*` 试跑；
3. POSCAR 元素顺序须与 `id_prop.csv` 一致，避免训练/推理口径错位；
4. 带 GPU 的 Kaggle Notebook 流程相同（Settings -> Accelerator -> GPU T4）。
