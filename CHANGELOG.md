# Changelog

本文件记录项目的所有重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.13.0] - 2026-10-21

**核心变化：把 DRX/富锂候选的「无序装饰噪声」量化成误差条 —— 稳定性可忽略，带隙不可忽略。**

### 背景

v1.12.0 用「同构替换 + MLIP 选相」补结构时，每个目标只生成了**一个**随机装饰，
等于把"某个偶然的装饰"当成结论。对无序岩盐（DRX）与富锂材料而言，这层**装饰噪声**必须量化。

### 新增

- `src/drx_ensemble.py`：无序系综
  1. 对每个目标固定其最优模板族（来自 v1.12.0 选相结果），生成 **N 个不同种子**的随机装饰（默认 8）；
  2. 每个装饰都用 **CHGNet** 弛豫 → 能量 E_i；
  3. 对全部 N 个装饰用 CGCNN 预测（每原子形成能 / 带隙）→ **系综均值 ± 标准差**；
  4. 代表结构取系综内**能量最低**者（与"按能量选相"一致）。
- `screening_struct.py` 接入系综：结构来源优先级变为
  **MP 结构 → 无序系综代表结构 → 同构替换相 → 岩盐原型**；
  新增误差条列 `ef_ens_std_eV_atom` / `gap_ens_std_eV` 与指标 `decoration_noise`；
  `n_from_ensemble` 计入 A 层（仍为 40 条单层主排名）。

### 实测（9 个目标 × 8 个装饰）

| 量 | 系综标准差（均值） | 最大 | 判读 |
|---|---|---|---|
| CGCNN 每原子形成能 | **0.0209 eV/atom** | 0.0604 | 远小于模型误差 0.0962 与结构残差 0.0903 → **稳定性用单个装饰即可** |
| CGCNN 带隙 | **0.1043 eV** | 0.2608 | 与带隙模型误差（0.53 eV）同量级 → **必须作为误差条引用** |

**实际影响**：改用系综代表结构后，DRX-A 从第 7 名落到第 21 名 ——
说明 v1.12.0 的名次部分来自"幸运装饰"，系综代表结构更稳健。

### 变更

- 测试 151 → **161**（160 通过 + 1 条件跳过）；新增 `TestDrxEnsemble`、`TestEnsembleErrorBars`
- 入库 `data/raw/mp_structures_candidates_ensemble.json.gz`（系综代表结构 + 统计）
- 版本 1.12.0 → 1.13.0

## [1.12.0] - 2026-10-21

**核心突破：换成「同构替换 + MLIP 选相」后，补出来的结构与 MP 真实结构几乎等价，
9 条缺失候选正式并入主排名 —— 候选结构覆盖率真正达到 40/40。**

### 新增

- `src/template_substitution.py`：多晶型枚举 + 同构替换（按 MLIP 能量选相）
  1. 模板库取自仓库内已有的 MP **弛豫**结构（层状 R-3m / 尖晶石 / 橄榄石 / 磷酸盐…）；
  2. 按**阴离子骨架相同 + 阳离子:阴离子比接近**筛兼容模板，按元素重叠度排序取前 N（默认 4）；
  3. 只替换**阳离子**位点（整数占位，晶格与阴离子骨架沿用模板的弛豫几何）；
  4. 每个候选结构用 **CHGNet** 弛豫，**按能量/原子选最低相** —— 即用 MLIP 做多晶型筛选，
     与 MP 收录结构的生成逻辑（枚举 + 总能排序）一致；
  5. 输出 schema 与原型文件相同，可直接喂给 `prototype_sensitivity.py`。
- `prototype_sensitivity.py` 增加 `--out-json`（同时保留「岩盐原型」与「同构替换」两份对比）

### 关键实测：残差降低一个数量级

| 补结构方式 | 每原子形成能 bias | MAE | Spearman ρ | 带隙 bias | 带隙 MAE |
|---|---|---|---|---|---|
| 岩盐型原型（v1.11.0） | +1.2490 eV/atom | 1.2594 | -0.053 | −1.1172 eV | 1.1818 |
| **同构替换 + CHGNet 选相（v1.12.0）** | **+0.0903 eV/atom** | **0.1470** | **0.972** | **-0.0094 eV** | 0.2095 |

- 偏差缩小 **14×**（1.249 → 0.090 eV/atom），MAE 缩小 **8.6×**，排序相关性从 **−0.05 升到 0.97**；
- 残差 MAE 0.1470 eV/atom 已与 CGCNN 自身误差（0.0962 eV/atom）同量级，
  说明**结构替代物几乎不引入额外误差** → 可与 MP 结构同层比较。

### 变更

- `screening_struct.py` 结构来源改为三级优先：**MP 结构 → 同构替换相 → 岩盐原型**；
  分层改为 `A_comparable`（MP + 同构替换，验证可比）/ `B_prototype_rocksalt`（仅组内可比）/ `C_none`；
  **A 层 40 条候选单层主排名**（v1.11.0 是 31 + 9 两层）；
- 新增指标 `n_from_mp_structure`、`n_from_polymorph_substitution`、`polymorph_residual_uncertainty`；
- 新增测试：同构替换语义（阴离子骨架保留、阳离子比匹配、不兼容时返回 None）与
  **可并入主排名的残差门槛**（`|bias| < 0.3`、`ρ > 0.8`），另加"岩盐残差必须显著更大"的对照；

### 主排名（40 条，Top 8）

| 排名 | 候选 | 结构来源 | 稳定性 (eV/atom) | 电压 (V) | 容量 (mAh/g) | 评分 |
|---|---|---|---|---|---|---|
| 1 | LiMn₀.₈Fe₀.₂PO₄ | MP | −2.530 | 3.806 | 170.7 | 72.8 |
| 2 | Li₂MnSiO₄ | MP | −2.603 | 3.629 | 166.6 | 72.3 |
| 3 | LiMnPO₄ | MP | −2.562 | 3.577 | 170.9 | 72.0 |
| 4 | LiMn₀.₆Fe₀.₄PO₄ | MP | −2.496 | 3.781 | 170.5 | 71.0 |
| 5 | LiV₃O₈ | MP | −2.261 | 3.382 | 279.4 | 69.8 |
| 6 | Li₃V₂(PO₄)₃ | MP | −2.674 | 3.627 | 131.5 | 67.5 |
| 7 | **DRX-B** Li₁.₂Mn₀.₄Zr₀.₄O₂ | 同构替换 | −2.618 | 1.902 | 325.6 | 67.3 |
| 8 | LiFePO₄ | MP | −2.432 | 3.407 | 169.9 | 67.3 |

### 变更（文档与其他）

- `docs/CGCNN_colab_extension.md` 重写：本项目的自包含 CGCNN 已成为推荐路径，
  Colab + 官方仓库降级为备用方案；补齐已达成指标与踩坑清单
- 测试 151 → **156**（155 通过 + 1 个条件跳过）；版本 1.11.0 → 1.12.0

## [1.11.0] - 2026-10-21

**核心变化：给 9 个 MP 无结构的候选生成原型结构（含 CHGNet MLIP 弛豫），
但实测证明「原型 ≠ 真实多晶型」，因此筛选中新增独立置信层级而不是硬并入主排名。**

### 新增

- `src/generate_prototype_structures.py`：为缺失候选生成**岩盐型阳离子原型**
  - pymatgen 构建，阳离子按目标化学计量比分布；两种模式：
    - `--decor ordered`（默认）：**整数占位有序超胞**（最大余数法取整 + 固定种子随机分配），
      成分偏差 0.006–0.019；ASE/CHGNet 只能弛豫有序结构，故默认此模式
    - `--decor disordered`：**部分占位**（精确计量比，但无法弛豫）
  - `--relax chgnet`（默认）：用 **CHGNet 通用机器学习势**做 MLIP 弛豫（fmax 0.05、300 步）
- `src/prototype_sensitivity.py`：量化「原型 vs MP 真实结构」的预测差异
- `cgcnn_data.build_graph` 支持**占位度加权的节点 one-hot**
  （有序结构退化为普通 one-hot，已验证旧预测完全不变：LiCoO₂ 仍为 −1.7435 eV/atom）
- `screening_struct.py` 改为 **A/B/C 三层置信结构**：
  - **A 层**（31 条）：MP 弛豫结构 → 主排名 `cathode_ranking_struct.csv`
  - **B 层**（9 条）：原型结构 → 单列 `cathode_ranking_struct_prototype.csv`，**仅组内可比**
  - **C 层**（0 条）：无结构 → `cathode_ranking_struct_fallback.csv`
  - 候选结构覆盖率：**31/40 → 40/40**（但分属两个置信层级）

### 关键实测结论（为什么不能直接并入主排名）

在 31 条同时有两种结构的候选上比较 CGCNN 预测：

| 指标 | 偏差 (bias) | MAE | Spearman ρ |
|---|---|---|---|
| 每原子形成能 | **+1.2490 eV/atom** | 1.2594 | −0.053 |
| 带隙 | −1.1172 eV | 1.1818 | 0.365 |

- 该差异**不是弛豫误差**：晶胞常数扫描 3.80→4.20 Å 偏差都停在 ≈1.1 eV/atom
  （未弛豫原型 +1.10、CHGNet 弛豫后 +1.25，几乎没变）；
- 真正原因是**多晶型能量差**：原型是岩盐型**亚稳**多晶型，MP 给的是该成分的**基态**多晶型
  （如 LiCoO₂ 的层状相）；
- 因此原型候选只能**组内相对比较**，与 A 层混排会产生系统性偏差 → 分层的依据即此。

### 变更

- `.gitignore` 放行候选结构/电极对/原型文件，离线可复现结构驱动筛选
- 测试 146 → **151**（原型生成器、部分占位语义、敏感性 schema、三层产物契约；另修正 `tests/conftest.py` 中名为 `test_element_list` 的 fixture 反模式）
- 版本 1.10.0 → 1.11.0

## [1.10.0] - 2026-10-21

**核心变化：训练了结构版带隙模型（CGCNN on MP band_gap），并得到一个重要的负面结论。**

### 新增

- `train_cgcnn_mp.py` 泛化为支持 `--target`（e_form_per_atom / band_gap）与 `--log1p`：
  带隙分布严重右偏（33% 为金属、gap=0），必须用 log1p 训练、预测时反变换并裁剪到 ≥0
- 结构版带隙模型（CGCNN，MP 含 Li 材料）

  | 数据 | test MAE | 成分基线 MAE | 提升 | 半导体子集 MAE | 金属/非金属判对率 |
  |---|---|---|---|---|---|
  | 3,000 条（仓库内） | 0.7353 eV | 0.6479 eV | **0.88x（更差）** | 0.9532 eV | n/a |
  | 8,000 条（本地） | 0.5360 eV | 0.5605 eV | 1.05x | 0.6472 eV | n/a |

- 新增带隙专用评估指标：**半导体子集 MAE** 与 **金属/非金属判对率**
  （单一 MAE 会被 33% 的金属样本主导，掩盖半导体上的真实表现）
- `screening_struct.py` 增加结构带隙列（`cgcnn_gap_eV`）、`gap_diff_eV` 与两类新标记

### 负面结论（本轮最重要的发现）

结构带隙（PBE 口径）与实验口径的成分带隙，在 31 条结构候选上的
**Spearman rho = -0.5662（负相关）**，
另有 4 条出现「结构判金属、成分判半导体」。
根因：**PBE 系统性低估带隙**，MP 中 33% 的样本 gap=0。

因此工程决策：

- **导电性评分仍用实验口径的成分带隙模型**（与"实际导电性"标签域一致）；
- 结构带隙作为**交叉检查列**保留，新增标记 `结构判金属/成分判半导体(PBE低估)`、`带隙结构/成分差异大`；
- 需要纯结构口径时用 `--prefer-struct-gap` 切换（默认关闭）。

与形成能形成鲜明对比（结构带来 12x 提升）：**并非所有目标都能从结构信息同等获益** ——
带隙的瓶颈在交换关联泛函，不在几何。

### 变更

- 测试 144 → **146**；版本 1.9.0 → 1.10.0

## [1.9.0] - 2026-10-21

**核心变化：候选筛选从「纯成分模型」升级为「结构模型驱动」，并修掉一个方法学陷阱。**

### 新增

- `src/screening_struct.py`：结构模型驱动的候选筛选（40 条候选）
  - 稳定性：**CGCNN（MP 含 Li 域，每原子形成能）**，覆盖率 **31/40**
  - 电压：**双塔 CGCNN**（有电极对时，20/40），否则成分电压模型
  - 导电性：成分带隙模型（尚无结构版，已列入 caveats）
  - 评分：0.40×稳定性 + 0.30×导电性 + 0.30×理论容量（与原 `screening.py` 同权重）
  - 输出：`results/cathode_ranking_struct.csv`（主排名 31 条）、
    `cathode_ranking_struct_fallback.csv`（无结构 9 条）、`cathode_ranking_struct.png`、
    `metrics_screening_struct.json`
- `fetch_mp_structures.py` 支持**三级结构匹配**：精确 → pymatgen 约化式 →
  元素集合+成分容差（固溶体候选）。候选结构覆盖率 **22/40 → 31/40**
  （例：NMC111→mp-1222510、NMC811→mp-1120802、LiMn0.8Fe0.2PO4→mp-849356，偏差 0.007）
- 测试：结构筛选产物契约（主排名只含结构候选、两表不重叠、分数降序、总数守恒）

### 修复（方法学）

- **两套稳定性口径不可混排**：成分形成能模型的目标是 matbench 5 原子胞口径
  （数值为正，如 +0.11 eV/atom），而 CGCNN 是 MP 每原子口径（稳定相为负，如 −2.5 eV/atom）。
  第一版把两者直接放进同一个百分位排名，导致**有结构的候选被系统性抬高**
  （Top12 全是有结构的）。现在改为：
  - **主排名只收录有结构的候选**（同口径，尺度一致）；
  - 无结构的候选单列 `*_fallback.csv`，不参与主排名；
  - 「分歧」不再比绝对值，改用**尺度无关的排名偏离**（|Δrank| > 15 才标记）。
- 结构 vs 成分两套稳定性的 **Spearman ρ = 0.746**（在 31 条结构候选上），
  说明两者大体一致但排序不可互换——这也是必须分开排名的量化依据。

### 主要结果（主排名 Top 5）

| 排名 | 候选 | 结构稳定性 (eV/atom) | 电压来源 | 电压 (V) | 容量 (mAh/g) | 评分 |
|---|---|---|---|---|---|---|
| 1 | LiMn₀.₈Fe₀.₂PO₄ | −2.530 | 成分回退 | 3.806 | 170.7 | 74.8 |
| 2 | Li₂MnSiO₄ | −2.603 | 双塔 CGCNN | 3.629 | 166.6 | 73.3 |
| 3 | LiMnPO₄ | −2.562 | 双塔 CGCNN | 3.577 | 170.9 | 73.0 |
| 4 | LiMn₀.₆Fe₀.₄PO₄ | −2.496 | 成分回退 | 3.781 | 170.5 | 73.0 |
| 5 | LiV₃O₈ | −2.261 | 双塔 CGCNN | 3.382 | 279.4 | 71.8 |

### 变更

- 测试 142 → **144**；版本 1.8.0 → 1.9.0

## [1.8.0] - 2026-10-21

**核心变化：双塔电压模型接入 CLI —— 化学式进去、电压出来。**

### 新增

- `src/fetch_mp_structures.py --mode pairs`：按化学式取**插入电极对**
  （两相结构 + MP 真实电压 + id_charge/id_discharge），实测 10/12 个常见正极命中
- `src/predict_voltage_cgcnn.py`：**双塔 CGCNN 电压推理 CLI**
  - `--pairs`：读电极对文件（仓库内附 10 个常见正极 demo，离线可用）
  - `--formulas`：给化学式联网取电极对（需 MP Key），自动过滤 >6 V 的非物理条目
  - **按数据划分分组报告精度**（train / val / test / unseen），避免拿训练样本冒充精度：

    | 分组 | n | MAE |
    |---|---|---|
    | 训练集内（in-sample，仅供参考） | 7 | 0.1435 V |
    | **测试划分（held-out，可信）** | 2 | **0.3855 V**（与完整测试集 0.3525 V 一致） |

- `data/raw/mp_voltage_pairs_demo.json.gz` 随仓库分发（10 条电极对，含 `dataset_split` 标注）
- 测试：demo 电极对 schema + **必须包含 held-out 样本**（防止 demo 失真）

### 变更

- `build_pair_net(pretrained=None)` 不再打印误导性的"随机初始化"警告
- README 4.2 小节补充 CLI 用法；项目报告附录 H.5 更新为「已接入」（原为下一步）
- 测试 140 → **142**；版本 1.7.0 → 1.8.0

### 说明（诚实标注）

MP 的 Li 插入电极**全库**（2,774 条）都用于训练，因此任何来自该库的 demo 电极对
大多是 in-sample；无偏指标只能看训练脚本的 held-out 划分（0.3525 V）
或按成分分组的模型（0.4007 V）。CLI 已按分组分别展示，不再混为一谈。

## [1.7.0] - 2026-10-21

**核心变化：电压预测改用「两相结构对」端到端建模，MAE 从 0.419 V 降到 0.353 V。**

### 新增

- `src/voltage_from_cgcnn.py`：**物理化路径诊断**（重要负面结果）
  - 把逐原子形成能模型套进热力学公式 `V = [E_f(脱锂) − E_f(嵌锂)] / n_Li`
  - **实测 MAE 0.7004 V、ρ 0.615（n=2,532，覆盖 94.8%）**，还不如同子集上的成分模型（0.4157 V）
  - 根因：每原子误差 ~0.1 eV/atom 被「单胞原子数（可达 80）/ n_Li（常为 1–2）」放大，
    且两相误差各自独立、不会抵消 → **必须学"差异"，不能分开算再相减**
- `src/train_cgcnn_voltage.py`：**双塔 CGCNN**（端到端电压模型）
  - 共享编码器（由 `cgcnn_mp_8k.pt` 初始化）分别编码脱锂相/嵌锂相；
    拼接 `[h_frame, h_lith, h_frame − h_lith]` + 标量（log n_Li、两相原子数）→ MLP 回归电压
  - 编码器与头部使用不同学习率（2e-4 / 1e-3），ReduceLROnPlateau + 早停

  | 划分 | test MAE (V) | R² | 轮数 / 用时 |
  |---|---|---|---|
  | 随机 8:1:1 | **0.3525** | 0.742 | 94 / 155s |
  | 成分分组（更严格） | **0.4007** | 0.747 | 66 / 92s |

- 参照：成分模型 OOF **0.4187 V**（随机）/ 0.4733 V（分组）→ 双塔模型提升 **约 15%**；
  CGCNN-ΔE 物理路径 0.7004 V；中位数基线 MAD 0.849 V

### 变更

- `cgcnn_model.py` 暴露 `encode()` / `head()`，供双塔模型复用编码器
- `voltage_from_cgcnn.py` 的成分模型对比改用 **OOF** 预测（原先误用全量拟合，口径不公平）
- 新增测试：**带 argparse 的脚本必须把 torch 导入放进函数内**（CI 无 torch 仍需 `--help` 可用）
- 测试 135 → **140**；版本 1.6.0 → 1.7.0

## [1.6.0] - 2026-10-21

**核心变化：CGCNN 从「评估集模型」变成「可推理的工具」，并把结构信息用到电压任务上。**

### 新增

- `src/cgcnn_model.py`：把 CGCNN 模型定义抽出（perovskites 训练 / MP 训练 / 推理三处共用）；
  `train_cgcnn.py` 的 `batch_tensors` 一并共用，消除重复
- `src/fetch_mp_structures.py`：从 MP 拉**晶体结构**，三种模式
  - `--mode li`：含 Li 材料摘要 + 结构 + 每原子形成能（CGCNN 训练集）
  - `--mode formulas`：按化学式取**最稳定结构**（推理用）
  - `--mode ids` / `--ids-file`：按 material_id 批量取（电压侧消歧用；5344 个 id 实测 7.8 MB）
- `src/train_cgcnn_mp.py`：CGCNN 训练于 **MP 含 Li 材料**（目标 eV/atom），自带成分基线对照

  | 数据 | test MAE (eV/atom) | 同数据成分基线 | 提升 | 用时 |
  |---|---|---|---|---|
  | 3,000 条（**随仓库分发**，`data/processed/mp_structures_Li.json.gz`） | **0.1921** | 0.2750 | 1.4× | 113 s |
  | 8,000 条（本地 `--mode li` 拉取） | **0.0962** | 0.1813 | 1.9× | 286 s |

- `src/predict_cgcnn.py`：**CGCNN 推理 CLI**（结构 → 每原子形成能）
  - 支持 `--structures`（结构文件，仓库内 10 个常见正极 demo 可离线跑）与 `--formulas`（联网取最稳定结构）
  - 实测：10 个常见正极 MAE 0.058 eV/atom；其中 5 条**不在训练集内**，干净外推 MAE **0.0896 eV/atom**
- `src/prepare_voltage_struct.py` + `src/train_voltage_struct.py`：**电压侧结构消歧**
  - 给电压数据补 44 维结构特征：嵌锂相/脱锂相各 21 维几何描述符 + 脱锂收缩率 + 原子数差
  - 结果：随机 5 折 MAE **0.4187 → 0.4082 V**（R² 0.682 → 0.705）；
    GroupKFold（按框架分组）**0.4733 → 0.4449 V**；结构特征占 RF 重要性 **32%**
  - 只用结构特征反而更差（0.5731 V）：电压主要由成分/氧化还原决定，结构是改良项
- `tests/test_voltage_struct.py`（5 例）：结构特征拼接、单相记录的 NaN 处理、缺失文件错误码、
  量结构表可用性、CLI 接口

### 变更

- `prepare_mp_data.py` 保留 `id_charge` / `id_discharge` / `max_delta_volume`（结构侧消歧所需）
- `data/processed/mp_voltage_struct.csv.gz`（0.97 MB）与 `data/processed/mp_structures_Li.json.gz`（3.5 MB）
  随仓库分发 → 无 API Key 也能复现结构与电压实验
- CI 增加「电压结构消歧（冒烟子集）」步骤与产物校验
- README 新增「MP 域 CGCNN 与推理 CLI」小节；项目报告新增附录 G
- 测试 130 → **135**；版本 1.5.2 → 1.6.0

### 注意事项（诚实记录）

- `predict_cgcnn.py` 的 demo 结构中 8/10 出现在 8,000 条训练池里，**整体 MAE 0.058 有污染**；
  干净的 5 条外推样本 MAE 为 0.0896，训练脚本自带的 held-out 划分（0.0962 @8k）才是无偏指标
- 电压结构消歧的增益有限（随机 2.5%、分组 6%）：该任务的内禀噪声来自**电化学区间**而非几何

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
