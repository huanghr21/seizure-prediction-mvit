# 癫痫发作预测项目 （已弃用，新代码见code_new目录）

**code_new prompt:**
请你阅读文献中的方法和baseline实验，我之前打算复现论文的全部实验，发现工作量过大，所以我现在打算只做一个简单版本：
1：参照论文中选取发作前30min到发作前3min的片段作为发作前数据，非发作数据随机选取和发作前数据等量的片段作为非发作数据，这样可以保证数据量充足且平衡；
2：对选取数据的预处理流程和窗口划分等使用论文中的方式，4s窗口，48hz滤波（注意这些参数要和论文中一样，否则输入图片大小会有变化），、23通道（论文中描述的其实有重合，最终只有21个），s变换；如果遇到了缺失，直接跳过这个窗口；
3：模型也和论文一样，Multi-channel Vision Transformer，这个模型的细节是这样的，每个通道的(32，32)图片会各自经过一个vit模型，最后把21个通道的transformer encoder模块输出拼接起来，输入mlp做分类，细节你可以参考论文；
4：我打算只做第一个受试者的实验，于是我们有很多可以简化，首先数据部分我们就把所有发作事件前期的数据收集起来，然后随机选取等量的非发作数据，直接划分训练集和测试集就行了，最终结果就选用这个测试集上的结果，模型训练也只需要单个受试者的数据就行了。
5：所有总的来说我们不选用baseline的任何设置，只做论文中的方法，但是简化成单个受试者的实验，最终的项目也不需要很多脚本，主要就是数据处理、模型和训练几个模块。因为最终结果会在划分的测试集中评估，这些在训练脚本中实现就行了。
6：old_readme是之前项目的readme，你看看就好，自己找一找可用信息。
最终这个工作重点是侧重于深度学习模型的实现，分析那一块就简化了。


## 项目概述

本项目基于CHB-MIT脑电图（EEG）数据集，使用机器学习方法进行**癫痫发作预测（Seizure Prediction）**，目标是在癫痫即将发作前提前识别出异常脑电活动模式，为临床预警系统提供足够的干预时间。

**核心任务**：这是**预测任务**而非检测任务
- ❌ 检测（Detection）：识别癫痫正在发作
- ✅ 预测（Prediction）：在发作前提前预警
  - **Baseline方法**：发作前10秒识别（短期预警）
  - **优化方法（论文）**：发作前30分钟到发作前3分钟识别（提供27分钟临床干预窗口）

**任务类型**：二分类问题（PreSeizure vs non-PreSeizure）

**数据集**：CHB-MIT Scalp EEG Database
- 来源：波士顿儿童医院
- 包含24名癫痫患者的长时程EEG记录
- 采样频率：256 Hz
- 通道数：21个有效通道（去除重复后）

---

## 项目结构

```
code/
├── baseline/              # 基线实验脚本（已移入此目录）
│   ├── 癫痫前识别实验一.py  # 单被试SVM实验
│   └── 癫痫前识别实验二.py  # 24被试LOOCV实验
├── src/                   # 核心源代码
│   ├── data_processing.py # EEG数据加载、预处理、Dataset、发作时间标注读取
│   ├── stockwell.py       # S-transform时频分析
│   ├── models.py          # MultiChannelViT模型
│   ├── train.py           # Trainer训练类
│   └── evaluate.py        # 评估指标、K-of-N后处理
├── experiments/           # 实验脚本
│   ├── train_mvit.py      # 单被试或多被试训练
│   └── cross_validation.py # K折/LOOCV交叉验证
├── utils/                 # 辅助工具
│   ├── logger.py          # 日志工具
│   └── visualization.py   # 可视化工具
├── checkpoints/           # 模型检查点（运行时生成）
├── results/               # 实验结果（运行时生成）
├── logs/                  # 训练日志（运行时生成）
├── config.py              # 全局配置文件
├── test_annotation_loading.py  # 测试发作时间标注加载
├── requirements.txt       # Python依赖
└── README.md              # 项目文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

**下载CHB-MIT数据集**：

**方式1：官网下载**
- 官网：https://physionet.org/content/chbmit/1.0.0/
- ZIP下载：https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip
- 大小：约40GB

**方式2：HuggingFace镜像站（国内访问更快）**
```bash
# 设置镜像站
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
set HF_ENDPOINT=https://hf-mirror.com     # Windows

# 下载
pip install huggingface_hub
hf download --repo-type dataset Yuchang-Zhao/CHB-MIT --local-dir /path/to/chbmit
```

**解压到合适位置**，例如：
```
D:\dataset\chbmit\rawData\
├── seizure_summary.csv      # 集中式发作时间标注文件（包含所有被试）
├── chb01/
│   ├── chb01-summary.txt   # 单个被试的发作时间标注文件
│   ├── chb01_01.edf
│   ├── chb01_02.edf
│   └── ...
├── chb02/
│   ├── chb02-summary.txt
│   └── ...
├── ...
└── chb24/
    ├── chb24-summary.txt
    └── ...
```

**重要**：每个被试目录下都包含一个 `chbXX-summary.txt` 文件，记录了该被试所有EDF文件中的癫痫发作时间。此外，数据集根目录下还有一个集中式的 `seizure_summary.csv` 文件，包含所有被试的发作时间信息。

**发作时间标注文件**：
1. **集中式CSV文件**（推荐）：`seizure_summary.csv`
   - 位置：数据集根目录
   - 格式：`File_name, Seizure_start, Seizure_stop`
   - 包含所有24个被试的发作时间
   - 项目会优先使用此文件（如果存在）

2. **单独SUMMARY文件**：`chbXX-summary.txt`
   - 位置：各被试目录下
   - 格式：文本文件，包含详细的发作信息
   - 作为CSV文件不存在时的备选方案

**Pre-ictal（发作前）定义**（基于论文方法）：
- **SOP (Seizure Onset Period)** = 30分钟 - 预期发作的时间窗口
- **SPH (Seizure Prediction Horizon)** = 3分钟 - 临床干预期
- **Pre-ictal period** = 从发作前30分钟到发作前3分钟（共27分钟）

> 注：这与baseline实验中的"发作前10秒"定义不同。论文方法使用更长的预测窗口以提供足够的临床干预时间。

修改 `config.py` 中的 `DATA_ROOT` 为你的数据集路径（默认Windows: `D:/dataset/chbmit/rawData`，Linux: `/root/local-nvme/datasets/chbmit`）。

### 3. 测试数据加载

运行测试脚本验证发作时间标注是否正确加载：

```bash
python test_annotation_loading.py
```

该脚本会：
- 扫描所有被试的SUMMARY文件
- 自动计算发作前时间标注（发作前30分钟到发作前3分钟）
- 显示每个被试的标注统计信息
- 验证数据加载功能是否正常

### 4. 运行实验

#### 单被试训练
```bash
python experiments/train_mvit.py --subject chb01 --epochs 50
```

#### K折交叉验证（单被试内）
```bash
python experiments/cross_validation.py --cv_type kfold --subject chb01 --n_folds 10 --epochs 30
```

#### 留一被试交叉验证（24被试）
```bash
python experiments/cross_validation.py --cv_type loocv --epochs 30
```

### 4. 查看结果

训练完成后，结果将保存在：
- `results/` 目录：JSON格式的评估指标
- `checkpoints/` 目录：最佳模型权重
- `logs/` 目录：训练日志文件

---

## Git工作流

### 本地开发
```bash
# 修改代码后
git add .
git commit -m "描述修改内容"
git push
```

### 服务器同步
```bash
# 拉取最新代码
git pull

# 运行实验
python experiments/train_mvit.py --subject chb01
```

**注意**：
- 数据集不会上传到Git（已在.gitignore中排除）
- 在服务器上需单独下载数据集或使用已有数据
- 修改 `config.py` 中的数据路径适配不同环境

---

## 优化方法说明

### 基线方法
- **特征**：手工提取（能量率、频谱质心、频谱熵）
- **分类器**：SVM（RBF核）
- **通道**：1-3个通道
- **窗口**：10秒

### 优化方法（S-transform + MViT）
- **时频分析**：Stockwell Transform（1024点信号 → 32×32时频图）
- **多通道融合**：21个EEG通道的时频图堆叠
- **深度学习**：Multi-channel Vision Transformer
  - 6层Transformer编码器
  - 8个注意力头
  - 512维嵌入
- **窗口**：4秒（非重叠）
- **后处理**：K-of-N平滑（K=4, N=9）

### 关键改进点
1. **时频分析**：从1D时间序列到2D时频表示，捕获频域动态
2. **多通道融合**：利用空间信息和通道间关联
3. **深度特征**：Transformer自动学习判别特征
4. **后处理**：K-of-N规则减少误报

---

## 基线实验说明

以下两个实验脚本为**基线实验**，建立了基础的特征提取和分类框架，后续工作将在此基础上进行优化改进。

---

## 基线实验一：单受试者验证实验

**文件**：`癫痫前识别实验一.py`

### 实验目标
验证基础特征提取和分类方法在单个受试者（chb01）上的可行性，作为快速原型验证。

### 数据配置
- **受试者**：chb01
- **通道**：C3-P3（单通道）
- **发作前标注**：7个时间段
- **窗口设置**：10秒窗口，5秒步长

### 预处理流程
1. **带通滤波**：0.5-25 Hz（去除基线漂移和高频噪声）
2. **Min-Max归一化**：将信号归一化到[0,1]范围
3. **滑动窗口分段**：提取重叠时间片段

### 特征提取
对每个10秒片段提取三类特征：
1. **能量时间变化率**：`np.mean(np.diff(np.square(segment)))`
   - 反映信号能量的时间变化
2. **频谱质心**：`np.sum(freqs * spectrum) / np.sum(spectrum)`
   - 表示频率分布的重心位置
3. **频谱熵**：`-np.sum(spectrum_norm * np.log2(spectrum_norm))`
   - 度量频谱复杂度

### 类别平衡策略
- 从大量non-PreSeizure样本中随机抽取29000个
- 对PreSeizure样本进行**过采样**，使两类数量相等
- 避免类别不平衡导致的模型偏差

### 模型配置
- **分类器**：SVM（RBF核）
- **参数**：`class_weight='balanced'`, `random_state=42`
- **数据划分**：80%训练，20%测试
- **评估指标**：精确率、召回率、F1值

### 实验特点
- ✅ 快速验证特征有效性
- ✅ 单通道降低计算复杂度
- ⚠️ 仅针对单个受试者，泛化能力未验证

---

## 基线实验二：跨受试者交叉验证实验

**文件**：`癫痫前识别实验二.py`

### 实验目标
评估模型在**跨受试者场景**下的泛化能力，模拟临床应用中对新患者的预测能力。

### 数据配置
- **受试者**：chb01-chb24（24名患者）
- **通道**：FP2-F8, FP2-F4, C3-P3（三通道）
- **发作前标注**：200+个时间区间（覆盖所有受试者）
- **窗口设置**：10秒窗口

### 多通道特征提取
改进的`extract_features()`函数：
- 对每个通道独立计算三类特征
- 输出形状：(3_channels, 3_features) = 9维特征向量
- 包含更丰富的空间信息

### 数据处理流程（`process_data()`）
1. **发作前片段提取**：根据时间标注精确提取
2. **随机非发作片段提取**：
   - 数量与发作前片段相同
   - 随机选择10秒窗口
3. **通道完整性检查**：确保所有目标通道可用

### 24折留一交叉验证（LOOCV）
```
迭代1：测试chb01，训练chb02-chb24
迭代2：测试chb02，训练chb01+chb03-chb24
...
迭代24：测试chb24，训练chb01-chb23
```

### 训练数据平衡
- 统计每次迭代训练集中的PreSeizure和non-PreSeizure样本数
- 使用`resample()`下采样到较小类别的数量
- 确保每次训练的类别平衡

### 评估指标
- **混淆矩阵**：TP, TN, FP, FN详细统计
- **准确率**：每个受试者的分类准确率
- **平均性能**：
  - 24个混淆矩阵的平均
  - 24个准确率的平均值

### 实验特点
- ✅ 严格的跨受试者验证
- ✅ 多通道提供更丰富特征
- ✅ 更接近临床实际应用场景
- ⚠️ 计算量大（24次完整训练）

---

## 基线方法技术总结

### 数据处理流程
```
原始EEG数据 
  → 通道选择
  → 带通滤波 (0.5-25 Hz)
  → 归一化
  → 滑动窗口分段
  → 特征提取
  → 类别平衡
  → SVM分类
```

### 核心技术栈
- **数据处理**：MNE-Python（专业EEG工具）
- **机器学习**：Scikit-learn（SVM分类器）
- **数值计算**：NumPy（信号处理、FFT）

### 特征工程
| 特征类型 | 计算方法 | 生理意义 |
|---------|---------|---------|
| 能量时间变化率 | 能量一阶导数 | 信号强度变化速度 |
| 频谱质心 | 频谱加权平均 | 主导频率位置 |
| 频谱熵 | 香农熵 | 频率成分复杂度 |

### 类别平衡方法
- **过采样**（实验一）：重复少数类样本
- **随机配对**（实验二）：动态生成平衡的non-PreSeizure样本

### 两个实验的对比

| 维度 | 实验一 | 实验二 |
|------|--------|--------|
| 受试者数量 | 1个 (chb01) | 24个 (chb01-24) |
| 通道数 | 1个 (C3-P3) | 3个 (FP2-F8, FP2-F4, C3-P3) |
| 特征维度 | 3维 | 9维 |
| 验证方式 | 单次train-test split | 24折留一交叉验证 |
| 泛化能力评估 | 受试者内 | 跨受试者 |
| 计算复杂度 | 低 | 高 |
| 临床相关性 | 概念验证 | 实际应用导向 |

---

## 优化实验：基于S-transform和Vision Transformer的方法

### 方法来源
复现论文方法，采用**Stockwell变换（S-transform）**进行时频分析 + **Multi-channel Vision Transformer (MViT)** 进行分类。

### 核心改进点

#### 1. 数据配置升级
- **通道数量**：21个共有通道（CHB-MIT数据库中所有患者都具备）
  ```
  # 左侧通道
  FP1-F7, F7-T7, T7-P7, P7-O1,
  FP1-F3, F3-C3, C3-P3, P3-O1,
  
  # 右侧通道
  FP2-F4, F4-C4, C4-P4, P4-O2,
  FP2-F8, F8-T8, T8-P8, P8-O2,
  
  # 中线通道
  FZ-CZ, CZ-PZ,
  
  # 额外通道
  T7-FT9, FT9-FT10, FT10-T8
  ```
  > **注**：论文原文列出23个通道，但存在重复（P7-T7和T8-P8各出现2次），去重后为21个唯一通道
  
- **窗口大小**：从10秒改为**4秒**（非重叠滑动窗口）
- **数据增强**：随机分割与组合方法处理文件间隙

#### 2. 特征提取：S-transform（Stockwell变换）
**替代传统FFT的时频分析方法**

- **原理**：结合STFT和小波变换优势的时频分析
- **实现步骤**：
  1. 对4秒EEG片段（1024个采样点）进行S-transform
  2. 得到 (1024, 513) 时频矩阵（0-128 Hz）
  3. 提取0-48 Hz部分 → (1024, 192) 矩阵
  4. 时间维度32倍压缩 + 频率维度6倍压缩
  5. 最终得到 **(32, 32)** 时频特征图（每个通道）
  
- **优势**：
  - 自适应时频分辨率
  - 更好地捕获癫痫发作前的时频模式
  - 保留完整的相位信息

**S-transform公式**：
$$S_X(\tau, f) = \int_{-\infty}^{+\infty} x(t) w(t-\tau, f) e^{-i 2\pi ft} dt$$

$$w(t-\tau, f) = \frac{|f|}{\sqrt{2\pi}} e^{\frac{-(t-\tau)^2 f^2}{2}}$$

#### 3. 分类模型：Multi-channel Vision Transformer (MViT)
**从传统SVM升级到深度学习**

**模型架构**：
- **输入**：(batch, 21_channels, 32, 32) - 21个通道的时频图
- **Patch划分**：将32×32图像分成4个patch
- **Transformer Encoder**：
  - 层数：6层
  - 注意力头数：8个
  - 自注意力机制：捕获全局依赖关系
- **输出**：(batch, 2) - PreSeizure/non-PreSeizure概率

**训练配置**：
- 优化器：Adam (lr=0.00001)
- Batch size: 32
- 最大迭代：50 epochs
- 损失函数：CrossEntropyLoss

#### 4. 后处理：K-of-N平滑
- **方法**：连续N=9个结果中至少K=4个为阳性才报警
- **目的**：降低假阳性率，提高预警可靠性

### 与基线方法对比

| 维度 | 基线方法 | 优化方法 |
|------|---------|---------|
| **通道数** | 1-3个 | 21个（全通道） |
| **窗口大小** | 10秒 | 4秒 |
| **特征提取** | 手工特征（能量、频谱质心、频谱熵）<br>3-9维 | S-transform时频图<br>21×32×32=21,504维 |
| **分类器** | SVM (RBF核) | Vision Transformer |
| **技术栈** | Scikit-learn | PyTorch + MNE |
| **后处理** | 无 | K-of-N平滑 (K=4, N=9) |
| **评估** | 准确率、混淆矩阵 | 准确率+敏感性+特异性+F1<br>事件级评估(SPH, SOP, FPR) |

### 实施计划

#### 阶段1：环境准备与S-transform实现（1周）
- [ ] 安装深度学习环境（PyTorch, CUDA）
- [ ] 实现S-transform函数
- [ ] 在单个文件上验证时频图生成

#### 阶段2：数据预处理优化（1周）
- [ ] 实现21通道数据加载
- [ ] 4秒非重叠窗口分割
- [ ] 批量生成时频特征图
- [ ] 数据增强：随机分割组合

#### 阶段3：MViT模型实现（1-2周）
- [ ] 构建Multi-channel Vision Transformer架构
- [ ] 实现训练循环
- [ ] 在chb01上进行概念验证
- [ ] 调优超参数

#### 阶段4：完整实验与评估（1-2周）
- [ ] 10折交叉验证（论文方法）
- [ ] 24折留一交叉验证（与基线对比）
- [ ] K-of-N后处理
- [ ] 事件级评估（SPH=3min, SOP=30min）

#### 阶段5：结果分析与对比（1周）
- [ ] 与基线实验对比分析
- [ ] 可视化时频图和注意力权重
- [ ] 消融实验（通道数、窗口大小、模型深度）

### 所需依赖

```bash
# 深度学习框架
pip install torch torchvision torchaudio

# S-transform实现
pip install stockwell

# Vision Transformer
pip install timm  # PyTorch Image Models

# 已有依赖
pip install mne numpy scikit-learn matplotlib
```

### 预期效果

根据类似研究和论文结果，预期相比基线方法：
- **准确率提升**：10-15%
- **敏感性提升**：15-20%
- **假阳性率降低**：显著减少
- **泛化能力**：跨受试者性能更稳定

---

## 参考文献

1. CHB-MIT Scalp EEG Database: https://physionet.org/content/chbmit/1.0.0/
2. MNE-Python Documentation: https://mne.tools/
3. Stockwell Transform: Stockwell, R.G., et al. "Localization of the complex spectrum: the S transform." IEEE Trans. Signal Processing, 1996.
4. Vision Transformer: Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

---

## 更新日志

- **2025-12-23**：完成基线实验框架，建立README文档
- 后续优化工作进行中...

---

*本项目为深度学习前沿与交叉课程项目*
