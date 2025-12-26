# 癫痫发作预测 - Multi-Channel Vision Transformer

基于CHB-MIT脑电图数据集的癫痫发作预测项目

## 📋 项目简介

本项目实现了基于Multi-Channel Vision Transformer (MViT)的癫痫发作预测方法，参考论文《Deep learning based automatic seizure prediction with EEG time-frequency representation》。通过简单修改配置文件，可以在不同受试者上进行实验。

**核心特点：**
- ✅ 使用S-transform进行时频分析
- ✅ **参数共享的Multi-Channel ViT架构**（~620万参数）
- ✅ 发作前30分钟到发作前3分钟的预测窗口（27分钟干预时间）
- ✅ **三分法数据划分**：训练集(60%) / 验证集(20%) / 测试集(20%)
- ✅ 数据自动平衡（Pre-ictal vs Inter-ictal）
- ✅ **多受试者支持**：只需修改config.py即可切换受试者

## 📁 项目结构

```
code_new/
├── config.py                       # 单被试者配置文件
├── config_multi_subject.py         # 多被试者配置文件（新增）
├── train.py                        # 单被试者训练脚本
├── train_multi_subject.py          # 多被试者训练脚本（新增）
├── train_loso.py                   # LOSO交叉验证脚本（新增）
├── data/
│   ├── preprocessor.py            # 单被试者数据预处理
│   └── multi_subject_preprocessor.py  # 多被试者数据预处理（新增）
├── model/
│   └── mvit.py                    # Multi-Channel ViT模型
├── utils/                         # 工具函数包
│   ├── __init__.py
│   ├── training_utils.py          # 训练工具函数
│   └── cross_subject_utils.py     # 跨被试者评估工具（新增）
├── baseline/                       # baseline实验代码
├── doc/                           # 文档
├── output/                        # 输出目录（运行时生成）
│   ├── checkpoints/              # 单被试者模型检查点
│   ├── results/                  # 单被试者实验结果
│   ├── multi_subject_results/    # 多被试者实验结果（新增）
│   └── loso_results/             # LOSO交叉验证结果（新增）
└── README.md                      # 本文件
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（推荐）
conda create -n seizure_prediction python=3.8
conda activate seizure_prediction

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

**下载CHB-MIT数据集：**

- 官网：https://physionet.org/content/chbmit/1.0.0/
- 下载并解压到本地目录，例如：`D:/ML/chb-mit-scalp-eeg-database-1.0.0`

**数据集结构：**
```
chb-mit-scalp-eeg-database-1.0.0/
├── chb01/
│   ├── chb01-summary.txt      # 发作时间标注
│   ├── chb01_01.edf
│   ├── chb01_02.edf
│   └── ...
├── chb02/
├── ...
└── chb24/
```

**修改配置：**

编辑 `config.py` 中的数据路径：
```python
DATA_ROOT = "D:/ML/chb-mit-scalp-eeg-database-1.0.0"  # 改为你的数据集路径
```

### 3. 运行训练

```bash
python train.py
```

训练过程包括：
1. 加载EEG数据并应用滤波（0.5-48 Hz）
2. 提取发作前数据（发作前30min到发作前3min）
3. 随机采样等量的非发作数据
4. 应用S-transform生成时频特征（32x32）
5. 训练Multi-Channel ViT模型
6. 评估并保存结果

### 4. 查看结果

训练完成后，结果保存在 `output/` 目录：

```
output/
├── checkpoints/
│   ├── best_model_chb01.pth          # 最佳模型权重（验证集最优）
│   └── training_curves_chb01.png     # 训练/验证曲线图
└── results/
    ├── results_chb01.json            # 详细评估指标（JSON格式）
    └── confusion_matrix_chb01.png    # 混淆矩阵可视化
```

**文件说明**：

1. **best_model_chb01.pth**
   - 保存在验证集上表现最好的模型权重
   - 可用于继续训练或推理
   - 包含模型参数、优化器状态等

2. **results_chb01.json**
   - 最终测试集上的评估结果
   - 包含详细指标和混淆矩阵数据
   - 示例格式：
   ```json
   {
     "subject": "chb01",
     "test_metrics": {
       "sensitivity": 0.9800,
       "specificity": 0.9750,
       "accuracy": 0.9775,
       "precision": 0.9760,
       "f1_score": 0.9780
     },
     "confusion_matrix": {
       "TP": 420, "TN": 419,
       "FP": 11, "FN": 10
     },
     "training_info": {
       "epochs_trained": 35,
       "best_epoch": 25,
       "total_time": "12.5 minutes"
     }
   }
   ```

3. **confusion_matrix_chb01.png**
   - 可视化混淆矩阵
   - 清晰展示TP、TN、FP、FN

4. **training_curves_chb01.png**
   - 训练过程中的损失和准确率曲线
   - 帮助分析模型收敛情况

## 🔧 配置说明

主要配置参数（在 `config.py` 中）：

### 数据参数
- `SUBJECT = "chb01"` - 受试者（可改为 "chb02", "chb03" 等）
- `DATA_ROOT` - 数据集根目录
- `SAMPLING_RATE = 256` - 采样率（Hz）
- `N_CHANNELS = 23` - 通道数（所有受试者共有的通道）
- `WINDOW_SIZE = 4` - 窗口大小（秒）
- `FILTER_LOW = 0.5` - 低通滤波（Hz）
- `FILTER_HIGH = 48` - 高通滤波（Hz）

### 通道配置
本项目使用23个标准双极导联通道（CHB-MIT数据集所有受试者共有）：
```
前额：FP1-F7, FP1-F3, FP2-F4, FP2-F8
颞区：F7-T7, T7-P7, F8-T8, T8-P8 (×2)
中央：F3-C3, C3-P3, F4-C4, C4-P4, FZ-CZ, CZ-PZ
枕区：P7-O1, P3-O1, P4-O2, P8-O2
额外：P7-T7, T7-FT9, FT9-FT10, FT10-T8
```
**注意**：T8-P8通道在数据集中出现两次（位置15和23），MNE会自动重命名为T8-P8-0和T8-P8-1。

### 时间定义
- `SOP = 30 * 60` - 发作预测窗口（Seizure Onset Period）：30分钟
- `SPH = 3 * 60` - 临床干预期（Seizure Prediction Horizon）：3分钟
- **Pre-ictal period**：发作前30min到发作前3min（共27分钟）
- **Post-ictal period**：发作后5分钟（避免混淆）
- **Inter-ictal period**：排除Pre-ictal、发作期、Post-ictal的其他时间

### 模型参数
- `PATCH_SIZE = 8` - Patch大小（将32×32分成16个patches）
- `EMBED_DIM = 256` - 嵌入维度
- `NUM_HEADS = 8` - 注意力头数
- `NUM_LAYERS = 6` - Transformer层数
- `MLP_DIM = 512` - MLP隐藏层维度（MLP_RATIO = 512/256 = 2）
- `DROPOUT = 0.1` - Dropout率

### 训练参数
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 1e-5` (0.00001)
- `NUM_EPOCHS = 100` - 最大训练轮数（实际由早停机制决定）
- `TRAIN_RATIO = 0.6` - 训练集比例
- `VAL_RATIO = 0.2` - 验证集比例
- `TEST_RATIO = 0.2` - 测试集比例
- `EARLY_STOPPING_PATIENCE = 10` - 早停耐心值
- `SCHEDULER_PATIENCE = 5` - 学习率调度耐心值

## 📊 方法说明

### 1. 数据预处理流程

#### 1.1 时间定义

```
时间轴 ─────────────────────────────────────────────────────────►

├─────────────┤─────────────────────────┤──────┤────────┤─────────┤
  Inter-ictal      Pre-ictal Period     SPH   Seizure  Post-ictal  Inter-ictal
   (正常期)          (预测窗口)        (干预期) (发作期)  (发作后)     (正常期)
                     27分钟             3分钟            5分钟
                                                    
├──────────── 30分钟(SOP) ─────────────►│
                                       发作时刻
                                        ↑
                                    目标预测点
```

**时期说明**：

- **Pre-ictal（发作前）**：发作前30分钟到发作前3分钟
  - 提供27分钟的干预时间窗口
  - 标签：1
  
- **Inter-ictal（非发作）**：排除Pre-ictal、发作期、Post-ictal后的正常脑电
  - 排除发作前30分钟
  - 排除发作期间
  - 排除发作后5分钟
  - 标签：0

#### 1.2 预处理步骤

**步骤1：EEG数据加载**
- 读取EDF格式的脑电图文件
- 选择23个标准双极导联通道
- 处理重复通道名（MNE自动重命名）
- 采样率：256 Hz

**步骤2：信号滤波**
- 带通滤波：0.5-48 Hz
- FIR滤波器设计
- 去除低频漂移和高频噪声

**步骤3：时间窗口分割**
- 窗口大小：4秒（1024个采样点）
- 无重叠滑动窗口
- 遍历所有EEG记录文件

**步骤4：标签分配**
- 根据发作时间标注（summary.txt）判断窗口类别
- Pre-ictal：[发作时刻-30min, 发作时刻-3min]
- Inter-ictal：其他时间（排除发作期和Post-ictal）

**步骤5：数据平衡**
- 统计Pre-ictal样本数量（如2,148个）
- 随机采样等量的Inter-ictal样本
- 保证类别平衡，避免模型偏向

**步骤6：数据集划分**
- 训练集：60%（用于模型训练）
- 验证集：20%（用于早停和超参数选择）
- 测试集：20%（用于最终评估）
- 使用分层采样保持类别比例

### 2. 特征提取：S-Transform

#### 2.1 S-Transform原理

S-Transform（Stockwell Transform）是一种时频分析方法，结合了短时傅里叶变换（STFT）和小波变换的优点：

```
S(τ, f) = ∫ x(t) · w(t-τ, f) · e^(-i2πft) dt
```

其中窗函数 w(t-τ, f) 的宽度随频率自适应变化。

#### 2.2 实现细节

**输入**：4秒EEG信号（1024个采样点）

**处理流程**：
1. 对每个通道独立应用S-transform
2. 计算结果：(1024, 513) 时频矩阵
   - 时间维度：1024个时间点
   - 频率维度：513个频率点（0-128Hz）

3. 频率截取：保留0-48Hz的频率成分
   - 提取前192个频率点
   - 结果：(1024, 192)

4. 特征压缩：降维到(32, 32)
   - 时间维度：1024 → 32（32倍压缩）
   - 频率维度：192 → 32（6倍压缩）
   - 使用平均池化或插值

**输出**：(23, 32, 32) 多通道时频图

**优势**：
- 时频分辨率自适应
- 保留相位信息
- 对癫痫发作前的频率变化敏感

### 3. 模型架构：参数共享的Multi-Channel ViT

#### 3.1 整体架构

```
输入: (batch, 23, 32, 32)
    ↓
┌─────────────────────────────┐
│  共享的 Single-Channel ViT  │  ← 所有23个通道共用同一个ViT
│  (参数共享，节省参数量)      │     参数量：~256K
└─────────────────────────────┘
    ↓
对每个通道独立前向传播
    ├─ Channel 1: (batch, 32, 32) → (batch, 256)
    ├─ Channel 2: (batch, 32, 32) → (batch, 256)
    ├─ ...
    └─ Channel 23: (batch, 32, 32) → (batch, 256)
    ↓
通道特征拼接: (batch, 23×256) = (batch, 5888)
    ↓
┌─────────────────────────────┐
│      MLP分类器头            │
│  Linear(5888 → 512) + ReLU  │
│  Dropout(0.1)               │
│  Linear(512 → 2)            │
└─────────────────────────────┘
    ↓
输出: (batch, 2) [Pre-ictal, Inter-ictal]
```

**总参数量：~6,200,000 (6.2M)**

#### 3.2 Single-Channel ViT详解

每个通道的ViT结构（所有通道共享）：

**A. Patch Embedding**
```
输入: (batch, 1, 32, 32)
    ↓
分割成16个patches: 每个8×8
    ↓
展平: (batch, 16, 64)
    ↓
线性投影: (batch, 16, 256)
    ↓
添加位置编码
    ↓
添加CLS token: (batch, 17, 256)
```

**B. Transformer Encoder（6层）**

每一层包含：
1. **多头自注意力（Multi-Head Attention）**
   - 8个注意力头
   - 每个头维度：256/8 = 32
   - 计算：Q, K, V = Linear(x)
   - Attention(Q,K,V) = softmax(QK^T/√d)V
   - 学习patches之间的空间关系

2. **前馈网络（MLP）**
   - Linear(256 → 512) + GELU
   - Linear(512 → 256)
   - MLP_RATIO = 2

3. **残差连接与LayerNorm**
   - x = x + Attention(LN(x))
   - x = x + MLP(LN(x))

**C. 输出**
- 取CLS token的输出：(batch, 256)
- 作为该通道的全局特征表示

#### 3.3 参数共享的优势


**参数共享的好处**：
1. **大幅减少参数量**：从124M降到6M（减少95%）
2. **防止过拟合**：参数少，泛化能力强
3. **训练更快**：内存占用小，训练速度快
4. **跨通道一致性**：所有通道使用相同的特征提取器

**为什么可以共享参数？**
- 不同通道记录的是同一大脑不同位置的电活动
- 癫痫发作的模式在不同通道表现相似
- 共享参数相当于学习一个通用的时频模式检测器

### 4. 训练策略

#### 4.1 损失函数与优化器

- **损失函数**：CrossEntropyLoss
  - 适合二分类问题
  - 自动处理softmax

- **优化器**：Adam
  - 学习率：1e-5（0.00001）
  - 权重衰减：1e-4（L2正则化）
  - 自适应学习率

#### 4.2 学习率调度

**ReduceLROnPlateau**：
- 监控验证集损失
- 如果5个epoch没有改善，学习率×0.5
- 防止学习率过大导致震荡

#### 4.3 早停策略

**EarlyStopping**：
- 监控验证集损失
- 如果10个epoch没有改善，停止训练
- 防止过拟合，节省时间

**为什么设置100个epoch？**
- 论文使用固定50个epoch（无早停）
- 本项目加入早停机制后，设置更大的epoch上限（100）
- 让早停机制自动决定最佳停止点
- 实际训练通常在20-40个epoch停止
- 既充分训练，又避免过拟合

#### 4.4 模型保存

- 每个epoch结束后验证
- 如果验证集性能提升，保存模型
- 训练结束后加载最佳模型在测试集上评估

#### 4.5 数据增强

本项目未使用传统的数据增强（如翻转、旋转），因为：
- 时频图的时间轴和频率轴具有物理意义
- 随意变换可能破坏信号特征
- 已通过数据平衡保证类别均衡

#### 4.6 关于后处理方法

**原论文中的后处理方法**：
- 论文采用 **K-of-N 平滑策略**（K=4, N=9）
- 目的：在连续时间窗口中进行平滑，连续N个预测中至少K个为阳性才报警
- 应用场景：**实时、连续的脑电图预测场景**（随时间滑动窗口验证）

**本项目的实现选择**：
- ❌ **未实现后处理方法**
- 原因：本项目采用的是**静态数据集划分实验**，而非连续时间流预测
- 数据划分为训练集/验证集/测试集，在测试集上直接评估性能
- 不涉及连续时间窗口的实时预测场景

**后处理方法的价值**：
- ✅ 在实时监测系统中**显著提升预测稳定性**
- ✅ **大幅降低假阳性率**（减少误报）
- ✅ 提高临床应用中的**预警可靠性**
- 论文结果显示：K-of-N后处理可使假阳性率降低30-50%

## 📈 评估指标

### 混淆矩阵

```
              预测
          Pre  Inter
实际 Pre  [ TP    FN ]
    Inter [ FP    TN ]
```

### 评估指标定义

- **Sensitivity（灵敏度/召回率）**: TP/(TP+FN)
  - 正确识别Pre-ictal的比例
  - 衡量模型**发现发作前状态的能力**
  - **越高越好**（减少漏检）

- **Specificity（特异度）**: TN/(TN+FP)
  - 正确识别Inter-ictal的比例
  - 衡量模型**避免误报的能力**
  - **越高越好**（减少虚警）

- **Accuracy（准确率）**: (TP+TN)/(TP+TN+FP+FN)
  - 整体预测正确的比例
  - 综合评价指标

- **Precision（精确率）**: TP/(TP+FP)
  - 预测为Pre-ictal的样本中实际为Pre-ictal的比例
  - 衡量模型**预测的可信度**

- **F1 Score**: 2×Precision×Recall/(Precision+Recall)
  - Precision和Recall的调和平均
  - 综合考虑精确率和召回率

### 临床意义

- **高Sensitivity**：不错过真正的发作前状态，保护患者安全
- **高Specificity**：减少虚警，避免不必要的干预和焦虑
- **平衡点**：在实际应用中需要权衡两者

## 🔧 切换受试者

得益于良好的代码设计，切换受试者非常简单：

### 方法1：修改配置文件（推荐）

编辑 `config.py`：
```python
# 只需修改这一行
SUBJECT = "chb02"  # 或 "chb03", "chb04", ..., "chb24"
```

### 方法2：通过环境变量

```bash
export SUBJECT=chb02
python train.py
```

### 自动适配

代码会自动处理：
- ✅ 数据路径：`DATA_ROOT/chb02/`
- ✅ 标注文件：`chb02-summary.txt`
- ✅ EEG文件：`chb02_*.edf`
- ✅ 模型保存：`best_model_chb02.pth`
- ✅ 结果保存：`results_chb02.json`
- ✅ 混淆矩阵：`confusion_matrix_chb02.png`



## 🔍 注意事项

1. **通道配置**：23个通道是所有CHB-MIT受试者的共有通道，无需修改
2. **重复通道**：T8-P8在数据集中出现两次，MNE会自动重命名为T8-P8-0和T8-P8-1
3. **窗口参数**：4秒窗口和48Hz滤波必须保持一致，否则特征图尺寸会改变
4. **数据缺失**：部分文件可能通道数不足，代码会自动跳过
5. **计算资源**：
   - GPU推荐：NVIDIA GPU with 6GB+ VRAM
   - CPU可运行但较慢
   - 参数共享设计显著降低内存需求
6. **随机种子**：已设置随机种子(42)保证实验可复现
7. **数据平衡**：自动采样等量的Pre-ictal和Inter-ictal样本


## 🛠️ 依赖库

详见 `requirements.txt`

## 📚 参考文献

Xingchen Dong, Landi He, Haotian Li, et al. "Deep learning based automatic seizure prediction with EEG time-frequency representation." (论文方法)

## 👤 作者

深度学习前沿与交叉课程项目

## 📄 许可证

MIT License

---

## 📎 附录A：Baseline实验说明

本项目包含两个Baseline实验脚本（位于 `baseline/` 目录），使用传统机器学习方法作为对比基线。

### Baseline 1：单受试者验证实验

**文件**：`baseline/癫痫前识别实验一.py`

**实验目标**：验证基础特征提取和分类方法在单个受试者上的可行性

**配置**：
- 受试者：chb01
- 通道：C3-P3（单通道）
- 窗口：10秒窗口，5秒步长
- 发作前定义：发作前10秒（与本项目的30min-3min不同）

**方法**：
1. 特征提取（3维特征）：
   - 能量时间变化率：`np.mean(np.diff(np.square(segment)))`
   - 频谱质心：`np.sum(freqs * spectrum) / np.sum(spectrum)`
   - 频谱熵：`-np.sum(spectrum_norm * np.log2(spectrum_norm))`

2. 分类器：SVM (RBF核)

3. 数据平衡：对PreSeizure样本进行过采样

**特点**：
- ✅ 快速验证，计算量小
- ⚠️ 仅单受试者，泛化能力未验证

### Baseline 2：跨受试者交叉验证实验

**文件**：`baseline/癫痫前识别实验二.py`

**实验目标**：评估模型在跨受试者场景下的泛化能力

**配置**：
- 受试者：chb01-chb24（24名患者）
- 通道：FP2-F8, FP2-F4, C3-P3（三通道）
- 窗口：10秒窗口
- 验证方式：24折留一交叉验证（LOOCV）

**方法**：
1. 多通道特征提取（9维特征）：
   - 对每个通道独立计算3类特征
   - 输出：3通道 × 3特征 = 9维向量

2. 分类器：SVM (RBF核)

3. 交叉验证：
   ```
   迭代1：测试chb01，训练chb02-chb24
   迭代2：测试chb02，训练chb01+chb03-chb24
   ...
   迭代24：测试chb24，训练chb01-chb23
   ```

**特点**：
- ✅ 严格的跨受试者验证
- ✅ 更接近临床实际应用
- ⚠️ 计算量大（24次完整训练）

### 与本项目方法的对比

| 维度 | Baseline方法 | 本项目（MViT方法） |
|------|-------------|------------------|
| **通道数** | 1-3个 | 23个（全通道） |
| **窗口大小** | 5秒 | 4秒 |
| **发作前定义** | 发作前10秒 | 发作前30min到3min |
| **特征提取** | 手工特征（3-9维） | S-transform时频图（23×32×32） |
| **分类器** | SVM | Vision Transformer |
| **参数量** | ~数千 | ~620万 |
| **训练方式** | 传统ML | 深度学习（GPU加速） |
| **后处理** | 无 | 早停+学习率调度 |

### 运行Baseline实验

```bash
# Baseline 1：单受试者
python baseline/癫痫前识别实验一.py

# Baseline 2：跨受试者LOOCV
python baseline/癫痫前识别实验二.py
```

**注意**：Baseline实验使用的是传统特征提取方法，主要用于建立性能基线和方法对比，不是本项目的核心方法。

---

## 📎 附录B：多被试者实验（新增模块）

### 概述

除了单被试者实验外，本项目还提供了**多被试者实验框架**，用于评估模型的跨被试者泛化能力。这是临床应用的关键评估指标。

### 实验方案

提供两种多被试者实验方案：

#### **方案1：被试者级别划分**
- 文件：`train_multi_subject.py`
- 配置：18个训练被试者 / 3个验证被试者 / 3个测试被试者
- 特点：快速评估跨被试者性能（~1-2小时）
- 输出：整体指标 + 每个测试被试者的详细报告

#### **方案2：LOSO交叉验证**
- 文件：`train_loso.py`
- 配置：24折留一交叉验证（每个被试者轮流作测试集）
- 特点：最严格的跨被试者评估（~12-24小时）
- 输出：每折结果 + 汇总统计

### 快速使用

```bash
# 方案1：被试者级别划分（推荐先运行）
python train_multi_subject.py

# 方案2：LOSO交叉验证（需确认）
python train_loso.py
```

### 配置文件

多被试者实验使用独立的配置文件 `config_multi_subject.py`：
- 可自定义被试者划分
- 支持两种数据平衡策略
- 独立的输出目录

### 主要特性

- ✅ **样本溯源**：跟踪每个样本来源被试者
- ✅ **详细报告**：生成每个测试被试者的性能指标
- ✅ **丰富可视化**：柱状图、对比图、混淆矩阵等
- ✅ **容错设计**：自动跳过有问题的被试者数据

### 与单被试者实验的关系

- **单被试者实验**：验证方法在个体上的有效性（本README主要内容）
- **多被试者实验**：评估方法的泛化能力（新增模块，可选）

详细使用方法和参数说明请参考代码中的注释和docstring。

---

**祝训练顺利！** 🎉

如有问题，请检查：
1. 数据集路径是否正确
2. 依赖库是否安装完整
3. 发作时间标注文件是否存在（chb01-summary.txt）
