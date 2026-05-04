"""
配置文件 - 包含所有实验参数和路径配置
"""

import os

# ============= 路径配置 =============
# 数据集根目录（请根据实际情况修改）
DATA_ROOT = "/root/local-nvme/datasets/chbmit/rawData"

# 受试者选择
SUBJECT = "chb12"
SUBJECT_PATH = os.path.join(DATA_ROOT, SUBJECT)

# 输出目录
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 创建输出目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============= 数据参数 =============
# 采样率
SAMPLING_RATE = 256  # Hz

# 通道配置（论文中提到的23个通道）
CHANNEL_NAMES = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",  # 第一个 T8-P8（MNE会重命名为 T8-P8-0）
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8",  # 第二个 T8-P8（MNE会重命名为 T8-P8-1）
]

# 实际使用的通道数
N_CHANNELS = 23

# 滤波参数
FILTER_LOW = 0.5  # Hz
FILTER_HIGH = 48  # Hz

# 窗口参数
WINDOW_SIZE = 4  # 秒
WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE  # 1024个采样点

# ============= 时间定义 =============
# Pre-ictal period: 发作前30分钟到发作前3分钟
SOP = 30 * 60  # Seizure Onset Period: 30分钟（秒）
SPH = 3 * 60  # Seizure Prediction Horizon: 3分钟（秒）

# Post-ictal period: 发作后5分钟
POST_ICTAL = 5 * 60  # 秒

# ============= S-Transform参数 =============
# 频率范围
FREQ_MIN = 0  # Hz
FREQ_MAX = 48  # Hz

# S-transform后的时频矩阵尺寸
ST_TIME_BINS = 1024  # 时间维度
ST_FREQ_BINS = 192  # 频率维度（0-48Hz，从513个频率点中选择前192个）

# 压缩参数
TIME_COMPRESSION = 32  # 时间维度压缩倍数
FREQ_COMPRESSION = 6  # 频率维度压缩倍数

# 最终特征图尺寸
FEATURE_SIZE = 32  # 32x32

# ============= 模型参数 =============
# ViT参数
PATCH_SIZE = 8  # 将32x32分成4个patch，每个patch是8x8（32/4=8）
EMBED_DIM = 256  # 嵌入维度
NUM_HEADS = 8  # 注意力头数
NUM_LAYERS = 6  # Transformer层数
MLP_DIM = 512  # MLP隐藏层维度
DROPOUT = 0.1  # Dropout率

# 分类参数
NUM_CLASSES = 2  # 二分类：PreSeizure vs non-PreSeizure

# ============= 训练参数 =============
# 数据划分（训练/验证/测试）
TRAIN_RATIO = 0.6  # 训练集比例
VAL_RATIO = 0.2  # 验证集比例
TEST_RATIO = 0.2  # 测试集比例
RANDOM_SEED = 42  # 随机种子

# 训练超参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-5  # 0.00001
NUM_EPOCHS = 200  # 使用早停机制，设置较大上限让模型充分训练
WEIGHT_DECAY = 1e-4

# 优化器
OPTIMIZER = "Adam"

# 学习率调度器
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# 早停
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 25

# ============= 设备配置 =============
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12  # DataLoader的工作进程数

# ============= 其他配置 =============
# 是否打印详细信息
VERBOSE = True

# 保存配置
SAVE_BEST_ONLY = True  # 只保存最佳模型

# 日志配置
LOG_INTERVAL = 10  # 每多少个batch打印一次日志

print(f"Configuration loaded. Device: {DEVICE}")
print(f"Subject: {SUBJECT}, Channels: {N_CHANNELS}, Window: {WINDOW_SIZE}s")
