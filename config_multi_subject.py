"""
多被试者实验配置
支持两种模式：
1. 被试者级别划分（train_subjects / val_subjects / test_subjects）
2. LOSO留一交叉验证（自动生成24折）
"""

import os

from config import *  # 继承原始配置

# ============================================
# 多被试者模式选择
# ============================================
USE_LOSO = False  # True: 使用LOSO交叉验证, False: 使用固定划分

# ============================================
# 所有可用被试者列表（CHB-MIT数据集共24名患者）
# ============================================
ALL_SUBJECTS = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
    "chb21",
    "chb22",
    "chb23",
    "chb24",
]

# ============================================
# 方案1：被试者级别划分（18/3/3）
# ============================================
TRAIN_SUBJECTS = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
]  # 18个被试者用于训练

VAL_SUBJECTS = ["chb19", "chb20", "chb21"]  # 3个被试者用于验证（早停、超参数调整）

TEST_SUBJECTS = ["chb22", "chb23", "chb24"]  # 3个被试者用于最终测试（完全未见过）

# ============================================
# 数据平衡策略
# ============================================
BALANCE_STRATEGY = (
    "per_subject"  # "per_subject": 每个被试者内部平衡, "global": 全局平衡
)

# ============================================
# 输出路径（多被试者专用）
# ============================================
OUTPUT_DIR_MULTI = os.path.join(OUTPUT_DIR, "multi_subject_results")
CHECKPOINT_DIR_MULTI = os.path.join(OUTPUT_DIR_MULTI, "checkpoints")
RESULTS_DIR_MULTI = os.path.join(OUTPUT_DIR_MULTI, "results")
LOGS_DIR_MULTI = os.path.join(OUTPUT_DIR_MULTI, "logs")

# LOSO输出路径
OUTPUT_DIR_LOSO = os.path.join(OUTPUT_DIR, "loso_results")

# ============================================
# 训练参数调整（可选）
# ============================================
# 多被试者数据量更大，可能需要更多epoch或更早收敛
NUM_EPOCHS_MULTI = 100  # 最大epoch数
EARLY_STOPPING_PATIENCE_MULTI = 15  # 早停耐心值（数据量大可以增加）
SCHEDULER_PATIENCE_MULTI = 7  # 学习率调度耐心值

# LOSO可能每折数据较少，适当调整
NUM_EPOCHS_LOSO = 80
EARLY_STOPPING_PATIENCE_LOSO = 12


# ============================================
# 显示配置信息
# ============================================
def print_config():
    """打印当前配置信息"""
    print("=" * 60)
    print("多被试者实验配置")
    print("=" * 60)

    if USE_LOSO:
        print(f"模式: LOSO留一交叉验证")
        print(f"总被试者数: {len(ALL_SUBJECTS)}")
        print(f"交叉验证折数: {len(ALL_SUBJECTS)}")
        print(
            f"每折 - 测试: 1个被试者, 验证: 2个被试者, 训练: {len(ALL_SUBJECTS) - 3}个被试者"
        )
    else:
        print(f"模式: 固定被试者划分")
        print(f"训练集被试者 ({len(TRAIN_SUBJECTS)}个): {', '.join(TRAIN_SUBJECTS)}")
        print(f"验证集被试者 ({len(VAL_SUBJECTS)}个): {', '.join(VAL_SUBJECTS)}")
        print(f"测试集被试者 ({len(TEST_SUBJECTS)}个): {', '.join(TEST_SUBJECTS)}")

    print(f"\n数据平衡策略: {BALANCE_STRATEGY}")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"输出目录: {OUTPUT_DIR_MULTI if not USE_LOSO else OUTPUT_DIR_LOSO}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
