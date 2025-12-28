"""
多被试者数据预处理器
支持批量加载多个被试者的数据，并跟踪样本来源
"""

import os
import sys
import warnings

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 过滤 MNE 的重复通道名警告（这是预期行为）
warnings.filterwarnings("ignore", message="Channel names are not unique")


# 导入单被试者预处理器中的函数
from data.preprocessor import (
    EEGDataset,
    SeizureAnnotation,
    apply_stockwell_transform,
    load_eeg_data,
)
from data.preprocessor import (
    extract_interictal_segments as extract_interictal_from_dict,
)
from data.preprocessor import (
    extract_preictal_segments as extract_preictal_from_dict,
)


class MultiSubjectEEGDataset(Dataset):
    """
    多被试者EEG数据集
    除了返回数据和标签，还返回被试者ID用于分析
    """

    def __init__(self, data, labels, subject_ids):
        """
        Args:
            data: (N, C, H, W) 时频特征
            labels: (N,) 标签
            subject_ids: (N,) 被试者ID列表
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids  # 保持为list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_ids[idx]


def load_subject_data(
    subject,
    data_root,
    channels,
    sampling_rate,
    filter_low,
    filter_high,
    window_size,
    sop,
    sph,
):
    """
    加载单个被试者的数据

    Returns:
        preictal_features: (N_pre, C, H, W)
        interictal_features: (N_inter, C, H, W)
        subject_id: str
    """
    print(f"\n{'=' * 60}")
    print(f"加载被试者: {subject}")
    print(f"{'=' * 60}")

    subject_dir = os.path.join(data_root, subject)

    # 读取发作标注（SeizureAnnotation接收目录路径）
    annotation = SeizureAnnotation(subject_dir)

    # 统计总发作次数
    total_seizures = sum(len(seizures) for seizures in annotation.annotations.values())
    if total_seizures == 0:
        print(f"警告: {subject} 没有发作记录，跳过该被试者")
        return None, None, subject

    print(f"发作次数: {total_seizures}")

    # 使用preprocessor中的函数加载整个被试者的EEG数据
    print("加载EEG数据...")
    eeg_data = load_eeg_data(subject_dir, channels)

    if len(eeg_data) == 0:
        print(f"警告: {subject} 没有有效的EDF文件，跳过该被试者")
        return None, None, subject

    print(f"EDF文件数: {len(eeg_data)}")

    # 提取发作前数据
    try:
        preictal_segments = extract_preictal_from_dict(eeg_data, annotation)
    except ValueError as e:
        print(f"警告: {subject} 提取发作前数据失败: {e}")
        return None, None, subject

    # 提取等量的非发作数据
    interictal_segments = extract_interictal_from_dict(
        eeg_data, annotation, len(preictal_segments)
    )

    # 提取等量的非发作数据
    interictal_segments = extract_interictal_from_dict(
        eeg_data, annotation, len(preictal_segments)
    )

    print(f"\n提取的片段数:")
    print(f"  - Pre-ictal: {len(preictal_segments)}")
    print(f"  - Inter-ictal: {len(interictal_segments)}")

    # 应用S-transform（批量处理）
    print("应用S-transform...")
    preictal_features = apply_stockwell_transform(preictal_segments)
    interictal_features = apply_stockwell_transform(interictal_segments)

    print(f"\nS-transform后的特征形状:")
    print(f"  - Pre-ictal: {preictal_features.shape}")
    print(f"  - Inter-ictal: {interictal_features.shape}")

    return preictal_features, interictal_features, subject


def prepare_multi_subject_data(
    subjects,
    data_root,
    channels,
    sampling_rate,
    filter_low,
    filter_high,
    window_size,
    sop,
    sph,
    balance_strategy="per_subject",
    normalization="z-score",  # 新增：跨被试者标准化方法
):
    """
    准备多被试者数据

    Args:
        subjects: 被试者ID列表
        balance_strategy: "per_subject" 或 "global"
            - "per_subject": 每个被试者内部平衡后再合并
            - "global": 先合并再全局平衡
        normalization: "z-score", "min-max", "robust", 或 None
            - 跨被试者标准化，减少个体差异

    Returns:
        all_data: (N, C, H, W)
        all_labels: (N,)
        all_subject_ids: (N,) list of subject IDs
    """
    print("\n" + "=" * 60)
    print("多被试者数据准备")
    print("=" * 60)
    print(f"被试者列表: {subjects}")
    print(f"数据平衡策略: {balance_strategy}")
    print(f"标准化方法: {normalization}")
    print("=" * 60)

    all_preictal = []
    all_interictal = []
    all_subject_ids_pre = []
    all_subject_ids_inter = []

    # 加载所有被试者的数据
    for subject in subjects:
        pre_feat, inter_feat, subj_id = load_subject_data(
            subject,
            data_root,
            channels,
            sampling_rate,
            filter_low,
            filter_high,
            window_size,
            sop,
            sph,
        )

        if pre_feat is None or len(pre_feat) == 0:
            continue

        if balance_strategy == "per_subject":
            # 每个被试者内部平衡
            n_pre = len(pre_feat)
            n_inter = len(inter_feat)
            n_samples = min(n_pre, n_inter)

            if n_samples == 0:
                print(f"警告: {subject} 没有足够的样本，跳过")
                continue

            # 随机采样
            if n_pre > n_samples:
                indices = np.random.choice(n_pre, n_samples, replace=False)
                pre_feat = pre_feat[indices]

            if n_inter > n_samples:
                indices = np.random.choice(n_inter, n_samples, replace=False)
                inter_feat = inter_feat[indices]

            print(f"{subject} 平衡后: Pre={len(pre_feat)}, Inter={len(inter_feat)}")

        all_preictal.append(pre_feat)
        all_interictal.append(inter_feat)
        all_subject_ids_pre.extend([subj_id] * len(pre_feat))
        all_subject_ids_inter.extend([subj_id] * len(inter_feat))

    if len(all_preictal) == 0:
        raise ValueError("没有加载到任何有效数据！")

    # 合并所有被试者的数据
    all_preictal = np.concatenate(all_preictal, axis=0)
    all_interictal = np.concatenate(all_interictal, axis=0)

    print(f"\n合并后的数据形状:")
    print(f"  - Pre-ictal: {all_preictal.shape}")
    print(f"  - Inter-ictal: {all_interictal.shape}")

    if balance_strategy == "global":
        # 全局平衡
        n_pre = len(all_preictal)
        n_inter = len(all_interictal)
        n_samples = min(n_pre, n_inter)

        if n_pre > n_samples:
            indices = np.random.choice(n_pre, n_samples, replace=False)
            all_preictal = all_preictal[indices]
            all_subject_ids_pre = [all_subject_ids_pre[i] for i in indices]

        if n_inter > n_samples:
            indices = np.random.choice(n_inter, n_samples, replace=False)
            all_interictal = all_interictal[indices]
            all_subject_ids_inter = [all_subject_ids_inter[i] for i in indices]

        print(f"\n全局平衡后:")
        print(f"  - Pre-ictal: {all_preictal.shape}")
        print(f"  - Inter-ictal: {all_interictal.shape}")

    # 合并并创建标签
    all_data = np.concatenate([all_preictal, all_interictal], axis=0)
    all_labels = np.concatenate(
        [np.ones(len(all_preictal)), np.zeros(len(all_interictal))]
    )
    all_subject_ids = all_subject_ids_pre + all_subject_ids_inter

    # 跨被试者标准化
    if normalization is not None:
        from data.cross_subject_normalization import apply_cross_subject_normalization
        
        print(f"\n应用跨被试者标准化: {normalization}")
        print(f"  标准化前: mean={all_data.mean():.4f}, std={all_data.std():.4f}, "
              f"range=[{all_data.min():.4f}, {all_data.max():.4f}]")
        
        all_data = apply_cross_subject_normalization(all_data, method=normalization)
        
        print(f"  标准化后: mean={all_data.mean():.4f}, std={all_data.std():.4f}, "
              f"range=[{all_data.min():.4f}, {all_data.max():.4f}]")

    print(f"\n最终数据集:")
    print(f"  - 总样本数: {len(all_data)}")
    print(f"  - Pre-ictal: {np.sum(all_labels == 1)}")
    print(f"  - Inter-ictal: {np.sum(all_labels == 0)}")
    print(f"  - 数据形状: {all_data.shape}")

    # 统计每个被试者的样本数
    from collections import Counter

    subject_counts = Counter(all_subject_ids)
    print(f"\n每个被试者的样本数:")
    for subj, count in sorted(subject_counts.items()):
        print(f"  - {subj}: {count}")

    return all_data, all_labels, all_subject_ids


def create_subject_split_dataloaders(
    train_subjects,
    val_subjects,
    test_subjects,
    data_root,
    channels,
    sampling_rate,
    filter_low,
    filter_high,
    window_size,
    sop,
    sph,
    batch_size,
    balance_strategy="per_subject",
    normalization="z-score",  # 新增：跨被试者标准化方法
):
    """
    创建按被试者划分的数据加载器

    Args:
        normalization: "z-score", "min-max", "robust", 或 None
            跨被试者标准化，减少个体差异

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "=" * 60)
    print("创建被试者级别划分的数据加载器")
    print("=" * 60)

    # 准备训练集
    print("\n【训练集】")
    train_data, train_labels, train_subject_ids = prepare_multi_subject_data(
        train_subjects,
        data_root,
        channels,
        sampling_rate,
        filter_low,
        filter_high,
        window_size,
        sop,
        sph,
        balance_strategy,
        normalization,  # 传递标准化参数
    )
    train_dataset = MultiSubjectEEGDataset(train_data, train_labels, train_subject_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 准备验证集
    print("\n【验证集】")
    val_data, val_labels, val_subject_ids = prepare_multi_subject_data(
        val_subjects,
        data_root,
        channels,
        sampling_rate,
        filter_low,
        filter_high,
        window_size,
        sop,
        sph,
        balance_strategy,
        normalization,  # 传递标准化参数
    )
    val_dataset = MultiSubjectEEGDataset(val_data, val_labels, val_subject_ids)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 准备测试集
    print("\n【测试集】")
    test_data, test_labels, test_subject_ids = prepare_multi_subject_data(
        test_subjects,
        data_root,
        channels,
        sampling_rate,
        filter_low,
        filter_high,
        window_size,
        sop,
        sph,
        balance_strategy,
        normalization,  # 传递标准化参数
    )
    test_dataset = MultiSubjectEEGDataset(test_data, test_labels, test_subject_ids)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("数据加载器创建完成")
    print("=" * 60)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print("=" * 60)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """简单测试：加载少量被试者验证数据流程"""
    import sys

    sys.path.append("..")

    # 导入配置
    import config_multi_subject as cfg

    print("\n" + "=" * 60)
    print("多被试者数据预处理器 - 测试模式")
    print("=" * 60)

    # 使用少量被试者进行快速测试
    test_train_subjects = cfg.TRAIN_SUBJECTS[:2]  # 只用前2个训练被试者
    test_val_subjects = cfg.VAL_SUBJECTS[:1]  # 只用1个验证被试者
    test_test_subjects = cfg.TEST_SUBJECTS[:1]  # 只用1个测试被试者

    print(f"\n测试配置:")
    print(f"  - 训练被试者: {test_train_subjects}")
    print(f"  - 验证被试者: {test_val_subjects}")
    print(f"  - 测试被试者: {test_test_subjects}")
    print(f"  - 数据平衡策略: {cfg.BALANCE_STRATEGY}")
    print(f"  - 批次大小: {cfg.BATCH_SIZE}")

    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_subject_split_dataloaders(
            test_train_subjects,
            test_val_subjects,
            test_test_subjects,
            cfg.DATA_ROOT,
            cfg.CHANNEL_NAMES,
            cfg.SAMPLING_RATE,
            cfg.FILTER_LOW,
            cfg.FILTER_HIGH,
            cfg.WINDOW_SIZE,
            cfg.SOP,
            cfg.SPH,
            cfg.BATCH_SIZE,
            cfg.BALANCE_STRATEGY,
        )

        print("\n" + "=" * 60)
        print("测试成功！数据加载器已创建")
        print("=" * 60)

        # 测试一个批次
        print("\n测试读取一个批次...")
        data, labels, subject_ids = next(iter(train_loader))
        print(f"  - 数据形状: {data.shape}")
        print(f"  - 标签形状: {labels.shape}")
        print(f"  - 被试者ID数量: {len(subject_ids)}")
        print(f"  - 批次中的被试者: {set(subject_ids)}")
        print(f"  - Pre-ictal样本数: {labels.sum().item()}")
        print(f"  - Inter-ictal样本数: {(labels == 0).sum().item()}")

        print("\n✓ 所有测试通过！")

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
