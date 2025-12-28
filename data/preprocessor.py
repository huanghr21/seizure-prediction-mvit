"""
数据预处理模块
包含：发作时间标注读取、EEG数据加载、窗口分割、S-transform、Dataset类
"""

import os
import sys
import re
import warnings
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 过滤 MNE 的重复通道名警告（这是预期行为）
warnings.filterwarnings('ignore', message='Channel names are not unique')

import mne
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

import config


class SeizureAnnotation:
    """发作时间标注类"""

    def __init__(self, subject_path: str):
        """
        从summary文件读取发作时间标注

        Args:
            subject_path: 受试者数据目录路径
        """
        self.subject_path = subject_path
        self.annotations = {}  # {filename: [(start_time, end_time), ...]}
        self._load_annotations()

    def _load_annotations(self):
        """加载发作时间标注"""
        summary_file = os.path.join(
            self.subject_path, f"{os.path.basename(self.subject_path)}-summary.txt"
        )

        if not os.path.exists(summary_file):
            print(f"Warning: Summary file not found: {summary_file}")
            return

        with open(summary_file, "r") as f:
            lines = f.readlines()

        current_file = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 匹配文件名
            if line.startswith("File Name:"):
                current_file = line.split(":")[-1].strip()
                if current_file not in self.annotations:
                    self.annotations[current_file] = []

            # 匹配发作开始时间
            elif "Seizure Start Time:" in line or "Seizure 1 Start Time:" in line:
                match = re.search(r"(\d+)\s+seconds", line)
                if match and current_file:
                    start_time = int(match.group(1))
                    # 查找对应的结束时间（在后续行中）
                    for j in range(i + 1, min(i + 10, len(lines))):  # 向后查找最多10行
                        next_line = lines[j].strip()
                        if (
                            "Seizure End Time:" in next_line
                            or "Seizure 1 End Time:" in next_line
                        ):
                            end_match = re.search(r"(\d+)\s+seconds", next_line)
                            if end_match:
                                end_time = int(end_match.group(1))
                                self.annotations[current_file].append(
                                    (start_time, end_time)
                                )
                                print(f"  Found seizure in {current_file}: {start_time}s - {end_time}s")
                            break
            i += 1

        print(f"\nLoaded annotations for {len(self.annotations)} files")
        total_seizures = sum(len(seizures) for seizures in self.annotations.values())
        print(f"Total seizures found: {total_seizures}")
        for filename, seizures in self.annotations.items():
            print(f"  {filename}: {len(seizures)} seizure(s)")

    def get_seizure_times(self, filename: str) -> List[Tuple[int, int]]:
        """
        获取指定文件的发作时间列表

        Args:
            filename: EDF文件名

        Returns:
            [(start_time, end_time), ...] 单位：秒
        """
        return self.annotations.get(filename, [])


def load_eeg_data(subject_path: str, channel_names: List[str]) -> Dict[str, np.ndarray]:
    """
    加载指定受试者的所有EDF文件并应用滤波

    Args:
        subject_path: 受试者数据目录路径
        channel_names: 要提取的通道名称列表

    Returns:
        {filename: (n_channels, n_samples)} 字典
    """
    eeg_data = {}

    # 获取所有EDF文件
    edf_files = [f for f in os.listdir(subject_path) if f.endswith(".edf")]
    edf_files.sort()

    print(f"Loading EEG data from {len(edf_files)} files...")

    for edf_file in edf_files:
        edf_path = os.path.join(subject_path, edf_file)

        try:
            # 读取EDF文件（关闭重复警告和缩放因子警告）
            # CHB-MIT数据集中部分文件的占位通道缺少缩放因子，这不影响实际EEG通道
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Scaling factor is not defined')
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # 处理重复的通道名：MNE会自动为所有重复的通道添加编号
            # 例如两个 T8-P8 会被重命名为 T8-P8-0 和 T8-P8-1
            available_channels = []
            channel_count = {}  # 跟踪每个通道名出现的次数
            
            for ch_name in channel_names:
                # 初始化计数
                if ch_name not in channel_count:
                    channel_count[ch_name] = 0
                
                # 先检查原始名称是否存在（没有重复的情况）
                if ch_name in raw.ch_names:
                    actual_name = ch_name
                else:
                    # 如果原始名称不存在，说明有重复，MNE添加了后缀
                    # MNE为重复通道添加 -0, -1, -2, ... 后缀
                    actual_name = f"{ch_name}-{channel_count[ch_name]}"
                
                # 检查通道是否存在
                if actual_name in raw.ch_names:
                    available_channels.append(actual_name)
                    channel_count[ch_name] += 1
                # 注意：如果通道不存在，继续检查下一个通道（不break）
                # 某些被试者可能缺少部分通道，但只要总数足够即可
            
            if len(available_channels) < config.N_CHANNELS:
                # 有些文件可能通道数不足，跳过
                print(f"  Warning: {edf_file} has only {len(available_channels)}/{config.N_CHANNELS} channels, skipping")
                continue

            # 选择通道
            raw.pick(available_channels)

            # 应用带通滤波 0.5-48 Hz
            raw.filter(
                config.FILTER_LOW,
                config.FILTER_HIGH,
                fir_design="firwin",
                verbose=False,
            )

            # 获取数据
            data, _ = raw.get_data(return_times=True)

            # 归一化（Min-Max）
            data = (data - np.min(data, axis=1, keepdims=True)) / (
                np.max(data, axis=1, keepdims=True)
                - np.min(data, axis=1, keepdims=True)
                + 1e-12
            )

            eeg_data[edf_file] = data
            print(f"  Loaded {edf_file}: shape={data.shape}")

        except Exception as e:
            print(f"  Error loading {edf_file}: {e}")

    return eeg_data


def extract_preictal_segments(
    eeg_data: Dict[str, np.ndarray], annotations: SeizureAnnotation
) -> np.ndarray:
    """
    提取发作前数据段（发作前30分钟到发作前3分钟）

    Args:
        eeg_data: {filename: (n_channels, n_samples)}
        annotations: 发作时间标注对象

    Returns:
        (n_samples, n_channels, window_samples) 的numpy数组
    """
    preictal_segments = []

    print(f"\nExtracting pre-ictal segments...")
    
    # 调试信息
    print(f"Available EEG files: {len(eeg_data)}")
    print(f"Files with annotations: {len([f for f in eeg_data.keys() if annotations.get_seizure_times(f)])}")

    for filename, data in eeg_data.items():
        seizure_times = annotations.get_seizure_times(filename)

        if not seizure_times:
            continue
        
        print(f"Processing {filename}: {len(seizure_times)} seizure(s)")

        n_channels, n_samples = data.shape
        total_duration = n_samples / config.SAMPLING_RATE

        for seizure_start, seizure_end in seizure_times:
            # 定义发作前区间：[发作前30min, 发作前3min]
            preictal_start = max(0, seizure_start - config.SOP)
            preictal_end = max(0, seizure_start - config.SPH)
            
            print(f"  Seizure at {seizure_start}s, pre-ictal: {preictal_start}s - {preictal_end}s")

            if preictal_end <= preictal_start:
                print(f"  Warning: Invalid pre-ictal period, skipping")
                continue

            # 转换为采样点
            start_sample = int(preictal_start * config.SAMPLING_RATE)
            end_sample = int(preictal_end * config.SAMPLING_RATE)

            # 使用4秒无重叠滑动窗口
            window_samples = config.WINDOW_SAMPLES
            
            n_windows_before = len(preictal_segments)

            for i in range(
                start_sample, end_sample - window_samples + 1, window_samples
            ):
                segment = data[:, i : i + window_samples]

                # 检查窗口完整性
                if segment.shape[1] == window_samples:
                    preictal_segments.append(segment)
            
            n_windows_extracted = len(preictal_segments) - n_windows_before
            print(f"    Extracted {n_windows_extracted} windows from this seizure")

    if len(preictal_segments) == 0:
        print("\nError: No pre-ictal segments found!")
        print("Possible reasons:")
        print("  1. No seizure annotations loaded")
        print("  2. All pre-ictal periods are too short (< 27 min)")
        print("  3. EDF files don't match annotation file names")
        raise ValueError("No pre-ictal segments found!")

    preictal_segments = np.array(preictal_segments)
    print(f"Total pre-ictal segments: {preictal_segments.shape[0]}")

    return preictal_segments


def extract_interictal_segments(
    eeg_data: Dict[str, np.ndarray], annotations: SeizureAnnotation, n_samples: int
) -> np.ndarray:
    """
    从非发作期随机采样数据段

    Args:
        eeg_data: {filename: (n_channels, n_samples)}
        annotations: 发作时间标注对象
        n_samples: 需要采样的段数（与发作前数据相等）

    Returns:
        (n_samples, n_channels, window_samples) 的numpy数组
    """
    interictal_segments = []

    print(f"\nExtracting inter-ictal segments (target: {n_samples})...")

    # 收集所有可用的非发作期窗口
    available_windows = []

    for filename, data in eeg_data.items():
        seizure_times = annotations.get_seizure_times(filename)

        n_channels, n_samples_file = data.shape
        total_duration = n_samples_file / config.SAMPLING_RATE

        # 创建时间掩码（标记哪些时间段不能用）
        excluded_mask = np.zeros(int(total_duration), dtype=bool)

        for seizure_start, seizure_end in seizure_times:
            # 排除：发作前30分钟、发作期、发作后5分钟
            exclude_start = max(0, seizure_start - config.SOP)
            exclude_end = min(total_duration, seizure_end + config.POST_ICTAL)
            excluded_mask[int(exclude_start) : int(exclude_end)] = True

        # 找到可用的窗口位置
        window_samples = config.WINDOW_SAMPLES
        for i in range(0, n_samples_file - window_samples + 1, window_samples):
            start_time = i / config.SAMPLING_RATE
            end_time = (i + window_samples) / config.SAMPLING_RATE

            # 检查窗口是否完全在非发作期
            if not np.any(excluded_mask[int(start_time) : int(end_time)]):
                available_windows.append((filename, i))

    print(f"Found {len(available_windows)} available inter-ictal windows")

    # 随机采样
    if len(available_windows) < n_samples:
        print(
            f"Warning: Only {len(available_windows)} windows available, less than requested {n_samples}"
        )
        n_samples = len(available_windows)

    np.random.seed(config.RANDOM_SEED)
    selected_indices = np.random.choice(
        len(available_windows), n_samples, replace=False
    )

    for idx in selected_indices:
        filename, start_idx = available_windows[idx]
        data = eeg_data[filename]
        segment = data[:, start_idx : start_idx + config.WINDOW_SAMPLES]
        interictal_segments.append(segment)

    interictal_segments = np.array(interictal_segments)
    print(f"Total inter-ictal segments: {interictal_segments.shape[0]}")

    return interictal_segments


def stockwell_transform(signal_data: np.ndarray) -> np.ndarray:
    """
    对单通道信号应用Stockwell Transform

    Args:
        signal_data: (n_samples,) 一维信号

    Returns:
        (time_bins, freq_bins) 时频表示
    """
    from scipy.signal import stft

    # 使用STFT近似S-transform（简化实现）
    # 如果需要真正的S-transform，可以使用stockwell库
    f, t, Zxx = stft(signal_data, fs=config.SAMPLING_RATE, nperseg=256, noverlap=128)

    # 取幅值的平方（能量）
    magnitude = np.abs(Zxx) ** 2

    # 选择0-48Hz的频率范围
    freq_mask = (f >= config.FREQ_MIN) & (f <= config.FREQ_MAX)
    magnitude = magnitude[freq_mask, :]

    # 调整大小到目标尺寸 (1024, 192)
    from scipy.ndimage import zoom

    target_shape = (config.ST_FREQ_BINS, config.ST_TIME_BINS)
    zoom_factors = (
        target_shape[0] / magnitude.shape[0],
        target_shape[1] / magnitude.shape[1],
    )
    magnitude = zoom(magnitude, zoom_factors, order=1)

    return magnitude.T  # 返回 (time, freq)


def apply_stockwell_transform(segments: np.ndarray) -> np.ndarray:
    """
    对所有段的所有通道应用S-transform并压缩

    Args:
        segments: (n_samples, n_channels, window_samples)

    Returns:
        (n_samples, n_channels, 32, 32) 压缩后的时频特征
    """
    n_samples, n_channels, _ = segments.shape
    features = np.zeros(
        (n_samples, n_channels, config.FEATURE_SIZE, config.FEATURE_SIZE)
    )

    print(f"\nApplying S-transform to {n_samples} segments...")

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{n_samples}...")

        for ch in range(n_channels):
            # S-transform
            st_result = stockwell_transform(segments[i, ch, :])  # (1024, 192)

            # 压缩：时间维度32倍，频率维度6倍
            compressed = st_result.reshape(
                config.FEATURE_SIZE,
                config.TIME_COMPRESSION,
                config.FEATURE_SIZE,
                config.FREQ_COMPRESSION,
            ).sum(axis=(1, 3))

            features[i, ch, :, :] = compressed

    print("S-transform completed.")
    return features


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (n_samples, n_channels, 32, 32)
            labels: (n_samples,) 0 for inter-ictal, 1 for pre-ictal
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_data(subject_path: str) -> Tuple[EEGDataset, EEGDataset, EEGDataset]:
    """
    准备训练、验证和测试数据集的完整流程

    Args:
        subject_path: 受试者数据目录路径

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print("=" * 50)
    print("Starting data preparation...")
    print("=" * 50)

    # 1. 加载发作时间标注
    annotations = SeizureAnnotation(subject_path)

    # 2. 加载EEG数据
    eeg_data = load_eeg_data(subject_path, config.CHANNEL_NAMES)

    # 3. 提取发作前数据
    preictal_segments = extract_preictal_segments(eeg_data, annotations)

    # 4. 提取等量的非发作数据
    interictal_segments = extract_interictal_segments(
        eeg_data, annotations, len(preictal_segments)
    )

    # 5. 应用S-transform
    preictal_features = apply_stockwell_transform(preictal_segments)
    interictal_features = apply_stockwell_transform(interictal_segments)

    # 6. 合并数据和标签
    all_features = np.concatenate([preictal_features, interictal_features], axis=0)
    all_labels = np.concatenate(
        [
            np.ones(len(preictal_features)),  # 1 for pre-ictal
            np.zeros(len(interictal_features)),  # 0 for inter-ictal
        ]
    )

    print(f"\nTotal samples: {len(all_features)}")
    print(f"  Pre-ictal: {np.sum(all_labels == 1)}")
    print(f"  Inter-ictal: {np.sum(all_labels == 0)}")

    # 7. 划分训练集、验证集和测试集（60% / 20% / 20%）
    from sklearn.model_selection import train_test_split

    # 首先划分出训练集（60%）
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_features,
        all_labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=all_labels,
    )

    # 再将剩余的40%划分成验证集和测试集（各20%）
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,  # 50%的临时数据 = 20%的总数据
        random_state=config.RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"\nTrain set: {len(X_train)} samples ({len(X_train)/len(all_features)*100:.1f}%)")
    print(f"Val set: {len(X_val)} samples ({len(X_val)/len(all_features)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(all_features)*100:.1f}%)")

    # 8. 创建Dataset对象
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    print("\nData preparation completed!")
    print("=" * 50)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # 测试数据预处理流程
    train_ds, val_ds, test_ds = prepare_data(config.SUBJECT_PATH)
    print(f"\nDataset shapes:")
    print(f"  Train: {train_ds.features.shape}, {train_ds.labels.shape}")
    print(f"  Val: {val_ds.features.shape}, {val_ds.labels.shape}")
    print(f"  Test: {test_ds.features.shape}, {test_ds.labels.shape}")
