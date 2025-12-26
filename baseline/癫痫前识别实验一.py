import os
import random
import matplotlib
matplotlib.use('Qt5Agg')  # 使用交互式后端
import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import resample


# 定义文件夹路径
edf_folder_path = "D:\\ML\\chb-mit-scalp-eeg-database-1.0.0\\chb01"

# 定义“PreSeizure”标签的时间区间（单位：秒）
pre_seizure_times = {
    "chb01_03.edf": [(2986, 2996)],
    "chb01_04.edf": [(1457, 1467)],
    "chb01_15.edf": [(1722, 1732)],
    "chb01_16.edf": [(1005, 1015)],
    "chb01_18.edf": [(1710, 1720)],
    "chb01_21.edf": [(317, 327)],
    "chb01_26.edf": [(1852, 1862)],
}

# 滑动窗口参数
window_size = 10  # 窗口大小（秒）
step_size = 5     # 步长（秒）

# 特征提取函数
def extract_features(segment):
    # 能量时间变化率
    energy_rate = np.mean(np.diff(np.square(segment)))
    # 频谱质心
    spectrum = np.abs(np.fft.fft(segment))
    freqs = np.fft.fftfreq(len(segment))
    spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    # 频谱熵
    spectrum_norm = spectrum / np.sum(spectrum)
    spectral_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-12))
    return [energy_rate, spectral_centroid, spectral_entropy]

# 遍历文件夹中的所有 EDF 文件
samples = []
labels = []
for edf_file in os.listdir(edf_folder_path):
    if edf_file.endswith(".edf"):
        edf_path = os.path.join(edf_folder_path, edf_file)
        print(f"正在处理文件：{edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # 选择目标通道 C3-P3
        target_channels = ['C3-P3']
        available_channels = [ch for ch in target_channels if ch in raw.ch_names]
        if len(available_channels) < 1:
            print(f"文件 {edf_file} 缺少目标通道，跳过。")
            continue

        # 使用 inst.pick() 保留目标通道
        raw.pick(available_channels)
        print(f"保留的通道名：{raw.ch_names}")

        # 滤波处理：0.5 Hz 到 25 Hz 带通滤波
        raw.filter(0.5, 25, fir_design='firwin', verbose=False)

        # 获取采样频率
        sfreq = raw.info['sfreq']
        data, _ = raw.get_data(return_times=True)

        # 信号归一化：Min-Max 归一化
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)

        # 滑动窗口分段
        total_time = int(raw.times[-1])
        for start_time in range(0, total_time - window_size + 1, step_size):
            end_time = start_time + window_size
            start_sample = int(start_time * sfreq)
            end_sample = int(end_time * sfreq)
            segment = data[:, start_sample:end_sample]

            # 跳过异常段
            if segment.shape[1] != window_size * sfreq:
                continue

            # 提取特征
            features = extract_features(segment[0])  # 仅对 C3-P3 通道提取特征

            # 标注
            label = "non-PreSeizure"
            if edf_file in pre_seizure_times:
                for seizure_start, seizure_end in pre_seizure_times[edf_file]:
                    if seizure_start <= start_time < seizure_end:
                        label = "PreSeizure"
                        break

            samples.append(features)
            labels.append(label)

# 打印样本类别分布
print("原始数据类别分布：", Counter(labels))

# 数据转换为数组
samples = np.array(samples)
labels = np.array(labels)

# 分离类别
pre_seizure_samples = samples[labels == "PreSeizure"]
non_pre_seizure_samples = samples[labels == "non-PreSeizure"]

# 随机抽取 10000 段 non-PreSeizure 样本
non_pre_seizure_samples = resample(non_pre_seizure_samples,
                                   replace=False,
                                   n_samples=29000,
                                   random_state=42)

# 合并所有 PreSeizure 样本和抽取的 non-PreSeizure 样本
reduced_samples = np.vstack([pre_seizure_samples, non_pre_seizure_samples])
reduced_labels = np.array(["PreSeizure"] * len(pre_seizure_samples) +
                          ["non-PreSeizure"] * len(non_pre_seizure_samples))

# 数据平衡：过采样 PreSeizure 类别
pre_seizure_samples_resampled = resample(pre_seizure_samples,
                                         replace=True,
                                         n_samples=len(non_pre_seizure_samples),
                                         random_state=42)

balanced_samples = np.vstack([pre_seizure_samples_resampled, non_pre_seizure_samples])
balanced_labels = np.array(["PreSeizure"] * len(pre_seizure_samples_resampled) +
                           ["non-PreSeizure"] * len(non_pre_seizure_samples))

# 打印平衡后的类别分布
print("平衡后数据类别分布：", Counter(balanced_labels))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    balanced_samples, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels)

# 训练 SVM 模型
clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 测试模型
y_pred = clf.predict(X_test)
print("分类报告：")
print(classification_report(y_test, y_pred, zero_division=0))