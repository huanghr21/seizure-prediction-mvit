import os
import mne
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample

# 数据路径
base_folder = "D:\\ML\\chb-mit-scalp-eeg-database-1.0.0"

# 滑动窗口参数
window_size = 10  # 秒
step_size = 5     # 秒

# 目标通道
target_channels = ['FP2-F8', 'FP2-F4', 'C3-P3']

# 癫痫前片段时间段 (pre_seizure_times)
pre_seizure_times = {
    "chb01/chb01_03.edf": [(2986, 2996)],
    "chb01/chb01_04.edf": [(1457, 1467)],
    "chb01/chb01_15.edf": [(1722, 1732)],
    "chb01/chb01_16.edf": [(1005, 1015)],
    "chb01/chb01_18.edf": [(1710, 1720)],
    "chb01/chb01_21.edf": [(317, 327)],
    "chb01/chb01_26.edf": [(1852, 1862)],
    "chb02/chb02_16.edf": [(120, 130)],
    "chb02/chb02_16+.edf": [(2962, 2972)],
    "chb02/chb02_19.edf": [(3359, 3369)],
    "chb03/chb03_01.edf": [(352, 362)],
    "chb03/chb03_02.edf": [(721, 731)],
    "chb03/chb03_03.edf": [(422, 432)],
    "chb03/chb03_04.edf": [(2162, 2172)],
    "chb03/chb03_34.edf": [(1972, 1982)],
    "chb03/chb03_35.edf": [(2582, 2592)],
    "chb03/chb03_36.edf": [(1715, 1725)],
    "chb04/chb04_05.edf": [(7794, 7804)],
    "chb04/chb04_08.edf": [(6436, 6446)],
    "chb04/chb04_28.edf": [(1669, 1679), (3772, 3782)],
    "chb05/chb05_06.edf": [(407, 417)],
    "chb05/chb05_13.edf": [(1076, 1086)],
    "chb05/chb05_16.edf": [(2307, 2317)],
    "chb05/chb05_17.edf": [(2441, 2451)],
    "chb05/chb05_22.edf": [(2338, 2348)],
    "chb06/chb06_01.edf": [(1714, 1724), (7451, 7461), (13515, 13525)],
    "chb06/chb06_04.edf": [(317, 327), (6201, 6211)],
    "chb06/chb06_09.edf": [(12490, 12500)],
    "chb06/chb06_10.edf": [(10823, 10833)],
    "chb06/chb06_13.edf": [(496, 506)],
    "chb06/chb06_18.edf": [(7789, 7799)],
    "chb06/chb06_24.edf": [(9377, 9387)],
    "chb07/chb07_12.edf": [(4910, 4920)],
    "chb07/chb07_13.edf": [(3275, 3285)],
    "chb07/chb07_18.edf": [(13678, 13688)],
    "chb08/chb08_02.edf": [(2660, 2670)],
    "chb08/chb08_05.edf": [(2846, 2856)],
    "chb08/chb08_11.edf": [(2978, 2988)],
    "chb08/chb08_13.edf": [(2407, 2417)],
    "chb08/chb08_21.edf": [(2073, 2083)],
    "chb09/chb09_06.edf": [(12221, 12231)],
    "chb09/chb09_08.edf": [(2941, 2951), (9186, 9196)],
    "chb09/chb09_19.edf": [(5289, 5299)],
    "chb10/chb10_12.edf": [(6303, 6313)],
    "chb10/chb10_20.edf": [(6878, 6888)],
    "chb10/chb10_27.edf": [(2372, 2382)],
    "chb10/chb10_30.edf": [(3011, 3021)],
    "chb10/chb10_31.edf": [(3791, 3801)],
    "chb10/chb10_38.edf": [(4608, 4618)],
    "chb10/chb10_89.edf": [(1373, 1383)],
    "chb11/chb11_82.edf": [(288, 298)],
    "chb11/chb11_92.edf": [(2685, 2695)],
    "chb11/chb11_99.edf": [(1444, 1454)],
    "chb12/chb12_06.edf": [(1655, 1665), (3405, 3415)],
    "chb12/chb12_08.edf": [(1416, 1426), (1581, 1591), (1947, 1957), (2788, 2798)],
    "chb12/chb12_09.edf": [(3072, 3082), (3493, 3503)],
    "chb12/chb12_10.edf": [(583, 593), (801, 811)],
    "chb12/chb12_11.edf": [(1075, 1085)],
    "chb12/chb12_23.edf": [(243, 253), (415, 425), (620, 630)],
    "chb12/chb12_27.edf": [(906, 916), (1087, 1097), (1718, 1728), (1911, 1921), (2378, 2388), (2611, 2621)],
    "chb12/chb12_28.edf": [(171, 181)],
    "chb12/chb12_29.edf": [(97, 107), (544, 554), (1153, 1163), (1391, 1401), (1874, 1884), (3547, 3557)],
    "chb12/chb12_33.edf": [(2175, 2185), (2417, 2427)],
    "chb12/chb12_36.edf": [(643, 653)],
    "chb12/chb12_38.edf": [(1538, 1548), (2788, 2798), (2956, 2966), (3136, 3146), (3354, 3364)],
    "chb12/chb12_42.edf": [(689, 699), (935, 945), (1160, 1170), (1189, 1199), (1666, 1676), (2203, 2213)],
    "chb13/chb13_19.edf": [(2067, 2077)],
    "chb13/chb13_21.edf": [(924, 934)],
    "chb13/chb13_40.edf": [(132, 142), (520, 530)],
    "chb13/chb13_55.edf": [(448, 458), (2426, 2436)],
    "chb13/chb13_58.edf": [(2464, 2474)],
    "chb13/chb13_59.edf": [(3329, 3339)],
    "chb13/chb13_60.edf": [(628, 638)],
    "chb13/chb13_62.edf": [(841, 851), (1616, 1626), (2654, 2664)],
    "chb14/chb14_03.edf": [(1976, 1986)],
    "chb14/chb14_04.edf": [(1362, 1372), (2807, 2817)],
    "chb14/chb14_06.edf": [(1901, 1911)],
    "chb14/chb14_11.edf": [(1828, 1838)],
    "chb14/chb14_17.edf": [(3229, 3239)],
    "chb14/chb14_18.edf": [(1029, 1039)],
    "chb14/chb14_27.edf": [(2823, 2833)],
    "chb15/chb15_06.edf": [(262, 272)],
    "chb15/chb15_10.edf": [(1072, 1082)],
    "chb15/chb15_15.edf": [(1581, 1591)],
    "chb15/chb15_17.edf": [(1915, 1925)],
    "chb15/chb15_20.edf": [(597, 607)],
    "chb15/chb15_22.edf": [(750, 760)],
    "chb15/chb15_28.edf": [(866, 876)],
    "chb15/chb15_31.edf": [(1741, 1751)],
    "chb15/chb15_40.edf": [(824, 834), (2368, 2378), (3352, 3362)],
    "chb15/chb15_46.edf": [(3312, 3322)],
    "chb15/chb15_49.edf": [(1098, 1108)],
    "chb15/chb15_52.edf": [(768, 778)],
    "chb15/chb15_54.edf": [(253, 263), (833, 843), (1514, 1524), (2169, 2179), (3418, 3428)],
    "chb15/chb15_62.edf": [(741, 751)],
    "chb16/chb16_10.edf": [(2280, 2290)],
    "chb16/chb16_11.edf": [(1110, 1120)],
    "chb16/chb16_14.edf": [(1844, 1854)],
    "chb16/chb16_16.edf": [(1204, 1214)],
    "chb16/chb16_17.edf": [(217, 227), (1684, 1694), (2152, 2162), (3280, 3290)],
    "chb16/chb16_18.edf": [(617, 627), (1899, 1909)],
    "chb17/chb17a_03.edf": [(2272, 2282)],
    "chb17/chb17a_04.edf": [(3015, 3025)],
    "chb17/chb17b_63.edf": [(3126, 3136)],
    "chb18/chb18_29.edf": [(3467, 3477)],
    "chb18/chb18_30.edf": [(531, 541)],
    "chb18/chb18_31.edf": [(2077, 2087)],
    "chb18/chb18_32.edf": [(1898, 1908)],
    "chb18/chb18_35.edf": [(2186, 2196)],
    "chb18/chb18_36.edf": [(453, 463)],
    "chb19/chb19_28.edf": [(289, 299)],
    "chb19/chb19_29.edf": [(2954, 2964)],
    "chb19/chb19_30.edf": [(3149, 3159)],
    "chb20/chb20_12.edf": [(84, 94)],
    "chb20/chb20_13.edf": [(1430, 1440), (2488, 2498)],
    "chb20/chb20_14.edf": [(1961, 1971)],
    "chb20/chb20_15.edf": [(380, 390), (1679, 1689)],
    "chb20/chb20_16.edf": [(2216, 2226)],
    "chb20/chb20_68.edf": [(1383, 1393)],
    "chb21/chb21_19.edf": [(1278, 1288)],
    "chb21/chb21_20.edf": [(2617, 2627)],
    "chb21/chb21_21.edf": [(1993, 2003)],
    "chb21/chb21_22.edf": [(2543, 2553)],
    "chb22/chb22_20.edf": [(3357, 3367)],
    "chb22/chb22_25.edf": [(3129, 3139)],
    "chb22/chb22_38.edf": [(1253, 1263)],
    "chb23/chb23_06.edf": [(3952, 3962)],
    "chb23/chb23_08.edf": [(315, 325), (5094, 5104)],
    "chb23/chb23_09.edf": [(2579, 2589), (6875, 6885), (8495, 8505), (9570, 9580)],
    "chb24/chb24_01.edf": [(470, 480), (2441, 2451)],
    "chb24/chb24_03.edf": [(221, 231), (2873, 2883)],
    "chb24/chb24_04.edf": [(1078, 1088), (1401, 1411), (1735, 1745)],
    "chb24/chb24_06.edf": [(1219, 1229)],
    "chb24/chb24_07.edf": [(28, 38)],
    "chb24/chb24_09.edf": [(1735, 1745)],
    "chb24/chb24_11.edf": [(3517, 3527)],
    "chb24/chb24_13.edf": [(3278, 3288)],
    "chb24/chb24_14.edf": [(1929, 1939)],
    "chb24/chb24_15.edf": [(3542, 3552)],
    "chb24/chb24_17.edf": [(3505, 3515)],
    "chb24/chb24_21.edf": [(2794, 2804)],
}
def extract_features(segment):
    """
    特征提取函数：计算能量时间变化率、频谱质心和频谱熵
    """
    energy_rate = np.mean(np.diff(np.square(segment), axis=1), axis=1)
    spectrum = np.abs(np.fft.rfft(segment, axis=1))
    freqs = np.fft.rfftfreq(segment.shape[1])
    spectral_centroid = np.sum(freqs * spectrum, axis=1) / np.sum(spectrum, axis=1)
    spectrum_norm = spectrum / np.sum(spectrum, axis=1, keepdims=True)
    spectral_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-12), axis=1)
    features = np.vstack([energy_rate, spectral_centroid, spectral_entropy]).T
    return features

def process_data(pre_seizure_times, base_folder, target_files, sfreq=256):
    """
    处理数据并提取特征，确保训练数据中包含平衡的非发作前片段
    """
    samples, labels = [], []
    for file_path, seizure_intervals in pre_seizure_times.items():
        if not any(file_path.startswith(folder) for folder in target_files):
            continue
        file_full_path = os.path.join(base_folder, file_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(file_full_path, preload=True, verbose=False)

        raw.filter(0.5, 25, fir_design='firwin', verbose=False)

        available_channels = [ch for ch in target_channels if ch in raw.ch_names]
        if len(available_channels) < len(target_channels):
            continue
        raw.pick(available_channels)

        sfreq = raw.info['sfreq']
        data = raw.get_data()

        # 归一化
        data = (data - np.min(data, axis=1, keepdims=True)) / (
                np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True) + 1e-12)

        # 提取发作前片段
        for start, end in seizure_intervals:
            start_sample = int(start * sfreq)
            end_sample = int(end * sfreq)
            segment = data[:, start_sample:end_sample]
            if segment.shape[1] != (end - start) * sfreq:
                continue
            features = extract_features(segment)
            samples.append(features)
            labels.append("PreSeizure")

        # 提取随机的非发作片段
        num_non_seizure = len(seizure_intervals)  # 根据发作片段的数量确定非发作片段的数量
        total_samples = data.shape[1]
        for _ in range(num_non_seizure):
            random_start = np.random.randint(0, total_samples - int(window_size * sfreq))
            random_end = random_start + int(window_size * sfreq)
            segment = data[:, random_start:random_end]
            features = extract_features(segment)
            samples.append(features)
            labels.append("non-PreSeizure")

    return np.array(samples), np.array(labels)

# 执行24折交叉验证
subject_ids = [f"chb{i:02d}" for i in range(1, 25)]
confusion_matrices = []
accuracies = []

for test_subject in subject_ids:
    print(f"正在处理测试受试者：{test_subject}")

    # 训练集与测试集划分
    train_subjects = [subj for subj in subject_ids if subj != test_subject]
    train_samples, train_labels = process_data(pre_seizure_times, base_folder, train_subjects)
    test_samples, test_labels = process_data(pre_seizure_times, base_folder, [test_subject])

    # 数据展平
    if len(train_samples.shape) == 3:
        train_samples = train_samples.reshape(train_samples.shape[0], -1)
    if len(test_samples.shape) == 3:
        test_samples = test_samples.reshape(test_samples.shape[0], -1)

    # 平衡训练数据
    pre_seizure_samples = train_samples[train_labels == "PreSeizure"]
    non_pre_seizure_samples = train_samples[train_labels == "non-PreSeizure"]
    min_samples = min(len(pre_seizure_samples), len(non_pre_seizure_samples))

    # 采样平衡后的训练集
    balanced_pre_seizure_samples = resample(pre_seizure_samples, n_samples=min_samples, random_state=42)
    balanced_non_pre_seizure_samples = resample(non_pre_seizure_samples, n_samples=min_samples, random_state=42)

    balanced_train_samples = np.vstack([balanced_pre_seizure_samples, balanced_non_pre_seizure_samples])
    balanced_train_labels = np.array(["PreSeizure"] * min_samples + ["non-PreSeizure"] * min_samples)

    # 模型训练
    clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    clf.fit(balanced_train_samples, balanced_train_labels)

    # 模型预测
    y_pred = clf.predict(test_samples)

    # 混淆矩阵与准确率
    cm = confusion_matrix(test_labels, y_pred, labels=["PreSeizure", "non-PreSeizure"])
    acc = accuracy_score(test_labels, y_pred)

    confusion_matrices.append(cm)
    accuracies.append(acc)

    print(f"受试者 {test_subject} 的混淆矩阵：\n{cm}")
    print(f"受试者 {test_subject} 的准确率：{acc:.2f}")

# 汇总结果
confusion_matrices = np.array(confusion_matrices)
mean_cm = np.mean(confusion_matrices, axis=0)

print(f"\n平均混淆矩阵：\n{mean_cm}")
print(f"\n平均准确率：{np.mean(accuracies):.2f}")