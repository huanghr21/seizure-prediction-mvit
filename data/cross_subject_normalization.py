"""
跨被试者标准化模块
减少个体差异，提取共性特征
"""

import numpy as np


def z_score_normalize(data, axis=(1, 2)):
    """
    Z-score标准化
    
    Args:
        data: (N, C, H, W) 时频特征
        axis: 标准化的维度（默认对每个通道的时频图标准化）
    
    Returns:
        normalized_data: 标准化后的数据
    """
    # 确保使用float32以节省内存
    data = data.astype(np.float32, copy=False)
    mean = data.mean(axis=axis, keepdims=True).astype(np.float32)
    std = data.std(axis=axis, keepdims=True).astype(np.float32)
    std = np.where(std == 0, 1, std)  # 避免除零
    return (data - mean) / std


def min_max_normalize(data, axis=(1, 2), feature_range=(0, 1)):
    """
    Min-Max归一化到指定范围
    
    Args:
        data: (N, C, H, W) 时频特征
        axis: 归一化的维度
        feature_range: 目标范围
    
    Returns:
        normalized_data: 归一化后的数据
    """
    min_val = data.min(axis=axis, keepdims=True)
    max_val = data.max(axis=axis, keepdims=True)
    
    # 避免除零
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    
    # 归一化到[0,1]
    normalized = (data - min_val) / range_val
    
    # 缩放到目标范围
    min_target, max_target = feature_range
    return normalized * (max_target - min_target) + min_target


def robust_normalize(data, axis=(1, 2)):
    """
    基于中位数和四分位距的鲁棒标准化
    对异常值更稳健
    
    Args:
        data: (N, C, H, W) 时频特征
        axis: 标准化的维度
    
    Returns:
        normalized_data: 标准化后的数据
    """
    median = np.median(data, axis=axis, keepdims=True)
    q75 = np.percentile(data, 75, axis=axis, keepdims=True)
    q25 = np.percentile(data, 25, axis=axis, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 1, iqr)  # 避免除零
    
    return (data - median) / iqr


def apply_cross_subject_normalization(data, method='z-score'):
    """
    应用跨被试者标准化
    
    Args:
        data: (N, C, H, W) 时频特征
        method: 'z-score', 'min-max', 或 'robust'
    
    Returns:
        normalized_data: 标准化后的数据
    """
    if method == 'z-score':
        return z_score_normalize(data)
    elif method == 'min-max':
        return min_max_normalize(data)
    elif method == 'robust':
        return robust_normalize(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


if __name__ == "__main__":
    print("测试跨被试者标准化...")
    
    # 创建模拟数据 (100样本, 23通道, 32x32时频图)
    # 模拟不同被试者的幅度差异
    data1 = np.random.randn(50, 23, 32, 32) * 10 + 100  # 被试者1: 高基线
    data2 = np.random.randn(50, 23, 32, 32) * 2 + 20    # 被试者2: 低基线
    data = np.concatenate([data1, data2], axis=0)
    
    print(f"\n原始数据:")
    print(f"  整体范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  被试者1范围: [{data1.min():.2f}, {data1.max():.2f}]")
    print(f"  被试者2范围: [{data2.min():.2f}, {data2.max():.2f}]")
    
    # 测试各种标准化方法
    for method in ['z-score', 'min-max', 'robust']:
        normalized = apply_cross_subject_normalization(data, method=method)
        print(f"\n{method}标准化后:")
        print(f"  范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
        print(f"  均值: {normalized.mean():.4f}, 标准差: {normalized.std():.4f}")
    
    print("\n跨被试者标准化测试完成!")
