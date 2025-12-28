"""
EEG数据增强模块
用于提升模型泛化能力,特别是跨被试者场景
"""

import numpy as np
import torch


class EEGAugmentation:
    """EEG时频特征数据增强"""
    
    def __init__(self, 
                 noise_std=0.01,          # 高斯噪声标准差
                 time_shift_max=3,        # 时间平移最大像素
                 freq_shift_max=2,        # 频率平移最大像素
                 magnitude_scale_range=(0.9, 1.1),  # 幅度缩放范围
                 prob=0.5):               # 每种增强的应用概率
        """
        Args:
            noise_std: 添加高斯噪声的标准差
            time_shift_max: 时间维度随机平移的最大像素数
            freq_shift_max: 频率维度随机平移的最大像素数
            magnitude_scale_range: 幅度随机缩放的范围(min, max)
            prob: 每种增强技术的应用概率
        """
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.freq_shift_max = freq_shift_max
        self.magnitude_scale_range = magnitude_scale_range
        self.prob = prob
    
    def __call__(self, x):
        """
        对时频特征进行增强
        
        Args:
            x: (C, H, W) 时频特征，C=通道数, H=频率bins, W=时间bins
            
        Returns:
            augmented_x: 增强后的特征
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
            was_tensor = True
        else:
            was_tensor = False
        
        # 1. 添加高斯噪声
        if np.random.rand() < self.prob:
            noise = np.random.normal(0, self.noise_std, x.shape)
            x = x + noise
        
        # 2. 幅度缩放
        if np.random.rand() < self.prob:
            scale = np.random.uniform(*self.magnitude_scale_range)
            x = x * scale
        
        # 3. 时间平移 (循环移位)
        if np.random.rand() < self.prob and self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            x = np.roll(x, shift, axis=2)  # axis=2 是时间维度
        
        # 4. 频率平移 (循环移位)
        if np.random.rand() < self.prob and self.freq_shift_max > 0:
            shift = np.random.randint(-self.freq_shift_max, self.freq_shift_max + 1)
            x = np.roll(x, shift, axis=1)  # axis=1 是频率维度
        
        if was_tensor:
            x = torch.from_numpy(x).float()
        
        return x


class MixUp:
    """
    MixUp数据增强
    将两个样本线性混合,提升模型对插值样本的鲁棒性
    """
    
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Beta分布的参数,控制混合比例
        """
        self.alpha = alpha
    
    def __call__(self, x1, x2, y1, y2):
        """
        混合两个样本
        
        Args:
            x1, x2: (C, H, W) 两个样本
            y1, y2: 标签
            
        Returns:
            mixed_x: 混合后的样本
            mixed_y: 混合后的标签(软标签)
        """
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        
        # 软标签
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y


def get_augmentation(mode='train'):
    """
    获取数据增强策略
    
    Args:
        mode: 'train' 或 'eval'
        
    Returns:
        augmentation: 数据增强函数或None
    """
    # 暂时关闭数据增强，先测试基础配置能否稳定学习
    return None
    
    # if mode == 'train':
    #     return EEGAugmentation(
    #         noise_std=0.01,
    #         time_shift_max=3,
    #         freq_shift_max=2,
    #         magnitude_scale_range=(0.9, 1.1),
    #         prob=0.5
    #     )
    # else:
    #     return None


if __name__ == "__main__":
    # 测试数据增强
    print("测试EEG数据增强...")
    
    # 创建模拟数据 (23通道, 32x32时频图)
    x = np.random.randn(23, 32, 32).astype(np.float32)
    
    # 创建增强器
    aug = EEGAugmentation()
    
    # 应用增强
    x_aug = aug(x)
    
    print(f"原始数据形状: {x.shape}")
    print(f"增强数据形状: {x_aug.shape}")
    print(f"数据变化: mean diff = {np.abs(x - x_aug).mean():.6f}")
    
    # 测试MixUp
    print("\n测试MixUp...")
    mixup = MixUp(alpha=0.2)
    x1, x2 = np.random.randn(23, 32, 32), np.random.randn(23, 32, 32)
    y1, y2 = np.array([1, 0]), np.array([0, 1])
    
    mixed_x, mixed_y = mixup(x1, x2, y1, y2)
    print(f"混合样本形状: {mixed_x.shape}")
    print(f"混合标签: {mixed_y}")
    
    print("\n数据增强模块测试完成!")
