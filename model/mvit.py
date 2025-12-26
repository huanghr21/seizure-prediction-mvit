"""
Multi-channel Vision Transformer (MViT) 模型
所有通道共享同一个ViT（参数共享），最后拼接输出进行分类
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PatchEmbedding(nn.Module):
    """将图像分割成patches并进行嵌入"""

    def __init__(self, img_size=32, patch_size=8, in_channels=1, embed_dim=256):
        """
        Args:
            img_size: 输入图像大小（32x32）
            patch_size: patch大小（8x8，将32x32分成16个patches）
            in_channels: 输入通道数（1，单通道灰度图）
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 16个patches

        # 使用卷积进行patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, 32)

        Returns:
            (batch, n_patches+1, embed_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.proj(x)  # (batch, embed_dim, 4, 4)
        x = x.flatten(2)  # (batch, embed_dim, 16)
        x = x.transpose(1, 2)  # (batch, 16, embed_dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 17, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed

        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, n_patches+1, embed_dim)

        Returns:
            (batch, n_patches+1, embed_dim)
        """
        batch_size, n_tokens, embed_dim = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(
            batch_size, n_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算attention
        attn = (q @ k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (batch, num_heads, n_tokens, n_tokens)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, embed_dim)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout=0.1
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, embed_dim, dropout)

    def forward(self, x):
        # 多头注意力 + 残差连接
        x = x + self.attn(self.norm1(x))

        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器（堆叠多层）"""

    def __init__(
        self, embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class SingleChannelViT(nn.Module):
    """单通道的Vision Transformer"""

    def __init__(
        self,
        img_size=32,
        patch_size=8,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, mlp_dim, num_layers, dropout
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, 32) 单通道输入

        Returns:
            (batch, embed_dim) CLS token的输出
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches+1, embed_dim)

        # Transformer encoding
        x = self.encoder(x)  # (batch, n_patches+1, embed_dim)

        # 取CLS token作为输出
        x = x[:, 0]  # (batch, embed_dim)

        return x


class MultiChannelViT(nn.Module):
    """
    Multi-channel Vision Transformer
    所有通道共享同一个ViT（参数共享），最后拼接输出进行分类
    """

    def __init__(
        self,
        n_channels=21,
        img_size=32,
        patch_size=8,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=6,
        num_classes=2,
        dropout=0.1,
    ):
        """
        Args:
            n_channels: 通道数（21）
            img_size: 输入图像大小（32）
            patch_size: patch大小（8）
            embed_dim: 嵌入维度（256）
            num_heads: 注意力头数（8）
            mlp_dim: MLP隐藏层维度（512）
            num_layers: Transformer层数（6）
            num_classes: 分类类别数（2）
            dropout: Dropout率
        """
        super().__init__()

        self.n_channels = n_channels
        self.embed_dim = embed_dim

        # 所有通道共享同一个ViT（参数共享）
        self.shared_vit = SingleChannelViT(
            img_size,
            patch_size,
            embed_dim,
            num_heads,
            mlp_dim,
            num_layers,
            dropout,
        )

        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, 32, 32)

        Returns:
            (batch, num_classes)
        """
        # 每个通道通过共享的ViT
        channel_outputs = []
        for ch in range(self.n_channels):
            ch_input = x[:, ch : ch + 1, :, :]  # (batch, 1, 32, 32)
            ch_output = self.shared_vit(ch_input)  # (batch, embed_dim)
            channel_outputs.append(ch_output)

        # 拼接所有通道的输出
        x = torch.cat(channel_outputs, dim=1)  # (batch, n_channels * embed_dim)

        # 分类
        x = self.classifier(x)  # (batch, num_classes)

        return x


def create_model():
    """创建MViT模型"""
    model = MultiChannelViT(
        n_channels=config.N_CHANNELS,
        img_size=config.FEATURE_SIZE,
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        mlp_dim=config.MLP_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_model()
    print(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 21, 32, 32)  # (batch=2, channels=21, H=32, W=32)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
