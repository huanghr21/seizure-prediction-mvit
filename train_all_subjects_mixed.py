"""
所有被试者混合训练
将所有24个被试者的数据混合在一起，然后随机划分训练/验证/测试集
不考虑跨被试者泛化，目标是获得最佳性能指标
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和模块
from torch.utils.data import DataLoader, Dataset

import config as cfg
from data.multi_subject_preprocessor import prepare_multi_subject_data
from model.mvit import MultiChannelViT
from utils import EarlyStopping, compute_metrics, plot_training_curves, save_checkpoint

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class EEGDataset(Dataset):
    """简单的EEG数据集（延迟转换以节省内存）"""

    def __init__(self, data, labels):
        # 保持numpy格式，不一次性转换为tensor以节省内存
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 只在取数据时转换为tensor
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    save_dir,
):
    """训练模型"""
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 早停和检查点
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_val_acc = 0.0

    # 训练历史
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print("-" * 60)

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics["accuracy"]

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印指标
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(
            f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, "
            f"Val Specificity: {val_metrics['specificity']:.4f}"
        )

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_acc,
                os.path.join(save_dir, "best_model_all_subjects_mixed.pth"),
            )
            print(f"✓ 最佳模型已保存 (Val Acc: {val_acc:.4f})")

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n早停触发！训练在第 {epoch + 1} 轮停止")
            break

    training_time = (time.time() - start_time) / 60
    print(f"\n训练完成！总耗时: {training_time:.2f} 分钟")

    # 绘制训练曲线
    plot_training_curves(
        history, os.path.join(save_dir, "training_curves_all_subjects_mixed.png")
    )

    return history, training_time


def main(test_mode=False):
    """主函数"""
    print("\n" + "=" * 60)
    if test_mode:
        print("测试模式：只使用1个被试者快速验证")
    else:
        print("所有被试者混合训练")
    print("=" * 60)
    print("将所有24个被试者数据混合后随机划分")
    print("=" * 60)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # 创建输出目录
    output_dir = os.path.join(
        cfg.OUTPUT_DIR, "test_mixed" if test_mode else "all_subjects_mixed"
    )
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 准备所有被试者的数据
    print("\n" + "=" * 60)
    if test_mode:
        print("加载测试被试者数据（chb01）...")
        all_subjects = ["chb01"]
        max_epochs = 10
        patience = 5
    else:
        print("加载所有24个被试者的数据...")
        all_subjects = [
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
        max_epochs = cfg.NUM_EPOCHS
        patience = 20

    all_data, all_labels, all_subject_ids = prepare_multi_subject_data(
        subjects=all_subjects,
        data_root=cfg.DATA_ROOT,
        channels=cfg.CHANNEL_NAMES,
        sampling_rate=cfg.SAMPLING_RATE,
        filter_low=cfg.FILTER_LOW,
        filter_high=cfg.FILTER_HIGH,
        window_size=cfg.WINDOW_SIZE,
        sop=cfg.SOP,
        sph=cfg.SPH,
        balance_strategy="global",  # 全局平衡
        normalization=None,  # 混合训练不需要标准化
    )

    print(f"\n总数据量: {len(all_data)} 样本")
    print(f"数据形状: {all_data.shape}")

    # 随机划分数据集 (60% / 20% / 20%)
    print("\n随机划分数据集 (60% / 20% / 20%)...")

    # 先划分出测试集
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 再从训练+验证中划分出验证集
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data,
        train_val_labels,
        test_size=0.25,
        random_state=42,
        stratify=train_val_labels,
    )

    print(f"\n数据划分完成:")
    print(
        f"  训练集: {len(train_data)} 样本 (Pre={np.sum(train_labels == 1)}, Inter={np.sum(train_labels == 0)})"
    )
    print(
        f"  验证集: {len(val_data)} 样本 (Pre={np.sum(val_labels == 1)}, Inter={np.sum(val_labels == 0)})"
    )
    print(
        f"  测试集: {len(test_data)} 样本 (Pre={np.sum(test_labels == 1)}, Inter={np.sum(test_labels == 0)})"
    )

    # 创建数据加载器
    train_dataset = EEGDataset(train_data, train_labels)
    val_dataset = EEGDataset(val_data, val_labels)
    test_dataset = EEGDataset(test_data, test_labels)

    # 优化的DataLoader参数（云端服务器可进一步调整batch_size和num_workers）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print("\n" + "=" * 60)
    print("数据加载器创建完成")
    print("=" * 60)

    # 创建模型
    print("\n创建模型...")
    model = MultiChannelViT(
        n_channels=cfg.N_CHANNELS,
        img_size=32,
        patch_size=cfg.PATCH_SIZE,
        embed_dim=cfg.EMBED_DIM,
        num_layers=cfg.NUM_LAYERS,
        num_heads=cfg.NUM_HEADS,
        mlp_dim=cfg.MLP_DIM,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
    ).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    # 训练模型
    history, training_time = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        max_epochs,
        patience,
        checkpoint_dir,
    )

    # 加载最佳模型
    print("\n加载最佳模型进行测试...")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model_all_subjects_mixed.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估")
    print("=" * 60)

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")

    # 保存结果
    results = {
        "model": "MultiChannelViT",
        "training_mode": "all_subjects_mixed",
        "total_subjects": 24,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "training_time_minutes": training_time,
        "best_val_acc": float(max(history["val_acc"])),
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_metrics["accuracy"]),
            "sensitivity": float(test_metrics["sensitivity"]),
            "specificity": float(test_metrics["specificity"]),
            "f1_score": float(test_metrics["f1_score"]),
        },
        "config": {
            "dropout": cfg.DROPOUT,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "num_layers": cfg.NUM_LAYERS,
            "normalization": "z-score",
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = os.path.join(results_dir, "results_all_subjects_mixed.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {results_path}")

    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)
    print(f"训练耗时: {training_time:.2f} 分钟")
    print(f"最佳验证准确率: {max(history['val_acc']):.4f}")
    print(f"测试准确率: {test_metrics['accuracy']:.4f}")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="所有被试者混合训练")
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：只使用chb01进行快速验证",
    )
    args = parser.parse_args()

    main(test_mode=args.test)
