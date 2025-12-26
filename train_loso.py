"""
方案2: LOSO (Leave-One-Subject-Out) 交叉验证
每折使用1个被试者测试，2个被试者验证，其余21个被试者训练
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置和模块
import config_multi_subject as cfg
from data.multi_subject_preprocessor import create_subject_split_dataloaders
from model.mvit import MultiChannelViT
from utils import EarlyStopping, compute_metrics, save_checkpoint
from utils.cross_subject_utils import save_loso_summary

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, labels, subject_ids in train_loader:
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
        for data, labels, subject_ids in data_loader:
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

    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def train_one_fold(
    fold_idx, test_subject, val_subjects, train_subjects, device, cfg_params
):
    """训练一折"""
    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx + 1}/{len(cfg.ALL_SUBJECTS)}")
    print("=" * 80)
    print(f"测试被试者: {test_subject}")
    print(f"验证被试者: {val_subjects}")
    print(f"训练被试者数量: {len(train_subjects)}")
    print("=" * 80)

    # 创建数据加载器
    print("\n准备数据...")
    train_loader, val_loader, test_loader = create_subject_split_dataloaders(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=[test_subject],
        data_root=cfg.DATA_ROOT,
        channels=cfg.CHANNELS,
        sampling_rate=cfg.SAMPLING_RATE,
        filter_low=cfg.FILTER_LOW,
        filter_high=cfg.FILTER_HIGH,
        window_size=cfg.WINDOW_SIZE,
        sop=cfg.SOP,
        sph=cfg.SPH,
        batch_size=cfg.BATCH_SIZE,
        balance_strategy=cfg.BALANCE_STRATEGY,
    )

    # 创建模型
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

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.SCHEDULER_PATIENCE_MULTI
    )

    # 早停
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE_LOSO, verbose=False
    )
    best_val_acc = 0.0
    best_epoch = 0

    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(cfg.NUM_EPOCHS_LOSO):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics["accuracy"]

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型（在内存中）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"  早停触发于第 {epoch + 1} 轮")
            break

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{cfg.NUM_EPOCHS_LOSO} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    training_time = time.time() - start_time

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 在测试集上评估
    print(f"\n在测试被试者 {test_subject} 上评估...")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n测试结果 ({test_subject}):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"  训练耗时: {training_time / 60:.2f} 分钟")

    # 返回结果
    fold_result = {
        "fold": fold_idx + 1,
        "test_subject": test_subject,
        "val_subjects": val_subjects,
        "train_subjects_count": len(train_subjects),
        "best_epoch": best_epoch + 1,
        "epochs_trained": epoch + 1,
        "training_time_minutes": training_time / 60,
        "test_metrics": test_metrics,
    }

    return fold_result


def main():
    # 打印配置
    print("=" * 80)
    print("LOSO (Leave-One-Subject-Out) 交叉验证")
    print("=" * 80)
    print(f"总被试者数: {len(cfg.ALL_SUBJECTS)}")
    print(f"交叉验证折数: {len(cfg.ALL_SUBJECTS)}")
    print(f"每折配置: 1个测试 + 2个验证 + {len(cfg.ALL_SUBJECTS) - 3}个训练")
    print("=" * 80)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR_LOSO, exist_ok=True)

    # 存储所有折的结果
    all_fold_results = []

    # 开始交叉验证
    total_start_time = time.time()

    for fold_idx, test_subject in enumerate(cfg.ALL_SUBJECTS):
        # 确定验证集和训练集
        remaining_subjects = [s for s in cfg.ALL_SUBJECTS if s != test_subject]

        # 选择2个被试者作为验证集（循环选择以保证多样性）
        val_idx_1 = fold_idx % len(remaining_subjects)
        val_idx_2 = (fold_idx + 1) % len(remaining_subjects)
        if val_idx_2 == val_idx_1:
            val_idx_2 = (val_idx_2 + 1) % len(remaining_subjects)

        val_subjects = [remaining_subjects[val_idx_1], remaining_subjects[val_idx_2]]
        train_subjects = [s for s in remaining_subjects if s not in val_subjects]

        # 训练这一折
        try:
            fold_result = train_one_fold(
                fold_idx, test_subject, val_subjects, train_subjects, device, cfg
            )
            all_fold_results.append(fold_result)

            # 保存中间结果
            temp_path = os.path.join(
                cfg.OUTPUT_DIR_LOSO, f"fold_{fold_idx + 1:02d}_result.json"
            )
            with open(temp_path, "w") as f:
                json.dump(fold_result, f, indent=2)

        except Exception as e:
            print(f"\n警告: Fold {fold_idx + 1} 失败 - {str(e)}")
            continue

    total_time = time.time() - total_start_time

    # 保存汇总结果
    print("\n" + "=" * 80)
    print("LOSO交叉验证完成！")
    print("=" * 80)
    print(f"成功完成的折数: {len(all_fold_results)}/{len(cfg.ALL_SUBJECTS)}")
    print(f"总耗时: {total_time / 3600:.2f} 小时")
    print("=" * 80)

    if len(all_fold_results) > 0:
        summary_path = os.path.join(cfg.OUTPUT_DIR_LOSO, "loso_summary.json")
        save_loso_summary(all_fold_results, summary_path)

        print(f"\n结果已保存到: {cfg.OUTPUT_DIR_LOSO}")
    else:
        print("\n警告: 没有成功完成的折数！")


if __name__ == "__main__":
    # 询问用户确认
    print("=" * 80)
    print("警告: LOSO交叉验证需要训练24个模型，可能耗时12-24小时！")
    print("=" * 80)
    response = input("\n确定要开始吗？(yes/no): ")

    if response.lower() in ["yes", "y"]:
        main()
    else:
        print("已取消")
