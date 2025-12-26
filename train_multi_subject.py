"""
方案1: 被试者级别划分训练
训练集(18个被试者) / 验证集(3个被试者) / 测试集(3个被试者)
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
from utils import EarlyStopping, compute_metrics, plot_training_curves, save_checkpoint
from utils.cross_subject_utils import (
    compute_per_subject_metrics,
    plot_all_metrics_comparison,
    plot_per_subject_confusion_matrices,
    plot_per_subject_metrics,
    print_summary_statistics,
    save_per_subject_results,
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, labels, subject_ids) in enumerate(train_loader):
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

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, return_predictions=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_subject_ids = []

    with torch.no_grad():
        for data, labels, subject_ids in data_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_subject_ids.extend(subject_ids)

    avg_loss = total_loss / len(data_loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算整体指标
    metrics = compute_metrics(all_labels, all_preds)

    if return_predictions:
        return avg_loss, metrics, all_labels, all_preds, all_subject_ids
    else:
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
            checkpoint_path = os.path.join(save_dir, "best_model_multi_subject.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"✓ 最佳模型已保存 (Val Acc: {val_acc:.4f})")

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n早停触发！训练在第 {epoch + 1} 轮停止")
            break

    training_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {training_time / 60:.2f} 分钟")

    # 绘制训练曲线
    curves_path = os.path.join(save_dir, "training_curves_multi_subject.png")
    plot_training_curves(history, curves_path)

    return history, training_time


def main():
    # 打印配置
    cfg.print_config()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR_MULTI, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR_MULTI, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR_MULTI, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR_MULTI, exist_ok=True)

    # 创建数据加载器
    print("\n" + "=" * 60)
    print("准备数据...")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_subject_split_dataloaders(
        train_subjects=cfg.TRAIN_SUBJECTS,
        val_subjects=cfg.VAL_SUBJECTS,
        test_subjects=cfg.TEST_SUBJECTS,
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.SCHEDULER_PATIENCE_MULTI
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
        cfg.NUM_EPOCHS_MULTI,
        cfg.EARLY_STOPPING_PATIENCE_MULTI,
        cfg.CHECKPOINT_DIR_MULTI,
    )

    # 加载最佳模型
    print("\n加载最佳模型进行测试...")
    checkpoint_path = os.path.join(
        cfg.CHECKPOINT_DIR_MULTI, "best_model_multi_subject.pth"
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估")
    print("=" * 60)

    test_loss, test_metrics, test_labels, test_preds, test_subject_ids = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )

    print(f"\n整体测试指标:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")

    # 计算每个被试者的指标
    print("\n计算每个测试被试者的性能...")
    per_subject_results = compute_per_subject_metrics(
        test_labels, test_preds, test_subject_ids
    )

    print("\n每个测试被试者的结果:")
    for subject, metrics in sorted(per_subject_results.items()):
        print(f"\n{subject}:")
        print(f"  样本数: {metrics['n_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    # 打印统计汇总
    print_summary_statistics(per_subject_results)

    # 保存结果
    print("\n保存结果...")

    # 保存JSON结果
    results_json_path = os.path.join(
        cfg.RESULTS_DIR_MULTI, "results_multi_subject.json"
    )
    save_per_subject_results(per_subject_results, test_metrics, results_json_path)

    # 保存详细结果
    detailed_results = {
        "experiment_info": {
            "train_subjects": cfg.TRAIN_SUBJECTS,
            "val_subjects": cfg.VAL_SUBJECTS,
            "test_subjects": cfg.TEST_SUBJECTS,
            "balance_strategy": cfg.BALANCE_STRATEGY,
            "training_time_minutes": training_time / 60,
            "epochs_trained": len(history["train_loss"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "overall_test_metrics": test_metrics,
        "per_subject_results": per_subject_results,
    }

    detailed_path = os.path.join(cfg.RESULTS_DIR_MULTI, "detailed_results.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)

    # 绘制可视化图表
    print("\n生成可视化图表...")

    # 每个指标的柱状图
    for metric in ["accuracy", "sensitivity", "specificity", "f1_score"]:
        plot_per_subject_metrics(per_subject_results, cfg.RESULTS_DIR_MULTI, metric)

    # 所有指标对比图
    plot_all_metrics_comparison(per_subject_results, cfg.RESULTS_DIR_MULTI)

    # 每个被试者的混淆矩阵
    plot_per_subject_confusion_matrices(per_subject_results, cfg.RESULTS_DIR_MULTI)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"结果保存在: {cfg.RESULTS_DIR_MULTI}")
    print(f"模型保存在: {cfg.CHECKPOINT_DIR_MULTI}")
    print("=" * 60)


if __name__ == "__main__":
    main()
