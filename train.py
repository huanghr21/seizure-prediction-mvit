"""
训练脚本
主训练流程：数据加载、模型训练、评估和结果保存
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data.preprocessor import prepare_data
from model.mvit import MultiChannelViT, create_model
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    print_metrics,
    save_checkpoint,
    save_results,
    set_seed,
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备

    Returns:
        avg_loss, avg_acc
    """
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()

        # 更新统计信息
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc.item(), data.size(0))

        # 更新进度条
        pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{accuracies.avg:.4f}"})

    return losses.avg, accuracies.avg


def evaluate(model, dataloader, criterion, device):
    """
    评估模型

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        avg_loss, avg_acc, y_true, y_pred
    """
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 计算准确率
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean()

            # 更新统计信息
            losses.update(loss.item(), data.size(0))
            accuracies.update(acc.item(), data.size(0))

            # 收集预测结果
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            # 更新进度条
            pbar.set_postfix(
                {"loss": f"{losses.avg:.4f}", "acc": f"{accuracies.avg:.4f}"}
            )

    return losses.avg, accuracies.avg, np.array(all_targets), np.array(all_preds)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_dir,
):
    """
    完整的训练流程

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（用于早停和选择最佳模型）
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        num_epochs: 训练轮数
        save_dir: 保存目录

    Returns:
        best_metrics: 最佳模型的评估指标
    """
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_acc = 0.0
    best_metrics = None

    # 早停
    if config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE, verbose=True
        )

    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print("-" * 50)

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 评估（在验证集上）
        val_loss, val_acc, y_true, y_pred = evaluate(
            model, val_loader, criterion, device
        )

        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 打印结果
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Learning Rate: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = compute_metrics(y_true, y_pred)

            if config.SAVE_BEST_ONLY:
                checkpoint_path = os.path.join(
                    save_dir, f"best_model_{config.SUBJECT}.pth"
                )
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                print(f"✓ Best model saved! Accuracy: {best_acc:.4f}")

        # 早停检查
        if config.USE_EARLY_STOPPING:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("\nEarly stopping triggered!")
                break

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

    # 绘制训练曲线
    curves_path = os.path.join(save_dir, f"training_curves_{config.SUBJECT}.png")
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    plot_training_curves(history, curves_path)

    return best_metrics, train_losses, val_losses, train_accs, val_accs


def main():
    """主函数"""
    print("=" * 50)
    print("Seizure Prediction with Multi-Channel ViT")
    print("=" * 50)
    print(f"Subject: {config.SUBJECT}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 50)

    # 设置随机种子
    set_seed(config.RANDOM_SEED)

    # 1. 准备数据
    print("\n[1/5] Preparing data...")
    train_dataset, val_dataset, test_dataset = prepare_data(config.SUBJECT_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows上建议设置为0
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 2. 创建模型
    print("\n[2/5] Creating model...")
    model = create_model()
    model = model.to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. 定义损失函数和优化器
    print("\n[3/5] Setting up training...")
    criterion = nn.CrossEntropyLoss()

    if config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY,
        )

    # 学习率调度器
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )

    # 4. 训练模型
    print("\n[4/5] Training model...")
    start_time = time.time()

    best_metrics, train_losses, val_losses, train_accs, val_accs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        save_dir=config.CHECKPOINT_DIR,
    )

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time / 60:.2f} minutes")

    # 5. 最终评估
    print("\n[5/5] Final evaluation...")

    # 加载最佳模型
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, f"best_model_{config.SUBJECT}.pth"
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # 在测试集上评估
    _, _, y_true, y_pred = evaluate(model, test_loader, criterion, config.DEVICE)
    final_metrics = compute_metrics(y_true, y_pred)

    # 打印评估指标
    print_metrics(final_metrics, title=f"Final Test Results - {config.SUBJECT}")

    # 绘制混淆矩阵
    cm_path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_{config.SUBJECT}.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # 6. 保存结果
    results = {
        "subject": config.SUBJECT,
        "model": "MultiChannelViT",
        "config": {
            "n_channels": config.N_CHANNELS,
            "window_size": config.WINDOW_SIZE,
            "filter_range": f"{config.FILTER_LOW}-{config.FILTER_HIGH} Hz",
            "sop": f"{config.SOP / 60} min",
            "sph": f"{config.SPH / 60} min",
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
            "num_layers": config.NUM_LAYERS,
            "embed_dim": config.EMBED_DIM,
            "num_heads": config.NUM_HEADS,
        },
        "data": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "train_preictal": int(train_dataset.labels.sum().item()),
            "train_interictal": len(train_dataset)
            - int(train_dataset.labels.sum().item()),
            "val_preictal": int(val_dataset.labels.sum().item()),
            "val_interictal": len(val_dataset) - int(val_dataset.labels.sum().item()),
            "test_preictal": int(test_dataset.labels.sum().item()),
            "test_interictal": len(test_dataset)
            - int(test_dataset.labels.sum().item()),
        },
        "training": {
            "total_time_minutes": round(training_time / 60, 2),
            "final_train_loss": round(train_losses[-1], 4),
            "final_train_acc": round(train_accs[-1], 4),
            "final_val_loss": round(val_losses[-1], 4),
            "final_val_acc": round(val_accs[-1], 4),
        },
        "metrics": final_metrics,
    }

    results_path = os.path.join(config.RESULTS_DIR, f"results_{config.SUBJECT}.json")
    save_results(results, results_path)

    print("\n" + "=" * 50)
    print("All done! 🎉")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {results_path}")
    print(f"Confusion Matrix: {cm_path}")
    print("=" * 50)


def test_main():
    """
    测试主函数 - 使用生成的假数据快速验证代码逻辑
    不加载真实数据，只测试模型架构和训练流程
    """
    print("\n" + "=" * 50)
    print("TEST MODE - Quick validation with synthetic data")
    print("=" * 50)

    # 1. 生成假数据
    print("\n[1/5] Generating synthetic data...")
    batch_size = 16
    n_samples = 100

    # 生成随机数据：(n_samples, n_channels, 32, 32)
    fake_data = torch.randn(n_samples, config.N_CHANNELS, 32, 32)
    fake_labels = torch.randint(0, 2, (n_samples,))  # 二分类标签

    # 创建简单的TensorDataset
    from torch.utils.data import TensorDataset

    fake_dataset = TensorDataset(fake_data, fake_labels)

    # 分割数据集
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    test_size = n_samples - train_size - val_size

    train_data = TensorDataset(fake_data[:train_size], fake_labels[:train_size])
    val_data = TensorDataset(
        fake_data[train_size : train_size + val_size],
        fake_labels[train_size : train_size + val_size],
    )
    test_data = TensorDataset(
        fake_data[train_size + val_size :], fake_labels[train_size + val_size :]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # 2. 创建模型
    print("\n[2/5] Building model...")
    model = MultiChannelViT(
        n_channels=config.N_CHANNELS,
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        mlp_dim=config.MLP_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 测试前向传播
    print("\n[3/5] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        x, y = next(iter(train_loader))
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Labels shape: {y.shape}")
        print("✓ Forward pass successful")

    # 4. 训练测试
    print("\n[4/5] Training test (5 batches)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 5:  # 只训练5个batch
            break

        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"  Batch {i + 1}/5: Loss = {loss.item():.4f}")

    avg_loss = total_loss / min(5, len(train_loader))
    print(f"Average training loss: {avg_loss:.4f}")
    print("✓ Training loop successful")

    # 5. 测试评估
    print("\n[5/5] Testing evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    print_metrics(metrics)
    print("✓ Evaluation successful")

    # 计算指标
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    print_metrics(metrics)

    print("\n" + "=" * 50)
    print("Test completed successfully! ✓")
    print("You can now run the full training with main()")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    # 如果命令行参数包含 --test，运行测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_main()
    else:
        main()
