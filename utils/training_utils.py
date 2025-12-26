"""
工具函数模块
包含：评估指标计算、随机种子设置、结果保存等
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed=42):
    """
    设置随机种子以保证实验可复现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    """
    计算评估指标

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    Returns:
        metrics: 包含各项指标的字典
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 对于二分类：
    # cm = [[TN, FP],
    #       [FN, TP]]
    # 其中：0-inter-ictal, 1-pre-ictal

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # 计算各项指标
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # 灵敏度/召回率
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # 特异度
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 精确率

    # F1 score
    if precision + sensitivity > 0:
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1_score = 0

    metrics = {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "TP": int(TP),
        },
    }

    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    打印评估指标

    Args:
        metrics: compute_metrics返回的指标字典
        title: 标题
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print("=" * 50)

    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix:")
    print(f"  TN: {cm['TN']:4d}  |  FP: {cm['FP']:4d}")
    print(f"  FN: {cm['FN']:4d}  |  TP: {cm['TP']:4d}")
    print(f"{'-' * 50}")

    print(
        f"Sensitivity (Recall): {metrics['sensitivity']:.4f} ({metrics['sensitivity'] * 100:.2f}%)"
    )
    print(
        f"Specificity:          {metrics['specificity']:.4f} ({metrics['specificity'] * 100:.2f}%)"
    )
    print(
        f"Accuracy:             {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)"
    )
    print(
        f"Precision:            {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)"
    )
    print(f"F1 Score:             {metrics['f1_score']:.4f}")
    print("=" * 50)


def save_results(results, filepath):
    """
    保存结果到JSON文件

    Args:
        results: 结果字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {filepath}")


def load_results(filepath):
    """
    从JSON文件加载结果

    Args:
        filepath: 文件路径

    Returns:
        results: 结果字典
    """
    with open(filepath, "r") as f:
        results = json.load(f)

    return results


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径（可选）
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Inter-ictal", "Pre-ictal"],
        yticklabels=["Inter-ictal", "Pre-ictal"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.close()


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线

    Args:
        history: 包含训练历史的字典，需包含以下键：
                 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: 保存路径（可选）
    """
    # 兼容旧版本的函数调用
    if isinstance(history, dict):
        train_losses = history.get("train_loss", [])
        train_accs = history.get("train_acc", [])
        val_losses = history.get("val_loss", [])
        val_accs = history.get("val_acc", [])
    else:
        # 如果是旧版本的列表参数
        train_losses = history
        train_accs = save_path if isinstance(save_path, list) else []
        val_losses = []
        val_accs = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    if train_losses:
        axes[0].plot(train_losses, label="Train Loss")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    if train_accs:
        axes[1].plot(train_accs, label="Train Accuracy")
    if val_accs:
        axes[1].plot(val_accs, label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # save_path 作为最后一个参数
    if isinstance(history, dict) and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    plt.close()


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, verbose=True, delta=0):
        """
        Args:
            patience: 容忍多少个epoch没有改进
            verbose: 是否打印信息
            delta: 最小改进量
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    加载模型检查点

    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器（可选）

    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


if __name__ == "__main__":
    # 测试工具函数
    import sys

    sys.path.append("..")
    import config

    print("Testing utility functions...")

    # 测试设置随机种子
    set_seed(42)
    print("Random seed set to 42")

    # 测试评估指标计算
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    # 测试保存和加载结果（使用完整路径）
    test_results = {"subject": "chb01", "metrics": metrics}
    test_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    save_results(test_results, test_path)
    loaded_results = load_results(test_path)
    print("\nLoaded results:", loaded_results)

    # 清理测试文件
    if os.path.exists(test_path):
        os.remove(test_path)

    print("\nAll tests passed!")
