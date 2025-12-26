"""
跨被试者评估工具
提供每个被试者的性能统计和可视化功能
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_per_subject_metrics(all_labels, all_preds, all_subject_ids):
    """
    计算每个被试者的评估指标

    Args:
        all_labels: (N,) 真实标签
        all_preds: (N,) 预测标签
        all_subject_ids: (N,) 被试者ID列表

    Returns:
        per_subject_results: dict {subject_id: metrics}
    """
    unique_subjects = sorted(set(all_subject_ids))
    per_subject_results = {}

    for subject in unique_subjects:
        # 获取该被试者的数据
        mask = np.array([sid == subject for sid in all_subject_ids])
        subj_labels = all_labels[mask]
        subj_preds = all_preds[mask]

        if len(subj_labels) == 0:
            continue

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(
            subj_labels, subj_preds, labels=[0, 1]
        ).ravel()

        # 计算指标
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )

        per_subject_results[subject] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "f1_score": float(f1),
            "confusion_matrix": {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            },
            "n_samples": int(len(subj_labels)),
        }

    return per_subject_results


def save_per_subject_results(per_subject_results, overall_metrics, save_path):
    """
    保存每个被试者的结果到JSON文件

    Args:
        per_subject_results: dict {subject_id: metrics}
        overall_metrics: dict, 整体指标
        save_path: str, 保存路径
    """
    results = {
        "overall_metrics": overall_metrics,
        "per_subject_results": per_subject_results,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"每个被试者的结果已保存到: {save_path}")


def plot_per_subject_metrics(per_subject_results, save_dir, metric_name="accuracy"):
    """
    绘制每个被试者的性能柱状图

    Args:
        per_subject_results: dict {subject_id: metrics}
        save_dir: str, 保存目录
        metric_name: str, 要绘制的指标名称
    """
    os.makedirs(save_dir, exist_ok=True)

    subjects = sorted(per_subject_results.keys())
    values = [per_subject_results[s][metric_name] for s in subjects]

    plt.figure(figsize=(15, 6))
    bars = plt.bar(
        range(len(subjects)), values, color="skyblue", edgecolor="navy", alpha=0.7
    )

    # 添加平均线
    mean_value = np.mean(values)
    plt.axhline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.4f}",
    )

    # 设置标签
    plt.xlabel("Subject", fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    plt.title(f"Per-Subject {metric_name.capitalize()}", fontsize=14, fontweight="bold")
    plt.xticks(range(len(subjects)), subjects, rotation=45, ha="right")
    plt.ylim([0, 1.0])
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    # 在柱子上添加数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"per_subject_{metric_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"每被试者{metric_name}图已保存到: {save_path}")


def plot_all_metrics_comparison(per_subject_results, save_dir):
    """
    绘制所有指标的对比图

    Args:
        per_subject_results: dict {subject_id: metrics}
        save_dir: str, 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    subjects = sorted(per_subject_results.keys())
    metrics = ["sensitivity", "specificity", "accuracy", "precision", "f1_score"]

    # 准备数据
    data = {
        metric: [per_subject_results[s][metric] for s in subjects] for metric in metrics
    }

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = data[metric]

        bars = ax.bar(
            range(len(subjects)),
            values,
            color="lightcoral",
            edgecolor="darkred",
            alpha=0.7,
        )
        mean_value = np.mean(values)
        ax.axhline(
            mean_value,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_value:.4f}",
        )

        ax.set_xlabel("Subject", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_title(f"{metric.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    # 隐藏多余的子图
    axes[-1].axis("off")

    plt.suptitle("Per-Subject Metrics Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "all_metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"所有指标对比图已保存到: {save_path}")


def plot_per_subject_confusion_matrices(per_subject_results, save_dir):
    """
    为每个被试者绘制混淆矩阵

    Args:
        per_subject_results: dict {subject_id: metrics}
        save_dir: str, 保存目录
    """
    cm_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for subject, results in sorted(per_subject_results.items()):
        cm_dict = results["confusion_matrix"]
        cm = np.array([[cm_dict["TN"], cm_dict["FP"]], [cm_dict["FN"], cm_dict["TP"]]])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Inter-ictal", "Pre-ictal"],
            yticklabels=["Inter-ictal", "Pre-ictal"],
            cbar_kws={"label": "Count"},
        )

        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.title(
            f"Confusion Matrix - {subject}\n"
            f"Accuracy: {results['accuracy']:.4f}, "
            f"F1: {results['f1_score']:.4f}",
            fontsize=14,
            fontweight="bold",
        )

        save_path = os.path.join(cm_dir, f"{subject}_confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"每被试者混淆矩阵已保存到: {cm_dir}")


def print_summary_statistics(per_subject_results):
    """
    打印汇总统计信息

    Args:
        per_subject_results: dict {subject_id: metrics}
    """
    metrics = ["sensitivity", "specificity", "accuracy", "precision", "f1_score"]

    print("\n" + "=" * 60)
    print("跨被试者性能统计汇总")
    print("=" * 60)

    for metric in metrics:
        values = [results[metric] for results in per_subject_results.values()]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"\n{metric.capitalize()}:")
        print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Range: {max_val - min_val:.4f}")

    print("\n" + "=" * 60)


def save_loso_summary(all_fold_results, save_path):
    """
    保存LOSO所有折的汇总结果

    Args:
        all_fold_results: list of dict, 每折的结果
        save_path: str, 保存路径
    """
    # 计算平均指标
    metrics = ["sensitivity", "specificity", "accuracy", "precision", "f1_score"]
    summary = {
        "n_folds": len(all_fold_results),
        "per_fold_results": all_fold_results,
        "average_metrics": {},
    }

    for metric in metrics:
        values = [fold["test_metrics"][metric] for fold in all_fold_results]
        summary["average_metrics"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLOSO汇总结果已保存到: {save_path}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("LOSO交叉验证汇总统计")
    print("=" * 60)
    for metric in metrics:
        stats = summary["average_metrics"][metric]
        print(f"\n{metric.capitalize()}:")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print("=" * 60)
