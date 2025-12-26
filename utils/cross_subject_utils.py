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


if __name__ == "__main__":
    """测试跨被试者评估工具"""
    import tempfile

    print("\n" + "=" * 60)
    print("跨被试者评估工具 - 测试模式")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)

    # 模拟3个被试者的数据
    subjects = ["chb01", "chb02", "chb03"]
    n_samples_per_subject = 100

    all_labels = []
    all_preds = []
    all_subject_ids = []

    for subject in subjects:
        # 为每个被试者生成随机标签和预测
        labels = np.random.randint(0, 2, n_samples_per_subject)
        # 预测有80%的准确率
        preds = labels.copy()
        flip_indices = np.random.choice(
            n_samples_per_subject, size=int(n_samples_per_subject * 0.2), replace=False
        )
        preds[flip_indices] = 1 - preds[flip_indices]

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_subject_ids.extend([subject] * n_samples_per_subject)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print(f"\n生成测试数据:")
    print(f"  - 被试者数: {len(subjects)}")
    print(f"  - 总样本数: {len(all_labels)}")
    print(f"  - 每被试者样本数: {n_samples_per_subject}")

    # 测试1: 计算每个被试者的指标
    print("\n【测试1】计算每个被试者的指标...")
    per_subject_results = compute_per_subject_metrics(
        all_labels, all_preds, all_subject_ids
    )

    for subject, metrics in per_subject_results.items():
        print(f"\n  {subject}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"    F1 Score: {metrics['f1_score']:.4f}")

    # 测试2: 保存结果
    print("\n【测试2】保存结果到临时文件...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 模拟整体指标
        overall_metrics = {
            "accuracy": 0.80,
            "sensitivity": 0.78,
            "specificity": 0.82,
            "f1_score": 0.79,
        }

        results_path = os.path.join(tmpdir, "results", "test_results.json")
        save_per_subject_results(per_subject_results, overall_metrics, results_path)

        # 验证文件是否创建
        if os.path.exists(results_path):
            print(f"  ✓ 结果文件成功创建")
            with open(results_path, "r") as f:
                loaded = json.load(f)
                print(
                    f"  ✓ 文件包含 {len(loaded['per_subject_results'])} 个被试者的结果"
                )

        # 测试3: 绘制图表
        print("\n【测试3】绘制可视化图表...")

        # 绘制单个指标
        plot_per_subject_metrics(per_subject_results, tmpdir, "accuracy")
        accuracy_plot = os.path.join(tmpdir, "per_subject_accuracy.png")
        if os.path.exists(accuracy_plot):
            print(f"  ✓ Accuracy柱状图已生成")

        # 绘制所有指标对比
        plot_all_metrics_comparison(per_subject_results, tmpdir)
        comparison_plot = os.path.join(tmpdir, "all_metrics_comparison.png")
        if os.path.exists(comparison_plot):
            print(f"  ✓ 所有指标对比图已生成")

        # 绘制混淆矩阵
        plot_per_subject_confusion_matrices(per_subject_results, tmpdir)
        cm_dir = os.path.join(tmpdir, "confusion_matrices")
        if os.path.exists(cm_dir):
            cm_files = [f for f in os.listdir(cm_dir) if f.endswith(".png")]
            print(f"  ✓ {len(cm_files)} 个混淆矩阵图已生成")

    # 测试4: 打印统计摘要
    print("\n【测试4】打印统计摘要...")
    print_summary_statistics(per_subject_results)

    # 测试5: LOSO汇总
    print("\n【测试5】测试LOSO汇总功能...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 模拟LOSO的多折结果
        all_fold_results = []
        for i, subject in enumerate(subjects):
            fold_result = {
                "fold": i + 1,
                "test_subject": subject,
                "test_metrics": {
                    "accuracy": 0.75 + np.random.rand() * 0.15,
                    "sensitivity": 0.70 + np.random.rand() * 0.20,
                    "specificity": 0.75 + np.random.rand() * 0.15,
                    "precision": 0.70 + np.random.rand() * 0.20,
                    "f1_score": 0.72 + np.random.rand() * 0.18,
                },
            }
            all_fold_results.append(fold_result)

        summary_path = os.path.join(tmpdir, "loso_summary.json")
        save_loso_summary(all_fold_results, summary_path)

        if os.path.exists(summary_path):
            print(f"  ✓ LOSO汇总文件成功创建")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
