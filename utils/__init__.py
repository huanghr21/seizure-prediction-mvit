"""
工具函数包
用于单被试者和多被试者实验
"""

# 从 training_utils 导入常用函数
# 从 cross_subject_utils 导入跨被试者评估函数
from .cross_subject_utils import (
    compute_per_subject_metrics,
    plot_all_metrics_comparison,
    plot_per_subject_confusion_matrices,
    plot_per_subject_metrics,
    print_summary_statistics,
    save_loso_summary,
    save_per_subject_results,
)
from .training_utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    load_checkpoint,
    load_results,
    plot_confusion_matrix,
    plot_training_curves,
    print_metrics,
    save_checkpoint,
    save_results,
    set_seed,
)

__all__ = [
    # 基础训练工具
    "set_seed",
    "compute_metrics",
    "print_metrics",
    "save_results",
    "load_results",
    "plot_confusion_matrix",
    "plot_training_curves",
    "AverageMeter",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    # 跨被试者评估工具
    "compute_per_subject_metrics",
    "save_per_subject_results",
    "plot_per_subject_metrics",
    "plot_all_metrics_comparison",
    "plot_per_subject_confusion_matrices",
    "print_summary_statistics",
    "save_loso_summary",
]
