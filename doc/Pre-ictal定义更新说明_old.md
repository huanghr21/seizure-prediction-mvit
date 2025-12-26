# Pre-ictal（发作前）定义更新说明

## 重要变更

项目已从Baseline方法的Pre-ictal定义更新为**论文方法**的定义。

## 定义对比

### Baseline方法（旧）
```
Pre-ictal period = 发作前10秒
时间窗口: [seizure_start - 10s, seizure_start]
窗口长度: 10秒
```

**问题**：
- 时间窗口太短，临床上几乎无法进行有效干预
- 不符合实际应用场景
- 仅用于快速原型验证

### 论文方法（新）✅
```
Pre-ictal period = 发作前30分钟到发作前3分钟
时间窗口: [seizure_start - 30min, seizure_start - 3min]
窗口长度: 27分钟

关键参数:
- SOP (Seizure Onset Period) = 30分钟
- SPH (Seizure Prediction Horizon) = 3分钟
```

**优势**：
- ✅ 提供充足的临床干预时间（27分钟）
- ✅ SPH提供3分钟的安全缓冲期
- ✅ 符合临床实际应用需求
- ✅ 与论文方法保持一致

## 技术细节

### 时间线示意图

```
Inter-ictal          Pre-ictal Period (27min)     SPH (3min)    Seizure
   期间      |←-------- SOP (30min) -------->|←-- 干预期 -->|   发作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
            ↑                                ↑              ↑
      发作前30分钟                       发作前3分钟      发作开始
   (preictal_start)                  (preictal_end)  (seizure_start)
```

### 关键术语解释

**SOP (Seizure Onset Period)**
- 中文：发作预期期
- 定义：预期癫痫发作的时间窗口
- 本项目设置：30分钟
- 作用：定义Pre-ictal period的起始点

**SPH (Seizure Prediction Horizon)**
- 中文：发作预测地平线 / 临床干预期
- 定义：从预测警报到发作期开始的时间间隔
- 本项目设置：3分钟
- 作用：为临床干预提供缓冲时间，避免预警过晚

**Pre-ictal Period**
- 中文：发作前期
- 定义：[发作前SOP, 发作前SPH] = [发作前30分钟, 发作前3分钟]
- 长度：SOP - SPH = 27分钟
- 标签：正类（PreSeizure = 1）

**Inter-ictal Period**
- 中文：发作间期 / 正常期
- 定义：不在Pre-ictal period内的其他时间
- 标签：负类（non-PreSeizure = 0）

## 代码实现

### 配置文件 (config.py)

```python
# 癫痫预测配置（基于论文方法）
SPH = 3 * 60   # 3分钟 = 180秒
SOP = 30 * 60  # 30分钟 = 1800秒

# Pre-ictal period定义
PREICTAL_START_BEFORE = SOP  # 发作前30分钟开始
PREICTAL_END_BEFORE = SPH     # 到发作前3分钟结束
```

### 数据处理 (src/data_processing.py)

```python
def get_preictal_times(data_root, sop=30*60, sph=3*60):
    """计算发作前时间段"""
    for seizure_start, seizure_end in seizures:
        # Pre-ictal: [seizure_start - SOP, seizure_start - SPH]
        preictal_start = seizure_start - sop  # 发作前30分钟
        preictal_end = seizure_start - sph    # 发作前3分钟
        
        if preictal_end > preictal_start:
            preictal_times.append((preictal_start, preictal_end))
```

## 实际示例

假设某个癫痫发作发生在 `2996秒`：

### Baseline方法（旧）
```
Pre-ictal: [2986s, 2996s]
持续时间: 10秒
```

### 论文方法（新）
```
SOP = 30 * 60 = 1800秒
SPH = 3 * 60 = 180秒

preictal_start = 2996 - 1800 = 1196秒
preictal_end = 2996 - 180 = 2816秒

Pre-ictal: [1196s, 2816s]
持续时间: 1620秒 = 27分钟
```

## 影响范围

### 已更新的文件

1. **config.py**
   - 添加 `SOP`, `SPH` 配置
   - 移除旧的 `PREICTAL_DURATION`

2. **src/data_processing.py**
   - `get_preictal_times()`: 使用SOP/SPH计算
   - `get_preictal_annotations()`: 更新函数签名

3. **experiments/train_mvit.py**
   - 使用新的 `SOP`, `SPH` 配置

4. **test_annotation_loading.py**
   - 显示SOP/SPH信息

5. **README.md**
   - 更新Pre-ictal定义说明

6. **doc/数据标注自动加载说明.md**
   - 详细说明新的Pre-ictal定义

### 需要注意的文件

**baseline/** 目录下的文件保持不变
- `癫痫前识别实验一.py`：仍使用10秒定义（baseline验证）
- `癫痫前识别实验二.py`：仍使用10秒定义（baseline验证）

**原因**：Baseline实验是为了对比，保持原有定义以便比较优化效果。

## 使用建议

### 1. 新实验请使用论文方法
```python
from config import DATA_ROOT, SOP, SPH
from src.data_processing import get_preictal_annotations

# 加载标注（27分钟Pre-ictal窗口）
preictal_times = get_preictal_annotations(DATA_ROOT, SOP, SPH)
```

### 2. Baseline对比实验保持不变
```python
# baseline实验脚本中仍使用10秒定义
# 用于对比优化效果
```

### 3. 自定义SOP/SPH
```python
# 可以尝试不同的参数组合
custom_sop = 20 * 60  # 20分钟
custom_sph = 5 * 60   # 5分钟

preictal_times = get_preictal_annotations(
    DATA_ROOT, 
    sop=custom_sop, 
    sph=custom_sph
)
```

## 参考文献

论文中的相关描述：

> "The SPH was set at 3 min, and the SOP was established as 30 min."

> "The SOP refers to the period during which a seizure is anticipated, while the SPH is the timeframe from the prediction alert to the start of the SOP."

> "The SPH, known as the clinical intervention period, refers to when appropriate actions can be taken in response to the seizure."

## 常见问题

**Q: 为什么要排除发作前3分钟？**
A: 发作前3分钟（SPH）是临床干预期，在这个阶段进行预警可能太晚，无法进行有效干预。

**Q: 27分钟的窗口会不会太长？**
A: 这是基于临床实际需求的设置。27分钟提供了足够的时间进行药物治疗、神经刺激等干预措施。

**Q: 能否使用其他时间窗口？**
A: 可以。通过修改 `config.py` 中的 `SOP` 和 `SPH` 来调整。建议进行消融实验找到最优参数。

**Q: Baseline的10秒定义还有用吗？**
A: 有。它作为对比基准，用于评估优化方法的改进效果。

---

**更新日期**: 2025-12-24  
**版本**: 2.0（论文方法）  
**状态**: ✅ 已实现并测试
