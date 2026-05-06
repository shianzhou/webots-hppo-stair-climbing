# 日志系统 (log_code)

## 📁 项目结构

```
log_code/
├── __init__.py              # 包初始化，导出所有公共类
├── base.py                  # BaseLogWriter 基类和编码器
├── catch.py                 # CatchAgentLog（抓取智能体）
├── tai.py                   # TaiAgentLog（抬腿智能体）
├── decision.py              # DecisionAgentLog（决策智能体）
└── README.md               # 本文件
```

## 🎯 快速开始

### 导入

推荐使用方式：

```python
from python_scripts.PPO.log_code import CatchAgentLog, TaiAgentLog, DecisionAgentLog
```

### 基础使用

#### 抓取智能体日志

```python
from python_scripts.PPO.log_code import CatchAgentLog

# 创建日志记录器
log_catch = CatchAgentLog(keep_records=False)

# 记录一个 episode
log_catch.add_episode(
    episode_num=1,
    total_episode=100,
    loss_discrete=0.15,
    loss_continuous=0.22,
    episode_reward=42.5,
    episode_steps=50,
    catch_success=True
)

# 保存日志
log_catch.save('path/to/catch_log.json')
```

#### 抬腿智能体日志

```python
from python_scripts.PPO.log_code import TaiAgentLog

log_tai = TaiAgentLog(keep_records=False)

log_tai.add_episode(
    episode_num=1,
    total_episode=100,
    loss_discrete=0.18,
    loss_continuous=0.25,
    episode_reward=35.8,
    episode_steps=40,
    tai_success=True
)

log_tai.save('path/to/tai_log.json')
```

#### 决策智能体日志

```python
from python_scripts.PPO.log_code import DecisionAgentLog

log_decision = DecisionAgentLog(keep_records=False)

log_decision.add_cycle(
    total_episode=100,
    decision_action=0,  # 0=抓取, 1=抬腿
    loss_discrete=0.12,
    loss_continuous=0.08,
    decision_reward=5.0,
    route='catch',
    pre_catch_success=False,
    post_catch_success=True
)

log_decision.save('path/to/decision_log.json')
```

## 📊 日志数据结构

保存的 JSON 文件包含两种格式的数据：

```json
{
    "start time": "2026-05-06 10:30:45",
    "save time": ["2026-05-06 10:35:20"],
    "series": {
        "episode_num": [1, 2, 3, ...],
        "loss_discrete": [0.15, 0.14, 0.13, ...],
        "loss_continuous": [0.22, 0.21, 0.20, ...],
        "episode_reward": [42.5, 43.2, 41.8, ...],
        ...
    },
    "records": [
        {
            "episode_num": 1,
            "loss_discrete": 0.15,
            "loss_continuous": 0.22,
            ...
        },
        ...
    ]
}
```

- **series**: 按列组织，每个字段是一个数组 → **便于绘图和数据分析**
- **records**: 逐条存储（仅当 `keep_records=True` 时）

## 📝 完整 API 参考

### 通用基类：BaseLogWriter

```python
from python_scripts.PPO.log_code import BaseLogWriter

writer = BaseLogWriter(keep_records=False)

# 添加循环记录
writer.add_cycle_record(
    episode_num=1,
    loss=0.15,
    reward=42.5
)

# 保存到文件
writer.save('log.json')

# 重置日志
writer.reset()
```

### CatchAgentLog（抓取智能体）

| 方法 | 说明 |
|------|------|
| `__init__(keep_records=False)` | 初始化日志记录器 |
| `add_episode(...)` | 记录一个 episode |
| `save(file_path)` | 保存日志到文件 |
| `reset()` | 重置日志 |
| `get_data()` | 获取所有日志数据 |

**add_episode() 参数**：
- `episode_num` (int): 本阶段 episode 编号
- `total_episode` (int): 总 episode 编号
- `loss_discrete` (float): 离散动作网络 loss
- `loss_continuous` (float): 连续动作网络 loss
- `episode_reward` (float): episode 总奖励
- `episode_steps` (int): episode 步数
- `catch_success` (bool): 是否成功抓取
- `**extra`: 其他自定义字段

### TaiAgentLog（抬腿智能体）

同 CatchAgentLog，但 `catch_success` 改为 `tai_success`

### DecisionAgentLog（决策智能体）

**add_cycle() 参数**：
- `total_episode` (int): 总循环编号
- `decision_action` (int): 决策动作（0=抓取, 1=抬腿, ...）
- `loss_discrete` (float): 离散动作 loss
- `loss_continuous` (float): 连续动作 loss（可选）
- `decision_reward` (float): 决策获得的奖励
- `route` (str): 执行路由标签（'catch', 'tai', ...）
- `pre_catch_success` (bool): 决策前抓取成功状态
- `post_catch_success` (bool): 决策后抓取成功状态
- `**extra`: 其他自定义字段

## 🔗 集成到训练脚本

### PPO_episoid_1.py 中的使用

```python
from python_scripts.PPO.log_code import CatchAgentLog, TaiAgentLog, DecisionAgentLog
from python_scripts.PPO.preparation_tool.checkpoint_utils import _next_log_file
from python_scripts.Project_config import path_list

# ===== 初始化日志记录器 =====
log_catch = CatchAgentLog(keep_records=False)
log_tai = TaiAgentLog(keep_records=False)
log_decision = DecisionAgentLog(keep_records=False)

# ===== 获取日志文件路径 =====
log_file_catch = _next_log_file(path_list['catch_log_path_PPO'], 'catch_log')
log_file_tai = _next_log_file(path_list['tai_log_path_PPO'], 'tai_log')
log_file_decision = _next_log_file(path_list['decision_log_path_PPO'], 'decision_log')

# ===== 在训练循环中使用 =====
# 抓取阶段
log_catch.add_episode(
    episode_num=episode_num,
    total_episode=total_episode,
    loss_discrete=loss_d,
    loss_continuous=loss_c,
    episode_reward=episode_reward,
    episode_steps=steps,
    catch_success=catch_success
)
log_catch.save(log_file_catch)

# 抬腿阶段
log_tai.add_episode(
    episode_num=tai_episode,
    total_episode=total_episode,
    loss_discrete=tai_loss_d,
    loss_continuous=tai_loss_c,
    episode_reward=tai_reward,
    episode_steps=tai_steps,
    tai_success=tai_success
)
log_tai.save(log_file_tai)

# 决策阶段
log_decision.add_cycle(
    total_episode=total_episode,
    decision_action=decision,
    loss_discrete=decision_loss_d,
    decision_reward=decision_reward,
    route=route,
    pre_catch_success=pre_catch_success,
    post_catch_success=catch_success
)
log_decision.save(log_file_decision)
```

## 💡 高级特性

### 自定义字段

所有日志方法都支持 `**extra` 参数添加自定义字段：

```python
log_catch.add_episode(
    episode_num=1,
    loss_discrete=0.15,
    custom_field_1=100,
    custom_field_2="test",
    **other_params
)
```

### 双存储模式

- `keep_records=False` (默认): 仅存储 `series`（节省空间）
- `keep_records=True`: 同时存储 `series` 和 `records`（保留详细历史）

### 自动类型转换

系统自动处理：
- PyTorch Tensor → 数值
- NumPy 数组 → 列表
- numpy 数据类型 → Python 原生类型

## 📈 日志分析示例

```python
import json
import matplotlib.pyplot as plt

# 加载日志
with open('catch_log.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

series = data['series']

# 绘制 loss 曲线
plt.plot(series['episode_num'], series['loss_discrete'], label='loss_discrete')
plt.plot(series['episode_num'], series['loss_continuous'], label='loss_continuous')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 获取统计信息
print(f"Total episodes: {len(series['episode_num'])}")
print(f"Success rate: {series['catch_success'].count(True) / len(series['catch_success'])}")
```

## 🔗 相关文件

- 原始基类: `python_scripts/PPO_Log_write.py`（保持向后兼容）
- 主训练脚本: `PPO_episoid_1.py`
- 配置: `Project_config.py`

## ✅ 检查清单

- [x] 通用日志基类实现
- [x] 三个智能体专用模块
- [x] 完整 API 文档
- [x] 使用示例
- [x] 向后兼容
- [x] 自动类型转换
- [x] 双格式存储（series + records）

---

**版本**: 1.0.0  
**最后更新**: 2026-05-06
