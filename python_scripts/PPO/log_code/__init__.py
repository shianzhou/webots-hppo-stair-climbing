"""
PPO 训练日志系统
=====================

通用日志写入器和三个智能体专用日志模块

模块组成：
- base.py: BaseLogWriter（通用基类）、CustomJSONEncoder
- catch.py: CatchAgentLog（抓取智能体日志）
- tai.py: TaiAgentLog（抬腿智能体日志）
- decision.py: DecisionAgentLog（决策智能体日志）

快速开始：
    from python_scripts.PPO.log_code import CatchAgentLog, TaiAgentLog, DecisionAgentLog
    
    log_catch = CatchAgentLog()
    log_catch.add_episode(episode_num=1, loss_discrete=0.15, ...)
    log_catch.save('catch_log.json')
"""

# 导出基类和编码器
from .base import (
    BaseLogWriter,
    CustomJSONEncoder,
    CatchAgentLogWriter,
    TaiAgentLogWriter,
    DecisionAgentLogWriter,
    Log_write,
)

# 导出专用日志类
from .catch import CatchAgentLog
from .tai import TaiAgentLog
from .decision import DecisionAgentLog

__all__ = [
    # 基类
    'BaseLogWriter',
    'CustomJSONEncoder',
    'Log_write',
    # 基于基类的写入器（向后兼容）
    'CatchAgentLogWriter',
    'TaiAgentLogWriter',
    'DecisionAgentLogWriter',
    # 推荐使用的专用日志类
    'CatchAgentLog',
    'TaiAgentLog',
    'DecisionAgentLog',
]

__version__ = '1.0.0'
