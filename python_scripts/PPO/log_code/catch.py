"""
抓取智能体（Catch Agent）日志写入器
记录抓取阶段的所有训练数据、loss、reward 等
"""

import numpy as np
import json
from datetime import datetime
from .base import BaseLogWriter, CustomJSONEncoder


class CatchAgentLog:
    """抓取智能体专用日志写入器"""
    
    def __init__(self, keep_records=False):
        """
        初始化抓取智能体日志
        
        Args:
            keep_records (bool): 是否同时保存逐条记录和列式数据
        """
        self.writer = BaseLogWriter(keep_records=keep_records)
        self.agent_name = 'CatchAgent'
        
    def add_episode(self, episode_num=None, total_episode=None, loss_discrete=None, 
                   loss_continuous=None, episode_reward=None, episode_steps=None, 
                   catch_success=None, **extra):
        """
        记录一个完整的 episode（抓取阶段）
        
        Args:
            episode_num (int): 本阶段 episode 编号
            total_episode (int): 总 episode 编号
            loss_discrete (float): 离散动作网络 loss
            loss_continuous (float): 连续动作网络 loss
            episode_reward (float): 整个 episode 的累积奖励
            episode_steps (int): episode 步数
            catch_success (bool): 是否成功抓取
            **extra: 其他自定义字段
        """
        record = {
            'episode_num': episode_num,
            'total_episode': total_episode,
            'loss_discrete': loss_discrete,
            'loss_continuous': loss_continuous,
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'catch_success': catch_success,
        }
        for key, value in extra.items():
            record[key] = value
        self.writer.add_cycle_record(**record)
    
    def save(self, file_path):
        """保存日志到文件"""
        self.writer.save(file_path)
    
    def reset(self):
        """重置日志"""
        self.writer.reset()
    
    def get_data(self):
        """获取所有日志数据"""
        return self.writer.data
