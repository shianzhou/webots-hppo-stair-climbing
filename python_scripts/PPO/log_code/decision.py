"""
决策智能体（Decision Agent）日志写入器
记录决策层的所有训练数据、loss、决策选择、reward 等
"""

import numpy as np
import json
from datetime import datetime
from .base import BaseLogWriter, CustomJSONEncoder


class DecisionAgentLog:
    """决策智能体专用日志写入器"""
    
    def __init__(self, keep_records=False):
        """
        初始化决策智能体日志
        
        Args:
            keep_records (bool): 是否同时保存逐条记录和列式数据
        """
        self.writer = BaseLogWriter(keep_records=keep_records)
        self.agent_name = 'DecisionAgent'
        
    def add_cycle(self, total_episode=None, decision_action=None, loss_discrete=None, 
                 loss_continuous=None, decision_reward=None, route=None, 
                 pre_catch_success=None, post_catch_success=None, **extra):
        """
        记录一个决策循环
        
        Args:
            total_episode (int): 总循环编号
            decision_action (int): 决策动作（例如：0=抓取, 1=抬腿）
            loss_discrete (float): 离散动作网络 loss
            loss_continuous (float): 连续动作网络 loss（若有）
            decision_reward (float): 决策层获得的奖励
            route (str): 执行的路由标签（'catch', 'tai', 等）
            pre_catch_success (bool): 决策前抓取是否成功
            post_catch_success (bool): 决策后抓取是否成功
            **extra: 其他自定义字段
        """
        record = {
            'total_episode': total_episode,
            'decision_action': decision_action,
            'loss_discrete': loss_discrete,
            'loss_continuous': loss_continuous,
            'decision_reward': decision_reward,
            'route': route,
            'pre_catch_success': pre_catch_success,
            'post_catch_success': post_catch_success,
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
