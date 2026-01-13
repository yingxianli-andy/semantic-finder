"""
经验回放缓冲区模块

实现 DQN 的经验回放缓冲区，存储和采样经验元组 (state, action, reward, next_state, done)。
"""
import random
from collections import deque
from typing import Optional, Tuple
import networkx as nx


class ReplayBuffer:
    """
    经验回放缓冲区
    存储 (state, action, reward, next_state, done) 元组
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: nx.Graph,
        action: int,
        reward: float,
        next_state: nx.Graph,
        done: bool
    ):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态（图对象）
            action: 动作（节点索引）
            reward: 奖励值
            next_state: 下一状态（图对象）
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> list:
        """
        从缓冲区随机采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            batch: 经验批次列表
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 32) -> bool:
        """
        检查缓冲区是否有足够的经验用于训练
        
        Args:
            min_size: 最小经验数量
        
        Returns:
            ready: 是否准备好
        """
        return len(self.buffer) >= min_size


