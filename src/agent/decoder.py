"""
解码器模块

实现 DQN 解码器（MLP），将节点嵌入映射为 Q 值。
输出：每个节点的 Q(s, a) 值，表示选择该节点的预期累积奖励。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNDecoder(nn.Module):
    """
    DQN 解码器，使用 MLP 将节点嵌入映射为 Q 值
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入维度（节点嵌入维度）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（通常为1，表示Q值）
            dropout: Dropout 概率
        """
        super(DQNDecoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_embeddings: 节点嵌入 [N, input_dim]
        
        Returns:
            q_values: Q 值 [N, output_dim] 或 [N]（如果 output_dim=1）
        """
        q_values = self.mlp(node_embeddings)
        
        # 如果输出维度为1，压缩最后一维
        if q_values.shape[-1] == 1:
            q_values = q_values.squeeze(-1)
        
        return q_values


