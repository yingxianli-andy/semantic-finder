"""
Semantic-FINDER 主类

实现基于强化学习的关键节点发现算法。
结合 GraphSAGE/GCN 编码器和 DQN 解码器，学习选择最优干预节点。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from typing import Optional, Tuple
import copy

from .encoder import GraphEncoder, graph_to_pyg_data
from .decoder import DQNDecoder
from torch_geometric.data import Data


class SemanticFINDER(nn.Module):
    """
    Semantic-FINDER Agent
    
    基于 FINDER 的改进版本，用于在社交网络中选择关键节点进行干预，
    以降低网络极化度。
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # 默认：度数 + 观点值
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "GraphSAGE",
        decoder_hidden_dims: list = [128, 64],
        use_semantic_embedding: bool = False,
        semantic_embedding_dim: int = 0,
        learning_rate: float = 0.001,
        gamma: float = 0.99,  # 折扣因子
        device: str = "cpu"
    ):
        """
        Args:
            input_dim: 输入特征维度（不包括语义嵌入）
            hidden_dim: 隐藏层维度
            output_dim: 编码器输出维度
            num_layers: GNN 层数
            encoder_type: 编码器类型，"GraphSAGE" 或 "GCN"
            decoder_hidden_dims: 解码器隐藏层维度列表
            use_semantic_embedding: 是否使用语义嵌入
            semantic_embedding_dim: 语义嵌入维度
            learning_rate: 学习率
            gamma: 折扣因子
            device: 设备（"cpu" 或 "cuda"）
        """
        super(SemanticFINDER, self).__init__()
        
        self.device = torch.device(device)
        self.gamma = gamma
        
        # 编码器
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            encoder_type=encoder_type,
            use_semantic_embedding=use_semantic_embedding,
            semantic_embedding_dim=semantic_embedding_dim
        )
        
        # 解码器（Q-network）
        self.decoder = DQNDecoder(
            input_dim=output_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=1,
            dropout=0.1
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 移动到设备
        self.to(self.device)
        
        # 用于存储语义嵌入（如果使用）
        self.semantic_embeddings = None
        self.use_semantic_embedding = use_semantic_embedding
    
    def set_semantic_embeddings(self, semantic_embeddings: dict):
        """
        设置语义嵌入字典
        
        Args:
            semantic_embeddings: {node_id: embedding_vector}
        """
        self.semantic_embeddings = semantic_embeddings
    
    def forward(self, graph: nx.Graph = None, data: Optional[Data] = None, verbose: bool = False) -> torch.Tensor:
        """
        前向传播，计算所有节点的 Q 值
        
        Args:
            graph: NetworkX 图对象（如果data为None则使用）
            data: 预处理的PyG Data对象（如果提供则跳过图转换）
            verbose: 是否输出进度信息
        
        Returns:
            q_values: Q 值张量 [N]，N 为节点数量
        """
        import time
        start_time = time.time()
        
        # 如果提供了预处理的data，直接使用；否则从graph转换
        if data is None:
            if graph is None:
                raise ValueError("必须提供graph或data参数之一")
            num_nodes = graph.number_of_nodes()
        else:
            num_nodes = data.x.shape[0]
        
        if verbose and num_nodes > 10000:
            print(f"    开始计算Q值（{num_nodes:,} 节点）...", end="", flush=True)
        
        # 将图转换为 PyG Data 对象（优先使用预处理好的 data，避免重复转换）
        if data is None:
            convert_start = time.time()
            data = graph_to_pyg_data(
                graph,
                use_semantic_embedding=self.use_semantic_embedding,
                semantic_embeddings=self.semantic_embeddings,
                verbose=verbose
            )
            convert_time = time.time() - convert_start
            if verbose and num_nodes > 10000:
                print(f" 图转换完成 ({convert_time:.2f}s)", end="", flush=True)
        else:
            if verbose and num_nodes > 10000:
                print(f"    使用预处理数据（{num_nodes:,} 节点）...", end="", flush=True)

        data = data.to(self.device)
        
        # 编码：图 -> 节点嵌入
        encode_start = time.time()
        with torch.no_grad():
            node_embeddings = self.encoder(data)
        encode_time = time.time() - encode_start
        
        if verbose and num_nodes > 10000:
            print(f" 编码完成 ({encode_time:.2f}s)", end="", flush=True)
        
        # 解码：节点嵌入 -> Q 值
        decode_start = time.time()
        with torch.no_grad():
            q_values = self.decoder(node_embeddings)
        decode_time = time.time() - decode_start
        
        total_time = time.time() - start_time
        if verbose and num_nodes > 10000:
            print(f" 解码完成 ({decode_time:.2f}s) 总耗时: {total_time:.2f}s", flush=True)
        
        return q_values
    
    def select_action(
        self,
        graph: nx.Graph = None,
        data: Optional[Data] = None,
        mask: Optional[list] = None,
        epsilon: float = 0.0,
        training: bool = True
    ) -> int:
        """
        选择动作（节点）
        
        使用 epsilon-greedy 策略：
        - 以 epsilon 概率随机选择（探索）
        - 以 (1-epsilon) 概率选择 Q 值最大的节点（利用）
        
        Args:
            graph: 当前状态图（如果data为None则使用）
            data: 预处理的PyG Data对象（如果提供则跳过图转换）
            mask: 掩码列表，已选过的节点设为 True（不能选择）
            epsilon: 探索率（0-1）
            training: 是否在训练模式
        
        Returns:
            action: 选择的节点索引
        """
        if data is None:
            if graph is None:
                raise ValueError("必须提供graph或data参数之一")
            num_nodes = graph.number_of_nodes()
        else:
            num_nodes = data.x.shape[0]
        
        if num_nodes == 0:
            raise ValueError("Graph has no nodes")
        
        # 如果所有节点都被 mask 了，返回 -1（无效动作）
        if mask is not None and all(mask):
            return -1
        
        # Epsilon-greedy 策略
        if training and np.random.random() < epsilon:
            # 随机选择（探索）
            if mask is None:
                action = np.random.randint(0, num_nodes)
            else:
                # 从未被 mask 的节点中随机选择
                valid_actions = [i for i in range(num_nodes) if not mask[i]]
                if len(valid_actions) == 0:
                    return -1
                action = np.random.choice(valid_actions)
        else:
            # 选择 Q 值最大的节点（利用）
            self.eval()
            with torch.no_grad():
                # 对于大图，输出进度信息
                verbose = num_nodes > 10000
                q_values = self.forward(graph=graph, data=data, verbose=verbose)
            
            # 应用 mask：将已选节点的 Q 值设为负无穷
            if mask is not None:
                q_values_np = q_values.cpu().numpy()
                q_values_np[mask] = -np.inf
                q_values = torch.tensor(q_values_np, device=self.device)
            
            action = q_values.argmax().item()
        
        return action
    
    def update(
        self,
        replay_buffer,
        batch_size: int = 32
    ) -> float:
        """
        使用经验回放更新网络参数
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
        
        Returns:
            loss: 损失值
        """
        if not replay_buffer.is_ready(batch_size):
            return 0.0
        
        # 采样一批经验
        batch = replay_buffer.sample(batch_size)
        
        # 准备批次数据
        states = [exp[0] for exp in batch]
        actions = [exp[1] for exp in batch]
        rewards = [exp[2] for exp in batch]
        next_states = [exp[3] for exp in batch]
        dones = [exp[4] for exp in batch]
        
        # 计算当前 Q 值
        current_q_values = []
        for state, action in zip(states, actions):
            q_values = self.forward(state)
            current_q_values.append(q_values[action])
        
        current_q_values = torch.stack(current_q_values)
        
        # 计算目标 Q 值（使用目标网络或当前网络）
        # 这里使用当前网络（简化版），实际可以使用目标网络提高稳定性
        next_q_values = []
        for next_state, done in zip(next_states, dones):
            if done:
                next_q_values.append(torch.tensor(0.0, device=self.device))
            else:
                with torch.no_grad():
                    q_values = self.forward(next_state)
                    next_q_values.append(q_values.max())
        
        next_q_values = torch.stack(next_q_values)
        
        # 目标 Q 值：r + gamma * max Q(s', a')
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        target_q_values = rewards_tensor + self.gamma * next_q_values
        
        # 计算损失（MSE）
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



