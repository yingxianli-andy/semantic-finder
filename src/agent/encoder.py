"""
图编码器模块

实现 GraphSAGE 或 GCN 编码器，将图结构编码为节点嵌入。
输入特征：h_v = [d_v, x_v, e_v]
- d_v: 归一化度数
- x_v: 观点值（-1 到 1）
- e_v: 语义嵌入（可选）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data
from typing import Optional


class GraphEncoder(nn.Module):
    """
    图编码器，使用 GraphSAGE 或 GCN 将图结构编码为节点嵌入
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "GraphSAGE",
        use_semantic_embedding: bool = False,
        semantic_embedding_dim: int = 0
    ):
        """
        Args:
            input_dim: 输入特征维度（不包括语义嵌入）
                      通常为 2（度数 + 观点值）或 3（如果包含其他特征）
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_layers: GNN 层数
            encoder_type: 编码器类型，"GraphSAGE" 或 "GCN"
            use_semantic_embedding: 是否使用语义嵌入
            semantic_embedding_dim: 语义嵌入维度（如果使用）
        """
        super(GraphEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.use_semantic_embedding = use_semantic_embedding
        
        # 如果使用语义嵌入，输入维度需要加上语义嵌入维度
        if use_semantic_embedding:
            actual_input_dim = input_dim + semantic_embedding_dim
        else:
            actual_input_dim = input_dim
        
        self.layers = nn.ModuleList()
        
        # 第一层
        if encoder_type == "GraphSAGE":
            self.layers.append(SAGEConv(actual_input_dim, hidden_dim))
        elif encoder_type == "GCN":
            self.layers.append(GCNConv(actual_input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from: 'GraphSAGE', 'GCN'")
        
        # 中间层
        for _ in range(num_layers - 2):
            if encoder_type == "GraphSAGE":
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # 最后一层
        if num_layers > 1:
            if encoder_type == "GraphSAGE":
                self.layers.append(SAGEConv(hidden_dim, output_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyTorch Geometric Data 对象，包含：
                - x: 节点特征 [N, input_dim]
                - edge_index: 边索引 [2, E]
        
        Returns:
            node_embeddings: 节点嵌入 [N, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # 通过 GNN 层
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        return x


def graph_to_pyg_data(
    graph: nx.Graph,
    use_semantic_embedding: bool = False,
    semantic_embeddings: Optional[dict] = None,
    verbose: bool = False
) -> Data:
    """
    将 NetworkX 图转换为 PyTorch Geometric Data 对象
    
    Args:
        graph: NetworkX 图对象，节点必须有 'opinion' 属性
        use_semantic_embedding: 是否使用语义嵌入
        semantic_embeddings: 语义嵌入字典 {node_id: embedding_vector}
        verbose: 是否输出进度信息
    
    Returns:
        data: PyTorch Geometric Data 对象
    """
    import time
    start_time = time.time()
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    if verbose and num_nodes > 10000:
        print(f"  转换图 ({num_nodes:,} 节点, {num_edges:,} 边)...", end="", flush=True)
    
    # 优化：预先计算所有度数
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # 构建节点ID到索引的映射（确保顺序）
    nodes_list = list(graph.nodes())
    node_id_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    
    # 构建节点特征矩阵（优化：使用列表推导式）
    feature_start = time.time()
    node_features = []
    for node in nodes_list:
        # 特征1：归一化度数
        degree = degrees[node]
        normalized_degree = degree / max_degree if max_degree > 0 else 0.0
        
        # 特征2：观点值
        opinion = graph.nodes[node].get('opinion', 0.0)
        
        # 构建特征向量 [d_v, x_v]
        features = [normalized_degree, opinion]
        
        # 如果使用语义嵌入，添加语义特征
        if use_semantic_embedding and semantic_embeddings is not None:
            if node in semantic_embeddings:
                semantic_feat = semantic_embeddings[node]
                if isinstance(semantic_feat, (list, np.ndarray)):
                    features.extend(semantic_feat)
                else:
                    features.append(semantic_feat)
            else:
                # 如果没有语义嵌入，用零向量填充
                if isinstance(semantic_embeddings, dict) and len(semantic_embeddings) > 0:
                    # 获取第一个嵌入的维度
                    first_emb = next(iter(semantic_embeddings.values()))
                    if isinstance(first_emb, (list, np.ndarray)):
                        emb_dim = len(first_emb)
                    else:
                        emb_dim = 1
                    features.extend([0.0] * emb_dim)
        
        node_features.append(features)
    
    # 转换为张量
    x = torch.tensor(node_features, dtype=torch.float32)
    feature_time = time.time() - feature_start
    
    if verbose and num_nodes > 10000:
        print(f" 特征完成 ({feature_time:.2f}s)", end="", flush=True)
    
    # 构建边索引（优化：使用更高效的方法）
    edge_start = time.time()
    # 使用NetworkX的边列表，直接转换为索引
    edge_list = []
    for u, v in graph.edges():
        i, j = node_id_to_idx[u], node_id_to_idx[v]
        edge_list.append([i, j])
        # 无向图，添加反向边
        edge_list.append([j, i])
    
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # 使用更高效的张量创建方式
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    edge_time = time.time() - edge_start
    
    if verbose and num_nodes > 10000:
        print(f" 边索引完成 ({edge_time:.2f}s)", end="", flush=True)
    
    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index)
    
    total_time = time.time() - start_time
    if verbose and num_nodes > 10000:
        print(f" 总耗时: {total_time:.2f}s", flush=True)
    
    return data


