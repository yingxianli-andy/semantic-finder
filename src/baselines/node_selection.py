"""
基线节点选择方法实现。

用于IJCAI实验对比。
"""

import random
import numpy as np
from typing import List, Optional

import networkx as nx
import torch

from ..agent.semantic_finder import SemanticFINDER
from ..agent.encoder import graph_to_pyg_data
from ..llm.controller import LLMController
from torch_geometric.data import Data


def select_nodes_random(
    graph: nx.Graph, k: int, seed: Optional[int] = None
) -> List[int]:
    """
    随机选择 K 个节点。

    Args:
        graph: 图对象
        k: 需要选择的节点数量
        seed: 随机种子

    Returns:
        selected_nodes: 选中的节点索引列表
    """
    if seed is not None:
        random.seed(seed)
    n_nodes = graph.number_of_nodes()
    if k > n_nodes:
        k = n_nodes
    return random.sample(range(n_nodes), k)


def select_nodes_high_degree(graph: nx.Graph, k: int) -> List[int]:
    """
    选择度数最大的 K 个节点（大V策略）。

    Args:
        graph: 图对象
        k: 需要选择的节点数量

    Returns:
        selected_nodes: 选中的节点索引列表（按度数从大到小排序）
    """
    n_nodes = graph.number_of_nodes()
    if k > n_nodes:
        k = n_nodes

    # 计算所有节点的度数
    degrees = [(i, graph.degree(i)) for i in range(n_nodes)]
    # 按度数从大到小排序
    degrees.sort(key=lambda x: x[1], reverse=True)
    # 返回前k个节点
    return [node_id for node_id, _ in degrees[:k]]


def select_nodes_pagerank(
    graph: nx.Graph, k: int, alpha: float = 0.85
) -> List[int]:
    """
    选择 PageRank 值最高的 K 个节点。

    Args:
        graph: 图对象
        k: 需要选择的节点数量
        alpha: PageRank 阻尼系数（默认0.85）

    Returns:
        selected_nodes: 选中的节点索引列表（按 PageRank 值从大到小排序）
    """
    n_nodes = graph.number_of_nodes()
    if k > n_nodes:
        k = n_nodes

    # 使用 networkx 的 pagerank 函数
    pagerank = nx.pagerank(graph, alpha=alpha)
    # 按 PageRank 值从大到小排序
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    # 返回前k个节点
    return [node_id for node_id, _ in sorted_nodes[:k]]


def select_nodes_finder(
    graph: nx.Graph,
    k: int,
    model_path: str = "results/models/final_model.pth",
    device: str = "cpu",
    preprocessed_data: Optional[Data] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[int]:
    """
    使用预训练的 RL 模型（Original FINDER，无语义）选择节点。
    使用ε-greedy策略：75%概率选择Q值最高的节点（贪心），25%概率随机探索。

    Args:
        graph: 图对象
        k: 需要选择的节点数量
        model_path: 模型文件路径
        device: 设备（cpu 或 cuda）
        preprocessed_data: 预处理的PyG Data对象
        seed: 随机种子（用于ε-greedy的随机探索，确保可重复性）
        temperature: 已弃用，保留以保持接口兼容性

    Returns:
        selected_nodes: 选中的节点索引列表
    """
    n_nodes = graph.number_of_nodes()
    if k > n_nodes:
        k = n_nodes

    # 加载模型（优化：使用全局缓存，避免重复加载）
    cache_key = f"{model_path}_{device}"
    if not hasattr(select_nodes_finder, '_model_cache'):
        select_nodes_finder._model_cache = {}
    
    if cache_key not in select_nodes_finder._model_cache:
        try:
            model = SemanticFINDER(device=device)
            model.load(model_path)
            model.eval()
            select_nodes_finder._model_cache[cache_key] = model
        except Exception as e:
            raise RuntimeError(f"无法加载模型 {model_path}: {e}")
    else:
        model = select_nodes_finder._model_cache[cache_key]

    # 确保有 PyG Data，避免重复转换
    data_for_model = preprocessed_data if preprocessed_data is not None else graph_to_pyg_data(graph)

    # 一次性前向，获取所有节点的 Q 值
    with torch.no_grad():
        q_values = model.forward(graph=None, data=data_for_model, verbose=False).flatten()

    # 转到 CPU 做掩码筛选
    q_np = q_values.cpu().numpy()
    
    selected_nodes: List[int] = []
    mask = np.zeros(n_nodes, dtype=bool)

    # 使用seed初始化随机数生成器（用于ε-greedy的随机探索）
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    
    # ε-greedy策略：75%贪心选择，25%随机探索
    epsilon = 0.25  # 25%探索率，75%利用（贪心）

    for step in range(k):
        valid_indices = np.where(~mask)[0]
        if valid_indices.size == 0:
            break
        
        # 获取未选中节点的Q值
        valid_q = q_np[valid_indices]
        
        # ε-greedy策略
        if rng.random() < epsilon:
            # 25%概率：随机探索（从有效节点中随机选择）
            best_local_idx = rng.randint(0, len(valid_indices))
            best_idx = valid_indices[best_local_idx]
        else:
            # 75%概率：贪心选择（选择Q值最高的节点）
            best_local_idx = np.argmax(valid_q)
            best_idx = valid_indices[best_local_idx]

        selected_nodes.append(int(best_idx))
        mask[best_idx] = True
        q_np[best_idx] = -np.inf  # 置为 -inf，保证下轮不再选

    return selected_nodes


def select_nodes_semantic_finder(
    graph: nx.Graph,
    k: int,
    model_path: str = "results/models/final_model.pth",
    llm_controller: Optional[LLMController] = None,
    device: str = "cpu",
    preprocessed_data: Optional[Data] = None,
) -> List[int]:
    """
    使用 Semantic-FINDER（RL 选点 + LLM 权重）选择节点。

    注意：这个方法会在选点的同时获取 LLM 权重，但权重不在这里返回
    （权重在实验循环中使用）。

    Args:
        graph: 图对象
        k: 需要选择的节点数量
        model_path: 模型文件路径
        llm_controller: LLMController 实例（如果为None，则只使用RL选点）
        device: 设备（cpu 或 cuda）

    Returns:
        selected_nodes: 选中的节点索引列表
    """
    # 实际上，Semantic-FINDER 的选点逻辑和 Original FINDER 相同
    # 区别在于权重的获取（在实验循环中处理）
    return select_nodes_finder(graph, k, model_path, device, preprocessed_data=preprocessed_data)
