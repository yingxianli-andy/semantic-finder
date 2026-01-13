"""
奖励函数定义模块

实现三种极化度计算方法和奖励函数：
1. 方差（Variance）
2. 加权分歧（Weighted Disagreement）
3. 舆论回音室指数（Echo Chamber Score）

奖励 = 当前极化度 - 下一状态极化度（极化度下降量）
"""
import numpy as np
import networkx as nx
from typing import Optional


def compute_polarization_variance(graph: nx.Graph = None, opinions: np.ndarray = None) -> float:
    """
    计算极化度：方差方法（优化版）
    
    公式：P(G) = Var(x) = (1/N) * sum((x_i - x_bar)^2)
    其中 x_i 是节点 i 的观点值，x_bar 是平均观点值
    
    Args:
        graph: NetworkX 图对象，节点必须有 'opinion' 属性（如果提供）
        opinions: 观点值数组（如果提供，直接使用，避免图遍历）
    
    Returns:
        polarization: 极化度值（非负）
    """
    # 性能优化：如果提供了opinions数组，直接使用，避免图遍历
    if opinions is not None:
        if len(opinions) == 0:
            return 0.0
        mean_opinion = np.mean(opinions)
        variance = np.mean((opinions - mean_opinion) ** 2)
        return float(variance)
    
    # 向后兼容：如果没有提供opinions，从图读取
    if graph is None:
        raise ValueError("必须提供 graph 或 opinions 参数之一")
    
    opinions = np.array([graph.nodes[node].get('opinion', 0.0) for node in graph.nodes()])
    if len(opinions) == 0:
        return 0.0
    
    mean_opinion = np.mean(opinions)
    variance = np.mean((opinions - mean_opinion) ** 2)
    return float(variance)


def compute_polarization_weighted_disagreement(graph: nx.Graph) -> float:
    """
    计算极化度：加权分歧方法
    
    公式：P(G) = sum_{(i,j) in E} w_ij * (x_i - x_j)^2
    表示邻居间观点差异的加权和（Lyapunov 函数）
    
    Args:
        graph: NetworkX 图对象，节点必须有 'opinion' 属性
    
    Returns:
        polarization: 极化度值（非负）
    """
    total_disagreement = 0.0
    
    for edge in graph.edges():
        node_i, node_j = edge
        opinion_i = graph.nodes[node_i].get('opinion', 0.0)
        opinion_j = graph.nodes[node_j].get('opinion', 0.0)
        
        # 如果边有权重，使用权重；否则权重为1
        weight = graph.edges[edge].get('weight', 1.0)
        
        disagreement = weight * (opinion_i - opinion_j) ** 2
        total_disagreement += disagreement
    
    return float(total_disagreement)


def compute_polarization_echo_chamber(graph: nx.Graph, threshold: float = 0.0) -> float:
    """
    计算极化度：舆论回音室指数
    
    计算不同阵营间的边数占比。如果两个节点的观点在阈值两侧，则认为它们属于不同阵营。
    回音室指数 = 1 - (跨阵营边数 / 总边数)
    值越大，表示回音室效应越强（极化度越高）
    
    Args:
        graph: NetworkX 图对象，节点必须有 'opinion' 属性
        threshold: 观点阈值，用于划分阵营（默认0.0，即正负阵营）
    
    Returns:
        polarization: 回音室指数（0-1之间，1表示完全极化）
    """
    if graph.number_of_edges() == 0:
        return 0.0
    
    cross_camp_edges = 0
    total_edges = graph.number_of_edges()
    
    for edge in graph.edges():
        node_i, node_j = edge
        opinion_i = graph.nodes[node_i].get('opinion', 0.0)
        opinion_j = graph.nodes[node_j].get('opinion', 0.0)
        
        # 判断是否跨阵营：一个在阈值左侧，一个在阈值右侧
        if (opinion_i <= threshold and opinion_j > threshold) or \
           (opinion_i > threshold and opinion_j <= threshold):
            cross_camp_edges += 1
    
    # 回音室指数：跨阵营边数越少，回音室效应越强
    echo_chamber_score = 1.0 - (cross_camp_edges / total_edges)
    return float(echo_chamber_score)


def compute_reward(
    state: nx.Graph,
    next_state: nx.Graph,
    method: str = "variance"
) -> float:
    """
    计算奖励值：极化度的下降量
    
    公式：r_t = P(G_t) - P(G_{t+1})
    奖励为正表示极化度下降（好的行为），奖励为负表示极化度上升（不好的行为）
    
    Args:
        state: 当前状态（图对象）
        next_state: 下一状态（图对象）
        method: 极化度计算方法
            - "variance": 方差方法
            - "weighted_disagreement": 加权分歧方法
            - "echo_chamber": 回音室指数方法
    
    Returns:
        reward: 奖励值（标量）
    """
    # 计算当前状态的极化度
    if method == "variance":
        polarization_current = compute_polarization_variance(state)
        polarization_next = compute_polarization_variance(next_state)
    elif method == "weighted_disagreement":
        polarization_current = compute_polarization_weighted_disagreement(state)
        polarization_next = compute_polarization_weighted_disagreement(next_state)
    elif method == "echo_chamber":
        polarization_current = compute_polarization_echo_chamber(state)
        polarization_next = compute_polarization_echo_chamber(next_state)
    else:
        raise ValueError(f"Unknown polarization method: {method}. "
                       f"Choose from: 'variance', 'weighted_disagreement', 'echo_chamber'")
    
    # 奖励 = 极化度下降量
    reward = polarization_current - polarization_next
    
    return float(reward)


