"""
评估指标工具函数

提供各种评估指标的计算，用于评估模型性能和网络状态。
"""
import networkx as nx
import numpy as np
from typing import Dict, List


def compute_polarization_variance(graph: nx.Graph) -> float:
    """
    计算极化度（使用方差方法）
    
    Args:
        graph: 图对象
    
    Returns:
        polarization: 极化度（方差）
    """
    opinions = []
    for node in graph.nodes():
        if 'opinion' in graph.nodes[node]:
            opinions.append(graph.nodes[node]['opinion'])
    
    if not opinions:
        return 0.0
    
    return float(np.var(opinions))


def compute_polarization_weighted_disagreement(graph: nx.Graph) -> float:
    """
    计算极化度（使用加权分歧方法）
    
    Args:
        graph: 图对象
    
    Returns:
        polarization: 极化度（加权分歧）
    """
    total_disagreement = 0.0
    edge_count = 0
    
    for u, v in graph.edges():
        if 'opinion' in graph.nodes[u] and 'opinion' in graph.nodes[v]:
            op_u = graph.nodes[u]['opinion']
            op_v = graph.nodes[v]['opinion']
            disagreement = (op_u - op_v) ** 2
            total_disagreement += disagreement
            edge_count += 1
    
    if edge_count == 0:
        return 0.0
    
    return float(total_disagreement / edge_count)


def compute_echo_chamber_score(graph: nx.Graph, threshold: float = 0.5) -> float:
    """
    计算回音室指数
    
    Args:
        graph: 图对象
        threshold: 观点差异阈值（超过此值认为是不同阵营）
    
    Returns:
        score: 回音室指数（0-1之间，越高表示回音室效应越强）
    """
    if graph.number_of_edges() == 0:
        return 0.0
    
    cross_camp_edges = 0
    total_edges = 0
    
    for u, v in graph.edges():
        if 'opinion' in graph.nodes[u] and 'opinion' in graph.nodes[v]:
            op_u = graph.nodes[u]['opinion']
            op_v = graph.nodes[v]['opinion']
            opinion_diff = abs(op_u - op_v)
            
            if opinion_diff > threshold:
                cross_camp_edges += 1
            total_edges += 1
    
    if total_edges == 0:
        return 0.0
    
    # 回音室指数 = 1 - 跨阵营边比例
    cross_camp_ratio = cross_camp_edges / total_edges
    echo_chamber_score = 1.0 - cross_camp_ratio
    
    return float(echo_chamber_score)


def compute_network_modularity(graph: nx.Graph, communities: Dict[int, int] = None) -> float:
    """
    计算网络模块度（需要社区划分）
    
    Args:
        graph: 图对象
        communities: 节点到社区的映射（如果为 None，则基于观点值划分）
    
    Returns:
        modularity: 模块度
    """
    if communities is None:
        # 基于观点值划分社区（简单二分：正负观点）
        communities = {}
        for node in graph.nodes():
            if 'opinion' in graph.nodes[node]:
                opinion = graph.nodes[node]['opinion']
                communities[node] = 0 if opinion < 0 else 1
            else:
                communities[node] = 0
    
    try:
        modularity = nx.community.modularity(graph, [set(n for n, c in communities.items() if c == comm) 
                                                      for comm in set(communities.values())])
        return float(modularity)
    except Exception:
        return 0.0


def compute_opinion_distribution(graph: nx.Graph, bins: int = 20) -> Dict:
    """
    计算观点分布统计
    
    Args:
        graph: 图对象
        bins: 直方图分箱数
    
    Returns:
        distribution: 包含分布信息的字典
    """
    opinions = []
    for node in graph.nodes():
        if 'opinion' in graph.nodes[node]:
            opinions.append(graph.nodes[node]['opinion'])
    
    if not opinions:
        return {'hist': [], 'bins': [], 'mean': 0.0, 'std': 0.0}
    
    opinions = np.array(opinions)
    hist, bin_edges = np.histogram(opinions, bins=bins, range=(-1.0, 1.0))
    
    return {
        'hist': hist.tolist(),
        'bins': bin_edges.tolist(),
        'mean': float(np.mean(opinions)),
        'std': float(np.std(opinions)),
        'min': float(np.min(opinions)),
        'max': float(np.max(opinions)),
        'median': float(np.median(opinions)),
    }


def compute_intervention_effectiveness(
    initial_graph: nx.Graph,
    final_graph: nx.Graph,
    method: str = "variance"
) -> Dict:
    """
    计算干预效果
    
    Args:
        initial_graph: 干预前的图
        final_graph: 干预后的图
        method: 计算方法（"variance", "weighted_disagreement", "echo_chamber"）
    
    Returns:
        effectiveness: 包含效果指标的字典
    """
    if method == "variance":
        initial_pol = compute_polarization_variance(initial_graph)
        final_pol = compute_polarization_variance(final_graph)
    elif method == "weighted_disagreement":
        initial_pol = compute_polarization_weighted_disagreement(initial_graph)
        final_pol = compute_polarization_weighted_disagreement(final_graph)
    elif method == "echo_chamber":
        initial_pol = compute_echo_chamber_score(initial_graph)
        final_pol = compute_echo_chamber_score(final_graph)
    else:
        raise ValueError(f"未知的方法: {method}")
    
    reduction = initial_pol - final_pol
    reduction_percent = (reduction / initial_pol * 100) if initial_pol > 0 else 0.0
    
    return {
        'initial_polarization': float(initial_pol),
        'final_polarization': float(final_pol),
        'reduction': float(reduction),
        'reduction_percent': float(reduction_percent),
        'method': method,
    }
