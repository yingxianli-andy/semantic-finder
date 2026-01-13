"""
图处理工具函数

提供图数据的预处理、特征提取、统计信息等功能。
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple


def get_graph_statistics(graph: nx.Graph) -> Dict:
    """
    获取图的基本统计信息
    
    Args:
        graph: NetworkX 图对象
    
    Returns:
        stats: 包含统计信息的字典
    """
    stats = {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph) if isinstance(graph, nx.Graph) else False,
    }
    
    # 度统计
    degrees = dict(graph.degree())
    if degrees:
        stats['avg_degree'] = np.mean(list(degrees.values()))
        stats['max_degree'] = max(degrees.values())
        stats['min_degree'] = min(degrees.values())
    
    # 观点统计（如果有）
    if all('opinion' in graph.nodes[node] for node in graph.nodes()):
        opinions = [graph.nodes[node]['opinion'] for node in graph.nodes()]
        stats['opinion_mean'] = np.mean(opinions)
        stats['opinion_std'] = np.std(opinions)
        stats['opinion_min'] = np.min(opinions)
        stats['opinion_max'] = np.max(opinions)
    
    return stats


def normalize_graph(graph: nx.Graph) -> nx.Graph:
    """
    归一化图（确保节点索引从 0 开始连续）
    
    Args:
        graph: 输入图
    
    Returns:
        normalized_graph: 归一化后的图
    """
    # 如果节点索引已经是连续的 0..N-1，直接返回
    nodes = sorted(graph.nodes())
    if nodes == list(range(len(nodes))):
        return graph.copy()
    
    # 否则重新映射节点索引
    mapping = {old_node: new_node for new_node, old_node in enumerate(nodes)}
    normalized_graph = nx.relabel_nodes(graph, mapping)
    
    return normalized_graph


def extract_node_features(graph: nx.Graph, include_opinion: bool = True) -> np.ndarray:
    """
    提取节点特征
    
    Args:
        graph: 图对象
        include_opinion: 是否包含观点值
    
    Returns:
        features: 节点特征矩阵 [N, feature_dim]
    """
    n_nodes = graph.number_of_nodes()
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1.0
    
    features = []
    for node in sorted(graph.nodes()):
        node_features = []
        
        # 归一化度数
        normalized_degree = degrees.get(node, 0) / max_degree
        node_features.append(normalized_degree)
        
        # 观点值（如果有）
        if include_opinion and 'opinion' in graph.nodes[node]:
            node_features.append(graph.nodes[node]['opinion'])
        
        features.append(node_features)
    
    return np.array(features, dtype=np.float32)


def get_neighbor_opinions(graph: nx.Graph, node: int) -> List[float]:
    """
    获取节点的邻居观点值
    
    Args:
        graph: 图对象
        node: 节点索引
    
    Returns:
        neighbor_opinions: 邻居观点值列表
    """
    neighbors = list(graph.neighbors(node))
    opinions = []
    for neighbor in neighbors:
        if 'opinion' in graph.nodes[neighbor]:
            opinions.append(graph.nodes[neighbor]['opinion'])
    return opinions


def compute_opinion_similarity(graph: nx.Graph, node1: int, node2: int) -> float:
    """
    计算两个节点观点的相似度
    
    Args:
        graph: 图对象
        node1: 节点1索引
        node2: 节点2索引
    
    Returns:
        similarity: 相似度（0-1之间，1表示完全相同）
    """
    if 'opinion' not in graph.nodes[node1] or 'opinion' not in graph.nodes[node2]:
        return 0.0
    
    op1 = graph.nodes[node1]['opinion']
    op2 = graph.nodes[node2]['opinion']
    
    # 使用 1 - 归一化距离作为相似度
    distance = abs(op1 - op2) / 2.0  # 归一化到 [0, 1]
    similarity = 1.0 - distance
    
    return float(similarity)


def get_connected_components(graph: nx.Graph) -> List[nx.Graph]:
    """
    获取图的连通分量
    
    Args:
        graph: 图对象
    
    Returns:
        components: 连通分量列表
    """
    if isinstance(graph, nx.DiGraph):
        components = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
    else:
        components = [graph.subgraph(c) for c in nx.connected_components(graph)]
    
    return components


def filter_nodes_by_opinion(
    graph: nx.Graph,
    min_opinion: float = -1.0,
    max_opinion: float = 1.0
) -> List[int]:
    """
    根据观点值过滤节点
    
    Args:
        graph: 图对象
        min_opinion: 最小观点值
        max_opinion: 最大观点值
    
    Returns:
        filtered_nodes: 符合条件的节点索引列表
    """
    filtered = []
    for node in graph.nodes():
        if 'opinion' in graph.nodes[node]:
            opinion = graph.nodes[node]['opinion']
            if min_opinion <= opinion <= max_opinion:
                filtered.append(node)
    return filtered
