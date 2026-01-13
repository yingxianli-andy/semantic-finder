"""
LLM 策略模块：实现不同的干预权重生成策略。

用于IJCAI实验四（策略多样性）。
"""

import random
from typing import Optional

import networkx as nx
import numpy as np


def get_weight_conservative(
    node_id: int, graph: nx.Graph, subgraph: Optional[nx.Graph] = None
) -> float:
    """
    温和派策略：总是给低权重，慢慢感化。

    Args:
        node_id: 节点ID
        graph: 完整图对象
        subgraph: 节点周围的子图（可选，未使用）

    Returns:
        weight: 干预权重（0.3 到 0.4 之间）
    """
    return 0.3 + 0.1 * random.random()


def get_weight_aggressive(
    node_id: int, graph: nx.Graph, subgraph: Optional[nx.Graph] = None
) -> float:
    """
    激进派策略：总是给高权重，强力洗脑。

    Args:
        node_id: 节点ID
        graph: 完整图对象
        subgraph: 节点周围的子图（可选，未使用）

    Returns:
        weight: 干预权重（0.8 到 1.0 之间）
    """
    return 0.8 + 0.2 * random.random()


def get_weight_adaptive(
    node_id: int, graph: nx.Graph, subgraph: Optional[nx.Graph] = None
) -> float:
    """
    自适应策略：根据周围邻居的方差动态调整权重。

    逻辑：
    - 如果邻居方差大（吵得凶）：给高权重（1.0），强力介入
    - 如果邻居方差小（很和平）：给低权重（0.2），轻轻推一下

    Args:
        node_id: 节点ID
        graph: 完整图对象
        subgraph: 节点周围的子图（如果为None，从完整图中提取）

    Returns:
        weight: 干预权重（0.2 到 1.0 之间）
    """
    # 获取节点邻居
    if subgraph is not None:
        neighbors = list(subgraph.neighbors(node_id))
        if len(neighbors) == 0:
            # 如果没有邻居，使用默认权重
            return 0.5
        # 从子图中获取邻居观点
        neighbor_opinions = [
            subgraph.nodes[n].get("opinion", 0.0) for n in neighbors
        ]
    else:
        neighbors = list(graph.neighbors(node_id))
        if len(neighbors) == 0:
            # 如果没有邻居，使用默认权重
            return 0.5
        # 从完整图中获取邻居观点
        neighbor_opinions = [
            graph.nodes[n].get("opinion", 0.0) for n in neighbors
        ]

    # 计算邻居观点值的方差
    if len(neighbor_opinions) == 0:
        return 0.5

    variance = np.var(neighbor_opinions)

    # 根据方差决定权重
    var_high_threshold = 0.5  # 可调参数
    var_low_threshold = 0.1

    if variance > var_high_threshold:
        # 方差大，给高权重
        return 1.0
    elif variance < var_low_threshold:
        # 方差小，给低权重
        return 0.2
    else:
        # 中等方差，给中等权重（线性插值）
        normalized_var = (variance - var_low_threshold) / (
            var_high_threshold - var_low_threshold
        )
        return 0.2 + 0.8 * normalized_var
