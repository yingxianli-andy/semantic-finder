"""
Data loading and synthetic graph generation utilities for opinion dynamics.

由实习生A（哈工爷）实现。

接口规范参考 `doc/接口对接文档.md` 与 `doc/开发文档.md`：

- load_graph(dataset_name: str, opinion_init: str = "bimodal") -> nx.Graph
- generate_synthetic_graph(n_nodes: int = 50, graph_type: str = "BA") -> nx.Graph
"""

from __future__ import annotations

import os
import tarfile
from typing import Literal

import networkx as nx
import numpy as np


OpinionInitType = Literal["bimodal", "random", "uniform"]


def _ensure_integer_node_labels(graph: nx.Graph) -> nx.Graph:
    """
    Ensure node labels are consecutive integers starting from 0.

    Many SNAP datasets use arbitrary integer IDs. For RL we require
    0..N-1 indexing, so we relabel when necessary.
    """
    nodes = list(graph.nodes())
    # Fast path: already 0..N-1
    if all(isinstance(n, int) for n in nodes):
        max_id = max(nodes) if nodes else -1
        if set(nodes) == set(range(max_id + 1)):
            return graph

    mapping = {old: i for i, old in enumerate(nodes)}
    return nx.relabel_nodes(graph, mapping)


def _init_opinions(graph: nx.Graph, mode: OpinionInitType = "bimodal") -> None:
    """
    Initialize node 'opinion' attribute in-place.

    - bimodal: half near -0.8, half near +0.8
    - random: uniform in [-1, 1]
    - uniform: all zeros (near neutral) for now
    """
    num_nodes = graph.number_of_nodes()
    nodes = list(graph.nodes())

    if mode == "bimodal":
        # 随机打乱节点，然后前一半设为 -0.8，后一半设为 +0.8（可加少量噪声）
        rng = np.random.default_rng()
        rng.shuffle(nodes)
        half = num_nodes // 2
        for idx, n in enumerate(nodes):
            if idx < half:
                opinion = -0.8 + rng.normal(0.0, 0.05)
            else:
                opinion = 0.8 + rng.normal(0.0, 0.05)
            graph.nodes[n]["opinion"] = float(np.clip(opinion, -1.0, 1.0))
    elif mode == "random":
        rng = np.random.default_rng()
        opinions = rng.uniform(-1.0, 1.0, size=num_nodes)
        for n, op in zip(nodes, opinions):
            graph.nodes[n]["opinion"] = float(op)
    elif mode == "uniform":
        # 所有人接近中立 0，可以后续按需要调整为其他分布
        for n in nodes:
            graph.nodes[n]["opinion"] = 0.0
    else:
        raise ValueError(f"Unknown opinion_init mode: {mode}")


def load_graph(dataset_name: str, opinion_init: OpinionInitType = "bimodal") -> nx.Graph:
    """
    加载图数据并初始化观点值。

    Args:
        dataset_name: 数据集名称（如 "ego-Twitter"）。这里约定从
            `data/raw/{dataset_name}` 或 `data/raw/{dataset_name}.edgelist`
            / `.txt` 中读取。
        opinion_init: 观点初始化方式：
            - "bimodal": 双峰分布（一半节点 -0.8，一半节点 0.8）
            - "random": 随机分布（-1 到 1 之间均匀分布）
            - "uniform": 均匀分布（当前实现为 0）

    Returns:
        graph: NetworkX 图对象，每个节点具有:
            - int 型索引（0..N-1）
            - 'opinion' 属性（float, 范围 [-1, 1]）
    """
    # 尝试多种常见路径，方便后续落地实际数据时只需把文件放对位置
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    
    # 首先检查是否是tar.gz文件（如twitter.tar.gz）
    tar_path = os.path.join(base_dir, "twitter.tar.gz")
    if dataset_name == "ego-Twitter" and os.path.isfile(tar_path):
        # 解压tar.gz文件（如果还没解压）
        extracted_dir = os.path.join(base_dir, "twitter")
        if not os.path.isdir(extracted_dir):
            print(f"正在解压 {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(base_dir)
            print("解压完成")
        
        # 查找解压后的边文件（twitter_combined.txt）
        # 可能在twitter子目录，也可能直接在raw目录
        candidates = [
            os.path.join(extracted_dir, "twitter_combined.txt"),
            os.path.join(base_dir, "twitter_combined.txt"),
            # 也可能在解压后的其他位置
        ]
        # 如果extracted_dir存在，搜索其下的所有.txt文件
        if os.path.isdir(extracted_dir):
            for root, dirs, files in os.walk(extracted_dir):
                for file in files:
                    if file == "twitter_combined.txt" or file.endswith("_combined.txt"):
                        candidates.insert(0, os.path.join(root, file))
        
        edge_path = None
        for path in candidates:
            if os.path.isfile(path):
                edge_path = path
                break
    else:
        # 尝试其他常见路径
        candidates = [
            os.path.join(base_dir, dataset_name),
            os.path.join(base_dir, f"{dataset_name}.edgelist"),
            os.path.join(base_dir, f"{dataset_name}.txt"),
            os.path.join(base_dir, "twitter_combined.txt"),  # 解压后的文件
        ]

        edge_path = None
        for path in candidates:
            if os.path.isfile(path):
                edge_path = path
                break

    if edge_path is None or not os.path.isfile(edge_path):
        raise FileNotFoundError(
            f"未找到数据集 {dataset_name!r} 的边文件，"
            f"请将 SNAP 等数据放在 data/raw/ 下（支持 .edgelist、.txt 或 .tar.gz）。"
        )

    # 这里假设是简单的 edge list，每行两个节点 ID（可以是空格或制表符分隔）
    graph = nx.read_edgelist(edge_path, nodetype=int)

    # 统一为 0..N-1 整数索引
    graph = _ensure_integer_node_labels(graph)

    # 初始化观点
    _init_opinions(graph, mode=opinion_init)

    return graph


def sample_subgraph(graph: nx.Graph, max_nodes: int = 5000, method: str = "random") -> nx.Graph:
    """
    从大图中采样子图（用于快速测试）。
    
    Args:
        graph: 原始图
        max_nodes: 最大节点数
        method: 采样方法
            - "random": 随机采样节点
            - "largest_component": 最大连通分量
            - "degree": 按度数采样（保留高度数节点）
    
    Returns:
        subgraph: 采样后的子图
    """
    n_original = graph.number_of_nodes()
    
    if n_original <= max_nodes:
        return graph.copy()
    
    if method == "random":
        # 随机采样节点
        nodes_to_keep = np.random.choice(
            list(graph.nodes()),
            size=max_nodes,
            replace=False
        )
        subgraph = graph.subgraph(nodes_to_keep).copy()
    elif method == "largest_component":
        # 取最大连通分量，如果还是太大则随机采样
        components = list(nx.connected_components(graph))
        largest_component = max(components, key=len)
        if len(largest_component) <= max_nodes:
            subgraph = graph.subgraph(largest_component).copy()
        else:
            nodes_to_keep = np.random.choice(
                list(largest_component),
                size=max_nodes,
                replace=False
            )
            subgraph = graph.subgraph(nodes_to_keep).copy()
    elif method == "degree":
        # 按度数排序，保留高度数节点
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        nodes_to_keep = [node for node, _ in sorted_nodes[:max_nodes]]
        subgraph = graph.subgraph(nodes_to_keep).copy()
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # 重新标记节点为 0..N-1
    subgraph = _ensure_integer_node_labels(subgraph)
    
    print(f"  子图采样: {n_original} -> {subgraph.number_of_nodes()} 节点 "
          f"({subgraph.number_of_nodes()/n_original*100:.1f}%)")
    
    return subgraph


def generate_synthetic_graph(n_nodes: int = 50, graph_type: str = "BA") -> nx.Graph:
    """
    生成合成图（用于离线训练）。

    Args:
        n_nodes: 节点数量（建议 30-50）
        graph_type: 图类型
            - "BA": Barabasi-Albert 无标度网络

    Returns:
        graph: NetworkX 图对象，每个节点具有:
            - int 型索引（0..N-1）
            - 'opinion' 属性（float, 范围 [-1, 1]）
    """
    if graph_type.upper() == "BA":
        # m 取 3 比较常见，也避免太稀疏
        m = min(3, max(1, n_nodes - 1))
        graph = nx.barabasi_albert_graph(n=n_nodes, m=m)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type!r}")

    graph = _ensure_integer_node_labels(graph)
    _init_opinions(graph, mode="bimodal")
    return graph


__all__ = ["load_graph", "generate_synthetic_graph", "sample_subgraph"]

