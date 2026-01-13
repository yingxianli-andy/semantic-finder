"""
Opinion dynamics models (DeGroot / Friedkin-Johnsen).

由实习生A（哈工爷）实现，用于在图上更新节点的观点值。

核心接口:
- update_opinions(graph, model="degroot", steps=3, alpha=0.5) -> nx.Graph

性能要求:
- 使用 scipy.sparse 构建和运算稀疏矩阵
- 避免双重 for 循环
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境中可能没有 torch
    TORCH_AVAILABLE = False


DynamicsModelType = Literal["degroot", "friedkin_johnsen"]


def _build_normalized_adjacency(graph: nx.Graph) -> sp.csr_matrix:
    """
    构建行归一化的稀疏邻接矩阵 A_norm.

    A_norm[i, j] = 1 / deg(i) if (i, j) in E, else 0.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    # 假设节点已经是 0..N-1
    rows = []
    cols = []
    data = []

    for i in range(n):
        neighbors = list(graph.neighbors(i))
        deg = len(neighbors)
        if deg == 0:
            # 孤立点：保持自环，观点不变
            rows.append(i)
            cols.append(i)
            data.append(1.0)
        else:
            weight = 1.0 / deg
            for j in neighbors:
                rows.append(i)
                cols.append(j)
                data.append(weight)

    mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return mat.tocsr()


def _build_normalized_adjacency_torch(
    graph: nx.Graph, device: Optional[str | torch.device] = None
) -> "torch.sparse.FloatTensor":
    """
    使用 PyTorch 构建行归一化的稀疏邻接矩阵 A_norm（用于 GPU 加速）。

    A_norm[i, j] = 1 / deg(i) if (i, j) in E, else 0.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 不可用，无法使用 GPU 加速的观点动力学。")

    # 统一处理 device 参数（可能是字符串或 torch.device 对象）
    if device is None:
        device = "cpu"
    if isinstance(device, torch.device):
        device = str(device)
    
    dev = torch.device(device) if isinstance(device, str) else device
    
    n = graph.number_of_nodes()
    if n == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=dev)
        values = torch.empty((0,), dtype=torch.float32, device=dev)
        return torch.sparse_coo_tensor(indices, values, (0, 0), device=dev)

    # 使用更高效的方法：直接从图的边列表构建
    # 先收集所有边，然后批量处理
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # 假设节点已经是 0..N-1
    # 优化：先计算所有节点的度（避免在循环中重复计算）
    degrees = [graph.degree(i) for i in range(n)]
    
    for i in range(n):
        deg = degrees[i]
        if deg == 0:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
        else:
            weight = 1.0 / deg
            for j in graph.neighbors(i):
                rows.append(i)
                cols.append(j)
                data.append(weight)

    # 直接在 GPU 上构建 tensor（避免在 CPU 上先构建 Python list 再转换）
    # 对于超大数据集，考虑分批构建或使用更高效的方法
    indices = torch.tensor([rows, cols], dtype=torch.long, device=dev)
    values = torch.tensor(data, dtype=torch.float32, device=dev)
    mat = torch.sparse_coo_tensor(indices, values, (n, n), device=dev)
    return mat.coalesce()


def _update_opinions_torch(
    graph: nx.Graph,
    model: DynamicsModelType,
    steps: int,
    alpha: float,
    device: str = "cuda",
    intervention_dict: Optional[Dict[int, float]] = None,
    cached_adj_matrix: Optional["torch.sparse.FloatTensor"] = None,
    intervention_beta: float = 0.5,
) -> nx.Graph:
    """
    使用 PyTorch（可在 GPU 上）更新观点值。
    
    Args:
        cached_adj_matrix: 缓存的邻接矩阵（如果提供，跳过构建步骤）
    """
    if steps <= 0:
        return graph

    if not TORCH_AVAILABLE:
        # 回退到 NumPy / SciPy 实现
        return update_opinions(graph, model=model, steps=steps, alpha=alpha)

    n = graph.number_of_nodes()
    if n == 0:
        return graph

    dev_str = str(device) if isinstance(device, torch.device) else device
    dev = torch.device(dev_str) if isinstance(dev_str, str) else device

    # 性能优化：批量读取观点值（避免逐个访问图节点）
    # 使用列表推导式一次性读取所有观点值，比循环快
    opinions_list = [float(graph.nodes[i].get("opinion", 0.0)) for i in range(n)]
    opinions = torch.tensor(opinions_list, dtype=torch.float32, device=dev)

    # 对 FJ 模型，需要记住初始观点 x(0)
    x0 = opinions.clone()

    # 性能优化：如果提供了缓存的邻接矩阵，直接使用；否则构建并返回
    if cached_adj_matrix is not None:
        a_norm = cached_adj_matrix
    else:
        a_norm = _build_normalized_adjacency_torch(graph, device=dev_str)

    x = opinions
    model = model.lower()

    # 构建干预向量（如果有）
    if intervention_dict is not None:
        intervention_vector = torch.zeros(n, dtype=torch.float32, device=dev)
        for node_id, weight in intervention_dict.items():
            if 0 <= node_id < n:
                intervention_vector[node_id] = weight
    else:
        intervention_vector = None

    for _ in range(steps):
        if model == "degroot":
            if intervention_vector is not None:
                # 新公式：x(t+1) = alpha * x(t) + (1-alpha) * (A_norm * x(t) + beta * Intervention)
                # 引入强度系数beta，避免干预过强导致数值溢出
                neighbor_influence = torch.sparse.mm(a_norm, x.unsqueeze(1)).squeeze(1)
                total_influence = neighbor_influence + intervention_beta * intervention_vector
                x = alpha * x + (1.0 - alpha) * total_influence
            else:
                # 原公式：x(t+1) = A_norm * x(t)
                x = torch.sparse.mm(a_norm, x.unsqueeze(1)).squeeze(1)
        elif model == "friedkin_johnsen":
            # x(t+1) = alpha * A_norm * x(t) + (1-alpha) * x(0)
            # 注意：FJ模型暂时不支持Intervention，保持原逻辑
            ax = torch.sparse.mm(a_norm, x.unsqueeze(1)).squeeze(1)
            x = alpha * ax + (1.0 - alpha) * x0
        else:
            raise ValueError(f"Unknown dynamics model: {model}")

    # 性能优化：直接在GPU上clamp，然后批量写回（减少CPU-GPU传输）
    # 性能优化：在GPU上clamp，然后批量传输和写回
    x = torch.clamp(x, -1.0, 1.0)
    
    # 批量传输：一次性传输所有值到CPU
    x_cpu = x.detach().cpu().numpy()
    
    # 批量写回：使用字典更新，比逐个赋值快
    # 但NetworkX不支持批量更新，所以还是需要循环
    # 优化：使用enumerate和列表，减少属性查找
    for i, val in enumerate(x_cpu):
        graph.nodes[i]["opinion"] = float(val)

    return graph


def update_opinions(
    graph: nx.Graph,
    model: DynamicsModelType = "degroot",
    steps: int = 3,
    alpha: float = 0.5,
    device: str = "cpu",
    intervention_dict: Optional[Dict[int, float]] = None,
    cached_adj_matrix: Optional[any] = None,
    cached_adj_matrix_torch: Optional["torch.sparse.FloatTensor"] = None,
    intervention_beta: float = 0.5,
) -> nx.Graph:
    """
    更新图中所有节点的观点值（使用观点动力学模型）。

    Args:
        graph: 输入图（会被修改）
        model: 模型类型（"degroot" 或 "friedkin_johnsen"）
        steps: 演化步数（建议 3-5 步）
        alpha: 顽固度参数（0-1）
            - 对于 DeGroot 变体：alpha 表示对当前观点的坚持程度
            - 对于 FJ 模型：alpha 表示对初始观点的坚持程度
        device: 设备（"cpu" 或 "cuda"），当为 "cuda" 且 PyTorch 可用时，
            使用 GPU 加速的实现。
        intervention_dict: 干预字典 {节点ID: 干预权重}，用于新公式
            - 如果提供，DeGroot 变体将使用新公式：
              x(t+1) = alpha * x(t) + (1-alpha) * (A_norm * x(t) + beta * Intervention)
            - 如果为 None，使用原公式（向后兼容）
        intervention_beta: 干预强度系数（默认0.5）
            - 用于控制干预项的强度，避免数值溢出
            - beta=0.5 表示干预强度为原始权重的一半

    Returns:
        updated_graph: 更新后的图（当前实现直接在原图上就地修改并返回）
    """
    # 如果可以使用 GPU，则优先使用 torch 实现
    if device != "cpu" and TORCH_AVAILABLE:
        try:
            return _update_opinions_torch(
                graph=graph,
                model=model,
                steps=steps,
                alpha=alpha,
                device=device,
                intervention_dict=intervention_dict,
                cached_adj_matrix=cached_adj_matrix_torch,
                intervention_beta=intervention_beta,
            )
        except Exception as e:
            # 出现任何问题时回退到原始 CPU 实现，保证鲁棒性
            print(f"--- 警告: GPU 动力学计算失败，已自动回退到 CPU。错误: {e} ---")
            import traceback
            traceback.print_exc()
            print("----------------------------------------------------------")
            pass

    if steps <= 0:
        return graph

    n = graph.number_of_nodes()
    if n == 0:
        return graph

    # 读取当前观点向量 x(t)
    opinions = np.array(
        [float(graph.nodes[i].get("opinion", 0.0)) for i in range(n)],
        dtype=np.float32,
    )

    # 对 FJ 模型，需要记住初始观点 x(0)
    x0 = opinions.copy()

    # 性能优化：如果提供了缓存的邻接矩阵，直接使用；否则构建
    if cached_adj_matrix is not None:
        a_norm = cached_adj_matrix
    else:
        a_norm = _build_normalized_adjacency(graph)

    x = opinions
    model = model.lower()

    # 构建干预向量（如果有）
    if intervention_dict is not None:
        intervention_vector = np.zeros(n, dtype=np.float32)
        for node_id, weight in intervention_dict.items():
            if 0 <= node_id < n:
                intervention_vector[node_id] = weight
    else:
        intervention_vector = None

    for _ in range(steps):
        if model == "degroot":
            if intervention_vector is not None:
                # 新公式：x(t+1) = alpha * x(t) + (1-alpha) * (A_norm * x(t) + beta * Intervention)
                # 引入强度系数beta，避免干预过强导致数值溢出
                neighbor_influence = a_norm @ x
                total_influence = neighbor_influence + intervention_beta * intervention_vector
                x = alpha * x + (1.0 - alpha) * total_influence
            else:
                # 原公式：x(t+1) = A_norm * x(t)
                x = a_norm @ x
        elif model == "friedkin_johnsen":
            # x(t+1) = alpha * A_norm * x(t) + (1-alpha) * x(0)
            # 注意：FJ模型暂时不支持Intervention，保持原逻辑
            x = alpha * (a_norm @ x) + (1.0 - alpha) * x0
        else:
            raise ValueError(f"Unknown dynamics model: {model}")

    # 写回图的节点属性
    for i in range(n):
        graph.nodes[i]["opinion"] = float(np.clip(x[i], -1.0, 1.0))

    return graph


__all__ = ["update_opinions"]

