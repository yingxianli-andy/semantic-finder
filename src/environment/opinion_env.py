"""
Gymnasium environment for opinion dynamics.

由实习生A（哈工爷）实现，对外暴露给：
- 安迪：用于 RL 训练 / 测试
- 上交爷：通过训练/测试脚本间接调用

核心接口:
- reset() -> nx.Graph
- step(action_node: int, intervention_weight: float) -> (next_state, reward, done, info)

接口规范详见 `doc/接口对接文档.md` 与 `doc/开发文档.md`。
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np

from .dynamics import update_opinions, TORCH_AVAILABLE


class OpinionDynamicsEnv(gym.Env):
    """
    观点动力学环境（Gymnasium 接口）。

    状态: NetworkX Graph（节点具有 'opinion' 属性）
    动作: 选择一个节点索引（由 RL Agent 决定），并由外部提供干预权重。
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        graph: nx.Graph,
        budget: int,
        reward_fn: Callable[[nx.Graph, nx.Graph], float],
        dynamics_model: str = "degroot",
        dynamics_steps: int = 3,
        fj_alpha: float = 0.5,
        device: str = "cpu",
    ):
        """
        Args:
            graph: 初始图对象（NetworkX Graph）
            budget: 干预预算（每个 episode 最多选择 K 个节点）
            reward_fn: 奖励函数（由安迪提供）
                签名: reward_fn(state: nx.Graph, next_state: nx.Graph) -> float
            dynamics_model: 观点动力学模型类型（"degroot" 或 "friedkin_johnsen"）
            dynamics_steps: 每次 step 中动力学演化的步数
            fj_alpha: Friedkin-Johnsen 模型参数
            device: 设备（"cpu" 或 "cuda"），用于控制是否在 GPU 上加速动力学计算
        """
        super().__init__()

        if budget <= 0:
            raise ValueError("budget must be positive")

        self._original_graph = graph.copy()
        self.graph = graph.copy()

        self.budget = int(budget)
        self.reward_fn = reward_fn

        self.dynamics_model = dynamics_model
        self.dynamics_steps = int(dynamics_steps)
        self.fj_alpha = float(fj_alpha)

        # 保存设备信息，供 dynamics 层选择 CPU / GPU 实现
        self.device = device

        self.step_count = 0
        
        # 性能优化：缓存邻接矩阵（图结构不变，避免重复构建）
        # 在初始化时构建一次，后续复用
        from .dynamics import _build_normalized_adjacency, _build_normalized_adjacency_torch
        self._cached_adj_matrix = _build_normalized_adjacency(self.graph)
        if device != "cpu" and TORCH_AVAILABLE:
            try:
                self._cached_adj_matrix_torch = _build_normalized_adjacency_torch(self.graph, device=device)
            except:
                self._cached_adj_matrix_torch = None
        else:
            self._cached_adj_matrix_torch = None

        # Gym action/observation space（主要为了兼容性，真正状态直接用 Graph）
        n_nodes = self.graph.number_of_nodes()
        self.action_space = gym.spaces.Discrete(n_nodes)

        # 观测空间这里简单起见用一个 Box，但训练代码里通常直接用 Graph
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_nodes,),
            dtype=np.float32,
        )

    # ---- Gym API ----

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        """
        重置环境，返回初始状态（NetworkX Graph）。
        """
        super().reset(seed=seed)
        self.graph = self._original_graph.copy()
        self.step_count = 0
        return self.graph

    def step(
        self,
        action_node: int,
        intervention_weight: float,
        alpha: Optional[float] = None,
        use_new_formula: bool = False,
    ) -> Tuple[nx.Graph, float, bool, Dict]:
        """
        执行一步动作。

        Args:
            action_node: 被选中的节点索引（int, 0..N-1）
            intervention_weight: 干预权重（float, 0.0..1.0）
            alpha: 顽固度参数（如果提供，覆盖初始化时的值）
            use_new_formula: 是否使用新公式（带Intervention项）

        Returns:
            next_state: 下一状态（NetworkX Graph）
            reward: 奖励值（float）
            done: 是否结束（bool）
            info: 额外信息（dict）
        """
        if action_node < 0 or action_node >= self.graph.number_of_nodes():
            raise ValueError(f"action_node {action_node} 越界")

        w = float(np.clip(intervention_weight, 0.0, 1.0))

        # 性能优化：批量读取观点值（避免逐个访问图节点）
        n_nodes = self.graph.number_of_nodes()
        # 使用列表推导式一次性读取，比循环快
        prev_opinions = np.array([
            float(self.graph.nodes[i].get("opinion", 0.0))
            for i in range(n_nodes)
        ], dtype=np.float32)

        # 确定使用的alpha值
        current_alpha = alpha if alpha is not None else self.fj_alpha

        # 1. 干预：根据是否使用新公式决定处理方式
        if use_new_formula:
            # 新公式：干预项在动力学公式内部处理
            # 构建intervention_dict，传递给update_opinions
            intervention_dict = {action_node: w}
        else:
            # 旧方式：提前修改节点观点
            x_old = float(self.graph.nodes[action_node].get("opinion", 0.0))
            x_new = x_old * (1.0 - w)  # + 0 * w
            self.graph.nodes[action_node]["opinion"] = float(np.clip(x_new, -1.0, 1.0))
            intervention_dict = None

        # 2. 运行观点动力学模型若干步
        import time
        dynamics_start = time.time()
        # 性能优化：传递缓存的邻接矩阵，避免重复构建
        self.graph = update_opinions(
            self.graph,
            model=self.dynamics_model,
            steps=self.dynamics_steps,
            alpha=current_alpha,
            device=self.device,
            intervention_dict=intervention_dict,
            cached_adj_matrix=self._cached_adj_matrix,
            cached_adj_matrix_torch=self._cached_adj_matrix_torch,
        )
        dynamics_time = time.time() - dynamics_start

        # 3. 奖励计算：优化 - 只复制opinion值，不复制整个图
        # compute_polarization_variance只需要opinion值，不需要边信息
        copy_start = time.time()
        # 只复制opinion数组（numpy数组，非常快）
        prev_opinions_copy = prev_opinions.copy()  # 已经是numpy数组，copy很快
        copy_time = time.time() - copy_start
        
        reward_start = time.time()
        # 性能优化：批量读取当前opinions并直接计算奖励
        current_opinions = np.array([
            float(self.graph.nodes[i].get("opinion", 0.0))
            for i in range(n_nodes)
        ], dtype=np.float32)
        
        # 直接计算极化度（避免图遍历和创建）
        from src.agent.reward import compute_polarization_variance
        prev_polarization = compute_polarization_variance(opinions=prev_opinions_copy)
        current_polarization = compute_polarization_variance(opinions=current_opinions)
        reward = float(prev_polarization - current_polarization)
        reward_time = time.time() - reward_start
        
        # 激进优化：进一步减少日志输出频率（每100个step输出一次，减少I/O开销）
        if self.step_count % 100 == 0:
            print(f"      [env.step #{self.step_count}] 动力学: {dynamics_time:.3f}s | "
                  f"图复制: {copy_time:.3f}s | 奖励计算: {reward_time:.3f}s | "
                  f"总计: {dynamics_time+copy_time+reward_time:.3f}s", flush=True)

        # 4. 更新步数 & 是否结束
        self.step_count += 1
        done = self.step_count >= self.budget

        info: Dict = {
            "step_count": self.step_count,
        }

        return self.graph, reward, done, info


__all__ = ["OpinionDynamicsEnv"]

