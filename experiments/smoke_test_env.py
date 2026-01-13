"""
Minimal smoke test for OpinionDynamicsEnv.

由实习生A（哈工爷）使用：
- 生成一张合成图
- 初始化环境
- 随机选节点 & 随机权重
- 跑完一个 episode，打印 reward
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.agent.reward import compute_reward
from src.environment.data_loader import generate_synthetic_graph
from src.environment.opinion_env import OpinionDynamicsEnv


def main() -> None:
    graph = generate_synthetic_graph(n_nodes=20, graph_type="BA")

    env = OpinionDynamicsEnv(
        graph=graph,
        budget=5,
        reward_fn=lambda s, ns: compute_reward(s, ns, method="variance"),
        dynamics_model="degroot",
        dynamics_steps=3,
    )

    state = env.reset()
    done = False

    step = 0
    while not done:
        n_nodes = state.number_of_nodes()
        action_node = np.random.randint(0, n_nodes)
        intervention_weight = float(np.random.uniform(0.3, 0.7))

        next_state, reward, done, info = env.step(action_node, intervention_weight)
        print(f"step={step}, action_node={action_node}, weight={intervention_weight:.3f}, reward={reward:.4f}")

        state = next_state
        step += 1


if __name__ == "__main__":
    main()

