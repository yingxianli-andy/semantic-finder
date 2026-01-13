"""
环境模块演示脚本

这个脚本演示了如何使用 `OpinionDynamicsEnv` 环境，包括：
1. 生成合成图
2. 初始化环境
3. 运行一个 episode（使用随机动作）
4. 可视化结果
"""

import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environment.data_loader import generate_synthetic_graph
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward
from src.utils.graph_utils import get_graph_statistics

# 1. 生成合成图
print("1. 生成合成图...")
N_NODES = 50
GRAPH_TYPE = "BA"
initial_graph = generate_synthetic_graph(n_nodes=N_NODES, graph_type=GRAPH_TYPE)

# 打印图的基本信息
stats = get_graph_statistics(initial_graph)
print("图信息:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  - {key}: {value:.4f}")
    else:
        print(f"  - {key}: {value}")

# 2. 初始化环境
print("\n2. 初始化环境...")
BUDGET = 5
REWARD_METHOD = "variance"

def reward_fn(state, next_state):
    return compute_reward(state, next_state, method=REWARD_METHOD)

env = OpinionDynamicsEnv(
    graph=initial_graph,
    budget=BUDGET,
    reward_fn=reward_fn,
    dynamics_model="degroot",
    dynamics_steps=3
)
print(f"环境初始化完成. 预算: {BUDGET}, 奖励方法: {REWARD_METHOD}")

# 3. 运行一个 episode
print("\n3. 运行 episode...")
state = env.reset()
done = False
total_reward = 0.0
selected_nodes = []
step_count = 0

while not done and step_count < BUDGET * 2:  # 防止无限循环
    # 随机选择一个未被选过的节点
    available_nodes = [n for n in state.nodes() if n not in selected_nodes]
    if not available_nodes:
        print("没有更多可用节点，提前结束。")
        break
        
    action_node = np.random.choice(available_nodes)
    selected_nodes.append(action_node)
    
    # 随机生成干预权重
    intervention_weight = np.random.uniform(0.3, 0.7)
    
    # 执行一步
    next_state, reward, done, info = env.step(action_node, intervention_weight)
    
    total_reward += reward
    state = next_state
    step_count += 1
    
    print(f"Step {step_count:2d}: 选择节点 {action_node:3d}, "
          f"权重={intervention_weight:.2f}, 奖励={reward:+.4f}, "
          f"累计奖励={total_reward:+.4f}")

print(f"\nEpisode 结束. 总奖励: {total_reward:.4f}, 总步数: {step_count}")

# 4. 可视化结果
print("\n4. 可视化结果...")
initial_opinions = [initial_graph.nodes[n]['opinion'] for n in initial_graph.nodes()]
final_opinions = [state.nodes[n]['opinion'] for n in state.nodes()]

initial_variance = np.var(initial_opinions)
final_variance = np.var(final_opinions)

# 绘制观点分布
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(initial_opinions, bins=20, range=(-1, 1), color='red', alpha=0.7)
plt.title(f'初始观点分布\n方差: {initial_variance:.4f}')
plt.xlabel('观点值')
plt.ylabel('节点数量')
plt.xlim(-1, 1)

plt.subplot(1, 2, 2)
plt.hist(final_opinions, bins=20, range=(-1, 1), color='blue', alpha=0.7)
plt.title(f'最终观点分布 (干预后)\n方差: {final_variance:.4f}')
plt.xlabel('观点值')
plt.xlim(-1, 1)

plt.suptitle(f'干预前后观点分布对比 (预算: {BUDGET}, 奖励方法: {REWARD_METHOD})', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/opinion_distribution.png')
print("结果已保存到 'results/figures/opinion_distribution.png'")

# 显示图片
plt.show()