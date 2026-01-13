"""
测试详细日志输出
快速运行几个step，展示每一步的耗时
"""
import sys
sys.path.insert(0, '.')
import networkx as nx
import numpy as np
import time
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_polarization_variance
from src.baselines.node_selection import select_nodes_random

print("=" * 60)
print("详细日志测试 - 展示每一步的耗时")
print("=" * 60)
print()

# 创建测试图（模拟真实大小）
n_nodes = 70000
print(f"创建测试图: {n_nodes}节点...")
graph = nx.erdos_renyi_graph(n_nodes, 0.0001)
for i in range(n_nodes):
    graph.nodes[i]['opinion'] = np.random.uniform(-1, 1)
print(f"图创建完成: {graph.number_of_nodes()}节点, {graph.number_of_edges()}边")
print()

# 初始化环境
def reward_fn(prev, next_state):
    return compute_polarization_variance(prev) - compute_polarization_variance(next_state)

env = OpinionDynamicsEnv(graph, budget=10, reward_fn=reward_fn, device='cuda')
env.reset()

# 选择节点
print("选择节点（Random方法）...")
selected_nodes = select_nodes_random(graph, 10, seed=42)
print(f"已选择 {len(selected_nodes)} 个节点")
print()

# 执行几个step，展示详细日志
print("=" * 60)
print("开始执行step，详细日志如下：")
print("=" * 60)
print()

for step in range(min(5, len(selected_nodes))):
    print(f"\n--- Step {step+1} ---")
    action_node = selected_nodes[step]
    intervention_weight = 1.0
    
    step_start = time.time()
    _, reward, done, _ = env.step(action_node, intervention_weight)
    step_time = time.time() - step_start
    
    print(f"Step {step+1} 总耗时: {step_time:.3f}秒")
    print(f"当前极化度: {compute_polarization_variance(env.graph):.6f}")
    print(f"奖励: {reward:.6f}")

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
