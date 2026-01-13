# diagnose_methods.py
import torch
import numpy as np
import networkx as nx
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.data_loader import load_graph, sample_subgraph, _init_opinions
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward, compute_polarization_variance
from src.baselines.node_selection import (
    select_nodes_random, select_nodes_high_degree, 
    select_nodes_pagerank, select_nodes_finder
)

def run_simulation(graph, selected_nodes, intervention_weights):
    env = OpinionDynamicsEnv(
        graph=graph.copy(),
        budget=len(selected_nodes),
        reward_fn=lambda s, ns: compute_reward(s, ns, method="variance"),
        dynamics_model="degroot",
        dynamics_steps=3,
        device="cpu"
    )
    state = env.reset(seed=42)
    initial_polarization = compute_polarization_variance(state)
    polarization_history = [initial_polarization]

    for node, weight in zip(selected_nodes, intervention_weights):
        state, _, _, _ = env.step(
            action_node=node,
            intervention_weight=weight,
            alpha=0.5,
            use_new_formula=True
        )
        polarization_history.append(compute_polarization_variance(state))
    
    print(f"  Initial Polarization: {initial_polarization:.4f}")
    print(f"  Final Score: {polarization_history[-1]:.4f}")
    print(f"  History (first 5): {[f'{p:.4f}' for p in polarization_history[:5]]}")


def main():
    print("--- 诊断开始 ---")
    graph = load_graph('ego-Twitter', opinion_init="bimodal")
    small_graph = sample_subgraph(graph, max_nodes=2000, method="largest_component")
    # 重新初始化观点（子图采样后观点值丢失）
    _init_opinions(small_graph, mode="bimodal")
    k = 50  # 预算
    print(f"使用 {small_graph.number_of_nodes()} 节点的子图进行测试，预算 k={k}\n")

    methods = {
        "Random": lambda g: select_nodes_random(g, k, seed=42),
        "High-Degree": lambda g: select_nodes_high_degree(g, k),
        "PageRank": lambda g: select_nodes_pagerank(g, k),
        "FINDER": lambda g: select_nodes_finder(g, k, model_path="results/models/final_model.pth", device="cuda:0")
    }

    for name, func in methods.items():
        print(f"--- 测试方法: {name} ---")
        try:
            selected_nodes = func(small_graph)
            print(f"  选出的节点数: {len(selected_nodes)}")
            if not selected_nodes:
                print("  !! 警告: 未选出任何节点")
                continue
            
            print(f"  选出的节点 (前5个): {selected_nodes[:5]}")
            
            # 模拟环境运行
            if name == "FINDER":
                weights = [0.5] * len(selected_nodes)
            else:
                weights = [1.0] * len(selected_nodes)
            
            run_simulation(small_graph, selected_nodes, weights)

        except Exception as e:
            print(f"  !! 错误: {e}")
        print("\n")

if __name__ == "__main__":
    main()
