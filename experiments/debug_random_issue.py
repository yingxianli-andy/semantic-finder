#!/usr/bin/env python3
"""
调试Random方法异常：检查为什么budget=3812时效果突然提升
"""
import sys
import os
import numpy as np
import networkx as nx
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.data_loader import load_graph
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_polarization_variance
from src.baselines.node_selection import select_nodes_random

def debug_random_method(budget: int, seed: int = 0):
    """调试Random方法在指定budget下的行为"""
    print(f"\n{'='*60}")
    print(f"调试 Random 方法: budget={budget}, seed={seed}")
    print(f"{'='*60}\n")
    
    # 加载图
    graph = load_graph("ego-Twitter", opinion_init="bimodal")
    print(f"图大小: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    # 初始化环境
    reward_fn = lambda s, ns: compute_polarization_variance(s) - compute_polarization_variance(ns)
    env = OpinionDynamicsEnv(
        graph=graph.copy(),
        budget=budget,
        reward_fn=reward_fn,
        dynamics_model="degroot",
        dynamics_steps=3,
        device="cpu"
    )
    
    # 固定种子
    np.random.seed(seed)
    
    # 选择节点
    selected_nodes = select_nodes_random(graph, budget, seed=seed)
    print(f"选择的节点数: {len(selected_nodes)}")
    print(f"前10个节点: {selected_nodes[:10]}")
    
    # 检查是否有重复
    if len(selected_nodes) != len(set(selected_nodes)):
        print(f"⚠️  警告：有重复节点！")
        print(f"  唯一节点数: {len(set(selected_nodes))}")
    
    # 初始化环境
    state = env.reset(seed=seed)
    initial_polarization = compute_polarization_variance(state)
    print(f"\n初始极化度: {initial_polarization:.6f}")
    
    # 使用权重1.0进行干预（与实验脚本一致）
    intervention_weights = [1.0] * len(selected_nodes)
    
    polarization_history = [initial_polarization]
    step_details = []
    
    # 运行前10步，详细记录
    for step in range(min(10, len(selected_nodes))):
        node = selected_nodes[step]
        weight = intervention_weights[step]
        
        # 记录干预前的状态
        prev_opinion = state.nodes[node].get('opinion', 0.0)
        prev_polarization = compute_polarization_variance(state)
        
        # 执行干预
        next_state, reward, done, info = env.step(
            action_node=node,
            intervention_weight=weight,
            alpha=0.5,
            use_new_formula=True
        )
        
        # 记录干预后的状态
        next_opinion = next_state.nodes[node].get('opinion', 0.0)
        next_polarization = compute_polarization_variance(next_state)
        
        step_details.append({
            'step': step + 1,
            'node': node,
            'weight': weight,
            'prev_opinion': prev_opinion,
            'next_opinion': next_opinion,
            'opinion_change': next_opinion - prev_opinion,
            'prev_polarization': prev_polarization,
            'next_polarization': next_polarization,
            'polarization_change': next_polarization - prev_polarization,
            'reward': reward
        })
        
        polarization_history.append(next_polarization)
        state = next_state
        
        print(f"\nStep {step+1}:")
        print(f"  节点: {node}, 权重: {weight}")
        print(f"  观点变化: {prev_opinion:.6f} -> {next_opinion:.6f} (变化: {next_opinion - prev_opinion:.6f})")
        print(f"  极化度变化: {prev_polarization:.6f} -> {next_polarization:.6f} (变化: {next_polarization - prev_polarization:.6f})")
        print(f"  奖励: {reward:.6f}")
        
        # 检查是否有异常值
        if abs(next_opinion) > 1.0:
            print(f"  ⚠️  警告：观点值超出[-1, 1]范围！")
        if next_polarization < 0:
            print(f"  ⚠️  警告：极化度为负值！")
    
    # 继续运行剩余步骤（不详细记录）
    for step in range(10, len(selected_nodes)):
        node = selected_nodes[step]
        weight = intervention_weights[step]
        next_state, reward, done, info = env.step(
            action_node=node,
            intervention_weight=weight,
            alpha=0.5,
            use_new_formula=True
        )
        polarization_history.append(compute_polarization_variance(next_state))
        state = next_state
        if done:
            break
    
    final_polarization = polarization_history[-1]
    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"  初始极化度: {initial_polarization:.6f}")
    print(f"  最终极化度: {final_polarization:.6f}")
    print(f"  极化度下降: {initial_polarization - final_polarization:.6f}")
    print(f"  下降百分比: {(initial_polarization - final_polarization) / initial_polarization * 100:.2f}%")
    print(f"  总步数: {len(polarization_history) - 1}")
    print(f"{'='*60}\n")
    
    # 检查是否有异常
    if final_polarization < 0:
        print("⚠️  异常：最终极化度为负值！")
    if final_polarization > initial_polarization:
        print("⚠️  异常：最终极化度大于初始值！")
    if any(p < 0 for p in polarization_history):
        print("⚠️  异常：极化度历史中有负值！")
    
    # 检查观点值范围
    all_opinions = [state.nodes[i].get('opinion', 0.0) for i in range(state.number_of_nodes())]
    min_opinion = min(all_opinions)
    max_opinion = max(all_opinions)
    print(f"观点值范围: [{min_opinion:.6f}, {max_opinion:.6f}]")
    if min_opinion < -1.0 or max_opinion > 1.0:
        print("⚠️  异常：观点值超出[-1, 1]范围！")
    
    return {
        'budget': budget,
        'seed': seed,
        'initial_polarization': initial_polarization,
        'final_polarization': final_polarization,
        'polarization_history': polarization_history,
        'step_details': step_details
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="调试Random方法异常")
    parser.add_argument("--budget", type=int, default=762, help="预算大小")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()
    
    result = debug_random_method(args.budget, args.seed)
    
    # 保存结果
    output_file = f"debug_random_budget_{args.budget}_seed_{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存到: {output_file}")

